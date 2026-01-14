import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# KDE dependency
try:
    from scipy.stats import gaussian_kde
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ----------------------------
# Data loading (robust to comment/header junk)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_possessions(file_bytes: bytes) -> pd.DataFrame:
    """
    Handles files that start with comment lines like:
    # CSV-File created with merge-csv.com
    and possible blank lines, by locating a plausible header row.
    """
    text = file_bytes.decode("utf-8", errors="ignore").splitlines()

    header_idx = None
    for i, line in enumerate(text[:150]):
        s = line.strip()
        if s.startswith("event_id,") and "match_id" in s and ("x_end" in s or "y_end" in s):
            header_idx = i
            break

    if header_idx is None:
        return pd.read_csv(io.BytesIO(file_bytes))

    data_str = "\n".join(text[header_idx:])
    return pd.read_csv(io.StringIO(data_str), engine="python")


@st.cache_data(show_spinner=False)
def load_from_path(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        return load_possessions(f.read())


# ----------------------------
# Pitch drawing (x in [-52.5, 52.5], y in [-34, 34])
# ----------------------------
def draw_pitch(ax):
    ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5],
            [-34, -34, 34, 34, -34], linewidth=1)
    ax.plot([0, 0], [-34, 34], linewidth=1)

    centre_circle = plt.Circle((0, 0), 9.15, fill=False, linewidth=1)
    ax.add_artist(centre_circle)

    # Penalty areas (16.5m from goal line, 40.3m wide -> y +/-20.15)
    ax.plot([-52.5, -36], [-20.15, -20.15], linewidth=1)
    ax.plot([-36, -36], [-20.15, 20.15], linewidth=1)
    ax.plot([-36, -52.5], [20.15, 20.15], linewidth=1)

    ax.plot([52.5, 36], [-20.15, -20.15], linewidth=1)
    ax.plot([36, 36], [-20.15, 20.15], linewidth=1)
    ax.plot([36, 52.5], [20.15, 20.15], linewidth=1)

    ax.set_aspect("equal")
    ax.set_xlim(-55, 55)
    ax.set_ylim(-37, 37)
    ax.axis("off")

# ----------------------------
# Thirds helpers (based on x coordinate)
# ----------------------------
def label_third_from_x(x: float) -> str:
    """
    Splits the pitch length (-52.5 to 52.5) into 3 equal thirds using boundaries at -17.5 and 17.5.
    Defensive: x < -17.5
    Middle:    -17.5 <= x <= 17.5
    Attacking: x > 17.5
    """
    if pd.isna(x):
        return "Unknown"
    if x < -17.5:
        return "Defensive"
    if x > 17.5:
        return "Attacking"
    return "Middle"


def thirds_progressed(start_x: float, end_x: float) -> int:
    """
    Counts how many thirds a possession moves through, ignoring directionality:
    Defensive -> Middle = 1
    Middle -> Attacking = 1
    Defensive -> Attacking = 2
    Otherwise = 0
    """
    s = label_third_from_x(start_x)
    e = label_third_from_x(end_x)
    if "Unknown" in (s, e):
        return 0
    mapping = {
        ("Defensive", "Middle"): 1,
        ("Middle", "Attacking"): 1,
        ("Defensive", "Attacking"): 2,
    }
    return mapping.get((s, e), 0)



# ----------------------------
# Aggregation helpers
# ----------------------------
def add_team_possession_sequence_id(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """
    Adds a derived `team_possession_seq` per match by cumulatively summing
    `first_player_possession_in_team_possession` (when present).

    This works well when each team possession has its first row flagged.
    """
    df = df.copy()
    if "index" in df.columns:
        df = df.sort_values(["match_id", "index"])
    else:
        df = df.sort_values(["match_id", "frame_start"])

    if "first_player_possession_in_team_possession" in df.columns:
        # Ensure boolean-ish values are treated properly
        first = df["first_player_possession_in_team_possession"].fillna(False).astype(bool)
        df["team_possession_seq"] = first.groupby(df["match_id"]).cumsum()
        warn = None
    else:
        # Fallback: treat every row as its own sequence (not great, but keeps app usable)
        df["team_possession_seq"] = np.arange(len(df))
        warn = (
            "Column 'first_player_possession_in_team_possession' not found. "
            "Falling back to treating each row as a separate sequence (effectiveness metrics will be less meaningful)."
        )
    return df, warn


def get_team_possession_ends(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """
    Uses last_player_possession_in_team_possession == True when present.
    If the flag is missing, returns last row per (match_id, team_possession_seq).
    """
    df = df.copy()
    if "last_player_possession_in_team_possession" in df.columns:
        out = df[df["last_player_possession_in_team_possession"] == True].copy()
        return out, None

    # Fallback: last row per sequence
    if "team_possession_seq" in df.columns:
        if "index" in df.columns:
            out = df.sort_values(["match_id", "team_possession_seq", "index"]).groupby(
                ["match_id", "team_possession_seq"], as_index=False
            ).tail(1)
        else:
            out = df.sort_values(["match_id", "team_possession_seq", "frame_end"]).groupby(
                ["match_id", "team_possession_seq"], as_index=False
            ).tail(1)
        return out.copy(), (
            "Column 'last_player_possession_in_team_possession' not found. "
            "Using last row per derived team possession sequence as a fallback."
        )

    return df.copy(), (
        "Could not aggregate to team possession ends (missing flags and derived sequence id). Using all rows."
    )


def summarise_sequences(df: pd.DataFrame, team_col: str) -> pd.DataFrame:
    """
    One row per team possession sequence (match_id + team_possession_seq),
    with simple effectiveness proxies.

    IMPORTANT: We build everything inside a single groupby.agg so indices align.
    We also sort rows within sequences using the best available time/order column.
    """
    # Choose an order column that actually exists in this dataset
    order_candidates = ["index", "frame_start", "time_start", "frame_end", "time_end", "event_id"]
    order_col = next((c for c in order_candidates if c in df.columns), None)

    sort_cols = ["match_id", "team_possession_seq"] + ([order_col] if order_col else [])
    d = df.sort_values(sort_cols).copy()

    gb = d.groupby(["match_id", "team_possession_seq"], as_index=False)

    def first(series):
        return series.iloc[0]

    def last(series):
        return series.iloc[-1]

    agg_spec = {
        "team": (team_col, first),
        "start_x": ("x_start", first),
        "start_y": ("y_start", first),
        "end_x": ("x_end", last),
        "end_y": ("y_end", last),
        "duration": ("duration", "sum"),
        "n_player_possessions": ("event_id", "size"),
    }

    # Optional outcome flags
    if "lead_to_shot" in d.columns:
        agg_spec["lead_to_shot"] = ("lead_to_shot", "max")
    else:
        # Will fill after agg
        pass

    if "lead_to_goal" in d.columns:
        agg_spec["lead_to_goal"] = ("lead_to_goal", "max")

    # Optional end-zone labels/flags (take last value in the sequence)
    if "third_end" in d.columns:
        agg_spec["third_end"] = ("third_end", last)

    if "penalty_area_end" in d.columns:
        agg_spec["penalty_area_end"] = ("penalty_area_end", last)

    out = gb.agg(**agg_spec)

    # Ensure missing optional columns exist with sensible defaults
    if "lead_to_shot" not in out.columns:
        out["lead_to_shot"] = False
    if "lead_to_goal" not in out.columns:
        out["lead_to_goal"] = False
    if "third_end" not in out.columns:
        out["third_end"] = np.nan
    if "penalty_area_end" not in out.columns:
        out["penalty_area_end"] = np.nan

    out["progression_x"] = out["end_x"] - out["start_x"]
    out["progression_abs_x"] = out["progression_x"].abs()
    return out



# ----------------------------
# KDE heatmap plotter
# ----------------------------
def kde_density_grid(x, y, xmin=-52.5, xmax=52.5, ymin=-34, ymax=34, nx=300, ny=200):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    xx, yy = np.mgrid[xmin:xmax:complex(nx), ymin:ymax:complex(ny)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])

    kde = gaussian_kde(values)  # Scott's rule
    zz = np.reshape(kde(positions).T, xx.shape)
    return xx, yy, zz


def plot_kde(ax, x, y, alpha=0.85):
    xmin, xmax = -52.5, 52.5
    ymin, ymax = -34, 34

    if not SCIPY_OK:
        draw_pitch(ax)
        ax.set_title("scipy not installed (needed for KDE)")
        return None

    if len(x) < 2:
        draw_pitch(ax)
        ax.set_title("Not enough points for KDE")
        return None

    _, _, zz = kde_density_grid(x, y, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    draw_pitch(ax)
    img = ax.imshow(
        zz.T,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        alpha=alpha,
        aspect="equal",
    )
    return img


# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="Possession KDE Comparison", layout="wide")
st.title("Possession Analysis – KDE Heatmaps + Effectiveness + Player Links")

st.caption(
    "Compares two teams using KDE heatmaps of **team possession sequence end locations**, "
    "plus simple effectiveness metrics and player involvement / combinations."
)

default_path = os.path.join("data", "possessions.csv")
df = None

with st.sidebar:
    st.header("Data")
    use_local = st.checkbox("Use local file data/possessions.csv", value=os.path.exists(default_path))
    uploaded = None if use_local else st.file_uploader("Upload CSV", type=["csv"])

if use_local and os.path.exists(default_path):
    df = load_from_path(default_path)
elif uploaded is not None:
    df = load_possessions(uploaded.getvalue())
else:
    st.info("Upload a CSV, or add your file to `data/possessions.csv` and tick the checkbox in the sidebar.")
    st.stop()

# Required columns
required = {"match_id", "x_start", "y_start", "x_end", "y_end"}
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Type coercions
for c in ["x_start", "y_start", "x_end", "y_end", "duration"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

team_col = "team_shortname" if "team_shortname" in df.columns else ("team_id" if "team_id" in df.columns else None)
if team_col is None:
    st.error("Could not find a team identifier column (team_shortname or team_id).")
    st.stop()

# --- Base filtering dataset (player possessions) ---
match_ids = sorted(df["match_id"].dropna().unique().tolist())

with st.sidebar:
    st.header("Filters")
    selected_matches = st.multiselect("Match(es)", match_ids, default=match_ids)

dff = df[df["match_id"].isin(selected_matches)].copy()
dff_all = dff.copy()  # snapshot before any team-scope filtering

# Optional filters (phase and half)
with st.sidebar:
    if "team_in_possession_phase_type" in dff_all.columns:
        phase_opts = ["All"] + sorted([x for x in dff["team_in_possession_phase_type"].dropna().unique().tolist()])
        phase_choice = st.selectbox("In-possession phase", phase_opts, index=0)
        if phase_choice != "All":
            dff = dff[dff["team_in_possession_phase_type"] == phase_choice]

    half_col = None
    for cand in ["period", "half", "match_period"]:
        if cand in dff.columns:
            half_col = cand
            break
    if half_col is not None:
        half_opts = ["All"] + sorted([x for x in dff[half_col].dropna().unique().tolist()])
        half_choice = st.selectbox("Half / period", half_opts, index=0)
        if half_choice != "All":
            dff = dff[dff[half_col] == half_choice]

# Add derived sequence id


# Optional: zone filters (by thirds, pitch zones, or custom rectangle)
with st.sidebar:
    st.subheader("Zone filter")
    zone_target = st.selectbox(
        "Apply zone filter to",
        ["None", "Possession end location (x_end, y_end)", "Possession start location (x_start, y_start)"],
        index=0,
        help="Filters the underlying rows used for heatmaps, effectiveness, and player sections."
    )

    zone_mode = st.selectbox("Zone mode", ["Thirds", "Pitch zones", "Custom rectangle"], index=1)

# Apply zone filter (if chosen)
if zone_target != "None":
    if "end location" in zone_target:
        zx, zy = "x_end", "y_end"
    else:
        zx, zy = "x_start", "y_start"

    # Ensure numeric
    dff[zx] = pd.to_numeric(dff[zx], errors="coerce")
    dff[zy] = pd.to_numeric(dff[zy], errors="coerce")

    if zone_mode == "Thirds":
        with st.sidebar:
            third_choice = st.selectbox("Select third", ["Defensive", "Middle", "Attacking"])
        dff["_zone_third"] = dff[zx].apply(label_third_from_x)
        dff = dff[dff["_zone_third"] == third_choice].copy()

    elif zone_mode == "Pitch zones":
        # Define a pitch grid: 6 columns (length) x 4 rows (width) = 24 zones
        x_edges = np.linspace(-52.5, 52.5, 7)  # 6 bins
        y_edges = np.linspace(-34.0, 34.0, 5)  # 4 bins

        def zone_id_for_xy(x, y):
            if pd.isna(x) or pd.isna(y):
                return None
            xi = np.searchsorted(x_edges, x, side="right") - 1
            yi = np.searchsorted(y_edges, y, side="right") - 1
            if xi < 0 or xi >= 6 or yi < 0 or yi >= 4:
                return None
            return f"Z{yi+1}-{xi+1}"  # row-col (1-indexed), row is bottom->top

        # Build zone centers for clickable selection
        zone_centers = []
        zone_ids = []
        for r in range(4):      # rows (bottom->top)
            for c in range(6):  # cols (left->right)
                x0, x1 = x_edges[c], x_edges[c+1]
                y0, y1 = y_edges[r], y_edges[r+1]
                zone_centers.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
                zone_ids.append(f"Z{r+1}-{c+1}")

        # Persist selection in session_state
        if "selected_zones" not in st.session_state:
            st.session_state["selected_zones"] = []

        with st.sidebar:
            st.markdown("**Clickable pitch – select zones**")
            st.caption("Click points to select. Use box/lasso to multi-select. Double-click to reset selection.")
            show_zone_labels = st.checkbox("Show zone labels", value=False)

        # Plotly pitch + clickable zone points
        def make_pitch_zone_fig(selected):
            xs = [p[0] for p in zone_centers]
            ys = [p[1] for p in zone_centers]
            texts = zone_ids if show_zone_labels else [""] * len(zone_ids)

            fig = go.Figure()

            # Zone points
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="markers+text" if show_zone_labels else "markers",
                text=texts,
                textposition="middle center",
                customdata=zone_ids,
                marker=dict(size=16, opacity=0.85),
                selected=dict(marker=dict(size=18, opacity=1.0)),
                unselected=dict(marker=dict(opacity=0.45)),
            ))

            # Pitch outline + key markings as shapes
            shapes = []
            # Outline
            shapes.append(dict(type="rect", x0=-52.5, y0=-34, x1=52.5, y1=34, line=dict(width=2)))
            # Halfway
            shapes.append(dict(type="line", x0=0, y0=-34, x1=0, y1=34, line=dict(width=2)))
            # Centre circle (approx with many points via shape circle)
            shapes.append(dict(type="circle", x0=-9.15, y0=-9.15, x1=9.15, y1=9.15, line=dict(width=2)))
            # Penalty areas
            shapes.append(dict(type="rect", x0=-52.5, y0=-20.15, x1=-36, y1=20.15, line=dict(width=2)))
            shapes.append(dict(type="rect", x0=36, y0=-20.15, x1=52.5, y1=20.15, line=dict(width=2)))

            # Grid lines
            for xe in x_edges[1:-1]:
                shapes.append(dict(type="line", x0=xe, y0=-34, x1=xe, y1=34, line=dict(width=1, dash="dot")))
            for ye in y_edges[1:-1]:
                shapes.append(dict(type="line", x0=-52.5, y0=ye, x1=52.5, y1=ye, line=dict(width=1, dash="dot")))

            # Highlight selected zones as translucent rectangles
            for zid in selected:
                r, c = zid[1:].split("-")
                r = int(r) - 1
                c = int(c) - 1
                x0, x1 = x_edges[c], x_edges[c+1]
                y0, y1 = y_edges[r], y_edges[r+1]
                shapes.append(dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                                   fillcolor="rgba(0,0,0,0.15)", line=dict(width=0)))

            fig.update_layout(
                shapes=shapes,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(range=[-55, 55], showgrid=False, zeroline=False, visible=False),
                yaxis=dict(range=[-37, 37], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
                dragmode="select",  # enables box select (and lasso via modebar)
                height=320,
            )
            return fig

        fig = make_pitch_zone_fig(st.session_state["selected_zones"])

        # Streamlit selection events (no extra component needed)
        sel = st.plotly_chart(fig, use_container_width=True, key="zone_pitch", on_select="rerun")
        # On Streamlit versions that support selection events, selection is stored in session_state:
        # st.session_state["zone_pitch"] contains {"selection": {...}} when available.
        sel_state = st.session_state.get("zone_pitch", {})
        selection = sel_state.get("selection", None)

        if selection and "points" in selection and selection["points"]:
            newly_selected = [p.get("customdata") for p in selection["points"] if p.get("customdata")]
            # Update selection (toggle behavior)
            current = set(st.session_state["selected_zones"])
            for z in newly_selected:
                if z in current:
                    current.remove(z)
                else:
                    current.add(z)
            st.session_state["selected_zones"] = sorted(current)

        selected = st.session_state["selected_zones"]

        if selected:
            dff["_zone_id"] = [zone_id_for_xy(x, y) for x, y in zip(dff[zx], dff[zy])]
            dff = dff[dff["_zone_id"].isin(selected)].copy()
        else:
            st.sidebar.warning("No zones selected — zone filter not applied.")


    else:
        with st.sidebar:
            x_min, x_max = st.slider("x range", min_value=-52.5, max_value=52.5, value=(-52.5, 52.5), step=0.5)
            y_min, y_max = st.slider("y range", min_value=-34.0, max_value=34.0, value=(-34.0, 34.0), step=0.5)
        dff = dff[(dff[zx] >= x_min) & (dff[zx] <= x_max) & (dff[zy] >= y_min) & (dff[zy] <= y_max)].copy()

dff, seq_warn = add_team_possession_sequence_id(dff)
if seq_warn:
    st.warning(seq_warn)
dff_all, _seq_warn_all = add_team_possession_sequence_id(dff_all)
# We intentionally do not show _seq_warn_all to avoid duplicate warnings.


# Teams after filtering
teams = sorted(dff[team_col].dropna().unique().tolist())
if not teams:
    st.error("No teams found after filtering.")
    st.stop()

# Team picker with MU heuristic
default_mu = None
for t in teams:
    s = str(t).lower()
    if "man" in s and ("utd" in s or "united" in s):
        default_mu = t
        break
if default_mu is None:
    default_mu = teams[0]

with st.sidebar:
    st.header("Compare")
    left_team = st.selectbox("Team A", teams, index=teams.index(default_mu) if default_mu in teams else 0)
    other_teams = [t for t in teams if t != left_team]
    right_team = st.selectbox("Team B", other_teams, index=0 if other_teams else 0)

    scope = st.radio(
        "Comparison scope",
        ["Across selected matches", "Only matches where both teams appear"],
        index=0,
        help="Second option restricts to match_ids containing both Team A and Team B.",
    )

# Apply scope restriction at the sequence level (and propagate to player possessions)
if scope == "Only matches where both teams appear":
    matches_with_A = set(dff.loc[dff[team_col] == left_team, "match_id"].dropna().unique().tolist())
    matches_with_B = set(dff.loc[dff[team_col] == right_team, "match_id"].dropna().unique().tolist())
    common_matches = sorted(list(matches_with_A.intersection(matches_with_B)))
    dff = dff[dff["match_id"].isin(common_matches)].copy()

# --- Sequence summaries (effectiveness) ---
seq_summary = summarise_sequences(dff, team_col=team_col)
seq_summary_all = summarise_sequences(dff_all, team_col=team_col)

# Ends for heatmaps
ends, ends_warn = get_team_possession_ends(dff)
if ends_warn:
    st.warning(ends_warn)

A_ends = ends[ends[team_col] == left_team].dropna(subset=["x_end", "y_end"])
B_ends = ends[ends[team_col] == right_team].dropna(subset=["x_end", "y_end"])

# --- Top metrics ---
A_seq = seq_summary[seq_summary["team"] == left_team].dropna(subset=["end_x", "end_y"])
B_seq = seq_summary[seq_summary["team"] == right_team].dropna(subset=["end_x", "end_y"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Matches in view", int(dff["match_id"].nunique()))
c2.metric(f"{left_team} sequences", len(A_seq))
c3.metric(f"{right_team} sequences", len(B_seq))
c4.metric("KDE available", "Yes" if SCIPY_OK else "No (install scipy)")

# --- Heatmaps ---
st.subheader("Where possessions end (KDE heatmaps)")
colA, colB = st.columns(2)

with colA:
    fig, ax = plt.subplots(figsize=(8, 5))
    img = plot_kde(ax, A_ends["x_end"].values, A_ends["y_end"].values)
    ax.set_title(f"{left_team} – Possession End KDE")
    if img is not None:
        plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, use_container_width=True)

with colB:
    fig, ax = plt.subplots(figsize=(8, 5))
    img = plot_kde(ax, B_ends["x_end"].values, B_ends["y_end"].values)
    ax.set_title(f"{right_team} – Possession End KDE")
    if img is not None:
        plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, use_container_width=True)

# --- Effectiveness comparisons ---


# --- Where possessions start and end (common areas) ---
st.subheader("Most common areas for possessions to start vs end")

st.caption(
    "These plots use **team possession sequences** (one per sequence). "
    "Start = first location in the sequence, End = last location in the sequence."
)

team_for_zones = st.radio("Team for start/end zones", [str(left_team), str(right_team), "Both"], horizontal=True)

def plot_start_end_for_team(team_name: str):
    seq = seq_summary[seq_summary["team"] == team_name].dropna(subset=["start_x", "start_y", "end_x", "end_y"])
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        img = plot_kde(ax, seq["start_x"].values, seq["start_y"].values)
        ax.set_title(f"{team_name} – Sequence START locations (KDE)")
        if img is not None:
            plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        img = plot_kde(ax, seq["end_x"].values, seq["end_y"].values)
        ax.set_title(f"{team_name} – Sequence END locations (KDE)")
        if img is not None:
            plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, use_container_width=True)

if team_for_zones == "Both":
    t1, t2 = st.tabs([str(left_team), str(right_team)])
    with t1:
        plot_start_end_for_team(str(left_team))
    with t2:
        plot_start_end_for_team(str(right_team))
else:
    plot_start_end_for_team(team_for_zones)

st.subheader("Effectiveness of possession sequences (simple comparisons)")

def safe_mean(s):
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if len(s) else float("nan")

def rate_true(s):
    if len(s) == 0:
        return float("nan")
    return float(pd.Series(s).fillna(False).astype(bool).mean())

def pct(cond):
    return float(cond.mean()) if len(cond) else float("nan")

# End-in-zone flags
A_final_third = None
B_final_third = None
if "third_end" in A_seq.columns and A_seq["third_end"].notna().any():
    # Heuristic: treat anything containing "Attacking" as final third
    A_final_third = A_seq["third_end"].astype(str).str.contains("attacking", case=False, na=False)
    B_final_third = B_seq["third_end"].astype(str).str.contains("attacking", case=False, na=False)

A_pen = None
B_pen = None
if "penalty_area_end" in A_seq.columns and A_seq["penalty_area_end"].notna().any():
    # penalty_area_end is often boolean
    A_pen = A_seq["penalty_area_end"].fillna(False).astype(bool)
    B_pen = B_seq["penalty_area_end"].fillna(False).astype(bool)

# KPI row
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric(f"{left_team} shot rate", f"{100*rate_true(A_seq.get('lead_to_shot', False)):.1f}%")
k2.metric(f"{right_team} shot rate", f"{100*rate_true(B_seq.get('lead_to_shot', False)):.1f}%")
k3.metric(f"{left_team} avg duration", f"{safe_mean(A_seq['duration']):.2f}s")
k4.metric(f"{right_team} avg duration", f"{safe_mean(B_seq['duration']):.2f}s")
k5.metric(f"{left_team} avg x-prog", f"{safe_mean(A_seq['progression_x']):.2f}")
k6.metric(f"{right_team} avg x-prog", f"{safe_mean(B_seq['progression_x']):.2f}")

# Bar chart: key rates
labels = ["Lead to shot", "Lead to goal"]
A_rates = [rate_true(A_seq.get("lead_to_shot", False)), rate_true(A_seq.get("lead_to_goal", False))]
B_rates = [rate_true(B_seq.get("lead_to_shot", False)), rate_true(B_seq.get("lead_to_goal", False))]

if A_final_third is not None and B_final_third is not None:
    labels.append("End in final third")
    A_rates.append(pct(A_final_third))
    B_rates.append(pct(B_final_third))

if A_pen is not None and B_pen is not None:
    labels.append("End in penalty area")
    A_rates.append(pct(A_pen))
    B_rates.append(pct(B_pen))

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(x - width/2, A_rates, width, label=str(left_team))
ax.bar(x + width/2, B_rates, width, label=str(right_team))
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylim(0, max(0.01, np.nanmax(A_rates + B_rates) * 1.25))
ax.set_ylabel("Rate")
ax.legend()
st.pyplot(fig, use_container_width=True)

# Distribution: progression and duration
dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(A_seq["progression_x"].dropna().values, bins=30, alpha=0.6, label=str(left_team))
    ax.hist(B_seq["progression_x"].dropna().values, bins=30, alpha=0.6, label=str(right_team))
    ax.set_title("Distribution: x progression (end_x - start_x)")
    ax.set_xlabel("x progression")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

with dist_col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(A_seq["duration"].dropna().values, bins=30, alpha=0.6, label=str(left_team))
    ax.hist(B_seq["duration"].dropna().values, bins=30, alpha=0.6, label=str(right_team))
    ax.set_title("Distribution: sequence duration (sum of player possessions)")
    ax.set_xlabel("seconds")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

# --- Player involvement ---

# ============================================================
# Advanced effectiveness dashboard (filters + multi-team charts)
# ============================================================
st.subheader("Who is most involved?")

player_col = "player_name" if "player_name" in dff.columns else None
if player_col is None:
    st.info("No player_name column found, so player involvement charts are unavailable.")
else:
    p1, p2 = st.columns(2)

    def player_involvement(team_name: str) -> pd.DataFrame:
        dd = dff[dff[team_col] == team_name].dropna(subset=[player_col]).copy()

        # Total involvements (count of player possessions)
        # Use reset_index(name=...) to be robust across pandas versions.
        total = (
            dd[player_col]
            .value_counts()
            .reset_index(name="player_possessions")
            .rename(columns={player_col: "player"})
        )

        # Unique sequences involved in
        seqs = (
            dd.groupby(player_col)["team_possession_seq"]
            .nunique()
            .reset_index(name="sequences_involved")
            .rename(columns={player_col: "player"})
        )

        out = total.merge(seqs, on="player", how="left")
        return out

    A_inv = player_involvement(left_team)
    B_inv = player_involvement(right_team)

    with p1:
        st.markdown(f"**Top players – {left_team}**")
        topA = A_inv.head(12)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(topA["player"][::-1], topA["player_possessions"][::-1])
        ax.set_xlabel("Player possessions")
        ax.set_ylabel("Player")
        st.pyplot(fig, use_container_width=True)
        st.dataframe(topA, use_container_width=True, hide_index=True)

    with p2:
        st.markdown(f"**Top players – {right_team}**")
        topB = B_inv.head(12)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(topB["player"][::-1], topB["player_possessions"][::-1])
        ax.set_xlabel("Player possessions")
        ax.set_ylabel("Player")
        st.pyplot(fig, use_container_width=True)
        st.dataframe(topB, use_container_width=True, hide_index=True)

# --- Player combinations (passing links) ---


# --- Player ball progression value (through thirds) ---
st.subheader("Who adds the most ball progression through the thirds?")

st.caption(
    "Simple 'value' proxy: counts how often a player moves the ball from Defensive→Middle, Middle→Attacking (1 each), "
    "or Defensive→Attacking (2). Boundaries are x=-17.5 and x=17.5 in your coordinate system."
)

if player_col is None:
    st.info("No player_name column found, so progression-by-player is unavailable.")
else:
    def player_third_progression(team_name: str) -> pd.DataFrame:
        dd = dff[dff[team_col] == team_name].dropna(subset=[player_col, "x_start", "y_start", "x_end", "y_end"]).copy()
        dd["start_third"] = dd["x_start"].apply(label_third_from_x)
        dd["end_third"] = dd["x_end"].apply(label_third_from_x)
        dd["thirds_progressed"] = [thirds_progressed(s, e) for s, e in zip(dd["x_start"], dd["x_end"])]
        dd["dx"] = (dd["x_end"] - dd["x_start"])

        out = (
            dd.groupby(player_col, as_index=False)
            .agg(
                player_possessions=(player_col, "size"),
                thirds_progressed=("thirds_progressed", "sum"),
                avg_thirds_progressed=("thirds_progressed", "mean"),
                total_x_progression=("dx", "sum"),
                avg_x_progression=("dx", "mean"),
            )
            .rename(columns={player_col: "player"})
            .sort_values("thirds_progressed", ascending=False)
        )
        return out

    pr1, pr2 = st.columns(2)
    with pr1:
        st.markdown(f"**Top ball progressors – {left_team}**")
        progA = player_third_progression(left_team)
        topA = progA.head(12)
        if len(topA) == 0:
            st.info("No player possessions under current filters.")
        else:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.barh(topA["player"][::-1], topA["thirds_progressed"][::-1])
            ax.set_xlabel("Total thirds progressed (proxy)")
            ax.set_ylabel("Player")
            st.pyplot(fig, use_container_width=True)
            st.dataframe(topA, use_container_width=True, hide_index=True)

    with pr2:
        st.markdown(f"**Top ball progressors – {right_team}**")
        progB = player_third_progression(right_team)
        topB = progB.head(12)
        if len(topB) == 0:
            st.info("No player possessions under current filters.")
        else:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.barh(topB["player"][::-1], topB["thirds_progressed"][::-1])
            ax.set_xlabel("Total thirds progressed (proxy)")
            ax.set_ylabel("Player")
            st.pyplot(fig, use_container_width=True)
            st.dataframe(topB, use_container_width=True, hide_index=True)




# --- Player ball losses / turnovers ---
st.subheader("Who loses the ball the most?")

st.caption(
    "This uses the best available loss/turnover indicator in your dataset. "
    "If a dedicated turnover column isn't present, you may need to map event types to losses."
)

if player_col is None:
    st.info("No player_name column found, so ball-loss analysis is unavailable.")
else:
    # Heuristic: pick a loss column if present
    loss_candidates = [
        "possession_lost", "lost_possession", "is_possession_lost",
        "turnover", "is_turnover", "dispossessed", "is_dispossessed",
        "ball_lost", "loss", "team_possession_loss_in_phase"
    ]
    loss_col = None
    lower_cols = {c.lower(): c for c in dff.columns}
    for cand in loss_candidates:
        if cand in lower_cols:
            loss_col = lower_cols[cand]
            break

    # Fallback: if event_type exists, look for common loss event labels
    event_col = None
    for cand in ["event_type", "type", "event_name"]:
        if cand in dff.columns:
            event_col = cand
            break

    dd_loss = dff[dff[team_col].isin([left_team, right_team])].copy()
    dd_loss = dd_loss.dropna(subset=[player_col])

    if loss_col is not None:
        # Convert to boolean-ish
        loss_flag = dd_loss[loss_col].fillna(False)
        if loss_flag.dtype != bool:
            # handle strings like "True"/"False"
            loss_flag = loss_flag.astype(str).str.lower().isin(["true", "1", "yes", "y"])
        dd_loss["_loss_flag"] = loss_flag
        loss_source = f"Using column `{loss_col}` as the loss indicator."
    elif event_col is not None:
        # Basic mapping of event strings
        ev = dd_loss[event_col].astype(str).str.lower()
        dd_loss["_loss_flag"] = ev.str.contains("turnover|dispossess|miscontrol|lost|intercepted", regex=True)
        loss_source = f"No explicit loss column found; using keyword match on `{event_col}`."
    else:
        dd_loss["_loss_flag"] = False
        loss_source = "No loss indicator found (no suitable columns)."

    st.info(loss_source)

    def player_losses(team_name: str) -> pd.DataFrame:
        t = dd_loss[dd_loss[team_col] == team_name].copy()
        out = (
            t.groupby(player_col, as_index=False)
            .agg(
                player_possessions=(player_col, "size"),
                ball_losses=("_loss_flag", "sum"),
            )
            .rename(columns={player_col: "player"})
        )
        out["loss_rate"] = out["ball_losses"] / out["player_possessions"]
        out = out.sort_values(["ball_losses", "loss_rate"], ascending=False)
        return out

    l1, l2 = st.columns(2)
    with l1:
        st.markdown(f"**{left_team} – most ball losses**")
        LA = player_losses(left_team).head(15)
        if len(LA) == 0:
            st.info("No player possessions under current filters.")
        else:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.barh(LA["player"][::-1], LA["ball_losses"][::-1])
            ax.set_xlabel("Ball losses")
            ax.set_ylabel("Player")
            st.pyplot(fig, use_container_width=True)
            st.dataframe(LA, use_container_width=True, hide_index=True)

    with l2:
        st.markdown(f"**{right_team} – most ball losses**")
        LB = player_losses(right_team).head(15)
        if len(LB) == 0:
            st.info("No player possessions under current filters.")
        else:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.barh(LB["player"][::-1], LB["ball_losses"][::-1])
            ax.set_xlabel("Ball losses")
            ax.set_ylabel("Player")
            st.pyplot(fig, use_container_width=True)
            st.dataframe(LB, use_container_width=True, hide_index=True)


st.subheader("Which players combine the most? (pass links)")

target_col = "player_targeted_name" if "player_targeted_name" in dff.columns else None
if player_col is None or target_col is None:
    st.info("No player_targeted_name column found, so combination charts are unavailable.")
else:
    st.caption("Uses rows where `player_targeted_name` is present (treated as a pass to that player).")

    def top_links(team_name: str) -> pd.DataFrame:
        dd = dff[dff[team_col] == team_name].copy()
        dd = dd.dropna(subset=[player_col, target_col])

        # Keep links within the same team roster (prevents odd links if present)
        roster = set(dd[player_col].dropna().unique().tolist())
        dd = dd[dd[target_col].isin(roster)]

        dd = dd[dd[player_col] != dd[target_col]]

        links = (
            dd.groupby([player_col, target_col])
            .size()
            .rename("passes")
            .reset_index()
            .rename(columns={player_col: "from", target_col: "to"})
            .sort_values("passes", ascending=False)
        )

        # Combine bidirectionally (A<->B) for "combination" view
        a = links["from"].astype(str)
        b = links["to"].astype(str)
        links["pair"] = np.where(a < b, a + " ↔ " + b, b + " ↔ " + a)

        combo = links.groupby("pair", as_index=False)["passes"].sum().sort_values("passes", ascending=False)
        return combo

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Top combinations – {left_team}**")
        comboA = top_links(left_team).head(15)
        if len(comboA) == 0:
            st.info("No pass links found under current filters.")
        else:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.barh(comboA["pair"][::-1], comboA["passes"][::-1])
            ax.set_xlabel("Pass links (both directions)")
            ax.set_ylabel("Pair")
            st.pyplot(fig, use_container_width=True)
            st.dataframe(comboA, use_container_width=True, hide_index=True)

    with c2:
        st.markdown(f"**Top combinations – {right_team}**")
        comboB = top_links(right_team).head(15)
        if len(comboB) == 0:
            st.info("No pass links found under current filters.")
        else:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.barh(comboB["pair"][::-1], comboB["passes"][::-1])
            ax.set_xlabel("Pass links (both directions)")
            ax.set_ylabel("Pair")
            st.pyplot(fig, use_container_width=True)
            st.dataframe(comboB, use_container_width=True, hide_index=True)

# --- Data table & download ---


# --- Passing network (network-style graph) ---
st.subheader("Passing network (who connects with who?)")

if player_col is None or target_col is None:
    st.info("No player_targeted_name column found, so passing networks are unavailable.")
else:
    with st.sidebar:
        st.header("Network settings")
        show_bg_heatmap = st.checkbox("Background heatmap behind network", value=True)
        bg_heatmap_target = st.selectbox("Heatmap source", ["Possession end locations", "Possession start locations"], index=0)
        min_link_weight = st.slider("Minimum links to show", min_value=1, max_value=20, value=3, step=1)
        top_edges = st.slider("Max edges to draw", min_value=10, max_value=200, value=60, step=10)
        overlay_on_pitch = st.checkbox("Overlay network on pitch", value=True)
        loc_mode = st.selectbox("Player location", ["avg start location", "avg midpoint (start/end)"], index=0)
        fallback_layout = st.selectbox("Fallback layout (if no locations)", ["spring", "circular"], index=0)

    try:
        import networkx as nx
        NX_OK = True
    except Exception:
        NX_OK = False

    if not NX_OK:
        st.info("networkx is not installed (needed for passing networks). Install it via requirements.txt.")
    else:
        def build_pass_links(team_name: str) -> pd.DataFrame:
            dd = dff[dff[team_col] == team_name].copy()
            dd = dd.dropna(subset=[player_col, target_col])

            roster = set(dd[player_col].dropna().unique().tolist())
            dd = dd[dd[target_col].isin(roster)]
            dd = dd[dd[player_col] != dd[target_col]]

            links = (
                dd.groupby([player_col, target_col])
                .size()
                .rename("w")
                .reset_index()
                .rename(columns={player_col: "from", target_col: "to"})
                .sort_values("w", ascending=False)
            )

            links = links[links["w"] >= min_link_weight].head(top_edges)
            return links

        def build_positions(team_name: str) -> dict:
            dd = dff[dff[team_col] == team_name].copy()
            dd = dd.dropna(subset=[player_col, "x_start", "y_start", "x_end", "y_end"])

            if loc_mode == "avg midpoint (start/end)":
                dd["px"] = (dd["x_start"] + dd["x_end"]) / 2.0
                dd["py"] = (dd["y_start"] + dd["y_end"]) / 2.0
            else:
                dd["px"] = dd["x_start"]
                dd["py"] = dd["y_start"]

            pos_df = dd.groupby(player_col)[["px", "py"]].mean().reset_index()
            return {str(r[player_col]): (float(r["px"]), float(r["py"])) for _, r in pos_df.iterrows()}

        def links_to_graph(links: pd.DataFrame) -> nx.DiGraph:
            G = nx.DiGraph()
            for _, r in links.iterrows():
                G.add_edge(str(r["from"]), str(r["to"]), weight=float(r["w"]))
            return G

        def draw_layout_graph(ax, G, title: str, layout: str):
            ax.axis("off")
            ax.set_title(title)

            if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                ax.text(0.5, 0.5, "No links under current filters", ha="center", va="center")
                return

            if layout == "circular":
                pos = nx.circular_layout(G)
            else:
                pos = nx.spring_layout(G, seed=7, k=0.7)

            node_strength = {}
            for n in G.nodes():
                out_w = sum(d.get("weight", 1.0) for _, _, d in G.out_edges(n, data=True))
                in_w = sum(d.get("weight", 1.0) for _, _, d in G.in_edges(n, data=True))
                node_strength[n] = out_w + in_w

            strengths = np.array([node_strength[n] for n in G.nodes()], dtype=float)
            sizes = 300 + 1200 * (strengths / strengths.max()) if strengths.max() > 0 else 400

            weights = np.array([d.get("weight", 1.0) for _, _, d in G.edges(data=True)], dtype=float)
            widths = 0.5 + 4.0 * (weights / weights.max()) if weights.max() > 0 else 1.0

            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, alpha=0.9)
            nx.draw_networkx_edges(G, pos, ax=ax, width=widths, arrows=True, arrowsize=12, alpha=0.6)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)

        def draw_pitch_overlay(ax, G, positions: dict, title: str, team_name: str):
            # Optional background density heatmap
            if show_bg_heatmap and SCIPY_OK:
                src_x, src_y = ('x_end','y_end') if bg_heatmap_target.startswith('Possession end') else ('x_start','y_start')
                ddh = dff[dff[team_col] == team_name].dropna(subset=[src_x, src_y]).copy()
                if len(ddh) > 1:
                    _, _, zz = kde_density_grid(ddh[src_x].values, ddh[src_y].values)
                    ax.imshow(
                        zz.T,
                        origin='lower',
                        extent=[-52.5, 52.5, -34, 34],
                        alpha=0.35,
                        aspect='equal'
                    )
            draw_pitch(ax)
            ax.set_title(title)

            if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                ax.text(0, 0, "No links under current filters", ha="center", va="center")
                return

            # Keep only nodes that have positions
            nodes_with_pos = {n for n in G.nodes() if n in positions}
            if len(nodes_with_pos) < 2:
                ax.text(0, 0, "Not enough player locations (try changing Player location setting)", ha="center", va="center")
                return

            # Node strength for sizing
            node_strength = {}
            for n in nodes_with_pos:
                out_w = sum(d.get("weight", 1.0) for _, _, d in G.out_edges(n, data=True) if _ in nodes_with_pos or True)
                in_w = sum(d.get("weight", 1.0) for _, _, d in G.in_edges(n, data=True) if _ in nodes_with_pos or True)
                node_strength[n] = out_w + in_w

            strengths = np.array([node_strength[n] for n in nodes_with_pos], dtype=float)
            sizes = 80 + 520 * (strengths / strengths.max()) if strengths.max() > 0 else 200

            # Edge widths
            edges = [(u, v, d.get("weight", 1.0)) for u, v, d in G.edges(data=True) if u in nodes_with_pos and v in nodes_with_pos]
            if not edges:
                ax.text(0, 0, "No edges with player locations", ha="center", va="center")
                return

            weights = np.array([w for _, _, w in edges], dtype=float)
            widths = 0.3 + 3.0 * (weights / weights.max()) if weights.max() > 0 else 1.0

            # Draw edges as arrows
            for (u, v, w), lw in zip(edges, widths):
                x1, y1 = positions[u]
                x2, y2 = positions[v]
                ax.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=lw, alpha=0.55),
                )

            # Draw nodes + labels
            xs = [positions[n][0] for n in nodes_with_pos]
            ys = [positions[n][1] for n in nodes_with_pos]
            ax.scatter(xs, ys, s=sizes, alpha=0.9)

            for n in nodes_with_pos:
                x, y = positions[n]
                ax.text(x, y, n, fontsize=8, ha="center", va="center")

        def make_team_network(team_name: str):
            links = build_pass_links(team_name)
            G = links_to_graph(links)
            pos = build_positions(team_name) if overlay_on_pitch else {}
            return G, links, pos

        netA, linksA, posA = make_team_network(left_team)
        netB, linksB, posB = make_team_network(right_team)

        n1, n2 = st.columns(2)
        with n1:
            fig, ax = plt.subplots(figsize=(7, 6))
            if overlay_on_pitch:
                draw_pitch_overlay(ax, netA, posA, f"{left_team} passing network (pitch overlay)", str(left_team))
            else:
                draw_layout_graph(ax, netA, f"{left_team} passing network", layout=fallback_layout)
            st.pyplot(fig, use_container_width=True)
            with st.expander("Show edges table"):
                st.dataframe(linksA, use_container_width=True, hide_index=True)

        with n2:
            fig, ax = plt.subplots(figsize=(7, 6))
            if overlay_on_pitch:
                draw_pitch_overlay(ax, netB, posB, f"{right_team} passing network (pitch overlay)", str(right_team))
            else:
                draw_layout_graph(ax, netB, f"{right_team} passing network", layout=fallback_layout)
            st.pyplot(fig, use_container_width=True)
            with st.expander("Show edges table"):
                st.dataframe(linksB, use_container_width=True, hide_index=True)


st.divider()
st.subheader("Sequence table (one row per team possession sequence)")

show_cols = ["match_id", "team_possession_seq", "team", "duration", "n_player_possessions", "progression_x", "lead_to_shot", "lead_to_goal"]
for c in ["third_end", "penalty_area_end"]:
    if c in seq_summary.columns:
        show_cols.append(c)

st.dataframe(seq_summary[show_cols].copy(), use_container_width=True, hide_index=True)

csv_out = seq_summary.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download sequence summary as CSV",
    data=csv_out,
    file_name="sequence_summary.csv",
    mime="text/csv",
)


st.header("Effectiveness dashboard (all teams)")

st.caption(
    "This section compares possession-sequence effectiveness across teams using one-row-per-sequence data. "
    "Filters apply at the **sequence level** (not individual player possessions)."
)

# --- Build sequence-level helper columns ---
seq_eff = seq_summary_all.copy()

# Add thirds labels for starts/ends (always available from x coords)
seq_eff["start_third"] = seq_eff["start_x"].apply(label_third_from_x) if "start_x" in seq_eff.columns else "Unknown"
seq_eff["end_third"] = seq_eff["end_x"].apply(label_third_from_x) if "end_x" in seq_eff.columns else "Unknown"
seq_eff["thirds_progressed_seq"] = [thirds_progressed(s, e) for s, e in zip(seq_eff.get("start_x", np.nan), seq_eff.get("end_x", np.nan))]

# --- Sequence-level filters ---
with st.sidebar:
    st.header("Effectiveness filters")

    # Team selection for this section
    eff_teams_all = sorted(seq_eff["team"].dropna().astype(str).unique().tolist())
    eff_teams = st.multiselect("Teams to include (effectiveness)", eff_teams_all, default=eff_teams_all)

    # Game state / score filters if present
    gs_col = None
    for cand in ["game_state", "gameState", "possession_game_state", "match_state"]:
        if cand in seq_eff.columns:
            gs_col = cand
            break

    score_diff_col = None
    for cand in ["score_diff", "goal_diff", "score_difference", "scoreDelta"]:
        if cand in seq_eff.columns:
            score_diff_col = cand
            break

    # Time filters if present
    minute_col = None
    for cand in ["minute_start", "minute", "start_minute"]:
        if cand in seq_eff.columns:
            minute_col = cand
            break

# Apply filters
if eff_teams:
    seq_eff = seq_eff[seq_eff["team"].astype(str).isin([str(t) for t in eff_teams])].copy()

# Game state filter
if gs_col is not None:
    with st.sidebar:
        gs_opts = ["All"] + sorted([x for x in seq_eff[gs_col].dropna().unique().tolist()])
        gs_choice = st.selectbox("Game state", gs_opts, index=0)
    if gs_choice != "All":
        seq_eff = seq_eff[seq_eff[gs_col] == gs_choice].copy()
else:
    with st.sidebar:
        st.caption("Game state filter: column not found (skipping).")

# Score diff filter
if score_diff_col is not None:
    seq_eff[score_diff_col] = pd.to_numeric(seq_eff[score_diff_col], errors="coerce")
    with st.sidebar:
        sd_min = float(np.nanmin(seq_eff[score_diff_col].values)) if len(seq_eff) else -3.0
        sd_max = float(np.nanmax(seq_eff[score_diff_col].values)) if len(seq_eff) else 3.0
        # keep sane bounds
        sd_min = min(sd_min, 0.0)
        sd_max = max(sd_max, 0.0)
        sd_range = st.slider("Score diff range", min_value=float(sd_min), max_value=float(sd_max), value=(float(sd_min), float(sd_max)))
    seq_eff = seq_eff[(seq_eff[score_diff_col] >= sd_range[0]) & (seq_eff[score_diff_col] <= sd_range[1])].copy()
else:
    with st.sidebar:
        st.caption("Score-diff filter: column not found (skipping).")

# Minute range filter
if minute_col is not None:
    seq_eff[minute_col] = pd.to_numeric(seq_eff[minute_col], errors="coerce")
    with st.sidebar:
        m_min = int(np.nanmin(seq_eff[minute_col].values)) if len(seq_eff) else 0
        m_max = int(np.nanmax(seq_eff[minute_col].values)) if len(seq_eff) else 90
        m_range = st.slider("Minute range", min_value=int(m_min), max_value=int(m_max), value=(int(m_min), int(m_max)))
    seq_eff = seq_eff[(seq_eff[minute_col] >= m_range[0]) & (seq_eff[minute_col] <= m_range[1])].copy()
else:
    with st.sidebar:
        st.caption("Minute filter: column not found (skipping).")

# Start/end third filters
with st.sidebar:
    st.subheader("Location filters (sequence-level)")
    start_third_choice = st.multiselect("Start third", ["Defensive", "Middle", "Attacking"], default=["Defensive", "Middle", "Attacking"])
    end_third_choice = st.multiselect("End third", ["Defensive", "Middle", "Attacking"], default=["Defensive", "Middle", "Attacking"])
    min_thirds_prog = st.slider("Min thirds progressed", min_value=0, max_value=2, value=0)

seq_eff = seq_eff[seq_eff["start_third"].isin(start_third_choice)].copy()
seq_eff = seq_eff[seq_eff["end_third"].isin(end_third_choice)].copy()
seq_eff = seq_eff[seq_eff["thirds_progressed_seq"] >= min_thirds_prog].copy()

# Optional: filter by in-possession phase type if present (sequence-level best-effort)
if "team_in_possession_phase_type" in dff_all.columns:
    with st.sidebar:
        phase_seq_opts = ["All"] + sorted([x for x in dff_all["team_in_possession_phase_type"].dropna().unique().tolist()])
        phase_seq_choice = st.selectbox("Phase (in-possession)", phase_seq_opts, index=0)
    if phase_seq_choice != "All":
        # Approx: keep sequences that contain at least one row with this phase type
        keep = (
            dff_all[dff_all["team_in_possession_phase_type"] == phase_seq_choice][["match_id", "team_possession_seq"]]
            .dropna()
            .drop_duplicates()
        )
        seq_eff = seq_eff.merge(keep, on=["match_id", "team_possession_seq"], how="inner")

# --- Summary table: effectiveness per team ---
def team_effectiveness_table(d: pd.DataFrame) -> pd.DataFrame:
    if len(d) == 0:
        return pd.DataFrame(columns=["team", "sequences", "shot_rate", "goal_rate", "avg_duration", "avg_x_prog", "end_final_third_rate", "end_pen_area_rate"])

    out = d.groupby("team", as_index=False).agg(
        sequences=("team", "size"),
        shot_rate=("lead_to_shot", lambda s: pd.Series(s).fillna(False).astype(bool).mean()),
        goal_rate=("lead_to_goal", lambda s: pd.Series(s).fillna(False).astype(bool).mean()),
        avg_duration=("duration", "mean"),
        avg_x_prog=("progression_x", "mean"),
        avg_thirds_progressed=("thirds_progressed_seq", "mean"),
    )

    # End zone rates if present
    if "third_end" in d.columns and d["third_end"].notna().any():
        out["end_final_third_rate"] = d.groupby("team")["third_end"].apply(lambda s: s.astype(str).str.contains("attacking", case=False, na=False).mean()).values
    else:
        out["end_final_third_rate"] = np.nan

    if "penalty_area_end" in d.columns and d["penalty_area_end"].notna().any():
        out["end_pen_area_rate"] = d.groupby("team")["penalty_area_end"].apply(lambda s: pd.Series(s).fillna(False).astype(bool).mean()).values
    else:
        out["end_pen_area_rate"] = np.nan

    return out.sort_values("sequences", ascending=False)

team_tbl = team_effectiveness_table(seq_eff)

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Teams in view", int(team_tbl["team"].nunique()) if len(team_tbl) else 0)
kpi2.metric("Sequences in view", int(seq_eff.shape[0]))
kpi3.metric("Avg thirds progressed (all)", f"{seq_eff['thirds_progressed_seq'].mean():.2f}" if len(seq_eff) else "—")

tabs = st.tabs(["Overview", "Shot & territory", "Progression & tempo", "Start vs end geography"])

with tabs[0]:
    st.subheader("Team effectiveness table")
    st.dataframe(team_tbl, use_container_width=True, hide_index=True)

    if len(team_tbl) == 0:
        st.info("No sequences available under current effectiveness filters.")
    else:
        st.subheader("Quick leaderboard")
        metric_key = st.selectbox(
            "Leaderboard metric",
            ["shot_rate", "goal_rate", "avg_duration", "avg_x_prog", "avg_thirds_progressed"],
            index=0,
        )
        metric_label_map = {
            "shot_rate": "Shot rate",
            "goal_rate": "Goal rate",
            "avg_duration": "Avg duration",
            "avg_x_prog": "Avg x progression",
            "avg_thirds_progressed": "Avg thirds progressed",
        }
        metric_label = metric_label_map.get(metric_key, metric_key)

        fig, ax = plt.subplots(figsize=(10, 5))
        plot_df = team_tbl.sort_values(metric_key, ascending=False).copy()
        ax.bar(plot_df["team"].astype(str), plot_df[metric_key].astype(float))
        ax.set_title(f"Teams by {metric_label}")
        ax.set_ylabel(metric_label)
        ax.set_xticklabels(plot_df["team"].astype(str), rotation=20, ha="right")
        st.pyplot(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Shot & territory")
    if len(team_tbl) == 0:
        st.info("No sequences available under current effectiveness filters.")
    else:
        # Shot / goal rates + territory proxies (final third / box if available)
        cols = st.columns(2)
        with cols[0]:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            plot_df = team_tbl.sort_values("shot_rate", ascending=False).copy()
            ax.bar(plot_df["team"].astype(str), plot_df["shot_rate"].astype(float))
            ax.set_title("Shot rate by team")
            ax.set_ylabel("Shot rate")
            ax.set_xticklabels(plot_df["team"].astype(str), rotation=20, ha="right")
            st.pyplot(fig, use_container_width=True)

        with cols[1]:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            plot_df = team_tbl.sort_values("goal_rate", ascending=False).copy()
            ax.bar(plot_df["team"].astype(str), plot_df["goal_rate"].astype(float))
            ax.set_title("Goal rate by team")
            ax.set_ylabel("Goal rate")
            ax.set_xticklabels(plot_df["team"].astype(str), rotation=20, ha="right")
            st.pyplot(fig, use_container_width=True)

        # Territory: end in final third / penalty area (if available)
        terr_cols = [c for c in ["end_final_third_rate", "end_pen_area_rate"] if c in team_tbl.columns and team_tbl[c].notna().any()]
        if terr_cols:
            for c in terr_cols:
                fig, ax = plt.subplots(figsize=(10, 4.5))
                plot_df = team_tbl.sort_values(c, ascending=False).copy()
                ax.bar(plot_df["team"].astype(str), plot_df[c].astype(float))
                ax.set_title(f"{c.replace('_', ' ').title()} by team")
                ax.set_ylabel("Rate")
                ax.set_xticklabels(plot_df["team"].astype(str), rotation=20, ha="right")
                st.pyplot(fig, use_container_width=True)
        else:
            st.info("Territory end-zone rates not available in this dataset/filters.")

with tabs[2]:
    st.subheader("Progression & tempo")
    if len(team_tbl) == 0:
        st.info("No sequences available under current effectiveness filters.")
    else:
        cols = st.columns(2)
        with cols[0]:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            plot_df = team_tbl.sort_values("avg_x_prog", ascending=False).copy()
            ax.bar(plot_df["team"].astype(str), plot_df["avg_x_prog"].astype(float))
            ax.set_title("Average x progression by team")
            ax.set_ylabel("Avg x progression")
            ax.set_xticklabels(plot_df["team"].astype(str), rotation=20, ha="right")
            st.pyplot(fig, use_container_width=True)

        with cols[1]:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            plot_df = team_tbl.sort_values("avg_duration", ascending=False).copy()
            ax.bar(plot_df["team"].astype(str), plot_df["avg_duration"].astype(float))
            ax.set_title("Average sequence duration by team")
            ax.set_ylabel("Seconds")
            ax.set_xticklabels(plot_df["team"].astype(str), rotation=20, ha="right")
            st.pyplot(fig, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        plot_df = team_tbl.sort_values("avg_thirds_progressed", ascending=False).copy()
        ax.bar(plot_df["team"].astype(str), plot_df["avg_thirds_progressed"].astype(float))
        ax.set_title("Average thirds progressed by team")
        ax.set_ylabel("Thirds progressed (avg)")
        ax.set_xticklabels(plot_df["team"].astype(str), rotation=20, ha="right")
        st.pyplot(fig, use_container_width=True)

with tabs[3]:
    st.subheader("Start vs end geography (filtered sequences)")
    if len(seq_eff) < 2:
        st.info("Not enough sequences under current filters to draw KDE maps.")
    else:
        tcols = st.columns(2)
        with tcols[0]:
            fig, ax = plt.subplots(figsize=(8, 5))
            img = plot_kde(ax, seq_eff["start_x"].dropna().values, seq_eff["start_y"].dropna().values)
            ax.set_title("START locations (all selected teams)")
            if img is not None:
                plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig, use_container_width=True)

        with tcols[1]:
            fig, ax = plt.subplots(figsize=(8, 5))
            img = plot_kde(ax, seq_eff["end_x"].dropna().values, seq_eff["end_y"].dropna().values)
            ax.set_title("END locations (all selected teams)")
            if img is not None:
                plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig, use_container_width=True)
