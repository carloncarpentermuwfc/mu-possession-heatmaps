import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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
    """
    sort_cols = ["match_id", "team_possession_seq", "index"] if "index" in df.columns else ["match_id", "team_possession_seq", "frame_start"]
    d = df.sort_values(sort_cols).copy()

    gb = d.groupby(["match_id", "team_possession_seq"], as_index=False)

    def first(series):
        return series.iloc[0]

    def last(series):
        return series.iloc[-1]

    out = gb.agg(
        team=(team_col, first),
        start_x=("x_start", first),
        start_y=("y_start", first),
        end_x=("x_end", last),
        end_y=("y_end", last),
        duration=("duration", "sum"),
        n_player_possessions=("event_id", "size"),
    )

    # Optional: "lead_to_shot"/"lead_to_goal" are great outcome flags if present
    if "lead_to_shot" in d.columns:
        out["lead_to_shot"] = gb["lead_to_shot"].max()
    else:
        out["lead_to_shot"] = False

    if "lead_to_goal" in d.columns:
        out["lead_to_goal"] = gb["lead_to_goal"].max()
    else:
        out["lead_to_goal"] = False

    # End location categories if present
    if "third_end" in d.columns:
        out["third_end"] = gb["third_end"].apply(lambda s: s.iloc[-1])
    else:
        out["third_end"] = np.nan

    if "penalty_area_end" in d.columns:
        out["penalty_area_end"] = gb["penalty_area_end"].apply(lambda s: s.iloc[-1])
    else:
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

# Optional filters (phase and half)
with st.sidebar:
    if "team_in_possession_phase_type" in dff.columns:
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
dff, seq_warn = add_team_possession_sequence_id(dff)
if seq_warn:
    st.warning(seq_warn)

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
st.subheader("Who is most involved?")

player_col = "player_name" if "player_name" in dff.columns else None
if player_col is None:
    st.info("No player_name column found, so player involvement charts are unavailable.")
else:
    p1, p2 = st.columns(2)

    def player_involvement(team_name: str):
        dd = dff[dff[team_col] == team_name].dropna(subset=[player_col])
        # Total involvements (count of player possessions)
        total = dd[player_col].value_counts().rename("player_possessions").reset_index().rename(columns={"index": "player"})
        # Unique sequences involved in
        seqs = dd.groupby(player_col)["team_possession_seq"].nunique().rename("sequences_involved").reset_index().rename(columns={player_col: "player"})
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
        min_link_weight = st.slider("Minimum links to show", min_value=1, max_value=20, value=3, step=1)
        top_edges = st.slider("Max edges to draw", min_value=10, max_value=200, value=60, step=10)
        layout_choice = st.selectbox("Layout", ["spring", "circular"], index=0)

    try:
        import networkx as nx
        NX_OK = True
    except Exception:
        NX_OK = False

    if not NX_OK:
        st.info("networkx is not installed (needed for passing networks). Install it via requirements.txt.")
    else:
        def build_pass_graph(team_name: str):
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

            # Filter + cap edges
            links = links[links["w"] >= min_link_weight].head(top_edges)

            G = nx.DiGraph()
            for _, r in links.iterrows():
                G.add_edge(str(r["from"]), str(r["to"]), weight=float(r["w"]))
            return G, links

        def draw_graph(ax, G, title: str):
            ax.axis("off")
            ax.set_title(title)

            if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                ax.text(0.5, 0.5, "No links under current filters", ha="center", va="center")
                return

            if layout_choice == "circular":
                pos = nx.circular_layout(G)
            else:
                # spring layout with fixed seed for stability
                pos = nx.spring_layout(G, seed=7, k=0.7)

            # Node sizes: total involvement (in+out degree weighted)
            node_strength = {}
            for n in G.nodes():
                out_w = sum(d.get("weight", 1.0) for _, _, d in G.out_edges(n, data=True))
                in_w = sum(d.get("weight", 1.0) for _, _, d in G.in_edges(n, data=True))
                node_strength[n] = out_w + in_w

            strengths = np.array([node_strength[n] for n in G.nodes()], dtype=float)
            # Scale sizes nicely
            sizes = 300 + 1200 * (strengths / strengths.max()) if strengths.max() > 0 else 400

            # Edge widths proportional to weight
            weights = np.array([d.get("weight", 1.0) for _, _, d in G.edges(data=True)], dtype=float)
            widths = 0.5 + 4.0 * (weights / weights.max()) if weights.max() > 0 else 1.0

            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, alpha=0.9)
            nx.draw_networkx_edges(G, pos, ax=ax, width=widths, arrows=True, arrowsize=12, alpha=0.6)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)

        netA, linksA = build_pass_graph(left_team)
        netB, linksB = build_pass_graph(right_team)

        n1, n2 = st.columns(2)
        with n1:
            fig, ax = plt.subplots(figsize=(7, 6))
            draw_graph(ax, netA, f"{left_team} passing network (filtered)")
            st.pyplot(fig, use_container_width=True)

            with st.expander("Show edges table"):
                st.dataframe(linksA, use_container_width=True, hide_index=True)

        with n2:
            fig, ax = plt.subplots(figsize=(7, 6))
            draw_graph(ax, netB, f"{right_team} passing network (filtered)")
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
