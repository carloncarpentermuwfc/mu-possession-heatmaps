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
    for i, line in enumerate(text[:100]):
        s = line.strip()
        if s.startswith("event_id,") and "match_id" in s and ("x_end" in s or "y_end" in s):
            header_idx = i
            break

    if header_idx is None:
        # Fallback: try reading as-is
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
    # Outline
    ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5],
            [-34, -34, 34, 34, -34], linewidth=1)

    # Halfway line
    ax.plot([0, 0], [-34, 34], linewidth=1)

    # Centre circle (9.15m)
    centre_circle = plt.Circle((0, 0), 9.15, fill=False, linewidth=1)
    ax.add_artist(centre_circle)

    # Penalty areas (16.5m from goal line, 40.3m wide -> y +/-20.15)
    # Left
    ax.plot([-52.5, -36], [-20.15, -20.15], linewidth=1)
    ax.plot([-36, -36], [-20.15, 20.15], linewidth=1)
    ax.plot([-36, -52.5], [20.15, 20.15], linewidth=1)
    # Right
    ax.plot([52.5, 36], [-20.15, -20.15], linewidth=1)
    ax.plot([36, 36], [-20.15, 20.15], linewidth=1)
    ax.plot([36, 52.5], [20.15, 20.15], linewidth=1)

    ax.set_aspect("equal")
    ax.set_xlim(-55, 55)
    ax.set_ylim(-37, 37)
    ax.axis("off")


# ----------------------------
# Aggregation: end of TEAM possession sequences
# ----------------------------
def get_team_possession_ends(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """
    Uses last_player_possession_in_team_possession == True when present.
    If the flag is missing, returns the original df and a warning message.
    """
    if "last_player_possession_in_team_possession" in df.columns:
        out = df[df["last_player_possession_in_team_possession"] == True].copy()
        return out, None

    return df.copy(), (
        "Column 'last_player_possession_in_team_possession' not found. "
        "Falling back to using ALL rows (not true team-possession aggregation)."
    )


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
st.set_page_config(page_title="MU Possession KDE Comparison", layout="wide")
st.title("Possession Analysis – KDE Heatmaps (Team Possession Sequence Ends)")

st.caption(
    "Upload the possessions CSV (or place it at `data/possessions.csv`). "
    "This app plots KDE heatmaps of *team possession sequence end locations* and enables team comparisons."
)

# --- Input source: local file or upload ---
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

# Basic checks
required = {"match_id", "x_end", "y_end"}
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df["x_end"] = pd.to_numeric(df["x_end"], errors="coerce")
df["y_end"] = pd.to_numeric(df["y_end"], errors="coerce")

team_col = "team_shortname" if "team_shortname" in df.columns else ("team_id" if "team_id" in df.columns else None)
if team_col is None:
    st.error("Could not find a team identifier column (team_shortname or team_id).")
    st.stop()

# --- Sidebar filters ---
with st.sidebar:
    st.header("Filters")

match_ids = sorted(df["match_id"].dropna().unique().tolist())
selected_matches = st.multiselect("Match(es)", match_ids, default=match_ids)

dff = df[df["match_id"].isin(selected_matches)].copy()

# Optional filters
if "team_in_possession_phase_type" in dff.columns:
    phase_opts = ["All"] + sorted([x for x in dff["team_in_possession_phase_type"].dropna().unique().tolist()])
    phase_choice = st.selectbox("In-possession phase", phase_opts, index=0)
    if phase_choice != "All":
        dff = dff[dff["team_in_possession_phase_type"] == phase_choice]

# If present, allow filtering by half / period
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

# Aggregate
ends, agg_warning = get_team_possession_ends(dff)
if agg_warning:
    st.warning(agg_warning)

teams = sorted(ends[team_col].dropna().unique().tolist())
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

# Comparison scope
scope = st.sidebar.radio(
    "Comparison scope",
    ["Across selected matches", "Only matches where both teams appear"],
    index=0,
    help="Second option restricts to match_ids containing both Team A and Team B.",
)

if scope == "Only matches where both teams appear":
    matches_with_A = set(ends.loc[ends[team_col] == left_team, "match_id"].dropna().unique().tolist())
    matches_with_B = set(ends.loc[ends[team_col] == right_team, "match_id"].dropna().unique().tolist())
    common_matches = sorted(list(matches_with_A.intersection(matches_with_B)))
    ends = ends[ends["match_id"].isin(common_matches)].copy()

A = ends[(ends[team_col] == left_team)].dropna(subset=["x_end", "y_end"])
B = ends[(ends[team_col] == right_team)].dropna(subset=["x_end", "y_end"])

# --- Metrics ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Matches selected", len(selected_matches))
c2.metric(f"{left_team} ends", len(A))
c3.metric(f"{right_team} ends", len(B))
c4.metric("KDE available", "Yes" if SCIPY_OK else "No (install scipy)")

# --- Plots ---
colA, colB = st.columns(2)

with colA:
    fig, ax = plt.subplots(figsize=(8, 5))
    img = plot_kde(ax, A["x_end"].values, A["y_end"].values)
    ax.set_title(f"{left_team} – Team Possession End KDE")
    if img is not None:
        plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, use_container_width=True)

with colB:
    fig, ax = plt.subplots(figsize=(8, 5))
    img = plot_kde(ax, B["x_end"].values, B["y_end"].values)
    ax.set_title(f"{right_team} – Team Possession End KDE")
    if img is not None:
        plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, use_container_width=True)

st.divider()
st.subheader("Difference View (A – B)")

if SCIPY_OK and len(A) > 1 and len(B) > 1:
    xmin, xmax = -52.5, 52.5
    ymin, ymax = -34, 34
    xx, yy = np.mgrid[xmin:xmax:300j, ymin:ymax:200j]
    pos = np.vstack([xx.ravel(), yy.ravel()])

    kdeA = gaussian_kde(np.vstack([A["x_end"].values, A["y_end"].values]))
    kdeB = gaussian_kde(np.vstack([B["x_end"].values, B["y_end"].values]))
    zA = np.reshape(kdeA(pos).T, xx.shape)
    zB = np.reshape(kdeB(pos).T, xx.shape)
    zD = zA - zB

    fig, ax = plt.subplots(figsize=(10, 6))
    draw_pitch(ax)
    img = ax.imshow(
        zD.T,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        alpha=0.85,
        aspect="equal",
    )
    ax.set_title(f"Density Difference: {left_team} – {right_team}")
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, use_container_width=True)
else:
    st.info("Difference view needs scipy and at least 2 points per team.")

st.divider()
st.subheader("Filtered team possession-sequence ends (table)")

# Show a useful subset if columns exist
cols_show = [team_col, "match_id", "x_end", "y_end"]
for c in ["time_start", "time_end", "team_in_possession_phase_type"]:
    if c in ends.columns:
        cols_show.append(c)

st.dataframe(ends[cols_show].copy(), use_container_width=True)

csv_out = ends.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered ends as CSV",
    data=csv_out,
    file_name="filtered_team_possession_ends.csv",
    mime="text/csv",
)
