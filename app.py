import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# KDE
try:
    from scipy.stats import gaussian_kde
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


@st.cache_data(show_spinner=False)
def load_possessions(file_bytes: bytes) -> pd.DataFrame:
    """
    Robust CSV loader:
    - Handles comment/header junk at the top of a file
    - Finds the real header row if needed
    """
    text = file_bytes.decode("utf-8", errors="ignore").splitlines()

    header_idx = None
    for i, line in enumerate(text[:80]):
        line_stripped = line.strip()
        # look for a likely header row
        if line_stripped.startswith("event_id,") and "match_id" in line_stripped and "x_end" in line_stripped:
            header_idx = i
            break

    if header_idx is None:
        return pd.read_csv(io.BytesIO(file_bytes))

    data_str = "\n".join(text[header_idx:])
    return pd.read_csv(io.StringIO(data_str), engine="python")


def draw_pitch(ax):
    # Outline
    ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5],
            [-34, -34, 34, 34, -34], linewidth=1)

    # Halfway line
    ax.plot([0, 0], [-34, 34], linewidth=1)

    # Centre circle (9.15m)
    centre_circle = plt.Circle((0, 0), 9.15, fill=False, linewidth=1)
    ax.add_artist(centre_circle)

    # Penalty areas (16.5m from goal line; width 40.3m -> y +/- 20.15)
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


def get_team_possession_ends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates to TEAM possession sequences by selecting the final player possession
    for each team possession, when the boolean flag exists.
    """
    if "last_player_possession_in_team_possession" in df.columns:
        return df[df["last_player_possession_in_team_possession"] == True].copy()

    # If the explicit flag isn't present, we can't reliably aggregate.
    return df.copy()


def kde_heatmap(ax, x, y, gridsize=(300, 200), alpha=0.85):
    xmin, xmax = -52.5, 52.5
    ymin, ymax = -34, 34

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2 or len(y) < 2:
        draw_pitch(ax)
        ax.set_title("Not enough points for KDE")
        return None

    if not SCIPY_OK:
        draw_pitch(ax)
        ax.set_title("scipy not installed (needed for KDE)")
        return None

    xx, yy = np.mgrid[xmin:xmax:complex(gridsize[0]), ymin:ymax:complex(gridsize[1])]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])

    kde = gaussian_kde(values)  # Scott's rule bandwidth by default
    zz = np.reshape(kde(positions).T, xx.shape)

    draw_pitch(ax)
    img = ax.imshow(
        zz.T,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        alpha=alpha,
        aspect="equal",
    )
    return img


def main():
    st.set_page_config(page_title="Possession End KDE Heatmaps", layout="wide")
    st.title("Possession Analysis – KDE Heatmaps (Team Possession Sequence Ends)")

    uploaded = st.file_uploader("Upload your possessions CSV", type=["csv"])
    st.caption("Expected columns include: match_id, x_end, y_end, and a team identifier (team_shortname or team_id).")

    if not uploaded:
        st.stop()

    df = load_possessions(uploaded.getvalue())

    # Required columns
    required = {"match_id", "x_end", "y_end"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # Coerce coords
    df["x_end"] = pd.to_numeric(df["x_end"], errors="coerce")
    df["y_end"] = pd.to_numeric(df["y_end"], errors="coerce")

    # Team column
    team_col = "team_shortname" if "team_shortname" in df.columns else ("team_id" if "team_id" in df.columns else None)
    if team_col is None:
        st.error("Could not find a team identifier column (team_shortname or team_id).")
        st.stop()

    st.sidebar.header("Filters")

    match_ids = sorted(df["match_id"].dropna().unique().tolist())
    selected_matches = st.sidebar.multiselect("Match(es)", match_ids, default=match_ids)
    dff = df[df["match_id"].isin(selected_matches)].copy()

    # Optional filters
    if "team_in_possession_phase_type" in dff.columns:
        phase_opts = ["All"] + sorted(dff["team_in_possession_phase_type"].dropna().unique().tolist())
        phase_choice = st.sidebar.selectbox("In-possession phase", phase_opts, index=0)
        if phase_choice != "All":
            dff = dff[dff["team_in_possession_phase_type"] == phase_choice]

    if "possession_direction" in dff.columns:
        dir_opts = ["All"] + sorted(dff["possession_direction"].dropna().unique().tolist())
        dir_choice = st.sidebar.selectbox("Possession direction", dir_opts, index=0)
        if dir_choice != "All":
            dff = dff[dff["possession_direction"] == dir_choice]

    # Aggregate to team possession ends
    ends = get_team_possession_ends(dff)

    if "last_player_possession_in_team_possession" not in df.columns:
        st.warning(
            "Column 'last_player_possession_in_team_possession' not found. "
            "Aggregation to team possession sequences may be unreliable; using all rows instead."
        )

    teams = sorted(ends[team_col].dropna().unique().tolist())
    if not teams:
        st.error("No teams found after filtering.")
        st.stop()

    # Choose default MU if possible
    default_mu = None
    for t in teams:
        ts = str(t).lower()
        if ("man" in ts and "utd" in ts) or ("manchester" in ts and "united" in ts) or ("utd" in ts and "man" in ts):
            default_mu = t
            break
    if default_mu is None:
        default_mu = teams[0]

    team_a = st.sidebar.selectbox("Team A", teams, index=teams.index(default_mu) if default_mu in teams else 0)
    other = [t for t in teams if t != team_a]
    if not other:
        st.info("Only one team available after filtering.")
        st.stop()
    team_b = st.sidebar.selectbox("Team B", other, index=0)

    A = ends[ends[team_col] == team_a].dropna(subset=["x_end", "y_end"])
    B = ends[ends[team_col] == team_b].dropna(subset=["x_end", "y_end"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches selected", len(selected_matches))
    c2.metric(f"{team_a} (team-seq ends)", len(A))
    c3.metric(f"{team_b} (team-seq ends)", len(B))
    c4.metric("KDE available", "Yes" if SCIPY_OK else "No (install scipy)")

    colA, colB = st.columns(2)

    with colA:
        fig, ax = plt.subplots(figsize=(8, 5))
        img = kde_heatmap(ax, A["x_end"].values, A["y_end"].values)
        ax.set_title(f"{team_a} – Team Possession End KDE")
        if img is not None:
            plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, use_container_width=True)

    with colB:
        fig, ax = plt.subplots(figsize=(8, 5))
        img = kde_heatmap(ax, B["x_end"].values, B["y_end"].values)
        ax.set_title(f"{team_b} – Team Possession End KDE")
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
        ax.set_title(f"Density Difference: {team_a} – {team_b}")
        plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Difference view needs scipy and at least 2 points per team.")

    st.divider()
