
# app.py
# Streamlit: Crossing Dashboard + Full Team Comparison + Possession Effectiveness (inferred possessions)
#
# What you get (all requested):
# - Crossing pitch map (your existing Plotly pitch code stays, with filters)
# - Team-by-team comparison charts
# - Possession sequence effectiveness (inferred possessions)
# - Speed vs Control (passes per possession, duration)
# - Directness (meters progressed per possession)
# - Crossing-possession effectiveness toggle
# - Counter-attack detection (heuristic) + team comparison
# - Possession outcome breakdown (stacked bars)
# - Possession flow Sankey (build → cross → box entry → shot / turnover)
# - Percentile ranks table (league-wide)
# - Match-state filtering (if score columns exist; otherwise hidden)
# - Download buttons for filtered data + team tables
#
# Requirements (requirements.txt):
#   streamlit
#   pandas
#   numpy
#   plotly
#
# If you see "No module named plotly", add plotly to requirements.txt and redeploy / clear cache.

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Crossing & Possession Effectiveness", layout="wide")

DEFAULT_CSV = "Events.csv"

# ----------------------------
# Data loading
# ----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_data(uploaded) -> pd.DataFrame:
    if uploaded is not None:
        st.caption("Using uploaded file.")
        return pd.read_csv(uploaded)
    if os.path.exists(DEFAULT_CSV):
        st.caption(f"Using local file: {DEFAULT_CSV}")
        return load_csv(DEFAULT_CSV)
    return None

# ----------------------------
# Helpers (crossing pitch)
# ----------------------------
def build_pitch_shapes_vertical(length=120, width=80):
    """Vertical pitch (attack up): x axis = width (0..80), y axis = length (0..120)"""
    shapes = []
    shapes.append(dict(type="rect", x0=0, y0=0, x1=width, y1=length, line=dict(width=2)))
    shapes.append(dict(type="line", x0=0, y0=length / 2, x1=width, y1=length / 2, line=dict(width=2)))

    pa_y = 18
    pa_x0, pa_x1 = 18, 62
    shapes.append(dict(type="rect", x0=pa_x0, y0=0, x1=pa_x1, y1=pa_y, line=dict(width=2)))
    shapes.append(dict(type="rect", x0=pa_x0, y0=length - pa_y, x1=pa_x1, y1=length, line=dict(width=2)))

    sb_y = 6
    sb_x0, sb_x1 = 30, 50
    shapes.append(dict(type="rect", x0=sb_x0, y0=0, x1=sb_x1, y1=sb_y, line=dict(width=2)))
    shapes.append(dict(type="rect", x0=sb_x0, y0=length - sb_y, x1=sb_x1, y1=length, line=dict(width=2)))

    shapes.append(
        dict(
            type="circle",
            x0=width / 2 - 10,
            y0=length / 2 - 10,
            x1=width / 2 + 10,
            y1=length / 2 + 10,
            line=dict(width=2),
        )
    )
    return shapes

def make_segment_trace(x0, y0, x1, y1, color: str):
    """Many line segments separated by NaN."""
    xs = np.column_stack([x0, x1, np.full_like(x0, np.nan)]).ravel()
    ys = np.column_stack([y0, y1, np.full_like(y0, np.nan)]).ravel()
    return go.Scattergl(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(width=2, color=color),
        hoverinfo="skip",
        showlegend=False,
    )

def classify_zone(start_x: float, start_y: float) -> str:
    """Start-zone classification using StatsBomb-ish coords: X 0..120, Y 0..80"""
    if start_x < 80:
        return "Other"
    wide = (start_y <= 20) or (start_y >= 60)
    halfspace = ((20 < start_y <= 34) or (46 <= start_y < 60))
    if wide and start_x >= 114:
        return "Cutbacks & stand ups"
    if wide and 108 <= start_x < 114:
        return "Driven"
    if wide and 96 <= start_x < 108:
        return "Whipped"
    if halfspace and 84 <= start_x < 108:
        return "Diagonals"
    return "Other"

def add_zone_overlays(fig: go.Figure, attack_up: bool, flip_len: bool, invert_width: bool):
    def maybe_flip_x(x):
        return 120 - x if flip_len else x
    def width_map(y):
        return 80 - y if invert_width else y
    def rect(xmin, xmax, ymin, ymax):
        x0, x1 = maybe_flip_x(xmin), maybe_flip_x(xmax)
        x_low, x_high = (min(x0, x1), max(x0, x1))
        wy0 = width_map(ymin)
        wy1 = width_map(ymax)
        w_low, w_high = (min(wy0, wy1), max(wy0, wy1))
        if attack_up:
            return dict(type="rect", x0=w_low, x1=w_high, y0=x_low, y1=x_high,
                        line=dict(width=1, dash="dot"), opacity=0.15)
        return dict(type="rect", x0=x_low, x1=x_high, y0=w_low, y1=w_high,
                    line=dict(width=1, dash="dot"), opacity=0.15)
    for args in [(84,108,20,34),(84,108,46,60),(96,108,0,20),(96,108,60,80),(108,114,0,20),(108,114,60,80),(114,120,0,20),(114,120,60,80)]:
        fig.add_shape(rect(*args))

# ----------------------------
# Possession inference + metrics
# ----------------------------
def _time_seconds(df: pd.DataFrame) -> pd.Series:
    if "Minute" in df.columns and "Second" in df.columns:
        return df["Minute"].fillna(0).astype(float) * 60 + df["Second"].fillna(0).astype(float)
    return pd.Series(np.arange(len(df), dtype=float), index=df.index)

def infer_possessions(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Match", "Team"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for possession inference: {missing}")
    d = df.copy()
    d["_t"] = _time_seconds(d)
    d = d.sort_values(["Match", "_t"]).reset_index(drop=True)

    def is_shot_row(r) -> bool:
        for c in ["Event", "Type", "Action"]:
            if c in d.columns and isinstance(r.get(c), str) and "shot" in r[c].lower():
                return True
        return False

    def is_turnover_row(r) -> bool:
        if "Outcome" in d.columns and isinstance(r.get("Outcome"), str):
            if r["Outcome"].lower() in ["incomplete", "out", "lost"]:
                return True
        for c in ["Turnover", "turnover", "is_turnover"]:
            if c in d.columns:
                v = r.get(c)
                if isinstance(v, (int, float)) and int(v) == 1:
                    return True
                if isinstance(v, str) and v.lower() in ["1","true","yes"]:
                    return True
        return False

    def is_foul_row(r) -> bool:
        for c in ["Event", "Type", "Action"]:
            if c in d.columns and isinstance(r.get(c), str) and "foul" in r[c].lower():
                return True
        return False

    poss_ids = []
    current = 0
    prev_match = None
    prev_team = None
    prev_end = True

    for _, r in d.iterrows():
        m = r["Match"]
        team = r["Team"]
        new_poss = False
        if prev_match is None or m != prev_match:
            new_poss = True
        elif team != prev_team:
            new_poss = True
        elif prev_end:
            new_poss = True
        if new_poss:
            current += 1
        poss_ids.append(current)
        prev_match = m
        prev_team = team
        prev_end = is_shot_row(r) or is_turnover_row(r) or is_foul_row(r)

    d["possession_id"] = poss_ids
    return d

def add_match_state(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    candidates = [
        ("team_score", "opp_score"),
        ("Team Score", "Opp Score"),
        ("score_for", "score_against"),
        ("Home Score", "Away Score"),
    ]
    found = None
    for a, b in candidates:
        if a in d.columns and b in d.columns:
            found = (a, b)
            break
    if found is None:
        return d
    a, b = found
    diff = pd.to_numeric(d[a], errors="coerce") - pd.to_numeric(d[b], errors="coerce")
    d["match_state"] = np.where(diff > 0, "Winning", np.where(diff < 0, "Losing", "Drawing"))
    return d

def compute_possession_tables(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["_t"] = _time_seconds(d)

    d["_shot"] = False
    for c in ["Event", "Type", "Action"]:
        if c in d.columns:
            d["_shot"] = d["_shot"] | d[c].astype(str).str.lower().str.contains("shot", na=False)

    d["_cross"] = True
    if "Action" in d.columns:
        d["_cross"] = d["Action"].astype(str).str.lower().str.contains("cross", na=False)
    if "is_cross" in d.columns:
        d["_cross"] = d["is_cross"].fillna(0).astype(int).eq(1)

    d["_box_entry"] = False
    if {"End X", "End Y"}.issubset(d.columns):
        d["_box_entry"] = (pd.to_numeric(d["End X"], errors="coerce") >= 102) & pd.to_numeric(d["End Y"], errors="coerce").between(18, 62)

    if {"Start X", "End X"}.issubset(d.columns):
        d["_prog"] = (pd.to_numeric(d["End X"], errors="coerce") - pd.to_numeric(d["Start X"], errors="coerce")).fillna(0.0)
    else:
        d["_prog"] = 0.0

    xg_col = None
    for c in ["xG", "xg", "shot_xg", "expected_goals"]:
        if c in d.columns:
            xg_col = c
            break
    if xg_col:
        d["_xg"] = pd.to_numeric(d[xg_col], errors="coerce").fillna(0.0)
    else:
        d["_xg"] = np.nan

    poss = d.groupby(["Match", "Team", "possession_id"], dropna=False).agg(
        events=("Team", "size"),
        duration=("_t", lambda s: float(s.max() - s.min()) if len(s) else 0.0),
        progressed=("_prog", "sum"),
        has_cross=("_cross", "max"),
        has_shot=("_shot", "max"),
        has_box=("_box_entry", "max"),
        xg_sum=("_xg", "sum") if xg_col else ("Team", lambda s: np.nan),
    ).reset_index()

    poss["outcome"] = np.where(poss["has_shot"], "Shot",
                       np.where(poss["has_box"], "Box Entry",
                       np.where(poss["has_cross"], "Cross",
                       "Turnover/Other")))

    poss["is_counter"] = (poss["duration"] <= 10) & (poss["progressed"] >= 25) & (poss["events"] <= 6)
    poss["directness"] = poss["progressed"] / poss["events"].replace(0, np.nan)

    team = poss.groupby("Team").agg(
        possessions=("possession_id", "nunique"),
        shot_poss_pct=("has_shot", lambda x: 100 * x.mean()),
        box_poss_pct=("has_box", lambda x: 100 * x.mean()),
        cross_poss_pct=("has_cross", lambda x: 100 * x.mean()),
        xg_per_poss=("xg_sum", lambda x: np.nan if x.isna().all() else x.sum() / max(len(x), 1)),
        passes_per_poss=("events", "mean"),
        duration_s=("duration", "mean"),
        meters_progressed=("progressed", "mean"),
        directness=("directness", "mean"),
        counter_poss_pct=("is_counter", lambda x: 100 * x.mean()),
    ).reset_index()

    out = poss.pivot_table(index="Team", columns="outcome", values="possession_id", aggfunc="count", fill_value=0).reset_index()
    total = out.drop(columns=["Team"]).sum(axis=1).replace(0, np.nan)
    for c in out.columns:
        if c != "Team":
            out[c + "_pct"] = 100 * out[c] / total

    flow = poss.copy()
    flow["stage1"] = "Build"
    flow["stage2"] = np.where(flow["has_cross"], "Cross", "No Cross")
    flow["stage3"] = np.where(flow["has_box"], "Box Entry", "No Box Entry")
    flow["stage4"] = np.where(flow["has_shot"], "Shot", "No Shot")
    sankey_counts = flow.groupby(["stage1","stage2","stage3","stage4"]).size().reset_index(name="count")

    return {"poss": poss, "team": team, "outcomes": out, "sankey": sankey_counts}

def percentile_ranks(team_df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = team_df.copy()
    for c in cols:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce")
            out[c + "_pctile"] = s.rank(pct=True) * 100
    return out


def compute_player_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Player profiles: progression (xT proxy) vs chance creation (xA proxy)."""
    d = df.copy()

    if {"Start X", "End X"}.issubset(d.columns):
        d["_prog"] = pd.to_numeric(d["End X"], errors="coerce") - pd.to_numeric(d["Start X"], errors="coerce")
    else:
        d["_prog"] = 0.0

    d["_xa"] = 0.0
    if "Outcome" in d.columns:
        complete = d["Outcome"].astype(str).str.lower().eq("complete")
    else:
        complete = True

    if {"End X", "End Y"}.issubset(d.columns):
        box = (pd.to_numeric(d["End X"], errors="coerce") >= 102) & pd.to_numeric(d["End Y"], errors="coerce").between(18, 62)
        d.loc[complete & box, "_xa"] = 0.05

    if "Player" not in d.columns:
        return pd.DataFrame()

    return (
        d.groupby("Player")
        .agg(progression=("_prog", "sum"), xa_proxy=("_xa", "sum"), events=("Player", "size"))
        .reset_index()
    )


# ----------------------------
# App
# ----------------------------
st.title("Crossing Dashboard + Team Comparisons + Possession Effectiveness")

left, right = st.columns([1, 2], vertical_alignment="top")

with left:
    st.subheader("Data")
    uploaded = st.file_uploader("Upload Events CSV (optional)", type=["csv"])
    df = load_data(uploaded)
    if df is None:
        st.error("No file uploaded and Events.csv not found. Please upload your CSV or add Events.csv next to app.py.")
        st.stop()

    required_cols = ["Start X", "Start Y", "End X", "End Y"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns for pitch map: {missing}")
        st.stop()

    df = add_match_state(df)

    st.subheader("Filters")

    teams = sorted(df["Team"].dropna().unique()) if "Team" in df.columns else []
    team = None
    if teams:
        default_team = "Manchester United" if "Manchester United" in teams else teams[0]
        team = st.selectbox("Team", teams, index=teams.index(default_team))

    pass_heights = sorted(df["Pass height"].dropna().unique()) if "Pass height" in df.columns else []
    selected_heights = None
    if pass_heights:
        height_mode = st.radio("Cross height", ["All", "High only", "Low + Ground only", "Custom"], index=0)
        if height_mode == "Custom":
            selected_heights = st.multiselect("Select heights", pass_heights, default=pass_heights)
        elif height_mode == "High only":
            selected_heights = ["high"] if "high" in pass_heights else pass_heights[:1]
        elif height_mode == "Low + Ground only":
            selected_heights = [h for h in ["low", "ground"] if h in pass_heights] or pass_heights
        else:
            selected_heights = pass_heights

    outcomes = sorted(df["Outcome"].dropna().unique()) if "Outcome" in df.columns else []
    selected_outcomes = st.multiselect("Outcome", outcomes, default=outcomes) if outcomes else None

    matches = sorted(df["Match"].dropna().unique()) if "Match" in df.columns else []
    selected_matches = st.multiselect("Match", matches, default=matches) if matches else None

    players = sorted(df["Player"].dropna().unique()) if "Player" in df.columns else []
    selected_players = st.multiselect("Player", players, default=players) if players else None

    match_states = sorted(df["match_state"].dropna().unique()) if "match_state" in df.columns else []
    selected_states = st.multiselect("Match state", match_states, default=match_states) if match_states else None

    st.subheader("View options")
    attack_up = st.toggle("Attack upwards (rotate pitch)", value=True)
    flip_direction = st.toggle("Flip direction (length)", value=False)
    invert_width = st.toggle("Invert width (fix left/right)", value=True)
    show_zone_overlays = st.toggle("Show start-zone overlays", value=True)
    show_endpoints = st.toggle("Show endpoints", value=False)
    show_table = st.toggle("Show data table", value=True)

    st.subheader("Start zone filter")
    zone_options = ["Diagonals", "Whipped", "Driven", "Cutbacks & stand ups", "Other"]
    selected_zones = st.multiselect("Start Zones", zone_options,
                                    default=["Diagonals", "Whipped", "Driven", "Cutbacks & stand ups"])

    st.subheader("Start side filter")
    side_choice = st.radio("Crosses delivered from", ["Both", "Left", "Right"], index=0, horizontal=True)

    st.subheader("End location filters")
    use_end_filters = st.toggle("Enable end-location filtering", value=False)
    end_x_range = (0.0, 120.0)
    end_y_range = (0.0, 80.0)
    in_box_only = False
    if use_end_filters:
        end_x_range = st.slider("End X range (0–120)", 0.0, 120.0, (0.0, 120.0), 1.0)
        end_y_range = st.slider("End Y range (0–80)", 0.0, 80.0, (0.0, 80.0), 1.0)
        in_box_only = st.toggle("End location inside penalty area only", value=False)

# Apply filters for map
f = df.copy()
if team and "Team" in f.columns:
    f = f[f["Team"] == team]
if selected_heights is not None and "Pass height" in f.columns:
    f = f[f["Pass height"].isin(selected_heights)]
if selected_outcomes is not None and "Outcome" in f.columns:
    f = f[f["Outcome"].isin(selected_outcomes)]
if selected_matches is not None and "Match" in f.columns:
    f = f[f["Match"].isin(selected_matches)]
if selected_players is not None and "Player" in f.columns:
    f = f[f["Player"].isin(selected_players)]
if selected_states is not None and "match_state" in f.columns:
    f = f[f["match_state"].isin(selected_states)]

f = f.dropna(subset=["Start X", "Start Y", "End X", "End Y"]).copy()

if flip_direction:
    f["_sx"] = 120 - f["Start X"]
    f["_ex"] = 120 - f["End X"]
else:
    f["_sx"] = f["Start X"]
    f["_ex"] = f["End X"]

f["_sy"] = f["Start Y"]
f["_ey"] = f["End Y"]

def width_map(arr):
    return 80 - arr if invert_width else arr

f["Start Zone"] = [classify_zone(x, y) for x, y in zip(f["_sx"].to_numpy(), f["_sy"].to_numpy())]
if selected_zones:
    f = f[f["Start Zone"].isin(selected_zones)]
else:
    f = f.iloc[0:0]

f["Start Side"] = np.where(width_map(f["Start Y"].to_numpy()) < 40, "Left", "Right")
if side_choice != "Both":
    f = f[f["Start Side"] == side_choice]

if use_end_filters and len(f) > 0:
    f = f[(f["_ex"].between(end_x_range[0], end_x_range[1])) & (f["_ey"].between(end_y_range[0], end_y_range[1]))]
    if in_box_only:
        f = f[(f["_ex"] >= 102) & (f["_ey"].between(18, 62))]

if "Outcome" in f.columns:
    success = f["Outcome"].astype(str).str.lower().eq("complete").to_numpy()
else:
    success = np.zeros(len(f), dtype=bool)

pitch_len, pitch_wid = 120, 80
plot_start_x = f["_sx"].to_numpy()
plot_end_x = f["_ex"].to_numpy()
plot_start_w = width_map(f["_sy"].to_numpy())
plot_end_w = width_map(f["_ey"].to_numpy())

if attack_up:
    sx, sy = plot_start_w, plot_start_x
    ex, ey = plot_end_w, plot_end_x
    x_range, y_range = (0, pitch_wid), (0, pitch_len)
else:
    sx, sy = plot_start_x, plot_start_w
    ex, ey = plot_end_x, plot_end_w
    x_range, y_range = (0, pitch_len), (0, pitch_wid)

# Possession analytics
df_poss = infer_possessions(df)
if selected_states is not None and "match_state" in df_poss.columns and len(selected_states) > 0:
    df_poss = df_poss[df_poss["match_state"].isin(selected_states)]

tables = compute_possession_tables(df_poss)
poss_team = tables["team"]
poss_out = tables["outcomes"]
sankey_counts = tables["sankey"]
poss_df = tables["poss"]

rank_cols = ["shot_poss_pct","box_poss_pct","xg_per_poss","passes_per_poss","duration_s","meters_progressed","directness","counter_poss_pct"]
poss_team_ranks = percentile_ranks(poss_team, rank_cols)

with right:
    tab_map, tab_team, tab_poss_eff = st.tabs(["🗺️ Crossing Map", "📊 Team Comparison", "⚙️ Possession Effectiveness"])

    with tab_map:
        st.subheader("Pitch map")
        st.caption(f"Showing {len(f):,} crosses after filters.")

        fig = go.Figure()
        if attack_up:
            fig.update_layout(
                xaxis=dict(range=x_range, showgrid=False, zeroline=False, visible=False, scaleanchor="y", scaleratio=1),
                yaxis=dict(range=y_range, showgrid=False, zeroline=False, visible=False),
                shapes=build_pitch_shapes_vertical(length=pitch_len, width=pitch_wid),
                margin=dict(l=10, r=10, t=10, b=10),
            )
        else:
            shapes = []
            shapes.append(dict(type="rect", x0=0, y0=0, x1=pitch_len, y1=pitch_wid, line=dict(width=2)))
            shapes.append(dict(type="line", x0=pitch_len / 2, y0=0, x1=pitch_len / 2, y1=pitch_wid, line=dict(width=2)))
            pa_x = 18
            pa_y0, pa_y1 = 18, 62
            shapes.append(dict(type="rect", x0=0, y0=pa_y0, x1=pa_x, y1=pa_y1, line=dict(width=2)))
            shapes.append(dict(type="rect", x0=pitch_len - pa_x, y0=pa_y0, x1=pitch_len, y1=pa_y1, line=dict(width=2)))
            sb_x = 6
            sb_y0, sb_y1 = 30, 50
            shapes.append(dict(type="rect", x0=0, y0=sb_y0, x1=sb_x, y1=sb_y1, line=dict(width=2)))
            shapes.append(dict(type="rect", x0=pitch_len - sb_x, y0=sb_y0, x1=pitch_len, y1=sb_y1, line=dict(width=2)))
            shapes.append(dict(type="circle", x0=pitch_len / 2 - 10, y0=pitch_wid / 2 - 10,
                               x1=pitch_len / 2 + 10, y1=pitch_wid / 2 + 10, line=dict(width=2)))
            fig.update_layout(
                xaxis=dict(range=x_range, showgrid=False, zeroline=False, visible=False, scaleanchor="y", scaleratio=1),
                yaxis=dict(range=y_range, showgrid=False, zeroline=False, visible=False),
                shapes=shapes,
                margin=dict(l=10, r=10, t=10, b=10),
            )

        if show_zone_overlays:
            add_zone_overlays(fig, attack_up=attack_up, flip_len=flip_direction, invert_width=invert_width)

        hover_cols = [c for c in ["Start Side", "Start Zone", "Player", "Recipient player", "Outcome", "Pass height", "Match", "Date", "Minute", "Second"] if c in f.columns]
        def row_to_hover(row):
            parts = []
            for c in hover_cols:
                v = row.get(c, "")
                if pd.isna(v):
                    continue
                parts.append(f"<b>{c}</b>: {v}")
            parts.append(f"<b>Start</b>: ({row['Start X']:.1f}, {row['Start Y']:.1f})")
            parts.append(f"<b>End</b>: ({row['End X']:.1f}, {row['End Y']:.1f})")
            return "<br>".join(parts)

        hover_text = f.apply(row_to_hover, axis=1).to_list()
        hover_arr = np.array(hover_text, dtype=object)

        sx_s, sy_s, ex_s, ey_s = sx[success], sy[success], ex[success], ey[success]
        sx_f, sy_f, ex_f, ey_f = sx[~success], sy[~success], ex[~success], ey[~success]

        if len(sx_s) > 0:
            fig.add_trace(make_segment_trace(sx_s, sy_s, ex_s, ey_s, color="green"))
        if len(sx_f) > 0:
            fig.add_trace(make_segment_trace(sx_f, sy_f, ex_f, ey_f, color="red"))

        if len(sx_s) > 0:
            fig.add_trace(go.Scattergl(x=sx_s, y=sy_s, mode="markers",
                                       marker=dict(size=7, opacity=0.85, color="green"),
                                       hovertext=hover_arr[success], hoverinfo="text", showlegend=False))
        if len(sx_f) > 0:
            fig.add_trace(go.Scattergl(x=sx_f, y=sy_f, mode="markers",
                                       marker=dict(size=7, opacity=0.85, color="red"),
                                       hovertext=hover_arr[~success], hoverinfo="text", showlegend=False))

        if show_endpoints and len(f) > 0:
            fig.add_trace(go.Scattergl(x=ex, y=ey, mode="markers",
                                       marker=dict(size=6, opacity=0.6, symbol="x"),
                                       hoverinfo="skip", showlegend=False))

        st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Crosses", f"{len(f):,}")
        with m2: st.metric("Completed", f"{int(success.sum()):,}")
        with m3: st.metric("Completion %", f"{(100*success.mean() if len(success) else 0):.1f}%")

        if show_table:
            st.subheader("Filtered data")
            st.dataframe(f, use_container_width=True, height=350)

        st.download_button(
            "Download filtered crosses CSV",
            data=f.to_csv(index=False).encode("utf-8"),
            file_name="filtered_crosses.csv",
            mime="text/csv",
        )

    with tab_team:

        st.subheader("Player profiles: progression vs chance creation")

        prof = compute_player_profiles(df)
        if not prof.empty:
            figp = px.scatter(
                prof,
                x="progression",
                y="xa_proxy",
                text="Player",
                size="events",
                title="Player Profiles: Progression (xT proxy) vs Chance Creation (xA proxy)",
            )
            figp.update_traces(textposition="top center")
            figp.update_layout(
                xaxis_title="Progression (xT proxy)",
                yaxis_title="Chance Creation (xA proxy)",
            )
            st.plotly_chart(figp, use_container_width=True)

        st.subheader("Team-by-Team Comparison")
        metric = st.selectbox("Metric", [
            "possessions","shot_poss_pct","box_poss_pct","xg_per_poss","passes_per_poss",
            "duration_s","meters_progressed","directness","counter_poss_pct"
        ], index=1)

        plot_df = poss_team.sort_values(metric, ascending=False)
        fig = px.bar(plot_df, x="Team", y=metric, title=f"Teams — {metric.replace('_',' ')}")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            poss_team,
            x="passes_per_poss",
            y="xg_per_poss",
            size="possessions",
            hover_name="Team",
            title="Speed vs Control: passes/possession vs xG/possession",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Percentile ranks (0–100)")
        show_cols = ["Team"] + [c for c in poss_team_ranks.columns if c.endswith("_pctile")]
        st.dataframe(poss_team_ranks[show_cols], use_container_width=True)

        st.download_button(
            "Download team metrics CSV",
            data=poss_team.to_csv(index=False).encode("utf-8"),
            file_name="team_possession_metrics.csv",
            mime="text/csv",
        )

    with tab_poss_eff:
        st.subheader("Possession Effectiveness (Inferred)")

        cross_only = st.toggle("Analyse crossing possessions only", value=False)
        poss_focus = poss_df.copy()
        if cross_only:
            poss_focus = poss_focus[poss_focus["has_cross"] == 1]

        st.markdown("**Possession outcomes** (share of possessions)")
        out_pct_cols = [c for c in poss_out.columns if c.endswith("_pct")]
        if out_pct_cols:
            melt = poss_out.melt(id_vars=["Team"], value_vars=out_pct_cols, var_name="Outcome", value_name="Pct")
            melt["Outcome"] = melt["Outcome"].str.replace("_pct","", regex=False)
            fig = px.bar(melt, x="Team", y="Pct", color="Outcome", barmode="stack",
                         title="Possession outcome distribution (%)")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        # Team focus from subset
        team_focus = poss_team.copy()
        if cross_only:
            team_focus = poss_focus.groupby("Team").agg(
                possessions=("possession_id","nunique"),
                shot_poss_pct=("has_shot", lambda x: 100*x.mean()),
                box_poss_pct=("has_box", lambda x: 100*x.mean()),
                xg_per_poss=("xg_sum", lambda x: np.nan if x.isna().all() else x.sum()/max(len(x),1)),
                passes_per_poss=("events","mean"),
                duration_s=("duration","mean"),
                meters_progressed=("progressed","mean"),
                directness=("directness","mean"),
                counter_poss_pct=("is_counter", lambda x: 100*x.mean()),
            ).reset_index()

        fig3 = px.scatter(
            team_focus,
            x="shot_poss_pct",
            y="xg_per_poss",
            size="possessions",
            hover_name="Team",
            title="Shot-ending % vs xG per possession (size=possessions)",
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("**Counter-attacks (heuristic)**: duration ≤ 10s, progressed ≥ 25m, events ≤ 6")
        fig4 = px.bar(team_focus.sort_values("counter_poss_pct", ascending=False),
                      x="Team", y="counter_poss_pct", title="Counter-attack possessions (%)")
        fig4.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("**Possession flow (Sankey)**")
        nodes = ["Build", "Cross", "No Cross", "Box Entry", "No Box Entry", "Shot", "No Shot"]
        node_index = {n:i for i,n in enumerate(nodes)}
        sources, targets, values = [], [], []
        for _, r in sankey_counts.iterrows():
            c = int(r["count"])
            if c <= 0:
                continue
            sources += [node_index[r["stage1"]], node_index[r["stage2"]], node_index[r["stage3"]]]
            targets += [node_index[r["stage2"]], node_index[r["stage3"]], node_index[r["stage4"]]]
            values  += [c, c, c]

        sank = go.Figure(data=[go.Sankey(
            node=dict(label=nodes, pad=12, thickness=15),
            link=dict(source=sources, target=targets, value=values),
        )])
        sank.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(sank, use_container_width=True)

        st.subheader("Team possession metrics (focus)")
        st.dataframe(team_focus.sort_values("xg_per_poss", ascending=False), use_container_width=True)
        st.download_button(
            "Download team focus CSV",
            data=team_focus.to_csv(index=False).encode("utf-8"),
            file_name="team_possession_focus.csv",
            mime="text/csv",
        )

