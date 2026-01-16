# ==============================
# Imports
# ==============================
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="MUWFC Possession Dashboard")

# ==============================
# Automatic data loading (NO uploads)
# ==============================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

with st.sidebar:
    st.header("Data")
    st.caption("Data is automatically loaded from the GitHub /data folder.")

# ---- Required: possessions data ----
POSS_FILE = os.path.join(DATA_DIR, "MUWFCPOSSESSIONS.csv")

if not os.path.exists(POSS_FILE):
    st.error(
        "MUWFCPOSSESSIONS.csv not found.\n\n"
        "Please add it to the /data folder in your GitHub repo."
    )
    st.stop()

# Load possessions (skip metadata rows safely)
df = pd.read_csv(POSS_FILE, skiprows=2)

# ---- Optional player value tables ----
def load_optional(name):
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

xt_tbl = load_optional("Top_progression__xT__per_100_possessions.csv")
xa_tbl = load_optional("Top_chance_creators__xA__per_100_possessions.csv")
best_tbl = load_optional("Best_role-adjusted_attacking_value__per_100_possessions_.csv")
worst_tbl = load_optional("Worst_role-adjusted_attacking_value__per_100_possessions_.csv")
neg_tbl = load_optional("Net_negative_total_attacking_value__sample_filtered_.csv")

# ---- Data status panel ----
with st.sidebar.expander("Data status", expanded=False):
    st.write("Possessions:", df.shape)
    st.write("xT table:", "✅" if not xt_tbl.empty else "❌ missing")
    st.write("xA table:", "✅" if not xa_tbl.empty else "❌ missing")
    st.write("Best role-adjusted:", "✅" if not best_tbl.empty else "❌ missing")
    st.write("Worst role-adjusted:", "✅" if not worst_tbl.empty else "❌ missing")
    st.write("Net negative:", "✅" if not neg_tbl.empty else "❌ missing")

# ==============================
# App content (minimal example)
# ==============================
st.title("Manchester United Women – Possession & Attacking Analysis")

st.markdown(
    """
This dashboard automatically loads:
- Event-level possession data  
- Player progression (xT)
- Chance creation (xA)
- Role-adjusted attacking value
- Net negative attacking value  

All data is read directly from the GitHub `/data` folder.
"""
)

st.subheader("Quick sanity check")
st.dataframe(df.head(), use_container_width=True)

# ==============================
# Example: xT vs xA scatter
# ==============================
if not xt_tbl.empty and not xa_tbl.empty:
    st.subheader("Player profiles: progression vs creativity")

    # Try to merge safely on player name
    common_cols = set(xt_tbl.columns) & set(xa_tbl.columns)
    player_col = "player" if "player" in common_cols else list(common_cols)[0]

    prof = xt_tbl.merge(
        xa_tbl,
        on=player_col,
        suffixes=("_xT", "_xA"),
        how="inner"
    )

    if "xT_per_100" in prof.columns and "xA_per_100" in prof.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(prof["xT_per_100"], prof["xA_per_100"], alpha=0.7)
        ax.set_xlabel("xT per 100 possessions")
        ax.set_ylabel("xA per 100 possessions")
        ax.set_title("Progression vs Chance Creation")
        st.pyplot(fig, use_container_width=True)
