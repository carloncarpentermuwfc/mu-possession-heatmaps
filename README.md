# MU Possession KDE Comparison (Streamlit)

A Streamlit app for analysing **team possession sequence end locations** (e.g., where possessions finish) using **KDE heatmaps** over a pitch, and comparing **Manchester United vs opponents** (or any two teams in the dataset).

## Features
- Upload a possessions CSV (or place it at `data/possessions.csv`)
- Aggregates to **team possession sequence ends** using `last_player_possession_in_team_possession == True` (when present)
- KDE heatmap per team with a pitch layout
- Optional **difference view** (Team A density − Team B density)
- Filters for matches (and phase/half when available)

## Getting started

### 1) Create & activate a virtual environment (recommended)
```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the app
```bash
streamlit run app.py
```

## Data
- You can upload the CSV in the app **or**
- Put it at `data/possessions.csv` and tick “Use local file” in the sidebar.

The app expects at least these columns:
- `match_id`
- `x_end`
- `y_end`
- `team_shortname` (or `team_id`)

For true team-possession aggregation, the CSV should include:
- `last_player_possession_in_team_possession` (boolean)

## Deploy to Streamlit Community Cloud
1. Push this folder to GitHub
2. Go to Streamlit Community Cloud → “New app”
3. Choose your repo + branch
4. Main file path: `app.py`

## License
MIT
