# App

Flask web app for predicting MLB home team win probability. Takes manual stat input for both lineups (1 starting pitcher + 9 batters per team) and returns the predicted winner.

## Prerequisites

`model/model.pkl` must exist. Run the model pipeline first if it doesn't — see [model/README.md](../model/README.md).

## Setup

```bash
cd app
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
# Must be run from the app/ directory (model.pkl path is relative)
cd app
python app.py
```

Then open [http://localhost:5000](http://localhost:5000).

## Input

For each team, enter stats for the starting pitcher and 9 batters:

| Role | Stats |
|------|-------|
| Starting pitcher | ERA, WHIP, SO9, SO/W, IP |
| Each batter (1–9) | BA, OBP, SLG, OPS, K%, BB% |

All fields are required. The model computes home-minus-away diffs internally and passes them to the classifier.

## Notes

- The app currently requires manual stat entry. A planned improvement is to fetch stats automatically via the MLB Stats API or Baseball Reference.
- The model file is loaded once at startup from `../model/model.pkl`.
