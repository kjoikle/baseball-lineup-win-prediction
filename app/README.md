# App

Flask web app for predicting MLB home team win probability. Takes lineup stat input for both teams (1 starting pitcher + 9 batters per team) and returns the predicted winner with win probability.

The application is deployed here: https://baseball-lineup-win-prediction.vercel.app

## Prerequisites

`models/model.pkl` must exist. Run the model training pipeline first if it doesn't — see [model/README.md](../model/README.md). If you need to change the path to the model:

```bash
cp .env.example .env
```

And set your path there.

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

## Using the App

### Fill in lineup stats

Stats should reflect the players' **season-to-date or recent performance** at the time of the game being predicted. The model was trained on full-season stats (2023–2024), so using full-season stats will produce the most reliable results.

Additionally you can click "Fill Random Stats" to auto generate stats in a reasonable range.

### Autofill by player name

Instead of entering stats manually, you can type a player's name into the name field next to each slot and click **Autofill**. The app will look up the player's current-season stats from the MLB Stats API and populate the fields automatically.

Names must match the MLB Stats API spelling (e.g., "Shohei Ohtani", not "Ohtani").

## Todo Items

- Add a tab to import past games and check how the model's prediction compares to reality

- Implement the All-Time Lineups Tab

- Improve UX
