# Model

Binary classifier predicting MLB home team win probability from starting lineup stats. Trained on 2023–2024 Retrosheet gamelogs, tested on 2025.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# pybaseball must be installed from source (PyPI version is outdated)
pip install git+https://github.com/jldbc/pybaseball.git
```

## Data Prerequisites

Download Retrosheet gamelogs for the years you want and place them in `data/`:

```
data/gl2023.txt
data/gl2024.txt
data/gl2025.txt
```

Gamelogs are available at [retrosheet.org/gamelogs](https://www.retrosheet.org/gamelogs/). Column definitions are in [data/gamelog_col_description.txt](data/gamelog_col_description.txt).

## Training the Model

Edit the config block at the top of [train.py](train.py), then run:

```bash
python train.py
```

This trains a logistic regression on `data/train_features.parquet`, prints test accuracy and AUC, and saves the model to `OUTPUT_PATH` (default: `model.pkl`).

Key config options:

| Option | Default | Description |
|---|---|---|
| `OUTPUT_PATH` | `model.pkl` | Where to save the trained model |
| `USE_DIFFERENTIAL` | `True` | 59 diff features vs. 118 raw |
| `USE_SCALE` | `False` | Apply StandardScaler |
| `POLY_DEGREE` | `None` | Add polynomial/interaction features |
| `MODEL_PARAMS` | L2, C=0.2336 | kwargs passed to `LogisticRegression` |

## Running the Data Pipeline

The pipeline produces `data/train_features.parquet` and `data/test_features.parquet`.

```bash
# Season-to-date stats (default)
python scraper.py

# Trailing N days
python scraper.py --time-window trailing_N --window-days 60

# Full prior season
python scraper.py --time-window full_prior_season
```

Stats are fetched from pybaseball and disk-cached under `data/stats_cache/` — reruns skip the API calls.

## Pipeline Steps

1. **Parse gamelogs** → `data/games.csv`: extracts date, teams, score, home_win label, and 20 player ID columns per game.
2. **Map player IDs** → `data/player_id_map.csv`: maps Retrosheet IDs to MLBAM IDs via the Chadwick Bureau register.
3. **Fetch & cache stats** → `data/stats_cache/{window}/`: calls pybaseball for batting and pitching stats per game date.
4. **Build feature matrix** → train/test parquets: 108 batting features + 10 pitching features + metadata. Missing values filled with column mean.

## Notebooks

- [test_scraper.ipynb](test_scraper.ipynb) — interactive walkthrough of the data pipeline
- [experiments.ipynb](experiments.ipynb) — multi-config experiment comparison (LR + MLP)
