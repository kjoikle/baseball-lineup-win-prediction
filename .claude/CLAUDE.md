# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A binary classification ML project predicting home team win probability from MLB starting lineups. The pipeline transforms raw Retrosheet gamelogs into a feature matrix of per-player stats, then splits into train (2023–2024) and test (2025) sets.

## Commands

**Run the full data pipeline:**
```bash
python scraper.py                                          # season-to-date stats (default)
python scraper.py --time-window trailing_N --window-days 60  # last 60 days
python scraper.py --time-window full_prior_season
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Interactive exploration:**
```bash
jupyter notebook test_scraper.ipynb   # data pipeline walkthrough
jupyter notebook test_model.ipynb     # model training (in progress)
```

## Architecture

The pipeline in [scraper.py](../scraper.py) has 4 sequential steps:

1. **Parse gamelogs** (`data/gl2023-2025.txt` → `data/games.csv`): Reads Retrosheet fixed-format CSVs, extracts date/teams/score/home_win label and 20 player ID columns (1 SP + 9 batters per team).

2. **Map player IDs** → `data/player_id_map.csv`: Fetches Chadwick Bureau register to map Retrosheet player IDs to MLBAM IDs (required by pybaseball). 1,519 unique players, 100% coverage.

3. **Fetch & cache stats** → `data/stats_cache/{window}/batting|pitching/*.csv`: Calls `pybaseball.batting_stats_range()` and `pybaseball.pitching_stats_range()` for each unique game date. Results are disk-cached so reruns are instant.

4. **Build feature matrix** → `data/train_features.parquet` + `data/test_features.parquet`: Joins stats to each game. 122 columns total — 108 batting features (BA, OBP, SLG, OPS, K%, BB% × 9 batters × 2 teams) + 10 pitching features (ERA, WHIP, SO9, SO/W, IP × 1 SP × 2 teams) + metadata. Missing values filled with column mean.

**Train/test split:** 4,859 games (2023–2024) train, 2,430 games (2025) test.

## Key Data Files

Raw gamelogs (`data/gl*.txt`) and all generated outputs are gitignored. The column format for Retrosheet gamelogs is documented in `data/gamelog_col_description.txt`.

## Patterns

- Prioritize clean, reusable, well-documented code.
- Ask questions whenever you are unsure.
- The stat time window (season-to-date, trailing N days, full prior season) is a key design parameter — changes to it affect both caching structure and feature validity/leakage.
