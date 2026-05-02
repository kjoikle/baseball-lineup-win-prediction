# MLB Home Team Win Predictor

Binary classifier predicting home team win probability from MLB starting lineups. Built with Retrosheet gamelogs (2023–2025) and pybaseball stats.

## Structure

- [model/](model/) — data pipeline (scraper), feature engineering, and model training
- [app/](app/) — Flask web app for interactive predictions

## Quick Start

```bash
bash quickstart.sh
```

The script sets up a shared `.venv`, installs all dependencies, runs the data pipeline, and starts the app. It will stop and tell you what to do if any manual steps are needed (gamelogs download, model training).

See [model/data/README.md](model/data/README.md) for data download instructions and [model/README.md](model/README.md) / [app/README.md](app/README.md) for details on each component.
