#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"

echo "=== MLB Win Predictor — Quickstart ==="

# ── 1. Virtual environment ────────────────────────────────────────────────────
echo ""
echo "[1/4] Setting up virtual environment..."
if [ ! -d "$VENV" ]; then
    python -m venv "$VENV"
    echo "      Created .venv"
else
    echo "      .venv already exists, skipping"
fi
source "$VENV/bin/activate"

# ── 2. Install dependencies ───────────────────────────────────────────────────
echo ""
echo "[2/4] Installing dependencies..."
pip install -q -r "$ROOT/model/requirements.txt"
pip install -q git+https://github.com/jldbc/pybaseball.git
pip install -q -r "$ROOT/app/requirements.txt"
echo "      Done"

# ── 3. Data pipeline ─────────────────────────────────────────────────────────
echo ""
echo "[3/4] Checking for Retrosheet gamelogs..."

MISSING_LOGS=false
for year in 2023 2024 2025; do
    if [ ! -f "$ROOT/model/data/gl${year}.txt" ]; then
        echo "      MISSING: model/data/gl${year}.txt"
        MISSING_LOGS=true
    fi
done

if [ "$MISSING_LOGS" = true ]; then
    echo ""
    echo "  One or more gamelogs are missing. See model/data/README.md for"
    echo "  download instructions, then re-run this script."
    echo ""
    echo "  Skipping pipeline — if model/model.pkl already exists, the app"
    echo "  can still start."
else
    echo "      All gamelogs found. Running pipeline..."
    cd "$ROOT/model"
    python scraper.py
    echo ""
    echo "      Features written to model/data/. Open model/test_model.ipynb"
    echo "      to train the model and save model.pkl if you haven't already."
fi

# ── 4. Launch app ─────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Checking for model.pkl..."

if [ ! -f "$ROOT/model/model.pkl" ]; then
    echo ""
    echo "  model/model.pkl not found. Train the model first:"
    echo "    jupyter notebook model/test_model.ipynb"
    echo ""
    echo "  Then re-run this script to start the app."
    exit 1
fi

echo "      Found. Starting app at http://localhost:5000 ..."
echo ""
cd "$ROOT/app"
python app.py
