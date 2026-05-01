"""
scraper.py — Baseball Win Probability Data Pipeline

End-to-end scraper that produces train/test feature matrices from raw Retrosheet gamelogs.

Pipeline:
    1. Parse gamelogs → data/games.csv
    2. Map player IDs (Retrosheet → MLBAM) via Chadwick Bureau → data/player_id_map.csv
    3. Fetch & cache per-date player stats via pybaseball → data/stats_cache/
    4. Build feature matrix → data/train_features.parquet, data/test_features.parquet

Usage:
    python scraper.py
    python scraper.py --time-window season_to_date
    python scraper.py --time-window trailing_N --window-days 60
"""

import argparse
import io
from datetime import date as date_type, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pybaseball
import requests

# ---------------------------------------------------------------------------
# Paths & file lists
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
GAMELOG_FILES = [DATA_DIR / "gl2023.txt", DATA_DIR / "gl2024.txt", DATA_DIR / "gl2025.txt"]
STATS_CACHE_DIR = DATA_DIR / "stats_cache"

# ---------------------------------------------------------------------------
# Retrosheet gamelog column indices (0-indexed)
# Reference: data/gamelog_col_description.txt
# ---------------------------------------------------------------------------

COL_DATE = 0
COL_AWAY_TEAM = 3
COL_HOME_TEAM = 6
COL_AWAY_SCORE = 9
COL_HOME_SCORE = 10
COL_AWAY_SP_ID = 101  # visiting starting pitcher Retrosheet ID
COL_HOME_SP_ID = 103  # home starting pitcher Retrosheet ID

# Lineup ID columns: every 3rd column (ID, name, position)
AWAY_LINEUP_COLS = [105 + i * 3 for i in range(9)]
HOME_LINEUP_COLS = [132 + i * 3 for i in range(9)]

# All player ID columns used across lineup and pitching
ID_COLS = (
    ["away_sp_id", "home_sp_id"]
    + [f"away_bat{i}_id" for i in range(1, 10)]
    + [f"home_bat{i}_id" for i in range(1, 10)]
)

# ---------------------------------------------------------------------------
# Stat configuration
# ---------------------------------------------------------------------------

# Approximate MLB opening day by season year
OPENING_DAYS = {
    2022: date_type(2022, 4, 7),
    2023: date_type(2023, 3, 30),
    2024: date_type(2024, 3, 20),
    2025: date_type(2025, 3, 27),
}

# batting_stats_range returns BA, OBP, SLG, OPS, BB, SO, PA directly.
# K% and BB% are computed from counting stats.
BATTING_STATS = ["BA", "OBP", "SLG", "OPS", "K%", "BB%"]
PITCHING_STATS = ["ERA", "WHIP", "SO9", "SO/W", "IP"]

# Chadwick Bureau player register (split into 16 files)
CHADWICK_BASE = "https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-{}.csv"


# ---------------------------------------------------------------------------
# Step 1 — Parse Gamelogs
# ---------------------------------------------------------------------------


def parse_gamelogs(files=GAMELOG_FILES):
    """Parse raw Retrosheet gamelog files into a structured DataFrame.

    Args:
        files: List of paths to Retrosheet gamelog .txt files.

    Returns:
        DataFrame with one row per game (ties excluded), columns:
        date, away_team, home_team, away_score, home_score, home_win,
        away_sp_id, home_sp_id, away_bat1_id … home_bat9_id.
    """
    frames = [pd.read_csv(f, header=None, dtype=str) for f in files]
    raw = pd.concat(frames, ignore_index=True)

    games = pd.DataFrame()
    games["date"] = pd.to_datetime(raw[COL_DATE], format="%Y%m%d")
    games["away_team"] = raw[COL_AWAY_TEAM]
    games["home_team"] = raw[COL_HOME_TEAM]
    games["away_score"] = pd.to_numeric(raw[COL_AWAY_SCORE])
    games["home_score"] = pd.to_numeric(raw[COL_HOME_SCORE])
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
    games["away_sp_id"] = raw[COL_AWAY_SP_ID].str.strip('"')
    games["home_sp_id"] = raw[COL_HOME_SP_ID].str.strip('"')

    for i, col in enumerate(AWAY_LINEUP_COLS):
        games[f"away_bat{i+1}_id"] = raw[col].str.strip('"')
    for i, col in enumerate(HOME_LINEUP_COLS):
        games[f"home_bat{i+1}_id"] = raw[col].str.strip('"')

    # Drop ties (rare suspended games)
    games = games[games["away_score"] != games["home_score"]].reset_index(drop=True)
    return games


# ---------------------------------------------------------------------------
# Step 2 — Player ID Mapping (Retrosheet → MLBAM)
# ---------------------------------------------------------------------------


def fetch_chadwick_register():
    """Fetch the full Chadwick Bureau player register from GitHub.

    The register is split across 16 CSV files (people-0.csv … people-f.csv).

    Returns:
        DataFrame with 510k+ rows and 40 columns including key_retro and key_mlbam.
    """
    chunks = []
    for suffix in [*"0123456789", "a", "b", "c", "d", "e", "f"]:
        url = CHADWICK_BASE.format(suffix)
        resp = requests.get(url)
        resp.raise_for_status()
        chunks.append(
            pd.read_csv(io.StringIO(resp.content.decode("utf-8")), low_memory=False)
        )
    return pd.concat(chunks, ignore_index=True)


def build_id_mapping(all_retro_ids, raw_register):
    """Build a Retrosheet → MLBAM ID lookup table for a set of player IDs.

    Args:
        all_retro_ids: Iterable of Retrosheet player ID strings.
        raw_register:  Chadwick Bureau register DataFrame (from fetch_chadwick_register).

    Returns:
        Tuple of (mapping_df, players_with_mlbam) where:
        - mapping_df has columns: retro_id, mlbam_id, name
        - players_with_mlbam is the set of retro IDs that have a valid MLBAM ID
    """
    subset = raw_register[raw_register["key_retro"].isin(all_retro_ids)].copy()
    mapping = (
        subset[["key_retro", "key_mlbam", "name_first", "name_last"]]
        .rename(columns={"key_retro": "retro_id", "key_mlbam": "mlbam_id"})
    )
    mapping["name"] = (
        mapping["name_first"].str.strip() + " " + mapping["name_last"].str.strip()
    )
    mapping = (
        mapping.drop(columns=["name_first", "name_last"])
        .drop_duplicates(subset="retro_id")
        .reset_index(drop=True)
    )

    has_mlbam = mapping[mapping["mlbam_id"].notna()].copy()
    has_mlbam["mlbam_id"] = has_mlbam["mlbam_id"].astype(int)
    players_with_mlbam = set(has_mlbam["retro_id"])
    return mapping, players_with_mlbam


def filter_games_by_mlbam(games, players_with_mlbam, id_cols=ID_COLS):
    """Drop games where any lineup player lacks an MLBAM ID.

    Args:
        games:               Games DataFrame from parse_gamelogs.
        players_with_mlbam:  Set of Retrosheet IDs that have MLBAM mappings.
        id_cols:             List of player ID column names to check.

    Returns:
        Filtered DataFrame (reset index).
    """
    def all_mapped(row):
        for col in id_cols:
            pid = row.get(col)
            if pd.notna(pid) and pid.strip() and pid not in players_with_mlbam:
                return False
        return True

    mask = games.apply(all_mapped, axis=1)
    return games[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3 — Fetch & Cache Player Stats
# ---------------------------------------------------------------------------


def get_stat_window(game_date, time_window="season_to_date", window_days=30):
    """Compute the stat date range and cache label for a given game date.

    Args:
        game_date:   datetime.date or pandas Timestamp for the game.
        time_window: One of:
                     - 'season_to_date'    — opening day through game_date - 1 (no leakage)
                     - 'trailing_N'        — last window_days days before game
                     - 'full_prior_season' — full previous calendar year
                     - 'full_current_season' — full current season through Nov 1
        window_days: Number of days for 'trailing_N' window.

    Returns:
        Tuple of (start_date, end_date, cache_label) as date objects and str.
    """
    d = game_date if isinstance(game_date, date_type) else game_date.date()
    end = d - timedelta(days=1)

    if time_window == "season_to_date":
        start = OPENING_DAYS[d.year]
        label = "season_to_date"
    elif time_window == "trailing_N":
        start = d - timedelta(days=window_days)
        label = f"trailing_{window_days}"
    elif time_window == "full_prior_season":
        start = OPENING_DAYS.get(d.year - 1, date_type(d.year - 1, 3, 28))
        end = date_type(d.year - 1, 11, 1)
        label = f"prior_season_{d.year - 1}"
    elif time_window == "full_current_season":
        start = OPENING_DAYS[d.year]
        end = date_type(d.year, 11, 1)
        label = f"full_current_season_{d.year}"
    else:
        raise ValueError(f"Unknown time_window: {time_window}")

    return start, end, label


def fetch_batting_stats_cached(start, end, cache_label):
    """Fetch batting stats for a date range, using disk cache to avoid re-fetching.

    Args:
        start:       Start date (date object or str).
        end:         End date (date object or str).
        cache_label: Subdirectory name under stats_cache/ for this window.

    Returns:
        DataFrame of batting stats from pybaseball.batting_stats_range.
    """
    cache_dir = STATS_CACHE_DIR / cache_label / "batting"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{start}_{end}.csv"

    if cache_file.exists():
        return pd.read_csv(cache_file)

    df = pybaseball.batting_stats_range(str(start), str(end))
    df.to_csv(cache_file, index=False)
    return df


def fetch_pitching_stats_cached(start, end, cache_label):
    """Fetch pitching stats for a date range, using disk cache to avoid re-fetching.

    Args:
        start:       Start date (date object or str).
        end:         End date (date object or str).
        cache_label: Subdirectory name under stats_cache/ for this window.

    Returns:
        DataFrame of pitching stats from pybaseball.pitching_stats_range.
    """
    cache_dir = STATS_CACHE_DIR / cache_label / "pitching"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{start}_{end}.csv"

    if cache_file.exists():
        return pd.read_csv(cache_file)

    df = pybaseball.pitching_stats_range(str(start), str(end))
    df.to_csv(cache_file, index=False)
    return df


def fetch_stats_for_dates(all_dates, time_window="full_current_season", window_days=30):
    """Fetch (or load from cache) batting and pitching stats for all game dates.

    Args:
        all_dates:   Sorted list of datetime.date objects.
        time_window: Stat window type (see get_stat_window).
        window_days: Days for 'trailing_N' window.

    Returns:
        Tuple of (batting_stats_by_date, pitching_stats_by_date) dicts
        mapping date → DataFrame.
    """
    batting_stats_by_date = {}
    pitching_stats_by_date = {}

    for i, game_date in enumerate(all_dates):
        start, end, label = get_stat_window(game_date, time_window, window_days)

        if start >= end:
            # Opening day or before — no prior stats available yet
            batting_stats_by_date[game_date] = pd.DataFrame()
            pitching_stats_by_date[game_date] = pd.DataFrame()
            continue

        batting_stats_by_date[game_date] = fetch_batting_stats_cached(start, end, label)
        pitching_stats_by_date[game_date] = fetch_pitching_stats_cached(start, end, label)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(all_dates)} dates done")

    return batting_stats_by_date, pitching_stats_by_date


# ---------------------------------------------------------------------------
# Step 4 — Build Feature Matrix
# ---------------------------------------------------------------------------


def lookup_batting_stats(mlbam_id, stats_df):
    """Extract batting stats for a single player from a stats DataFrame.

    Computes K% = SO/PA and BB% = BB/PA from counting stats.

    Args:
        mlbam_id: MLBAM integer player ID.
        stats_df: DataFrame returned by fetch_batting_stats_cached.

    Returns:
        Dict with keys matching BATTING_STATS; values are NaN if player not found.
    """
    result = {col: np.nan for col in BATTING_STATS}
    if stats_df.empty or pd.isna(mlbam_id):
        return result
    row = stats_df[stats_df["mlbID"] == mlbam_id]
    if row.empty:
        return result
    r = row.iloc[0]
    result["BA"] = r.get("BA", np.nan)
    result["OBP"] = r.get("OBP", np.nan)
    result["SLG"] = r.get("SLG", np.nan)
    result["OPS"] = r.get("OPS", np.nan)
    pa = r.get("PA", np.nan)
    if pa and pa > 0:
        result["K%"] = round(r.get("SO", np.nan) / pa, 2)
        result["BB%"] = round(r.get("BB", np.nan) / pa, 2)
    return result


def lookup_pitching_stats(mlbam_id, stats_df):
    """Extract pitching stats for a single player from a stats DataFrame.

    Args:
        mlbam_id: MLBAM integer player ID.
        stats_df: DataFrame returned by fetch_pitching_stats_cached.

    Returns:
        Dict with keys matching PITCHING_STATS; values are NaN if player not found.
    """
    result = {col: np.nan for col in PITCHING_STATS}
    if stats_df.empty or pd.isna(mlbam_id):
        return result
    row = stats_df[stats_df["mlbID"] == mlbam_id]
    if row.empty:
        return result
    r = row.iloc[0]
    for col in PITCHING_STATS:
        result[col] = r.get(col, np.nan)
    return result


def build_game_features(game_row, bat_stats_by_date, pitch_stats_by_date, retro_to_mlbam):
    """Flatten all player stats for a single game into a feature dict.

    Produces 120 features: 9 batters × 6 batting stats × 2 teams
    + 2 starting pitchers × 5 pitching stats × 2 teams.

    Args:
        game_row:             Row from the games DataFrame.
        bat_stats_by_date:    Dict mapping date → batting stats DataFrame.
        pitch_stats_by_date:  Dict mapping date → pitching stats DataFrame.
        retro_to_mlbam:       Dict mapping Retrosheet ID → MLBAM ID.

    Returns:
        Dict of feature_name → value.
    """
    game_date = game_row["date"].date()
    b_df = bat_stats_by_date.get(game_date, pd.DataFrame())
    p_df = pitch_stats_by_date.get(game_date, pd.DataFrame())
    feat = {}

    for team in ("away", "home"):
        for i in range(1, 10):
            retro_id = game_row.get(f"{team}_bat{i}_id", "")
            mlbam_id = retro_to_mlbam.get(str(retro_id), pd.NA)
            stats = lookup_batting_stats(mlbam_id, b_df)
            for stat, val in stats.items():
                feat[f"{team}_bat{i}_{stat}"] = val

        sp_retro = game_row.get(f"{team}_sp_id", "")
        sp_mlbam = retro_to_mlbam.get(str(sp_retro), pd.NA)
        sp_stats = lookup_pitching_stats(sp_mlbam, p_df)
        for stat, val in sp_stats.items():
            feat[f"{team}_sp_{stat}"] = val

    return feat


def build_feature_matrix(games, batting_stats_by_date, pitching_stats_by_date, retro_to_mlbam):
    """Build the full feature matrix for all games.

    Args:
        games:                  Games DataFrame from parse_gamelogs.
        batting_stats_by_date:  Dict mapping date → batting stats DataFrame.
        pitching_stats_by_date: Dict mapping date → pitching stats DataFrame.
        retro_to_mlbam:         Dict mapping Retrosheet ID → MLBAM ID.

    Returns:
        DataFrame with 122 columns: 120 player stat features + date, home_team,
        away_team, home_win. NaN values are filled with column means.
    """
    rows = []
    for _, row in games.iterrows():
        feat = build_game_features(row, batting_stats_by_date, pitching_stats_by_date, retro_to_mlbam)
        feat["date"] = row["date"]
        feat["home_team"] = row["home_team"]
        feat["away_team"] = row["away_team"]
        feat["home_win"] = row["home_win"]
        rows.append(feat)

    features_df = pd.DataFrame(rows)

    # Fill missing values with column mean (league average proxy)
    feat_cols = [c for c in features_df.columns if c not in ("date", "home_team", "away_team", "home_win")]
    col_means = features_df[feat_cols].mean()
    features_df[feat_cols] = features_df[feat_cols].fillna(col_means)

    return features_df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(time_window="full_current_season", window_days=30):
    """Run the full data scraping and feature-building pipeline.

    Args:
        time_window: Stat window type passed to get_stat_window.
        window_days: Days for 'trailing_N' window.

    Outputs:
        data/games.csv
        data/player_id_map.csv
        data/stats_cache/  (populated with cached CSVs)
        data/train_features.parquet  (2023-2024 games)
        data/test_features.parquet   (2025 games)
    """
    # ── Step 1: Parse Gamelogs ──────────────────────────────────────────────
    print("Step 1 — Parsing gamelogs...")
    games = parse_gamelogs(GAMELOG_FILES)
    print(f"  Parsed {len(games)} games ({games['date'].min().date()} → {games['date'].max().date()})")
    games.to_csv(DATA_DIR / "games.csv", index=False)
    print("  Saved data/games.csv")

    # ── Step 2: Player ID Mapping ───────────────────────────────────────────
    print("\nStep 2 — Building player ID mapping...")
    all_retro_ids = (
        pd.concat([games[c] for c in ID_COLS]).dropna().unique()
    )
    all_retro_ids = [rid for rid in all_retro_ids if rid.strip()]
    print(f"  {len(all_retro_ids)} unique Retrosheet player IDs")

    print("  Fetching Chadwick Bureau register (16 files)...")
    raw_register = fetch_chadwick_register()

    mapping, players_with_mlbam = build_id_mapping(all_retro_ids, raw_register)
    not_mapped = [rid for rid in all_retro_ids if rid not in players_with_mlbam]
    print(f"  Mapped: {len(players_with_mlbam)}/{len(all_retro_ids)} | Not mapped: {len(not_mapped)}")

    games = filter_games_by_mlbam(games, players_with_mlbam)
    print(f"  Games after MLBAM filter: {len(games)}")

    mapping.to_csv(DATA_DIR / "player_id_map.csv", index=False)
    print("  Saved data/player_id_map.csv")

    # ── Step 3: Fetch & Cache Player Stats ──────────────────────────────────
    print(f"\nStep 3 — Fetching player stats (window={time_window})...")
    print("  First run may take several minutes; cached runs are instant.")
    all_dates = sorted(games["date"].dt.date.unique())
    print(f"  {len(all_dates)} unique game dates")

    batting_stats_by_date, pitching_stats_by_date = fetch_stats_for_dates(
        all_dates, time_window, window_days
    )
    print("  Stats fetch complete.")

    # ── Step 4: Build Feature Matrix ────────────────────────────────────────
    print("\nStep 4 — Building feature matrix...")
    id_map = pd.read_csv(DATA_DIR / "player_id_map.csv")
    retro_to_mlbam = dict(zip(id_map["retro_id"], id_map["mlbam_id"].astype("Int64")))

    features_df = build_feature_matrix(
        games, batting_stats_by_date, pitching_stats_by_date, retro_to_mlbam
    )

    feat_cols = [c for c in features_df.columns if c not in ("date", "home_team", "away_team", "home_win")]
    missing_pct = features_df[feat_cols].isna().mean().mean() * 100
    print(f"  Feature matrix shape: {features_df.shape} | Missing: {missing_pct:.1f}%")

    # Train/test split: 2023-2024 train, 2025 held out
    train_df = features_df[features_df["date"].dt.year.isin([2023, 2024])].reset_index(drop=True)
    test_df = features_df[features_df["date"].dt.year == 2025].reset_index(drop=True)
    print(f"  Train: {len(train_df)} games | Test: {len(test_df)} games")

    train_df.to_parquet(DATA_DIR / "train_features.parquet", index=False)
    test_df.to_parquet(DATA_DIR / "test_features.parquet", index=False)
    print("  Saved train_features.parquet and test_features.parquet")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseball win probability data scraper")
    parser.add_argument(
        "--time-window",
        default="full_current_season",
        choices=["season_to_date", "trailing_N", "full_prior_season", "full_current_season"],
        help="Stat aggregation window relative to each game date (default: full_current_season)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Number of days for trailing_N window (default: 30)",
    )
    args = parser.parse_args()
    main(time_window=args.time_window, window_days=args.window_days)
