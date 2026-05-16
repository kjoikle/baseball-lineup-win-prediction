#!/usr/bin/env python3
"""Build app/data/player_seasons.json from the MLB Stats API.

Covers all seasons from START_YEAR to the current year. Each year's raw API
response is cached in .build_cache/ so reruns are fast — only newly-added
seasons hit the network. Re-run each offseason to pick up the latest season.

Usage:
    python build_player_seasons.py

Output:
    app/data/player_seasons.json
"""
import datetime
import json
import os
import time

import mlbstatsapi

START_YEAR = 1974
BATTER_MIN_PA = 100
PITCHER_MIN_IP = 30.0
CACHE_DIR = ".build_cache"
OUT_PATH = os.path.join("app", "data", "player_seasons.json")

mlb = mlbstatsapi.Mlb()
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def fetch_bulk(year: int, group: str) -> list[dict]:
    cache_file = os.path.join(CACHE_DIR, f"{year}_{group}.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    resp = mlb._mlb_adapter_v1.get(
        endpoint="stats",
        ep_params={
            "stats": "season",
            "group": group,
            "gameType": "R",
            "season": year,
            "sportId": 1,
            "limit": 5000,
        },
    )
    splits = []
    for stat_group in resp.data.get("stats", []):
        splits.extend(stat_group.get("splits", []))

    with open(cache_file, "w") as f:
        json.dump(splits, f)
    time.sleep(0.3)
    return splits


def safe_float(val, fallback=None):
    try:
        f = float(val)
        if f != f:  # NaN
            return fallback
        return f
    except (TypeError, ValueError):
        return fallback


def best_split_per_player(splits: list[dict], pa_key: str) -> dict[int, dict]:
    """For players with multiple splits (traded mid-season), keep the row with the most PA/IP."""
    best: dict[int, dict] = {}
    for split in splits:
        pid = split.get("player", {}).get("id")
        if not pid:
            continue
        stat = split.get("stat", {})
        val = safe_float(stat.get(pa_key, 0), 0)
        existing_val = safe_float(best.get(pid, {}).get("stat", {}).get(pa_key, 0), 0)
        if val > existing_val:
            best[pid] = split
    return best


def build_batter_entry(stat: dict, pa: int) -> dict | None:
    try:
        so = stat.get("strikeOuts", 0)
        bb = stat.get("baseOnBalls", 0)
        return {
            "kind": "batter",
            "BA": round(float(stat["avg"]), 3),
            "OBP": round(float(stat["obp"]), 3),
            "SLG": round(float(stat["slg"]), 3),
            "OPS": round(float(stat["ops"]), 3),
            "K%": round(so / pa, 4),
            "BB%": round(bb / pa, 4),
        }
    except (KeyError, TypeError, ZeroDivisionError, ValueError):
        return None


def build_pitcher_entry(stat: dict) -> dict | None:
    try:
        ip = safe_float(stat.get("inningsPitched"), 0)
        era = safe_float(stat.get("era"))
        whip = safe_float(stat.get("whip"))
        so9 = safe_float(stat.get("strikeoutsPer9Inn"))
        sow = safe_float(stat.get("strikeoutWalkRatio"))
        if any(v is None for v in (era, whip, so9, sow)):
            return None
        return {
            "kind": "pitcher",
            "ERA": round(era, 2),
            "WHIP": round(whip, 3),
            "SO9": round(so9, 2),
            "SO/W": round(sow, 2),
            "IP": round(ip, 1),
        }
    except (KeyError, TypeError, ValueError):
        return None


def process_year(year: int) -> dict[str, dict]:
    batting_splits = fetch_bulk(year, "hitting")
    pitching_splits = fetch_bulk(year, "pitching")

    batters = best_split_per_player(batting_splits, "plateAppearances")
    pitchers = best_split_per_player(pitching_splits, "inningsPitched")

    # Build qualified entries
    batter_entries: dict[int, tuple[str, str, dict]] = {}  # pid -> (name, team, stats)
    for pid, split in batters.items():
        stat = split.get("stat", {})
        pa = stat.get("plateAppearances", 0)
        if pa < BATTER_MIN_PA:
            continue
        entry = build_batter_entry(stat, pa)
        if not entry:
            continue
        name = split.get("player", {}).get("fullName", "")
        team = split.get("team", {}).get("abbreviation", "")
        if name:
            batter_entries[pid] = (name, team, entry)

    pitcher_entries: dict[int, tuple[str, str, dict]] = {}
    for pid, split in pitchers.items():
        stat = split.get("stat", {})
        ip = safe_float(stat.get("inningsPitched"), 0)
        if ip < PITCHER_MIN_IP:
            continue
        entry = build_pitcher_entry(stat)
        if not entry:
            continue
        name = split.get("player", {}).get("fullName", "")
        team = split.get("team", {}).get("abbreviation", "")
        if name:
            pitcher_entries[pid] = (name, team, entry)

    two_way = set(batter_entries) & set(pitcher_entries)

    # Assign display keys — two-way players get a role suffix so they appear
    # twice (once as batter, once as pitcher) without overwriting each other.
    from collections import Counter

    def make_key(name: str, suffix: str, team: str, year: int, counts: Counter) -> str:
        key = f"{name}{suffix} ({year})"
        return f"{name}{suffix} - {team} ({year})" if counts[key] > 1 else key

    raw_keys: list[str] = []
    for pid, (name, team, _) in batter_entries.items():
        suffix = " - Batter" if pid in two_way else ""
        raw_keys.append(f"{name}{suffix} ({year})")
    for pid, (name, team, _) in pitcher_entries.items():
        suffix = " - Pitcher" if pid in two_way else ""
        raw_keys.append(f"{name}{suffix} ({year})")
    key_counts = Counter(raw_keys)

    year_result: dict[str, dict] = {}
    for pid, (name, team, entry) in batter_entries.items():
        suffix = " - Batter" if pid in two_way else ""
        key = make_key(name, suffix, team, year, key_counts)
        year_result[key] = entry
    for pid, (name, team, entry) in pitcher_entries.items():
        suffix = " - Pitcher" if pid in two_way else ""
        key = make_key(name, suffix, team, year, key_counts)
        year_result[key] = entry

    return year_result, len(batter_entries), len(pitcher_entries), len(two_way)


def main():
    end_year = datetime.date.today().year
    player_seasons: dict[str, dict] = {}

    for year in range(START_YEAR, end_year + 1):
        print(f"{year}...", end=" ", flush=True)
        try:
            result, nb, np_, ntw = process_year(year)
            player_seasons.update(result)
            print(f"{nb}B {np_}P {ntw}2W")
        except Exception as e:
            print(f"SKIP ({e})")

    print(f"\nTotal entries: {len(player_seasons)}")
    with open(OUT_PATH, "w") as f:
        json.dump(player_seasons, f, separators=(",", ":"))
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
