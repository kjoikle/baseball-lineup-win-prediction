# mlbstatsapi Reference

Stats needed for the lineup prediction app:
```python
BATTING_STATS  = ["BA", "OBP", "SLG", "OPS", "K%", "BB%"]
PITCHING_STATS = ["ERA", "WHIP", "SO9", "SO/W", "IP"]
```

---

## Setup

```python
import mlbstatsapi
mlb = mlbstatsapi.Mlb()
```

---

## Player ID Lookup

```python
player_id = mlb.get_people_id("Freddie Freeman")[0]  # returns list; take first
```

Returns the MLBAM integer ID needed for all stats calls.

---

## Fetching Stats

```python
# Batting
stats = mlb.get_player_stats(player_id, stats=["season"], groups=["hitting"], season=2025)
stat  = stats["hitting"]["season"].splits[0].stat

# Pitching
stats = mlb.get_player_stats(player_id, stats=["season"], groups=["pitching"], season=2025)
stat  = stats["pitching"]["season"].splits[0].stat
```

`season` is an int or string year. Returns `{}` if the player has no stats for that season.

---

## Pitching Stats

All five fields work correctly and return floats.

| App key  | Field                       | Example  |
|----------|-----------------------------|----------|
| `ERA`    | `stat.era`                  | `2.71`   |
| `WHIP`   | `stat.whip`                 | `0.94`   |
| `SO9`    | `stat.strikeouts_per_9_inn` | `11.73`  |
| `SO/W`   | `stat.strikeout_walk_ratio` | `5.91`   |
| `IP`     | `stat.innings_pitched`      | `149.2`  |

---

## Batting Stats

Four of the six fields work; `K%` requires a workaround (see below).

| App key  | Field / expression                                     | Type   | Notes                        |
|----------|--------------------------------------------------------|--------|------------------------------|
| `BA`     | `float(stat.avg)`                                      | string | convert with `float()`       |
| `OBP`    | `float(stat.obp)`                                      | string | convert with `float()`       |
| `SLG`    | `float(stat.slg)`                                      | string | convert with `float()`       |
| `OPS`    | `float(stat.ops)`                                      | string | convert with `float()`       |
| `BB%`    | `stat.base_on_balls / stat.plate_appearances`          | int    | both are ints, divide directly |
| `K%`     | **broken** — `stat.strikeouts` is always `None`        | —      | see workaround below         |

### K% Bug

The MLB Stats API returns `strikeOuts` (capital O in camelCase). The `MLBBaseModel`
alias generator converts the Python field name `strikeouts` → `strikeouts` (single
word, no underscores, no change), so pydantic never matches it. The raw JSON has the
right value; the model just can't bind it.

**Workaround** — fetch raw JSON alongside the model call:

```python
raw = mlb._mlb_adapter_v1.get(
    endpoint=f"people/{player_id}/stats",
    ep_params={"stats": ["season"], "group": ["hitting"], "season": year},
).data["stats"][0]["splits"][0]["stat"]

k_pct  = raw["strikeOuts"] / raw["plateAppearances"]
bb_pct = raw["baseOnBalls"] / raw["plateAppearances"]   # can also use stat.base_on_balls
```

Using the raw dict for all batting rate stats is the simplest approach since
`strikeOuts`, `plateAppearances`, `baseOnBalls`, `avg`, `obp`, `slg`, and `ops`
are all present in the raw JSON under their camelCase keys.

---

## Full Raw JSON Keys (for reference)

### Hitting
```
gamesPlayed, groundOuts, airOuts, runs, doubles, triples, homeRuns,
strikeOuts, baseOnBalls, intentionalWalks, hits, hitByPitch,
avg, atBats, obp, slg, ops,
caughtStealing, stolenBases, stolenBasePercentage,
groundIntoDoublePlay, numberOfPitches, plateAppearances,
totalBases, rbi, leftOnBase, sacBunts, sacFlies,
babip, groundOutsToAirouts, catchersInterference, atBatsPerHomeRun
```

### Pitching
```
gamesPlayed, gamesStarted, groundOuts, airOuts, runs, doubles, triples, homeRuns,
strikeOuts, baseOnBalls, intentionalWalks, hits, hitByPitch,
avg, atBats, obp, slg, ops,
era, inningsPitched, wins, losses, saves, earnedRuns, whip,
battersFaced, outs, gamesPitched, completeGames, shutouts,
strikes, strikePercentage, wildPitches, balks,
strikeoutWalkRatio, strikeoutsPer9Inn, walksPer9Inn,
hitsPer9Inn, runsScoredPer9, homeRunsPer9,
winPercentage, pitchesPerInning, gamesFinished,
inheritedRunners, inheritedRunnersScored
```
