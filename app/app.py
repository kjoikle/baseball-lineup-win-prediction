import datetime
import mlbstatsapi
from flask import Flask, jsonify, redirect, render_template, request, url_for
import joblib
import numpy as np

app = Flask(__name__)
mlb = mlbstatsapi.Mlb()

_players_cache: dict[int, list[str]] = {}

def current_season() -> int:
    return datetime.date.today().year


def get_players_for_season(year: int) -> list[str]:
    if year not in _players_cache:
        try:
            _players_cache[year] = sorted(
                {p.full_name for p in mlb.get_people(sport_id=1, season=year) if p.full_name}
            )
        except Exception:
            _players_cache[year] = []
    return _players_cache[year]


# Pre-load current season at startup
get_players_for_season(current_season())

BATTING_STATS = ["BA", "OBP", "SLG", "OPS", "K%", "BB%"]
PITCHING_STATS = ["ERA", "WHIP", "SO9", "SO/W", "IP"]
PITCHING_LOWER_IS_BETTER = {"ERA", "WHIP"}
TEAMS = ["away", "home"]

MODEL_PATH = "./models/model.pkl"
MODEL = joblib.load(MODEL_PATH)

# TODO: better way to search names (right now cannot search ppl with accents, hyphens cause problem?)
# easy way to autofill from lineups?

def fetch_player_hitting_stats(player_id: int, season: int) -> dict | None:
    resp = mlb._mlb_adapter_v1.get(
        endpoint=f"people/{player_id}/stats",
        ep_params={"stats": ["season"], "group": ["hitting"], "season": season},
    )
    splits = resp.data.get("stats", [{}])[0].get("splits", [])
    if not splits:
        return None
    raw = splits[0]["stat"]
    pa = raw.get("plateAppearances") or 0
    if not pa:
        return None
    return {
        "BA":  round(float(raw["avg"]), 3),
        "OBP": round(float(raw["obp"]), 3),
        "SLG": round(float(raw["slg"]), 3),
        "OPS": round(float(raw["ops"]), 3),
        "K%":  round(raw["strikeOuts"] / pa, 4),
        "BB%": round(raw["baseOnBalls"] / pa, 4),
    }


def fetch_player_pitching_stats(player_id: int, season: int) -> dict | None:
    stats = mlb.get_player_stats(player_id, stats=["season"], groups=["pitching"], season=season)
    if not stats or "pitching" not in stats:
        return None
    splits = stats["pitching"]["season"].splits
    if not splits:
        return None
    s = splits[0].stat
    return {
        "ERA":  s.era,
        "WHIP": s.whip,
        "SO9":  s.strikeouts_per_9_inn,
        "SO/W": s.strikeout_walk_ratio,
        "IP":   s.innings_pitched,
    }


@app.route("/games/today")
def games_today():
    """
    Returns:
        list: A list of live/finalized games for today.
    """
    today = datetime.date.today().strftime("%m/%d/%Y")
    games = mlb.get_scheduled_games_by_date(today)
    result = []
    for g in games:
        state = g.status.abstract_game_state
        if state not in ("Live", "Final"):
            continue
        result.append({
            "game_pk": g.game_pk,
            "status": state,
            "away": g.teams.away.team.name,
            "home": g.teams.home.team.name,
        })
    return jsonify(result)


@app.route("/game/<int:game_pk>/lineup")
def game_lineup(game_pk):
    box = mlb.get_game_box_score(game_pk)

    def resolve(team):
        batters = []
        for pid in team.batting_order:
            p = team.players.get(f"ID{pid}")
            batters.append({"id": pid, "name": p.person.full_name if p else ""})
        sp_id = team.pitchers[0] if team.pitchers else None
        sp_player = team.players.get(f"ID{sp_id}") if sp_id else None
        return {
            "team": team.team.name,
            "sp": {"id": sp_id, "name": sp_player.person.full_name if sp_player else ""},
            "batters": batters,
        }

    return jsonify({"away": resolve(box.teams.away), "home": resolve(box.teams.home)})


@app.route("/autofill", methods=["POST"])
def autofill():
    data = request.get_json(silent=True) or {}
    season = int(data.get("season", current_season()))
    players = data.get("players", [])

    result = {}
    for player in players:
        slot = player.get("slot", "")
        name = (player.get("name") or "").strip()
        kind = player.get("kind", "")
        if not name or not slot:
            continue
        pid = player.get("player_id") or (mlb.get_people_id(name) or [None])[0]
        if not pid:
            continue
        try:
            stats = fetch_player_pitching_stats(pid, season) if kind == "pitcher" \
                    else fetch_player_hitting_stats(pid, season)
            if stats:
                for stat_name, val in stats.items():
                    result[f"{slot}_{stat_name}"] = val
        except Exception:
            pass

    return jsonify(result)


@app.route("/players")
def players():
    year = request.args.get("year", current_season(), type=int)
    return jsonify(get_players_for_season(year))


def run_model(_features: dict) -> dict:
    feature_array = np.array(list(_features.values())).reshape(1, -1)
    prediction = MODEL.predict_proba(feature_array)[0][1]
    return {"home_win_prob": prediction}


def create_diff_features(features: dict) -> dict:
    diff_features = {}
    for stat in PITCHING_STATS:
        home_key = f"home_sp_{stat}"
        away_key = f"away_sp_{stat}"
        diff_key = f"diff_sp_{stat}"
        if stat in PITCHING_LOWER_IS_BETTER:
            diff_features[diff_key] = features.get(away_key) - features.get(home_key)
        else:
            diff_features[diff_key] = features.get(home_key) - features.get(away_key)
    for n in range(1, 10):
        for stat in BATTING_STATS:
            home_key = f"home_bat{n}_{stat}"
            away_key = f"away_bat{n}_{stat}"
            diff_key = f"diff_bat{n}_{stat}"
            diff_features[diff_key] = features.get(home_key) - features.get(away_key)
    return diff_features


def parse_features(form) -> dict:
    features = {}
    for team in TEAMS:
        for stat in PITCHING_STATS:
            key = f"{team}_sp_{stat}"
            val = form.get(key, "")
            features[key] = float(val) if val else None
        for n in range(1, 10):
            for stat in BATTING_STATS:
                key = f"{team}_bat{n}_{stat}"
                val = form.get(key, "")
                features[key] = float(val) if val else None
    return create_diff_features(features)


def validate_form(form) -> bool:
    for team in TEAMS:
        for stat in PITCHING_STATS:
            if not form.get(f"{team}_sp_{stat}", ""):
                return False
        for n in range(1, 10):
            for stat in BATTING_STATS:
                if not form.get(f"{team}_bat{n}_{stat}", ""):
                    return False
    return True


@app.route("/")
def index():
    error = request.args.get("error")
    return render_template("index.html", error=error, current_year=current_season())


@app.route("/predict", methods=["POST"])
def predict():
    if not validate_form(request.form):
        return redirect(url_for("index", error="Please fill in all stat fields."))

    home_team = request.form.get("home_team") or "Home"
    away_team = request.form.get("away_team") or "Away"

    if home_team == away_team:
        home_team += " (Home)"
        away_team += " (Away)"

    try:
        features = parse_features(request.form)
        result = run_model(features)
    except Exception:
        return redirect(
            url_for("index", error="Prediction failed. Please check your inputs and try again.")
        )

    winner = home_team if result["home_win_prob"] >= 0.5 else away_team
    winner_win_prob = (
        result["home_win_prob"] if winner == home_team else 1 - result["home_win_prob"]
    )

    return render_template(
        "result.html",
        home_team=home_team,
        away_team=away_team,
        winner=winner,
        winner_win_prob=winner_win_prob,
    )


if __name__ == "__main__":
    app.run(debug=True)


# --- Test Helpers ---

def _batter_stats_away(team, n):
    return {
        f"{team}_bat{n}_BA": 0.300,
        f"{team}_bat{n}_OBP": 0.360,
        f"{team}_bat{n}_SLG": 0.500,
        f"{team}_bat{n}_OPS": 0.860,
        f"{team}_bat{n}_K%": 0.20,
        f"{team}_bat{n}_BB%": 0.10,
    }


def _batter_stats_home(team, n):
    return {
        f"{team}_bat{n}_BA": 0.350,
        f"{team}_bat{n}_OBP": 0.380,
        f"{team}_bat{n}_SLG": 0.750,
        f"{team}_bat{n}_OPS": 0.860,
        f"{team}_bat{n}_K%": 0.10,
        f"{team}_bat{n}_BB%": 0.20,
    }


TEST_HOME_TEAM_STATS = {
    "home_sp_ERA": 3.50,
    "home_sp_WHIP": 1.20,
    "home_sp_SO9": 9.0,
    "home_sp_SO/W": 3.0,
    "home_sp_IP": 200.0,
    **{k: v for n in range(1, 10) for k, v in _batter_stats_home("home", n).items()},
}

TEST_AWAY_TEAM_STATS = {
    "away_sp_ERA": 3.50,
    "away_sp_WHIP": 1.20,
    "away_sp_SO9": 9.0,
    "away_sp_SO/W": 3.0,
    "away_sp_IP": 200.0,
    **{k: v for n in range(1, 10) for k, v in _batter_stats_away("away", n).items()},
}
