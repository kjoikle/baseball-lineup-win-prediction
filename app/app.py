from flask import Flask, redirect, render_template, request, url_for
import joblib
import numpy as np
import os

app = Flask(__name__)

BATTING_STATS = ["BA", "OBP", "SLG", "OPS", "K%", "BB%"]
PITCHING_STATS = ["ERA", "WHIP", "SO9", "SO/W", "IP"]
PITCHING_LOWER_IS_BETTER = {"ERA", "WHIP"}
TEAMS = ["away", "home"]

DEFAULT_MODEL_PATH = "./models/elastic-model.pkl"

MODEL_PATH = DEFAULT_MODEL_PATH

MODEL = joblib.load(MODEL_PATH)

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
    return render_template("index.html", error=error)


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
