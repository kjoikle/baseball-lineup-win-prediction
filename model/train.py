from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

# Paths — relative to model/
TRAIN_PATH  = Path("data/train_features.parquet")
TEST_PATH   = Path("data/test_features.parquet")
OUTPUT_PATH = Path("../app/models/model.pkl")  # app expects model/model.pkl

# Feature engineering
# USE_DIFFERENTIAL : True  → 59 home-minus-away diff features (recommended)
#                    False → 118 raw home/away features
USE_DIFFERENTIAL = True
USE_SCALE        = False  # StandardScaler before model; required when solver='saga'

# Polynomial expansion — set POLY_DEGREE=2 to add interaction terms, None to skip
POLY_DEGREE      = None
POLY_INTERACTION = True   # interaction-only (no x²); ignored when POLY_DEGREE is None

# LogisticRegression hyperparameters (sklearn ≥ 1.8 API — no penalty kwarg)
#   l1_ratio : 0.0 = pure L2 | 1.0 = pure L1 | (0, 1) = ElasticNet
#   C        : inverse regularization strength; smaller = stronger regularization
#   solver   
RANDOM_SEED  = 42
MODEL_PARAMS = {
    "solver": "saga",
    "C": 0.2336,
    "l1_ratio": 0.5,        # 0 = pure L2, 1 = pure L1
    "max_iter": 10000,
    "random_state": 42,
}

# ── Imports ───────────────────────────────────────────────────────────────────

import warnings
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

METADATA_COLS    = ["date", "home_team", "away_team", "home_win"]
BATTING_STATS    = ["BA", "OBP", "SLG", "OPS", "K%", "BB%"]
PITCHING_STATS   = ["ERA", "WHIP", "SO9", "SO/W", "IP"]
LOWER_IS_BETTER  = {"ERA", "WHIP"}

# ── Feature engineering ───────────────────────────────────────────────────────

def make_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return home-minus-away differential features (positive = home advantage)."""
    diffs = {}
    for n in range(1, 10):
        for stat in BATTING_STATS:
            diffs[f"diff_bat{n}_{stat}"] = df[f"home_bat{n}_{stat}"] - df[f"away_bat{n}_{stat}"]
    for stat in PITCHING_STATS:
        if stat in LOWER_IS_BETTER:
            diffs[f"diff_sp_{stat}"] = df[f"away_sp_{stat}"] - df[f"home_sp_{stat}"]
        else:
            diffs[f"diff_sp_{stat}"] = df[f"home_sp_{stat}"] - df[f"away_sp_{stat}"]
    return pd.DataFrame(diffs, index=df.index)


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    feat_cols = [c for c in train_df.columns if c not in METADATA_COLS]

    if USE_DIFFERENTIAL:
        X_tr = make_differential_features(train_df)
        X_te = make_differential_features(test_df)
    else:
        X_tr = train_df[feat_cols].copy()
        X_te = test_df[feat_cols].copy()

    feature_names = list(X_tr.columns)

    if USE_SCALE:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
    else:
        X_tr = X_tr.values
        X_te = X_te.values

    if POLY_DEGREE is not None:
        poly = PolynomialFeatures(
            degree=POLY_DEGREE,
            interaction_only=POLY_INTERACTION,
            include_bias=False,
        )
        X_tr = poly.fit_transform(X_tr)
        X_te = poly.transform(X_te)
        feature_names = poly.get_feature_names_out(feature_names).tolist()

    return X_tr, X_te, feature_names

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df  = pd.read_parquet(TEST_PATH)

    y_train  = train_df["home_win"].values
    y_test   = test_df["home_win"].values

    print(f"  Train: {train_df.shape}  Test: {test_df.shape}")

    print("Building features...")
    X_train, X_test, feature_names = build_features(train_df, test_df)
    print(f"  {X_train.shape[1]} features  ({'differential' if USE_DIFFERENTIAL else 'raw'})")

    print("Training...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(**MODEL_PARAMS)
        model.fit(X_train, y_train)

    if model.n_iter_[0] >= MODEL_PARAMS.get("max_iter", 100):
        print(f"  WARNING: max_iter reached ({model.n_iter_[0]}) — consider increasing max_iter")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"  Test accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Test AUC      : {roc_auc_score(y_test, y_prob):.4f}")

    joblib.dump(model, OUTPUT_PATH)
    print(f"Saved model to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
