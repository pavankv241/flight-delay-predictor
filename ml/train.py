"""Train a flight delay classifier and save artifacts for the API."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_flights.csv"
MODEL_DIR = ROOT / "backend" / "models"
MODEL_PATH = MODEL_DIR / "delay_model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"
WEIGHTS_PATH = ROOT / "frontend" / "src" / "model_weights.json"

CATEGORICAL = ["airline", "origin", "dest"]
NUMERIC = ["month", "day_of_week", "hour", "distance"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "delayed"


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
            ("num", StandardScaler(), NUMERIC),
        ]
    )


def build_gbm_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=120,
                    learning_rate=0.08,
                    max_depth=3,
                    random_state=42,
                ),
            ),
        ]
    )


def build_lr_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def export_weights(pipe: Pipeline, meta: dict) -> None:
    """Export a JSON scoring package for the Vercel TypeScript API."""
    prep: ColumnTransformer = pipe.named_steps["prep"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    ohe: OneHotEncoder = prep.named_transformers_["cat"]
    scaler: StandardScaler = prep.named_transformers_["num"]

    cat_features = list(ohe.get_feature_names_out(CATEGORICAL))
    payload = {
        "model_version": meta["model_version"],
        "algorithm": "LogisticRegression (Vercel export)",
        "categorical_features": CATEGORICAL,
        "numeric_features": NUMERIC,
        "one_hot_categories": {
            name: list(cats) for name, cats in zip(CATEGORICAL, ohe.categories_)
        },
        "one_hot_feature_names": cat_features,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "metrics": meta["metrics"],
        "airlines": meta["airlines"],
        "airports": meta["airports"],
    }
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(json.dumps(payload))
    print(f"Saved JS weights → {WEIGHTS_PATH}")


def main() -> None:
    if not DATA_PATH.exists():
        from generate_data import main as generate

        generate()

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_gbm_pipeline()
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    accuracy = float(accuracy_score(y_test, pred))
    auc = float(roc_auc_score(y_test, proba))

    print("GBM classification report:")
    print(classification_report(y_test, pred, digits=3))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    airlines = sorted(df["airline"].unique().tolist())
    airports = sorted(set(df["origin"].tolist()) | set(df["dest"].tolist()))

    meta = {
        "model_version": "1.0.0",
        "algorithm": "GradientBoostingClassifier",
        "features": FEATURES,
        "categorical_features": CATEGORICAL,
        "numeric_features": NUMERIC,
        "target": TARGET,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "roc_auc": round(auc, 4),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "delay_rate": round(float(y.mean()), 4),
        },
        "airlines": airlines,
        "airports": airports,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    lr_pipe = build_lr_pipeline()
    lr_pipe.fit(X_train, y_train)
    lr_auc = float(roc_auc_score(y_test, lr_pipe.predict_proba(X_test)[:, 1]))
    print(f"LR export ROC-AUC: {lr_auc:.4f}")
    export_meta = {
        **meta,
        "metrics": {
            **meta["metrics"],
            "vercel_lr_roc_auc": round(lr_auc, 4),
        },
    }
    export_weights(lr_pipe, export_meta)

    print(f"Saved model → {MODEL_PATH}")
    print(f"Saved meta  → {META_PATH}")


if __name__ == "__main__":
    main()
