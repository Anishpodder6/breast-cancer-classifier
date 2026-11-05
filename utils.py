"""Helpers shared by the Streamlit app. Kept free of Streamlit so they can be tested directly."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve,
)

MODEL_DIR = Path(__file__).resolve().parent / "model"


def load_metadata() -> dict:
    return json.loads((MODEL_DIR / "metadata.json").read_text())


def load_models(meta: dict) -> dict:
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in meta["models"].items()}


def encode_target(series: pd.Series, positive: str) -> pd.Series:
    """Accept M/B, malignant/benign or 1/0 in the target column."""
    s = series.astype(str).str.strip().str.lower()
    pos = {positive.lower(), "malignant", "1", "1.0"}
    neg = {"b", "benign", "0", "0.0"}
    unknown = sorted(set(s) - pos - neg)
    if unknown:
        raise ValueError(
            f"Target column has values the app cannot read: {unknown[:5]}. "
            "Use M/B, malignant/benign or 1/0."
        )
    return s.isin(pos).astype(int)


def validate_upload(df: pd.DataFrame, meta: dict):
    """Return (X, y) in the training feature order, or raise ValueError with a fix-it message."""
    df = df.rename(columns=lambda c: str(c).strip())
    target = meta["target"]
    if target not in df.columns:
        raise ValueError(f"The file has no '{target}' column. Add it so the predictions can be scored.")
    missing = [f for f in meta["features"] if f not in df.columns]
    if missing:
        raise ValueError(f"{len(missing)} feature column(s) are missing, for example: {missing[:4]}")
    df = df.dropna(subset=meta["features"] + [target])
    if df.empty:
        raise ValueError("Every row has a missing value, so nothing can be scored.")
    X = df[meta["features"]].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Non-numeric values found in: {bad[:4]}")
    y = encode_target(df[target], meta["positive_label"])
    return X, y


def positive_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def score_model(model, X, y) -> dict:
    y_pred = model.predict(X)
    y_score = positive_scores(model, X)
    both_classes = y.nunique() == 2
    return {
        "y_pred": y_pred,
        "y_score": y_score,
        "metrics": {
            "Accuracy": accuracy_score(y, y_pred),
            "AUC": roc_auc_score(y, y_score) if both_classes else np.nan,
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred, zero_division=0),
            "F1": f1_score(y, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y, y_pred),
        },
        "cm": confusion_matrix(y, y_pred, labels=[0, 1]),
        "report": pd.DataFrame(
            classification_report(
                y, y_pred, labels=[0, 1], target_names=["Benign (B)", "Malignant (M)"],
                output_dict=True, zero_division=0,
            )
        ).T,
        "roc": roc_curve(y, y_score) if both_classes else None,
    }
