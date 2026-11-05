"""
Train and evaluate five classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset.

Outputs
-------
model/<name>.joblib      fitted sklearn pipelines (scaler + estimator where needed)
model/metadata.json      feature order, label mapping, chosen hyper-parameters
model/metrics.csv        test-set metrics for all models (used in README)
test_data.csv            held-out 20% split, with the target column, for the Streamlit app

Run:  python train_models.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "wdbc.csv"
MODEL_DIR = ROOT / "model"
TARGET = "diagnosis"
POSITIVE = "M"          # malignant is the class we care about catching
SEED = 47
TEST_SIZE = 0.20


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # The Kaggle copy ships with an 'id' column and an empty trailing column.
    df = df.drop(columns=[c for c in df.columns if c.lower() == "id" or c.startswith("Unnamed")])
    df = df.dropna(axis=0)
    return df


def build_search_spaces():
    """Each model gets a small, explainable grid tuned with 5-fold CV on training data only."""
    return {
        "Logistic Regression": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000, random_state=SEED))]),
            {"clf__C": [0.01, 0.1, 0.5, 1, 5, 10]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=SEED),
            {"max_depth": [3, 4, 5, 6, 8, None],
             "min_samples_leaf": [1, 3, 5, 10],
             "criterion": ["gini", "entropy"]},
        ),
        "KNN": (
            Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [3, 5, 7, 9, 11, 15],
             "clf__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
        ),
        "Support Vector Machine": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", SVC(kernel="rbf", random_state=SEED))]),
            {"clf__C": [0.1, 1, 5, 10, 50], "clf__gamma": ["scale", 0.01, 0.05, 0.1]},
        ),
    }


def positive_scores(model, X):
    """Score for the positive class. SVC has no predict_proba unless calibrated,
    and the ROC curve only needs a ranking, so its decision_function is used."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def evaluate(model, X, y_true) -> dict:
    y_pred = model.predict(X)
    y_score = positive_scores(model, X)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def file_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_") + ".joblib"


def main():
    df = load_dataset(DATA_PATH)
    features = [c for c in df.columns if c != TARGET]
    X = df[features]
    y = (df[TARGET] == POSITIVE).astype(int)

    print(f"Dataset: {df.shape[0]} rows, {len(features)} features")
    print(f"Class balance: {y.value_counts().to_dict()}  (1 = {POSITIVE})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    MODEL_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, chosen_params = [], {}

    for name, (estimator, grid) in build_search_spaces().items():
        search = GridSearchCV(estimator, grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        chosen_params[name] = {k: (v if v is not None else "None") for k, v in search.best_params_.items()}

        joblib.dump(best, MODEL_DIR / file_name(name))
        metrics = evaluate(best, X_test, y_test)
        rows.append({"Model": name, **metrics})
        print(f"{name:<24} CV-F1={search.best_score_:.4f}  params={search.best_params_}")

    results = pd.DataFrame(rows).set_index("Model").round(4)
    results.to_csv(MODEL_DIR / "metrics.csv")
    print("\nHeld-out test metrics\n", results.to_string())

    test_df = X_test.copy()
    test_df[TARGET] = np.where(y_test == 1, "M", "B")
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    metadata = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic), UCI ML Repository",
        "target": TARGET,
        "positive_label": POSITIVE,
        "labels": {"1": "M (malignant)", "0": "B (benign)"},
        "features": features,
        "random_state": SEED,
        "test_size": TEST_SIZE,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "models": {name: file_name(name) for name in chosen_params},
        "best_params": chosen_params,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved models and metadata to {MODEL_DIR}/ and test split to test_data.csv")


if __name__ == "__main__":
    main()
