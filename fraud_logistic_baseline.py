"""Logistic Regression fraud baseline using chronological evaluation.

Run from your project folder:
    /opt/anaconda3/bin/python fraud_logistic_baseline.py

Requires credit_card_fraud_dataset.csv in the same folder.
Creates separate fraud_logistic_* files; never overwrites RF v1/v2 outputs.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ALERT_RATE = 0.01  # Target the highest-scoring ~1% of validation transactions


def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    # No class weighting: a straightforward, regularized baseline.
    # Evaluate probabilities separately for calibration before calling them risks.
    model = LogisticRegression(max_iter=3000, solver="lbfgs", C=1.0)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main():
    folder = Path(__file__).resolve().parent
    input_file = folder / "credit_card_fraud_dataset.csv"
    model_file = folder / "fraud_logistic_pipeline.pkl"
    scored_file = folder / "fraud_logistic_scored.csv"
    test_file = folder / "fraud_logistic_test_scored.csv"
    metrics_file = folder / "fraud_logistic_metrics.json"

    if not input_file.exists():
        raise FileNotFoundError(f"Dataset not found: {input_file}")

    original = pd.read_csv(input_file).dropna(how="all").copy()
    if original.empty:
        raise ValueError("The dataset contains no records.")
    needed = {"TransactionDate", "IsFraud"}
    missing = needed - set(original.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    if original["IsFraud"].isna().any() or set(original["IsFraud"].unique()) != {0, 1}:
        raise ValueError("IsFraud must contain only 0/1 and both classes must exist.")

    # Preserve original columns for reporting. Sort BEFORE engineering time features.
    original["TransactionDate"] = pd.to_datetime(original["TransactionDate"], errors="raise")
    original = original.sort_values("TransactionDate", kind="stable").reset_index(drop=True)
    dt = original["TransactionDate"]

    features = original.drop(columns=["IsFraud", "TransactionID", "TransactionDate"], errors="ignore").copy()
    features["txn_hour"] = dt.dt.hour
    features["txn_dayofweek"] = dt.dt.dayofweek
    features["txn_day"] = dt.dt.day
    features["txn_month"] = dt.dt.month
    if "MerchantID" in features:
        features["MerchantID"] = features["MerchantID"].astype(str)

    target = original["IsFraud"].astype(int)
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()

    n = len(features)
    train_end, val_end = int(n * 0.60), int(n * 0.80)
    X_train, y_train = features.iloc[:train_end], target.iloc[:train_end]
    X_val, y_val = features.iloc[train_end:val_end], target.iloc[train_end:val_end]
    X_test, y_test = features.iloc[val_end:], target.iloc[val_end:]

    print(f"[INFO] Dataset: {n:,} rows")
    print(f"[INFO] Features: {len(numeric)} numeric | {len(categorical)} categorical")
    for name, split_y in (("Training", y_train), ("Validation", y_val), ("Testing", y_test)):
        print(f"[INFO] {name}: {len(split_y):,} rows | Frauds: {int(split_y.sum()):,}")
        if split_y.nunique() != 2:
            raise ValueError(f"{name} data has only one class, so evaluation is not possible.")

    for label, point in (("training/validation", train_end), ("validation/testing", val_end)):
        if original.loc[point - 1, "TransactionDate"] == original.loc[point, "TransactionDate"]:
            print(f"[WARN] Identical timestamps on both sides of the {label} boundary.")

    pipeline = build_pipeline(numeric, categorical)
    print("[INFO] Training Logistic Regression on the oldest 60% of transactions...")
    pipeline.fit(X_train, y_train)

    # Choose the alert threshold on validation only; leave the test data untouched.
    val_scores = pipeline.predict_proba(X_val)[:, 1]
    threshold = float(np.quantile(val_scores, 1 - ALERT_RATE))
    val_alerts = (val_scores >= threshold).astype(int)

    test_scores = pipeline.predict_proba(X_test)[:, 1]
    test_alerts = (test_scores >= threshold).astype(int)
    roc_auc = float(roc_auc_score(y_test, test_scores))
    average_precision = float(average_precision_score(y_test, test_scores))
    cm = confusion_matrix(y_test, test_alerts, labels=[0, 1]).tolist()
    report = classification_report(y_test, test_alerts, output_dict=True, zero_division=0)

    print(f"[RESULT] Test ROC-AUC: {roc_auc:.4f}")
    print(f"[RESULT] Test average precision: {average_precision:.4f}")
    print(f"[RESULT] Test fraud base rate: {y_test.mean():.2%}")
    print(f"[RESULT] Fixed validation threshold: {threshold:.6f}")
    print(f"[RESULT] Validation alert rate: {val_alerts.mean():.2%}")
    print(f"[RESULT] Test alert rate: {test_alerts.mean():.2%}")
    print(f"[RESULT] Test confusion matrix: {cm}")
    print(classification_report(y_test, test_alerts, zero_division=0))

    # Full-data scores are descriptive; they include training records.
    full_scores = pipeline.predict_proba(features)[:, 1]
    full_scored = original.copy()
    full_scored["Fraud_Probability"] = full_scores
    full_scored["Fraud_Flag"] = (full_scores >= threshold).astype(int)
    full_scored.to_csv(scored_file, index=False)

    # This file contains only chronologically held-out test transactions.
    test_scored = original.iloc[val_end:].copy()
    test_scored["Fraud_Probability"] = test_scores
    test_scored["Fraud_Flag"] = test_alerts
    test_scored.to_csv(test_file, index=False)

    joblib.dump(pipeline, model_file)
    metrics = {
        "model_choice": "logistic_regression",
        "evaluation_split": "chronological_60_20_20",
        "roc_auc": roc_auc,
        "pr_auc": average_precision,
        "alert_rate": ALERT_RATE,
        "threshold_validation": threshold,
        "threshold_test": threshold,
        "threshold_full": threshold,
        "validation_alert_rate_observed": float(val_alerts.mean()),
        "test_alert_rate_observed": float(test_alerts.mean()),
        "test_fraud_base_rate": float(y_test.mean()),
        "confusion_matrix": cm,
        "classification_report": report,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "target_col": "IsFraud",
        "split_counts": {
            "train": len(X_train), "validation": len(X_val), "test": len(X_test)
        },
        "notes": [
            "Alert threshold selected using validation data only.",
            "Only held-out test metrics should be used to evaluate generalization.",
            "Full scored portfolio includes training records.",
            "Fraud probability calibration has not been verified.",
        ],
    }
    with open(metrics_file, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    for output in (scored_file, test_file, model_file, metrics_file):
        print(f"[SAVED] {output}")
    print("[DONE] Logistic Regression baseline completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}")
        sys.exit(1)
