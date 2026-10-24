"""Fraud detection model (v2): chronological validation and a fixed alert threshold.

Run with:
    /opt/anaconda3/bin/python fraud_model_v2.py

Input: credit_card_fraud_dataset.csv (in the same folder).
Outputs use _v2 suffix and do not overwrite original model artifacts.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def safe_read_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    return df


def is_binary_01(series: pd.Series) -> bool:
    observed = series.dropna()
    return not observed.empty and set(pd.unique(observed)) == {0, 1}


def detect_target_column(df: pd.DataFrame) -> str:
    for name in ("IsFraud", "Class", "is_fraud", "isFraud", "fraud", "Fraud", "target", "label", "y"):
        if name in df.columns:
            return name

    binary_candidates = [
        name for name in df.columns
        if df[name].nunique(dropna=True) <= 10 and is_binary_01(df[name])
    ]
    if len(binary_candidates) == 1:
        return binary_candidates[0]
    raise KeyError(
        "Could not determine the fraud target column. "
        f"Available columns: {df.columns.tolist()}"
    )


def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Derive model-ready time columns without treating date text as a feature."""
    result = df.copy()
    dt = pd.to_datetime(result[date_col], errors="raise")
    result["txn_hour"] = dt.dt.hour
    result["txn_dayofweek"] = dt.dt.dayofweek
    result["txn_day"] = dt.dt.day
    result["txn_month"] = dt.dt.month
    return result.drop(columns=[date_col])


def build_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=2,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "credit_card_fraud_dataset.csv"

    model_path = project_dir / "fraud_pipeline_v2.pkl"
    scored_path = project_dir / "fraud_scored_v2.csv"
    metrics_path = project_dir / "fraud_metrics_v2.json"
    test_scored_path = project_dir / "fraud_test_scored_v2.csv"

    print(f"[INFO] Loading fraud dataset from: {data_path}")
    original = safe_read_csv(data_path).dropna(how="all").copy()
    if original.empty:
        raise ValueError("There are no nonempty transactions to train on.")

    target_col = detect_target_column(original)
    print(f"[INFO] Fraud target column: {target_col}")
    if original[target_col].isna().any() or not is_binary_01(original[target_col]):
        raise ValueError(f"'{target_col}' must contain nonmissing binary values 0 and 1.")
    if "TransactionDate" not in original.columns:
        raise KeyError("Missing required TransactionDate column.")

    # Maintain original dates and identifiers for audit-friendly scored CSVs.
    original["TransactionDate"] = pd.to_datetime(
        original["TransactionDate"], errors="raise"
    )
    original = original.sort_values("TransactionDate", kind="stable").reset_index(drop=True)

    model_data = add_time_features(original, date_col="TransactionDate")
    X = model_data.drop(columns=[target_col, "TransactionID"], errors="ignore").copy()
    if "MerchantID" in X.columns:
        # Merchant identity is categorical, not a continuous numeric measurement.
        X["MerchantID"] = X["MerchantID"].astype(str)
    y = model_data[target_col].astype(int)

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()
    print(
        f"[INFO] Numeric features: {len(numeric_features)} | "
        f"Categorical features: {len(categorical_features)}"
    )

    # Chronological, non-shuffled 60%/20%/20% split.
    n = len(X)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    print("\n[INFO] Chronological dataset split:")
    for name, y_part in (
        ("Training", y_train),
        ("Validation", y_val),
        ("Testing", y_test),
    ):
        print(f"{name:11s}: {len(y_part):,} rows | Frauds: {int(y_part.sum()):,}")
        if y_part.nunique() < 2:
            raise ValueError(f"{name} set has only one target class; cannot evaluate.")

    # If dates are tied across a boundary, report this limitation.
    for label, boundary in (("train/validation", train_end), ("validation/test", val_end)):
        if original.loc[boundary - 1, "TransactionDate"] == original.loc[boundary, "TransactionDate"]:
            print(f"[WARN] Identical timestamps span the {label} split boundary.")

    pipeline = build_pipeline(numeric_features, categorical_features)
    print("[INFO] Training v2 RandomForest on training data only...")
    pipeline.fit(X_train, y_train)

    # Choose the operating threshold using VALIDATION scores only.
    ALERT_RATE = 0.01
    val_prob = pipeline.predict_proba(X_val)[:, 1]
    fixed_threshold = float(np.quantile(val_prob, 1 - ALERT_RATE))
    val_flags = (val_prob >= fixed_threshold).astype(int)

    # Evaluate once on future TEST transactions, with the same fixed threshold.
    test_prob = pipeline.predict_proba(X_test)[:, 1]
    test_flags = (test_prob >= fixed_threshold).astype(int)
    roc_auc = float(roc_auc_score(y_test, test_prob))
    pr_auc = float(average_precision_score(y_test, test_prob))
    cm = confusion_matrix(y_test, test_flags, labels=[0, 1]).tolist()
    report = classification_report(y_test, test_flags, output_dict=True, zero_division=0)

    print(f"[RESULT] Test ROC-AUC: {roc_auc:.4f}")
    print(f"[RESULT] Test average precision: {pr_auc:.4f}")
    print(f"[RESULT] Fraud base rate (test): {y_test.mean():.2%}")
    print(f"[RESULT] Target validation alert rate: {ALERT_RATE:.2%}")
    print(f"[RESULT] Threshold chosen from validation: {fixed_threshold:.6f}")
    print(f"[RESULT] Actual validation alert rate: {val_flags.mean():.2%}")
    print(f"[RESULT] Actual test alert rate: {test_flags.mean():.2%}")
    print("[RESULT] Test confusion matrix:", cm)
    print(classification_report(y_test, test_flags, zero_division=0))

    # Full scored portfolio is DESCRIPTIVE: includes training records.
    # Use exactly the SAME threshold, never recompute it on test/full data.
    print("[INFO] Scoring full dataset with the fixed validation threshold...")
    full_prob = pipeline.predict_proba(X)[:, 1]
    scored = original.copy()
    scored["Fraud_Probability"] = full_prob
    scored["Fraud_Flag"] = (full_prob >= fixed_threshold).astype(int)
    scored.to_csv(scored_path, index=False)

    # Separate held-out test record file for defensible reporting.
    test_scored = original.iloc[val_end:].copy()
    test_scored["Fraud_Probability"] = test_prob
    test_scored["Fraud_Flag"] = test_flags
    test_scored.to_csv(test_scored_path, index=False)

    joblib.dump(pipeline, model_path)
    metrics = {
        "model_choice": "rf",
        "evaluation_split": "chronological_60_20_20",
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "alert_rate": ALERT_RATE,
        "threshold_validation": fixed_threshold,
        # Keep legacy keys for compatibility; both now use the fixed threshold.
        "threshold_test": fixed_threshold,
        "threshold_full": fixed_threshold,
        "validation_alert_rate_observed": float(val_flags.mean()),
        "test_alert_rate_observed": float(test_flags.mean()),
        "test_fraud_base_rate": float(y_test.mean()),
        "confusion_matrix": cm,
        "classification_report": report,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_col": target_col,
        "split_counts": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "notes": [
            "Threshold selected on validation data, not test data.",
            "Full-data scores include training records; use test metrics for evaluation.",
            "Fraud probabilities have not been assessed for calibration.",
        ],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Saved scored portfolio: {scored_path}")
    print(f"[INFO] Saved held-out test predictions: {test_scored_path}")
    print(f"[INFO] Saved fraud pipeline: {model_path}")
    print(f"[INFO] Saved fraud metrics: {metrics_path}")
    print("\n[DONE] Fraud model v2 completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        sys.exit(1)
