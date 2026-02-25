import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier


def safe_read_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    return df


def is_binary_01(series: pd.Series) -> bool:
    s = series.dropna()
    if s.empty:
        return False
    uniq = set(pd.unique(s))
    return uniq.issubset({0, 1}) and len(uniq) >= 2


def detect_target_column(df: pd.DataFrame) -> str:
    common = ["IsFraud", "Class", "is_fraud", "isFraud", "fraud", "Fraud", "target", "label", "y"]
    for col in common:
        if col in df.columns:
            return col

    binary_candidates = []
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 10 and is_binary_01(df[col]):
            binary_candidates.append(col)

    for pref in ["fraud", "class", "label", "target"]:
        for col in binary_candidates:
            if pref in col.lower():
                return col

    if len(binary_candidates) == 1:
        return binary_candidates[0]

    raise KeyError(
        "Could not confidently detect target column.\n"
        f"Columns: {df.columns.tolist()}\n"
        f"Binary candidates: {binary_candidates}\n"
        "Fix: set TARGET_COL manually."
    )


def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Convert date column to datetime and extract useful fraud time features.
    """
    if date_col not in df.columns:
        return df

    dt = pd.to_datetime(df[date_col], errors="coerce")
    df = df.copy()

    df["txn_hour"] = dt.dt.hour
    df["txn_dayofweek"] = dt.dt.dayofweek
    df["txn_day"] = dt.dt.day
    df["txn_month"] = dt.dt.month

    # Drop original datetime to avoid raw datetime text issues
    df = df.drop(columns=[date_col])
    return df


def build_pipeline(numeric_features, categorical_features) -> Pipeline:
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


def main():
    project_dir = Path(r"C:\Users\Dell\Desktop\CreditFraudRisk")
    data_path = project_dir / "credit_card_fraud_dataset.csv"

    model_path = project_dir / "fraud_pipeline.pkl"
    scored_path = project_dir / "fraud_scored.csv"
    metrics_path = project_dir / "fraud_metrics.json"

    print(f"[INFO] Loading fraud dataset from: {data_path}")
    df = safe_read_csv(data_path).dropna(how="all")

    TARGET_COL = None  # set manually if needed
    target_col = TARGET_COL if TARGET_COL else detect_target_column(df)
    print(f"[INFO] Fraud target column: {target_col}")

    # Feature engineering: keep signal from TransactionDate
    df = add_time_features(df, date_col="TransactionDate")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # Identify feature types
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    print(f"[INFO] Numeric features: {len(numeric_features)} | Categorical features: {len(categorical_features)}")

    # Train/test split (stratify keeps fraud ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(numeric_features, categorical_features)

    print("[INFO] Training fraud model (RandomForest + preprocessing)...")
    pipeline.fit(X_train, y_train)

    # Predict probabilities on test
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print(f"[RESULT] ROC-AUC: {roc_auc:.4f}")
    print(f"[RESULT] PR-AUC : {pr_auc:.4f}")


    # Use a realistic alert rate: flag only the top 1% highest-risk transactions
    ALERT_RATE = 0.01  
    test_threshold = float(np.quantile(y_prob, 1 - ALERT_RATE))
    y_pred = (y_prob >= test_threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"[RESULT] Alert rate target: {ALERT_RATE:.2%}")
    print(f"[RESULT] Threshold used (test): {test_threshold:.6f}")
    print("[RESULT] Confusion Matrix:", cm)
    print("[RESULT] Classification summary:")
    print(classification_report(y_test, y_pred))

    # Score full dataset
    print("[INFO] Scoring full dataset...")
    prob_all = pipeline.predict_proba(X)[:, 1]

    # Use the same alert-rate rule on full dataset
    full_threshold = float(np.quantile(prob_all, 1 - ALERT_RATE))

    scored = df.copy()
    scored["Fraud_Probability"] = prob_all
    scored["Fraud_Flag"] = (scored["Fraud_Probability"] >= full_threshold).astype(int)

    scored.to_csv(scored_path, index=False)

    # Save pipeline (preprocessing + model)
    joblib.dump(pipeline, model_path)

    # Save metrics
    metrics = {
        "model_choice": "rf",
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "alert_rate": ALERT_RATE,
        "threshold_test": test_threshold,
        "threshold_full": full_threshold,
        "confusion_matrix": cm,
        "classification_report": report,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_col": target_col,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Saved scored fraud file: {scored_path}")
    print(f"[INFO] Saved fraud pipeline model: {model_path}")
    print(f"[INFO] Saved fraud metrics: {metrics_path}")
    print("\n[DONE] Fraud modeling completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)