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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)


def safe_read_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    return df


def build_credit_pipeline(numeric_features, categorical_features) -> Pipeline:
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

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",  # helps if defaults are minority class
        solver="lbfgs",
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def assign_risk_tier(pd_scores: pd.Series) -> pd.Series:
    # Simple, interview-friendly tiers (you can tune later)
    return pd.cut(
        pd_scores,
        bins=[0.0, 0.05, 0.15, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )


def main():
    project_dir = Path(r"C:\Users\Dell\Desktop\CreditFraudRisk")
    data_path = project_dir / "credit_risk_dataset.csv"

    # Outputs
    model_path = project_dir / "credit_risk_pipeline.pkl"
    scored_path = project_dir / "credit_scored.csv"
    metrics_path = project_dir / "credit_metrics.json"

    print(f"[INFO] Loading credit dataset from: {data_path}")
    df = safe_read_csv(data_path)

    target_col = "loan_status"
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")

    # Drop obvious ID-like columns if present (optional safety)
    possible_id_cols = [c for c in df.columns if "id" in c.lower()]
    if possible_id_cols:
        print(f"[INFO] Dropping possible ID columns: {possible_id_cols}")
        df = df.drop(columns=possible_id_cols)

    # Split X/y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify feature types
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    print(f"[INFO] Numeric features: {len(numeric_features)} | Categorical features: {len(categorical_features)}")

    # Train/test split (stratify maintains default ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_credit_pipeline(numeric_features, categorical_features)

    print("[INFO] Training credit risk model (Logistic Regression + preprocessing pipeline)...")
    pipeline.fit(X_train, y_train)

    # Predict PD
    pd_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (pd_test >= 0.5).astype(int)

    # Metrics
    roc_auc = roc_auc_score(y_test, pd_test)
    pr_auc = average_precision_score(y_test, pd_test)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"[RESULT] ROC-AUC: {roc_auc:.4f}")
    print(f"[RESULT] PR-AUC : {pr_auc:.4f}")
    print("[RESULT] Confusion Matrix:", cm)

    # Score full dataset for dashboard (PD, tiers, EL)
    print("[INFO] Scoring full dataset for dashboard outputs...")
    pd_all = pipeline.predict_proba(X)[:, 1]
    scored = df.copy()
    scored["PD"] = pd_all
    scored["Risk_Tier"] = assign_risk_tier(scored["PD"]).astype(str)

    # Expected Loss: EL = PD * LGD * EAD
    # EAD: we will use loan_amnt if present, else skip EL
    LGD_ASSUMPTION = 0.45  # interview-friendly assumption (can mention rationale)
    if "loan_amnt" in scored.columns:
        scored["EAD"] = scored["loan_amnt"]
        scored["LGD"] = LGD_ASSUMPTION
        scored["Expected_Loss"] = scored["PD"] * scored["LGD"] * scored["EAD"]
    else:
        print("[WARN] 'loan_amnt' not found; skipping Expected Loss calculation.")

    # Save scored CSV
    scored.to_csv(scored_path, index=False)
    print(f"[INFO] Saved scored credit file: {scored_path}")

    # Save model pipeline (preprocessing + model)
    joblib.dump(pipeline, model_path)
    print(f"[INFO] Saved credit pipeline model: {model_path}")

    # Save metrics JSON
    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "lgd_assumption": LGD_ASSUMPTION,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved credit metrics: {metrics_path}")

    print("\n[DONE] Credit Risk modeling completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)