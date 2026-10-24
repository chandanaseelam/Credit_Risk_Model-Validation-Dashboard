
"""Credit risk model v2 with probability calibration."""

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
LGD_ASSUMPTION = 0.45


def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    classifier = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", classifier),
    ])


def log_odds(probabilities):
    """Convert probabilities into stable log-odds."""
    p = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def predict_calibrated(pipeline, calibrator, X):
    raw = pipeline.predict_proba(X)[:, 1]
    calibrated = calibrator.predict_proba(log_odds(raw))[:, 1]
    return raw, calibrated


def evaluate(y_true, probabilities):
    labels = (probabilities >= 0.5).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "log_loss": float(
            log_loss(y_true, probabilities, labels=[0, 1])
        ),
        "confusion_matrix_at_0_5": confusion_matrix(
            y_true, labels, labels=[0, 1]
        ).tolist(),
        "classification_report_at_0_5": classification_report(
            y_true,
            labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def calibration_bins(y_true, scores, label):
    frame = pd.DataFrame({
        "observed": np.asarray(y_true),
        "score": np.asarray(scores),
    })

    frame["bucket"] = pd.qcut(
        frame["score"], q=10, duplicates="drop"
    )

    result = frame.groupby(
        "bucket", observed=True
    ).agg(
        records=("observed", "size"),
        mean_predicted_pd=("score", "mean"),
        observed_default_rate=("observed", "mean"),
    ).reset_index(drop=True)

    result.insert(0, "score_type", label)
    result.insert(1, "bin_number", range(1, len(result) + 1))
    return result


def main():
    folder = Path(__file__).resolve().parent
    source = folder / "credit_risk_dataset.csv"

    if not source.exists():
        raise FileNotFoundError(
            f"Dataset not found: {source}"
        )

    df = pd.read_csv(source)

    if df.empty or "loan_status" not in df:
        raise ValueError(
            "Dataset must include loan_status."
        )

    y = pd.to_numeric(
        df["loan_status"], errors="raise"
    )

    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError(
            "loan_status must contain 0 and 1."
        )

    y = y.astype(int)

    # Remove identifiers from training features.
    id_columns = [
        c for c in df.columns
        if c.lower() == "id"
        or c.lower().endswith("_id")
    ]

    X = df.drop(
        columns=["loan_status"] + id_columns
    )

    numeric = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if not numeric and not categorical:
        raise ValueError("No usable features found.")

    # Keep the original 20% test indices.
    X_trainval, X_test, y_trainval, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    # Reserve a separate 20% for calibration.
    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_trainval,
            y_trainval,
            test_size=0.25,
            stratify=y_trainval,
            random_state=RANDOM_STATE,
        )
    )

    print(f"[INFO] Dataset: {len(df):,} loans")
    print(
        f"[INFO] Features: {len(numeric)} numeric | "
        f"{len(categorical)} categorical"
    )

    if id_columns:
        print(
            f"[INFO] Excluded identifiers: {id_columns}"
        )

    for name, labels in [
        ("Training", y_train),
        ("Validation", y_val),
        ("Test", y_test),
    ]:
        print(
            f"[INFO] {name}: {len(labels):,} loans | "
            f"defaults: {int(labels.sum()):,}"
        )

    # Train the original classifier approach.
    model = build_pipeline(numeric, categorical)

    print("[INFO] Training credit classifier...")
    model.fit(X_train, y_train)

    # Calibrate using validation data only.
    val_raw = model.predict_proba(X_val)[:, 1]

    calibrator = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
    )

    calibrator.fit(log_odds(val_raw), y_val)

    print(
        "[INFO] Sigmoid calibrator fitted on "
        "validation predictions."
    )

    # Evaluate on held-out test data.
    test_raw, test_cal = predict_calibrated(
        model, calibrator, X_test
    )

    before = evaluate(y_test, test_raw)
    after = evaluate(y_test, test_cal)

    baseline_brier = float(
        brier_score_loss(
            y_test,
            np.full(len(y_test), float(y_train.mean())),
        )
    )

    print("\n[RESULT] Held-out TEST results")
    print("Metric                 Raw model       Calibrated")

    for metric in (
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
    ):
        print(
            f"{metric:22s} "
            f"{before[metric]:.4f}          "
            f"{after[metric]:.4f}"
        )

    print(
        f"[RESULT] Test default rate: "
        f"{y_test.mean():.2%}"
    )
    print(
        f"[RESULT] Test mean raw PD: "
        f"{test_raw.mean():.2%}"
    )
    print(
        f"[RESULT] Test mean calibrated PD: "
        f"{test_cal.mean():.2%}"
    )
    print(
        f"[RESULT] Constant-prevalence baseline Brier: "
        f"{baseline_brier:.4f}"
    )
    print(
        "[RESULT] Calibrated confusion matrix: "
        f"{after['confusion_matrix_at_0_5']}"
    )

    # Generate full-dataset dashboard scores.
    raw_all, cal_all = predict_calibrated(
        model, calibrator, X
    )

    scored = df.copy()
    scored["PD_Raw"] = raw_all
    scored["PD"] = cal_all

    scored["Risk_Tier"] = pd.cut(
        scored["PD"],
        bins=[0, 0.05, 0.15, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)

    # Expected loss calculation.
    if "loan_amnt" in scored.columns:
        scored["EAD"] = pd.to_numeric(
            scored["loan_amnt"], errors="coerce"
        )
        scored["LGD"] = LGD_ASSUMPTION
        scored["Expected_Loss"] = (
            scored["PD"]
            * scored["LGD"]
            * scored["EAD"]
        )
    else:
        print(
            "[WARN] loan_amnt unavailable; "
            "expected loss omitted."
        )

    test_scored = scored.loc[X_test.index].copy()

    bins = pd.concat([
        calibration_bins(y_test, test_raw, "raw"),
        calibration_bins(y_test, test_cal, "calibrated"),
    ], ignore_index=True)

    # Output filenames are separate from V1.
    scored_path = folder / "credit_scored_v2.csv"
    test_path = folder / "credit_test_scored_v2.csv"
    bins_path = folder / "credit_calibration_bins_v2.csv"
    model_path = folder / "credit_risk_pipeline_v2.pkl"
    metrics_path = folder / "credit_metrics_v2.json"
    plot_path = folder / "credit_calibration_plot_v2.png"

    scored.to_csv(scored_path, index=False)
    test_scored.to_csv(test_path, index=False)
    bins.to_csv(bins_path, index=False)

    # Save trained classifier and calibrator.
    joblib.dump({
        "base_pipeline": model,
        "calibrator": calibrator,
        "feature_columns": X.columns.tolist(),
    }, model_path)

    metrics = {
        "evaluation": (
            "stratified 60% train / "
            "20% calibration / 20% test"
        ),
        "train_size": len(X_train),
        "validation_size": len(X_val),
        "test_size": len(X_test),
        "test_default_rate": float(y_test.mean()),
        "mean_test_pd_raw": float(test_raw.mean()),
        "mean_test_pd_calibrated": float(test_cal.mean()),
        "baseline_brier_score": baseline_brier,
        "before_calibration": before,
        "after_calibration": after,
        "lgd_assumption": LGD_ASSUMPTION,
        "notes": [
            "Base model uses balanced class weights.",
            "Sigmoid calibration uses validation data only.",
            "Full-data scores contain training records.",
            "45% LGD is an assumed demonstration value.",
            "Previously examined test results make "
            "cross-experiment comparisons exploratory.",
        ],
    }

    with open(
        metrics_path, "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    # Create reliability / calibration plot.
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    for name, subset in bins.groupby(
        "score_type", sort=False
    ):
        ax.plot(
            subset["mean_predicted_pd"],
            subset["observed_default_rate"],
            marker="o",
            label=name.title(),
        )

    ax.set(
        xlabel="Mean predicted default probability",
        ylabel="Observed default rate",
        title="Credit Model Calibration (Held-Out Test)",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    for path in (
        scored_path,
        test_path,
        bins_path,
        model_path,
        metrics_path,
        plot_path,
    ):
        print(f"[SAVED] {path.name}")

    print(
        "[DONE] Credit Model V2 completed. "
        "Original files are unchanged."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"[ERROR] {type(exc).__name__}: {exc}"
        )
        sys.exit(1)
