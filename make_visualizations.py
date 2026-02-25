import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


PROJECT_DIR = Path(r"C:\Users\Dell\Desktop\CreditFraudRisk")
VIZ_DIR = PROJECT_DIR / "visualizations"
CREDIT_SCORED = PROJECT_DIR / "credit_scored.csv"
FRAUD_SCORED = PROJECT_DIR / "fraud_scored.csv"


def ensure_viz_folder():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(filename: str):
    out = VIZ_DIR / filename
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"[SAVED] {out}")


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"File is empty: {path}")
    return df


def pick_amount_column(df: pd.DataFrame):
    for c in ["Amount", "TransactionAmount", "TxnAmount", "amt", "amount"]:
        if c in df.columns:
            return c
    return None


# -----------------------------
# CREDIT VISUALIZATIONS
# -----------------------------
def credit_visuals(df: pd.DataFrame):
    required = {"PD", "loan_status"}
    missing = required - set(df.columns)
    if missing:
        print(f"[SKIP] Credit visuals missing columns: {missing}")
        return

    # (1) PD Distribution (hist)
    plt.figure(figsize=(8, 5))
    plt.hist(df["PD"].dropna(), bins=40)
    plt.title("Credit Risk: PD Distribution")
    plt.xlabel("Probability of Default (PD)")
    plt.ylabel("Count")
    save_fig("01_credit_pd_distribution.png")

    # (2) Credit ROC curve (line)
    y_true = df["loan_status"].astype(int).values
    y_score = df["PD"].astype(float).values
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("Credit Model ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    save_fig("02_credit_roc_curve.png")

    # (3) Credit PR curve (line)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.title("Credit Model Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    save_fig("03_credit_pr_curve.png")

    # (4) PD by Actual Outcome (boxplot) — not a bar chart
    # Shows separation between defaulters vs non-defaulters
    pd_default = df.loc[df["loan_status"] == 1, "PD"].dropna()
    pd_nondefault = df.loc[df["loan_status"] == 0, "PD"].dropna()

    plt.figure(figsize=(7, 5))
    plt.boxplot([pd_nondefault, pd_default], labels=["No Default (0)", "Default (1)"])
    plt.title("Credit Risk: PD by Actual Outcome (Boxplot)")
    plt.ylabel("PD")
    save_fig("04_credit_pd_boxplot_by_outcome.png")

    # (5) Expected vs Actual Loss (total + by tier if possible)
    # We will compute "Actual Loss" using LGD assumption if loan_amnt exists
    LGD = 0.45
    if "loan_amnt" in df.columns:
        ead = df["loan_amnt"].astype(float)
        # Expected Loss
        if "Expected_Loss" in df.columns:
            expected_loss = df["Expected_Loss"].astype(float)
        else:
            expected_loss = df["PD"].astype(float) * LGD * ead

        actual_loss = df["loan_status"].astype(int) * LGD * ead

        plt.figure(figsize=(6, 4))
        plt.bar(["Expected Loss", "Actual Loss"], [expected_loss.sum(), actual_loss.sum()])
        plt.title("Credit Portfolio: Expected vs Actual Loss (Total)")
        plt.ylabel("Loss Amount")
        save_fig("05_credit_expected_vs_actual_loss_total.png")

        # By risk tier (line chart) if Risk_Tier exists
        if "Risk_Tier" in df.columns:
            tmp = df.copy()
            tmp["Risk_Tier"] = tmp["Risk_Tier"].astype(str)
            tmp["_expected"] = expected_loss
            tmp["_actual"] = actual_loss
            by_tier = tmp.groupby("Risk_Tier")[["_expected", "_actual"]].sum()

            # order tiers if present
            tier_order = ["Low", "Medium", "High"]
            by_tier = by_tier.reindex([t for t in tier_order if t in by_tier.index])

            plt.figure(figsize=(7, 5))
            plt.plot(by_tier.index, by_tier["_expected"].values, marker="o", label="Expected Loss")
            plt.plot(by_tier.index, by_tier["_actual"].values, marker="o", label="Actual Loss")
            plt.title("Credit Portfolio: Expected vs Actual Loss by Risk Tier")
            plt.xlabel("Risk Tier")
            plt.ylabel("Loss Amount")
            plt.legend()
            save_fig("06_credit_expected_vs_actual_loss_by_tier.png")
    else:
        print("[SKIP] Credit loss comparisons: missing 'loan_amnt' (EAD).")


# -----------------------------
# FRAUD VISUALIZATIONS
# -----------------------------
def fraud_visuals(df: pd.DataFrame):
    required = {"Fraud_Probability", "Fraud_Flag"}
    missing = required - set(df.columns)
    if missing:
        print(f"[SKIP] Fraud visuals missing columns: {missing}")
        return

    # (7) Fraud Probability Distribution (hist)
    plt.figure(figsize=(8, 5))
    plt.hist(df["Fraud_Probability"].dropna(), bins=40)
    plt.title("Fraud Risk: Fraud Probability Distribution")
    plt.xlabel("Fraud Probability")
    plt.ylabel("Count")
    save_fig("07_fraud_probability_distribution.png")

    # Need IsFraud for ROC/PR and actual vs predicted comparisons
    if "IsFraud" in df.columns:
        y_true = df["IsFraud"].astype(int).values
        y_score = df["Fraud_Probability"].astype(float).values

        # (8) Fraud ROC curve (line)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)

        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("Fraud Model ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        save_fig("08_fraud_roc_curve.png")

        # (9) Fraud PR curve (line)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        plt.figure(figsize=(7, 5))
        plt.plot(recall, precision, label=f"AP={ap:.3f}")
        plt.title("Fraud Model Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        save_fig("09_fraud_pr_curve.png")

        # (10) Confusion Matrix Heatmap-like (matshow) — not bar
        cm = confusion_matrix(df["IsFraud"].astype(int), df["Fraud_Flag"].astype(int))
        plt.figure(figsize=(5.5, 4.5))
        plt.matshow(cm, fignum=1)
        plt.title("Fraud Model Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center")
        save_fig("10_fraud_confusion_matrix.png")

        # (11) Actual vs Predicted Fraud Rates by Segment (TransactionType preferred)
        segment_col = None
        if "TransactionType" in df.columns:
            segment_col = "TransactionType"
        elif "Location" in df.columns:
            segment_col = "Location"

        if segment_col:
            tmp = df.copy()
            tmp[segment_col] = tmp[segment_col].astype(str)

            # choose top 10 by volume for stability
            top = tmp[segment_col].value_counts().head(10).index
            actual_rate = tmp.groupby(segment_col)["IsFraud"].mean().loc[top]
            predicted_rate = tmp.groupby(segment_col)["Fraud_Flag"].mean().loc[top]

            x = np.arange(len(top))
            width = 0.4

            plt.figure(figsize=(10, 5))
            plt.bar(x - width/2, actual_rate.values, width, label="Actual Fraud Rate")
            plt.bar(x + width/2, predicted_rate.values, width, label="Predicted Alert Rate")
            plt.title(f"Fraud: Actual vs Predicted by {segment_col} (Top 10)")
            plt.xlabel(segment_col)
            plt.ylabel("Rate")
            plt.xticks(x, top, rotation=30, ha="right")
            plt.legend()
            save_fig(f"11_fraud_actual_vs_predicted_by_{segment_col.lower()}.png")
        else:
            print("[SKIP] Fraud segment comparison: no TransactionType/Location found.")
    else:
        print("[SKIP] Fraud ROC/PR/confusion/segment: missing 'IsFraud' in fraud_scored.csv")

    # (12) Executive Risk Summary View (combined KPI snapshot)
    # We will build a single view from both datasets later in main() (needs credit_df too).


def executive_risk_summary_view(credit_df: pd.DataFrame, fraud_df: pd.DataFrame):
    # Collect KPIs (simple and interview-friendly)
    summary = {}

    # Credit KPIs
    if "loan_status" in credit_df.columns:
        summary["Credit: Default Rate"] = float(credit_df["loan_status"].mean())
        summary["Credit: Total Loans"] = float(len(credit_df))

    if "PD" in credit_df.columns:
        summary["Credit: Avg PD"] = float(credit_df["PD"].mean())

    if "Expected_Loss" in credit_df.columns:
        summary["Credit: Total Expected Loss"] = float(credit_df["Expected_Loss"].sum())

    if "loan_amnt" in credit_df.columns:
        summary["Credit: Total Exposure (EAD)"] = float(credit_df["loan_amnt"].sum())

    # Fraud KPIs
    if "IsFraud" in fraud_df.columns:
        summary["Fraud: Actual Rate"] = float(fraud_df["IsFraud"].mean())
        summary["Fraud: Total Txns"] = float(len(fraud_df))

    if "Fraud_Flag" in fraud_df.columns:
        summary["Fraud: Predicted Alert Rate"] = float(fraud_df["Fraud_Flag"].mean())

    if "Fraud_Probability" in fraud_df.columns:
        summary["Fraud: Avg Probability"] = float(fraud_df["Fraud_Probability"].mean())

    amt_col = pick_amount_column(fraud_df)
    if amt_col and "IsFraud" in fraud_df.columns:
        summary["Fraud: Total Amount"] = float(fraud_df[amt_col].sum())
        summary["Fraud: Fraud Amount"] = float(fraud_df.loc[fraud_df["IsFraud"] == 1, amt_col].sum())

    # Save CSV too (useful for dashboard)
    out_csv = PROJECT_DIR / "executive_risk_summary.csv"
    pd.DataFrame([summary]).to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")

    # KPI plot (horizontal bars for readability)
    labels = list(summary.keys())
    values = [summary[k] for k in labels]

    plt.figure(figsize=(12, 6))
    y = np.arange(len(labels))
    plt.barh(y, values)
    plt.yticks(y, labels)
    plt.title("Executive Risk Summary (Credit + Fraud KPIs)")
    plt.xlabel("Value")
    save_fig("12_executive_risk_summary_view.png")


def main():
    ensure_viz_folder()

    # Load
    if not CREDIT_SCORED.exists():
        raise FileNotFoundError(f"Missing: {CREDIT_SCORED}")
    if not FRAUD_SCORED.exists():
        raise FileNotFoundError(f"Missing: {FRAUD_SCORED}")

    print(f"[INFO] Loading {CREDIT_SCORED}")
    credit_df = safe_read(CREDIT_SCORED)

    print(f"[INFO] Loading {FRAUD_SCORED}")
    fraud_df = safe_read(FRAUD_SCORED)

    # Create credit visuals (includes ROC/PR + boxplot + loss comparisons)
    credit_visuals(credit_df)

    # Create fraud visuals (includes ROC/PR + confusion matrix + segment comparison)
    fraud_visuals(fraud_df)

    # Combined executive view
    executive_risk_summary_view(credit_df, fraud_df)

    print("\n[DONE] Visualizations created in:", VIZ_DIR)


if __name__ == "__main__":
    main()