"""Credit Risk & Fraud Monitoring dashboard — calibrated credit model edition.

Place this script inside Desktop/Credit_Risk_Project, rename it streamlit_app.py,
then run: /opt/anaconda3/bin/python -m streamlit run streamlit_app.py

Required: credit_scored_v2.csv, credit_metrics_v2.json
Fraud: fraud_test_scored_v2.csv (preferred), or fraud_scored_v2.csv / fraud_scored.csv
Optional: fraud_metrics_v2.json, fraud_logistic_metrics.json, fraud_metrics.json,
          credit_calibration_plot_v2.png, credit_calibration_bins_v2.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Credit & Fraud Risk Monitoring", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def csv_file(filename: str) -> pd.DataFrame:
    path = ROOT / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {filename} in {ROOT}")
    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner=False)
def json_file(filename: str) -> dict:
    path = ROOT / filename
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def metric_value(value, fmt=".4f") -> str:
    try:
        return format(float(value), fmt)
    except (ValueError, TypeError):
        return "N/A"


def choices(df: pd.DataFrame, col: str) -> list[str]:
    if col not in df:
        return []
    return sorted(df[col].dropna().astype(str).unique().tolist())


def confusion_matrix_view(matrix, heading: str) -> None:
    st.markdown(f"**{heading}**")
    if not isinstance(matrix, list) or len(matrix) != 2 or any(
        not isinstance(row, list) or len(row) != 2 for row in matrix
    ):
        st.info("No confusion matrix found in the selected metrics file.")
        return
    table = pd.DataFrame(
        matrix,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )
    st.dataframe(table, use_container_width=True)
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    st.caption(f"Class 1 precision: {precision:.2%} · Recall: {recall:.2%} · TP: {tp:,} · FN: {fn:,}")


def side_by_side_bars(table: pd.DataFrame, ylabel: str, title: str, percent=False):
    """True grouped bars (not stacked), with lightweight matplotlib dependency."""
    if table.empty:
        st.info("No records available for this chart.")
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    groups = list(table.index.astype(str))
    x = np.arange(len(groups))
    columns = list(table.columns)
    width = 0.76 / max(len(columns), 1)
    for i, col in enumerate(columns):
        offset = (i - (len(columns) - 1) / 2) * width
        ax.bar(x + offset, table[col].fillna(0), width=width, label=str(col))
    ax.set_xticks(x, groups, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent:
        from matplotlib.ticker import PercentFormatter
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# Required calibrated-credit files; preserve older files untouched.
st.title("Credit Risk & Fraud Monitoring")
st.caption("Calibrated credit probabilities · Fraud model validation · Runs locally on macOS")
try:
    credit = csv_file("credit_scored_v2.csv")
except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
    st.error(str(exc))
    st.info("First run `/opt/anaconda3/bin/python credit_model_v2.py` to generate credit_scored_v2.csv.")
    st.stop()

credit_metrics = json_file("credit_metrics_v2.json")
fraud_v2_metrics = json_file("fraud_metrics_v2.json")
fraud_lr_metrics = json_file("fraud_logistic_metrics.json")
fraud_v1_metrics = json_file("fraud_metrics.json")

# Prefer genuinely held-out fraud scores over full-data scores.
fraud_sources = {
    "RF V2 — held-out test (recommended)": "fraud_test_scored_v2.csv",
    "RF V2 — full scored (includes training)": "fraud_scored_v2.csv",
    "Original RF V1 — full scored (includes training)": "fraud_scored.csv",
    "Logistic — held-out test": "fraud_logistic_test_scored.csv",
}
fraud_sources = {label: filename for label, filename in fraud_sources.items() if (ROOT / filename).exists()}
if not fraud_sources:
    st.error("Fraud data unavailable. Run fraud_model_v2.py or fraud_model.py first.")
    st.stop()

st.sidebar.header("Portfolio filters")
fraud_source_label = st.sidebar.selectbox("Fraud dataset", list(fraud_sources))
fraud = csv_file(fraud_sources[fraud_source_label])

risk_levels = choices(credit, "Risk_Tier")
selected_risks = st.sidebar.multiselect("Credit risk tier", risk_levels, default=risk_levels)
credit_view = credit[credit["Risk_Tier"].astype(str).isin(selected_risks)].copy() if risk_levels else credit.copy()

grade_col = next((name for name in ("loan_grade", "CreditGrade", "grade") if name in credit.columns), None)
if grade_col:
    grades = choices(credit_view, grade_col)
    selected_grades = st.sidebar.multiselect("Loan grade", grades, default=grades)
    credit_view = credit_view[credit_view[grade_col].astype(str).isin(selected_grades)].copy()

fraud_group = next((name for name in ("TransactionType", "Location") if name in fraud.columns), None)
if fraud_group:
    groups = choices(fraud, fraud_group)
    selected_groups = st.sidebar.multiselect(f"Fraud: {fraud_group}", groups, default=groups)
    fraud_view = fraud[fraud[fraud_group].astype(str).isin(selected_groups)].copy()
else:
    fraud_view = fraud.copy()

st.sidebar.caption("Filters affect the displayed datasets, not the saved held-out model metrics.")
if credit_view.empty or fraud_view.empty:
    st.warning("One of your filters selected zero records. Restore at least one value in each filter.")
    st.stop()

st.info(
    "Credit portfolio scores include training records; the calibrated model was evaluated on a held-out test split. "
    "Choose the RF V2 held-out fraud dataset for the most defensible fraud monitoring view. "
    "The credit test set has been examined during development; independent external validation remains necessary."
)

overview, credit_tab, fraud_tab, validation_tab, explorer_tab = st.tabs([
    "Executive Overview", "Credit Risk", "Fraud Monitoring", "Model Validation", "Data Explorer"
])

with overview:
    st.subheader("Credit portfolio snapshot — calibrated scores")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Loans", f"{len(credit_view):,}")
    c2.metric("Observed default rate", metric_value(numeric(credit_view, "loan_status").mean(), ".2%"))
    c3.metric("Mean calibrated PD", metric_value(numeric(credit_view, "PD").mean(), ".2%"))
    c4.metric("Expected loss (dataset units)", metric_value(numeric(credit_view, "Expected_Loss").sum(), ",.0f"))
    if "PD_Raw" in credit_view:
        st.caption(f"Mean raw PD: {numeric(credit_view, 'PD_Raw').mean():.2%}  →  "
                   f"mean calibrated PD: {numeric(credit_view, 'PD').mean():.2%}. "
                   "These portfolio means include training records.")

    st.subheader(f"Transaction snapshot — {fraud_source_label}")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Transactions", f"{len(fraud_view):,}")
    f2.metric("Observed fraud rate", metric_value(numeric(fraud_view, "IsFraud").mean(), ".2%"))
    f3.metric("Fraud alert rate", metric_value(numeric(fraud_view, "Fraud_Flag").mean(), ".2%"))
    f4.metric("Mean model fraud score", metric_value(numeric(fraud_view, "Fraud_Probability").mean(), ".2%"))
    st.caption("Expected loss uses an assumed 45% LGD and available loan exposure; no currency is assumed. "
               "Fraud model scores are not proven to be calibrated probabilities.")

    raw = credit_metrics.get("before_calibration", {})
    cal = credit_metrics.get("after_calibration", {})
    st.subheader("Held-out evaluation highlights")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Credit ROC-AUC (calibrated)", metric_value(cal.get("roc_auc")))
    k2.metric("Credit Brier (calibrated)", metric_value(cal.get("brier_score")))
    k3.metric("Fraud RF V2 ROC-AUC", metric_value(fraud_v2_metrics.get("roc_auc")))
    k4.metric("Fraud RF V2 average precision", metric_value(fraud_v2_metrics.get("pr_auc")))

with credit_tab:
    st.subheader("Credit model — calibrated probability of default")
    col_a, col_b = st.columns(2)
    with col_a:
        if "Risk_Tier" in credit_view:
            tiers = credit_view["Risk_Tier"].astype(str).value_counts().reindex(
                ["Low", "Medium", "High"], fill_value=0
            )
            st.markdown("**Loans by calibrated risk tier**")
            st.bar_chart(tiers.rename("Loans"))
            st.caption("Illustrative cutoffs: Low ≤5%, Medium >5–15%, High >15% PD.")
    with col_b:
        score = numeric(credit_view, "PD").dropna().clip(0, 1)
        if not score.empty:
            values, edges = np.histogram(score, bins=np.linspace(0, 1, 21))
            hist = pd.DataFrame({
                "PD interval": [f"{edges[i]:.0%}–{edges[i+1]:.0%}" for i in range(len(values))],
                "Loans": values,
            }).set_index("PD interval")
            st.markdown("**Distribution of calibrated PD**")
            st.bar_chart(hist)

    if grade_col:
        by_grade = credit_view.assign(_default=numeric(credit_view, "loan_status")).groupby(
            grade_col, dropna=False
        ).agg(Loans=(grade_col, "size"), Default_Rate=("_default", "mean"))
        st.markdown("**Observed default rate by loan grade**")
        st.bar_chart(by_grade["Default_Rate"])
        st.dataframe(by_grade.rename(columns={"Default_Rate": "Observed default rate"}), use_container_width=True)
        st.caption("Interpret grade-specific rates alongside the number of loans in each grade.")

    if {"PD", "loan_status", "Risk_Tier"}.issubset(credit_view.columns):
        st.subheader("Expected loss vs default-based loss proxy")
        temp = credit_view.copy()
        ead = numeric(temp, "EAD") if "EAD" in temp else numeric(temp, "loan_amnt")
        if ead.notna().any():
            lgd = numeric(temp, "LGD").fillna(0.45) if "LGD" in temp else 0.45
            temp["Expected loss"] = numeric(temp, "Expected_Loss").fillna(numeric(temp, "PD") * lgd * ead)
            temp["Default-based loss proxy"] = numeric(temp, "loan_status") * lgd * ead
            comparison = temp.groupby("Risk_Tier")[["Expected loss", "Default-based loss proxy"]].sum()
            comparison = comparison.reindex([t for t in ("Low", "Medium", "High") if t in comparison.index])
            side_by_side_bars(comparison, "Dataset amount units", "Loss comparison by model risk tier")
            st.caption("Both amounts use an assumed LGD. The default-based proxy is NOT an observed realized loss.")

with fraud_tab:
    st.subheader(f"Fraud detection — {fraud_source_label}")
    if {"IsFraud", "Fraud_Flag"}.issubset(fraud_view.columns):
        actual = numeric(fraud_view, "IsFraud").fillna(0).astype(int)
        flag = numeric(fraud_view, "Fraud_Flag").fillna(0).astype(int)
        tp = int(((actual == 1) & (flag == 1)).sum())
        fp = int(((actual == 0) & (flag == 1)).sum())
        fn = int(((actual == 1) & (flag == 0)).sum())
        a, b, c = st.columns(3)
        a.metric("Detected frauds", f"{tp:,}")
        b.metric("False alerts", f"{fp:,}")
        c.metric("Missed frauds", f"{fn:,}")
        st.caption("These figures apply to the dataset and filters selected in the sidebar. "
                   "Only held-out datasets should be used to assess generalization.")
    if fraud_group and {"IsFraud", "Fraud_Flag"}.issubset(fraud_view.columns):
        temp = fraud_view.assign(actual=numeric(fraud_view, "IsFraud"), flagged=numeric(fraud_view, "Fraud_Flag"))
        rates = temp.groupby(fraud_group)[["actual", "flagged"]].mean()
        rates.columns = ["Observed fraud rate", "Alert rate"]
        side_by_side_bars(rates.head(15), "Share of transactions", f"Rates by {fraud_group}", percent=True)
    if "Fraud_Probability" in fraud_view:
        vals = numeric(fraud_view, "Fraud_Probability").dropna().clip(0, 1)
        if not vals.empty:
            counts, edges = np.histogram(vals, bins=np.linspace(0, 1, 21))
            st.markdown("**Fraud score distribution**")
            st.bar_chart(pd.DataFrame({
                "Score interval": [f"{edges[i]:.0%}–{edges[i+1]:.0%}" for i in range(20)],
                "Transactions": counts,
            }).set_index("Score interval"))
    st.warning("Fraud RF V2 detected only 1 of 203 actual frauds on its chronological test set. "
               "This model is not suitable for operational decisions.")

with validation_tab:
    st.subheader("Credit model — held-out test calibration")
    before = credit_metrics.get("before_calibration", {})
    after = credit_metrics.get("after_calibration", {})
    metrics_table = pd.DataFrame({
        "Before calibration": {
            "ROC-AUC": before.get("roc_auc"),
            "Average precision": before.get("average_precision"),
            "Brier score ↓": before.get("brier_score"),
            "Log loss ↓": before.get("log_loss"),
        },
        "After calibration": {
            "ROC-AUC": after.get("roc_auc"),
            "Average precision": after.get("average_precision"),
            "Brier score ↓": after.get("brier_score"),
            "Log loss ↓": after.get("log_loss"),
        },
    })
    st.dataframe(metrics_table.style.format("{:.4f}", na_rep="N/A"), use_container_width=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Observed test default rate", metric_value(credit_metrics.get("test_default_rate"), ".2%"))
    m2.metric("Raw test mean PD", metric_value(credit_metrics.get("mean_test_pd_raw"), ".2%"))
    m3.metric("Calibrated test mean PD", metric_value(credit_metrics.get("mean_test_pd_calibrated"), ".2%"))
    plot_file = ROOT / "credit_calibration_plot_v2.png"
    if plot_file.exists():
        st.image(str(plot_file), caption="Held-out credit reliability diagram; dashed line shows ideal calibration.")
    else:
        st.info("Calibration graph missing. Run credit_model_v2.py to create it.")
    confusion_matrix_view(after.get("confusion_matrix_at_0_5"), "Calibrated credit test confusion matrix (threshold 0.50)")

    st.subheader("Fraud models — chronological held-out results")
    rows = []
    for name, info in (("Random Forest V2", fraud_v2_metrics), ("Logistic Regression", fraud_lr_metrics)):
        if info:
            cm = info.get("confusion_matrix", [[0, 0], [0, 0]])
            tp = cm[1][1] if len(cm) == 2 and len(cm[1]) == 2 else None
            rows.append({"Model": name, "ROC-AUC": info.get("roc_auc"),
                         "Average precision": info.get("pr_auc"), "True positives": tp,
                         "Test alert rate": info.get("test_alert_rate_observed")})
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
    else:
        st.info("Fraud V2 metrics not found. Run fraud_model_v2.py and/or fraud_logistic_baseline.py.")
    if fraud_v2_metrics:
        confusion_matrix_view(fraud_v2_metrics.get("confusion_matrix"), "Random Forest V2 test confusion matrix")
    with st.expander("Original Random Forest V1 (not directly comparable)"):
        if fraud_v1_metrics:
            st.write({"ROC-AUC": fraud_v1_metrics.get("roc_auc"),
                      "Average precision": fraud_v1_metrics.get("pr_auc")})
            st.caption("V1 used a random test split and selected the alert threshold on test scores. "
                       "These results are not an equivalent comparison with V2.")
        else:
            st.info("No original fraud_metrics.json file found.")
    st.markdown("**Model governance notes**")
    st.markdown(
        "- Credit sigmoid calibration improved held-out Brier and log loss; good average agreement does not guarantee perfection in every subgroup.\n"
        "- Calibrated credit scores are still from a demonstration dataset; risk tiers and 45% LGD are illustrative.\n"
        "- Fraud RF V2 and Logistic Regression perform near random ranking on held-out data.\n"
        "- The credit test set has been reviewed during development. New untouched data is needed for a final external validation.\n"
        "- Full scored datasets contain training observations, so full-dataset detection statistics are descriptive."
    )

with explorer_tab:
    st.subheader("Data explorer")
    dataset = st.radio("Select dataset", ["Calibrated credit", "Selected fraud dataset"], horizontal=True)
    displayed = credit_view if dataset == "Calibrated credit" else fraud_view
    st.caption(f"{len(displayed):,} filtered rows · {len(displayed.columns)} columns")
    st.dataframe(displayed.head(500), use_container_width=True)
    st.download_button(
        "Download filtered CSV",
        data=displayed.to_csv(index=False).encode("utf-8"),
        file_name="calibrated_credit_filtered.csv" if dataset == "Calibrated credit" else "fraud_filtered.csv",
        mime="text/csv",
    )
    st.caption("Check the data and remove sensitive records before publishing or sharing exports.")

st.divider()
st.caption("Educational project · Source: local Python model outputs · Not for banking or fraud operational use")
