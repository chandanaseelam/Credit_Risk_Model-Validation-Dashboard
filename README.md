# Credit Risk & Fraud Monitoring Dashboard (macOS)

An educational, end-to-end Python machine learning and Streamlit project exploring **credit default risk**, **fraud detection**, **model validation**, and **probability calibration**.

> **Project origin and attribution:** This learning project was developed from the public GitHub repository [Credit Risk Model Validation & Governance Dashboard by Chandan Seelam](https://github.com/chandanaseelam/Credit_Risk_Model-Validation-Dashboard). The macOS adaptations, additional model experiments, calibration workflow, and Streamlit dashboard were developed as extensions to that starting project. See **Reuse and data permissions** before republishing any original code or data.
>
> **Educational use only.** These models are not appropriate for live lending or fraud enforcement decisions. In particular, the tested fraud models did **not** generalize reliably.

## Overview

The application provides five interactive views:

1. **Executive Overview** — loan volume, observed default rate, calibrated model score, estimated loss, and fraud monitoring KPIs.
2. **Credit Risk** — borrower risk tiers, calibrated score distribution, grade-level observed default rates, and loss comparisons.
3. **Fraud Monitoring** — transaction alerts, false alerts, missed fraud, and fraud-score patterns; a held-out fraud test set is selected by default.
4. **Model Validation** — before/after credit calibration metrics, a reliability plot, confusion matrices, and chronological fraud model comparison.
5. **Data Explorer** — filterable credit and fraud tables with CSV export.

## Technology

- **Python 3.11**, Pandas, NumPy, scikit-learn, Matplotlib, Joblib
- **Streamlit** for the local interactive dashboard
- Tested by the project developer on **macOS using Anaconda Python**

## Data

| Dataset | Records | Target |
| --- | ---: | --- |
| `credit_risk_dataset.csv` | 32,581 loans | `loan_status`: 1 = default, 0 = non-default |
| `credit_card_fraud_dataset.csv` | 100,000 transactions | `IsFraud`: 1 = fraud, 0 = legitimate |

The fraud dataset includes 1,000 labeled frauds (1%). Loan-level fields include borrower characteristics, loan attributes, and grade; transaction-level fields include amount, merchant, time, type, and location. Source datasets are required to reproduce model training, **but should be redistributed only where permissions allow**.

## Methodology

### Credit risk

- Preprocess numeric and categorical features in a scikit-learn pipeline.
- Train a **balanced Logistic Regression** model.
- Use **stratified 60% train / 20% calibration / 20% test** subsets.
- Fit a separate sigmoid (Platt-style) calibrator on validation-set model scores; evaluate raw and calibrated scores on the test set.
- Generate calibrated probability of default (**PD**) and illustrative `Low`, `Medium`, `High` risk categories (cutoffs: ≤5%, >5–15%, >15%).
- Estimate loss using `Expected Loss = PD × LGD × EAD`, with an **assumed LGD of 45%** and loan amount as an illustrative exposure proxy.

### Fraud monitoring

- Sort transactions by `TransactionDate` and use **chronological 60% train / 20% validation / 20% test** subsets.
- Exclude `TransactionID` from modeling; encode `MerchantID` as a category; derive date-based features.
- Compare **Random Forest V2** and a **Logistic Regression baseline**.
- Select a threshold aiming to alert on ~1% of **validation** transactions, then apply that fixed threshold to the later test transactions.
- Report ROC-AUC, average precision, confusion matrices, alert rate, precision and recall.

The original Random Forest V1 is retained for historical comparison, but its threshold was selected using test predictions; it is **not directly comparable** with the more carefully separated V2 experiments.

## Results

### Credit risk: held-out test, before vs after calibration

| Metric | Raw credit model | Calibrated credit model |
| --- | ---: | ---: |
| ROC-AUC | 0.8711 | 0.8711 |
| Average precision | 0.7230 | 0.7230 |
| Brier score ↓ | 0.1381 | **0.1024** |
| Log loss ↓ | 0.4359 | **0.3393** |
| Mean predicted PD | 37.98% | **21.96%** |

Observed default rate on the credit test set: **21.82%**. Lower Brier score and log loss indicate improved probability quality on this test split; matching average prevalence alone does not establish perfect calibration. The credit test set was also examined during development, so further untouched validation would be needed.

### Fraud models: chronological test period

| Metric | Random Forest V2 | Logistic Regression |
| --- | ---: | ---: |
| ROC-AUC | 0.4586 | 0.4752 |
| Average precision | 0.0091 | 0.0098 |
| Detected frauds | 1 / 203 | 1 / 203 |
| Test alert rate | 0.52% | 0.90% |

**Finding:** Both fraud models performed near or below random ranking on unseen transactions. Random Forest strongly overfit the training records (training ROC-AUC ~1.0 vs test 0.4586). Features available in this sample data exhibited little *demonstrated* stable fraud signal. Do not treat their fraud scores as calibrated probabilities or use these models for operational decisions.

## Project files

| File | Description |
| --- | --- |
| `credit_model_v2.py` | Credit Logistic Regression, probability calibration, metrics, and score generation |
| `fraud_model_v2.py` | Chronological Random Forest fraud evaluation |
| `fraud_logistic_baseline.py` | Chronological Logistic Regression fraud baseline |
| `streamlit_app.py` | Five-page Streamlit dashboard (V2 code) |
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Excludes virtual environments, model artifacts, and scored dataset outputs |
| `credit_risk_dataset.csv` | Input credit data (obtain and use with appropriate permission) |
| `credit_card_fraud_dataset.csv` | Input fraud data (obtain and use with appropriate permission) |

**Generated by the scripts** (normally not committed): `credit_scored_v2.csv`, `fraud_test_scored_v2.csv`, other `*_scored*.csv`, and `*.pkl`. Metrics JSON and `credit_calibration_plot_v2.png` are also generated.

Original example scripts (`credit_model.py`, `fraud_model.py`, `make_visualizations.py`) and the `.pbix` Power BI file are part of the starting project; use or redistribute them only in accordance with the original author's permissions.

## Run on a Mac

Open Terminal and navigate to the project folder:

```bash
cd ~/Desktop/Credit_Risk_Project
```

Create and activate an environment if desired:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If you already have a functioning Anaconda installation with these libraries, use that interpreter consistently rather than mixing Python environments. For the setup used during this project, scripts were run with `/opt/anaconda3/bin/python`.

Place both input CSV datasets in the same directory as the Python scripts. Then execute **in order**:

```bash
python credit_model_v2.py
python fraud_model_v2.py
python fraud_logistic_baseline.py
python -m streamlit run streamlit_app.py
```

If running with Anaconda without activating `.venv`, replace `python` with `/opt/anaconda3/bin/python` in all four commands.

Open the local URL displayed by Streamlit (commonly `http://localhost:8501`; it can use another port if that one is occupied). To stop Streamlit, press **Control+C** in the Terminal.

**Note:** The dashboard requires its generated V2 files to be present. If they have been excluded from Git, run the model scripts first.

## Reading the dashboard responsibly

- **Full-portfolio credit scores contain training records.** They describe the portfolio; evaluate generalization using held-out test metrics rather than recomputing ROC-AUC on all scored rows.
- **Calibration:** sigmoid calibration improved the measured test Brier score, but independent validation and monitoring remain necessary.
- **Expected loss:** the 45% LGD and exposure proxy are assumptions, not documented recoveries or verified realized bank losses. Currency is intentionally unspecified.
- **Fraud:** the held-out test set is the most defensible available view, but fraud models had inadequate discrimination and should not drive real-world decisions.
- **Data provenance:** confirm origins, rights, confidentiality and suitability before publishing any dataset or using it for training.

## Reuse and data permissions

The starting repository is publicly viewable, but **public visibility does not automatically grant permission to republish source code, datasets, images or the Power BI file**. Check the source repository's current license and data provenance, and seek permission where required. This README does not grant rights to those original materials. The attribution above should remain visible in any adapted publication.

## Next development ideas

- Test probability calibration on a new, independent credit dataset or later time period.
- Add confidence intervals, drift diagnostics and clear population definitions to validation reports.
- Obtain a fraud dataset with meaningful predictive features and validate with an untouched future period.
- Add automated tests and a Streamlit deployment guide after resolving data redistribution and licensing.
