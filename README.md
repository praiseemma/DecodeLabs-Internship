# DecodeLabs Data Science Internship

![dashboard tests](https://github.com/praiseemma/DecodeLabs-Internship/actions/workflows/dashboard-tests.yml/badge.svg)

Two projects from the DecodeLabs Data Science batch (November 2026): a data-preparation
project, then an end-to-end fraud detection pipeline shipped as a live Streamlit app.

**Live app:** <https://decodelabs-internship-nqrofsyqqhd38nnzlf6ad2.streamlit.app/> · **Author:** Praise Emma · **Instructor:** Vishal Rajpoot

| | Project 1 | Project 2 |
|---|---|---|
| Question | Can a raw order table be made machine-learning-ready, with every decision justified? | Can fraud be found when 99.83% of rows are legitimate? |
| Data | 1,200 e-commerce orders, 14 columns | 284,807 card transactions, 473 frauds (0.17%) |
| Result | Data cleaned, 5 features engineered, every choice logged | Random Forest: precision 0.60, recall 0.81, ROC-AUC 0.97 |
| Folder | [`Project-1-praiseemma/`](Project-1-praiseemma) | [`Project-2-praiseemma/`](Project-2-praiseemma) |

## Project 1 — Advanced EDA and feature engineering

`CouponCode` is missing in 25.75% of orders, and a blank appears to mean "no coupon used",
so it is kept as its own category instead of being imputed. Mean, median and KNN imputation
are still demonstrated and scored on a separate copy of the data. The IQR rule flags 8
`TotalPrice` outliers, which are capped rather than deleted; five features are engineered
with vectorised operations, and two features correlated above 0.80 are dropped. The exported
files are checked with assertions, not just written out.

Full write-up, assumptions and outputs: [`Project-1-praiseemma/README.md`](Project-1-praiseemma/README.md).

## Project 2 — Fraud detection pipeline

The brief asks for SMOTE, two algorithms, and precision/recall/ROC-AUC instead of accuracy.
Accuracy is exactly why: a model that labels every transaction legitimate scores **99.83%**
and catches **zero** fraud. This project measures what that metric hides.

- 1,081 duplicate rows removed, then a stratified 80/20 split: 56,746 test rows, 95 real frauds.
- SMOTE lives inside the scikit-learn pipeline, so it is fitted on training folds only and never
  touches validation or test data.
- Logistic Regression and Random Forest tuned with 5-fold stratified cross-validation, then
  evaluated once on the held-out set.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Baseline (always legitimate) | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.053 | 0.874 | 0.101 | 0.962 |
| **Random Forest** | **0.602** | **0.811** | **0.691** | **0.973** |

The Random Forest catches **77 of 95** test frauds and raises **51 false alarms**. Logistic
Regression catches more fraud but buries it in noise: **19 of every 20** alerts are legitimate.

Scores are also checked for calibration rather than trusted. SMOTE balances the training set,
so the forest reports a much higher fraud rate than reality (mean score 2.5% against 0.17%);
the ordering is sound, the level is not, and the app says so instead of printing a score as a
probability. Moving the cut-off from 0.50 to 0.40 buys 1.3 points of recall for 16 points of
precision — a business decision, which is why it is a slider and not a constant.

Notebook, change log and the comparison table: [`Project-2-praiseemma/`](Project-2-praiseemma).
App, model card and page-by-page detail: [`Project-2-praiseemma/dashboard/README.md`](Project-2-praiseemma/dashboard/README.md).

## Repository layout

```
.
├── Project-1-praiseemma/     notebook, change log, cleaned and engineered data
├── Project-2-praiseemma/
│   ├── project2_fraud_detection_pipeline.ipynb
│   ├── Project2_Change_Log.pdf
│   ├── outputs/model_comparison.csv
│   └── dashboard/            Streamlit app, exported model, 82 tests
├── .streamlit/config.toml    light theme, read from the folder Streamlit is launched in
└── .github/workflows/        dashboard tests on every push
```

## Run the dashboard

From the **repository root**, so Streamlit picks up `.streamlit/config.toml`:

```powershell
python -m pip install -r Project-2-praiseemma/dashboard/requirements.txt
python -m streamlit run Project-2-praiseemma/dashboard/app.py
```

Then open <http://localhost:8501>. The app reads the exported model and the 2,473-row demo
sample committed in `dashboard/artifacts/` and `dashboard/data/`, so it runs from a fresh
clone without the dataset. The 150 MB raw CSV and the split files are gitignored.

Check the app with `cd Project-2-praiseemma/dashboard; python -m pytest -q` — 82 tests, and
the same suite runs in CI.
