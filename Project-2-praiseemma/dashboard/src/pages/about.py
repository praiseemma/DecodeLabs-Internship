"""About and limits: the model card, and what this thing must not be used for."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import model, state, ui

results = state.results()
info = results["model"]
split = results["split"]
data = results["data"]
rf = results["metrics"]["Random Forest"]

ui.header(
    "About this model and its limits",
    "A model card: what it is, what it was trained on, how it performs, and where it stops "
    "being trustworthy.",
)

st.subheader("Model card")
card = pd.DataFrame([
    ("Purpose", "Flag card transactions that look fraudulent, for a demonstration of an "
                "imbalanced classification pipeline."),
    ("Model", f"Random Forest classifier ({info['n_estimators']} trees, "
              f"max_depth={info['params']['classifier__max_depth']}) behind SMOTE in an "
              "imbalanced-learn Pipeline."),
    ("Inputs", "30 numbers: `Time`, `V1` to `V28`, `Amount`. The V columns are anonymised "
               "PCA components of the original, private features."),
    ("Output", "A fraud probability between 0 and 1, turned into a decision by a threshold "
               f"the user sets (default {info['default_threshold']:.2f})."),
    ("Training data",
     f"Kaggle \"Credit Card Fraud Detection\" (ULB): {data['raw_rows']:,} transactions, "
     f"{data['duplicate_rows']:,} exact duplicates removed, leaving {data['rows']:,} rows "
     f"and {data['frauds']} frauds."),
    ("Training split", f"{split['train_rows']:,} rows for training ({split['train_frauds']} "
                       f"frauds), {split['test_rows']:,} rows held back ({split['test_frauds']} "
                       "frauds), stratified, seed 42."),
    ("Hyperparameters", "Chosen by 5-fold stratified GridSearchCV scored on ROC-AUC, on a "
                        f"stratified {100_000:,}-row sample of the training set. Best: "
                        "`smote__k_neighbors=5`, `classifier__max_depth=10`."),
    ("Test performance",
     f"Precision {rf['precision']:.4f}, recall {rf['recall']:.4f}, F1 {rf['f1']:.4f}, "
     f"ROC-AUC {rf['roc_auc']:.4f}, average precision {rf['avg_precision']:.4f}, at a "
     "threshold of 0.50."),
    ("Intended use", "Teaching, portfolio and review: showing a leak-free pipeline, the "
                     "cost of imbalance, and the threshold trade-off."),
    ("Not intended for", "Real fraud screening, credit decisions, or any decision about a "
                         "person. It has never seen live data and has no monitoring."),
], columns=["Aspect", "Detail"])
st.dataframe(card, width="stretch", hide_index=True)

st.divider()

st.subheader("Honest limits")
st.markdown(
    f"- **Anonymised inputs.** `V1` to `V28` are PCA components. The model cannot take raw "
    "card data, merchant names or a customer's history, so it cannot be dropped into a real "
    "payment system as it stands.\n"
    f"- **It is a demonstration.** This is an academic project on a public dataset, not a "
    "fraud product. There is no retraining schedule, no drift monitoring, no case management "
    "and no review of false positives.\n"
    f"- **Only {data['frauds']} fraud cases exist.** {split['test_frauds']} of them are in the "
    f"test set, so each one is worth about {100 / split['test_frauds']:.1f} percentage points "
    "of recall. Differences of a couple of points between models are noise.\n"
    f"- **The forest was tuned on a sample.** The search ran on a stratified 100,000-row "
    f"sample of the training set for time, then the winner was retrained on all "
    f"{split['train_rows']:,} training rows. A different sample could have picked a slightly "
    "different depth.\n"
    f"- **The threshold is a business choice.** 0.50 was kept because lowering it bought "
    "almost no recall for a large loss of precision. A bank with different costs might "
    "choose differently, and that choice should be revisited with real cost figures.\n"
    "- **The score is not a probability.** The forest was trained on a "
    "SMOTE-balanced set, where half the examples were fraud, so its output is a "
    "ranking, not a percentage chance. Averaged over the test set it reports "
    "2.5% fraud when the truth is 0.17%. \"0.80\" means \"more fraud-like than "
    "most\", not \"80% chance\". The table below shows the measured rates."
)

st.subheader("What a score means, measured")
arrays = state.predictions()
score_table = pd.DataFrame(
    model.calibration_table(arrays["test_proba_rf"], arrays["y_test"])
)
for column in ("Mean score", "Actual fraud rate"):
    score_table[column] = score_table[column] * 100
st.dataframe(
    score_table, width="stretch", hide_index=True,
    column_config={
        "Rows": st.column_config.NumberColumn("Rows", format="%d"),
        "Frauds": st.column_config.NumberColumn("Frauds", format="%d"),
        "Mean score": st.column_config.NumberColumn("Mean score", format="%.1f%%"),
        "Actual fraud rate": st.column_config.NumberColumn(
            "Actual fraud rate", format="%.1f%%"
        ),
    },
)
st.caption(
    "The model's **mean score** is its own opinion, averaged over that band; the "
    "**actual fraud rate** is what the true labels say happened. A calibrated "
    "model would have those two columns agree. This one ranks well but runs high, "
    "because SMOTE trained it on a 50/50 world instead of the real 0.17%. The "
    "0.50–0.90 and 0.90+ bands hold only 53 and 75 rows, so their ranges are wide. "
    "At the deployed cut-off, 128 test transactions were flagged and 77 were real "
    "fraud: about 60% of alerts are genuine (95% range 52% to 68%). That figure, "
    "not the raw score, is the one to quote."
)

st.divider()

st.subheader("Environment")
versions = pd.DataFrame(model.version_report(results["versions"]))
st.dataframe(versions, width="stretch", hide_index=True)
if (versions["Match"] == "no").any():
    st.warning(
        "The running versions differ from the ones the model was saved with. Re-run "
        "`scripts/export_model.py` in this environment, or install "
        "`requirements.txt`, before trusting the scores."
    )
else:
    st.success("The model is loaded under the exact library versions it was saved with.")

st.divider()

st.subheader("Reproducing this")
st.markdown(
    "From the `Project-2-praiseemma` folder, in PowerShell:\n\n"
    "```powershell\n"
    "# 1. install the pinned versions (once)\n"
    "& \"$env:USERPROFILE\\anaconda3\\python.exe\" -m pip install -r dashboard\\requirements.txt\n\n"
    "# 2. rebuild the model and the numbers this app shows (needs data/creditcard.csv)\n"
    "& \"$env:USERPROFILE\\anaconda3\\python.exe\" dashboard\\scripts\\export_model.py\n\n"
    "# 3. start the app\n"
    "& \"$env:USERPROFILE\\anaconda3\\python.exe\" -m streamlit run dashboard\\app.py\n"
    "```\n\n"
    "The export script ends with a PASS/FAIL check against the notebook's scores "
    "(precision 0.6016, recall 0.8105, 77 caught, 18 missed, 51 false alarms). If it fails, "
    "the app shows the failure rather than hiding it."
)

st.subheader("Data attribution")
st.markdown(
    "Dataset: Machine Learning Group, Université Libre de Bruxelles (ULB), "
    "\"Credit Card Fraud Detection\", published on Kaggle for research on highly imbalanced "
    "classification. The 150 MB file is deliberately not stored in this repository; the app "
    "ships a 2,473-row sample instead, described plainly as a sample."
)

ui.provenance(results)
