"""Overview: what this is, how imbalanced the data is, and how to read the numbers."""

from __future__ import annotations

import streamlit as st

from src import config, plots, state, ui

results = state.results()
eda = results["eda"]
rf = results["metrics"]["Random Forest"]
balance = eda["class_balance"]

ui.header(
    "Catching fraudulent card transactions",
    "A Random Forest trained on 283,726 anonymised card transactions, deployed as an "
    "app you can poke at. DecodeLabs Data Science, Project 2.",
)

ui.metric_row([
    {"label": "Transactions", "value": f"{balance['legitimate'] + balance['fraud']:,}",
     "help": "Rows left after removing 1,081 exact duplicates."},
    {"label": "Frauds", "value": f"{balance['fraud']:,}",
     "help": f"{balance['fraud_share']:.2f}% of all rows. This is the whole problem."},
    {"label": "Fraud caught", "value": f"{rf['recall']:.1%}",
     "help": "Recall: of the 95 real frauds held back for testing, 77 were caught."},
    {"label": "Alerts that were right", "value": f"{rf['precision']:.1%}",
     "help": "Precision: of the 128 transactions it flagged, 77 were really fraud."},
])

left, right = st.columns([1.05, 1], gap="large")

with left:
    st.subheader("The imbalance is the project")
    st.markdown(
        f"Only **{balance['fraud']:,} of {balance['legitimate'] + balance['fraud']:,}** "
        f"transactions are fraud ({balance['fraud_share']:.2f}%). A model that says "
        "\"legitimate\" to everything is right 99.83% of the time and catches nothing. "
        "That is why accuracy is not used here, and why the model is judged on "
        "**precision** (when it says fraud, how often is it right?), **recall** "
        "(how much of the fraud does it find?) and **ROC-AUC** (does it rank fraud "
        "above legitimate?)."
    )
    st.plotly_chart(plots.class_balance(eda), width="stretch",
                    config=config.PLOTLY_CONFIG)

with right:
    st.subheader("How to read this app")
    st.markdown(
        "1. **Score a transaction** - pick a known fraud or legitimate example, or type "
        "your own values, and see the probability and the decision.\n"
        "2. **Batch scoring** - upload a CSV of transactions and get a scored file back.\n"
        "3. **Threshold explorer** - move the cut-off and watch precision and recall "
        "trade places.\n"
        "4. **Model performance** - the full comparison, ROC and precision-recall curves, "
        "confusion matrices, feature importances.\n"
        "5. **Data insights** - what makes fraud look different in this dataset.\n"
        "6. **About and limits** - what this model must not be used for."
    )
    if results["reproduction"]["all_passed"]:
        st.success(
            "This app is serving a model that reproduces the notebook exactly: "
            f"precision {rf['precision']:.4f}, recall {rf['recall']:.4f} on the test set."
        )
    else:
        st.error("The exported model did **not** reproduce the notebook's scores. "
                 "See Model performance > reproduction check.")

st.divider()

col1, col2, col3 = st.columns(3, gap="large")
with col1:
    st.markdown("**What it takes as input**")
    st.caption(
        "30 numbers: `Time`, `V1` to `V28` and `Amount`. The V columns are anonymised "
        "PCA components from the original transaction data, so they cannot be read as "
        "\"merchant\" or \"country\" - they are just numbers that separate fraud from "
        "normal behaviour."
    )
with col2:
    st.markdown("**What it gives back**")
    st.caption(
        "A probability between 0 and 1, and a decision made by comparing it with a "
        "cut-off you control. The default cut-off is 0.50, the one Praise kept after "
        "testing 0.05 to 0.95."
    )
with col3:
    st.markdown("**What it cannot do**")
    st.caption(
        "Score raw card data, replace human review, or promise the same scores on data "
        "from another bank. Only 473 frauds exist in the whole dataset, so every metric "
        "here is noisy."
    )

ui.not_a_real_system()
ui.provenance(results)
