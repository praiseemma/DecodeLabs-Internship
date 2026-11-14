"""Data insights: what makes fraud look different in this dataset."""

from __future__ import annotations

import streamlit as st

from src import config, plots, state, ui

results = state.results()
eda = results["eda"]
summary = eda["outlier_summary"]

ui.header(
    "Data insights",
    "What the exploratory analysis found, and why it justifies the modelling choices. "
    "All of these numbers are descriptive: none of them fed a model, so none of them leak.",
)

ui.metric_row([
    {"label": "Fraud share", "value": f"{eda['class_balance']['fraud_share']:.2f}%",
     "help": "473 frauds in 283,726 rows after removing duplicates."},
    {"label": "Fraud rows unusual in a V column",
     "value": f"{summary['v_outlier_share_fraud']:.0f}%",
     "help": f"Against {summary['v_outlier_share_legitimate']:.0f}% of legitimate rows."},
    {"label": "V columns flagged per fraud row",
     "value": f"{summary['mean_v_flagged_fraud']:.1f}",
     "help": f"Against {summary['mean_v_flagged_legitimate']:.1f} for a legitimate row."},
    {"label": "Amount outliers",
     "value": f"{summary['amount_outlier_share_fraud']:.1f}%",
     "help": f"Fraud rows unusually large: {summary['amount_outlier_share_fraud']:.1f}% "
             f"against {summary['amount_outlier_share_legitimate']:.1f}% of legitimate rows."},
])

left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("Amount does not separate the classes")
    st.plotly_chart(plots.amount_by_class(eda), width="stretch",
                    config=config.PLOTLY_CONFIG)
    st.caption(
        "The two distributions overlap almost completely. Fraud is slightly more common at "
        "small amounts and at a few high ones, but no cut-off on amount alone would work. "
        "That is why `Amount` alone is a weak feature."
    )
with right:
    st.subheader("Fraud moves differently through the day")
    st.plotly_chart(plots.hour_of_day(eda), width="stretch", config=config.PLOTLY_CONFIG)
    st.caption(
        "Legitimate transactions cluster in the daytime; a larger share of fraud falls in "
        "the early hours. With 473 frauds this is suggestive, not conclusive. `Time` counts "
        "seconds from the first transaction, so the hour is approximate."
    )

st.divider()

left, right = st.columns([1, 1.4], gap="large")
with left:
    st.subheader("Which features relate to fraud")
    st.plotly_chart(plots.correlations(eda), width="stretch", config=config.PLOTLY_CONFIG)
    st.caption(
        "Correlation with `Class`, as an absolute value. `V17`, `V14`, `V12` and `V10` lead. "
        "The values are anonymised components, so this says which numbers carry signal, not "
        "what the signal means."
    )
with right:
    st.subheader("Fraud rows are unusual, in many columns at once")
    st.plotly_chart(plots.outlier_columns_per_row(eda), width="stretch",
                    config=config.PLOTLY_CONFIG)
    st.caption(
        f"About {summary['v_outlier_share_fraud']:.0f}% of fraud rows are an IQR outlier in "
        f"at least one V column, against {summary['v_outlier_share_legitimate']:.0f}% of "
        f"legitimate rows. The average fraud row is extreme in about "
        f"{summary['mean_v_flagged_fraud']:.0f} of the 28 columns; the average legitimate row "
        f"in about {summary['mean_v_flagged_legitimate']:.0f}. That is why the columns are "
        "kept as they are, and why the model can separate the classes at all."
    )

st.subheader("Outlier rate for every column, by class")
st.plotly_chart(plots.outlier_rate(eda), width="stretch", config=config.PLOTLY_CONFIG)
st.caption(
    "Each row is one column; the blue dot is the share of legitimate transactions that are "
    "IQR outliers there, the red dot the share of fraud. Where red sits far to the right of "
    "blue, that column separates the classes. This is descriptive only: no model uses these "
    "outlier flags, and the forest is given the raw values."
)

st.divider()

left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("The imbalance, again")
    st.plotly_chart(plots.class_balance(eda), width="stretch", config=config.PLOTLY_CONFIG)
    st.caption(
        "On a log scale, because on a linear scale the fraud bar would be invisible. This "
        "is what SMOTE has to work with: it creates new fraud examples by interpolating "
        "between existing ones."
    )
with right:
    st.subheader("Why outliers were kept here")
    st.markdown(
        "- In Project 1 extreme values were capped. Here they are kept, deliberately.\n"
        "- Fraud rows *are* the unusual rows. Capping them would erase the pattern the "
        "model is meant to learn.\n"
        "- With only 473 frauds, every one matters, and none were removed.\n"
        "- 1,081 exact duplicate rows were removed before the split, so an identical "
        "transaction cannot sit in both training and test."
    )

ui.provenance(results)
