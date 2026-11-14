"""Score one transaction: pick a known example or type your own 30 values."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config, model, plots, state, ui
from src.validation import ValidationError, coerce_threshold, validate_transaction

results = state.results()
demo = state.demo_scored()
pipeline = state.pipeline()

FRAUD = demo[demo[config.LABEL_COLUMN] == 1]
LEGIT = demo[demo[config.LABEL_COLUMN] == 0]


@st.cache_data(show_spinner=False)
def _picks() -> list[int]:
    """A small, deliberately chosen set of examples rather than 2,473 options.

    High-confidence fraud, fraud near the boundary, confident legitimate, and a
    few legitimate transactions the model gets wrong - the interesting cases.
    """
    by_score = FRAUD.sort_values("fraud_probability", ascending=False)
    near_legit = by_score[by_score["fraud_probability"] < 0.9].head(6)
    flagged_legit = LEGIT[LEGIT["fraud_probability"] >= 0.5].sort_values(
        "fraud_probability", ascending=False).head(6)
    chosen = pd.concat([
        by_score.head(12),
        near_legit,
        flagged_legit,
        LEGIT.sort_values("fraud_probability").head(6),
    ]).drop_duplicates()
    return [int(i) for i in chosen.index]


PICKS = _picks()

# Options are the labels themselves, not row numbers: a string value survives
# Streamlit's state serialisation cleanly, and the row number in the label makes
# every option unique and traceable back to the demo CSV.
OPTIONS: dict[str, int] = {}
for _index in PICKS:
    _row = demo.loc[_index]
    _truth = "Known fraud" if _row[config.LABEL_COLUMN] == 1 else "Known legitimate"
    OPTIONS[f"{_truth} · row {_index} · Amount {_row['Amount']:,.2f} · "
            f"{_row['Time'] / 3600:.0f} h in"] = _index


def _load(row: pd.Series) -> None:
    for feature in config.FEATURES:
        st.session_state[f"value_{feature}"] = float(row[feature])


if "value_V1" not in st.session_state:
    _load(demo.loc[PICKS[0]])

ui.header(
    "Score a single transaction",
    "The model needs 30 numbers. The V columns are anonymised, so typing them by hand "
    "is impractical - start from a known example, then edit anything you like.",
)

controls, spacer = st.columns([1, 1])
with controls:
    threshold = st.slider(
        "Decision threshold",
        min_value=config.MIN_THRESHOLD, max_value=config.MAX_THRESHOLD,
        value=config.DEFAULT_THRESHOLD, step=config.THRESHOLD_STEP,
        help="Probabilities at or above this value are called fraud. 0.50 is the "
             "threshold Praise kept; the Threshold explorer shows what moving it costs.",
    )
threshold = coerce_threshold(threshold)

mode = st.radio(
    "Where do the 30 numbers come from?",
    ["Pick a known transaction", "Enter my own values"],
    horizontal=True,
    help="Known transactions are taken from the committed demo sample, so the true "
         "label is available for comparison.",
)

if mode == "Pick a known transaction":
    choice = st.selectbox(
        "Example transaction",
        options=list(OPTIONS),
        index=0,
        key="pick",
    )
    index = OPTIONS[choice]
    st.caption(
        "Every example comes from the demo sample: all 473 frauds from the cleaned "
        "dataset plus 2,000 legitimate transactions. The label is the truth recorded "
        "in the dataset, not something the model produced. The row number matches the "
        "row in `data/demo_transactions.csv`."
    )
    _load(demo.loc[index])
    truth = int(demo.loc[index, config.LABEL_COLUMN])
else:
    left, right = st.columns(2)
    if left.button("Fill with a fraud example", width="stretch"):
        _load(FRAUD.sort_values("fraud_probability", ascending=False).iloc[0])
    if right.button("Fill with a legitimate example", width="stretch"):
        _load(LEGIT.sort_values("fraud_probability").iloc[0])
    truth = None
    st.caption("No true label is known for values you type, so only the prediction is shown.")

with st.expander("The 30 inputs: Time, V1 to V28, Amount", expanded=False):
    st.caption(
        "`Time` is seconds since the first transaction in the file. `Amount` is the "
        "transaction value. `V1` to `V28` are PCA components of the original, private "
        "features: they carry the signal but have no individual meaning."
    )
    time_col, amount_col = st.columns(2)
    time_col.number_input("Time (seconds)", step=1.0, format="%.0f", min_value=0.0,
                          key="value_Time",
                          help="Seconds since the first transaction. Hour of day is Time / 3600, mod 24.")
    amount_col.number_input("Amount", step=0.01, format="%.2f", min_value=0.0,
                            key="value_Amount")
    st.markdown("**V1 to V28**")
    columns = st.columns(4)
    for position, feature in enumerate(config.V_FEATURES):
        columns[position % 4].number_input(
            feature, step=0.01, format="%.5f", key=f"value_{feature}",
            label_visibility="visible",
        )

values = {feature: st.session_state[f"value_{feature}"] for feature in config.FEATURES}

st.divider()

try:
    frame = validate_transaction(values)
    probability = float(model.predict_proba(pipeline, frame)[0])
except ValidationError as error:
    st.error(error.as_markdown())
    st.stop()
except Exception:  # noqa: BLE001 - never show a stack trace to a visitor
    st.error(
        "The model could not score these values. Check that every field holds a "
        "number, then try again."
    )
    st.stop()

is_fraud = probability >= threshold

score_col, verdict_col = st.columns([1.25, 1], gap="large")

with score_col:
    ui.metric_row([
        {"label": "Fraud probability", "value": f"{probability:.2%}",
         "help": "The share of trees in the forest that voted fraud, averaged."},
        {"label": "Decision", "value": "Fraud" if is_fraud else "Legitimate",
         "help": f"Compared with the cut-off of {threshold:.2f}."},
    ])
    st.plotly_chart(plots.probability_bar(probability, threshold), width="stretch",
                    config=config.PLOTLY_CONFIG)
    ui.decision_badge(is_fraud, threshold)

with verdict_col:
    band, colour, explanation = model.confidence_band(probability)
    st.markdown(f"**Reading this score:** {band}")
    st.caption(explanation)
    if truth is not None:
        truth_name = "fraud" if truth == 1 else "legitimate"
        if (truth == 1) == is_fraud:
            st.success(f"The true label is **{truth_name}**, so at this cut-off the model is right.")
        else:
            st.warning(f"The true label is **{truth_name}**, so at this cut-off the model is wrong.")
    st.caption(
        f"At a cut-off of {threshold:.2f}, this transaction would be "
        f"{'blocked or reviewed' if is_fraud else 'approved'}."
    )

st.divider()

st.subheader("Which of these values are unusual?")
st.caption(
    "This is **not** how the Random Forest explains itself - a forest does not expose "
    "per-row reasons that can be quoted honestly. It answers a narrower question the "
    "exploratory analysis can support: are any of these 30 numbers far from typical? "
    "About 97% of fraud rows in this dataset are IQR outliers in at least one V column, "
    "against about 45% of legitimate rows."
)
st.dataframe(
    model.unusual_values(frame, demo, top_n=6),
    width="stretch", hide_index=True,
)

with st.expander("What a score like this means in practice"):
    rf = results["metrics"]["Random Forest"]
    st.markdown(
        f"On the 56,746 held-back test transactions, at a cut-off of 0.50 the model "
        f"flagged 128 of them: 77 were real fraud and 51 were not. So a very high score "
        f"is usually right, and a score just above the line is much less certain.\n\n"
        f"Overall it caught **{rf['recall']:.1%}** of the fraud and **{rf['precision']:.1%}** "
        f"of its alerts were real. Both numbers are small and noisy because only 95 "
        f"frauds sit in the test set: each one is worth about one point of recall."
    )

ui.provenance(results)
