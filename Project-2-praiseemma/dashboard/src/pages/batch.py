"""Batch scoring: upload a CSV, get probabilities and decisions back."""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import streamlit as st

from src import config, model, state, ui
from src.validation import ValidationError, coerce_threshold, validate_batch

results = state.results()
pipeline = state.pipeline()

ui.header(
    "Score a batch of transactions",
    "Upload a CSV with the 30 model inputs and get a probability for every row. "
    "Nothing is stored: the file is scored in memory and forgotten when you leave.",
)

threshold = coerce_threshold(st.slider(
    "Decision threshold",
    min_value=config.MIN_THRESHOLD, max_value=config.MAX_THRESHOLD,
    value=config.DEFAULT_THRESHOLD, step=config.THRESHOLD_STEP,
    help="Same cut-off as the single-transaction page. 0.50 is the kept threshold.",
))

template = state.demo().head(5)
st.download_button(
    "Download a 5-row template",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="fraud_scoring_template.csv",
    mime="text/csv",
    help="The columns the model expects: Time, V1 to V28, Amount, and an optional "
         "Class column holding the true label.",
)
st.caption(
    f"Rules enforced here: at most **{config.MAX_BATCH_ROWS:,} rows** and "
    f"**{config.MAX_UPLOAD_MB:.0f} MB** per file, all 30 columns present, numeric, "
    "no blanks, and no negative `Time` or `Amount`. A `Class` column is optional; if "
    "it is there it must hold 0 or 1 and is used only to check the predictions."
)

uploaded = st.file_uploader("Choose a CSV file", type=["csv"], accept_multiple_files=False)

if uploaded is not None:
    st.session_state["use_demo"] = False
if st.button("Score the committed demo sample instead (2,473 rows)"):
    st.session_state["use_demo"] = True

source_frame: pd.DataFrame | None = None
source_name = ""

if uploaded is not None:
    payload = uploaded.getvalue()
    size_mb = len(payload) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_MB:
        st.error(
            f"That file is {size_mb:.1f} MB, over the {config.MAX_UPLOAD_MB:.0f} MB limit. "
            f"Split it, or score at most {config.MAX_BATCH_ROWS:,} rows at a time."
        )
        st.stop()
    try:
        source_frame = pd.read_csv(io.BytesIO(payload))
        source_name = uploaded.name
    except pd.errors.EmptyDataError:
        st.error("That file is empty. It has no header and no rows.")
        st.stop()
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(
            "That file could not be read as a CSV. "
            "Re-export it from Excel or pandas and try again."
        )
        st.caption(f"Technical detail for the curious: {type(error).__name__}")
        st.stop()
elif st.session_state.get("use_demo"):
    source_frame = state.demo()
    source_name = "demo_transactions.csv (committed sample)"

if source_frame is None:
    st.info(
        "Nothing loaded yet. Upload a file, or press **Score the committed demo sample** "
        "to try the app without preparing anything."
    )
    st.stop()

try:
    batch = validate_batch(source_frame, source=source_name)
except ValidationError as error:
    st.error(error.as_markdown())
    st.stop()

for warning in batch.warnings:
    st.warning(warning)

started = time.perf_counter()
try:
    probabilities = model.predict_proba(pipeline, batch.features)
except Exception:  # noqa: BLE001 - never show a stack trace
    st.error("The model could not score this file. Check the column values and try again.")
    st.stop()
elapsed = time.perf_counter() - started

scored = batch.features.copy()
scored.insert(0, "Row", np.arange(1, len(scored) + 1))
scored["Fraud probability"] = probabilities
scored["Decision"] = np.where(probabilities >= threshold, "Fraud", "Legitimate")
if batch.labels is not None:
    scored["Actual"] = batch.labels.map({0: "Legitimate", 1: "Fraud"})
    scored["Outcome"] = np.where(
        (batch.labels.to_numpy() == 1) == (probabilities >= threshold), "correct", "incorrect"
    )

flagged = int((probabilities >= threshold).sum())
st.subheader("Summary")
ui.metric_row([
    {"label": "Rows scored", "value": f"{batch.rows:,}"},
    {"label": "Flagged as fraud", "value": f"{flagged:,}",
     "help": f"{(flagged / batch.rows):.2%} of the file."},
    {"label": "Time taken", "value": f"{elapsed:.2f} s"},
    {"label": "Cut-off used", "value": f"{threshold:.2f}"},
])

if batch.labels is not None:
    truth = batch.labels.to_numpy()
    predicted = (probabilities >= threshold).astype(int)
    tp = int(((truth == 1) & (predicted == 1)).sum())
    fp = int(((truth == 0) & (predicted == 1)).sum())
    fn = int(((truth == 1) & (predicted == 0)).sum())
    tn = int(((truth == 0) & (predicted == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    left, right = st.columns([1, 1], gap="large")
    with left:
        ui.metric_row([
            {"label": "Precision on this file", "value": f"{precision:.3f}"},
            {"label": "Recall on this file", "value": f"{recall:.3f}"},
        ])
        st.caption(
            f"Counts: {tp} caught, {fn} missed, {fp} false alarms, {tn} correctly passed. "
            f"F1 {f1:.3f}. This file is {truth.mean():.2%} fraud."
        )
        if source_name.startswith("demo"):
            st.warning(
                "The demo file deliberately over-represents fraud (473 of 2,473 rows, "
                "about 19%) so that examples are easy to find. Precision measured on it "
                "is therefore **not** the model's real-world precision. The honest "
                "figures are the 56,746-row test set on the Model performance page."
            )
    with right:
        st.markdown("**What to read from this**")
        st.caption(
            "Lowering the cut-off catches more fraud and raises more false alarms. On this "
            "file you can see that trade-off directly: move the slider above and watch "
            "precision and recall swap."
        )

st.subheader("Results")
view = st.toggle("Show all 30 input columns", value=False)
display = scored if view else scored[["Row", "Time", "Amount", "Fraud probability",
                                      "Decision"] + (["Actual", "Outcome"]
                                                     if "Actual" in scored else [])]
st.caption(
    f"Showing {min(len(display), config.MAX_CSV_PREVIEW_ROWS):,} of {len(display):,} rows. "
    "The download below contains every row and every column."
)
st.dataframe(
    display.head(config.MAX_CSV_PREVIEW_ROWS),
    width="stretch",
    hide_index=True,
    column_config={
        "Fraud probability": st.column_config.ProgressColumn(
            "Fraud probability", min_value=0.0, max_value=1.0, format="%.3f"
        ),
        "Amount": st.column_config.NumberColumn("Amount", format="%.2f"),
        "Time": st.column_config.NumberColumn("Time", format="%.0f"),
    },
)

st.download_button(
    "Download the scored file (CSV)",
    data=scored.to_csv(index=False).encode("utf-8"),
    file_name="scored_transactions.csv",
    mime="text/csv",
    type="primary",
)

with st.expander("How the rules are enforced"):
    st.markdown(
        "Every check runs before the model sees the data: column names are matched "
        "case-insensitively and stray spaces are trimmed, an index column saved by "
        "pandas is dropped, and then the file is rejected if a column is missing or "
        "unexpected, a value is not a number, a cell is blank, or `Time`/`Amount` is "
        "negative. The same function checks the single-transaction page, so the two "
        "cannot drift apart."
    )

ui.provenance(results)
