"""Threshold explorer: what moving the cut-off does to precision and recall."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import config, plots, state, ui
from src.validation import coerce_threshold

results = state.results()
arrays = state.predictions()
reference = results["threshold_reference"]

training_sweep = results["threshold_sweep"]["training"]
test_sweep = results["threshold_sweep"]["test"]
has_training = bool(training_sweep)

ui.header(
    "Threshold explorer",
    "The model outputs a probability. A threshold turns that probability into a decision. "
    "Move it and watch what happens to precision and recall.",
)

if not has_training:
    st.warning(
        "The training-data sweep was skipped when this app's data was exported "
        "(`--no-oof`), so only the test-set view is available. Re-run "
        "`scripts/export_model.py` without that flag to restore the honest view."
    )

source_label = st.radio(
    "Which probabilities should this page use?",
    options=["Cross-validated training data (the honest choice)",
             "Test set (for illustration only)"],
    index=0 if has_training else 1,
    horizontal=False,
)
use_training = source_label.startswith("Cross-validated")

if use_training:
    probabilities = arrays["oof_proba_rf"]
    labels = arrays["y_train"]
    sweep = training_sweep
    st.success(
        "**Training data, out-of-fold.** Each training transaction was scored by a model "
        f"that never saw it during training ({results['threshold_sweep']['training_note']}). "
        "This is the right place to choose a threshold."
    )
else:
    probabilities = arrays["test_proba_rf"]
    labels = arrays["y_test"]
    sweep = test_sweep
    st.error(
        "**Test set.** Useful for checking what a threshold *would have done*, but a "
        "threshold chosen by looking at the test set makes the test score meaningless: "
        "the number stops being an honest estimate for unseen data."
    )

threshold = coerce_threshold(st.slider(
    "Threshold for calling a transaction fraud",
    min_value=config.MIN_THRESHOLD, max_value=config.MAX_THRESHOLD,
    value=config.DEFAULT_THRESHOLD, step=config.THRESHOLD_STEP,
))

predicted = (probabilities >= threshold).astype(int)
tp = int(((labels == 1) & (predicted == 1)).sum())
fp = int(((labels == 0) & (predicted == 1)).sum())
fn = int(((labels == 1) & (predicted == 0)).sum())
tn = int(((labels == 0) & (predicted == 0)).sum())
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

ui.metric_row([
    {"label": "Precision", "value": f"{precision:.3f}",
     "help": "Of the transactions flagged, the share that really are fraud."},
    {"label": "Recall", "value": f"{recall:.3f}",
     "help": f"Of the {tp + fn} real frauds, the share caught."},
    {"label": "F1", "value": f"{f1:.3f}", "help": "One number balancing the two."},
    {"label": "False alarms", "value": f"{fp:,}",
     "help": "Legitimate transactions wrongly flagged at this threshold."},
])

left, right = st.columns([1, 1], gap="large")
with left:
    matrix = {"true_negatives": tn, "false_alarms": fp, "missed": fn, "caught": tp}
    st.plotly_chart(plots.confusion(matrix, f"At a threshold of {threshold:.2f}"),
                    width="stretch", config=config.PLOTLY_CONFIG)
with right:
    st.markdown("**What this threshold would do**")
    st.markdown(
        f"- Catch **{tp:,}** of the **{tp + fn:,}** real frauds, missing {fn:,}.\n"
        f"- Raise **{fp:,}** false alarms out of **{fp + tn:,}** legitimate transactions.\n"
        f"- For every 100 alerts, about **{precision * 100:.0f}** would be real fraud."
    )
    st.caption(
        f"Each real fraud here is worth "
        f"{100 / max(tp + fn, 1):.2f} percentage points of recall, which is why small "
        "differences between thresholds should not be over-read."
    )

st.plotly_chart(
    plots.threshold_curves(sweep, threshold,
                           (reference["f1_optimal_training"] or {}).get("threshold")
                           if use_training else None),
    width="stretch", config=config.PLOTLY_CONFIG,
)
st.caption(
    f"Precision and recall against the threshold. The dashed grey line is the default "
    f"0.50; the solid black line is the threshold selected above."
)

st.subheader("How the cut-off splits the two classes")
st.plotly_chart(plots.score_distribution(probabilities, labels, threshold),
                width="stretch", config=config.PLOTLY_CONFIG)
st.caption(
    "The model's probabilities for the two true classes, on a log scale. The further "
    "apart the blue and red masses, the easier the decision. The red mass left of the "
    "line is missed fraud; the blue mass right of it is false alarms."
)

st.divider()


def _row_at(target: float) -> dict | None:
    for row in sweep:
        if abs(row["threshold"] - target) < 1e-9:
            return row
    return None


at_50, at_40, at_30 = _row_at(0.5), _row_at(0.4), _row_at(0.3)
f1_best = max(sweep, key=lambda r: r["f1"])

st.subheader("The finding this page exists to show")
if at_50 and at_40 and at_30:
    st.markdown(
        f"- At **0.50**: precision {at_50['precision']:.3f}, recall {at_50['recall']:.3f}.\n"
        f"- At **0.40**: precision {at_40['precision']:.3f}, recall {at_40['recall']:.3f} "
        f"(recall {at_40['recall'] - at_50['recall']:+.3f}, precision "
        f"{at_40['precision'] - at_50['precision']:+.3f}).\n"
        f"- At **0.30**: precision {at_30['precision']:.3f}, recall {at_30['recall']:.3f}."
    )
    st.markdown(
        "Lowering the cut-off from 0.50 to 0.40 bought almost no extra recall for a large "
        "loss of precision. That is why **0.50 was kept**: the rule Praise fixed in advance "
        "was the highest recall among thresholds whose precision is at least the precision "
        "at 0.50, and no lower threshold qualified."
    )
f1_peak_training = reference.get("f1_optimal_training")
peak_sentence = ""
if f1_peak_training:
    peak_sentence = (
        f" On the training data the F1 peak sat at a threshold of "
        f"{f1_peak_training['threshold']:.2f}, where precision is higher and recall lower; "
        "moving there trades recall for precision, which is only worth it if a false alarm "
        "costs more than a missed fraud."
    )
st.markdown(
    f"The threshold that maximises F1 on this data is **{f1_best['threshold']:.2f}** "
    f"(precision {f1_best['precision']:.3f}, recall {f1_best['recall']:.3f})."
    + peak_sentence
)

st.info(
    "A threshold is a business decision, not a statistical one. It encodes the cost of a "
    "missed fraud against the cost of blocking a customer. Nothing in this data tells you "
    "those costs; they come from the business."
)

with st.expander("The numbers as a table (every 0.05)"):
    table = pd.DataFrame(sweep)
    table = table[np.isclose(table["threshold"] % 0.05, 0, atol=1e-9)]
    st.dataframe(
        table.round(3), width="stretch", hide_index=True,
        column_config={"threshold": st.column_config.NumberColumn("Threshold", format="%.2f")},
    )

ui.provenance(results)
