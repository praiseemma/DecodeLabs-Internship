"""Model performance: the evidence the deployed model is the better of the two."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config, plots, state, ui

results = state.results()
metrics = results["metrics"]
curves = results["curves"]
split = results["split"]
rf = metrics["Random Forest"]
lr = metrics["Logistic Regression"]
base = metrics["Baseline (always legitimate)"]

ui.header(
    "Model performance",
    f"Every number here comes from the {split['test_rows']:,} test transactions held back "
    f"before training, containing {split['test_frauds']} real frauds. The models never saw "
    "them during training or tuning.",
)

rows = []
for name in ["Random Forest", "Logistic Regression", "Baseline (always legitimate)"]:
    m = metrics[name]
    rows.append({
        "Model": name,
        "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"],
        "ROC-AUC": m["roc_auc"], "Avg precision": m["avg_precision"],
    })
table = pd.DataFrame(rows)

st.subheader("Three models, one test set")
st.dataframe(
    table, width="stretch", hide_index=True,
    column_config={
        column: st.column_config.NumberColumn(column, format="%.4f")
        for column in ["Precision", "Recall", "F1", "ROC-AUC", "Avg precision"]
    },
)

st.warning(
    f"**The accuracy illusion.** The baseline labels every transaction legitimate. It scores "
    f"**{base['accuracy']:.2%} accuracy** and catches **0%** of the fraud. Accuracy is "
    "therefore not used anywhere in this project: on data where 1 row in 600 is fraud, a "
    "model can be 99.8% right and completely useless."
)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("**Why the Random Forest was chosen**")
    st.markdown(
        f"It catches nearly as much fraud as Logistic Regression "
        f"({rf['recall']:.1%} against {lr['recall']:.1%}) while being right about far more "
        f"of its alerts: **{rf['precision']:.1%}** against **{lr['precision']:.1%}**. "
        f"Logistic Regression's precision means roughly 19 of every 20 alerts are wasted "
        "investigations."
    )
with col2:
    st.markdown("**Why ROC-AUC alone would mislead**")
    st.markdown(
        f"On ROC-AUC the two look close ({rf['roc_auc']:.3f} against {lr['roc_auc']:.3f}), "
        f"because the ROC curve is dominated by the {split['test_rows'] - split['test_frauds']:,} "
        f"legitimate rows. Average precision tells the truth about the rare class: "
        f"{rf['avg_precision']:.3f} against {lr['avg_precision']:.3f}, where a random guess "
        f"would score about {base['avg_precision']:.4f}."
    )

st.divider()

left, right = st.columns(2, gap="large")
with left:
    st.subheader("ROC curve")
    st.plotly_chart(plots.roc(curves), width="stretch", config=config.PLOTLY_CONFIG)
    st.caption(
        "True positive rate against false positive rate. Good for ranking overall, "
        "flattering when the rare class is tiny."
    )
with right:
    st.subheader("Precision-recall curve")
    st.plotly_chart(plots.precision_recall(curves, base["avg_precision"]),
                    width="stretch", config=config.PLOTLY_CONFIG)
    st.caption(
        "Precision against recall. This is the honest picture for a 0.17% positive class: "
        "the gap between the two models is obvious here and invisible on ROC."
    )

st.divider()

st.subheader("Confusion matrices at the default 0.50 cut-off")
left, right = st.columns(2, gap="large")
for column, name in ((left, "Random Forest"), (right, "Logistic Regression")):
    with column:
        if name in metrics:
            st.plotly_chart(plots.confusion(metrics[name], name), width="stretch",
                            config=config.PLOTLY_CONFIG)
            m = metrics[name]
            st.caption(
                f"{m['caught']} caught, {m['missed']} missed, {m['false_alarms']:,} false "
                f"alarms out of {split['test_rows']:,} rows."
            )

st.divider()

st.subheader("What the Random Forest relied on")
st.plotly_chart(plots.feature_importance(results["feature_importances"], top=12),
                width="stretch", config=config.PLOTLY_CONFIG)
st.caption(
    "Importance is how much each feature reduced impurity across all 100 trees. It is a "
    "measure of use, not of meaning: the top features are anonymised PCA components, so "
    "the chart says which numbers the forest leaned on, not why those numbers matter. "
    "`Time` and `Amount` barely register, matching the EDA finding that amount alone "
    "separates the classes weakly."
)

st.divider()

st.subheader("Reproduction check")
repro = results["reproduction"]
if repro["all_passed"]:
    st.success(
        "The model this app serves was rebuilt from `creditcard.csv` with the same split "
        "and seed, and it matches the notebook's recorded test scores."
    )
else:
    st.error("The rebuilt model does not match the notebook. Do not treat these numbers as final.")

checks = pd.DataFrame(repro["checks"])
# The confusion-matrix row holds a list, which Streamlit cannot display as a
# number, so every value is turned into text for the table.
checks_display = pd.DataFrame({
    "Check": checks["metric"].str.title(),
    "Notebook": checks["expected"].map(lambda value: str(value)),
    "Rebuilt model": checks["got"].map(lambda value: str(value)),
    "Result": checks["ok"].map({True: "PASS", False: "FAIL"}),
})
st.dataframe(checks_display, width="stretch", hide_index=True)
st.caption(
    f"Tolerance {repro['tolerance']}, because the notebook records four decimal places. "
    f"The model was exported with scikit-learn {results['versions']['scikit_learn']} and "
    f"imbalanced-learn {results['versions']['imbalanced_learn']} on Python "
    f"{results['versions']['python']}."
)

ui.provenance(results)
