"""The saved model behaves the way it did when it was exported."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, model


def test_pipeline_has_the_notebook_steps(pipeline):
    steps = [name for name, _ in pipeline.steps]
    assert steps == ["smote", "classifier"]
    assert pipeline.named_steps["classifier"].max_depth == 10
    assert pipeline.named_steps["classifier"].n_estimators == 100
    assert pipeline.named_steps["smote"].k_neighbors == 5


def test_probabilities_are_between_zero_and_one(pipeline, demo):
    probabilities = model.predict_proba(pipeline, demo[config.FEATURES].head(200))
    assert probabilities.shape == (200,)
    assert np.isfinite(probabilities).all()
    assert probabilities.min() >= 0.0 and probabilities.max() <= 1.0


def test_scoring_is_deterministic(pipeline, demo):
    """Same input, same output.

    Not bit-for-bit: the 100 trees are averaged in parallel, so the last couple
    of digits shift by about 1e-16 between runs. That is far below anything that
    can change a decision.
    """
    frame = demo[config.FEATURES].head(50)
    first = model.predict_proba(pipeline, frame)
    second = model.predict_proba(pipeline, frame)
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-12)


def test_column_order_does_not_change_the_result(pipeline, demo):
    frame = demo[config.FEATURES].head(25)
    shuffled = frame[list(reversed(config.FEATURES))]
    np.testing.assert_allclose(model.predict_proba(pipeline, frame),
                              model.predict_proba(pipeline, shuffled), rtol=0, atol=1e-12)


def test_empty_frame_gives_an_empty_result(pipeline):
    empty = pd.DataFrame(columns=config.FEATURES, dtype=float)
    assert model.predict_proba(pipeline, empty).size == 0


def test_known_rows_give_the_recorded_probabilities(pipeline, results):
    """Six rows saved at export time, with the probability this model gave them.

    If a different library version changes how the forest scores, this is the test
    that notices.
    """
    rows = results["regression_rows"]
    assert len(rows) >= 4
    for row in rows:
        frame = pd.DataFrame([row["features"]])[config.FEATURES]
        probability = float(model.predict_proba(pipeline, frame)[0])
        assert probability == pytest.approx(row["expected_probability"], abs=1e-6), (
            f"row {row['row_index']} moved from {row['expected_probability']} to {probability}"
        )


def test_a_known_fraud_scores_higher_than_a_typical_legitimate_row(pipeline, demo):
    fraud = demo[demo[config.LABEL_COLUMN] == 1]
    legitimate = demo[demo[config.LABEL_COLUMN] == 0]
    fraud_scores = model.predict_proba(pipeline, fraud[config.FEATURES])
    legit_scores = model.predict_proba(pipeline, legitimate[config.FEATURES])
    assert np.median(fraud_scores) > np.median(legit_scores)
    assert fraud_scores.mean() > 0.8        # the demo holds the easiest frauds too
    assert legit_scores.mean() < 0.2


@pytest.mark.parametrize("probability,expected", [
    (0.95, "Very likely fraud"),
    (0.60, "Flagged as fraud"),
    (0.40, "Uncertain"),
    (0.10, "Likely legitimate"),
    (0.001, "Very likely legitimate"),
])
def test_confidence_band_wording(probability, expected):
    assert model.confidence_band(probability)[0] == expected


def test_unusual_values_returns_the_requested_number_of_rows(pipeline, demo, one_row):
    table = model.unusual_values(one_row, demo, top_n=5)
    assert len(table) == 5
    assert set(table.columns) == {"Input", "Value", "Typical (demo sample)",
                                  "How far out", "Outside demo range"}


def test_version_report_flags_a_mismatch(results):
    rows = model.version_report(results["versions"])
    libs = {row["Library"]: row for row in rows}
    assert "scikit-learn" in libs
    assert libs["scikit-learn"]["Match"] in {"yes", "no"}

    tampered = dict(results["versions"])
    tampered["scikit_learn"] = "0.0.1"
    rows = model.version_report(tampered)
    assert {row["Library"]: row for row in rows}["scikit-learn"]["Match"] == "no"


def test_wilson_interval_stays_inside_zero_and_one():
    low, high = model.wilson_interval(65, 75)
    assert 0.0 <= low < 65 / 75 < high <= 1.0
    assert model.wilson_interval(0, 0) == (0.0, 0.0)
    assert model.wilson_interval(0, 10)[0] == 0.0
    # A small sample is a wide range; that is the point of the interval.
    wide = model.wilson_interval(4, 13)
    assert wide[1] - wide[0] > 0.4


def test_calibration_table_shows_the_score_running_high(predictions):
    rows = model.calibration_table(predictions["test_proba_rf"], predictions["y_test"])
    by_band = {row["Score band"]: row for row in rows}

    assert set(by_band) == {"below 0.50", "0.50 to 0.90", "0.90 and above", "all rows"}
    everything = by_band["all rows"]
    # The defect the About page admits: the model's own numbers overstate reality.
    assert everything["Mean score"] > 10 * everything["Actual fraud rate"]

    # The ordering is still right, which is why the threshold works.
    assert (by_band["below 0.50"]["Actual fraud rate"]
            < by_band["0.50 to 0.90"]["Actual fraud rate"]
            < by_band["0.90 and above"]["Actual fraud rate"])

    assert (by_band["below 0.50"]["Rows"] + by_band["0.50 to 0.90"]["Rows"]
            + by_band["0.90 and above"]["Rows"]) == everything["Rows"]
    assert everything["Rows"] == len(predictions["y_test"])
    assert everything["Frauds"] == int(predictions["y_test"].sum())
