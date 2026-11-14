"""The app's numbers must be the notebook's numbers.

These are the tests that matter most. If the model is retrained, the dataset
changes or a library version shifts behaviour, one of these fails before a
reviewer sees it.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score

from src import config

NOTEBOOK = {                      # from Project-2-praiseemma/outputs/model_comparison.csv
    "precision": 0.6016,
    "recall": 0.8105,
    "f1": 0.6906,
    "roc_auc": 0.9729,
    "avg_precision": 0.7716,
}
TOLERANCE = 5e-4


@pytest.mark.parametrize("metric,expected", NOTEBOOK.items())
def test_random_forest_matches_the_notebook(results, metric, expected):
    assert results["metrics"]["Random Forest"][metric] == pytest.approx(expected, abs=TOLERANCE)


def test_logistic_regression_matches_the_notebook(results):
    lr = results["metrics"]["Logistic Regression"]
    assert lr["precision"] == pytest.approx(0.0534, abs=TOLERANCE)
    assert lr["recall"] == pytest.approx(0.8737, abs=TOLERANCE)
    assert lr["roc_auc"] == pytest.approx(0.9623, abs=TOLERANCE)


def test_export_declares_a_passing_reproduction_check(results):
    assert results["reproduction"]["all_passed"] is True, results["reproduction"]["checks"]
    for check in results["reproduction"]["checks"]:
        assert check["ok"], f"{check['metric']}: expected {check['expected']}, got {check['got']}"


def test_saved_test_probabilities_reproduce_the_confusion_matrix(predictions):
    """The probabilities shipped to the app, thresholded, must give 77/18/51."""
    probabilities = predictions["test_proba_rf"]
    labels = predictions["y_test"]
    predicted = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    assert (tp, fn, fp) == (77, 18, 51)
    assert precision_score(labels, predicted) == pytest.approx(0.6016, abs=TOLERANCE)
    assert recall_score(labels, predicted) == pytest.approx(0.8105, abs=TOLERANCE)


def test_saved_test_probabilities_reproduce_roc_auc(predictions):
    auc = roc_auc_score(predictions["y_test"], predictions["test_proba_rf"])
    assert auc == pytest.approx(0.9729, abs=TOLERANCE)


def test_split_sizes_match_the_notebook(results):
    split = results["split"]
    assert split["train_rows"] == 226_980
    assert split["test_rows"] == 56_746
    assert split["train_frauds"] == 378
    assert split["test_frauds"] == 95
    assert split["train_rows"] + split["test_rows"] == results["data"]["rows"]


def test_data_profile_matches_the_notebook(results):
    data = results["data"]
    assert data["raw_rows"] == 284_807
    assert data["duplicate_rows"] == 1_081
    assert data["rows"] == 283_726
    assert data["frauds"] == 473
    assert data["missing_values"] == 0


def test_baseline_shows_the_accuracy_illusion(results):
    baseline = results["metrics"]["Baseline (always legitimate)"]
    assert baseline["accuracy"] > 0.998
    assert baseline["recall"] == 0.0


def test_threshold_sweep_covers_the_range(results):
    sweep = results["threshold_sweep"]
    if sweep["training"] is None:
        pytest.skip("training sweep was skipped with --no-oof")
    thresholds = [row["threshold"] for row in sweep["training"]]
    assert thresholds == sorted(thresholds)
    assert thresholds[0] == pytest.approx(0.01)
    assert thresholds[-1] == pytest.approx(0.99)
    # The finding the dashboard states in words.
    at = {round(row["threshold"], 2): row for row in sweep["training"]}
    assert at[0.5]["precision"] > at[0.4]["precision"] > at[0.3]["precision"]
    assert at[0.4]["recall"] - at[0.5]["recall"] < 0.02


def test_demo_sample_is_a_sample_of_all_frauds(demo, results):
    assert len(demo) == results["demo"]["rows"]
    assert int(demo[config.LABEL_COLUMN].sum()) == 473
    assert len(demo) < 5_000, "the demo file should stay small enough to commit"
    assert demo[config.FEATURES].notna().all().all()


def test_dataset_split_reproduces_when_the_full_file_is_present(results):
    """Needs the 150 MB dataset, so it is skipped in CI and anywhere it is absent."""
    from sklearn.model_selection import train_test_split

    if not (config.APP_DIR.parent / "data" / "creditcard.csv").exists():
        pytest.skip("data/creditcard.csv not present (it is never committed)")

    raw = pd.read_csv(config.APP_DIR.parent / "data" / "creditcard.csv")
    clean = raw.drop_duplicates().reset_index(drop=True)
    assert len(clean) == results["data"]["rows"]
    assert int(clean["Class"].sum()) == results["data"]["frauds"]

    X, y = clean.drop(columns="Class"), clean["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    assert len(X_train) == results["split"]["train_rows"]
    assert len(X_test) == results["split"]["test_rows"]
    assert int(y_train.sum()) == results["split"]["train_frauds"]
    assert int(y_test.sum()) == results["split"]["test_frauds"]
