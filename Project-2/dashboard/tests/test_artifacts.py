"""Guards on the committed artifacts: sizes, keys and shapes.

The repository is public and GitHub rejects files over 100 MB, so the size of
what is committed is part of the specification, not an afterthought.
"""

from __future__ import annotations

import pytest

from src import config


def test_model_file_is_small_enough_to_commit():
    assert config.MODEL_PATH.exists()
    size_mb = config.MODEL_PATH.stat().st_size / (1024 * 1024)
    assert size_mb < 20, f"model file is {size_mb:.1f} MB"


def test_no_dataset_is_committed():
    dataset = config.APP_DIR.parent / "data" / "creditcard.csv"
    if dataset.exists():
        assert dataset.stat().st_size > 100 * 1024 * 1024
    # Whatever its size, it must be ignored by git.
    ignore_file = config.APP_DIR.parent.parent / ".gitignore"
    assert "creditcard.csv" in ignore_file.read_text(encoding="utf-8")


def test_results_has_every_key_the_app_reads(results):
    required = {"data", "split", "features", "versions", "model", "metrics", "curves",
                "feature_importances", "threshold_sweep", "threshold_reference", "eda",
                "demo", "reproduction", "regression_rows"}
    missing = required - set(results)
    assert not missing, f"results.json is missing {missing}"


def test_results_declares_the_thirty_features(results):
    assert results["features"] == config.FEATURES
    assert len(results["features"]) == 30


def test_metrics_cover_three_models(results):
    assert set(results["metrics"]) == {"Random Forest", "Logistic Regression",
                                       "Baseline (always legitimate)"}
    for name, metrics in results["metrics"].items():
        for key in ("precision", "recall", "f1", "roc_auc"):
            assert 0.0 <= metrics[key] <= 1.0, f"{name}.{key} out of range"


def test_curves_have_points_for_both_models(results):
    for name in ("Random Forest", "Logistic Regression"):
        curve = results["curves"][name]
        assert len(curve["roc"]["x"]) >= 2
        assert len(curve["pr"]["x"]) >= 2
        assert curve["roc"]["x"][0] <= 1.0


def test_predictions_have_matching_lengths(predictions, results):
    assert len(predictions["test_proba_rf"]) == results["split"]["test_rows"]
    assert len(predictions["test_proba_lr"]) == results["split"]["test_rows"]
    assert len(predictions["y_test"]) == results["split"]["test_rows"]
    if "oof_proba_rf" in predictions:
        assert len(predictions["oof_proba_rf"]) == results["split"]["train_rows"]
        assert len(predictions["y_train"]) == results["split"]["train_rows"]


@pytest.mark.parametrize("name", ["Random Forest", "Logistic Regression"])
def test_confusion_matrices_add_up(results, name):
    metrics = results["metrics"][name]
    total = (metrics["caught"] + metrics["missed"] + metrics["false_alarms"]
             + metrics["true_negatives"])
    assert total == results["split"]["test_rows"]
    assert metrics["caught"] + metrics["missed"] == results["split"]["test_frauds"]


def test_feature_importances_cover_every_input(results):
    importances = results["feature_importances"]
    assert len(importances) == 30
    assert {row["feature"] for row in importances} == set(config.FEATURES)
    assert sum(row["importance"] for row in importances) == pytest.approx(1.0, abs=1e-4)


def test_eda_numbers_are_internally_consistent(results):
    eda = results["eda"]
    balance = eda["class_balance"]
    assert balance["legitimate"] + balance["fraud"] == results["data"]["rows"]
    assert balance["fraud_share"] == pytest.approx(
        balance["fraud"] / results["data"]["rows"] * 100, abs=1e-3)
    assert len(eda["outlier_rate_by_column"]) == 30
    assert sum(row["legitimate"] for row in eda["outlier_columns_per_row"]) == pytest.approx(100, abs=0.5)


def test_requirements_pin_the_versions_the_model_was_saved_with(results):
    text = (config.APP_DIR / "requirements.txt").read_text(encoding="utf-8")
    trained = results["versions"]
    expected = {
        "scikit-learn": trained["scikit_learn"],
        "imbalanced-learn": trained["imbalanced_learn"],
        "pandas": trained["pandas"],
        "numpy": trained["numpy"],
        "joblib": trained["joblib"],
    }
    for package, version in expected.items():
        assert f"{package}=={version}" in text, (
            f"requirements.txt should pin {package}=={version} (the version the model "
            "was exported with)"
        )
