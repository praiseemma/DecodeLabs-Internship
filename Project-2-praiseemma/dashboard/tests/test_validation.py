"""The input rules. These are the tests that stop a bad file reaching the model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.validation import ValidationError, coerce_threshold, validate_batch, validate_transaction


def make_frame(rows: int = 3) -> pd.DataFrame:
    """A small, valid table with the 30 model inputs."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(0, 1, size=(rows, len(config.FEATURES))),
                         columns=config.FEATURES)
    frame["Time"] = np.arange(rows, dtype=float)
    frame["Amount"] = np.linspace(1, 100, rows)
    return frame


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------
def test_valid_frame_passes_and_is_reordered():
    frame = make_frame()
    shuffled = frame[list(reversed(config.FEATURES))]
    result = validate_batch(shuffled)
    assert list(result.features.columns) == config.FEATURES
    assert result.rows == len(frame)
    assert result.labels is None
    assert any("Class" in warning for warning in result.warnings)


def test_column_names_are_matched_case_insensitively():
    frame = make_frame()
    frame.columns = [c.lower() for c in frame.columns]
    result = validate_batch(frame)
    assert list(result.features.columns) == config.FEATURES


def test_surrounding_spaces_in_column_names_are_ignored():
    frame = make_frame()
    frame.columns = [f" {c} " for c in frame.columns]
    result = validate_batch(frame)
    assert list(result.features.columns) == config.FEATURES


def test_an_index_column_saved_by_pandas_is_dropped():
    frame = make_frame()
    frame.insert(0, "Unnamed: 0", np.arange(len(frame)))
    result = validate_batch(frame)
    assert "Unnamed: 0" not in result.features.columns
    assert any("index column" in warning for warning in result.warnings)


def test_labels_are_returned_when_they_are_valid():
    frame = make_frame()
    frame[config.LABEL_COLUMN] = [0, 1, 0]
    result = validate_batch(frame)
    assert result.labels is not None
    assert result.labels.tolist() == [0, 1, 0]
    assert config.LABEL_COLUMN not in result.features.columns


# --------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------
def test_missing_column_is_rejected_and_named():
    frame = make_frame().drop(columns=["V7"])
    with pytest.raises(ValidationError) as error:
        validate_batch(frame)
    assert "V7" in error.value.as_markdown()


def test_extra_column_is_rejected_and_named():
    frame = make_frame()
    frame["Merchant"] = "shop"
    with pytest.raises(ValidationError) as error:
        validate_batch(frame)
    assert "Merchant" in error.value.as_markdown()


def test_text_in_a_numeric_column_is_rejected():
    frame = make_frame()
    frame.loc[1, "Amount"] = "twelve"
    with pytest.raises(ValidationError) as error:
        validate_batch(frame)
    assert "Amount" in error.value.as_markdown()
    assert "twelve" in error.value.as_markdown()


def test_blank_cell_is_rejected():
    frame = make_frame()
    frame.loc[0, "V3"] = np.nan
    with pytest.raises(ValidationError) as error:
        validate_batch(frame)
    assert "V3" in error.value.as_markdown()


def test_infinite_value_is_rejected():
    frame = make_frame()
    frame.loc[0, "V3"] = np.inf
    with pytest.raises(ValidationError):
        validate_batch(frame)


@pytest.mark.parametrize("column", ["Time", "Amount"])
def test_negative_values_are_rejected(column):
    frame = make_frame()
    frame.loc[2, column] = -1.0
    with pytest.raises(ValidationError) as error:
        validate_batch(frame)
    assert column in error.value.as_markdown()


def test_duplicate_column_names_are_rejected():
    frame = make_frame()
    frame = pd.concat([frame, frame[["Amount"]]], axis=1)
    with pytest.raises(ValidationError) as error:
        validate_batch(frame)
    assert "same name" in error.value.headline


def test_empty_file_is_rejected():
    with pytest.raises(ValidationError) as error:
        validate_batch(make_frame().iloc[0:0])
    assert "empty" in error.value.headline


def test_too_many_rows_is_rejected():
    frame = make_frame(rows=5)
    with pytest.raises(ValidationError) as error:
        validate_batch(frame, max_rows=4)
    assert "more than" in error.value.headline


@pytest.mark.parametrize("value", [2, -1, 1.5, "fraud"])
def test_labels_outside_zero_and_one_are_rejected(value):
    frame = make_frame()
    frame[config.LABEL_COLUMN] = [0, 1, value]
    with pytest.raises(ValidationError):
        validate_batch(frame)


# --------------------------------------------------------------------------
# Single row
# --------------------------------------------------------------------------
def test_single_row_matches_the_batch_path(one_row):
    values = {name: float(one_row.iloc[0][name]) for name in config.FEATURES}
    from_single = validate_transaction(values)
    from_batch = validate_batch(one_row).features
    assert list(from_single.columns) == list(from_batch.columns)
    np.testing.assert_allclose(from_single.to_numpy(), from_batch.to_numpy())


def test_single_row_reports_a_missing_value(one_row):
    values = {name: float(one_row.iloc[0][name]) for name in config.FEATURES}
    values["V9"] = None
    with pytest.raises(ValidationError) as error:
        validate_transaction(values)
    assert "V9" in str(error.value)


# --------------------------------------------------------------------------
# Threshold
# --------------------------------------------------------------------------
def test_threshold_is_clamped_into_range():
    assert coerce_threshold(-3) == config.MIN_THRESHOLD
    assert coerce_threshold(50) == config.MAX_THRESHOLD
    assert coerce_threshold(0.5) == 0.5


@pytest.mark.parametrize("value", ["abc", float("nan"), float("inf")])
def test_nonsense_threshold_is_rejected(value):
    with pytest.raises(ValidationError):
        coerce_threshold(value)
