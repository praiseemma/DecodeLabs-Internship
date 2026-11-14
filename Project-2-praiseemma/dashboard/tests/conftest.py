"""Shared fixtures. Run from the dashboard folder: `python -m pytest`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DASHBOARD = Path(__file__).resolve().parents[1]
if str(DASHBOARD) not in sys.path:
    sys.path.insert(0, str(DASHBOARD))

from src import config, model  # noqa: E402


@pytest.fixture(scope="session")
def results() -> dict:
    if not config.RESULTS_PATH.exists():
        pytest.skip("artifacts missing: run scripts/export_model.py first")
    return json.loads(config.RESULTS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def pipeline():
    if not config.MODEL_PATH.exists():
        pytest.skip("model missing: run scripts/export_model.py first")
    return model.load_pipeline(config.MODEL_PATH)


@pytest.fixture(scope="session")
def demo() -> pd.DataFrame:
    if not config.DEMO_PATH.exists():
        pytest.skip("demo data missing: run scripts/export_model.py first")
    return pd.read_csv(config.DEMO_PATH)


@pytest.fixture(scope="session")
def predictions() -> dict[str, np.ndarray]:
    if not config.PREDICTIONS_PATH.exists():
        pytest.skip("predictions missing: run scripts/export_model.py first")
    with np.load(config.PREDICTIONS_PATH) as arrays:
        return {name: arrays[name] for name in arrays.files}


@pytest.fixture
def one_row(demo: pd.DataFrame) -> pd.DataFrame:
    """A single known-fraud row, in the model's column order."""
    row = demo[demo[config.LABEL_COLUMN] == 1].head(1)
    return row[config.FEATURES].reset_index(drop=True)
