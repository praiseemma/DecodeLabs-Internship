"""Cached access to the exported model and the numbers the app displays.

Two caches, for two different reasons:
- `st.cache_resource` holds the model object itself. Loading a pickle takes about
  a second and there is no reason to do it on every click.
- `st.cache_data` holds tables and dictionaries. Streamlit copies cached data per
  session, so one visitor cannot change what another visitor sees.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from . import config, model


class ArtifactsMissing(RuntimeError):
    """The export script has not been run yet."""


def artifacts_status() -> dict:
    """Which exported files exist. Used to show a helpful setup screen."""
    files = {
        "model": config.MODEL_PATH,
        "results": config.RESULTS_PATH,
        "predictions": config.PREDICTIONS_PATH,
        "demo data": config.DEMO_PATH,
    }
    missing = [name for name, path in files.items() if not path.exists()]
    return {"ready": not missing, "missing": missing,
            "sizes_kb": {name: round(p.stat().st_size / 1024, 1)
                         for name, p in files.items() if p.exists()}}


@st.cache_data(show_spinner=False)
def results(path: str = str(config.RESULTS_PATH)) -> dict:
    path_obj = Path(path)
    if not path_obj.exists():
        raise ArtifactsMissing(
            "results.json is missing. Run dashboard/scripts/export_model.py once."
        )
    return json.loads(path_obj.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def demo(path: str = str(config.DEMO_PATH)) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise ArtifactsMissing(
            "demo_transactions.csv is missing. Run dashboard/scripts/export_model.py once."
        )
    return pd.read_csv(path_obj)


@st.cache_data(show_spinner=False)
def predictions(path: str = str(config.PREDICTIONS_PATH)) -> dict[str, np.ndarray]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise ArtifactsMissing(
            "predictions.npz is missing. Run dashboard/scripts/export_model.py once."
        )
    with np.load(path_obj) as arrays:
        return {name: arrays[name] for name in arrays.files}


@st.cache_resource(show_spinner="Loading the Random Forest ...")
def pipeline(path: str = str(config.MODEL_PATH)):
    return model.load_pipeline(path)


@st.cache_data(show_spinner="Scoring the demo transactions ...")
def demo_scored(path: str = str(config.DEMO_PATH)) -> pd.DataFrame:
    """The demo sample with each row's fraud probability already added.

    Scoring 2,473 rows takes well under a second, but it only needs doing once
    per deployment, so it is cached rather than repeated on every click.
    """
    frame = demo(path).copy()
    frame["fraud_probability"] = model.predict_proba(
        pipeline(str(config.MODEL_PATH)), frame[config.FEATURES]
    )
    return frame
