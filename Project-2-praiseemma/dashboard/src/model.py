"""Loading and using the exported Random Forest.

Kept free of Streamlit calls so the tests can import it directly. The cached
wrappers live in `state.py`.
"""

from __future__ import annotations

import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import config


class ModelNotAvailable(RuntimeError):
    """The exported model file is missing or unreadable."""


def load_pipeline(path: Path | str = config.MODEL_PATH):
    """Load the saved pipeline (scaler/SMOTE/classifier wrapped together).

    Returns the whole `imblearn` Pipeline, not just the classifier. Calling
    `.predict_proba` on the pipeline runs the steps in order, and SMOTE is
    skipped at prediction time: it is a training-only step. That is why the
    app and the notebook cannot disagree about what "the model" is.
    """
    path = Path(path)
    if not path.exists():
        raise ModelNotAvailable(
            f"The model file is missing: {path.name}. "
            "Run scripts/export_model.py once to create it."
        )
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - the message matters more than the type
        raise ModelNotAvailable(
            f"The model file could not be read ({type(exc).__name__}). It was "
            "probably saved with a different scikit-learn version. Re-run "
            "scripts/export_model.py with the current environment."
        ) from exc


def predict_proba(pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Fraud probability for each row, as a plain float array in [0, 1].

    The columns are reordered to the notebook's order before scoring, so a file
    with the right columns in the wrong order still scores correctly.
    """
    if list(frame.columns) != config.FEATURES:
        frame = frame[config.FEATURES]
    if frame.empty:
        return np.empty(0, dtype=float)
    proba = pipeline.predict_proba(frame)[:, 1]
    return np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)


def confidence_band(probability: float) -> tuple[str, str, str]:
    """Plain-language reading of a probability, with a colour.

    Deliberately tied to the 0.50 cut-off and the notebook's finding that most
    probabilities sit near the two ends.
    """
    if probability >= 0.9:
        return "Very likely fraud", config.RED, "The model is strongly indicating fraud."
    if probability >= config.DEFAULT_THRESHOLD:
        return "Flagged as fraud", config.RED, "Above the chosen cut-off, so it would be blocked or reviewed."
    if probability >= 0.25:
        return "Uncertain", config.AMBER, "Below the cut-off, but not a confident pass."
    if probability >= 0.01:
        return "Likely legitimate", config.BLUE, "Below the cut-off, so no action would be taken."
    return "Very likely legitimate", config.BLUE, "The model sees almost nothing fraud-like here."


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% range for a measured rate, which stays honest when `total` is small.

    The naive rate `successes / total` on 53 rows looks far more certain than it
    is. The Wilson interval is the standard small-sample fix, and it never
    escapes [0, 1].
    """
    if total <= 0:
        return (0.0, 0.0)
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    half = (z * np.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
            / denominator)
    return (max(0.0, centre - half), min(1.0, centre + half))


def calibration_table(probabilities, labels, low: float = 0.5,
                      high: float = 0.9) -> list[dict]:
    """Measured fraud rate by score band, beside the model's own mean score.

    A calibrated model would show the two columns agreeing. This one does not:
    SMOTE balanced the training set, so the forest reports a share of trees as
    if half of all transactions were fraud. The ordering is sound and the
    numbers are inflated, which is exactly why the app shows this table instead
    of letting a visitor read a score as a percentage chance.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)
    bands = [
        (f"below {low:.2f}", probabilities < low),
        (f"{low:.2f} to {high:.2f}", (probabilities >= low) & (probabilities < high)),
        (f"{high:.2f} and above", probabilities >= high),
        ("all rows", np.ones(len(probabilities), dtype=bool)),
    ]

    rows = []
    for name, mask in bands:
        total = int(mask.sum())
        frauds = int((labels[mask] == 1).sum())
        if total == 0:
            continue
        low_ci, high_ci = wilson_interval(frauds, total)
        rows.append({
            "Score band": name,
            "Rows": total,
            "Frauds": frauds,
            "Mean score": float(probabilities[mask].mean()),
            "Actual fraud rate": frauds / total,
            "95% range": f"{low_ci:.1%} to {high_ci:.1%}",
        })
    return rows


def running_versions() -> dict[str, str]:
    """Versions of the libraries actually loaded in this process."""
    import imblearn

    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "imbalanced_learn": imblearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }


def version_report(trained: dict[str, str]) -> list[dict]:
    """Compare the versions the model was trained with to the ones running now.

    A pickle assumes the library that wrote it. Same versions means the model
    cannot quietly change behaviour between training and deployment, which is
    why requirements.txt pins them.
    """
    now = running_versions()
    rows = []
    for name, trained_version in trained.items():
        if name not in now:
            continue
        current = now[name]
        if name == "python":
            # A patch-level difference in Python is normal and harmless, so this
            # row is shown for information rather than as a pass/fail.
            ok = "informational"
        else:
            ok = "yes" if current == trained_version else "no"
        rows.append({
            "Library": name.replace("_", "-"),
            "Trained with": trained_version,
            "Running now": current,
            "Match": ok,
        })
    return rows


def unusual_values(frame: pd.DataFrame, reference: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Which of a row's 30 values are furthest from typical, using the demo sample.

    This is *not* the Random Forest's explanation: a forest does not expose
    per-row reasons in a form we can quote honestly. It answers a narrower
    question that the EDA can support: "are any of these values unusual?" A
    fraud row is typically an outlier in about 12 of the 28 V columns, against
    about 1 for a legitimate row, so this is a useful sanity check.
    """
    row = frame.iloc[0]
    mean = reference[config.FEATURES].mean()
    std = reference[config.FEATURES].std().replace(0, np.nan)
    z = ((row[config.FEATURES] - mean) / std).abs().dropna()
    low = reference[config.FEATURES].min()
    high = reference[config.FEATURES].max()

    out = []
    for feature, z_score in z.sort_values(ascending=False).head(top_n).items():
        out.append({
            "Input": feature,
            "Value": round(float(row[feature]), 4),
            "Typical (demo sample)": round(float(mean[feature]), 4),
            "How far out": round(float(z_score), 2),
            "Outside demo range": "yes" if not (low[feature] <= row[feature] <= high[feature]) else "no",
        })
    return pd.DataFrame(out)
