"""One set of input rules, used by both the single-transaction page and the batch page.

The single row form and the CSV upload go through the same function on purpose:
a rule that lives in two places eventually gets fixed in one of them. Everything
here raises `ValidationError` with a message written for a person, never a
traceback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


class ValidationError(ValueError):
    """A problem the user can fix, described in plain language."""

    def __init__(self, headline: str, problems: list[str] | None = None):
        self.headline = headline
        self.problems = problems or []
        super().__init__(headline)

    def as_markdown(self) -> str:
        text = f"**{self.headline}**"
        if self.problems:
            text += "\n\n" + "\n".join(f"- {problem}" for problem in self.problems)
        return text


@dataclass
class BatchResult:
    """Clean model inputs plus anything worth telling the user about."""

    features: pd.DataFrame
    labels: pd.Series | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return len(self.features)


def _resolve_columns(frame: pd.DataFrame, source: str) -> tuple[pd.DataFrame, list[str]]:
    """Match the file's columns to the model's inputs, case-insensitively.

    Returns the frame with tidy column names and a list of ignored columns.
    """
    ignored: list[str] = []

    # A CSV saved from a notebook often carries the index as its first column.
    first = frame.columns[0]
    if str(first).startswith("Unnamed:") and frame[first].is_monotonic_increasing:
        frame = frame.drop(columns=first)
        ignored.append("an unnamed index column")

    frame = frame.rename(columns=lambda c: str(c).strip())
    duplicated = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated:
        raise ValidationError(
            f"{source} has two columns with the same name.",
            [f"'{name}' appears more than once. Rename one of them." for name in duplicated],
        )

    lookup = {str(c).strip().lower(): c for c in frame.columns}

    rename: dict[str, str] = {}
    missing: list[str] = []
    for name in config.FEATURES:
        actual = lookup.get(name.lower())
        if actual is None:
            missing.append(name)
        else:
            rename[actual] = name
    # The label column is optional: it is only used to check the predictions.
    for name in config.ALLOWED_EXTRA_COLUMNS:
        actual = lookup.get(name.lower())
        if actual is not None:
            rename[actual] = name
    frame = frame.rename(columns=rename)

    if missing:
        raise ValidationError(
            f"{source} is missing {len(missing)} of the {len(config.FEATURES)} columns the model needs.",
            [
                "Missing: " + ", ".join(f"`{m}`" for m in missing[:12])
                + (" ..." if len(missing) > 12 else ""),
                "The model takes exactly these inputs: Time, V1 to V28, Amount.",
                "Use the demo file on this page as a template if that helps.",
            ],
        )

    extra = [c for c in frame.columns
             if c not in config.FEATURES and c not in config.ALLOWED_EXTRA_COLUMNS]
    if extra:
        raise ValidationError(
            f"{source} has {len(extra)} column(s) the model does not use.",
            [
                "Unexpected: " + ", ".join(f"`{c}`" for c in extra[:12])
                + (" ..." if len(extra) > 12 else ""),
                "Remove them and try again, or start from the demo file.",
            ],
        )

    return frame, ignored


def _check_numeric(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Convert everything to numbers, naming anything that is not one."""
    out = frame.copy()
    problems: list[str] = []

    for column in config.FEATURES:
        series = out[column]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            out[column] = series.astype("float64")
            continue
        converted = pd.to_numeric(series.astype(str).str.strip(), errors="coerce")
        bad = converted.isna() & ~series.isna()
        if bad.any():
            example = str(series[bad].iloc[0])[:40]
            rows = ", ".join(str(i + 1) for i in np.flatnonzero(bad)[:3])
            problems.append(
                f"`{column}` has text where a number is expected "
                f"({int(bad.sum())} row(s), first at row {rows}: '{example}')"
            )
        out[column] = converted

    if problems:
        raise ValidationError(f"{source} has values that are not numbers.", problems)
    return out


def _check_blank_and_finite(frame: pd.DataFrame, source: str) -> None:
    blank = frame[config.FEATURES].isna()
    if blank.any().any():
        per_column = blank.sum()
        per_column = per_column[per_column > 0]
        problems = [
            f"`{column}`: {int(count)} empty value(s)" for column, count in per_column.items()
        ]
        first_rows = np.flatnonzero(blank.any(axis=1))[:3] + 1
        problems.append("First affected rows: " + ", ".join(str(r) for r in first_rows))
        raise ValidationError(
            f"{source} has {int(blank.sum().sum())} empty value(s). The model cannot score a blank.",
            problems,
        )

    numeric = frame[config.FEATURES].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        column = config.FEATURES[int(np.argwhere(~np.isfinite(numeric))[0][1])]
        raise ValidationError(
            f"{source} contains an infinite value in `{column}` (inf or -inf).",
            ["Replace it with a real number or remove the row."],
        )


def _check_ranges(frame: pd.DataFrame, source: str) -> None:
    problems: list[str] = []
    for column in ("Time", "Amount"):
        negative = frame[column] < 0
        if negative.any():
            rows = ", ".join(str(i + 1) for i in np.flatnonzero(negative)[:3])
            problems.append(
                f"`{column}` is negative in {int(negative.sum())} row(s) "
                f"(first at row {rows}). Time and Amount cannot be below zero."
            )
    if problems:
        raise ValidationError(f"{source} has values that cannot be real.", problems)


def _check_labels(frame: pd.DataFrame, source: str) -> pd.Series | None:
    if config.LABEL_COLUMN not in frame.columns:
        return None
    labels = pd.to_numeric(frame[config.LABEL_COLUMN], errors="coerce")
    bad = labels.isna() | ~labels.isin([0, 1])
    if bad.any():
        rows = ", ".join(str(i + 1) for i in np.flatnonzero(bad)[:3])
        raise ValidationError(
            f"`{config.LABEL_COLUMN}` must be 0 (legitimate) or 1 (fraud).",
            [f"Row(s) {rows} hold something else.", "Remove the column if it is not the true label."],
        )
    return labels.astype(int)


def validate_batch(
    frame: pd.DataFrame,
    *,
    max_rows: int = config.MAX_BATCH_ROWS,
    source: str = "The file",
) -> BatchResult:
    """Check a whole table before the model sees it."""
    if frame is None or len(frame) == 0:
        raise ValidationError(
            f"{source} is empty.",
            ["It has a header but no rows, or no header at all."],
        )
    if len(frame) > max_rows:
        raise ValidationError(
            f"{source} has {len(frame):,} rows, which is more than this demo will score.",
            [
                f"Split it into files of {max_rows:,} rows or fewer.",
                "Scoring millions of rows needs a batch job, not a browser tab.",
            ],
        )

    frame, ignored = _resolve_columns(frame, source)
    frame = _check_numeric(frame, source)
    _check_blank_and_finite(frame, source)
    _check_ranges(frame, source)
    labels = _check_labels(frame, source)

    warnings: list[str] = []
    if ignored:
        warnings.append("Ignored " + " and ".join(ignored) + ".")
    if labels is None:
        warnings.append(
            f"No `{config.LABEL_COLUMN}` column, so the results cannot be checked against "
            "the truth. Predictions are still valid."
        )
    return BatchResult(features=frame[config.FEATURES].reset_index(drop=True),
                       labels=labels, warnings=warnings)


def validate_transaction(values: dict) -> pd.DataFrame:
    """Check one transaction typed or picked on the single-row page.

    Returns a one-row frame in the model's column order.
    """
    for name in config.FEATURES:
        if name not in values:
            raise ValidationError(f"The value for `{name}` is missing.")
        value = values[name]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            raise ValidationError(f"The value for `{name}` is empty.")

    frame = pd.DataFrame([{name: values[name] for name in config.FEATURES}])
    result = validate_batch(frame, source="This transaction")
    return result.features


def coerce_threshold(value) -> float:
    """Keep a threshold inside the range where it means something."""
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"'{value}' is not a number, so it cannot be a threshold.") from None
    if not math.isfinite(threshold):
        raise ValidationError("The threshold must be a normal number between 0 and 1.")
    return float(min(max(threshold, config.MIN_THRESHOLD), config.MAX_THRESHOLD))
