"""Small shared pieces of layout so every page looks like the same app."""

from __future__ import annotations

import streamlit as st

from . import config


def header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def metric_row(items: list[dict]) -> None:
    """`items` is a list of {label, value, help} dicts."""
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        column.metric(item["label"], item["value"], help=item.get("help"))


def provenance(results: dict, extra: str = "") -> None:
    """Say where the numbers came from. Small, at the bottom, always."""
    stamp = results.get("generated_at", "unknown")
    text = (f"Numbers computed {stamp} by `dashboard/scripts/export_model.py`. "
            "The notebook produced the decisions; this export re-ran them so the app "
            "cannot drift from it.")
    if extra:
        text += " " + extra
    st.caption(text)


def not_a_real_system() -> None:
    st.info(config.NOT_A_REAL_SYSTEM)


def decision_badge(is_fraud: bool, threshold: float) -> None:
    if is_fraud:
        st.error(
            f"**Flagged as fraud** at a cut-off of {threshold:.2f}. "
            "In a real system this transaction would be blocked or sent for review."
        )
    else:
        st.success(
            f"**Passed as legitimate** at a cut-off of {threshold:.2f}. "
            "No action would be taken."
        )
