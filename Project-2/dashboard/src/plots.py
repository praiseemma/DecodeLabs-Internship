"""Plotly charts. Every figure here is a plain function of exported numbers.

Keeping them out of the pages means the same chart can be reused (the confusion
matrix appears on two pages) and the pages stay readable.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from . import config

FONT = dict(family="Helvetica Neue, Helvetica, Arial, sans-serif", size=13,
            color="#1f2937")


def _figure(height: int = 320, **kwargs) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=36, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_size=12),
        **kwargs,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#d5dbe7")
    fig.update_yaxes(gridcolor="#eef1f7", zeroline=False)
    return fig


def _as_percent(values) -> list[float]:
    """Turn histogram densities into "share of this class per band" percentages."""
    arr = np.asarray(values, dtype=float)
    total = arr.sum()
    return list(arr / total * 100) if total > 0 else list(arr)


# --------------------------------------------------------------------------
# Overview and data insights
# --------------------------------------------------------------------------
def class_balance(eda: dict) -> go.Figure:
    counts = eda["class_balance"]
    labels = [f"Legitimate<br>{counts['legitimate']:,}", f"Fraud<br>{counts['fraud']:,}"]
    fig = _figure(height=300)
    fig.add_bar(x=labels, y=[counts["legitimate"], counts["fraud"]],
                marker_color=[config.BLUE, config.RED], width=0.5,
                text=[f"{counts['legitimate']:,}", f"{counts['fraud']:,}"],
                textposition="outside", cliponaxis=False, showlegend=False)
    fig.update_yaxes(type="log", title="Transactions (log scale)", dtick=1)
    fig.update_xaxes(title=None)
    return fig


def amount_by_class(eda: dict) -> go.Figure:
    hist = eda["amount_histogram"]
    centres_log = [(left + right) / 2 for left, right in
                   zip(hist["bin_left"], hist["bin_right"])]
    amounts = [10 ** v - 1 for v in centres_log]
    fig = _figure(height=330)
    for key, colour, name in (("legitimate", config.BLUE, "Legitimate"),
                              ("fraud", config.RED, "Fraud")):
        fig.add_scatter(x=amounts, y=_as_percent(hist[key]), name=name,
                        mode="lines", line=dict(color=colour, width=2),
                        fill="tozeroy", fillcolor=_rgba(colour, 0.14))
    fig.update_xaxes(type="log", title="Amount")
    fig.update_yaxes(title="Share of each class (%)")
    return fig


def _rgba(hex_colour: str, alpha: float) -> str:
    value = hex_colour.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def hour_of_day(eda: dict) -> go.Figure:
    hist = eda["hour_histogram"]
    fig = _figure(height=310)
    for key, colour, name in (("legitimate", config.BLUE, "Legitimate"),
                              ("fraud", config.RED, "Fraud")):
        fig.add_scatter(x=hist["hour"], y=_as_percent(hist[key]), name=name,
                        mode="lines+markers", line=dict(color=colour, width=2),
                        marker=dict(size=5))
    fig.update_xaxes(title="Hour of day (from Time, seconds since the first transaction)",
                     dtick=3)
    fig.update_yaxes(title="Share of each class (%)")
    return fig


def correlations(eda: dict) -> go.Figure:
    rows = eda["top_correlations"]
    features = [r["feature"] for r in rows][::-1]
    values = [r["abs_correlation"] for r in rows][::-1]
    fig = _figure(height=340)
    fig.add_bar(x=values, y=features, orientation="h", marker_color=config.BLUE,
                text=[f"{v:.3f}" for v in values], textposition="outside",
                cliponaxis=False, showlegend=False)
    fig.update_xaxes(title="|correlation| with Class", range=[0, max(values) * 1.18])
    fig.update_yaxes(showgrid=False)
    return fig


def outlier_rate(eda: dict) -> go.Figure:
    rows = eda["outlier_rate_by_column"]
    features = [r["feature"] for r in rows]
    fig = _figure(height=560)
    fig.add_scatter(x=[r["legitimate"] for r in rows], y=features, mode="markers",
                    name="Legitimate", marker=dict(color=config.BLUE, size=7))
    fig.add_scatter(x=[r["fraud"] for r in rows], y=features, mode="markers",
                    name="Fraud", marker=dict(color=config.RED, size=7))
    for row in rows:
        fig.add_scatter(x=[row["legitimate"], row["fraud"]], y=[row["feature"]] * 2,
                        mode="lines", line=dict(color="#d5dbe7", width=1.5),
                        showlegend=False, hoverinfo="skip")
    fig.update_xaxes(title="% of rows that are IQR outliers in this column",
                     range=[0, 100])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=10))
    return fig


def outlier_columns_per_row(eda: dict) -> go.Figure:
    rows = eda["outlier_columns_per_row"]
    x = [r["n_columns"] for r in rows]
    fig = _figure(height=320)
    fig.add_bar(x=x, y=[r["legitimate"] for r in rows], name="Legitimate",
                marker_color=config.BLUE, opacity=0.75)
    fig.add_bar(x=x, y=[r["fraud"] for r in rows], name="Fraud",
                marker_color=config.RED, opacity=0.75)
    fig.update_layout(barmode="group", bargap=0.15)
    fig.update_xaxes(title="Number of V1 to V28 columns that are outliers in the same row",
                     dtick=2)
    fig.update_yaxes(title="% of rows in the class")
    return fig


# --------------------------------------------------------------------------
# Model performance
# --------------------------------------------------------------------------
def roc(curves: dict) -> go.Figure:
    fig = _figure(height=380)
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Coin flip (0.50)",
                    line=dict(color="#9aa3b2", dash="dash", width=1.5))
    for name, colour in (("Random Forest", config.RED), ("Logistic Regression", config.BLUE)):
        if name not in curves:
            continue
        curve = curves[name]["roc"]
        fig.add_scatter(x=curve["x"], y=curve["y"], mode="lines", name=name,
                        line=dict(color=colour, width=2.5))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate (recall)", range=[0, 1.01])
    return fig


def precision_recall(curves: dict, baseline_rate: float) -> go.Figure:
    fig = _figure(height=380)
    fig.add_scatter(x=[0, 1], y=[baseline_rate, baseline_rate], mode="lines",
                    name="Coin flip", line=dict(color="#9aa3b2", dash="dash", width=1.5))
    for name, colour in (("Random Forest", config.RED), ("Logistic Regression", config.BLUE)):
        if name not in curves:
            continue
        curve = curves[name]["pr"]
        fig.add_scatter(x=curve["x"], y=curve["y"], mode="lines",
                        name=f"{name} (AP {curves[name]['avg_precision']:.3f})",
                        line=dict(color=colour, width=2.5))
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1.01])
    return fig


def confusion(matrix: dict, title: str = "") -> go.Figure:
    """2x2 confusion matrix, coloured by how much of each true class it holds."""
    tn, fp, fn, tp = matrix["true_negatives"], matrix["false_alarms"], matrix["missed"], matrix["caught"]
    counts = [[tn, fp], [fn, tp]]
    shares = [[tn / max(tn + fn, 1) * 100, fp / max(tn + fn, 1) * 100],
              [fn / max(fn + tp, 1) * 100, tp / max(fn + tp, 1) * 100]]
    # One number per cell: a two-line label overflows neighbouring cells when the
    # window is narrow. The row share moves into the hover text instead.
    text = [[f"{value:,}" for value in row] for row in counts]
    custom = [[f"{share:.1f}% of this true class" for share in row] for row in shares]

    fig = _figure(height=300, title=dict(text=title, x=0.02, font=dict(size=14, color="#1f2937")))
    fig.add_heatmap(z=shares, x=["Predicted legitimate", "Predicted fraud"],
                    y=["Actually legitimate", "Actually fraud"],
                    colorscale=[[0, "#ffffff"], [1, "#b9cdf3"]], zmin=0, zmax=100,
                    text=text, texttemplate="%{text}", textfont=dict(size=15),
                    customdata=custom,
                    hovertemplate="%{y} / %{x}<br>%{text} rows (%{customdata})<extra></extra>",
                    showscale=False, xgap=3, ygap=3)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def feature_importance(importances: list[dict], top: int = 12) -> go.Figure:
    rows = importances[:top][::-1]
    features = [r["feature"] for r in rows]
    values = [r["importance"] for r in rows]
    fig = _figure(height=380)
    fig.add_bar(x=values, y=features, orientation="h", marker_color=config.BLUE,
                text=[f"{v:.3f}" for v in values], textposition="outside",
                cliponaxis=False, showlegend=False)
    fig.update_xaxes(title="How much the forest relied on the feature",
                     range=[0, max(values) * 1.2])
    fig.update_yaxes(showgrid=False)
    return fig


# --------------------------------------------------------------------------
# Threshold explorer and single scoring
# --------------------------------------------------------------------------
def threshold_curves(sweep: list[dict], chosen: float, f1_opt: float | None = None) -> go.Figure:
    thresholds = [r["threshold"] for r in sweep]
    fig = _figure(height=380)
    fig.add_scatter(x=thresholds, y=[r["precision"] for r in sweep], name="Precision",
                    line=dict(color=config.RED, width=2.5))
    fig.add_scatter(x=thresholds, y=[r["recall"] for r in sweep], name="Recall",
                    line=dict(color=config.BLUE, width=2.5))
    fig.add_scatter(x=thresholds, y=[r["f1"] for r in sweep], name="F1",
                    line=dict(color=config.GREEN, width=2, dash="dot"))
    fig.add_vline(x=config.DEFAULT_THRESHOLD, line=dict(color="#9aa3b2", dash="dash", width=1.5))
    fig.add_vline(x=chosen, line=dict(color="#111827", width=2))
    if f1_opt and abs(f1_opt - chosen) > 0.005:
        fig.add_vline(x=f1_opt, line=dict(color=config.GREEN, dash="dot", width=1.5))
    fig.add_annotation(x=chosen, y=1.04, yref="paper", text=f"chosen {chosen:.2f}",
                       showarrow=False, font=dict(size=11, color="#111827"))
    fig.update_xaxes(title="Threshold for calling a transaction fraud",
                     range=[0, 1], dtick=0.1)
    fig.update_yaxes(title="Score", range=[0, 1.02])
    return fig


def score_distribution(probabilities, labels, threshold: float) -> go.Figure:
    """How the model's probabilities split between the two true classes."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)
    bins = np.linspace(0, 1, 41)
    fig = _figure(height=330)
    for value, colour, name in ((0, config.BLUE, "Actually legitimate"),
                                (1, config.RED, "Actually fraud")):
        mask = labels == value
        counts, _ = np.histogram(probabilities[mask], bins=bins)
        fig.add_bar(x=(bins[:-1] + bins[1:]) / 2, y=counts, name=f"{name} ({mask.sum():,})",
                    marker_color=colour, opacity=0.8, width=0.024)
    fig.add_vline(x=threshold, line=dict(color="#111827", width=2))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Fraud probability the model gave", range=[0, 1], dtick=0.1)
    fig.update_yaxes(title="Transactions (log scale)", type="log", dtick=1)
    return fig


def probability_bar(probability: float, threshold: float) -> go.Figure:
    """A single readable bar: the score, and where the cut-off sits."""
    probability = float(min(max(probability, 0.0), 1.0))
    colour = config.RED if probability >= threshold else config.BLUE
    text_colour = "white" if probability > 0.35 else "#111827"
    fig = _figure(height=110)
    fig.add_bar(x=[1.0], y=[""], orientation="h", marker_color="#eef1f7",
                hoverinfo="skip", showlegend=False, width=0.5)
    fig.add_bar(x=[probability], y=[""], orientation="h", marker_color=colour,
                text=[f"{probability:.1%}"], textposition="inside",
                insidetextanchor="end", textfont=dict(color=text_colour, size=13),
                cliponaxis=False,
                hovertemplate="Fraud probability %{x:.1%}<extra></extra>",
                showlegend=False, width=0.5)
    fig.add_vline(x=threshold, line=dict(color="#111827", width=2))
    fig.add_annotation(x=threshold, y=1.22, yref="paper",
                       text=f"cut-off {threshold:.2f}", showarrow=False,
                       font=dict(size=11, color="#111827"))
    fig.update_xaxes(range=[0, 1], dtick=0.1, showticklabels=True)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=10, r=10, t=28, b=10), bargap=0)
    return fig
