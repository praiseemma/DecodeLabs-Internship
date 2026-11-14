"""
Fraud detection dashboard - the entry point.

Run it with:
    streamlit run app.py

Structure, in one sentence: `app.py` sets the frame (navigation, theme, a status
panel), `src/pages/` holds one file per screen, and `src/` holds the logic the
screens share. No page does its own model loading or its own validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `import src...` work no matter which folder the command was typed in.
# Streamlit Community Cloud starts the app from the repository root, so without
# this the imports would fail there but work locally - a nasty difference.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src import model, state, ui  # noqa: E402  (import after sys.path is set)

st.set_page_config(
    page_title="Fraud detection - DecodeLabs Project 2",
    page_icon=":material/credit_card:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Small amount of CSS: readable line length, tidy cards, and a footer we control.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1180px; }
      h1 { font-size: 1.9rem !important; letter-spacing: -0.01em; }
      div[data-testid="stMetric"] {
          background: #f7f9fd; border: 1px solid #e3e8f2;
          border-radius: 10px; padding: 0.7rem 0.9rem;
      }
      div[data-testid="stMetricLabel"] p { font-size: 0.82rem; color: #4b5563; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### Fraud detection")
        st.caption("DecodeLabs Data Science · Project 2 · Batch 2026")

        status = state.artifacts_status()
        with st.expander("Model and data status", expanded=False):
            if status["ready"]:
                info = state.results()
                trained = info["versions"]
                running = model.running_versions()
                sklearn_ok = running.get("scikit_learn") == trained.get("scikit_learn")
                st.markdown(
                    f"**Model:** Random Forest, `max_depth="
                    f"{info['model']['params']['classifier__max_depth']}`, "
                    f"{info['model']['n_estimators']} trees\n\n"
                    f"**File size:** {info['model']['size_kb']:,.0f} KB\n\n"
                    f"**Trained on:** {info['model']['trained_on_rows']:,} rows, "
                    f"seed {info['model']['random_seed']}\n\n"
                    f"**Test set:** {info['split']['test_rows']:,} rows, "
                    f"{info['split']['test_frauds']} frauds\n\n"
                    f"**Demo sample:** {info['demo']['rows']:,} rows "
                    f"({info['demo']['frauds']} frauds)"
                )
                if sklearn_ok:
                    st.caption("scikit-learn matches the version the model was saved with.")
                else:
                    st.warning("scikit-learn differs from the version used to train the model.")
            else:
                st.error("Missing: " + ", ".join(status["missing"]))

        st.divider()
        st.caption(
            "Blue means legitimate, red means fraud, everywhere in this app - "
            "the same colours as the notebook."
        )
        st.caption("Academic demo on a public dataset. Not a real fraud system.")


def setup_screen(status: dict) -> None:
    """Shown instead of the pages when the export has not been run yet."""
    ui.header("The dashboard has no model yet",
              "One command produces the four files this app reads.")
    st.error(f"Missing: {', '.join(status['missing'])}")
    st.markdown(
        "Open PowerShell in the `Project-2-praiseemma` folder and run:\n\n"
        "```powershell\n"
        "& \"$env:USERPROFILE\\anaconda3\\python.exe\" dashboard\\scripts\\export_model.py\n"
        "```\n\n"
        "It reads `data/creditcard.csv`, rebuilds the Random Forest with the notebook's "
        "settings, checks the test precision (0.6016) and recall (0.8105) come back, and "
        "writes the model, the results and a demo sample. It takes a few minutes."
    )
    st.caption(
        "The 150 MB dataset is deliberately not in the repository. It has to be in "
        "`Project-2-praiseemma/data/creditcard.csv` for this command to work."
    )
    st.stop()


status = state.artifacts_status()
if not status["ready"]:
    sidebar()
    setup_screen(status)

pages = [
    st.Page("src/pages/overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("src/pages/score.py", title="Score a transaction", icon=":material/search:",
            url_path="score"),
    st.Page("src/pages/batch.py", title="Batch scoring", icon=":material/upload_file:",
            url_path="batch"),
    st.Page("src/pages/threshold.py", title="Threshold explorer", icon=":material/tune:",
            url_path="threshold"),
    st.Page("src/pages/performance.py", title="Model performance",
            icon=":material/bar_chart:", url_path="performance"),
    st.Page("src/pages/insights.py", title="Data insights", icon=":material/insights:",
            url_path="insights"),
    st.Page("src/pages/about.py", title="About and limits", icon=":material/info:",
            url_path="about"),
]

sidebar()
st.navigation(pages).run()
