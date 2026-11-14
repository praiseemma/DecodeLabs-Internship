"""Shared settings for the fraud dashboard.

Everything that another file might need to agree on lives here, in one place:
file paths, the 30 model inputs, the colours, and the display limits.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Paths are built from this file's own location, never from the folder the
# command was typed in. That is what makes `streamlit run` and Streamlit
# Community Cloud (which starts the app from the repository root) behave the
# same way.
SRC_DIR = Path(__file__).resolve().parent
APP_DIR = SRC_DIR.parent                     # Project-2/dashboard
ARTIFACTS_DIR = APP_DIR / "artifacts"
DATA_DIR = APP_DIR / "data"

RESULTS_PATH = ARTIFACTS_DIR / "results.json"
MODEL_PATH = ARTIFACTS_DIR / "rf_fraud_pipeline.joblib"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.npz"
DEMO_PATH = DATA_DIR / "demo_transactions.csv"

# --------------------------------------------------------------------------
# The model's inputs, in the order the notebook trained them
# --------------------------------------------------------------------------
FEATURES: list[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
V_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]
LABEL_COLUMN = "Class"

# Columns a batch file may carry that are not model inputs.
# `Class` is the true label: optional, and only used to score the results.
ALLOWED_EXTRA_COLUMNS = [LABEL_COLUMN]

# --------------------------------------------------------------------------
# Guards (the app refuses to work outside these)
# --------------------------------------------------------------------------
MAX_BATCH_ROWS = 20_000          # scoring more than this in a browser tab is a bad idea
MAX_UPLOAD_MB = 10.0             # the Kaggle export of 20k rows is about 6 MB
MAX_CSV_PREVIEW_ROWS = 200       # rows shown in the results table

DEFAULT_THRESHOLD = 0.50
MIN_THRESHOLD = 0.01
MAX_THRESHOLD = 0.99
THRESHOLD_STEP = 0.01

# --------------------------------------------------------------------------
# Looks: same blue and red as the notebook (blue = legitimate, red = fraud)
# --------------------------------------------------------------------------
BLUE = "#1d5fd1"
RED = "#d1432d"
GREY = "#6b7280"
GREEN = "#2e7d32"
AMBER = "#b26a00"

COLOURS = {0: BLUE, 1: RED}
CLASS_NAMES = {0: "Legitimate", 1: "Fraud"}

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

# --------------------------------------------------------------------------
# Wording reused across pages
# --------------------------------------------------------------------------
TEST_SET_WARNING = (
    "**Training data only.** A threshold must be chosen on training data. "
    "Picking it on the test set would mean the test score is no longer an "
    "honest measure of how the model behaves on data it has never seen."
)

NOT_A_REAL_SYSTEM = (
    "This is an academic demo on a public dataset. It reads anonymised "
    "components (V1 to V28), not raw card data, so it cannot be pointed at a "
    "real payment system."
)
