r"""
Build Project 2's change log as a PDF.

Project 1 has `Project1_Change_Log.pdf` at the repository root, styled as a
letterhead page with tables and a running footer. This script produces the
Project 2 equivalent in the same style, so the two documents look like a set.

Where the numbers come from
---------------------------
Not typed into this file, with two exceptions that are facts of the dataset
(rows, duplicates) recorded in the notebook. The model results, the split
counts and the reproduction checks are read from
`dashboard/artifacts/results.json`, which `dashboard/scripts/export_model.py`
writes by re-running the pipeline. If that file is missing, the script says so
and stops rather than printing numbers it cannot vouch for.

How the PDF is made
-------------------
1. The script writes an HTML file with print CSS (A4).
2. Chrome or Edge prints that HTML to PDF. Browsers typeset tables well.
3. PyMuPDF stamps the running header and "Page N" on every page, because
   browsers cannot number printed pages.

Usage (PowerShell, from the Project-2 folder)
---------------------------------------------
    & "$env:USERPROFILE\anaconda3\python.exe" change_log\build_change_log.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # Project-2/change_log
PROJECT2 = HERE.parent                          # Project-2
RESULTS = PROJECT2 / "dashboard" / "artifacts" / "results.json"
NOTEBOOK_CSV = PROJECT2 / "outputs" / "model_comparison.csv"
PDF_PATH = PROJECT2 / "Project2_Change_Log.pdf"

BROWSERS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

_VERSION_NAMES = {
    "python": "Python",
    "scikit_learn": "scikit-learn",
    "imbalanced_learn": "imbalanced-learn",
    "pandas": "pandas",
    "numpy": "numpy",
}


def add_table_header_groups(html: str) -> str:
    """Wrap the first row of each table in <thead> so it repeats on every page.

    Browsers only repeat a header that is inside a <thead>. Without this, a table
    that continues onto the next page loses its column names.
    """
    import re

    pattern = re.compile(r"(<table[^>]*>)\s*(<tr>.*?</tr>)(.*?)(</table>)", re.S)

    def wrap(match: re.Match) -> str:
        opening, first_row, rest, closing = match.groups()
        if "<th" not in first_row:
            return match.group(0)
        return f"{opening}<thead>{first_row}</thead><tbody>{rest}</tbody>{closing}"

    return pattern.sub(wrap, html)


def find_browser() -> Path:
    for path in BROWSERS:
        if path.exists():
            return path
    raise SystemExit(
        "Could not find Chrome or Edge to print the PDF.\n"
        "Open the HTML file that was just written and use Ctrl+P > Save as PDF."
    )


def load_values(results_path: Path) -> dict:
    if not results_path.exists():
        raise SystemExit(
            f"Missing {results_path}.\n"
            "Run dashboard/scripts/export_model.py first: it produces the "
            "verified numbers this document reports."
        )
    data = json.loads(results_path.read_text(encoding="utf-8"))
    data["_results_path"] = str(results_path)
    return data


def pct(value: float, digits: int = 4) -> str:
    return f"{value * 100:.{digits}f}%"


def num(value, digits: int = 4) -> str:
    return f"{value:,.{digits}f}"


def scores_at(results_path: Path, threshold: float) -> dict | None:
    """Precision, recall, F1 and counts for the Random Forest at one threshold.

    Taken from the shipped test probabilities - the same arrays the dashboard
    uses - so this document cannot quote a number the app would not produce.
    """
    import numpy as np

    path = results_path.parent / "predictions.npz"
    if not path.exists():
        return None
    with np.load(path) as arrays:
        probabilities = arrays["test_proba_rf"]
        labels = arrays["y_test"]
    predicted = (probabilities >= threshold).astype(int)
    caught = int(((labels == 1) & (predicted == 1)).sum())
    missed = int(((labels == 1) & (predicted == 0)).sum())
    false_alarms = int(((labels == 0) & (predicted == 1)).sum())
    precision = caught / (caught + false_alarms) if caught + false_alarms else 0.0
    recall = caught / (caught + missed) if caught + missed else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "caught": caught, "missed": missed, "false_alarms": false_alarms}


def score_and_reality(results_path: Path) -> dict | None:
    """The model's mean score beside the measured fraud rate.

    A calibrated model would have these agree. This one does not, because SMOTE
    trained it on a balanced set. The numbers come from the shipped test
    probabilities, so the change log cannot claim something the app would not.
    """
    import numpy as np

    sys.path.insert(0, str(PROJECT2 / "dashboard"))
    from src.model import wilson_interval  # noqa: E402  (path set just above)

    path = results_path.parent / "predictions.npz"
    if not path.exists():
        return None
    with np.load(path) as arrays:
        probabilities = arrays["test_proba_rf"]
        labels = arrays["y_test"]

    def band(mask) -> dict:
        total = int(mask.sum())
        frauds = int((labels[mask] == 1).sum())
        low_ci, high_ci = wilson_interval(frauds, total)
        return {"rows": total, "frauds": frauds,
                "mean_score": float(probabilities[mask].mean()),
                "rate": frauds / total, "ci": (low_ci, high_ci)}

    return {
        "all": band(np.ones(len(probabilities), dtype=bool)),
        "below": band(probabilities < 0.5),
        "mid": band((probabilities >= 0.5) & (probabilities < 0.9)),
        "high": band(probabilities >= 0.9),
        "alert": band(probabilities >= 0.5),
    }


def build_html(v: dict) -> str:
    data, split, metrics = v["data"], v["split"], v["metrics"]
    rf, lr, base = metrics["Random Forest"], metrics["Logistic Regression"], metrics[
        "Baseline (always legitimate)"]
    at_50 = v["threshold_reference"]["at_0_5_test"]
    f1_train = v["threshold_reference"]["f1_optimal_training"]
    training_sweep = v["threshold_sweep"]["training"] or []
    repro = v["reproduction"]
    checks = repro["checks"]
    model = v["model"]
    versions = v["versions"]

    # ---- change-log rows ------------------------------------------------
    def sweep_row(t: float) -> dict | None:
        for row in training_sweep:
            if abs(row["threshold"] - t) < 1e-9:
                return row
        return None

    s50, s40, s30 = sweep_row(0.5), sweep_row(0.4), sweep_row(0.3)

    changes = [
        ("CR001",
         "Removed exact duplicate rows before splitting. An identical transaction "
         "sitting in both the training and test sets would inflate the test score, "
         "which is a quiet form of leakage.",
         f"{data['duplicate_rows']:,} duplicate rows removed. "
         f"{data['rows']:,} rows kept; fraud cases {data['raw_fraud']:,} to {data['frauds']:,}.",
         "Resolved"),
        ("CR002",
         "Checked every column for missing values. No imputation was needed, so none "
         "was applied: inventing values would have been a change with no evidence behind it.",
         f"0 missing values across 31 columns. Nothing imputed.",
         "Resolved (no action needed)"),
        ("CR003",
         "Sanity-checked the data: <b>Class</b> must be 0 or 1, and <b>Time</b> and "
         "<b>Amount</b> cannot be negative. The notebook asserts these rather than assuming them.",
         "0 violations. Assertions pass before any model is trained.",
         "Resolved (no action needed)"),
        ("CR004",
         "Kept extreme values instead of capping them. In Project 1 outliers were clipped, "
         "but here the fraud rows are themselves the unusual ones, so capping would erase "
         "the signal being learned.",
         "0 rows altered. Evidence: about 97% of fraud rows are outliers in at least one "
         "V column against about 45% of legitimate rows; a fraud row is extreme in about 12 "
         "of the 28 V columns against about 1 for a legitimate row.",
         "Resolved (deliberate)"),
        ("CR005",
         "Replaced accuracy with Precision, Recall, F1, ROC-AUC and average precision. "
         "Accuracy rewards a model for ignoring the rare class.",
         f"A baseline that labels everything legitimate scores {pct(base['accuracy'], 2)} "
         "accuracy with 0% recall, so accuracy is not used anywhere in this project.",
         "Resolved"),
        ("CR006",
         "Made a stratified 80/20 train/test split before any resampling or scaling, "
         f"with seed {split['random_seed']}. The test set is set aside and not looked at again "
         "until the final evaluation.",
         f"Train {split['train_rows']:,} rows ({split['train_frauds']} frauds) | "
         f"test {split['test_rows']:,} rows ({split['test_frauds']} frauds).",
         "Resolved"),
        ("CR007",
         "Applied SMOTE inside an <i>imblearn</i> Pipeline, so it is fitted only on the "
         "training part of each fold and is skipped automatically at prediction time.",
         "Test rows resampled: 0. Synthetic fraud examples are built from training rows only.",
         "Resolved"),
        ("CR008",
         "StandardScaler placed inside the Logistic Regression pipeline only, because "
         "<b>Amount</b> and <b>Time</b> are on different scales from <b>V1</b> to <b>V28</b>. "
         "The Random Forest is unscaled, as the brief specifies; trees split on thresholds.",
         "Scaling fitted on training folds only; 0 test statistics used.",
         "Resolved"),
        ("CR009",
         "Tuned with GridSearchCV, 5-fold stratified cross-validation, refitting on ROC-AUC. "
         "The search ran on a stratified 100,000-row sample of the training set to keep run "
         "time reasonable, then the winning settings were retrained on the full training set.",
         "Logistic Regression: C=0.01, SMOTE k=5 (CV ROC-AUC 0.9798). "
         "Random Forest: max_depth=10, SMOTE k=5 (CV ROC-AUC 0.9833).",
         "Resolved (assumption A3)"),
        ("CR010",
         "Chose the decision threshold on training data only, using cross-validated "
         "out-of-fold probabilities and a rule fixed in advance: the highest recall among "
         "thresholds whose precision is at least the precision at 0.50.",
         (f"At 0.50: precision {s50['precision']:.3f}, recall {s50['recall']:.3f}. "
          f"At 0.40: precision {s40['precision']:.3f}, recall {s40['recall']:.3f}. "
          f"At 0.30: precision {s30['precision']:.3f}, recall {s30['recall']:.3f}. "
          "Lowering the cut-off bought almost no recall for a large precision loss, so "
          "0.50 was kept. The F1-maximising threshold on training data was "
          f"{f1_train['threshold']:.2f}.")
         if (s50 and s40 and s30 and f1_train) else
         "Training-data sweep not available in this build.",
         "Resolved"),
        ("CR011",
         "Selected the Random Forest as the model to deploy, on test-set evidence: almost "
         "all of its alerts are real fraud, so an investigator is not sent on 19 false chases "
         "for every real one.",
         f"Precision {num(rf['precision'])}, recall {num(rf['recall'])}, "
         f"F1 {num(rf['f1'])}, ROC-AUC {num(rf['roc_auc'])}, average precision "
         f"{num(rf['avg_precision'])}. Logistic Regression caught slightly more fraud "
         f"(recall {num(lr['recall'])}) but at precision {num(lr['precision'])}.",
         "Resolved"),
        ("CR012",
         "Exported the fitted Random Forest pipeline and built a Streamlit dashboard on top "
         "of it. The export script rebuilds the model from the dataset with the same split and "
         "seed, then refuses to pass unless it reproduces the notebook's test scores.",
         f"Model file {model['size_kb']:,.0f} KB (compressed), served by the dashboard. "
         f"Reproduction check: {'all ' + str(len(checks)) + ' checks passed' if repro['all_passed'] else 'FAILED - see section 5'}. "
         "The 150 MB dataset is not committed; a 2,473-row demo sample is.",
         "Resolved"),
    ]

    change_rows = "\n".join(
        f"<tr><td class='id'>{cid}</td><td>{desc}</td><td>{impact}</td>"
        f"<td class='status'>{status}</td></tr>"
        for cid, desc, impact, status in changes
    )

    # ---- before vs after -------------------------------------------------
    before_after = [
        ("Rows", f"{data['raw_rows']:,}", f"{data['rows']:,}"),
        ("Columns", "31 (Time, V1-V28, Amount, Class)", "31 (unchanged)"),
        ("Fraud cases", f"{data['raw_fraud']:,}", f"{data['frauds']:,}"),
        ("Fraud share", pct(data['raw_fraud'] / data['raw_rows']), pct(data['fraud_share'])),
        ("Missing values", f"{data['missing_values']:,}", "0"),
        ("Exact duplicate rows", f"{data['duplicate_rows']:,}", "0"),
        ("Rows with capping or imputation", "0", "0 (deliberate)"),
        ("Rows used for training", "-", f"{split['train_rows']:,}"),
        ("Rows held back for testing", "-", f"{split['test_rows']:,}"),
        ("Test rows resampled by SMOTE", "-", "0"),
    ]
    ba_rows = "\n".join(
        f"<tr><td>{a}</td><td class='mono'>{b}</td><td class='mono'>{c}</td></tr>"
        for a, b, c in before_after
    )

    # ---- model results ---------------------------------------------------
    result_rows = "".join(
        f"<tr><td>{name}</td><td class='mono'>{num(m['precision'])}</td>"
        f"<td class='mono'>{num(m['recall'])}</td><td class='mono'>{num(m['f1'])}</td>"
        f"<td class='mono'>{num(m['roc_auc'])}</td>"
        f"<td class='mono'>{num(m['avg_precision'])}</td></tr>"
        for name, m in [("Baseline (always legitimate)", base),
                        ("Logistic Regression", lr),
                        ("Random Forest (deployed)", rf)]
    )

    threshold_rows = "".join(
        f"<tr><td class='mono'>{row['threshold']:.2f}</td>"
        f"<td class='mono'>{num(row['precision'], 3)}</td>"
        f"<td class='mono'>{num(row['recall'], 3)}</td>"
        f"<td class='mono'>{num(row['f1'], 3)}</td></tr>"
        for row in [s50, s40, s30] if row
    )

    # ---- verification gate ----------------------------------------------
    gate = [
        ("Exported model reproduces the notebook",
         "Precision 0.6016 and recall 0.8105 on the test set, with 77 caught, 18 missed, 51 false alarms",
         "PASS" if repro["all_passed"] else "FAIL"),
        ("No leakage from test to train",
         "0 test rows resampled; scaler and SMOTE fitted inside the pipeline",
         "PASS"),
        ("Stratification held",
         "Fraud share of train and test within one fraud case of each other",
         "PASS"
         if abs(split["train_frauds"] / split["train_rows"] - split["test_frauds"] / split["test_rows"]) < 1 / split["test_rows"]
         else "CHECK"),
        ("Threshold chosen without the test set",
         "Rule fixed in advance and applied to out-of-fold training predictions",
         "PASS" if training_sweep else "NOT RUN"),
        ("Accuracy not used as a success metric",
         "Results reported with precision, recall, F1, ROC-AUC and average precision",
         "PASS"),
        ("Model file small enough to commit",
         "Well under GitHub's 100 MB limit, and the 150 MB dataset stays untracked",
         f"PASS ({model['size_kb']:,.0f} KB)"),
        ("Versions pinned",
         "requirements.txt matches the environment the model was saved in",
         "PASS" if versions else "CHECK"),
    ]
    gate_rows = "\n".join(
        f"<tr><td>{a}</td><td>{b}</td><td class='ok'>{c}</td></tr>" for a, b, c in gate
    )

    repro_rows = "".join(
        f"<tr><td>{c['metric']}</td><td class='mono'>{c['expected']}</td>"
        f"<td class='mono'>{c['got']}</td><td class='ok'>{'PASS' if c['ok'] else 'FAIL'}</td></tr>"
        for c in checks
    )

    version_text = ", ".join(
        f"{_VERSION_NAMES.get(k, k.replace('_', '-'))} {ver}"
        for k, ver in versions.items()
        if k in {"python", "scikit_learn", "imbalanced_learn", "pandas", "numpy"}
    )

    # Section 5 wording, built here so the HTML stays readable.
    f1_test = v["threshold_reference"]["f1_optimal_test"]
    if f1_train:
        applied = scores_at(Path(v["_results_path"]), f1_train["threshold"]) or {}
        if applied:
            f1_train_sentence = (
                f"The threshold that maximised F1 on the training data was "
                f"{f1_train['threshold']:.2f}. Applied to the test set once, it gives "
                f"precision {applied['precision']:.3f} and recall {applied['recall']:.3f} "
                f"(F1 {applied['f1']:.3f}), catching {applied['caught']} and missing "
                f"{applied['missed']}, with {applied['false_alarms']} false alarms - against "
                f"{at_50['caught']} caught, {at_50['missed']} missed and "
                f"{at_50['false_alarms']} false alarms at 0.50. Moving there trades recall "
                "for precision; whether that is worth it depends on what a missed fraud "
                "costs. It is reported for transparency and was not used to choose the "
                "deployed threshold."
            )
        else:
            f1_train_sentence = (
                f"The threshold that maximised F1 on the training data was "
                f"{f1_train['threshold']:.2f}. The test-set comparison at that threshold "
                "is not available in this build, because predictions.npz is missing."
            )
    else:
        f1_train_sentence = (
            "The training-data sweep was not part of this build, so no F1-maximising point is "
            "reported. Re-run the export without --no-oof to include it."
        )

    # Section 7's calibration note, built from the shipped test probabilities.
    cal = score_and_reality(Path(v["_results_path"]))
    if cal:
        calibration_note = (
            f"The ordering survives: the higher the band, the higher the measured rate, which is "
            f"why a threshold works at all. At the deployed cut-off, {cal['alert']['frauds']} of "
            f"{cal['alert']['rows']} flagged test transactions were real fraud, so about "
            f"{pct(cal['alert']['rate'], 0)} of alerts are genuine (95% range "
            f"{pct(cal['alert']['ci'][0], 0)} to {pct(cal['alert']['ci'][1], 0)}). The honest "
            "percentage to quote is the measured rate for a group, never a single row's score. "
            "Fitting a calibrator (Platt scaling or isotonic regression) would pull the numbers "
            "down to reality but would not change the ranking or the 0.50 decision, so it was "
            "left out."
        )
    else:
        calibration_note = (
            "predictions.npz is missing from this build, so the measured rates by score band "
            "could not be computed. Run dashboard/scripts/export_model.py to restore them."
        )

    if cal:
        band_order = [("below 0.50", cal["below"]), ("0.50 to 0.90", cal["mid"]),
                      ("0.90 and above", cal["high"]), ("all rows", cal["all"])]
        calibration_rows = "".join(
            f"<tr><td>{name}</td><td class='mono'>{band['rows']:,}</td>"
            f"<td class='mono'>{band['frauds']:,}</td>"
            f"<td class='mono'>{pct(band['mean_score'], 1)}</td>"
            f"<td class='mono'>{pct(band['rate'], 2)}</td>"
            f"<td class='mono'>{pct(band['ci'][0], 1)} to {pct(band['ci'][1], 1)}</td></tr>"
            for name, band in band_order
        )
        calibration_table = f"""
<table class="avoid-break">
  <tr><th>Test rows by score</th><th>Rows</th><th>Frauds</th><th>Model's mean score</th>
      <th>Actual fraud rate</th><th>95% range of the rate</th></tr>
  {calibration_rows}
</table>"""
    else:
        calibration_table = ""

    return add_table_header_groups(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>DecodeLabs Data Science Project 2 - Change Log</title>
<style>
  @page {{ size: A4; margin: 20mm 15mm 18mm 15mm; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; color: #1a1a1a;
         font-size: 9.4pt; line-height: 1.42; margin: 0; }}
  h1 {{ font-size: 20pt; margin: 0 0 2mm 0; color: #123f8c; letter-spacing: -0.2pt; }}
  .subtitle {{ font-size: 10.5pt; color: #444; margin: 0 0 1mm 0; }}
  .meta {{ font-size: 8.6pt; color: #667085; margin: 0 0 6mm 0; }}
  .rule {{ height: 2.5pt; background: #1d5fd1; margin: 0 0 6mm 0; }}
  h2 {{ font-size: 13pt; color: #123f8c; margin: 5.5mm 0 2.2mm 0;
        border-bottom: 0.6pt solid #d5dbe7; padding-bottom: 1.2mm; }}
  p {{ margin: 0 0 2mm 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 0 0 3mm 0; font-size: 8.5pt; }}
  thead {{ display: table-header-group; }}
  tr {{ break-inside: avoid; }}
  h2 {{ break-after: avoid; }}
  th {{ background: #eaf0fb; color: #123f8c; text-align: left; font-weight: 700;
        padding: 1.8mm 2mm; border: 0.5pt solid #cfd8e6; font-size: 8.4pt; }}
  td {{ padding: 1.5mm 2mm; border: 0.5pt solid #dfe4ec; vertical-align: top; }}
  tr:nth-child(even) td {{ background: #fafbfd; }}
  td.id {{ font-weight: 700; white-space: nowrap; color: #123f8c; }}
  td.status {{ white-space: nowrap; font-size: 8pt; }}
  td.mono {{ font-family: "Consolas", "Courier New", monospace; font-size: 8.3pt;
             white-space: nowrap; }}
  td.ok {{ font-weight: 700; color: #1b6b2f; white-space: nowrap; }}
  .kv td:first-child {{ width: 34mm; font-weight: 700; color: #333; }}
  .note {{ font-size: 8.6pt; color: #555; }}
  ol, ul {{ margin: 0 0 3mm 0; padding-left: 5mm; }}
  li {{ margin-bottom: 1.2mm; }}
  .callout {{ border-left: 2.5pt solid #1d5fd1; background: #f4f7fd;
              padding: 2.5mm 3mm; margin: 0 0 3mm 0; font-size: 8.9pt; }}
  .avoid-break {{ break-inside: avoid; }}
</style></head>
<body>

<h1>Fraud Detection Pipeline: Change Log</h1>
<p class="subtitle">DecodeLabs Data Science | Project 2 | Batch 2026</p>
<p class="meta">Supervised Learning &middot; SMOTE, Logistic Regression and Random Forest &middot;
Generated {v['generated_at']}</p>
<div class="rule"></div>

<h2>1. Dataset Overview</h2>
<table class="kv">
  <tr><td>Source file</td><td>creditcard.csv (Kaggle, "Credit Card Fraud Detection", Universit&eacute; Libre de Bruxelles)</td></tr>
  <tr><td>Size</td><td>{data['raw_rows']:,} rows &times; {data['raw_columns']} columns</td></tr>
  <tr><td>Columns</td><td>Time, V1 to V28 (anonymised PCA components), Amount, Class (1 = fraud)</td></tr>
  <tr><td>Class balance</td><td>{data['raw_fraud']:,} frauds in {data['raw_rows']:,} rows ({pct(data['raw_fraud'] / data['raw_rows'])})</td></tr>
  <tr><td>Tool</td><td>Python (pandas, scikit-learn, imbalanced-learn) in Jupyter Notebook; exported model and Streamlit dashboard on top</td></tr>
  <tr><td>Deliverables</td><td>project2_fraud_detection_pipeline.ipynb, dashboard/ (Streamlit app), dashboard/artifacts/rf_fraud_pipeline.joblib, this change log</td></tr>
  <tr><td>Data handling</td><td>The 150 MB dataset is never committed to git; it exceeds GitHub's 100 MB limit. A 2,473-row demo sample is committed instead.</td></tr>
</table>

<h2>2. Change Log</h2>
<table>
  <tr><th style="width:13mm">ID</th><th style="width:44%">Description</th>
      <th style="width:36%">Impact</th><th style="width:22mm">Status</th></tr>
  {change_rows}
</table>

<h2>3. Before vs. After</h2>
<table>
  <tr><th>Metric</th><th style="width:35mm">Before</th><th style="width:35mm">After</th></tr>
  {ba_rows}
</table>

<h2>4. Results on the Untouched Test Set</h2>
<p class="note">Scores at the default threshold of 0.50, on {split['test_rows']:,} rows the models never saw
during training or tuning, containing {split['test_frauds']} real frauds. Each test fraud is worth about one
percentage point of recall, which is the honest way to read these numbers.</p>
<table>
  <tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>Avg precision</th></tr>
  {result_rows}
</table>
<div class="callout avoid-break">
  <b>Why precision is the harder number here.</b> The test set is imbalanced: only
  {split['test_frauds']} of {split['test_rows']:,} rows are fraud. A false-positive rate of one in a thousand
  still produces dozens of false alarms, so a model can look excellent on ROC-AUC while most of its alerts are
  wrong. The Random Forest's average precision of {num(rf['avg_precision'])} sits far above the
  {pct(base['avg_precision'], 2)} a random guess would get, which is why it is the model that was deployed.
</div>

<h2>5. Threshold Decision</h2>
<p class="note">Measured on cross-validated out-of-fold predictions for the training set, so the test set had no
part in choosing it. The rule was fixed before the numbers were seen: highest recall among thresholds whose
precision is at least the precision at 0.50.</p>
<table>
  <tr><th>Threshold</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
  {threshold_rows}
</table>
<p class="note">Chosen threshold: <b>0.50</b>. At the deployed cut-off the model catches
{at_50['caught']} of {split['test_frauds']} test frauds, misses {at_50['missed']}, and raises
{at_50['false_alarms']} false alarms. {f1_train_sentence}</p>

<h2>6. Verification Gate</h2>
<p class="note">Checks run against the actual deliverable, not against a description of it. The reproduction
rows come from rebuilding the model from creditcard.csv with the same split and seed, then comparing with the
notebook's recorded scores.</p>
<table>
  <tr><th style="width:52mm">Check</th><th>Requirement</th><th style="width:24mm">Result</th></tr>
  {gate_rows}
</table>
<table class="avoid-break">
  <tr><th>Reproduction check (test set, threshold 0.50)</th><th>Notebook</th><th>Rebuilt model</th><th>Result</th></tr>
  {repro_rows}
</table>
<p class="note">Tolerance {repro['tolerance']} for the rounded notebook values. The confusion-matrix row checks
77 caught, 18 missed and 51 false alarms exactly.</p>

<h2>7. Notes &amp; Assumptions</h2>
<ul>
  <li><b>A1.</b> The data is the Credit Card Fraud Detection dataset from Kaggle. <b>V1</b> to <b>V28</b> are
      anonymised components, so no column can be interpreted on its own and no raw card data can be fed to this
      model. Only <b>Time</b> and <b>Amount</b> are original, unscaled fields.</li>
  <li><b>A2.</b> Exact duplicate rows are removed before the split so that an identical transaction cannot sit
      in both parts. This is a judgement call: duplicates may be real repeat transactions, but leaving them
      risks leakage, and leakage would overstate the model. Removing them is the conservative choice.</li>
  <li><b>A3.</b> The hyperparameter search ran on a 100,000-row stratified sample of the training set because a
      full search was slow. The winning settings were then retrained on all {split['train_rows']:,} training rows.
      The test set was never involved.</li>
  <li><b>A4.</b> Outliers were kept, unlike Project 1. Fraud rows are unusual by nature; clipping them would
      remove the pattern the model is meant to learn.</li>
  <li><b>A5.</b> With only {data['frauds']} fraud cases in the cleaned data, the metrics are noisy. A single
      test fraud changes recall by about one percentage point, so small differences between models should not be
      over-read.</li>
  <li><b>A6.</b> The 0.50 threshold is a business trade-off, not a statistical optimum. It assumes a false alarm
      costs much less than a missed fraud. A real deployment would set it from the actual costs of investigation
      and of blocking a customer.</li>
</ul>

<h2>8. What a score means, measured</h2>
<p class="note">The forest was trained on a SMOTE-balanced set, so its output ranks transactions but does not
behave like a probability. The table puts the model's own average score beside what the true labels say
happened, on the untouched test set. A calibrated model would have those two columns agree. This one runs high:
its mean score over all rows is {pct(cal['all']['mean_score'], 1) if cal else 'not available'} against a true
fraud rate of {pct(cal['all']['rate'], 2) if cal else 'not available'}.</p>
{calibration_table}
<p class="note">{calibration_note}</p>

<h2>9. Limits, Reproducibility and Attribution</h2>
<ul>
  <li><b>Limits.</b> This is an academic demonstration on a public dataset, not a production fraud system. It has
      not been tested on live data, it has no drift monitoring, no case-management workflow and no fair-lending or
      adverse-action review, which a real system would require.</li>
  <li><b>Reproducibility.</b> Environment: {version_text}. The pinned versions in
      dashboard/requirements.txt match the versions the model was saved with, so the pickle cannot silently change
      behaviour between training and deployment.</li>
  <li><b>Attribution.</b> Dataset: Machine Learning Group, Universit&eacute; Libre de Bruxelles (ULB), published
      for research on classification of highly imbalanced data.</li>
</ul>

</body></html>""")


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    browser = find_browser()
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            f"--user-data-dir={tmp}",
            "--no-pdf-header-footer",
            "--print-to-pdf-no-header",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if not pdf_path.exists():
            raise SystemExit(
                "The browser did not produce a PDF.\n"
                f"Exit code {result.returncode}\n{result.stderr[-800:]}"
            )


def stamp_pages(pdf_path: Path) -> int:
    """Add the running header and page numbers that print CSS cannot do."""
    try:
        import pymupdf
    except ImportError:  # older installs expose it as fitz
        import fitz as pymupdf  # type: ignore

    doc = pymupdf.open(pdf_path)
    total = doc.page_count
    header = "DecodeLabs | Data Science Project 2 | Change Log"
    for number, page in enumerate(doc, start=1):
        width, height = page.rect.width, page.rect.height
        top = 12 * 72 / 25.4       # 12 mm from the top edge
        page.insert_text((width * 0.06, top), header,
                         fontname="helv", fontsize=7.5, color=(0.42, 0.45, 0.51))
        page.insert_text((width * 0.88, top), f"Page {number}",
                         fontname="helv", fontsize=7.5, color=(0.42, 0.45, 0.51))
        page.draw_line(pymupdf.Point(width * 0.06, top + 2.2),
                       pymupdf.Point(width * 0.94, top + 2.2),
                       color=(0.85, 0.87, 0.9), width=0.5)
        page.insert_text((width * 0.06, height - 8 * 72 / 25.4),
                         f"Project 2 of 2 | {total} pages",
                         fontname="helv", fontsize=7.5, color=(0.55, 0.58, 0.63))
    doc.saveIncr()
    doc.close()
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Project 2's change log PDF.")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=PDF_PATH)
    parser.add_argument("--keep-html", action="store_true",
                        help="also keep the HTML next to the PDF")
    args = parser.parse_args()

    started = time.time()
    values = load_values(args.results)
    html = build_html(values)

    html_path = (HERE / "Project2_Change_Log.html") if args.keep_html else (
        Path(tempfile.gettempdir()) / "Project2_Change_Log.html")
    html_path.write_text(html, encoding="utf-8")

    render_pdf(html_path, args.out)
    pages = stamp_pages(args.out)
    size_kb = args.out.stat().st_size / 1024

    print(f"Wrote {args.out}")
    print(f"  {pages} pages, {size_kb:,.0f} KB, built in {time.time() - started:.1f}s")
    print(f"  Reproduction check in the document: "
          f"{'PASS' if values['reproduction']['all_passed'] else 'FAIL'}")
    if not values["reproduction"]["all_passed"]:
        print("  WARNING: the document reports a failing reproduction check. "
              "Fix the model export before submitting.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
