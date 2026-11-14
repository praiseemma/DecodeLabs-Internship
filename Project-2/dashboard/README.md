# Fraud detection dashboard

A Streamlit app that serves the Random Forest from
`Project-2/project2_fraud_detection_pipeline.ipynb`, plus the evidence behind it.

The model was trained on the Kaggle *Credit Card Fraud Detection* dataset
(283,726 transactions after removing duplicates, of which 473 are fraud). It reaches
**0.6016 precision** and **0.8105 recall** on 56,746 held-back test rows at a
threshold of 0.50. This app reproduces those numbers exactly — the export script
refuses to finish unless it does.

---

## What is in here

| Path | What it is |
|---|---|
| `app.py` | Entry point. Sets up navigation, the sidebar status panel and the page list. |
| `src/config.py` | Paths, the 30 model inputs, colours and limits — the things every file must agree on. |
| `src/state.py` | The only place that loads the model and the exported numbers, both cached. |
| `src/model.py` | Loading and scoring the pipeline, version checks, plain-language reading of a score. |
| `src/validation.py` | One set of input rules, used by both the single-row page and the upload page. |
| `src/plots.py` | Every chart, as a function of exported numbers. |
| `src/ui.py` | Small shared layout pieces (headers, metric cards, the not-a-real-system notice). |
| `src/pages/` | One file per screen: overview, score, batch, threshold, performance, insights, about. |
| `scripts/export_model.py` | Rebuilds the model from the dataset, checks it against the notebook, writes `artifacts/`. |
| `artifacts/` | The committed model (1.5 MB), `results.json`, `predictions.npz`. Produced by the script above. |
| `data/demo_transactions.csv` | 2,473-row sample: all 473 frauds plus 2,000 legitimate transactions. |
| `tests/` | 80 pytest checks: input rules, scoring, the saved numbers, the committed artifacts, and every page rendered through Streamlit's own test runner. |

---

## Run it locally (Windows, PowerShell)

Open **PowerShell**, then move into the dashboard folder:

```powershell
cd "$env:USERPROFILE\Documents\GitHub\DecodeLabs-Internship\Project-2\dashboard"
```

**Step 1 — install the pinned libraries** (once; a few minutes):

```powershell
& "$env:USERPROFILE\anaconda3\python.exe" -m pip install -r requirements.txt
```

This installs the exact versions the model was saved with. Running a pickled
model under a different scikit-learn version can change its predictions quietly,
which is why the versions are pinned rather than "latest".

**Step 2 — start the app:**

```powershell
& "$env:USERPROFILE\anaconda3\python.exe" -m streamlit run app.py
```

A browser tab opens at `http://localhost:8501`. Leave PowerShell open; closing it
stops the app. Press `Ctrl+C` in PowerShell to stop it yourself.

> `python` on its own may open the Microsoft Store instead of Anaconda. Using the
> full path above avoids that. If your Anaconda lives somewhere else, run
> `Get-Command python` in a fresh PowerShell and use the path it prints.

**Optional — check everything still works:**

```powershell
& "$env:USERPROFILE\anaconda3\python.exe" -m pytest -q
```

80 tests, about 12 seconds. They cover the input rules, scoring of known rows, the
saved metrics against the notebook, the size of the committed files, and every page
rendered through `streamlit.testing` (including moving the threshold slider and
scoring the demo file).

---

## Rebuilding the model (needs the dataset)

The committed `artifacts/` folder is enough to run the app. You only need this
step if you retrain, or if you want to verify the numbers yourself.

Put `creditcard.csv` (about 150 MB) in `Project-2/data/`, then from `Project-2`:

```powershell
& "$env:USERPROFILE\anaconda3\python.exe" dashboard\scripts\export_model.py
```

It takes about 7 minutes, most of it the cross-validated training-set sweep. It
ends with:

```text
[PASS] precision   notebook 0.6016 | now 0.6016
[PASS] recall      notebook 0.8105 | now 0.8105
...
ALL CHECKS PASSED - this model matches the notebook run.
```

If any line says FAIL, do not deploy that model.

Two flags are worth knowing:

- `--retune` re-runs the whole `GridSearchCV` (about 10 extra minutes) instead of
  using the winning settings the notebook already recorded, and reports whether
  the search picks the same ones again.
- `--no-oof` skips the slow cross-validated sweep. The threshold explorer then
  loses its honest view and says so on the page.

---

## Deploy to Streamlit Community Cloud (free)

> **Choose Python 3.13 in Advanced settings** — not the default, and not 3.14.
> `streamlit==1.51.0` needs `pyarrow<22`, and `pyarrow` publishes no prebuilt
> wheel for Python 3.14. On 3.14 the installer tries to compile Arrow from
> source and stops with **"Error installing requirements"**. On 3.13 (or 3.12)
> the whole file installs from wheels in about a minute.

1. Push this folder to the public GitHub repository, including `artifacts/`,
   `data/demo_transactions.csv`, `app.py` and `requirements.txt`.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. **New app** → pick the repository `praiseemma/DecodeLabs-Internship`, branch
   `main`.
4. **Main file path:** `Project-2/dashboard/app.py`
5. **Advanced settings → Python version:** choose **3.13**. Then **Deploy**.

Streamlit Cloud fixes the Python version when an app is created. If an existing
app was made on 3.14, create a new app from the same repository with 3.13 and
delete the broken one. The build log's first lines say which Python is in use.

Nothing needs to be set in Secrets: the app has no keys and calls no APIs.

The app finds its files from its own location, not from the folder the command
runs in, so it behaves the same on your machine and on Cloud.

---

## What is committed, and what is not

| Committed | Not committed |
|---|---|
| `artifacts/rf_fraud_pipeline.joblib` (1.5 MB) | `data/creditcard.csv` (150 MB, over GitHub's limit) |
| `artifacts/results.json` (79 KB) | the four `.csv.gz` upload parts |
| `artifacts/predictions.npz` (2.5 MB) | `__pycache__`, `.ipynb_checkpoints` |
| `data/demo_transactions.csv` (708 KB) | |

The demo file is **a sample**, and the app says so: 473 frauds plus 2,000
legitimate rows, chosen with seed 42. Its fraud rate (about 19%) is nothing like
the real 0.17%, so precision measured on it is not the model's real precision.
The honest figures come from the 56,746-row test set on the **Model performance**
page.

---

## Why it is built this way

| Decision | Reason |
|---|---|
| Model loaded with `st.cache_resource`, data with `st.cache_data` | Loading the pickle takes about a second; doing it on every click is waste. Data is copied per visitor so one person cannot change what another sees. |
| Paths built from `__file__` | Streamlit Cloud starts apps from the repository root while you start them from `dashboard/`. Hard-coded relative paths break in one of those two places. |
| One `validate_batch` used by both entry points | A rule duplicated in two places gets fixed in one of them. The single-row form goes through the same code as the CSV upload. |
| Every failure caught and rewritten | A visitor should read "this file has 2 empty values in `V7`", not a stack trace. |
| No accuracy anywhere | The baseline scores 99.83% accuracy and catches no fraud. Precision, recall, F1, ROC-AUC and average precision instead. |
| Threshold default 0.50, adjustable | 0.50 is what the notebook kept: lowering it to 0.40 raised recall by 0.013 and cut precision by 0.160. The trade-off belongs to whoever owns the cost of a missed fraud. |
| Test-set view labelled "illustration only" | Choosing a threshold on the test set destroys the test set's value. The default view is cross-validated training data. |
| Reproduction check in the app, not just in a README | If the exported model ever stops matching the notebook, the app shows it rather than claiming a score it cannot back up. |

---

## Limits worth repeating

- **Anonymised inputs.** `V1` to `V28` are PCA components, so the model cannot take
  raw card data. It cannot be pointed at a real payment system.
- **Academic demo.** Public dataset, no monitoring, no retraining schedule, no
  case-management workflow.
- **Only 473 frauds exist**, 95 of them in the test set. Each test fraud is worth
  about one point of recall, so small differences are noise.
- **Tuned on a sample.** The search ran on 100,000 stratified training rows, then
  the winner was retrained on all 226,980.
- **Not calibrated.** The model's mean score is 2.5%, against a true test fraud rate of
  0.17% — about 15× too high, because SMOTE trained it on a balanced set. Read a score
  as "more fraud-like than most", never as an 80% chance. Measured rates: rows scored
  0.90+ were fraud 87% of the time; 0.50–0.90, about 23%; below 0.50, 0.03%. At the
  deployed cut-off, about 60% of alerts are genuine. The About page shows the table.

---

## If something goes wrong

| Message | Fix |
|---|---|
| "The dashboard has no model yet" | Run `scripts/export_model.py` from `Project-2`; the dataset must be in `Project-2/data/`. |
| "Could not find the dataset at ..." | `creditcard.csv` is not stored in git. Put it in `Project-2/data/`, or pass `--data PATH`. |
| "The model file could not be read" | The pickle was written by a different scikit-learn version. Install `requirements.txt`, then re-run the export. |
| Cloud: "Error installing requirements" | The app was created on Python 3.14. Recreate it with Python 3.13 (Advanced settings); see the deploy section above. |
| `python` opens the Microsoft Store | Use the full Anaconda path shown above. |
| Port 8501 already in use | Another copy is running. Close it, or `... -m streamlit run app.py --server.port 8502`. |
