r"""
Export the Project 2 Random Forest so the dashboard can serve it.

What it does, in order
----------------------
1. Loads creditcard.csv, removes exact duplicates, and makes the *same*
   stratified 80/20 split with the same seed as the notebook.
2. Rebuilds the two pipelines from the notebook with the winning settings the
   search already found, and fits them on the full training set.
3. Saves the Random Forest pipeline to artifacts/ (compressed), everything the
   app displays to artifacts/results.json, and the probability arrays to
   artifacts/predictions.npz.
4. Writes a small demo file (all frauds + a sample of legitimate rows) so the
   app never needs the 150 MB dataset.
5. Prints PASS/FAIL against the precision and recall recorded in the notebook.

Why the settings are hard-coded instead of re-tuned
---------------------------------------------------
The hyperparameter search in the notebook already picked the winner
(Random Forest: max_depth=10, smote__k_neighbors=5, CV ROC-AUC 0.9833), and the
search took about 10 minutes. Re-running it with the same seed would return the
same settings, so the script builds them straight away. Pass --retune to re-run
the full GridSearchCV and prove that for yourself; it writes a small report
showing whether the search picks the same settings again.

Usage (PowerShell, from the Project-2-praiseemma folder)
---------------------------------------------
    & "$env:USERPROFILE\anaconda3\python.exe" dashboard\scripts\export_model.py

Optional flags:
    --data PATH     where creditcard.csv lives  (default Project-2-praiseemma/data/creditcard.csv)
    --retune        re-run GridSearchCV instead of using the recorded settings
    --no-oof        skip the (slow) cross-validated training-set probabilities
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import joblib
import sklearn
import imblearn

# --------------------------------------------------------------------------
# Paths and constants copied from the notebook (same values, on purpose)
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DASHBOARD = HERE.parent                 # Project-2-praiseemma-praiseemma/dashboard
PROJECT2 = DASHBOARD.parent             # Project-2-praiseemma-praiseemma
ARTIFACTS = DASHBOARD / "artifacts"
DEMO_PATH = DASHBOARD / "data" / "demo_transactions.csv"

RANDOM_SEED = 42
TEST_SIZE = 0.2
TUNE_SAMPLE_SIZE = 100_000

# Winning settings recorded by the notebook's GridSearchCV.
BEST_PARAMS = {
    "Logistic Regression": {"smote__k_neighbors": 5, "classifier__C": 0.01},
    "Random Forest": {"smote__k_neighbors": 5, "classifier__max_depth": 10},
}

# Notebook reference values being checked (test set, threshold 0.50).
EXPECTED = {
    "precision": 0.6016,
    "recall": 0.8105,
    "f1": 0.6906,
    "roc_auc": 0.9729,
    "avg_precision": 0.7716,
}
TOLERANCE = 0.0005          # the notebook rounds to 4 decimals
RF_CM_EXPECTED = (77, 18, 51)   # caught, missed, false alarms

DEMO_LEGIT_ROWS = 2_000
DEMO_DECIMALS = 6


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def log(message: str) -> None:
    print(message, flush=True)


def build_pipelines() -> dict[str, ImbPipeline]:
    """The two pipelines exactly as the notebook defines them."""
    lr_pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ])
    rf_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("classifier", RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)),
    ])
    return {"Logistic Regression": lr_pipe, "Random Forest": rf_pipe}


def load_and_clean(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load the dataset, drop duplicates, and describe what was found."""
    if not path.exists():
        raise SystemExit(
            f"\nCould not find the dataset at:\n  {path}\n"
            "Put creditcard.csv in Project-2-praiseemma/data/ (it is not stored in git), "
            "or pass --data PATH."
        )
    log(f"Reading {path} ...")
    raw = pd.read_csv(path)
    info = {
        "raw_rows": int(len(raw)),
        "raw_columns": int(raw.shape[1]),
        "raw_fraud": int(raw["Class"].sum()),
        "missing_values": int(raw.isnull().sum().sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
    }
    df = raw.drop_duplicates().reset_index(drop=True)

    assert df.isnull().sum().sum() == 0, "Missing values found"
    assert set(df["Class"].unique()) <= {0, 1}, "Class has values other than 0/1"
    assert (df["Amount"] >= 0).all() and (df["Time"] >= 0).all(), "Negative Time or Amount"

    info.update({
        "rows": int(len(df)),
        "frauds": int(df["Class"].sum()),
        "fraud_share": float(df["Class"].mean()),
    })
    log(f"  {info['raw_rows']:,} rows -> {info['rows']:,} after removing "
        f"{info['duplicate_rows']:,} duplicates; {info['frauds']} frauds")
    return df, info


def score_at(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    """Precision/recall/F1/confusion matrix for one probability threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "caught": int(tp),
        "missed": int(fn),
        "false_alarms": int(fp),
        "true_negatives": int(tn),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


def downsample(x: np.ndarray, y: np.ndarray, limit: int = 400) -> dict:
    """Shrink a curve to at most `limit` points so the JSON stays small."""
    if len(x) > limit:
        idx = np.linspace(0, len(x) - 1, limit).round().astype(int)
        x, y = x[idx], y[idx]
    return {"x": [round(float(v), 6) for v in x],
            "y": [round(float(v), 6) for v in y]}


def threshold_sweep(y_true: np.ndarray, proba: np.ndarray) -> list[dict]:
    """Precision, recall and F1 for thresholds 0.01 .. 0.99."""
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    rows = []
    for t in grid:
        row = score_at(y_true, proba, float(t))
        rows.append({
            "threshold": float(t),
            "precision": round(row["precision"], 6),
            "recall": round(row["recall"], 6),
            "f1": round(row["f1"], 6),
        })
    return rows


# --------------------------------------------------------------------------
# EDA aggregates: computed once here so the app never needs the dataset
# --------------------------------------------------------------------------
def eda_summary(df: pd.DataFrame) -> dict:
    feature_cols = [c for c in df.columns if c != "Class"]
    legit, fraud = df[df["Class"] == 0], df[df["Class"] == 1]

    # IQR outlier rate per column, by class (descriptive only, same as notebook)
    q1, q3 = df[feature_cols].quantile(0.25), df[feature_cols].quantile(0.75)
    iqr = q3 - q1
    is_outlier = (df[feature_cols] < q1 - 1.5 * iqr) | (df[feature_cols] > q3 + 1.5 * iqr)
    is_fraud = df["Class"] == 1
    v_cols = [c for c in feature_cols if c.startswith("V")]
    n_flagged = is_outlier[v_cols].sum(axis=1)

    outlier_rows = []
    for col in feature_cols:
        outlier_rows.append({
            "feature": col,
            "legitimate": round(float(is_outlier.loc[~is_fraud, col].mean() * 100), 2),
            "fraud": round(float(is_outlier.loc[is_fraud, col].mean() * 100), 2),
        })
    outlier_rows.sort(key=lambda r: r["fraud"])

    flagged_share = []
    for k in range(0, len(v_cols) + 1):
        flagged_share.append({
            "n_columns": k,
            "legitimate": round(float((n_flagged[~is_fraud] == k).mean() * 100), 4),
            "fraud": round(float((n_flagged[is_fraud] == k).mean() * 100), 4),
        })

    # Histograms for the amount panel, built on log10(1 + Amount) so the
    # long right tail stays readable (same treatment as the notebook).
    edges = np.linspace(0, np.log10(1 + df["Amount"].max()), 41)
    legit_hist, _ = np.histogram(np.log10(1 + legit["Amount"]), bins=edges, density=True)
    fraud_hist, _ = np.histogram(np.log10(1 + fraud["Amount"]), bins=edges, density=True)

    def describe(series: pd.Series) -> dict:
        q = series.quantile([0.25, 0.5, 0.75])
        return {
            "count": int(series.count()),
            "mean": round(float(series.mean()), 2),
            "median": round(float(q.loc[0.5]), 2),
            "q1": round(float(q.loc[0.25]), 2),
            "q3": round(float(q.loc[0.75]), 2),
            "max": round(float(series.max()), 2),
        }

    correlations = (
        df.corr(numeric_only=True)["Class"].drop("Class").abs()
        .sort_values(ascending=False).head(10)
    )

    hour_edges = np.arange(0, 25)
    legit_hour, _ = np.histogram((legit["Time"] / 3600) % 24, bins=hour_edges, density=True)
    fraud_hour, _ = np.histogram((fraud["Time"] / 3600) % 24, bins=hour_edges, density=True)

    return {
        "class_balance": {
            "legitimate": int((df["Class"] == 0).sum()),
            "fraud": int((df["Class"] == 1).sum()),
            "fraud_share": round(float(df["Class"].mean()) * 100, 4),
        },
        "amount": {"legitimate": describe(legit["Amount"]),
                   "fraud": describe(fraud["Amount"])},
        "amount_histogram": {
            "bin_left": [round(float(v), 4) for v in edges[:-1]],
            "bin_right": [round(float(v), 4) for v in edges[1:]],
            "legitimate": [round(float(v), 6) for v in legit_hist],
            "fraud": [round(float(v), 6) for v in fraud_hist],
        },
        "hour_histogram": {
            "hour": list(range(24)),
            "legitimate": [round(float(v), 6) for v in legit_hour],
            "fraud": [round(float(v), 6) for v in fraud_hour],
        },
        "top_correlations": [
            {"feature": str(k), "abs_correlation": round(float(v), 4)}
            for k, v in correlations.items()
        ],
        "outlier_rate_by_column": outlier_rows,
        "outlier_columns_per_row": flagged_share,
        "outlier_summary": {
            "v_outlier_share_fraud": round(float((n_flagged[is_fraud] > 0).mean() * 100), 1),
            "v_outlier_share_legitimate": round(float((n_flagged[~is_fraud] > 0).mean() * 100), 1),
            "mean_v_flagged_fraud": round(float(n_flagged[is_fraud].mean()), 2),
            "mean_v_flagged_legitimate": round(float(n_flagged[~is_fraud].mean()), 2),
            "amount_outlier_share_fraud": round(float(is_outlier.loc[is_fraud, "Amount"].mean() * 100), 1),
            "amount_outlier_share_legitimate": round(float(is_outlier.loc[~is_fraud, "Amount"].mean() * 100), 1),
        },
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Export the fraud model for the dashboard.")
    parser.add_argument("--data", type=Path, default=PROJECT2 / "data" / "creditcard.csv")
    parser.add_argument("--retune", action="store_true",
                        help="re-run GridSearchCV instead of using the recorded settings")
    parser.add_argument("--no-oof", action="store_true",
                        help="skip cross-validated training-set probabilities (faster)")
    args = parser.parse_args()

    started = time.time()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    DEMO_PATH.parent.mkdir(parents=True, exist_ok=True)

    df, data_info = load_and_clean(args.data)
    X, y = df.drop(columns="Class"), df["Class"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    split_info = {
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "train_frauds": int(y_train.sum()), "test_frauds": int(y_test.sum()),
        "test_size": TEST_SIZE, "random_seed": RANDOM_SEED,
    }
    log(f"  train {len(X_train):,} rows / {int(y_train.sum())} frauds | "
        f"test {len(X_test):,} rows / {int(y_test.sum())} frauds")

    pipelines = build_pipelines()
    tuned_pipes = {}
    retune_report = None

    if args.retune:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        if TUNE_SAMPLE_SIZE and len(X_train) > TUNE_SAMPLE_SIZE:
            X_tune, _, y_tune, _ = train_test_split(
                X_train, y_train, train_size=TUNE_SAMPLE_SIZE, stratify=y_train,
                random_state=RANDOM_SEED)
        else:
            X_tune, y_tune = X_train, y_train

        grids = {
            "Logistic Regression": {"smote__k_neighbors": [3, 5, 7],
                                    "classifier__C": [0.01, 0.1, 1.0]},
            "Random Forest": {"smote__k_neighbors": [3, 5],
                              "classifier__max_depth": [10, 20]},
        }
        retune_report = {}
        for name, pipe in pipelines.items():
            t0 = time.time()
            log(f"  re-tuning {name} on {len(X_tune):,} rows (this is the slow part) ...")
            search = GridSearchCV(pipe, grids[name], cv=cv, refit="roc_auc",
                                  scoring={"roc_auc": "roc_auc"}, n_jobs=-1)
            search.fit(X_tune, y_tune)
            log(f"    best {search.best_params_} | CV ROC-AUC {search.best_score_:.4f} "
                f"| {time.time() - t0:.0f}s")
            tuned_pipes[name] = clone(pipe).set_params(**search.best_params_)
            retune_report[name] = {
                "best_params": search.best_params_,
                "best_cv_roc_auc": float(search.best_score_),
                "matches_recorded": search.best_params_ == BEST_PARAMS[name],
            }
    else:
        for name, pipe in pipelines.items():
            tuned_pipes[name] = clone(pipe).set_params(**BEST_PARAMS[name])

    # ---- fit both models on the full training set -------------------------
    fitted = {}
    for name, pipe in tuned_pipes.items():
        t0 = time.time()
        log(f"  fitting {name} on {len(X_train):,} rows ...")
        fitted[name] = pipe.fit(X_train, y_train)
        log(f"    done in {time.time() - t0:.0f}s")

    # ---- evaluate once on the untouched test set -------------------------
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    metrics = {
        "Baseline (always legitimate)": {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "roc_auc": 0.5, "avg_precision": float(y_test.mean()),
            "accuracy": float(accuracy_score(y_test, baseline.predict(X_test))),
        }
    }
    test_proba, curves, matrices, importances = {}, {}, {}, {}
    for name, model in fitted.items():
        proba = model.predict_proba(X_test)[:, 1]
        test_proba[name] = proba
        pred = (proba >= 0.5).astype(int)
        metrics[name] = {
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "avg_precision": float(average_precision_score(y_test, proba)),
            "accuracy": float(accuracy_score(y_test, pred)),
        }
        metrics[name].update(score_at(y_test, proba, 0.5))
        matrices[name] = metrics[name]

        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        curves[name] = {
            "roc": downsample(fpr, tpr),
            "pr": downsample(rec, prec),
            "avg_precision": metrics[name]["avg_precision"],
            "roc_auc": metrics[name]["roc_auc"],
        }

    rf_final = fitted["Random Forest"]
    rf_importances = pd.Series(
        rf_final.named_steps["classifier"].feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    importances = [{"feature": str(k), "importance": round(float(v), 6)}
                   for k, v in rf_importances.items()]

    # ---- threshold analysis on training data only ------------------------
    log("  threshold analysis on the training set ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    if args.no_oof:
        sweep = None
        oof_proba = None
        oof_note = "skipped (--no-oof): the app falls back to the test-set sweep"
        log("    skipped (--no-oof)")
    else:
        t0 = time.time()
        oof_proba = cross_val_predict(clone(rf_final), X_train, y_train, cv=cv,
                                      method="predict_proba", n_jobs=1)[:, 1]
        sweep = threshold_sweep(y_train.to_numpy(), oof_proba)
        oof_note = "5-fold cross-validation on the training set, fold models never saw their own rows"
        log(f"    done in {time.time() - t0:.0f}s")

    test_sweep = threshold_sweep(y_test.to_numpy(), test_proba["Random Forest"])

    # The notebook's two reference points, recomputed for the app.
    at_50 = score_at(y_test.to_numpy(), test_proba["Random Forest"], 0.50)
    f1_optimal = max(test_sweep, key=lambda r: r["f1"])
    f1_optimal_training = (max(sweep, key=lambda r: r["f1"]) if sweep else None)

    # ---- EDA -------------------------------------------------------------
    log("  summarising the data for the insights page ...")
    eda = eda_summary(df)

    # ---- demo sample -----------------------------------------------------
    log("  writing the demo sample ...")
    fraud_rows = df[df["Class"] == 1]
    legit_rows = df[df["Class"] == 0].sample(
        n=min(DEMO_LEGIT_ROWS, int((df["Class"] == 0).sum())), random_state=RANDOM_SEED)
    demo = (pd.concat([legit_rows, fraud_rows])
              .sample(frac=1, random_state=RANDOM_SEED)
              .reset_index(drop=True))
    for col in feature_names:
        demo[col] = demo[col].round(DEMO_DECIMALS)
    demo = demo[feature_names + ["Class"]]
    demo.to_csv(DEMO_PATH, index=False, float_format=f"%.{DEMO_DECIMALS}f")
    demo_info = {
        "path": "data/demo_transactions.csv",
        "rows": int(len(demo)),
        "frauds": int(demo["Class"].sum()),
        "legitimate": int((demo["Class"] == 0).sum()),
        "size_kb": round(DEMO_PATH.stat().st_size / 1024, 1),
        "note": ("A sample of the cleaned dataset: every fraud row plus "
                 f"{DEMO_LEGIT_ROWS:,} randomly chosen legitimate rows. It is a sample, "
                 "not the full 283,726 rows."),
    }
    log(f"    {len(demo):,} rows ({demo_info['frauds']} frauds) -> {demo_info['size_kb']} KB")

    # ---- regression rows: fixed inputs and the probabilities this model gives
    regression_rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    picks = list(rng.choice(np.flatnonzero(y_test.to_numpy() == 1), 3, replace=False))
    picks += list(rng.choice(np.flatnonzero(y_test.to_numpy() == 0), 3, replace=False))
    for i in picks:
        row = X_test.iloc[[i]]
        regression_rows.append({
            "source": "test set",
            "row_index": int(i),
            "actual_class": int(y_test.iloc[i]),
            "features": {k: round(float(v), 6) for k, v in row.iloc[0].items()},
            "expected_probability": round(float(rf_final.predict_proba(row)[0, 1]), 6),
        })

    # ---- save the model --------------------------------------------------
    model_path = ARTIFACTS / "rf_fraud_pipeline.joblib"
    joblib.dump(rf_final, model_path, compress=3)
    model_kb = model_path.stat().st_size / 1024
    log(f"  saved model -> {model_path.name} ({model_kb:,.0f} KB)")

    versions = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "imbalanced_learn": imblearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
        "platform": platform.platform(),
    }

    # ---- PASS/FAIL against the notebook ---------------------------------
    checks = []
    for key in ("precision", "recall", "f1", "roc_auc", "avg_precision"):
        got = metrics["Random Forest"][key]
        ok = abs(got - EXPECTED[key]) <= TOLERANCE
        checks.append({"metric": key, "expected": EXPECTED[key], "got": round(got, 4), "ok": ok})
    cm_ok = (at_50["caught"], at_50["missed"], at_50["false_alarms"]) == RF_CM_EXPECTED
    checks.append({"metric": "confusion matrix at 0.50 (caught/missed/false alarms)",
                   "expected": list(RF_CM_EXPECTED),
                   "got": [at_50["caught"], at_50["missed"], at_50["false_alarms"]],
                   "ok": cm_ok})

    results = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_in_seconds": round(time.time() - started, 1),
        "data": data_info,
        "split": split_info,
        "features": feature_names,
        "versions": versions,
        "model": {
            "file": "artifacts/rf_fraud_pipeline.joblib",
            "name": "Random Forest",
            "params": BEST_PARAMS["Random Forest"],
            "n_estimators": 100,
            "random_seed": RANDOM_SEED,
            "size_kb": round(model_kb, 1),
            "trained_on_rows": int(len(X_train)),
            "default_threshold": 0.5,
        },
        "metrics": metrics,
        "curves": curves,
        "feature_importances": importances,
        "threshold_sweep": {
            "training": sweep,
            "training_note": oof_note,
            "test": test_sweep,
        },
        "threshold_reference": {
            "at_0_5_test": at_50,
            "f1_optimal_test": f1_optimal,
            "f1_optimal_training": f1_optimal_training,
        },
        "eda": eda,
        "demo": demo_info,
        "reproduction": {"checks": checks, "all_passed": all(c["ok"] for c in checks),
                         "tolerance": TOLERANCE},
        "retune": retune_report,
        "regression_rows": regression_rows,
    }

    results_path = ARTIFACTS / "results.json"
    results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    log(f"  saved results -> {results_path.name} ({results_path.stat().st_size / 1024:,.0f} KB)")

    arrays = {"test_proba_rf": test_proba["Random Forest"],
              "test_proba_lr": test_proba["Logistic Regression"],
              "y_test": y_test.to_numpy()}
    if oof_proba is not None:
        arrays["oof_proba_rf"] = oof_proba
        arrays["y_train"] = y_train.to_numpy()
    np.savez_compressed(ARTIFACTS / "predictions.npz", **arrays)
    log(f"  saved probabilities -> predictions.npz "
        f"({(ARTIFACTS / 'predictions.npz').stat().st_size / 1024:,.0f} KB)")

    # ---- report ----------------------------------------------------------
    print("\n" + "=" * 68)
    print("Reproduction check against the notebook (test set, threshold 0.50)")
    print("=" * 68)
    for c in checks:
        mark = "PASS" if c["ok"] else "FAIL"
        print(f"  [{mark}] {c['metric']:<45} notebook {c['expected']} | now {c['got']}")
    print("=" * 68)
    if all(c["ok"] for c in checks):
        print("ALL CHECKS PASSED - this model matches the notebook run.")
        print(f"Total time: {time.time() - started:.0f}s")
        return 0
    print("CHECKS FAILED - do not deploy this model. See the differences above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
