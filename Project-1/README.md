# DecodeLabs Data Science, Project 1: Advanced EDA & Feature Engineering

**Intern:** Praise · **Programme:** DecodeLabs Data Science, November 2026 Batch · **Instructor:** Vishal Rajpoot

This project turns a raw e-commerce order dataset (1,200 orders, 14 columns) into a clean, machine-learning-ready dataset. Every decision is written down in the notebook so a reviewer can follow and challenge it.

## What the notebook covers

The notebook follows the brief's Input → Process → Output structure.

| Module | Content |
|---|---|
| 0 | Load, profile and explore the data |
| 1 (Input) | Missing-value audit, integrity checks, imputation demonstration, outlier handling |
| 2 (Process) | Vectorized feature engineering, one-hot encoding, collinearity removal |
| 3 (Output) | Pandera schema validation, export, verification on the exported files, decision log |

## The three core requirements

1. **Statistical imputation (Mean / Median / KNN).** The only column with missing values is `CouponCode` (309 rows, 25.75%). A blank there appears to mean "no coupon used", so it is kept as a `No Coupon` category rather than imputed, because imputing would invent coupons. To still demonstrate the three methods, the notebook hides 10% of `UnitPrice` in a separate copy, fills it with Mean, Median and KNN, and compares each against the true values. The real data is never modified by this step.
2. **Outlier handling (IQR and Z-score).** The IQR rule flags 8 `TotalPrice` values and the Z-score rule flags none. The flagged orders are valid, so they are capped with `np.clip` into a new column, `TotalPrice_capped`. No rows are deleted and the original `TotalPrice` is kept.
3. **Feature engineering (5 new features).** `used_coupon`, `other_items_in_cart`, `order_month`, `is_weekend` and `price_tier`, all built with vectorized operations (no row loops).

Beyond the core requirements, the notebook one-hot encodes the categorical columns with `drop_first=True`, removes features correlated above 0.80 (keeping whichever of each pair is more related to the target), and validates the final table with Pandera.

## Assumptions

- **A1.** A blank `CouponCode` means no coupon was used.
- **A2.** The dataset has no target column, so `is_failed` is defined as an order whose status is Cancelled or Returned. It is used only to decide which feature to keep in a collinear pair. This is awaiting instructor confirmation.
- **A3.** The imputation demonstration runs on a separate copy of the data. This is awaiting instructor confirmation.

The data looks synthetic (near-uniform values, evenly split statuses), so weak predictive signal is expected. The aim of the project is a sound, documented process.

## Repository layout

```
project-1/
├── project1_eda_feature_engineering.ipynb
├── requirements.txt
└── data/
    ├── Dataset for Data Analytics.xlsx   (input)
    ├── cleaned_data.csv                  (output)
    ├── engineered_data.csv               (output)
    └── model_matrix.csv                  (output)
```

## How to run

1. Install the dependencies: `pip install -r requirements.txt`
2. Open `project1_eda_feature_engineering.ipynb` in Jupyter.
3. Choose **Kernel → Restart & Run All**. Paths are relative to the notebook, so it must stay next to the `data/` folder.

The three CSV files in `data/` are regenerated on every run.
