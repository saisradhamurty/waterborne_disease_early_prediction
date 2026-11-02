"""
src/preprocessing.py
---------------------
Reproducible preprocessing pipeline for model training.

Produces data/processed/model_training_dataset.csv — the final feature matrix
fed into src/train_models.py.  All steps are documented and reversible.

Rules
-----
- Fit imputers / scalers / encoders on training split only (no leakage).
- Keep record_origin / source_file / sample identifiers throughout.
- Never overwrite original raw files.
- Document every imputation decision.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_training_dataset

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Outlier capping (IQR, k=3) ───────────────────────────────────────────────
OUTLIER_COLS = [
    "turbidity_ntu", "bod_mg_l", "fecal_coliform_per_100ml", "total_coliform_per_100ml",
    "tds_mg_l", "nitrate_mg_l", "fluoride_mg_l", "arsenic_ug_l", "avg_rainfall_mm",
]

# ── Reference columns for min-max normalisation ──────────────────────────────
REF_COLS = [
    "turbidity_ntu", "fecal_coliform_per_100ml", "total_coliform_per_100ml", "bod_mg_l",
    "tds_mg_l", "nitrate_mg_l", "fluoride_mg_l", "arsenic_ug_l", "water_quality_index",
    "avg_rainfall_mm", "avg_humidity_pct", "open_defecation_rate", "sewage_treatment_pct",
]

# Disease-to-horizon mapping (medically grounded on incubation periods)
FAST_ONSET_DISEASES = {"Cholera", "Dysentery", "Leptospirosis"}
SLOW_ONSET_DISEASES  = {"Typhoid", "Giardiasis", "Hepatitis_A", "Hepatitis_E"}


def _cap_outliers_iqr(series: pd.Series, k: float = 3.0) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series.clip(lower=q1 - k * iqr, upper=q3 + k * iqr)


def clean_training_data(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Full cleaning pipeline on the training dataset.

    Steps
    -----
    1. Load raw dataset if not provided.
    2. Fill numeric NaNs with column median, categorical with mode.
    3. Drop exact duplicates (check all columns).
    4. IQR-based outlier capping (k=3) on OUTLIER_COLS.
    5. Cyclical month encoding.
    6. Binary-column type coercion.
    7. Categorical string normalisation.
    8. Create 14-day / 30-day binary target columns.
    """
    if df is None:
        df = load_training_dataset()

    df = df.copy()

    # Step 2: Missing values
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in num_cols:
        if df[col].isnull().sum():
            fill = df[col].median()
            df[col] = df[col].fillna(fill)
            logger.debug("Filled %s NaNs with median=%.3f", col, fill)
    for col in cat_cols:
        if df[col].isnull().sum():
            fill = df[col].mode()[0]
            df[col] = df[col].fillna(fill)

    # Step 3: Duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < before:
        logger.info("Dropped %d duplicate rows", before - len(df))

    # Step 4: Outlier capping
    for col in OUTLIER_COLS:
        if col in df.columns:
            df[col] = _cap_outliers_iqr(df[col])

    # Step 5: Cyclical month
    df["month"] = df["month"].astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Step 6: Binary columns
    binary_cols = ["is_urban", "toilet_access", "flooding"] + [
        c for c in df.columns if c.startswith("symptom_")
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Step 7: String normalisation
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()

    # Step 8: Target variables
    df["outbreak_risk_14d"] = df["disease"].apply(
        lambda d: 1 if d in FAST_ONSET_DISEASES else 0)
    df["outbreak_risk_30d"] = df["disease"].apply(
        lambda d: 1 if d in SLOW_ONSET_DISEASES else 0)

    return df


def compute_reference_artifacts(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Compute REF_STATS and REGION_RISK_LOOKUP from the cleaned training dataframe.
    Must be called on the TRAINING SPLIT only (not the full dataset) to avoid leakage.

    Returns
    -------
    ref_stats : {col: (min, max)}
    region_risk_lookup : {region: mean_outbreak_risk_14d}
    """
    ref_stats = {col: (float(df[col].min()), float(df[col].max()))
                 for col in REF_COLS if col in df.columns}
    region_risk_lookup = (df.groupby("region")["outbreak_risk_14d"]
                          .mean().to_dict())
    return ref_stats, region_risk_lookup


def make_processed_datasets(df_clean: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """
    Produce and save all processed output files to data/processed/.

    Files
    -----
    real_water_quality.csv         — only the physicochemical water-quality columns
    model_training_dataset.csv     — full cleaned dataset used for model training
    """
    if df_clean is None:
        df_clean = clean_training_data()

    # Water-quality subset (no disease label, no demographic)
    wq_cols = [
        "state", "district", "latitude", "longitude", "record_origin",
        "water_quality_index", "ph", "turbidity_ntu", "dissolved_oxygen_mg_l",
        "bod_mg_l", "fecal_coliform_per_100ml", "total_coliform_per_100ml",
        "tds_mg_l", "nitrate_mg_l", "fluoride_mg_l", "arsenic_ug_l",
    ]
    wq = df_clean[[c for c in wq_cols if c in df_clean.columns]]
    wq.to_csv(PROCESSED / "real_water_quality.csv", index=False)

    df_clean.to_csv(PROCESSED / "model_training_dataset.csv", index=False)

    logger.info("Processed datasets saved to %s", PROCESSED)
    return {
        "real_water_quality": wq,
        "model_training_dataset": df_clean,
    }
