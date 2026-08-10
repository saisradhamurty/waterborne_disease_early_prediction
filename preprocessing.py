"""
preprocessing.py
-----------------
Reproduces — exactly, step for step — the data-cleaning pipeline from Section 4 of the
original training notebook, so that the reference statistics used at prediction time
(min/max ranges, region risk lookup) are computed on the *same* cleaned data the models
were trained on.

Nothing here retrains or changes any model. This module only rebuilds the small set of
lookup artifacts (REF_STATS, REGION_RISK_LOOKUP) that the notebook computed in-memory
from the training dataframe but never exported to disk on their own. Recomputing them
from data/dataset.csv is required so utils/feature_engineering.py can reconstruct the
exact same engineered features for new, user-submitted records.
"""

import numpy as np
import pandas as pd
import streamlit as st

DATA_PATH = "data/dataset.csv"

# Columns that get IQR-based outlier capping (Section 4.3 of the notebook)
OUTLIER_COLS = [
    "turbidity_ntu", "bod_mg_l", "fecal_coliform_per_100ml", "total_coliform_per_100ml",
    "tds_mg_l", "nitrate_mg_l", "fluoride_mg_l", "arsenic_ug_l", "avg_rainfall_mm",
]

# Columns whose min/max define the 0-1 normalisation range used inside the engineered
# risk-score formulas (Section 6.2 of the notebook)
REF_COLS = [
    "turbidity_ntu", "fecal_coliform_per_100ml", "total_coliform_per_100ml", "bod_mg_l",
    "tds_mg_l", "nitrate_mg_l", "fluoride_mg_l", "arsenic_ug_l", "water_quality_index",
    "avg_rainfall_mm", "avg_humidity_pct", "open_defecation_rate", "sewage_treatment_pct",
]

# Disease -> horizon grouping used to build the 14-day / 30-day targets (Section 6.1)
FAST_ONSET_DISEASES = {"Cholera", "Dysentery", "Leptospirosis"}
SLOW_ONSET_DISEASES = {"Typhoid", "Giardiasis", "Hepatitis_A", "Hepatitis_E"}


def _cap_outliers_iqr(series: pd.Series, k: float = 3.0) -> pd.Series:
    """Winsorize a series to [Q1 - k*IQR, Q3 + k*IQR]. Matches notebook Section 4.3 exactly."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return series.clip(lower=lower, upper=upper)


@st.cache_data(show_spinner=False)
def load_clean_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Loads data/dataset.csv and applies the exact same cleaning steps as notebook
    Section 4 (missing values, duplicates, outlier capping, cyclical month encoding)
    plus the Section 6.1 target-variable creation, so downstream lookups match training.
    """
    df = pd.read_csv(path)

    # 4.1 Missing value handling
    num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_all = df.select_dtypes(include=["object"]).columns.tolist()
    for col in num_cols_all:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols_all:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 4.2 Duplicate removal
    df = df.drop_duplicates().reset_index(drop=True)

    # 4.3 Outlier capping (IQR, k=3.0)
    for col in OUTLIER_COLS:
        if col in df.columns:
            df[col] = _cap_outliers_iqr(df[col])

    # 4.4 Cyclical month encoding
    df["month"] = df["month"].astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # 4.7 Feature transformation (type sanity fixes)
    binary_like_cols = ["is_urban", "toilet_access", "flooding"] + [
        c for c in df.columns if c.startswith("symptom_")
    ]
    for col in binary_like_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    for col in cat_cols_all:
        df[col] = df[col].astype(str).str.strip()

    # 6.1 Target variable creation (medically-grounded incubation-period grouping)
    df["outbreak_risk_14d"] = df["disease"].apply(lambda d: 1 if d in FAST_ONSET_DISEASES else 0)
    df["outbreak_risk_30d"] = df["disease"].apply(lambda d: 1 if d in SLOW_ONSET_DISEASES else 0)

    return df


@st.cache_data(show_spinner=False)
def get_reference_artifacts(path: str = DATA_PATH):
    """
    Returns (REF_STATS, REGION_RISK_LOOKUP) computed from the cleaned dataset — the same
    reference values the training notebook used inside engineer_features() (Section 11).

    REF_STATS: {column: (min, max)} used to min-max normalise raw readings into the
               engineered 0-100 risk scores.
    REGION_RISK_LOOKUP: {region: historical mean outbreak_risk_14d} used to build the
               location_risk_score for a new input's region.
    """
    df = load_clean_dataset(path)
    ref_stats = {col: (float(df[col].min()), float(df[col].max())) for col in REF_COLS}
    region_risk_lookup = df.groupby("region")["outbreak_risk_14d"].mean().to_dict()
    return ref_stats, region_risk_lookup


@st.cache_data(show_spinner=False)
def get_input_defaults(path: str = DATA_PATH) -> dict:
    """Dataset-wide defaults for optional/demographic fields, matching the notebook's
    predict_outbreak_risk() fallback behaviour (Section 11)."""
    df = load_clean_dataset(path)
    return {
        "latitude": float(df["latitude"].mean()),
        "longitude": float(df["longitude"].mean()),
        "region": df["region"].mode()[0],
        "is_urban": int(df["is_urban"].mode()[0]),
        "population_density": float(df["population_density"].median()),
        "water_source": df["water_source"].mode()[0],
        "water_treatment": df["water_treatment"].mode()[0],
        "age": float(df["age"].median()),
        "gender": df["gender"].mode()[0],
    }
