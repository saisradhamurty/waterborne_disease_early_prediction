"""
src/data_loader.py
------------------
Loads and validates all raw data sources with full provenance tracking.

Sources
-------
A. data/raw/cgwb_cg_groundwater.csv
   Real CGWB groundwater chemical-parameter data for Chhattisgarh (1961-2025).
   Columns: pH, EC, TDS, Nitrate, Fluoride, Hardness, Ca, Mg, Fe, As, Mn, etc.
   Geographic: State-District-Block-Village-Lat-Lon

B. data/dataset.csv
   Existing training dataset (798 records) covering pan-India districts.
   Contains water quality, weather, sanitation, demographic, and disease label columns.

C. Synthetic disease surveillance reports (project_REPORT_1.pdf — 5 disease scenarios)
   Summarised here as structured records; raw PDF kept in data/raw/.

All datasets get a `record_origin` field: "cgwb_observed", "training_dataset", or "synthetic".
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

CGWB_CSV = DATA_RAW / "cgwb_cg_groundwater.csv"
TRAINING_CSV = ROOT / "data" / "dataset.csv"

# ── Column rename map: CGWB raw → standardised names ─────────────────────────
CGWB_RENAME = {
    "SlNo": "cgwb_sl_no",
    "Station": "station_name",
    "Agency": "agency",
    "State LGD Code": "state_lgd_code",
    "State": "state",
    "District LGD Code": "district_lgd_code",
    "District": "district",
    "Tehsil": "tehsil",
    "Block": "block",
    "Village": "village",
    "River": "river",
    "Basin": "basin",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Data Acquisition Time": "sample_date",
    "Potential of Hydrogen (pH)": "ph",
    "Electric Conductivity (μS/cm)": "ec_us_cm",
    "Total Dissolved Solids (mg/L)": "tds_mg_l",
    "Carbonate (mg/L)": "carbonate_mg_l",
    "Bicarbonate (mg/L)": "bicarbonate_mg_l",
    "Total Alkalinity (mg/L as CaCO3)": "total_alkalinity_mg_l",
    "Chloride (mg/L)": "chloride_mg_l",
    "Nitrate N (mgN/L)": "nitrate_n_mg_l",
    "Sulphate (mg/L)": "sulphate_mg_l",
    "Phosphate(mg/L)": "phosphate_mg_l",
    "Silica(mg/L)": "silica_mg_l",
    "Fluoride (mg/L)": "fluoride_mg_l",
    "Total Hardness (mgCaCO3/L)": "total_hardness_mg_l",
    "Calcium (mg/L)": "calcium_mg_l",
    "Magnesium (mg/L)": "magnesium_mg_l",
    "Sodium (mg/L)": "sodium_mg_l",
    "Potassium (mg/L)": "potassium_mg_l",
    "Iron(mg/L)": "iron_mg_l",
    "Arsenic (mg/L)": "arsenic_mg_l",
    "Uranium(mg/L)": "uranium_mg_l",
    "Manganese (mg/L)": "manganese_mg_l",
    "Copper (mg/L)": "copper_mg_l",
    "Lead (mg/L)": "lead_mg_l",
    "Zinc (mg/L)": "zinc_mg_l",
    "Nickel (mg/L)": "nickel_mg_l",
    "Cadmium (mg/L)": "cadmium_mg_l",
    "Chromium (mg/L)": "chromium_mg_l",
}

# IS:10500:2012 permissible limits for key parameters
IS10500_LIMITS = {
    "ph": (6.5, 8.5),           # acceptable range
    "tds_mg_l": (500, 2000),    # acceptable, max permissible
    "nitrate_n_mg_l": (45, 45), # max permissible (as NO3)
    "fluoride_mg_l": (1.0, 1.5),
    "iron_mg_l": (0.3, 1.0),
    "arsenic_mg_l": (0.01, 0.05),
    "manganese_mg_l": (0.1, 0.3),
    "total_hardness_mg_l": (200, 600),
    "chloride_mg_l": (250, 1000),
}


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Replace '-', '', whitespace with NaN, then coerce to float."""
    cleaned = series.astype(str).str.strip().replace({"-": np.nan, "": np.nan, "nan": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def load_cgwb_raw() -> pd.DataFrame:
    """
    Load the CGWB Chhattisgarh groundwater CSV.
    Returns a DataFrame with standardised column names, parsed dates,
    numeric types, and record_origin = 'cgwb_observed'.

    Data quality notes added as columns:
        cgwb_ph_missing, cgwb_tds_missing, ... (bool flags)
    """
    if not CGWB_CSV.exists():
        logger.warning("CGWB CSV not found at %s — skipping.", CGWB_CSV)
        return pd.DataFrame()

    df = pd.read_csv(CGWB_CSV, low_memory=False)
    df = df.rename(columns=CGWB_RENAME)

    # Parse sample date
    df["sample_date"] = pd.to_datetime(df["sample_date"], dayfirst=True, errors="coerce")
    df["sample_year"] = df["sample_date"].dt.year
    df["sample_month"] = df["sample_date"].dt.month

    # Coerce all numeric chemical columns
    numeric_cols = [c for c in CGWB_RENAME.values()
                    if c not in ("cgwb_sl_no", "station_name", "agency", "state_lgd_code",
                                 "state", "district_lgd_code", "district", "tehsil",
                                 "block", "village", "river", "basin", "sample_date")
                    and c not in ("latitude", "longitude")]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _coerce_numeric(df[col])

    # Lat/lon as float
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Normalise district and state names
    df["district"] = df["district"].astype(str).str.strip().str.title()
    df["state"] = df["state"].astype(str).str.strip().str.title()

    # Provenance
    df["record_origin"] = "cgwb_observed"
    df["source_file"] = "cgwb_cg_groundwater.csv"

    # IS:10500 violation flags
    for param, (lo, hi) in IS10500_LIMITS.items():
        if param in df.columns:
            col_flag = f"is10500_violation_{param}"
            if param == "ph":
                df[col_flag] = ((df[param] < lo) | (df[param] > hi)).astype(float)
            else:
                df[col_flag] = (df[param] > hi).astype(float)

    # Missingness report
    key_params = ["ph", "tds_mg_l", "nitrate_n_mg_l", "fluoride_mg_l",
                  "iron_mg_l", "arsenic_mg_l", "total_hardness_mg_l"]
    for p in key_params:
        if p in df.columns:
            df[f"missing_{p}"] = df[p].isna().astype(int)

    logger.info("CGWB raw loaded: %d rows, %d cols", len(df), len(df.columns))
    return df


def load_training_dataset() -> pd.DataFrame:
    """
    Load the existing 798-record training dataset.
    Adds record_origin = 'training_dataset'.
    """
    df = pd.read_csv(TRAINING_CSV)
    df["record_origin"] = "training_dataset"
    df["source_file"] = "dataset.csv"
    logger.info("Training dataset loaded: %d rows", len(df))
    return df


def load_synthetic_surveillance_reports() -> pd.DataFrame:
    """
    Returns structured records extracted from the 5 synthetic disease surveillance
    reports (project_REPORT_1.pdf).  These are ILLUSTRATIVE academic scenarios,
    NOT confirmed outbreak data.  record_origin = 'synthetic'.

    Parameters are manually transcribed from the PDF tables.
    """
    records = [
        # Report 1: Cholera, SIM-CHOL-001, Durg, 14-day
        dict(report_id="SIM-CHOL-001", disease="Cholera", district="Durg",
             state="Chhattisgarh", latitude=21.19, longitude=81.28,
             reporting_period="19-25 Oct 2026", forecast_horizon_days=14,
             turbidity_ntu=5.2, ph=7.4, tds_mg_l=420, fecal_coliform_per_100ml=18,
             e_coli_detected=1, residual_chlorine_mg_l=0.10,
             heavy_rainfall=1, flooding=0, water_stagnation=1,
             sewage_leakage=1, water_supply_interruption=0,
             suspected_cases=12, probable_cases=5, confirmed_cases=2,
             hospital_admissions=3, deaths=0,
             predicted_risk_category="High", data_source="Synthetic academic dataset"),

        # Report 2: Dysentery, SIM-DYS-002, Durg, 14-day
        dict(report_id="SIM-DYS-002", disease="Dysentery", district="Durg",
             state="Chhattisgarh", latitude=21.19, longitude=81.28,
             reporting_period="19-25 Oct 2026", forecast_horizon_days=14,
             turbidity_ntu=4.6, ph=7.2, tds_mg_l=390, fecal_coliform_per_100ml=12,
             e_coli_detected=1, bod_mg_l=3.2, residual_chlorine_mg_l=0.20,
             heavy_rainfall=1, flooding=0, water_stagnation=1,
             sewage_leakage=1, water_supply_interruption=1,
             suspected_cases=18, probable_cases=8, confirmed_cases=3,
             hospital_admissions=4, deaths=0,
             predicted_risk_category="High", data_source="Synthetic academic dataset"),

        # Report 3: Typhoid, SIM-TYP-003, Durg, 30-day
        dict(report_id="SIM-TYP-003", disease="Typhoid", district="Durg",
             state="Chhattisgarh", latitude=21.19, longitude=81.28,
             reporting_period="19 Oct - 18 Nov 2026", forecast_horizon_days=30,
             nitrate_n_mg_l=22, ph=7.1, tds_mg_l=510, fecal_coliform_per_100ml=9,
             e_coli_detected=0, residual_chlorine_mg_l=0.15,
             heavy_rainfall=0, flooding=0, water_stagnation=0,
             sewage_leakage=1, water_supply_interruption=0,
             suspected_cases=9, probable_cases=4, confirmed_cases=1,
             hospital_admissions=2, deaths=0,
             predicted_risk_category="Medium", data_source="Synthetic academic dataset"),

        # Report 4: Hepatitis A, SIM-HEP-004, Durg, 30-day
        dict(report_id="SIM-HEP-004", disease="Hepatitis_A", district="Durg",
             state="Chhattisgarh", latitude=21.19, longitude=81.28,
             reporting_period="19 Oct - 18 Nov 2026", forecast_horizon_days=30,
             turbidity_ntu=3.1, ph=7.6, tds_mg_l=350, fecal_coliform_per_100ml=6,
             e_coli_detected=0, residual_chlorine_mg_l=0.25,
             heavy_rainfall=0, flooding=0, water_stagnation=0,
             sewage_leakage=0, water_supply_interruption=0,
             suspected_cases=5, probable_cases=2, confirmed_cases=0,
             hospital_admissions=1, deaths=0,
             predicted_risk_category="Low", data_source="Synthetic academic dataset"),

        # Report 5: Giardiasis, SIM-GIA-005, Durg, 30-day
        dict(report_id="SIM-GIA-005", disease="Giardiasis", district="Durg",
             state="Chhattisgarh", latitude=21.19, longitude=81.28,
             reporting_period="19 Oct - 18 Nov 2026", forecast_horizon_days=30,
             turbidity_ntu=7.8, ph=6.9, tds_mg_l=480, fecal_coliform_per_100ml=14,
             e_coli_detected=1, residual_chlorine_mg_l=0.08,
             heavy_rainfall=1, flooding=1, water_stagnation=1,
             sewage_leakage=1, water_supply_interruption=1,
             suspected_cases=22, probable_cases=10, confirmed_cases=4,
             hospital_admissions=6, deaths=0,
             predicted_risk_category="High", data_source="Synthetic academic dataset"),
    ]

    df = pd.DataFrame(records)
    df["record_origin"] = "synthetic"
    df["source_file"] = "project_REPORT_1.pdf"
    logger.info("Synthetic surveillance records: %d", len(df))
    return df


def get_cgwb_durg_district() -> pd.DataFrame:
    """
    Filter CGWB data to Durg district with valid coordinates and key parameters.
    Used for the Chhattisgarh-focused water quality map.
    """
    df = load_cgwb_raw()
    if df.empty:
        return df
    durg = df[df["district"].str.upper() == "DURG"].copy()
    # Keep only rows with valid lat/lon
    durg = durg.dropna(subset=["latitude", "longitude"])
    durg = durg[(durg["latitude"] > 0) & (durg["longitude"] > 0)]
    return durg


def get_cgwb_chhattisgarh_summary() -> pd.DataFrame:
    """
    District-level mean of key parameters across all of Chhattisgarh.
    Used for the water-quality map overview.
    """
    df = load_cgwb_raw()
    if df.empty:
        return pd.DataFrame()

    key = ["ph", "tds_mg_l", "nitrate_n_mg_l", "fluoride_mg_l",
           "iron_mg_l", "total_hardness_mg_l", "manganese_mg_l"]
    present = [c for c in key if c in df.columns]
    agg = (df.groupby(["district", "state"])[present]
           .mean(numeric_only=True)
           .reset_index())

    # Add representative lat/lon (mean per district)
    coords = (df.groupby("district")[["latitude", "longitude"]]
              .mean(numeric_only=True)
              .reset_index())
    agg = agg.merge(coords, on="district", how="left")
    agg["record_origin"] = "cgwb_observed"
    return agg


def data_quality_report() -> dict:
    """
    Returns a summary dict describing missingness and coverage for all sources.
    """
    cgwb = load_cgwb_raw()
    train = load_training_dataset()

    report = {}
    if not cgwb.empty:
        key_cols = ["ph", "tds_mg_l", "nitrate_n_mg_l", "fluoride_mg_l",
                    "iron_mg_l", "total_hardness_mg_l"]
        report["cgwb"] = {
            "rows": len(cgwb),
            "districts": int(cgwb["district"].nunique()),
            "date_min": str(cgwb["sample_date"].min()),
            "date_max": str(cgwb["sample_date"].max()),
            "missingness_pct": {
                c: round(cgwb[c].isna().mean() * 100, 1)
                for c in key_cols if c in cgwb.columns
            },
        }

    report["training_dataset"] = {
        "rows": len(train),
        "states": int(train["state"].nunique()),
        "districts": int(train["district"].nunique()),
        "disease_counts": train["disease"].value_counts().to_dict(),
        "missingness_pct": {
            c: round(train[c].isna().mean() * 100, 1)
            for c in train.columns
        },
    }

    report["synthetic_surveillance"] = {
        "rows": 5,
        "diseases": ["Cholera", "Dysentery", "Typhoid", "Hepatitis_A", "Giardiasis"],
        "district": "Durg, Chhattisgarh",
        "note": "Synthetic academic scenarios only — not confirmed outbreak data.",
    }

    return report
