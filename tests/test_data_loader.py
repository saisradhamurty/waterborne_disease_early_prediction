"""
tests/test_data_loader.py
--------------------------
Tests for src/data_loader.py — verifies data loading, column renaming,
IS:10500 violation flags, and data provenance fields.

Run with:
    python -m pytest tests/ -v
"""

import pandas as pd
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── helpers ──────────────────────────────────────────────────────────────────

def _cgwb_raw_available():
    """Return True only if the CGWB raw CSV has been placed in data/raw/."""
    return (ROOT / "data" / "raw" / "cgwb_cg_groundwater.csv").exists()


# ── data_loader imports ───────────────────────────────────────────────────────

from src.data_loader import (
    load_training_dataset,
    load_synthetic_surveillance_reports,
    get_cgwb_chhattisgarh_summary,
    data_quality_report,
    IS10500_LIMITS,
)


class TestLoadTrainingDataset:
    def test_returns_dataframe(self):
        df = load_training_dataset()
        assert isinstance(df, pd.DataFrame), "Expected a DataFrame"

    def test_minimum_rows(self):
        df = load_training_dataset()
        assert len(df) >= 100, f"Expected >= 100 rows, got {len(df)}"

    def test_record_origin_column(self):
        df = load_training_dataset()
        assert "record_origin" in df.columns
        assert (df["record_origin"] == "training_dataset").all(), \
            "All training rows must have record_origin='training_dataset'"

    def test_required_columns_present(self):
        df = load_training_dataset()
        required = ["disease", "ph", "turbidity_ntu", "tds_mg_l", "nitrate_mg_l",
                    "fecal_coliform_per_100ml", "state", "district", "latitude", "longitude"]
        for col in required:
            assert col in df.columns, f"Missing required column: {col}"

    def test_disease_classes(self):
        df = load_training_dataset()
        expected_diseases = {
            "No_Disease", "Typhoid", "Giardiasis", "Dysentery",
            "Hepatitis_E", "Cholera", "Hepatitis_A", "Leptospirosis",
        }
        found = set(df["disease"].unique())
        assert found.issubset(expected_diseases | {"No_Disease"}), \
            f"Unexpected disease labels: {found - expected_diseases}"


class TestSurveillanceReports:
    def test_returns_5_records(self):
        df = load_synthetic_surveillance_reports()
        assert len(df) == 5, f"Expected 5 synthetic surveillance records, got {len(df)}"

    def test_record_origin_synthetic(self):
        df = load_synthetic_surveillance_reports()
        assert "record_origin" in df.columns
        assert (df["record_origin"] == "synthetic").all(), \
            "All surveillance records must be labelled as synthetic"

    def test_required_fields(self):
        df = load_synthetic_surveillance_reports()
        for col in ["report_id", "disease", "district", "state",
                    "suspected_cases", "confirmed_cases", "predicted_risk_category"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_risk_categories_valid(self):
        df = load_synthetic_surveillance_reports()
        valid = {"Low", "Medium", "High"}
        cats = set(df["predicted_risk_category"].dropna().unique())
        assert cats.issubset(valid), f"Unexpected risk categories: {cats - valid}"

    def test_diseases_are_known(self):
        df = load_synthetic_surveillance_reports()
        known = {
            "Cholera", "Dysentery", "Leptospirosis",
            "Typhoid", "Giardiasis", "Hepatitis_A", "Hepatitis_E",
        }
        for d in df["disease"].unique():
            assert d in known, f"Unknown disease: {d}"


class TestIS10500Limits:
    def test_limits_dict_non_empty(self):
        assert len(IS10500_LIMITS) > 0

    def test_ph_limits(self):
        assert "ph" in IS10500_LIMITS
        lo, hi = IS10500_LIMITS["ph"]
        assert lo == pytest.approx(6.5)
        assert hi == pytest.approx(8.5)

    def test_tds_limit(self):
        assert "tds_mg_l" in IS10500_LIMITS

    def test_fluoride_limit(self):
        assert "fluoride_mg_l" in IS10500_LIMITS


@pytest.mark.skipif(not _cgwb_raw_available(), reason="CGWB raw CSV not present in data/raw/")
class TestCgwbSummary:
    def test_returns_dataframe(self):
        df = get_cgwb_chhattisgarh_summary()
        assert isinstance(df, pd.DataFrame)

    def test_has_district_column(self):
        df = get_cgwb_chhattisgarh_summary()
        assert "district" in df.columns

    def test_has_coordinates(self):
        df = get_cgwb_chhattisgarh_summary()
        assert "latitude" in df.columns
        assert "longitude" in df.columns

    def test_record_origin(self):
        df = get_cgwb_chhattisgarh_summary()
        if "record_origin" in df.columns:
            assert (df["record_origin"] == "cgwb_observed").all()


class TestDataQualityReport:
    def test_returns_dict(self):
        report = data_quality_report()
        assert isinstance(report, dict)

    def test_has_training_key(self):
        report = data_quality_report()
        assert "training_dataset" in report

    def test_training_has_rows(self):
        report = data_quality_report()
        assert "rows" in report["training_dataset"]
        assert report["training_dataset"]["rows"] > 0

    def test_surveillance_key_present(self):
        report = data_quality_report()
        assert "synthetic_surveillance" in report
