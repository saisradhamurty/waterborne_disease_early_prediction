"""
tests/test_synthetic_data.py
------------------------------
Tests for src/synthetic_data_generator.py — verifies reproducibility,
disease rule logic, and record metadata.
"""

import pandas as pd
import pytest

from src.synthetic_data_generator import (
    generate_synthetic_training_data,
    DISEASE_RULES,
)

FAST = {"Cholera", "Dysentery", "Leptospirosis"}
FAST_ONSET_DISEASES = FAST  # alias for clarity
ALL_DISEASES = {
    "Cholera", "Dysentery", "Leptospirosis",
    "Typhoid", "Giardiasis", "Hepatitis_A", "Hepatitis_E", "No_Disease",
}


class TestSyntheticDataGeneration:
    @pytest.fixture(scope="class")
    def df(self):
        return generate_synthetic_training_data(n_samples=500, seed=42)

    def test_correct_row_count(self, df):
        assert len(df) == 500

    def test_reproducible_with_same_seed(self):
        df1 = generate_synthetic_training_data(n_samples=200, seed=99)
        df2 = generate_synthetic_training_data(n_samples=200, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_synthetic_training_data(n_samples=200, seed=1)
        df2 = generate_synthetic_training_data(n_samples=200, seed=2)
        assert not df1["disease"].equals(df2["disease"]), \
            "Different seeds should produce different disease labels"

    def test_record_origin_is_synthetic(self, df):
        assert "record_origin" in df.columns
        assert (df["record_origin"] == "synthetic").all()

    def test_source_file_column(self, df):
        assert "source_file" in df.columns

    def test_disease_column_present(self, df):
        assert "disease" in df.columns

    def test_only_known_diseases(self, df):
        unknown = set(df["disease"].unique()) - ALL_DISEASES
        assert len(unknown) == 0, f"Unknown disease labels: {unknown}"

    def test_no_disease_class_present(self, df):
        assert "No_Disease" in df["disease"].values, \
            "Expected some No_Disease rows in a 500-sample dataset"

    def test_multiple_diseases_represented(self, df):
        n_diseases = df["disease"].nunique()
        assert n_diseases >= 4, \
            f"Expected >= 4 distinct diseases in 500 samples, got {n_diseases}"

    def test_numeric_columns_finite(self, df):
        import numpy as np
        num_cols = df.select_dtypes(include=[float, int]).columns
        for col in num_cols:
            assert df[col].notna().all(), f"NaN in column {col}"
            assert not df[col].isin([float("inf"), float("-inf")]).any(), \
                f"Inf in column {col}"

    def test_latitude_in_india_range(self, df):
        assert df["latitude"].between(6, 37).all()

    def test_longitude_in_india_range(self, df):
        assert df["longitude"].between(68, 98).all()

    def test_flooding_binary(self, df):
        vals = set(df["flooding"].unique())
        assert vals.issubset({0, 1}), f"flooding should be 0/1, got {vals}"

    def test_ph_in_valid_range(self, df):
        assert df["ph"].between(0, 14).all()


class TestDiseaseRules:
    def test_all_seven_diseases_have_rules(self):
        expected = {
            "Cholera", "Dysentery", "Leptospirosis",
            "Typhoid", "Giardiasis", "Hepatitis_A", "Hepatitis_E",
        }
        assert set(DISEASE_RULES.keys()) == expected

    def test_cholera_rule_fires_on_contaminated_water(self):
        record = {
            "fecal_coliform_per_100ml": 100,
            "turbidity_ntu": 10,
            "residual_chlorine_mg_l": 0.05,
            "ph": 7.0,
        }
        assert DISEASE_RULES["Cholera"](record) is True

    def test_cholera_rule_does_not_fire_on_clean_water(self):
        record = {
            "fecal_coliform_per_100ml": 0,
            "turbidity_ntu": 0.5,
            "residual_chlorine_mg_l": 0.5,
            "ph": 7.5,
        }
        assert DISEASE_RULES["Cholera"](record) is False

    def test_leptospirosis_requires_flooding(self):
        record = {
            "flooding": 0,
            "avg_rainfall_mm": 300,
            "turbidity_ntu": 10,
        }
        assert DISEASE_RULES["Leptospirosis"](record) is False

    def test_leptospirosis_fires_on_flooding(self):
        record = {
            "flooding": 1,
            "avg_rainfall_mm": 200,
            "turbidity_ntu": 8,
        }
        assert DISEASE_RULES["Leptospirosis"](record) is True

    def test_hepatitis_e_requires_monsoon_season(self):
        record = {
            "flooding": 1,
            "fecal_coliform_per_100ml": 50,
            "season": "Summer",
        }
        assert DISEASE_RULES["Hepatitis_E"](record) is False

    def test_hepatitis_a_requires_no_toilet(self):
        record = {
            "fecal_coliform_per_100ml": 60,
            "toilet_access": 1,
            "handwashing_practice": "Never",
        }
        assert DISEASE_RULES["Hepatitis_A"](record) is False

    def test_hepatitis_a_fires_without_sanitation(self):
        record = {
            "fecal_coliform_per_100ml": 60,
            "toilet_access": 0,
            "handwashing_practice": "Never",
        }
        assert DISEASE_RULES["Hepatitis_A"](record) is True
