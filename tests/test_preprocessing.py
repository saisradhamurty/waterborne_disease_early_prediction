"""
tests/test_preprocessing.py
-----------------------------
Tests for src/preprocessing.py — cleaning pipeline, reference artifacts,
and target variable creation.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    clean_training_data,
    compute_reference_artifacts,
    FAST_ONSET_DISEASES,
    SLOW_ONSET_DISEASES,
    OUTLIER_COLS,
)


class TestCleanTrainingData:
    @pytest.fixture(scope="class")
    def clean_df(self):
        return clean_training_data()

    def test_returns_dataframe(self, clean_df):
        assert isinstance(clean_df, pd.DataFrame)

    def test_no_null_in_numeric_cols(self, clean_df):
        num_cols = clean_df.select_dtypes(include=[np.number]).columns
        nulls = clean_df[num_cols].isnull().sum()
        assert nulls.sum() == 0, f"Null values remain in numeric cols:\n{nulls[nulls > 0]}"

    def test_target_columns_created(self, clean_df):
        assert "outbreak_risk_14d" in clean_df.columns
        assert "outbreak_risk_30d" in clean_df.columns

    def test_target_14d_binary(self, clean_df):
        vals = set(clean_df["outbreak_risk_14d"].unique())
        assert vals.issubset({0, 1}), f"Non-binary values in outbreak_risk_14d: {vals}"

    def test_target_30d_binary(self, clean_df):
        vals = set(clean_df["outbreak_risk_30d"].unique())
        assert vals.issubset({0, 1}), f"Non-binary values in outbreak_risk_30d: {vals}"

    def test_fast_onset_diseases_map_to_14d(self, clean_df):
        for d in FAST_ONSET_DISEASES:
            subset = clean_df[clean_df["disease"] == d]
            if len(subset):
                assert (subset["outbreak_risk_14d"] == 1).all(), \
                    f"{d} should have outbreak_risk_14d=1"

    def test_slow_onset_diseases_map_to_30d(self, clean_df):
        for d in SLOW_ONSET_DISEASES:
            subset = clean_df[clean_df["disease"] == d]
            if len(subset):
                assert (subset["outbreak_risk_30d"] == 1).all(), \
                    f"{d} should have outbreak_risk_30d=1"

    def test_no_disease_has_both_targets_zero(self, clean_df):
        no_dis = clean_df[clean_df["disease"] == "No_Disease"]
        if len(no_dis):
            assert (no_dis["outbreak_risk_14d"] == 0).all()
            assert (no_dis["outbreak_risk_30d"] == 0).all()

    def test_cyclical_month_columns(self, clean_df):
        assert "month_sin" in clean_df.columns
        assert "month_cos" in clean_df.columns
        # Values must be in [-1, 1]
        assert clean_df["month_sin"].between(-1.0001, 1.0001).all()
        assert clean_df["month_cos"].between(-1.0001, 1.0001).all()

    def test_outlier_capping_applied(self, clean_df):
        """After IQR capping (k=3) no value should be extreme outlier."""
        for col in OUTLIER_COLS:
            if col not in clean_df.columns:
                continue
            q1, q3 = clean_df[col].quantile(0.25), clean_df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
            assert (clean_df[col] >= lower - 1e-6).all(), f"{col} has values below IQR lower cap"
            assert (clean_df[col] <= upper + 1e-6).all(), f"{col} has values above IQR upper cap"

    def test_no_exact_duplicates(self, clean_df):
        dupes = clean_df.duplicated().sum()
        assert dupes == 0, f"{dupes} duplicate rows remain after cleaning"

    def test_accepts_custom_dataframe(self, clean_df):
        """clean_training_data should accept an already-loaded df without re-reading."""
        result = clean_training_data(clean_df.copy())
        assert isinstance(result, pd.DataFrame)


class TestComputeReferenceArtifacts:
    @pytest.fixture(scope="class")
    def artifacts(self):
        df = clean_training_data()
        return compute_reference_artifacts(df)

    def test_returns_two_dicts(self, artifacts):
        ref_stats, region_risk = artifacts
        assert isinstance(ref_stats, dict)
        assert isinstance(region_risk, dict)

    def test_ref_stats_has_turbidity(self, artifacts):
        ref_stats, _ = artifacts
        assert "turbidity_ntu" in ref_stats

    def test_ref_stats_min_le_max(self, artifacts):
        ref_stats, _ = artifacts
        for col, (lo, hi) in ref_stats.items():
            assert lo <= hi, f"ref_stats[{col}]: min={lo} > max={hi}"

    def test_region_risk_values_in_0_1(self, artifacts):
        _, region_risk = artifacts
        for region, val in region_risk.items():
            assert 0.0 <= val <= 1.0, f"Region risk out of range for {region}: {val}"

    def test_known_regions_present(self, artifacts):
        _, region_risk = artifacts
        expected = {"North", "South", "East", "West", "Central", "Northeast"}
        found = set(region_risk.keys())
        assert len(found & expected) >= 3, \
            f"Expected at least 3 standard regions, found: {found}"


class TestDiseaseHorizonSets:
    def test_no_overlap_between_horizons(self):
        overlap = FAST_ONSET_DISEASES & SLOW_ONSET_DISEASES
        assert len(overlap) == 0, f"Diseases appear in both horizons: {overlap}"

    def test_fast_onset_contains_cholera(self):
        assert "Cholera" in FAST_ONSET_DISEASES

    def test_slow_onset_contains_typhoid(self):
        assert "Typhoid" in SLOW_ONSET_DISEASES
