"""
tests/test_feature_engineering.py
-----------------------------------
Tests for utils/feature_engineering.py — verifies that engineer_features()
produces the correct output shape, column set, and score ranges.
"""

import pandas as pd
import pytest

from utils.feature_engineering import (
    engineer_features,
    ALL_MODEL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    SEASON_RISK_MAP,
    HANDWASH_RISK_MAP,
    RISK_LEVEL_THRESHOLDS,
    RISK_COLORS,
)
from src.preprocessing import clean_training_data, compute_reference_artifacts


@pytest.fixture(scope="module")
def artifacts():
    df = clean_training_data()
    return compute_reference_artifacts(df)


@pytest.fixture(scope="module")
def sample_record():
    """A minimal valid record to run through the feature engineer."""
    return {
        "state": "Chhattisgarh",
        "district": "Durg",
        "region": "Central",
        "latitude": 21.19,
        "longitude": 81.28,
        "is_urban": 1,
        "population_density": 500.0,
        "age": 30.0,
        "gender": "Female",
        "water_source": "Piped",
        "water_treatment": "Chlorinated",
        "water_quality_index": 60.0,
        "ph": 7.2,
        "turbidity_ntu": 2.0,
        "dissolved_oxygen_mg_l": 7.0,
        "bod_mg_l": 1.5,
        "fecal_coliform_per_100ml": 5.0,
        "total_coliform_per_100ml": 20.0,
        "tds_mg_l": 300.0,
        "nitrate_mg_l": 10.0,
        "fluoride_mg_l": 0.5,
        "arsenic_ug_l": 1.0,
        "open_defecation_rate": 15.0,
        "toilet_access": 1,
        "sewage_treatment_pct": 60.0,
        "handwashing_practice": "Always",
        "month": 7,
        "season": "Monsoon",
        "avg_temperature_c": 28.0,
        "avg_rainfall_mm": 200.0,
        "avg_humidity_pct": 80.0,
        "flooding": 0,
        "month_sin": 0.866,
        "month_cos": -0.5,
        "outbreak_risk_14d": 0,
        "outbreak_risk_30d": 0,
        "disease": "No_Disease",
    }


class TestAllModelFeatures:
    def test_total_feature_count(self):
        assert len(ALL_MODEL_FEATURES) == 40, \
            f"Expected 40 features, got {len(ALL_MODEL_FEATURES)}"

    def test_no_duplicates(self):
        assert len(ALL_MODEL_FEATURES) == len(set(ALL_MODEL_FEATURES))

    def test_numeric_plus_categorical_equals_all(self):
        assert len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) == len(ALL_MODEL_FEATURES)


class TestEngineerFeatures:
    def test_returns_dict(self, sample_record, artifacts):
        ref_stats, region_risk_lookup = artifacts
        result = engineer_features(sample_record, ref_stats, region_risk_lookup)
        assert isinstance(result, dict)

    def test_all_model_features_present(self, sample_record, artifacts):
        ref_stats, region_risk_lookup = artifacts
        result = engineer_features(sample_record, ref_stats, region_risk_lookup)
        missing = [f for f in ALL_MODEL_FEATURES if f not in result]
        assert len(missing) == 0, f"Missing model features: {missing}"

    def test_scores_in_0_100(self, sample_record, artifacts):
        ref_stats, region_risk_lookup = artifacts
        result = engineer_features(sample_record, ref_stats, region_risk_lookup)
        score_cols = [
            "water_contamination_score", "environmental_risk_score",
            "sanitation_risk_score", "composite_outbreak_risk_score",
        ]
        for col in score_cols:
            if col in result:
                val = result[col]
                assert 0 <= val <= 100, f"{col} out of range: {val}"

    def test_flooding_record_has_higher_env_risk(self, sample_record, artifacts):
        ref_stats, region_risk_lookup = artifacts
        no_flood = {**sample_record, "flooding": 0}
        flood = {**sample_record, "flooding": 1}
        r_no = engineer_features(no_flood, ref_stats, region_risk_lookup)
        r_yes = engineer_features(flood, ref_stats, region_risk_lookup)
        assert r_yes.get("environmental_risk_score", 0) >= r_no.get("environmental_risk_score", 0), \
            "Flooding should raise the environmental risk score"

    def test_deterministic(self, sample_record, artifacts):
        ref_stats, region_risk_lookup = artifacts
        r1 = engineer_features(sample_record, ref_stats, region_risk_lookup)
        r2 = engineer_features(sample_record, ref_stats, region_risk_lookup)
        assert r1 == r2, "engineer_features must be deterministic"


class TestRiskConstants:
    def test_risk_thresholds(self):
        assert RISK_LEVEL_THRESHOLDS["low"] < RISK_LEVEL_THRESHOLDS["high"]

    def test_season_risk_map_covers_four_seasons(self):
        expected = {"Monsoon", "Post-Monsoon", "Summer", "Winter"}
        assert set(SEASON_RISK_MAP.keys()) == expected

    def test_handwash_risk_map(self):
        assert "Never" in HANDWASH_RISK_MAP
        assert "Always" in HANDWASH_RISK_MAP
        # Never should have higher risk than Always
        assert HANDWASH_RISK_MAP["Never"] > HANDWASH_RISK_MAP["Always"]

    def test_risk_colors_defined(self):
        for level in ("Low Risk", "Medium Risk", "High Risk"):
            assert level in RISK_COLORS
            assert RISK_COLORS[level].startswith("#")
