"""
feature_engineering.py
-----------------------
Reproduces the engineered risk-score features from Section 6 / Section 11 of the
training notebook, EXACTLY (same formulas, same weights, same column order) — so a
new user-submitted record is transformed into the identical feature vector the models
were trained on. Nothing here is re-derived or approximated; every weight below was
copied from the notebook's engineer_features() function.

This module does not fit or change any model. It only reproduces deterministic,
already-decided arithmetic.
"""

import numpy as np

# Final feature lists used by every saved model pipeline (order matters — this is the
# exact order extracted from the trained ColumnTransformer inside the .pkl files).
NUMERIC_FEATURES = [
    "latitude", "longitude", "is_urban", "population_density", "age",
    "water_quality_index", "ph", "turbidity_ntu", "dissolved_oxygen_mg_l", "bod_mg_l",
    "fecal_coliform_per_100ml", "total_coliform_per_100ml", "tds_mg_l", "nitrate_mg_l",
    "fluoride_mg_l", "arsenic_ug_l", "open_defecation_rate", "toilet_access",
    "sewage_treatment_pct", "avg_temperature_c", "avg_rainfall_mm", "avg_humidity_pct",
    "flooding", "month_sin", "month_cos",
    "water_contamination_score", "weather_risk_score", "seasonal_risk_score",
    "environmental_risk_score", "sanitation_risk_score", "location_risk_score",
    "rain_flood_interaction", "contamination_sanitation_interaction",
    "composite_outbreak_risk_score",
]

CATEGORICAL_FEATURES = [
    "region", "gender", "water_source", "water_treatment", "handwashing_practice", "season",
]

ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

SEASON_RISK_MAP = {"Monsoon": 1.0, "Post-Monsoon": 0.8, "Summer": 0.5, "Winter": 0.3}
HANDWASH_RISK_MAP = {"Always": 0.0, "Sometimes": 0.5, "Never": 1.0}

RISK_LEVEL_THRESHOLDS = {"low": 0.35, "high": 0.65}


def _minmax_single(value: float, series_min: float, series_max: float) -> float:
    return (value - series_min) / (series_max - series_min + 1e-9)


def engineer_features(record: dict, ref_stats: dict, region_risk_lookup: dict) -> dict:
    """
    Recreates every Section 6 engineered feature for a single new input record.

    Parameters
    ----------
    record : dict
        Raw input fields (water quality, weather, sanitation, location, demographic).
    ref_stats : dict
        {column: (min, max)} from utils.preprocessing.get_reference_artifacts().
    region_risk_lookup : dict
        {region: historical outbreak rate} from utils.preprocessing.get_reference_artifacts().

    Returns
    -------
    dict — the input record plus every engineered column, ready to be reduced to
    ALL_MODEL_FEATURES and passed straight into a model pipeline's .predict_proba().
    """
    r = dict(record)

    def mm(col, val):
        lo, hi = ref_stats[col]
        return _minmax_single(val, lo, hi)

    # 6.2.1 Water Contamination Score (higher = more contaminated)
    r["water_contamination_score"] = (
        mm("turbidity_ntu", r["turbidity_ntu"]) * 0.15
        + mm("fecal_coliform_per_100ml", r["fecal_coliform_per_100ml"]) * 0.25
        + mm("total_coliform_per_100ml", r["total_coliform_per_100ml"]) * 0.15
        + mm("bod_mg_l", r["bod_mg_l"]) * 0.15
        + mm("tds_mg_l", r["tds_mg_l"]) * 0.10
        + mm("nitrate_mg_l", r["nitrate_mg_l"]) * 0.05
        + mm("fluoride_mg_l", r["fluoride_mg_l"]) * 0.05
        + mm("arsenic_ug_l", r["arsenic_ug_l"]) * 0.05
        + (1 - mm("water_quality_index", r["water_quality_index"])) * 0.05
    ) * 100

    # 6.2.2 Weather Risk Score (short-term hazard)
    r["weather_risk_score"] = (
        mm("avg_rainfall_mm", r["avg_rainfall_mm"]) * 0.4
        + mm("avg_humidity_pct", r["avg_humidity_pct"]) * 0.3
        + r["flooding"] * 0.3
    ) * 100

    # 6.2.3 Seasonal Risk (baseline outbreak propensity by season)
    r["seasonal_risk_score"] = SEASON_RISK_MAP.get(r["season"], 0.5) * 100

    # 6.2.4 Environmental Risk = weather (short-term) + seasonal (baseline)
    r["environmental_risk_score"] = r["weather_risk_score"] * 0.6 + r["seasonal_risk_score"] * 0.4

    # 6.2.5 Sanitation Risk Score (higher = worse sanitation)
    r["sanitation_risk_score"] = (
        mm("open_defecation_rate", r["open_defecation_rate"]) * 0.35
        + (1 - r["toilet_access"]) * 0.25
        + (1 - mm("sewage_treatment_pct", r["sewage_treatment_pct"])) * 0.25
        + HANDWASH_RISK_MAP.get(r["handwashing_practice"], 0.5) * 0.15
    ) * 100

    # 6.2.7 Location-Based Risk (region's historical outbreak rate, rescaled 0-100)
    region_min = min(region_risk_lookup.values())
    region_max = max(region_risk_lookup.values())
    region_risk = region_risk_lookup.get(r["region"], float(np.mean(list(region_risk_lookup.values()))))
    r["location_risk_score"] = _minmax_single(region_risk, region_min, region_max) * 100

    # 6.2.8 Early-warning interaction features
    r["rain_flood_interaction"] = r["avg_rainfall_mm"] * r["flooding"]
    r["contamination_sanitation_interaction"] = (
        r["water_contamination_score"] * r["sanitation_risk_score"] / 100
    )

    # 6.2.9 Composite Outbreak Risk Score (single 0-100 summary number)
    r["composite_outbreak_risk_score"] = (
        r["water_contamination_score"] * 0.35
        + r["environmental_risk_score"] * 0.25
        + r["sanitation_risk_score"] * 0.25
        + r["location_risk_score"] * 0.15
    )

    # 4.4 Cyclical month encoding
    r["month_sin"] = np.sin(2 * np.pi * r["month"] / 12)
    r["month_cos"] = np.cos(2 * np.pi * r["month"] / 12)

    return r


def classify_risk_level(probability: float) -> str:
    """Low / Medium / High, using the same 35% / 65% thresholds as the notebook (Section 11)."""
    if probability < RISK_LEVEL_THRESHOLDS["low"]:
        return "Low Risk"
    elif probability < RISK_LEVEL_THRESHOLDS["high"]:
        return "Medium Risk"
    else:
        return "High Risk"


RISK_COLORS = {"Low Risk": "#2E7D32", "Medium Risk": "#F9A825", "High Risk": "#C62828"}
