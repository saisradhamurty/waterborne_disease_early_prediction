"""
prediction.py
--------------
Loads the pre-trained model pipelines (NOT retrained here) and wraps them into the
same predict_outbreak_risk() interface defined in Section 11 of the training notebook.

Each .pkl is a full scikit-learn Pipeline (ColumnTransformer preprocessing + tuned
classifier). We therefore pass the engineered feature row straight into
`pipeline.predict_proba(...)` — we do NOT separately apply models/scaler.pkl, since
that would double-transform the input (each pipeline already contains its own fitted
preprocessor). scaler.pkl is loaded for completeness/inspection only.
"""

import pandas as pd
import joblib
import streamlit as st

from utils.feature_engineering import (
    ALL_MODEL_FEATURES,
    engineer_features,
    classify_risk_level,
)
from utils.preprocessing import get_reference_artifacts, get_input_defaults

MODEL_14D_PATH = "models/model_14d.pkl"
MODEL_30D_PATH = "models/model_30d.pkl"
MODEL_GENERIC_PATH = "models/waterborne_outbreak_model.pkl"
SCALER_PATH = "models/scaler.pkl"


@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    """Loads the saved pipelines exactly as they were trained. No retraining occurs here."""
    model_14d = joblib.load(MODEL_14D_PATH)
    model_30d = joblib.load(MODEL_30D_PATH)
    scaler = joblib.load(SCALER_PATH)  # standalone artifact, kept for reference/inspection
    return model_14d, model_30d, scaler


def _build_message(state: str, district: str, risk_level: str, horizon_days: int) -> str:
    if risk_level == "High Risk":
        return (
            f"HIGH probability of a waterborne disease outbreak in {state}, {district} "
            f"within the next {horizon_days} days. Immediate intervention recommended "
            f"(water testing, chlorination/boil-water advisory, sanitation follow-up)."
        )
    elif risk_level == "Medium Risk":
        return (
            f"MODERATE outbreak risk detected in {state}, {district} for the next "
            f"{horizon_days} days. Increased monitoring and precautionary water safety "
            f"measures are advised."
        )
    return (
        f"LOW outbreak risk in {state}, {district} for the next {horizon_days} days. "
        f"Continue routine surveillance."
    )


def predict_outbreak_risk(
    state: str,
    district: str,
    water_quality_params: dict,
    environmental_params: dict,
    sanitation_params: dict,
    location_params: dict = None,
    demographic_params: dict = None,
) -> dict:
    """
    Early Warning System prediction function — same contract as the notebook's
    predict_outbreak_risk() (Section 11).

    Returns a dict with keys 'location', '14_day', '30_day', each carrying
    risk_probability_pct, risk_level, and a plain-language message.
    """
    location_params = location_params or {}
    demographic_params = demographic_params or {}

    model_14d, model_30d, _ = load_models()
    ref_stats, region_risk_lookup = get_reference_artifacts()
    defaults = get_input_defaults()

    record = {}
    record.update(water_quality_params)
    record.update(environmental_params)
    record.update(sanitation_params)
    record.update(location_params)
    record.update(demographic_params)

    for k, v in defaults.items():
        record.setdefault(k, v)

    engineered = engineer_features(record, ref_stats, region_risk_lookup)
    input_row = pd.DataFrame([engineered])[ALL_MODEL_FEATURES]

    proba_14d = float(model_14d.predict_proba(input_row)[0, 1])
    proba_30d = float(model_30d.predict_proba(input_row)[0, 1])

    risk_level_14d = classify_risk_level(proba_14d)
    risk_level_30d = classify_risk_level(proba_30d)

    return {
        "location": {"state": state, "district": district},
        "input_row": input_row,
        "14_day": {
            "risk_probability_pct": round(proba_14d * 100, 2),
            "risk_level": risk_level_14d,
            "message": _build_message(state, district, risk_level_14d, 14),
        },
        "30_day": {
            "risk_probability_pct": round(proba_30d * 100, 2),
            "risk_level": risk_level_30d,
            "message": _build_message(state, district, risk_level_30d, 30),
        },
    }


def get_feature_importance(horizon: str = "14d", top_n: int = 10) -> pd.Series:
    """
    Extracts feature importances from the tuned pipeline for the requested horizon
    ('14d' or '30d'). Works for both tree-based (feature_importances_) and linear
    (coef_) models, matching the notebook's Section 12 get_feature_importance().
    """
    model_14d, model_30d, _ = load_models()
    pipeline = model_14d if horizon == "14d" else model_30d

    clf = pipeline.named_steps["clf"]
    prep = pipeline.named_steps["prep"]

    from utils.feature_engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES

    cat_names = list(prep.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    all_names = NUMERIC_FEATURES + cat_names

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = abs(clf.coef_[0])
    else:
        raise ValueError("Model does not expose feature importances.")

    series = pd.Series(importances, index=all_names).sort_values(ascending=False)
    return series.head(top_n)
