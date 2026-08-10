"""
recommendations.py
-------------------
Dynamic, risk-level-based recommendation text for the dashboard and PDF report.
Pure presentation logic — does not touch the model or the data pipeline.
"""

HIGH_RISK_ACTIONS = [
    "Boil drinking water for at least 1 minute before consumption",
    "Avoid using untreated surface water (rivers, ponds, open wells) for drinking or cooking",
    "Increase chlorination dosage at the water source / storage point immediately",
    "Contact the local health authority or district health officer to report the risk",
    "Conduct on-site water testing (fecal coliform, turbidity) within 48 hours",
    "Issue a public boil-water / safe-water advisory to the affected community",
    "Pre-position ORS (oral rehydration salts) and basic medical supplies locally",
]

MEDIUM_RISK_ACTIONS = [
    "Increase the frequency of water quality monitoring at this location",
    "Review and reinforce chlorination / water treatment practices",
    "Communicate basic hygiene precautions (handwashing, safe storage) to residents",
    "Re-check sanitation infrastructure (toilet access, sewage handling) for gaps",
    "Re-assess risk after the next rainfall/weather update, especially if flooding is forecast",
]

LOW_RISK_ACTIONS = [
    "Water and sanitation conditions currently support safe use",
    "Continue routine water quality surveillance on the normal schedule",
    "No immediate intervention required — maintain existing safeguards",
]


def get_recommendations(risk_level: str) -> list:
    """Returns a list of recommendation strings for the given risk level."""
    if risk_level == "High Risk":
        return HIGH_RISK_ACTIONS
    elif risk_level == "Medium Risk":
        return MEDIUM_RISK_ACTIONS
    return LOW_RISK_ACTIONS


def get_recommendation_summary(risk_level: str) -> str:
    """One-line summary banner text for the given risk level."""
    if risk_level == "High Risk":
        return "⚠️ High risk detected — act now to reduce exposure."
    elif risk_level == "Medium Risk":
        return "🟡 Moderate risk — increase monitoring and precaution."
    return "✅ Low risk — water and sanitation conditions look safe."
