"""
src/synthetic_data_generator.py
---------------------------------
Generates a larger synthetic training dataset for the ML models.

IMPORTANT DISCLAIMER
--------------------
All records produced here are SYNTHETIC — they do not represent observed
measurements or confirmed disease events.  The disease labels are assigned
by documented, configurable scenario rules derived from public-health
literature (WHO 2017, IS:10500:2012).  The models trained on this data
learn these synthetic rules, NOT real-world outbreak patterns.

Design principles (per task spec section 5)
-------------------------------------------
- Reproducible random seed (default 42).
- Diverse parameter combinations — not simple duplication.
- record_origin = "synthetic" on every row.
- Disease-specific risk rules are separate and documented.
- Labels are never randomly assigned.
- Synthetic risk scores ≠ real-world disease probabilities.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Reproducible seed ────────────────────────────────────────────────────────
DEFAULT_SEED = 42

# ── Documented scenario rules (grounded in WHO 2017, IS:10500:2012) ──────────
# Each disease has threshold conditions that, when met, produce a "present" label.
# Rules are intentionally conservative to avoid false-positive labels at scale.
DISEASE_RULES = {
    # Cholera (14-day horizon): driven by fecal contamination + inadequate chlorination
    "Cholera": lambda r: (
        r["fecal_coliform_per_100ml"] > 50      # fecal contamination (WHO: 0 for safe)
        and r["turbidity_ntu"] > 4              # turbidity aids Vibrio cholerae survival
        and r["residual_chlorine_mg_l"] < 0.2   # inadequate disinfection
        and r["ph"] < 8.0                        # low pH favours Vibrio
    ),

    # Dysentery (14-day): Shigella / Entamoeba — fecal-oral, BOD-driven
    "Dysentery": lambda r: (
        r["fecal_coliform_per_100ml"] > 30
        and r["bod_mg_l"] > 3.0                  # organic load indicator
        and r["open_defecation_rate"] > 25        # sanitation driver
    ),

    # Leptospirosis (14-day): flooding + contaminated water contact
    "Leptospirosis": lambda r: (
        r["flooding"] == 1
        and r["avg_rainfall_mm"] > 150
        and r["turbidity_ntu"] > 5
    ),

    # Typhoid (30-day): Salmonella Typhi — nitrate proxy for fecal contamination
    "Typhoid": lambda r: (
        r["nitrate_mg_l"] > 30                   # proxy for sewage contamination
        and r["fecal_coliform_per_100ml"] > 20
        and r["sewage_treatment_pct"] < 40
    ),

    # Giardiasis (30-day): Giardia cysts, turbidity-driven
    "Giardiasis": lambda r: (
        r["turbidity_ntu"] > 5
        and r["tds_mg_l"] > 400
        and r["ph"] < 7.0                         # acidic water favours cysts
    ),

    # Hepatitis A (30-day): fecal-oral, poor sanitation
    "Hepatitis_A": lambda r: (
        r["fecal_coliform_per_100ml"] > 40
        and r["toilet_access"] == 0
        and r["handwashing_practice"] == "Never"
    ),

    # Hepatitis E (30-day): waterborne faecal-oral — monsoon flooding trigger
    "Hepatitis_E": lambda r: (
        r["flooding"] == 1
        and r["fecal_coliform_per_100ml"] > 25
        and r["season"] == "Monsoon"
    ),
}

STATES_REGIONS = [
    ("Chhattisgarh", "Central"), ("Uttar Pradesh", "North"),
    ("Bihar", "East"), ("Assam", "Northeast"), ("Kerala", "South"),
    ("Maharashtra", "West"), ("Rajasthan", "North"), ("West Bengal", "East"),
    ("Madhya Pradesh", "Central"), ("Telangana", "South"),
]

WATER_SOURCES = ["Piped", "River", "Borewell", "Open Well", "Pond", "Tanker"]
WATER_TREATMENTS = ["Untreated", "Chlorinated", "Filtered", "Boiled"]
HANDWASH_OPTIONS = ["Always", "Sometimes", "Never"]
SEASONS = {1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer",
           5: "Summer", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon",
           9: "Post-Monsoon", 10: "Post-Monsoon", 11: "Post-Monsoon", 12: "Winter"}


def _assign_disease(record: dict) -> str:
    """
    Check disease-specific rules in order.  Returns the first disease whose
    conditions are satisfied, or 'No_Disease' if none apply.
    """
    check_order = [
        "Cholera", "Dysentery", "Leptospirosis",
        "Typhoid", "Giardiasis", "Hepatitis_A", "Hepatitis_E",
    ]
    for disease in check_order:
        try:
            if DISEASE_RULES[disease](record):
                return disease
        except (KeyError, TypeError):
            continue
    return "No_Disease"


def generate_synthetic_training_data(
    n_samples: int = 1500,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic training dataset of `n_samples` rows.

    Parameters
    ----------
    n_samples : int
        Number of synthetic records to generate (default 1500).
    seed : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    pd.DataFrame with all training columns + record_origin = 'synthetic'.
    """
    rng = np.random.default_rng(seed)

    records = []
    for _ in range(n_samples):
        month = int(rng.integers(1, 13))
        season = SEASONS[month]
        state, region = STATES_REGIONS[int(rng.integers(0, len(STATES_REGIONS)))]
        flooding_bool = bool(rng.choice([0, 1], p=[0.85, 0.15]))
        rainfall_boost = rng.uniform(100, 300) if season == "Monsoon" else 0

        r = {
            # Location
            "state": state,
            "district": f"Dist_{rng.integers(1, 20):02d}",
            "region": region,
            "latitude": float(rng.uniform(8, 35)),
            "longitude": float(rng.uniform(68, 97)),
            "is_urban": int(rng.choice([0, 1], p=[0.55, 0.45])),
            "population_density": float(rng.uniform(50, 1500)),

            # Demographic
            "age": float(rng.uniform(5, 75)),
            "gender": rng.choice(["Male", "Female"]),

            # Water source / treatment
            "water_source": rng.choice(WATER_SOURCES),
            "water_treatment": rng.choice(WATER_TREATMENTS),

            # Water quality — drawn from realistic ranges
            "water_quality_index": float(rng.uniform(10, 95)),
            "ph": float(rng.uniform(5.5, 9.5)),
            "turbidity_ntu": float(rng.exponential(5)),  # most samples low, tail high
            "dissolved_oxygen_mg_l": float(rng.uniform(1, 14)),
            "bod_mg_l": float(rng.exponential(4)),
            "fecal_coliform_per_100ml": float(rng.exponential(80)),
            "total_coliform_per_100ml": float(rng.exponential(300)),
            "tds_mg_l": float(rng.uniform(80, 1500)),
            "nitrate_mg_l": float(rng.exponential(15)),
            "fluoride_mg_l": float(rng.uniform(0.1, 3.5)),
            "arsenic_ug_l": float(rng.exponential(8)),
            "residual_chlorine_mg_l": float(rng.uniform(0, 0.6)),

            # Sanitation
            "open_defecation_rate": float(rng.uniform(0, 80)),
            "toilet_access": int(rng.choice([0, 1], p=[0.3, 0.7])),
            "sewage_treatment_pct": float(rng.uniform(0, 95)),
            "handwashing_practice": rng.choice(HANDWASH_OPTIONS,
                                               p=[0.4, 0.45, 0.15]),
            # Weather
            "month": month,
            "season": season,
            "avg_temperature_c": float(rng.uniform(8, 42)),
            "avg_rainfall_mm": float(rng.uniform(0, 150) + rainfall_boost),
            "avg_humidity_pct": float(rng.uniform(30, 95)),
            "flooding": int(flooding_bool),

            # Symptoms (derived below from disease label)
            "symptom_diarrhea": 0, "symptom_vomiting": 0, "symptom_fever": 0,
            "symptom_abdominal_pain": 0, "symptom_dehydration": 0,
            "symptom_jaundice": 0, "symptom_bloody_stool": 0, "symptom_skin_rash": 0,
        }

        disease = _assign_disease(r)
        r["disease"] = disease

        # Symptom flags consistent with disease (informational, not features)
        if disease in ("Cholera", "Dysentery"):
            r["symptom_diarrhea"] = 1
            r["symptom_dehydration"] = 1
        if disease == "Dysentery":
            r["symptom_bloody_stool"] = 1
        if disease in ("Typhoid", "Leptospirosis"):
            r["symptom_fever"] = 1
        if disease in ("Hepatitis_A", "Hepatitis_E"):
            r["symptom_jaundice"] = 1
            r["symptom_fever"] = 1
        if disease in ("Giardiasis", "Typhoid"):
            r["symptom_abdominal_pain"] = 1
        if disease == "Leptospirosis":
            r["symptom_skin_rash"] = 1

        r["record_origin"] = "synthetic"
        r["source_file"] = "synthetic_data_generator.py"
        records.append(r)

    df = pd.DataFrame(records)
    # Drop internal helper column not in training schema
    df = df.drop(columns=["residual_chlorine_mg_l"], errors="ignore")

    logger.info(
        "Generated %d synthetic records. Disease distribution:\n%s",
        len(df), df["disease"].value_counts().to_string(),
    )
    return df


def save_synthetic_dataset(n_samples: int = 1500, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Generate and save to data/processed/synthetic_disease_scenarios.csv."""
    df = generate_synthetic_training_data(n_samples=n_samples, seed=seed)
    out = PROCESSED / "synthetic_disease_scenarios.csv"
    df.to_csv(out, index=False)
    logger.info("Synthetic dataset saved: %s", out)
    return df
