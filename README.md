# 💧 Waterborne Disease Early Warning System

A Streamlit dashboard for the **AI-Based Early Warning System for Waterborne Disease
Outbreak Prediction** project. This app is a presentation/deployment layer over the
already-trained models from the project notebook — **no retraining, no changed ML
logic**. It reuses the exact saved model pipelines and reproduces the notebook's
preprocessing and feature-engineering steps so predictions match the notebook exactly.

## Installation

```bash
pip install -r requirements.txt
```

⚠️ **scikit-learn version matters.** `requirements.txt` pins `scikit-learn==1.6.1`
because the saved `.pkl` files were serialized with that exact version. Installing a
newer scikit-learn (1.8+/1.9+) will make the models fail to load with an
`AttributeError: _RemainderColsList` error — this was confirmed while building this
app. Don't upgrade scikit-learn independently of re-saving the models.

## Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Folder structure

```
Waterborne_Early_Warning_System/
│
├── app.py                     # Main Streamlit app (Home, Prediction Dashboard, District Risk Map)
│
├── models/
│   ├── waterborne_outbreak_model.pkl   # Generic alias (= 30-day Random Forest model)
│   ├── model_14d.pkl                   # 14-Day Early Warning model (tuned Gradient Boosting)
│   ├── model_30d.pkl                   # 30-Day Early Warning model (tuned Random Forest)
│   └── scaler.pkl                      # Standalone fitted preprocessor (reference only —
│                                        #   each model pipeline already embeds its own
│                                        #   fitted preprocessing step, so this is not
│                                        #   re-applied at prediction time)
│
├── utils/
│   ├── preprocessing.py        # Reproduces notebook Section 4 cleaning steps; computes
│   │                            #   REF_STATS / REGION_RISK_LOOKUP from data/dataset.csv
│   ├── feature_engineering.py  # Reproduces notebook Section 6/11 engineered risk-score
│   │                            #   features (exact formulas/weights) + risk-level logic
│   ├── prediction.py           # Loads models, wraps predict_outbreak_risk() and
│   │                            #   get_feature_importance()
│   └── recommendations.py      # Risk-level -> recommended-actions text
│
├── data/
│   └── dataset.csv             # Training dataset (used to recompute reference stats,
│                                #   NOT to retrain any model)
│
├── assets/
│   └── style.css                # Custom visual identity (deep-teal/aqua public-health theme)
│
├── requirements.txt
└── README.md
```

## What each page does

- **🏠 Home** — problem statement, objective, how the AI predicts outbreaks, and the
  14-day vs 30-day horizon explanation (grounded in each disease's incubation period,
  exactly as documented in the notebook).
- **🔬 Prediction Dashboard** — full input form (location, water quality, weather,
  sanitation, water source) → on submit, shows:
  - 14-day and 30-day risk probability, risk level (Low/Medium/High), and message
  - Gauge charts + a 14-day vs 30-day comparison bar chart (Plotly)
  - Feature importance charts for both horizon models
  - Dynamic risk recommendations
  - A downloadable PDF prediction report
- **🗺️ District Risk Map** — an interactive Folium map colouring every sampled district
  green/yellow/red by model-estimated risk, with a toggle between the 14-day and
  30-day model.

## How predictions stay consistent with the notebook

1. A submitted form is combined with dataset-wide defaults for any optional fields
   (age, gender, etc. — same fallback behaviour as the notebook's `predict_outbreak_risk()`).
2. `utils/feature_engineering.engineer_features()` recreates every engineered column
   (water contamination score, weather risk, seasonal risk, environmental risk,
   sanitation risk, location risk, interaction terms, composite score) using the exact
   weights and formulas from the notebook.
3. The engineered row is reduced to the same 34 numeric + 6 categorical columns, in the
   same order, that the saved pipelines were trained on (extracted directly from the
   pipelines' `ColumnTransformer`).
4. `model_14d.pkl` / `model_30d.pkl` — each a full scikit-learn `Pipeline` containing its
   own fitted preprocessing step — are called directly via `.predict_proba()`. The
   standalone `scaler.pkl` is not reapplied separately, since doing so would
   double-transform the input.

## Known limitations (carried over from the notebook)

- The training dataset is cross-sectional, not a per-location day-level time series —
  the 14-day/30-day horizons are operationalised via disease incubation-period
  groupings, not sequential forecasting.
- Risk-level thresholds (35% / 65%) are reasonable defaults; a production deployment
  should calibrate them with public-health domain experts.
- The District Risk Map scores one aggregated (mean/mode) profile per district from the
  training sample — it is illustrative, not a live/real-time feed.
