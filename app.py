"""
app.py
------
AI-Based Early Warning System for Waterborne Disease Outbreak Prediction — Streamlit app.

This app is a thin presentation layer over the ALREADY-TRAINED models produced by the
project notebook. It does not retrain anything or change any ML logic:
  - utils/preprocessing.py       -> reproduces the notebook's data-cleaning steps
  - utils/feature_engineering.py -> reproduces the notebook's engineered risk-score features
  - utils/prediction.py          -> loads the saved .pkl pipelines and runs predict_outbreak_risk()
  - utils/recommendations.py     -> risk-level -> recommended-actions text

UI NOTE: This revision only reorganises navigation/pages and rewords copy. Every call into
utils/* keeps the exact same function names and arguments as before — no prediction,
preprocessing, feature-engineering or model logic was touched.

Run with:  streamlit run app.py
"""

import hashlib
import io
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import folium
from streamlit_folium import st_folium

from utils.preprocessing import load_clean_dataset, get_input_defaults
from utils.feature_engineering import RISK_COLORS
from utils.prediction import predict_outbreak_risk, get_feature_importance, load_models
from utils.recommendations import get_recommendations, get_recommendation_summary

# ----------------------------------------------------------------------------
# Page config + global styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Waterborne Disease Early Warning System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass  # app still works without custom styling

RISK_CSS_CLASS = {"Low Risk": "low", "Medium Risk": "medium", "High Risk": "high"}

STATE_OPTIONS = [
    "Andaman and Nicobar", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chandigarh", "Chhattisgarh", "Dadra Nagar Haveli and Daman Diu", "Delhi", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand",
    "Karnataka", "Kerala", "Ladakh", "Lakshadweep", "Madhya Pradesh", "Maharashtra",
    "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Puducherry", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal",
]
REGION_OPTIONS = ["North", "South", "East", "West", "Central", "Northeast"]
WATER_SOURCE_OPTIONS = ["Piped", "River", "Borewell", "Open Well", "Pond", "Tanker", "Rainwater"]
WATER_TREATMENT_OPTIONS = ["Untreated", "Chlorinated", "Filtered", "Boiled"]
HANDWASHING_OPTIONS = ["Always", "Sometimes", "Never"]
SEASON_OPTIONS = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
GENDER_OPTIONS = ["Female", "Male"]

DISCLAIMER_TEXT = (
    "AI-generated risk assessment for early public-health monitoring. It does not "
    "replace laboratory confirmation, medical diagnosis, or official public-health "
    "surveillance."
)

NAV_PAGES = [
    "🏠 Home",
    "📊 Dashboard",
    "🔬 Risk Assessment",
    "🗺️ District Risk Map",
    "📈 Risk Analysis",
    "📋 Reports",
    "💡 Recommendations",
    "ℹ️ About",
]

# ----------------------------------------------------------------------------
# Small navigation helpers (session-state based, no ML logic here)
# ----------------------------------------------------------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = NAV_PAGES[0]

# ---- DEMO / LOCAL authentication state -------------------------------------
# NOTE: This is a simple in-memory, session-only demo authentication system for
# presentation purposes. It is NOT production-grade auth (no persistence, no
# email verification, no rate-limiting, no secure session/cookie handling).
# Passwords are never stored in plain text -- they are hashed (see hash_password
# below) before being kept in st.session_state.users_db. This is kept modular
# (all read/write of "users" and "auth_user" go through the helpers below) so a
# real provider (e.g. Firebase Auth, Auth0, a proper backend + DB) can be swapped
# in later without touching the rest of the app.
if "users_db" not in st.session_state:
    # Pre-seeded demo account so reviewers can log in without signing up first.
    st.session_state.users_db = {
        "demo@example.com": {
            "full_name": "Demo Analyst",
            "email": "demo@example.com",
            "password_hash": hashlib.sha256("demo1234".encode("utf-8")).hexdigest(),
            "user_type": "Public User",
        }
    }
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None  # None = signed out; dict = signed in / guest


def hash_password(password: str) -> str:
    """Demo-only password hashing (SHA-256). Not a substitute for a real
    password-hashing scheme (e.g. bcrypt/argon2) in production."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def find_user_by_login(login_id: str):
    """Look up a demo user by email or full name (case-insensitive)."""
    login_id = (login_id or "").strip().lower()
    for email, u in st.session_state.users_db.items():
        if email.lower() == login_id or u["full_name"].strip().lower() == login_id:
            return u
    return None


def goto(page_name: str):
    """Switch the active page and rerun. Pure UI navigation helper."""
    st.session_state.current_page = page_name
    st.rerun()


def get_worst_level(r14: dict, r30: dict) -> str:
    """Combine the two horizon risk levels into a single overall level for
    recommendations / reports. Does not touch model outputs, just picks the
    more severe of the two already-computed labels."""
    if "High Risk" in (r14["risk_level"], r30["risk_level"]):
        return "High Risk"
    if "Medium Risk" in (r14["risk_level"], r30["risk_level"]):
        return "Medium Risk"
    return "Low Risk"


def render_disclaimer():
    st.markdown(
        f'<div class="ews-disclaimer">⚠️ <strong>Disclaimer:</strong> {DISCLAIMER_TEXT}</div>',
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="ews-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def no_assessment_prompt(destination_label: str):
    """Shown on Risk Analysis / Reports / Recommendations when no prediction
    has been run yet this session."""
    st.markdown(
        f"""
        <div class="ews-card ews-empty-state">
            <p class="ews-empty-icon">🔍</p>
            <h4>No risk assessment yet</h4>
            <p>Run a risk assessment first, then come back here to see {destination_label}.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🔬 Go to Risk Assessment", type="primary"):
        goto("🔬 Risk Assessment")


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
# NOTE: navigation is deliberately built from st.button (not st.radio). A radio's
# selection is persisted by Streamlit under its own widget key across reruns, which
# would fight with the Account/Logout buttons below overriding the active page —
# on the next rerun the radio would just reassert its last-clicked value. Buttons
# driven entirely through st.session_state.current_page avoid that conflict.
st.sidebar.markdown('<div class="sidebar-brand">💧 Waterborne EWS</div>', unsafe_allow_html=True)

for _label in NAV_PAGES:
    _is_active = st.session_state.current_page == _label
    if st.sidebar.button(
        _label,
        key=f"nav_{_label}",
        use_container_width=True,
        type="primary" if _is_active else "secondary",
    ):
        goto(_label)

st.sidebar.markdown("---")
if st.session_state.auth_user:
    _who = st.session_state.auth_user["full_name"]
    _role = st.session_state.auth_user["user_type"]
    st.sidebar.caption(f"Signed in as **{_who}** · {_role}")
else:
    st.sidebar.caption("Not signed in")
acc_col, logout_col = st.sidebar.columns(2)
with acc_col:
    if st.button(
        "👤 Account", use_container_width=True,
        type="primary" if st.session_state.current_page == "👤 Account" else "secondary",
    ):
        goto("👤 Account")
with logout_col:
    if st.button(
        "🚪 Logout", use_container_width=True,
        type="primary" if st.session_state.current_page == "🚪 Logout" else "secondary",
    ):
        goto("🚪 Logout")

st.sidebar.markdown("---")
st.sidebar.caption("AI-Based Early Warning System for Waterborne Disease Outbreak Prediction")

page = st.session_state.current_page

# ==============================================================================
# PAGE — HOME
# ==============================================================================
if page == "🏠 Home":
    render_hero(
        "AI-Based Waterborne Disease Early Warning System",
        "Estimate waterborne disease outbreak risk using water-quality, environmental "
        "and sanitation signals.",
    )

    hb1, hb2, hb3 = st.columns([1, 1, 2])
    with hb1:
        if st.button("🔬 Start Risk Assessment", type="primary", use_container_width=True):
            goto("🔬 Risk Assessment")
    with hb2:
        if st.button("📊 Explore Dashboard", use_container_width=True):
            goto("📊 Dashboard")

    st.write("")
    render_disclaimer()
    st.write("")

    # ---- Four overview cards ----
    overview_cards = [
        ("⏱️", "14-Day Early Warning", "Short-term outbreak risk assessment."),
        ("📆", "30-Day Early Warning", "Longer-term outbreak risk assessment."),
        ("💧", "Water Quality Analysis", "Assessment of contamination-related signals."),
        ("🗺️", "District Risk Monitoring", "Location-based risk visualization."),
    ]
    oc_cols = st.columns(4)
    for col, (icon, title, desc) in zip(oc_cols, overview_cards):
        with col:
            st.markdown(
                f"""
                <div class="ews-card overview-card">
                    <div class="overview-icon">{icon}</div>
                    <p class="overview-title">{title}</p>
                    <p class="overview-desc">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---- How it works ----
    st.markdown('<p class="section-eyebrow">HOW IT WORKS</p>', unsafe_allow_html=True)
    steps = [
        ("01", "Enter Local Conditions"),
        ("02", "Analyze Risk Factors"),
        ("03", "Estimate 14-Day & 30-Day Risk"),
        ("04", "Receive Early Warning & Recommendations"),
    ]
    step_cols = st.columns(4)
    for col, (num, title) in zip(step_cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="ews-card step-card">
                    <p class="step-num">{num}</p>
                    <p class="step-title">{title}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---- What does the system analyze ----
    st.markdown('<p class="section-eyebrow">WHAT DOES THE SYSTEM ANALYZE?</p>', unsafe_allow_html=True)
    analyze_items = [
        ("💧", "Water Quality", "pH, turbidity, fecal coliform, total coliform, BOD, TDS, etc."),
        ("🌦️", "Weather", "Rainfall, temperature, humidity and flooding."),
        ("🚻", "Sanitation", "Toilet access, sewage treatment, handwashing and open defecation."),
        ("📍", "Location", "State, district, region and urban/rural conditions."),
    ]
    an_cols = st.columns(4)
    for col, (icon, title, desc) in zip(an_cols, analyze_items):
        with col:
            st.markdown(
                f"""
                <div class="ews-card analyze-card">
                    <div class="analyze-icon">{icon}</div>
                    <p class="analyze-title">{title}</p>
                    <p class="analyze-desc">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---- Risk levels ----
    st.markdown('<p class="section-eyebrow">RISK LEVELS</p>', unsafe_allow_html=True)
    rl1, rl2, rl3 = st.columns(3)
    with rl1:
        st.markdown(
            """
            <div class="ews-card risk-level-card low">
                <span class="risk-badge low">LOW</span>
                <p class="risk-level-desc">Relatively low predicted risk.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with rl2:
        st.markdown(
            """
            <div class="ews-card risk-level-card medium">
                <span class="risk-badge medium">MODERATE</span>
                <p class="risk-level-desc">Conditions indicate increased monitoring may be appropriate.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with rl3:
        st.markdown(
            """
            <div class="ews-card risk-level-card high">
                <span class="risk-badge high">HIGH</span>
                <p class="risk-level-desc">Elevated predicted risk requiring attention and monitoring.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption("Risk levels are model estimates, not a guarantee that an outbreak will occur.")

    # ---- 14 vs 30 day ----
    st.markdown('<p class="section-eyebrow">14-DAY VS 30-DAY EARLY WARNING</p>', unsafe_allow_html=True)
    hv1, hv2 = st.columns(2)
    with hv1:
        st.markdown(
            """
            <div class="ews-card horizon-mini-card">
                <p class="horizon-mini-label">14-Day</p>
                <p class="horizon-mini-desc">Shorter-term early warning horizon.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hv2:
        st.markdown(
            """
            <div class="ews-card horizon-mini-card">
                <p class="horizon-mini-label">30-Day</p>
                <p class="horizon-mini-desc">Longer-term early warning horizon.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption("See the ℹ️ About page for the full clinical rationale behind these two horizons.")

# ==============================================================================
# PAGE — DASHBOARD
# ==============================================================================
elif page == "📊 Dashboard":
    render_hero(
        "📊 Dashboard",
        "Your quick-access hub for outbreak risk monitoring and past assessments.",
    )

    df_preview = load_clean_dataset()
    m1, m2, m3 = st.columns(3)
    m1.metric("Records in training data", f"{len(df_preview):,}")
    m2.metric("States covered", df_preview["state"].nunique())
    m3.metric("Districts covered", df_preview["district"].nunique())

    st.write("")

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        r14, r30 = result["14_day"], result["30_day"]
        assessed_on = st.session_state.get("assessed_on", "—")

        st.markdown('<p class="section-eyebrow">LAST ASSESSMENT</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="dashboard-location">📍 {result["location"]["district"]}, '
            f'{result["location"]["state"]} &nbsp;•&nbsp; assessed {assessed_on}</p>',
            unsafe_allow_html=True,
        )

        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown(
                f"""
                <div class="horizon-card">
                    <p class="horizon-label">14-Day Early Warning</p>
                    <p class="horizon-pct">{r14['risk_probability_pct']}%</p>
                    <span class="risk-badge {RISK_CSS_CLASS[r14['risk_level']]}">{r14['risk_level']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with dc2:
            st.markdown(
                f"""
                <div class="horizon-card">
                    <p class="horizon-label">30-Day Early Warning</p>
                    <p class="horizon-pct">{r30['risk_probability_pct']}%</p>
                    <span class="risk-badge {RISK_CSS_CLASS[r30['risk_level']]}">{r30['risk_level']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            if st.button("📈 View Full Analysis", use_container_width=True):
                goto("📈 Risk Analysis")
        with ac2:
            if st.button("💡 View Recommendations", use_container_width=True):
                goto("💡 Recommendations")
        with ac3:
            if st.button("📋 Download Report", use_container_width=True):
                goto("📋 Reports")
    else:
        st.markdown(
            """
            <div class="ews-card ews-empty-state">
                <p class="ews-empty-icon">🩺</p>
                <h4>No assessment run yet this session</h4>
                <p>Run your first risk assessment to see results summarised here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🔬 Start Risk Assessment", type="primary"):
            goto("🔬 Risk Assessment")

    st.write("")
    st.markdown('<p class="section-eyebrow">QUICK LINKS</p>', unsafe_allow_html=True)
    quick_links = [
        ("🔬", "Risk Assessment", "Run a new prediction for a location.", "🔬 Risk Assessment"),
        ("🗺️", "District Risk Map", "See risk across all sampled districts.", "🗺️ District Risk Map"),
        ("📋", "Reports", "Download the latest PDF report.", "📋 Reports"),
        ("ℹ️", "About", "Methodology, dataset and limitations.", "ℹ️ About"),
    ]
    ql_cols = st.columns(4)
    for col, (icon, title, desc, target) in zip(ql_cols, quick_links):
        with col:
            st.markdown(
                f"""
                <div class="ews-card quick-link-card">
                    <div class="overview-icon">{icon}</div>
                    <p class="overview-title">{title}</p>
                    <p class="overview-desc">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Open →", key=f"ql_{target}", use_container_width=True):
                goto(target)

# ==============================================================================
# PAGE — RISK ASSESSMENT (prediction input form)
# ==============================================================================
elif page == "🔬 Risk Assessment":
    render_hero(
        "🔬 Risk Assessment",
        "Enter a location's water-quality, weather and sanitation readings to get a "
        "14-day and 30-day outbreak risk estimate.",
    )
    render_disclaimer()
    st.write("")

    defaults = get_input_defaults()

    with st.form("prediction_form"):
        st.markdown('<p class="ews-eyebrow">Location</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        state = c1.selectbox("State", STATE_OPTIONS, index=STATE_OPTIONS.index("Assam"))
        district = c2.text_input("District", value="Tinsukia")
        region = c3.selectbox("Region", REGION_OPTIONS, index=REGION_OPTIONS.index("Northeast"))

        c4, c5, c6 = st.columns(3)
        latitude = c4.number_input("Latitude", value=float(defaults["latitude"]), format="%.4f")
        longitude = c5.number_input("Longitude", value=float(defaults["longitude"]), format="%.4f")
        is_urban = c6.selectbox("Urban / Rural", ["Rural", "Urban"], index=0)

        st.markdown('<p class="ews-eyebrow">Water Quality</p>', unsafe_allow_html=True)
        w1, w2, w3, w4 = st.columns(4)
        water_quality_index = w1.slider("Water Quality Index (0=worst,100=best)", 0.0, 100.0, 45.0)
        ph = w2.number_input("pH", 0.0, 14.0, 7.0, step=0.1)
        turbidity_ntu = w3.number_input("Turbidity (NTU)", 0.0, 200.0, 15.0)
        dissolved_oxygen_mg_l = w4.number_input("Dissolved Oxygen (mg/L)", 0.0, 15.0, 5.0)

        w5, w6, w7, w8 = st.columns(4)
        bod_mg_l = w5.number_input("BOD (mg/L)", 0.0, 60.0, 5.0)
        fecal_coliform_per_100ml = w6.number_input("Fecal Coliform (per 100ml)", 0.0, 6000.0, 500.0)
        total_coliform_per_100ml = w7.number_input("Total Coliform (per 100ml)", 0.0, 12000.0, 1500.0)
        tds_mg_l = w8.number_input("TDS (mg/L)", 0.0, 3000.0, 500.0)

        w9, w10, w11 = st.columns(3)
        nitrate_mg_l = w9.number_input("Nitrate (mg/L)", 0.0, 100.0, 10.0)
        fluoride_mg_l = w10.number_input("Fluoride (mg/L)", 0.0, 5.0, 0.8)
        arsenic_ug_l = w11.number_input("Arsenic (µg/L)", 0.0, 100.0, 10.0)

        st.markdown('<p class="ews-eyebrow">Weather</p>', unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        avg_temperature_c = e1.number_input("Avg Temperature (°C)", -5.0, 55.0, 28.0)
        avg_rainfall_mm = e2.number_input("Avg Rainfall (mm)", 0.0, 1200.0, 100.0)
        avg_humidity_pct = e3.number_input("Avg Humidity (%)", 0.0, 100.0, 70.0)

        e4, e5, e6 = st.columns(3)
        season = e4.selectbox("Season", SEASON_OPTIONS, index=SEASON_OPTIONS.index("Monsoon"))
        month = e5.selectbox("Month", list(range(1, 13)), index=6)
        flooding = e6.selectbox("Active Flooding?", ["No", "Yes"], index=0)

        st.markdown('<p class="ews-eyebrow">Sanitation</p>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        open_defecation_rate = s1.number_input("Open Defecation Rate (%)", 0.0, 100.0, 20.0)
        toilet_access = s2.selectbox("Toilet Access", ["Yes", "No"], index=0)
        sewage_treatment_pct = s3.number_input("Sewage Treatment (%)", 0.0, 100.0, 40.0)
        handwashing_practice = s4.selectbox("Handwashing Practice", HANDWASHING_OPTIONS, index=1)

        st.markdown('<p class="ews-eyebrow">Water Source</p>', unsafe_allow_html=True)
        v1, v2 = st.columns(2)
        water_source = v1.selectbox("Water Source", WATER_SOURCE_OPTIONS, index=1)
        water_treatment = v2.selectbox("Water Treatment", WATER_TREATMENT_OPTIONS, index=0)

        submitted = st.form_submit_button("🔍 Predict Outbreak Risk", use_container_width=True)

    if submitted:
        result = predict_outbreak_risk(
            state=state,
            district=district,
            water_quality_params=dict(
                water_quality_index=water_quality_index, ph=ph, turbidity_ntu=turbidity_ntu,
                dissolved_oxygen_mg_l=dissolved_oxygen_mg_l, bod_mg_l=bod_mg_l,
                fecal_coliform_per_100ml=fecal_coliform_per_100ml,
                total_coliform_per_100ml=total_coliform_per_100ml, tds_mg_l=tds_mg_l,
                nitrate_mg_l=nitrate_mg_l, fluoride_mg_l=fluoride_mg_l, arsenic_ug_l=arsenic_ug_l,
            ),
            environmental_params=dict(
                avg_temperature_c=avg_temperature_c, avg_rainfall_mm=avg_rainfall_mm,
                avg_humidity_pct=avg_humidity_pct, flooding=1 if flooding == "Yes" else 0,
                season=season, month=int(month),
            ),
            sanitation_params=dict(
                open_defecation_rate=open_defecation_rate,
                toilet_access=1 if toilet_access == "Yes" else 0,
                sewage_treatment_pct=sewage_treatment_pct,
                handwashing_practice=handwashing_practice,
            ),
            location_params=dict(
                latitude=latitude, longitude=longitude, region=region,
                is_urban=1 if is_urban == "Urban" else 0,
                water_source=water_source, water_treatment=water_treatment,
            ),
        )
        st.session_state["last_result"] = result
        st.session_state["assessed_on"] = date.today().isoformat()

    # ---- Show results if a prediction has been made this session ----
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        r14, r30 = result["14_day"], result["30_day"]

        st.markdown("### Prediction Output")
        oc1, oc2 = st.columns(2)

        with oc1:
            st.markdown('<div class="horizon-card">', unsafe_allow_html=True)
            st.markdown('<p class="horizon-label">14-Day Early Warning</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="horizon-pct">{r14["risk_probability_pct"]}%</p>', unsafe_allow_html=True)
            st.markdown(
                f'<span class="risk-badge {RISK_CSS_CLASS[r14["risk_level"]]}">{r14["risk_level"]}</span>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.write(r14["message"])
            st.markdown("</div>", unsafe_allow_html=True)

        with oc2:
            st.markdown('<div class="horizon-card">', unsafe_allow_html=True)
            st.markdown('<p class="horizon-label">30-Day Early Warning</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="horizon-pct">{r30["risk_probability_pct"]}%</p>', unsafe_allow_html=True)
            st.markdown(
                f'<span class="risk-badge {RISK_CSS_CLASS[r30["risk_level"]]}">{r30["risk_level"]}</span>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.write(r30["message"])
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.caption("Continue to the pages below for charts, recommendations and a downloadable report.")
        na1, na2, na3 = st.columns(3)
        with na1:
            if st.button("📈 View Detailed Risk Analysis", use_container_width=True):
                goto("📈 Risk Analysis")
        with na2:
            if st.button("💡 See Recommendations", use_container_width=True):
                goto("💡 Recommendations")
        with na3:
            if st.button("📋 Download Report", use_container_width=True):
                goto("📋 Reports")

# ==============================================================================
# PAGE — DISTRICT RISK MAP
# ==============================================================================
elif page == "🗺️ District Risk Map":
    render_hero(
        "🗺️ District Risk Map",
        "Model-estimated outbreak risk across all districts sampled in the training "
        "dataset — green (low), yellow (medium), red (high).",
    )

    horizon_choice = st.radio("Horizon", ["14-Day", "30-Day"], horizontal=True)

    @st.cache_data(show_spinner="Scoring districts...")
    def score_all_districts(horizon: str):
        df = load_clean_dataset()
        model_14d, model_30d, _ = load_models()
        from utils.feature_engineering import ALL_MODEL_FEATURES, engineer_features, classify_risk_level
        from utils.preprocessing import get_reference_artifacts

        ref_stats, region_lookup = get_reference_artifacts()
        model = model_14d if horizon == "14-Day" else model_30d

        # Aggregate to one representative row per state+district (mean for numeric,
        # mode for categorical) so we score each district once rather than every record.
        agg_rows = []
        for (state, district), grp in df.groupby(["state", "district"]):
            row = {}
            for col in grp.columns:
                if pd.api.types.is_numeric_dtype(grp[col]):
                    row[col] = grp[col].mean()
                else:
                    row[col] = grp[col].mode()[0]
            row["state"] = state
            row["district"] = district
            agg_rows.append(row)
        agg_df = pd.DataFrame(agg_rows)

        engineered_rows = [engineer_features(row.to_dict(), ref_stats, region_lookup) for _, row in agg_df.iterrows()]
        X = pd.DataFrame(engineered_rows)[ALL_MODEL_FEATURES]
        proba = model.predict_proba(X)[:, 1]

        agg_df["risk_probability_pct"] = (proba * 100).round(1)
        agg_df["risk_level"] = [classify_risk_level(p) for p in proba]
        return agg_df[["state", "district", "latitude", "longitude", "risk_probability_pct", "risk_level"]]

    scored = score_all_districts(horizon_choice)

    m1, m2, m3 = st.columns(3)
    m1.metric("Low Risk districts", int((scored["risk_level"] == "Low Risk").sum()))
    m2.metric("Medium Risk districts", int((scored["risk_level"] == "Medium Risk").sum()))
    m3.metric("High Risk districts", int((scored["risk_level"] == "High Risk").sum()))

    india_map = folium.Map(location=[22.5, 80.0], zoom_start=5, tiles="CartoDB positron")
    color_map = {"Low Risk": "green", "Medium Risk": "orange", "High Risk": "red"}

    for _, row in scored.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            color=color_map[row["risk_level"]],
            fill=True,
            fill_color=color_map[row["risk_level"]],
            fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>{row['district']}, {row['state']}</b><br>"
                f"{horizon_choice} risk: {row['risk_probability_pct']}%<br>"
                f"Level: {row['risk_level']}",
                max_width=220,
            ),
        ).add_to(india_map)

    st_folium(india_map, width=None, height=560, returned_objects=[])

    with st.expander("View underlying district risk table"):
        st.dataframe(
            scored.sort_values("risk_probability_pct", ascending=False).reset_index(drop=True),
            use_container_width=True,
        )

# ==============================================================================
# PAGE — RISK ANALYSIS (gauges, comparison chart, feature importance)
# ==============================================================================
elif page == "📈 Risk Analysis":
    render_hero(
        "📈 Risk Analysis",
        "Visual breakdown of the most recent risk assessment — gauges, horizon "
        "comparison and the factors driving the prediction.",
    )

    if "last_result" not in st.session_state:
        no_assessment_prompt("the detailed risk analysis")
    else:
        result = st.session_state["last_result"]
        r14, r30 = result["14_day"], result["30_day"]
        st.caption(f"📍 {result['location']['district']}, {result['location']['state']}")

        st.markdown("### Risk Visualization")
        g1, g2, g3 = st.columns([1, 1, 1])

        def make_gauge(value_pct, title, risk_level):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value_pct,
                title={"text": title, "font": {"size": 15}},
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": RISK_COLORS[risk_level]},
                    "steps": [
                        {"range": [0, 35], "color": "#E9F6EA"},
                        {"range": [35, 65], "color": "#FFF6DF"},
                        {"range": [65, 100], "color": "#FDEAEA"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 2}, "value": value_pct},
                },
            ))
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
            return fig

        with g1:
            st.plotly_chart(make_gauge(r14["risk_probability_pct"], "14-Day Risk", r14["risk_level"]),
                             use_container_width=True)
        with g2:
            st.plotly_chart(make_gauge(r30["risk_probability_pct"], "30-Day Risk", r30["risk_level"]),
                             use_container_width=True)
        with g3:
            comp_df = pd.DataFrame({
                "Horizon": ["14-Day", "30-Day"],
                "Risk %": [r14["risk_probability_pct"], r30["risk_probability_pct"]],
                "Level": [r14["risk_level"], r30["risk_level"]],
            })
            fig_bar = px.bar(
                comp_df, x="Horizon", y="Risk %", color="Level",
                color_discrete_map=RISK_COLORS, text="Risk %",
                title="14-Day vs 30-Day Risk",
            )
            fig_bar.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # ---- Feature Importance ----
        st.markdown("### Feature Importance — Top Factors Driving This Prediction")
        fi1, fi2 = st.columns(2)
        with fi1:
            imp14 = get_feature_importance("14d", top_n=10)
            fig_fi14 = px.bar(
                x=imp14.values[::-1], y=imp14.index[::-1], orientation="h",
                labels={"x": "Importance", "y": ""}, title="Top factors — 14-Day model",
                color_discrete_sequence=["#0B4F6C"],
            )
            fig_fi14.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_fi14, use_container_width=True)
        with fi2:
            imp30 = get_feature_importance("30d", top_n=10)
            fig_fi30 = px.bar(
                x=imp30.values[::-1], y=imp30.index[::-1], orientation="h",
                labels={"x": "Importance", "y": ""}, title="Top factors — 30-Day model",
                color_discrete_sequence=["#01A7C2"],
            )
            fig_fi30.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_fi30, use_container_width=True)

        st.write("")
        ra1, ra2 = st.columns(2)
        with ra1:
            if st.button("💡 View Recommendations", use_container_width=True):
                goto("💡 Recommendations")
        with ra2:
            if st.button("📋 Download Report", use_container_width=True):
                goto("📋 Reports")

# ==============================================================================
# PAGE — REPORTS (PDF download)
# ==============================================================================
elif page == "📋 Reports":
    render_hero(
        "📋 Reports",
        "Generate a downloadable PDF summary of the most recent risk assessment.",
    )

    if "last_result" not in st.session_state:
        no_assessment_prompt("your downloadable report")
    else:
        result = st.session_state["last_result"]
        r14, r30 = result["14_day"], result["30_day"]
        worst_level = get_worst_level(r14, r30)
        recs = get_recommendations(worst_level)

        st.markdown('<div class="ews-card">', unsafe_allow_html=True)
        st.markdown(f"**📍 {result['location']['district']}, {result['location']['state']}**")
        rp1, rp2 = st.columns(2)
        rp1.metric("14-Day Risk", f"{r14['risk_probability_pct']}%", r14["risk_level"])
        rp2.metric("30-Day Risk", f"{r30['risk_probability_pct']}%", r30["risk_level"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Download Report")

        def build_pdf_report():
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()

            def safe_multicell(pdf, height, text):
                """Robust wrapper around FPDF.multi_cell().

                Root cause of the original 'Not enough horizontal space to render a
                single character' error: multi_cell(0, ...) computes its available
                width from the CURRENT cursor X position (self.w - r_margin - x).
                After a previous multi_cell() call the cursor is left wherever that
                call ended (not necessarily back at the left margin), so calling
                multi_cell(0, ...) again — e.g. once per recommendation, in a loop —
                could hand FPDF a shrinking, and eventually zero/near-zero, width
                until it could no longer fit even a single character.

                Fix: always reset X to the left margin and pass an explicit,
                pre-validated positive width, computed fresh from the page geometry
                rather than the cursor position.
                """
                # 1) Safely coerce to a string (handles None, numbers, etc.).
                text = "" if text is None else str(text)
                if not text.strip():
                    return  # nothing to render for an empty recommendation

                # 2) Always start each block flush against the left margin.
                pdf.set_x(pdf.l_margin)

                # 3) Explicit, validated available width (never rely on width=0).
                available_width = pdf.w - pdf.l_margin - pdf.r_margin
                if available_width <= 0:
                    # Page geometry is degenerate (shouldn't happen with standard
                    # A4/Letter + default margins) — fall back to a safe minimum
                    # rather than letting FPDF raise.
                    available_width = 10

                # 4) Render, allowing a hard character-level break for any
                #    unbroken/very long token (e.g. a URL) that wouldn't otherwise
                #    fit on one line at all.
                pdf.multi_cell(available_width, height, text, new_x="LMARGIN", new_y="NEXT", wrapmode="CHAR")

            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Waterborne Disease Outbreak Risk Report", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 8, f"Generated: {date.today().isoformat()}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Location", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 7, f"State: {result['location']['state']}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 7, f"District: {result['location']['district']}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

            for label, r in [("14-Day Prediction", r14), ("30-Day Prediction", r30)]:
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, label, new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 11)
                pdf.cell(0, 7, f"Risk Probability: {r['risk_probability_pct']}%", new_x="LMARGIN", new_y="NEXT")
                pdf.cell(0, 7, f"Risk Level: {r['risk_level']}", new_x="LMARGIN", new_y="NEXT")
                safe_multicell(pdf, 6, r.get("message"))
                pdf.ln(2)

            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Recommendations", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            if not recs:
                safe_multicell(pdf, 6, "No specific recommendations available.")
            else:
                for rec in recs:
                    safe_multicell(pdf, 6, f"- {rec}")

            return bytes(pdf.output())

        pdf_bytes = build_pdf_report()
        st.download_button(
            "⬇️ Download Prediction Report (PDF)",
            data=pdf_bytes,
            file_name=f"outbreak_risk_report_{result['location']['district']}_{date.today().isoformat()}.pdf",
            mime="application/pdf",
            type="primary",
        )

# ==============================================================================
# PAGE — RECOMMENDATIONS
# ==============================================================================
elif page == "💡 Recommendations":
    render_hero(
        "💡 Recommendations",
        "Suggested actions based on the most recent risk assessment.",
    )

    if "last_result" not in st.session_state:
        no_assessment_prompt("your risk recommendations")
    else:
        result = st.session_state["last_result"]
        r14, r30 = result["14_day"], result["30_day"]
        worst_level = get_worst_level(r14, r30)
        recs = get_recommendations(worst_level)

        st.markdown(f'<span class="risk-badge {RISK_CSS_CLASS[worst_level]}">{worst_level}</span>',
                    unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="ews-card">', unsafe_allow_html=True)
        st.markdown(f"**{get_recommendation_summary(worst_level)}**")
        st.markdown(
            "<ul class='rec-list'>" + "".join(f"<li>{r}</li>" for r in recs) + "</ul>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# PAGE — ABOUT (technical / methodology content moved here from Home)
# ==============================================================================
elif page == "ℹ️ About":
    render_hero(
        "ℹ️ About This System",
        "Problem statement, methodology, dataset and known limitations.",
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="ews-card">', unsafe_allow_html=True)
        st.markdown("#### Problem Statement")
        st.write(
            "Waterborne diseases (Cholera, Typhoid, Dysentery, Hepatitis A/E, Giardiasis, "
            "Leptospirosis) remain a major public health burden in India, especially where "
            "water quality is poor, sanitation infrastructure is weak, and monsoon rainfall "
            "or flooding is heavy. Outbreaks are usually detected only after a cluster of "
            "patients has already fallen sick — by which point the disease has already spread."
        )
        st.markdown("#### Objective")
        st.write(
            "Estimate the probability that a waterborne disease outbreak is emerging in a "
            "given district, on **two forecasting horizons**, using only signals that are "
            "observable *before* a clinical outbreak is confirmed — water quality, weather, "
            "sanitation, and location context."
        )
        st.markdown("#### How the AI Predicts Outbreaks")
        st.write(
            "Each new location's raw readings are converted into the same engineered risk "
            "scores used during training (water contamination score, weather risk, seasonal "
            "risk, sanitation risk, location risk, and their interactions), then passed into "
            "tuned classification models (Gradient Boosting for the 14-day horizon, Random "
            "Forest for the 30-day horizon) selected by cross-validated ROC-AUC during "
            "development."
        )
        st.markdown("#### Known Limitations")
        st.write(
            "- The training dataset is cross-sectional, not a per-location day-level time "
            "series — the 14-day/30-day horizons are operationalised via disease "
            "incubation-period groupings, not sequential forecasting.\n"
            "- Risk-level thresholds (35% / 65%) are reasonable defaults; a production "
            "deployment should calibrate them with public-health domain experts.\n"
            "- The District Risk Map scores one aggregated (mean/mode) profile per district "
            "from the training sample — it is illustrative, not a live/real-time feed."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="ews-card">', unsafe_allow_html=True)
        st.markdown("#### 14-Day vs 30-Day Early Warning")
        st.write(
            "The two horizons are grounded in each disease's typical **clinical incubation "
            "period**, not an arbitrary split:"
        )
        st.markdown(
            """
            **14-Day Early Warning** — fast-onset diseases
            *(Cholera, Dysentery, Leptospirosis; incubation ≲ 2 weeks)*
            Flags risk that can turn into confirmed cases within about two weeks —
            the tightest response window.

            **30-Day Early Warning** — slow-onset diseases
            *(Typhoid, Giardiasis, Hepatitis A/E; incubation 2–8 weeks)*
            Gives a longer planning window for infrastructure and sanitation response.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="ews-card">', unsafe_allow_html=True)
        st.markdown("#### Dataset Snapshot")
        df_preview = load_clean_dataset()
        m1, m2, m3 = st.columns(3)
        m1.metric("Records", f"{len(df_preview):,}")
        m2.metric("States covered", df_preview["state"].nunique())
        m3.metric("Districts covered", df_preview["district"].nunique())
        st.markdown("</div>", unsafe_allow_html=True)

    render_disclaimer()

# ==============================================================================
# PAGE — ACCOUNT (demo/local login + signup — not production-grade auth)
# ==============================================================================
elif page == "👤 Account":
    render_hero("👤 Account", "Sign in, create an account, or continue as a guest.")

    if st.session_state.auth_user:
        # ---- Already signed in: show profile card ----
        u = st.session_state.auth_user
        st.markdown(
            f"""
            <div class="ews-card account-card">
                <div class="account-avatar">👤</div>
                <p class="account-name">{u['full_name']}</p>
                <p class="account-role">{u['user_type']} — {'Guest Session' if u.get('guest') else 'Registered Account'}</p>
                <p class="account-meta">{u.get('email', '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "This is a demo/local authentication session for the project presentation. "
            "No data leaves your browser session — everything resets when the app restarts."
        )
        ac1, ac2 = st.columns(2)
        with ac1:
            if st.button("🏠 Back to Home", use_container_width=True):
                goto("🏠 Home")
        with ac2:
            if st.button("🚪 Sign Out", use_container_width=True):
                goto("🚪 Logout")

    else:
        # ---- Not signed in: Login / Sign Up / Forgot Password tabs ----
        st.caption(
            "⚠️ Demo/local authentication only — for this presentation build. Not "
            "production-grade security. Passwords are hashed before being kept in "
            "this session (never stored in plain text), and everything is cleared "
            "when the app restarts. Swap in a real auth provider before deploying."
        )

        tab_login, tab_signup, tab_forgot = st.tabs(
            ["🔐 Login", "📝 Create Account", "❓ Forgot Password"]
        )

        # ---- LOGIN ----
        with tab_login:
            with st.form("login_form"):
                login_id = st.text_input("Username / Email")
                login_pw = st.text_input("Password", type="password")
                login_submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

            if login_submitted:
                user = find_user_by_login(login_id)
                if user is None or user["password_hash"] != hash_password(login_pw):
                    st.error("Invalid username/email or password.")
                else:
                    st.session_state.auth_user = {**user, "guest": False}
                    st.success(f"Welcome back, {user['full_name']}!")
                    goto("📊 Dashboard")

            st.markdown("---")
            st.caption("Just exploring the project?")
            if st.button("👀 Continue as Guest", use_container_width=True):
                st.session_state.auth_user = {
                    "full_name": "Guest User",
                    "email": "",
                    "user_type": "Public User",
                    "guest": True,
                }
                goto("📊 Dashboard")
            st.caption("Demo login: **demo@example.com** / **demo1234**")

        # ---- SIGN UP ----
        with tab_signup:
            with st.form("signup_form"):
                su_name = st.text_input("Full Name")
                su_email = st.text_input("Email")
                su_pw = st.text_input("Password", type="password")
                su_pw2 = st.text_input("Confirm Password", type="password")
                su_type = st.radio("User Type", ["Public User", "Health/Admin User"], horizontal=True)
                su_submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)

            if su_submitted:
                su_name_clean = (su_name or "").strip()
                su_email_clean = (su_email or "").strip().lower()
                if not su_name_clean or not su_email_clean or not su_pw:
                    st.error("Please fill in your name, email and password.")
                elif su_pw != su_pw2:
                    st.error("Passwords do not match.")
                elif len(su_pw) < 6:
                    st.error("Password should be at least 6 characters.")
                elif su_email_clean in st.session_state.users_db:
                    st.error("An account with this email already exists. Try logging in instead.")
                else:
                    st.session_state.users_db[su_email_clean] = {
                        "full_name": su_name_clean,
                        "email": su_email_clean,
                        "password_hash": hash_password(su_pw),
                        "user_type": su_type,
                    }
                    st.session_state.auth_user = {
                        **st.session_state.users_db[su_email_clean],
                        "guest": False,
                    }
                    st.success(f"Account created — welcome, {su_name_clean}!")
                    goto("📊 Dashboard")

        # ---- FORGOT PASSWORD ----
        with tab_forgot:
            st.write("Enter your account email and we'll simulate sending a reset link.")
            with st.form("forgot_form"):
                fp_email = st.text_input("Email")
                fp_submitted = st.form_submit_button("Send Reset Link", use_container_width=True)
            if fp_submitted:
                if (fp_email or "").strip().lower() in st.session_state.users_db:
                    st.success(
                        "If this were connected to a real email service, a password-reset "
                        "link would be sent now. (Demo build — no email is actually sent.)"
                    )
                else:
                    # Avoid confirming/denying account existence in a real system;
                    # kept generic here too, even though this is just a demo.
                    st.success(
                        "If an account exists for that email, a password-reset link "
                        "would be sent. (Demo build — no email is actually sent.)"
                    )

# ==============================================================================
# PAGE — LOGOUT (UI placeholder — clears the current session's assessment)
# ==============================================================================
elif page == "🚪 Logout":
    st.session_state.pop("last_result", None)
    st.session_state.pop("assessed_on", None)
    st.session_state.auth_user = None

    st.markdown(
        """
        <div class="ews-card logout-card">
            <div class="account-avatar">🚪</div>
            <p class="account-name">You have been signed out</p>
            <p class="account-meta">Thank you for using the Waterborne Disease Early Warning System.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🔐 Return to Home", type="primary"):
        goto("🏠 Home")
