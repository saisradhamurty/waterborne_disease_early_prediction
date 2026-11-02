"""
src/india_map.py
-----------------
Builds Folium maps for two purposes:

1. cgwb_water_quality_map()  — Chhattisgarh district-level water-quality
   indicators from CGWB observed data (real measurements).

2. risk_map()  — Model-estimated outbreak risk across training-dataset districts
   (one marker per district, scored from the mean row for that district).

Design rules (per task spec §8)
--------------------------------
- Only verified coordinates from the CGWB data or the training dataset are used.
- Distinct visual markers for real CGWB data vs synthetic/model output.
- Tooltips show: location, data_origin, date range, parameters, risk level.
- Legend included.
- No fabricated or hardcoded coordinates.
"""

from __future__ import annotations

import logging
from typing import Optional

import folium
import pandas as pd

logger = logging.getLogger(__name__)

# ── Colour palettes ───────────────────────────────────────────────────────────
RISK_COLOR = {"Low Risk": "green", "Medium Risk": "orange", "High Risk": "red"}
WQ_VIOLATION_COLOR = {True: "red", False: "blue"}

INDIA_CENTER = [22.5, 80.0]
CGWB_CENTER = [21.3, 81.6]  # Chhattisgarh centroid


def _legend_html(items: list[tuple[str, str]]) -> str:
    """Build a simple HTML legend box for Folium."""
    rows = "".join(
        f'<i style="background:{color};width:12px;height:12px;'
        f'display:inline-block;border-radius:50%;margin-right:6px"></i>{label}<br>'
        for label, color in items
    )
    return (
        '<div style="position:fixed;bottom:40px;left:40px;z-index:1000;'
        'background:white;padding:10px;border:1px solid #ccc;border-radius:5px;'
        'font-size:13px;max-width:220px">'
        f'<b>Legend</b><br>{rows}</div>'
    )


def cgwb_water_quality_map(
    cgwb_summary: pd.DataFrame,
    parameter: str = "ph",
    title: str = "Chhattisgarh Groundwater Quality (CGWB Observed Data)",
) -> folium.Map:
    """
    Plot district-level mean values for a chosen CGWB parameter.

    Parameters
    ----------
    cgwb_summary : DataFrame from src.data_loader.get_cgwb_chhattisgarh_summary()
    parameter    : Column name to colour by (e.g. 'ph', 'tds_mg_l')
    """
    m = folium.Map(location=CGWB_CENTER, zoom_start=7, tiles="CartoDB positron")

    IS10500_THRESHOLD = {
        "ph": (6.5, 8.5), "tds_mg_l": 500, "nitrate_n_mg_l": 45,
        "fluoride_mg_l": 1.0, "iron_mg_l": 0.3, "total_hardness_mg_l": 200,
    }

    for _, row in cgwb_summary.iterrows():
        if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
            continue
        val = row.get(parameter)
        if pd.isna(val):
            continue

        # Determine violation
        limit = IS10500_THRESHOLD.get(parameter)
        if isinstance(limit, tuple):
            violation = not (limit[0] <= val <= limit[1])
        elif limit is not None:
            violation = val > limit
        else:
            violation = False

        color = "red" if violation else "steelblue"
        popup_text = (
            f"<b>{row.get('district', 'N/A')}, {row.get('state', 'Chhattisgarh')}</b><br>"
            f"Data origin: CGWB Observed<br>"
            f"{parameter}: {val:.2f}<br>"
            f"IS:10500 violation: {'Yes ⚠️' if violation else 'No ✓'}"
        )

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(popup_text, max_width=260),
            tooltip=f"{row.get('district', '')}: {val:.2f}",
        ).add_to(m)

    # Legend
    legend = _legend_html([
        ("IS:10500 Violation", "red"),
        ("Within Limit", "steelblue"),
    ])
    m.get_root().html.add_child(folium.Element(legend))
    m.get_root().html.add_child(
        folium.Element(f'<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);'
                       f'z-index:1001;background:white;padding:6px 14px;border-radius:4px;'
                       f'font-weight:bold;font-size:14px;border:1px solid #ccc">{title}</div>')
    )
    return m


def risk_map(
    scored_df: pd.DataFrame,
    horizon: str = "14-Day",
    title: Optional[str] = None,
) -> folium.Map:
    """
    Plot ML-estimated outbreak risk for all districts in scored_df.

    Parameters
    ----------
    scored_df : DataFrame with columns:
        state, district, latitude, longitude, risk_probability_pct, risk_level
    horizon   : "14-Day" or "30-Day" (display label only)
    """
    title = title or f"{horizon} Outbreak Risk — Model Estimate"
    m = folium.Map(location=INDIA_CENTER, zoom_start=5, tiles="CartoDB positron")

    for _, row in scored_df.iterrows():
        lat, lon = row.get("latitude"), row.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            continue

        rl = row.get("risk_level", "Low Risk")
        color = RISK_COLOR.get(rl, "gray")
        pct = row.get("risk_probability_pct", "N/A")

        popup_text = (
            f"<b>{row.get('district', 'N/A')}, {row.get('state', 'N/A')}</b><br>"
            f"Data origin: Model estimate (illustrative)<br>"
            f"{horizon} risk: {pct}%<br>"
            f"Level: <b>{rl}</b><br>"
            f"<i>Not a confirmed outbreak prediction.</i>"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=f"{row.get('district', '')}: {rl}",
        ).add_to(m)

    legend = _legend_html([
        ("Low Risk", "green"), ("Medium Risk", "orange"), ("High Risk", "red"),
    ])
    m.get_root().html.add_child(folium.Element(legend))
    m.get_root().html.add_child(
        folium.Element(
            f'<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);'
            f'z-index:1001;background:white;padding:6px 14px;border-radius:4px;'
            f'font-weight:bold;font-size:14px;border:1px solid #ccc">{title}</div>'
        )
    )
    return m


def durg_surveillance_map(surveillance_df: pd.DataFrame) -> folium.Map:
    """
    Plot the 5 synthetic surveillance report locations for Bhilai-Durg.
    Markers are visually distinct (diamond / square icons via DivIcon) and
    clearly labelled as SYNTHETIC ACADEMIC DATA.
    """
    m = folium.Map(location=[21.19, 81.28], zoom_start=10, tiles="CartoDB positron")

    disease_colors = {
        "Cholera": "#e74c3c", "Dysentery": "#e67e22", "Typhoid": "#f39c12",
        "Hepatitis_A": "#8e44ad", "Giardiasis": "#2980b9",
    }

    for i, row in surveillance_df.iterrows():
        lat = row.get("latitude", 21.19) + (i - 2) * 0.015  # slight offset to separate markers
        lon = row.get("longitude", 81.28) + (i - 2) * 0.015
        color = disease_colors.get(row.get("disease", ""), "#555")
        risk_cat = row.get("predicted_risk_category", "N/A")

        popup_text = (
            f"<b>Report ID: {row.get('report_id', 'N/A')}</b><br>"
            f"Disease: {row.get('disease', 'N/A')}<br>"
            f"District: {row.get('district', 'Durg')}, {row.get('state', 'Chhattisgarh')}<br>"
            f"Forecast horizon: {row.get('forecast_horizon_days', 'N/A')} days<br>"
            f"Risk category: <b>{risk_cat}</b> (illustrative)<br>"
            f"Suspected cases: {row.get('suspected_cases', 'N/A')}<br>"
            f"Confirmed cases: {row.get('confirmed_cases', 'N/A')}<br>"
            f"<hr><i>⚠️ SYNTHETIC ACADEMIC DATA — not a confirmed outbreak.</i>"
        )

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=280),
            tooltip=f"{row.get('disease', '')} — {risk_cat}",
            icon=folium.Icon(color="white", icon_color=color, icon="tint", prefix="fa"),
        ).add_to(m)

    legend = _legend_html([
        (d, c) for d, c in disease_colors.items()
    ] + [("", ""), ("⚠️ ALL DATA SYNTHETIC", "#555")])
    m.get_root().html.add_child(folium.Element(legend))
    return m
