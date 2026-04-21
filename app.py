"""
app.py  –  NYC Taxi Fare Predictor
Streamlit portfolio app.

Run locally:   streamlit run app.py
Deploy:        push to GitHub → connect to Streamlit Community Cloud
"""

import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pandas.tseries.holiday import USFederalHolidayCalendar

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Taxi Fare Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8e8;
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2.5rem 3rem 3rem 3rem; max-width: 1200px; }

  /* ── Hero header ── */
  .hero { margin-bottom: 2.5rem; }
  .hero h1 {
    font-size: 3rem; font-weight: 800; letter-spacing: -0.03em;
    color: #f5f5f5; margin: 0; line-height: 1.1;
  }
  .hero .accent { color: #f5c518; }
  .hero p {
    font-family: 'DM Mono', monospace; font-size: 0.8rem;
    color: #666; margin-top: 0.5rem; letter-spacing: 0.05em;
  }

  /* ── Section labels ── */
  .section-label {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: #f5c518; margin-bottom: 0.75rem;
  }

  /* ── Cards ── */
  .card {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1rem;
  }
  .card-title {
    font-size: 0.7rem; font-family: 'DM Mono', monospace;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: #555; margin-bottom: 1rem;
  }

  /* ── Fare display ── */
  .fare-amount {
    font-size: 3.5rem; font-weight: 800; letter-spacing: -0.04em;
    color: #f5c518; line-height: 1;
  }
  .fare-range {
    font-family: 'DM Mono', monospace; font-size: 0.78rem;
    color: #555; margin-top: 0.3rem;
  }

  /* ── Badge ── */
  .badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 99px;
    font-size: 0.8rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .badge-low    { background: #0d2b1a; color: #4ade80; border: 1px solid #166534; }
  .badge-medium { background: #2b2100; color: #facc15; border: 1px solid #854d0e; }
  .badge-high   { background: #2b0a0a; color: #f87171; border: 1px solid #991b1b; }

  /* ── Metric row ── */
  .metric-row { display: flex; gap: 1.5rem; margin-top: 1rem; }
  .metric { flex: 1; }
  .metric-value {
    font-size: 1.4rem; font-weight: 700; color: #e8e8e8;
  }
  .metric-label {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    color: #555; letter-spacing: 0.1em; text-transform: uppercase;
    margin-top: 0.1rem;
  }

  /* ── Divider ── */
  .divider { border: none; border-top: 1px solid #1e1e2e; margin: 1.25rem 0; }

  /* ── Input styling ── */
  .stSelectbox > div > div, .stNumberInput > div > div > input,
  .stSlider > div { background: #0f0f18 !important; }

  label { font-family: 'DM Mono', monospace !important;
          font-size: 0.72rem !important; color: #888 !important;
          letter-spacing: 0.05em !important; text-transform: uppercase !important; }

  /* ── Predict button ── */
  .stButton > button {
    width: 100%;
    background: #f5c518 !important; color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    border: none !important; border-radius: 8px !important;
    padding: 0.75rem !important; margin-top: 0.5rem !important;
    transition: opacity 0.15s !important;
  }
  .stButton > button:hover { opacity: 0.85 !important; }

  /* ── Disclaimer ── */
  .disclaimer {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    color: #3a3a4a; margin-top: 2rem; line-height: 1.6;
  }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ───────────────────────────────────────────────────────────
ARTIFACTS = "model_artifacts"

@st.cache_resource(show_spinner=False)
def load_artifacts():
    pre   = joblib.load(f"{ARTIFACTS}/preprocessor.joblib")
    reg   = joblib.load(f"{ARTIFACTS}/xgb_regressor.joblib")
    clf   = joblib.load(f"{ARTIFACTS}/xgb_classifier.joblib")
    dem   = pd.read_csv(f"{ARTIFACTS}/demand_stats.csv")
    dem["PULocationID"] = dem["PULocationID"].astype(str)

    with open(f"{ARTIFACTS}/fare_thresholds.json")  as f: thresh = json.load(f)
    with open(f"{ARTIFACTS}/feature_names.json")    as f: feat_names = json.load(f)
    with open(f"{ARTIFACTS}/demand_fallbacks.json") as f: fallbacks = json.load(f)

    zones = pd.read_csv(f"{ARTIFACTS}/zone_lookup.csv")
    return pre, reg, clf, dem, thresh, feat_names, fallbacks, zones

pre, reg_model, clf_model, demand_stats, thresholds, feature_names, fallbacks, zones = load_artifacts()


# ── Helpers ──────────────────────────────────────────────────────────────────
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR", "Unknown"]
RATE_CODES = {
    "Standard rate": "1.0",
    "JFK flat rate": "2.0",
    "Newark":        "3.0",
    "Nassau/Westchester": "4.0",
    "Negotiated":    "5.0",
    "Group ride":    "6.0",
}
PAYMENT_TYPES = {
    "Credit card": "1.0",
    "Cash":        "2.0",
    "No charge":   "3.0",
    "Dispute":     "4.0",
}
LABEL_MAP = {0: "low", 1: "medium", 2: "high"}

cal = USFederalHolidayCalendar()
HOLIDAYS = set(cal.holidays(start="2025-01-01", end="2025-12-31").date)


def build_row(pu_zone_id, do_zone_id, pu_borough, do_borough,
              rate_code, payment_type, passenger_count,
              trip_distance, trip_duration_min,
              hour, day, day_of_week,
              temp_avg, humidity, precip, snow, cloudcover,
              rain_flag, snow_flag) -> pd.DataFrame:

    is_weekend    = int(day_of_week >= 5)
    is_rush_hour  = int((not is_weekend) and hour in [7, 8, 17, 18])
    is_night      = int(hour >= 22 or hour <= 5)
    avg_speed_mph = min((trip_distance / (trip_duration_min / 60)) if trip_duration_min > 0 else 0, 80)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin  = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos  = np.cos(2 * np.pi * day_of_week / 7)

    import datetime
    is_holiday = int(datetime.date(2025, 1, day) in HOLIDAYS)

    airport_ids = {"1", "132", "138"}
    is_airport  = int(str(pu_zone_id) in airport_ids or str(do_zone_id) in airport_ids)

    row = {
        "PULocationID":   str(pu_zone_id),
        "DOLocationID":   str(do_zone_id),
        "PU_Borough":     pu_borough,
        "DO_Borough":     do_borough,
        "RatecodeID":     str(rate_code),
        "payment_type":   str(payment_type),
        "passenger_count":float(passenger_count),
        "trip_distance":  float(trip_distance),
        "hour":           int(hour),
        "day_of_week":    int(day_of_week),
        "day":            int(day),
        "is_weekend":     is_weekend,
        "hour_sin":       hour_sin,
        "hour_cos":       hour_cos,
        "dow_sin":        dow_sin,
        "dow_cos":        dow_cos,
        "is_rush_hour":   is_rush_hour,
        "is_night":       is_night,
        "is_holiday":     is_holiday,
        "trip_duration_min": float(trip_duration_min),
        "avg_speed_mph":  avg_speed_mph,
        "is_airport_trip": is_airport,
        "humidity":       float(humidity),
        "precip":         float(precip),
        "snow":           float(snow),
        "cloudcover":     float(cloudcover),
        "temp_avg":       float(temp_avg),
        "rain_flag":      int(rain_flag),
        "snow_flag":      int(snow_flag),
    }
    return pd.DataFrame([row])


def attach_demand(df_row):
    pu  = str(df_row["PULocationID"].iloc[0])
    hr  = int(df_row["hour"].iloc[0])
    match = demand_stats[
        (demand_stats["PULocationID"] == pu) &
        (demand_stats["hour"] == hr)
    ]
    if len(match):
        df_row["zone_hour_avg_fare"]   = match["zone_hour_avg_fare"].iloc[0]
        df_row["zone_hour_ride_count"] = match["zone_hour_ride_count"].iloc[0]
        df_row["zone_hour_fare_std"]   = match["zone_hour_fare_std"].iloc[0]
    else:
        df_row["zone_hour_avg_fare"]   = fallbacks["zone_hour_avg_fare"]
        df_row["zone_hour_ride_count"] = fallbacks["zone_hour_ride_count"]
        df_row["zone_hour_fare_std"]   = fallbacks["zone_hour_fare_std"]
    return df_row


def predict(row_df):
    row_df = attach_demand(row_df)
    # Align column order to what the preprocessor expects
    row_df = row_df[feature_names]
    X_enc  = pre.transform(row_df)

    fare       = float(reg_model.predict(X_enc)[0])
    proba      = clf_model.predict_proba(X_enc)[0]
    label_idx  = int(np.argmax(proba))
    label      = LABEL_MAP[label_idx]

    # Feature importance (regressor)
    importances = reg_model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).head(10)

    return fare, proba, label, fi_df


def fare_range(fare):
    margin = fare * 0.12
    return fare - margin, fare + margin


# ── UI ────────────────────────────────────────────────────────────────────────
# Hero
st.markdown("""
<div class="hero">
  <h1>NYC Taxi<br><span class="accent">Fare Predictor</span></h1>
</div>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1.2], gap="large")

# ── LEFT: Input form ─────────────────────────────────────────────────────────
with left_col:
    st.markdown('<div class="section-label">Trip Details</div>', unsafe_allow_html=True)

    with st.container():
        zone_ids   = sorted(zones["LocationID"].astype(str).tolist())
        pu_zone    = st.selectbox("Pickup Zone ID", zone_ids, index=zone_ids.index("161") if "161" in zone_ids else 0)
        do_zone    = st.selectbox("Dropoff Zone ID", zone_ids, index=zone_ids.index("236") if "236" in zone_ids else 1)

        pu_borough = zones[zones["LocationID"].astype(str) == pu_zone]["Borough"].iloc[0] if len(zones[zones["LocationID"].astype(str) == pu_zone]) else "Manhattan"
        do_borough = zones[zones["LocationID"].astype(str) == do_zone]["Borough"].iloc[0] if len(zones[zones["LocationID"].astype(str) == do_zone]) else "Manhattan"

        c1, c2 = st.columns(2)
        with c1:
            trip_distance    = st.number_input("Distance (miles)", min_value=0.1, max_value=50.0, value=2.5, step=0.1)
        with c2:
            trip_duration    = st.number_input("Duration (minutes)", min_value=1, max_value=180, value=12)

        c3, c4 = st.columns(2)
        with c3:
            hour         = st.slider("Pickup Hour", 0, 23, 14)
        with c4:
            day_of_week  = st.selectbox("Day of Week", list(range(7)),
                                         format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        day = st.slider("Day of Month (Jan 2025)", 1, 31, 15)

    st.markdown('<div class="section-label" style="margin-top:1.5rem">Conditions</div>', unsafe_allow_html=True)

    with st.container():
        c5, c6 = st.columns(2)
        with c5:
            temp_avg  = st.number_input("Avg Temp (°F)", min_value=-10.0, max_value=60.0, value=32.0)
            rain_flag = st.toggle("Raining", value=False)
        with c6:
            humidity  = st.slider("Humidity (%)", 0, 100, 65)
            snow_flag = st.toggle("Snowing", value=False)

        c7, c8 = st.columns(2)
        with c7:
            precip     = st.number_input("Precipitation (in)", 0.0, 5.0, 0.0, step=0.01)
        with c8:
            snow       = st.number_input("Snow depth (in)", 0.0, 20.0, 0.0, step=0.1)
        cloudcover = st.slider("Cloud Cover (%)", 0, 100, 40)

    st.markdown('<div class="section-label" style="margin-top:1.5rem">Ride Info</div>', unsafe_allow_html=True)
    c9, c10 = st.columns(2)
    with c9:
        rate_label    = st.selectbox("Rate Code", list(RATE_CODES.keys()))
        rate_code     = RATE_CODES[rate_label]
    with c10:
        payment_label = st.selectbox("Payment Type", list(PAYMENT_TYPES.keys()))
        payment_type  = PAYMENT_TYPES[payment_label]

    passenger_count = st.slider("Passengers", 1, 6, 1)
    predict_btn     = st.button("Predict Fare →")


# ── RIGHT: Results ───────────────────────────────────────────────────────────
with right_col:
    st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)

    if predict_btn:
        with st.spinner(""):
            row_df = build_row(
                pu_zone, do_zone, pu_borough, do_borough,
                rate_code, payment_type, passenger_count,
                trip_distance, trip_duration,
                hour, day, day_of_week,
                temp_avg, humidity, precip, snow, cloudcover,
                rain_flag, snow_flag,
            )
            fare, proba, label, fi_df = predict(row_df)
            lo, hi = fare_range(fare)

        # ── Fare card ──
        badge_class = f"badge-{label}"
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Estimated Fare</div>
          <div class="fare-amount">${fare:.2f}</div>
          <div class="fare-range">Expected range · ${lo:.2f} – ${hi:.2f}</div>
          <hr class="divider">
          <div class="card-title" style="margin-bottom:0.5rem">Fare Range</div>
          <span class="badge {badge_class}">{label.upper()}</span>
          <div class="metric-row">
            <div class="metric">
              <div class="metric-value">{proba[0]*100:.0f}%</div>
              <div class="metric-label">Low</div>
            </div>
            <div class="metric">
              <div class="metric-value">{proba[1]*100:.0f}%</div>
              <div class="metric-label">Medium</div>
            </div>
            <div class="metric">
              <div class="metric-value">{proba[2]*100:.0f}%</div>
              <div class="metric-label">High</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability bar chart ──
        st.markdown('<div class="section-label" style="margin-top:1.5rem">Class Probabilities</div>', unsafe_allow_html=True)
        fig_proba = go.Figure(go.Bar(
            x=["Low", "Medium", "High"],
            y=[p * 100 for p in proba],
            marker_color=["#4ade80", "#facc15", "#f87171"],
            text=[f"{p*100:.1f}%" for p in proba],
            textposition="outside",
            textfont=dict(family="DM Mono", size=11, color="#aaa"),
        ))
        fig_proba.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Syne", color="#888"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=200,
            yaxis=dict(showgrid=False, showticklabels=False, range=[0, 120]),
            xaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig_proba, use_container_width=True)

        # ── Feature importance ──
        with st.expander("Feature Importance (Regressor)", expanded=True):
            clean_names = [
                n.replace("remainder__", "").replace("cat__", "").replace("_", " ").title()
                for n in fi_df["feature"].tolist()
            ]
            fig_fi = go.Figure(go.Bar(
                x=fi_df["importance"].tolist()[::-1],
                y=clean_names[::-1],
                orientation="h",
                marker=dict(
                    color=fi_df["importance"].tolist()[::-1],
                    colorscale=[[0, "#1e1e2e"], [1, "#f5c518"]],
                ),
                text=[f"{v:.3f}" for v in fi_df["importance"].tolist()[::-1]],
                textposition="outside",
                textfont=dict(family="DM Mono", size=9, color="#555"),
            ))
            fig_fi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Syne", color="#888", size=11),
                margin=dict(l=10, r=40, t=10, b=0),
                height=320,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False),
                showlegend=False,
            )
            st.plotly_chart(fig_fi, use_container_width=True)

    else:
        # Placeholder state
        st.markdown("""
        <div class="card" style="min-height: 320px; display:flex; align-items:center; justify-content:center; flex-direction:column; gap:0.75rem;">
          <div style="font-size:2.5rem">🚕</div>
          <div style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#333; letter-spacing:0.1em;">
            FILL IN TRIP DETAILS AND HIT PREDICT
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
  Trained on NYC TLC Yellow Taxi data · January 2025 only · XGBoost regression + classification ·
  Features: trip distance, duration, pickup zone, borough, time of day, weather conditions, demand patterns ·
  This is a portfolio project — predictions are estimates based on historical patterns.
</div>
""", unsafe_allow_html=True)
