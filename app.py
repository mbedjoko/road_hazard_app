import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Road Hazard Predictor – Cameroon",
    page_icon="🛣️",
    layout="centered"
)

# ── Load saved model & encoders ──────────────────────────────
@st.cache_resource
def load_model():
    model      = joblib.load('road_hazard_model.pkl')
    le_target  = joblib.load('le_target.pkl')
    le_dict    = joblib.load('le_features.pkl')
    feat_cols  = joblib.load('feature_columns.pkl')
    return model, le_target, le_dict, feat_cols

model, le_target, le_dict, feat_cols = load_model()

# ── App header ───────────────────────────────────────────────
st.title("Road Hazard Risk Predictor")
st.caption("Cameroon Intelligent Road Safety System — real-time inference")
st.divider()

# ── Input form ───────────────────────────────────────────────
st.subheader("Enter current road conditions")

col1, col2 = st.columns(2)

with col1:
    city         = st.selectbox("City", ["Douala", "Yaoundé", "Bafoussam", "Bamenda"])
    road_type    = st.selectbox("Road type", ["asphalt", "dirt", "mixed"])
    season       = st.selectbox("Season", ["rainy", "dry"])
    traffic      = st.selectbox("Traffic density", ["low", "medium", "high"])
    hazard_type  = st.selectbox("Observed hazard", ["none", "pothole", "flood", "accident", "construction"])
    day_of_week  = st.selectbox("Day of week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

with col2:
    rainfall     = st.slider("Rainfall (mm)",           0.0, 150.0, 20.0)
    humidity     = st.slider("Humidity (%)",            30.0, 99.0, 65.0)
    temperature  = st.slider("Temperature (°C)",        18.0, 42.0, 28.0)
    hazard_sev   = st.slider("Hazard severity (0–5)",   0, 5, 1)
    hour         = st.slider("Hour of day",             0, 23, 12)
    vehicle_cnt  = st.number_input("Vehicle count estimate", 0, 1000, 80)

st.divider()
st.subheader("Historical & sensor data")

col3, col4 = st.columns(2)
with col3:
    past_7d      = st.number_input("Hazards last 7 days",   0, 50, 3)
    past_30d     = st.number_input("Hazards last 30 days",  0, 150, 10)
    damage_score = st.slider("Road damage score (0–1)",     0.0, 1.0, 0.3)
    accident_hist= st.number_input("Accident history count",0, 50, 5)
    repair_freq  = st.slider("Repair frequency (per year)", 0.0, 5.0, 1.0)

with col4:
    avg_rain_hist    = st.number_input("Avg monthly rainfall (mm)", 0.0, 200.0, 60.0)
    avg_speed        = st.slider("Avg speed (km/h)",                5.0, 120.0, 40.0)
    curr_intensity   = st.slider("Current rainfall intensity",      0.0, 150.0, 10.0)
    recent_reports   = st.number_input("Reports last 1 hour",       0, 20, 1)
    last_report_min  = st.number_input("Last report (mins ago)",    0, 120, 30)
    sensor_alert     = st.selectbox("Sensor alert", [0, 1])

# ── Derived / feature store scores ───────────────────────────
flood_risk    = min(1.0, (rainfall/120)*0.5 + (0.3 if season=="rainy" else 0) + damage_score*0.2)
road_stab     = max(0.0, 1 - damage_score - (0.1 if road_type=="dirt" else 0))
cong_risk     = 0.8 if traffic=="high" else 0.4 if traffic=="medium" else 0.1
comb_score    = flood_risk*0.35 + (1-road_stab)*0.35 + cong_risk*0.30

# ── Build input row ──────────────────────────────────────────
def encode(col, val):
    if col in le_dict:
        try:
            return le_dict[col].transform([str(val)])[0]
        except:
            return 0
    return val

raw = {
    "road_type": road_type, "city": city,
    "rainfall_mm": rainfall, "humidity_percent": humidity,
    "temperature_celsius": temperature, "season": season,
    "traffic_density": traffic,
    "vehicle_count_estimate": vehicle_cnt,
    "avg_speed_kmh": avg_speed,
    "hazard_type": hazard_type, "hazard_severity": hazard_sev,
    "past_7day_hazard_count": past_7d,
    "past_30day_hazard_count": past_30d,
    "road_damage_history_score": damage_score,
    "avg_monthly_rainfall_history": avg_rain_hist,
    "accident_history_count": accident_hist,
    "road_repair_frequency": repair_freq,
    "current_rainfall_intensity": curr_intensity,
    "recent_reports_last_1h": recent_reports,
    "last_report_time_minutes_ago": last_report_min,
    "sensor_alert_flag": sensor_alert,
    "flood_risk_score": round(flood_risk, 3),
    "road_stability_score": round(road_stab, 3),
    "congestion_risk_score": round(cong_risk, 3),
    "combined_hazard_score": round(comb_score, 3),
    "day_of_week": day_of_week, "hour_of_day": hour,
}

input_row = {col: encode(col, raw[col]) for col in feat_cols if col in raw}
# Fill any missing columns with 0
for col in feat_cols:
    if col not in input_row:
        input_row[col] = 0

input_df = pd.DataFrame([input_row])[feat_cols]

# ── Predict ──────────────────────────────────────────────────
st.divider()

if st.button("Predict Risk Level", type="primary", use_container_width=True):
    pred_encoded = model.predict(input_df)[0]
    pred_label   = le_target.inverse_transform([pred_encoded])[0]
    proba        = model.predict_proba(input_df)[0]
    classes      = le_target.classes_

    # Risk level badge
    color = {"low": "green", "medium": "orange", "high": "red"}.get(pred_label, "gray")
    st.markdown(f"### Predicted risk: :{color}[**{pred_label.upper()}**]")

    # Confidence bars
    st.subheader("Confidence breakdown")
    for cls, prob in sorted(zip(classes, proba), key=lambda x: -x[1]):
        st.progress(float(prob), text=f"{cls.capitalize()}: {prob:.1%}")

    # Auto-computed feature store scores
    st.subheader("Auto-computed feature store scores")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flood risk",    f"{flood_risk:.2f}")
    c2.metric("Road stability",f"{road_stab:.2f}")
    c3.metric("Congestion",    f"{cong_risk:.2f}")
    c4.metric("Combined score",f"{comb_score:.2f}")

    # Advice
    st.subheader("Recommended action")
    advice = {
        "low":    "Road conditions are acceptable. Normal monitoring recommended.",
        "medium": "Moderate risk detected. Alert field teams and increase patrol frequency.",
        "high":   "HIGH RISK. Immediate intervention required. Consider road closure or emergency response."
    }
    st.info(advice[pred_label])