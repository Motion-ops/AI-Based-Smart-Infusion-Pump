"""
===========================================================
DASHBOARD V2
Smart AI Infusion Pump - Live Digital Twin Monitor

Runs a live virtual patient through the full V2 pipeline:
  digital twin (true mechanistic physiology)
        -> noisy sensor readings (Measured MAP/HR/SpO2)
        -> trained AI classifier + regressor (advisory recommendation)
        -> expert_controller (the ONLY thing that ever actuates the pump)
        -> ai_consistency_monitor (flags AI/controller disagreement)
        -> risk classifier (independent look-ahead deterioration score)

This is the AI-overlay version, now that generate_dataset.py /
train_ai.py / train_risk_model.py have been rebuilt against the V2
twin and models/*.pkl exist again. Feature construction here
(build_feature_row / build_risk_feature_row) mirrors generate_dataset.py
exactly - same noise SDs, same rolling-window update order (append
before computing stats) - so live inference sees the same feature
distribution the models were trained on.

Layout: ICU-style monitor cards -> target status strip -> risk monitor
-> live scrolling charts (MAP/HR/SpO2/Flow/Plasma/Risk) -> AI vs
controller decision panels -> patient/hemodynamics panel -> event log.

Run with:
    pip install streamlit
    streamlit run dashboard.py
===========================================================
"""

import time
import random
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import joblib
import streamlit as st

import constants as c
from digital_twin import VirtualPatient, TARGET_MAP_MMHG
from patient_generator import PROFILE_WEIGHTS
from expert_controller import expert_controller, TARGET_DEADBAND, EMERGENCY_MAP_MMHG, MAX_STEP_MCG_KG_MIN
from pharmacokinetics import mass_rate_to_flow_ml_hr
from ai_consistency_monitor import evaluate_ai_recommendation
from generate_dataset import (
    FEATURE_COLUMNS, RISK_FEATURE_COLUMNS, WINDOW, LOOKAHEAD,
    BP_NOISE_SD, HR_NOISE_SD, SPO2_NOISE_SD, _rolling_slope
)

st.set_page_config(page_title="Smart AI Infusion Pump", layout="wide")

GREEN, YELLOW, RED, GREY = "#22c55e", "#eab308", "#ef4444", "#888888"
SPEED_MAP = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0, "5x": 5.0}
MAX_EVENT_LOG = 30
RISK_ALERT_THRESHOLD = 0.5


# -----------------------------------------------------
# Model loading (cached)
# -----------------------------------------------------
@st.cache_resource
def load_models():
    clf = joblib.load("models/ai_action_classifier.pkl")
    reg = joblib.load("models/ai_flow_regressor.pkl")
    risk_clf = joblib.load("models/risk_classifier.pkl")
    return clf, reg, risk_clf


# -----------------------------------------------------
# Status/color helpers - heuristic clinical thresholds,
# for display only, not diagnostic claims.
# -----------------------------------------------------
def bp_status(bp, target, deadband):
    if bp < EMERGENCY_MAP_MMHG:
        return "Critical", RED
    if abs(bp - target) <= deadband:
        return "Normal", GREEN
    return "Off-Target", YELLOW


def hr_status(hr):
    if hr < 50 or hr > 140:
        return "Critical", RED
    if hr < 60 or hr > 100:
        return ("Low" if hr < 60 else "Elevated"), YELLOW
    return "Normal", GREEN


def spo2_status(spo2):
    if spo2 < 90:
        return "Critical", RED
    if spo2 < 95:
        return "Low", YELLOW
    return "Normal", GREEN


def flow_status(flow, max_flow=c.MAX_FLOW_ML_HR):
    if flow >= max_flow - 0.01:
        return "At Max", YELLOW
    if flow <= c.MIN_FLOW_ML_HR + 0.01:
        return "Off", GREY
    return "Normal", GREEN


def mode_status(mode):
    if mode == "Maintain":
        return "Stable", GREEN
    if mode in ("Increase", "Decrease"):
        return mode, YELLOW
    if mode == "Emergency":
        return "EMERGENCY", RED
    return mode, GREY


def render_card(col, title, value, status_label, color):
    col.markdown(f"""
    <div style="border:1px solid #333;border-radius:10px;padding:12px 8px;
                text-align:center;background-color:rgba(255,255,255,0.03);">
      <div style="font-size:13px;color:#999;">{title}</div>
      <div style="font-size:26px;font-weight:700;">{value}</div>
      <div style="font-size:13px;color:{color};">&#9679; {status_label}</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------
# Event log
# -----------------------------------------------------
def log_event(message):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.event_log.append(f"{ts}  {message}")
    st.session_state.event_log = st.session_state.event_log[-MAX_EVENT_LOG:]


def detect_events(prev, latest):
    """Compare two consecutive ticks and log anything worth noting."""
    if prev is None:
        return

    if latest["ControllerMode"] != prev["ControllerMode"]:
        if latest["ControllerMode"] == "Emergency":
            log_event("Controller escalated to EMERGENCY")
        elif prev["ControllerMode"] == "Emergency":
            log_event("Controller exited Emergency mode")
        else:
            log_event(f"Controller mode changed to {latest['ControllerMode']}")

    if latest["InTargetRange"] and not prev["InTargetRange"]:
        log_event("MAP entered target range")
    elif not latest["InTargetRange"] and prev["InTargetRange"]:
        log_event("MAP left target range")

    if not latest["ActionAgrees"] and prev["ActionAgrees"]:
        log_event(f"AI disagreed with controller ({latest['Alert']})")
    elif latest["ActionAgrees"] and not prev["ActionAgrees"]:
        log_event("AI agreement with controller restored")

    # Risk score crossing - this is the genuinely new signal category.
    # The controller has no equivalent: it cannot look ahead in time.
    if latest["RiskScore"] >= RISK_ALERT_THRESHOLD > prev["RiskScore"]:
        log_event(f"AI RISK ALERT: {latest['RiskScore']*100:.0f}% probability of "
                  f"deterioration in next {LOOKAHEAD}s (controller has no equivalent forecast)")
    elif prev["RiskScore"] >= RISK_ALERT_THRESHOLD > latest["RiskScore"]:
        log_event(f"AI risk alert cleared ({latest['RiskScore']*100:.0f}%)")

    if latest.get("InfusionComplete") and not prev.get("InfusionComplete"):
        log_event("VTBI reached - infusion complete")


# -----------------------------------------------------
# Feature construction - mirrors generate_dataset.py exactly, so live
# inference sees the same feature distribution the models were trained on.
# -----------------------------------------------------
def build_feature_row(patient, measured_map, measured_hr, measured_spo2, prev_measured_map):
    bp_error = patient.target_bp - measured_map
    bp_trend = measured_map - prev_measured_map
    shock_index = measured_hr / max(measured_map, 1)
    perfusion = measured_map * measured_spo2 / 100
    oxy_deficit = 100 - measured_spo2

    row = {
        "MeasuredMAP": measured_map, "MeasuredHR": measured_hr, "MeasuredSpO2": measured_spo2,
        "FlowRate": patient.flow_rate, "WeightKg": patient.weight_kg, "Age": patient.age,
        "TargetBP": patient.target_bp,
        "MeasuredBP_Error": bp_error, "MeasuredBP_Trend": bp_trend,
        "MeasuredShockIndex": shock_index, "MeasuredPerfusionIndex": perfusion,
        "MeasuredOxygenDeficit": oxy_deficit
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS], row


def build_risk_feature_row(base_row, map_window, hr_window, spo2_window):
    """Extends the single-tick feature row with rolling-window stats,
    computed identically to generate_dataset.py so the live risk model
    sees the same feature distribution it was trained on."""
    risk_row = dict(base_row)
    risk_row["RollMeanMAP"] = float(np.mean(map_window))
    risk_row["RollStdMAP"] = float(np.std(map_window))
    risk_row["RollSlopeMAP"] = _rolling_slope(map_window)
    risk_row["RollMeanSpO2"] = float(np.mean(spo2_window))
    risk_row["RollStdSpO2"] = float(np.std(spo2_window))
    risk_row["RollSlopeSpO2"] = _rolling_slope(spo2_window)
    risk_row["RollMeanHR"] = float(np.mean(hr_window))
    risk_row["RollSlopeHR"] = _rolling_slope(hr_window)
    return pd.DataFrame([risk_row])[RISK_FEATURE_COLUMNS]


def step_simulation(patient, clf, reg, risk_clf):
    """Advance the patient one tick. expert_controller is the sole
    actuator; both AI models are advisory only and never touch the
    pump. clf/reg predict the current-tick recommended action/flow
    (imitation-style advisory). risk_clf predicts deterioration risk
    over the next LOOKAHEAD ticks using rolling-window trend history -
    a genuinely different, forward-looking task, not a copy of the
    controller."""
    patient.update()
    expert_decision = expert_controller(patient)

    measured_map = float(np.clip(patient.map + np.random.normal(0, BP_NOISE_SD), 20, 200))
    measured_hr = float(np.clip(patient.hr + np.random.normal(0, HR_NOISE_SD), 30, 220))
    measured_spo2 = float(np.clip(patient.spo2 + np.random.normal(0, SPO2_NOISE_SD), 70, 100))

    prev_measured_map = st.session_state.get("prev_measured_map")
    if prev_measured_map is None:
        prev_measured_map = measured_map
    X, base_row = build_feature_row(patient, measured_map, measured_hr, measured_spo2, prev_measured_map)
    st.session_state["prev_measured_map"] = measured_map

    # Update rolling windows BEFORE computing risk features - same
    # order as generate_dataset.py, so live inference matches training
    # exactly.
    st.session_state.map_window.append(measured_map)
    st.session_state.hr_window.append(measured_hr)
    st.session_state.spo2_window.append(measured_spo2)

    risk_X = build_risk_feature_row(
        base_row, st.session_state.map_window, st.session_state.hr_window, st.session_state.spo2_window
    )
    risk_prob = float(risk_clf.predict_proba(risk_X)[0, 1])

    ai_action = clf.predict(X)[0]
    ai_flow = float(reg.predict(X)[0])

    # "Clinically meaningful divergence" scaled to THIS patient's
    # weight: one real titration step (MAX_STEP_MCG_KG_MIN) converted
    # to this patient's own mL/hr terms - see ai_consistency_monitor.py.
    flow_disagreement_threshold = mass_rate_to_flow_ml_hr(MAX_STEP_MCG_KG_MIN * patient.weight_kg)
    comparison = evaluate_ai_recommendation(expert_decision, ai_action, ai_flow, flow_disagreement_threshold)

    # The pump only ever obeys the controller.
    patient.flow_rate = expert_decision["RecommendedFlow"]

    in_target = abs(patient.map - patient.target_bp) <= TARGET_DEADBAND

    return {
        "Time": len(st.session_state.history),
        "MAP": round(patient.map, 2),
        "MeasuredMAP": round(measured_map, 2),
        "TargetBP": patient.target_bp,
        "InTargetRange": in_target,
        "HR": round(patient.hr, 2),
        "SpO2": round(patient.spo2, 2),
        "ControllerMode": comparison["ControllerMode"],
        "ControllerFlow": comparison["ControllerFlow"],
        "ClinicalExplanation": expert_decision["ClinicalExplanation"],
        "AI_Action": comparison["AI_Action"],
        "AI_SuggestedFlow": comparison["AI_SuggestedFlow"],
        "FlowDivergence": comparison["FlowDivergence"],
        "ActionAgrees": comparison["ActionAgrees"],
        "Alert": comparison["Alert"],
        "StabilityScore": expert_decision["StabilityScore"],
        "Confidence": expert_decision["Confidence"],
        "ShockIndex": expert_decision["ShockIndex"],
        "PlasmaConcentration": round(patient.plasma_concentration, 3),
        "DrugEffect": round(patient.drug_effect, 3),
        "SVR": round(patient.svr, 1),
        "CO": round(patient.co, 2),
        "RiskScore": risk_prob,
        "InfusionComplete": patient.infusion_complete,
        # Display-only: never fed back into dosing. Flags a patient
        # whose MAP is adequate but HR remains persistently high - worth
        # clinician review, not something NE dosing can/should address.
        "HRAdvisory": in_target and patient.hr > c.HR_ADVISORY_THRESHOLD_BPM,
        "TimeInTargetMin": round(patient.time_in_target_min, 1),
    }


def new_patient(profile_choice, seed, target_map):
    profile = None if profile_choice == "Random" else profile_choice
    st.session_state.patient = VirtualPatient(0, profile=profile, seed=seed, target_map=target_map)
    st.session_state.history = []
    st.session_state.event_log = []
    st.session_state.prev_measured_map = None
    st.session_state.tick_accumulator = 0.0
    st.session_state.running = False
    st.session_state.map_window = deque(maxlen=WINDOW)
    st.session_state.hr_window = deque(maxlen=WINDOW)
    st.session_state.spo2_window = deque(maxlen=WINDOW)
    log_event(f"Patient created (profile={st.session_state.patient.profile})")


# -----------------------------------------------------
# Sidebar controls
# -----------------------------------------------------
st.sidebar.title("Controls")

profile_choice = st.sidebar.radio(
    "Clinical Profile", ["Random"] + list(PROFILE_WEIGHTS.keys())
)

target_lo, target_hi = c.TARGET_MAP_RANGE_MMHG
target_map_choice = st.sidebar.number_input(
    "Target MAP (mmHg)", min_value=target_lo, max_value=target_hi,
    value=TARGET_MAP_MMHG, step=1.0,
    help=("Individualized MAP target, within the range actually studied by the "
          "SEPSISPAM trial (65-70 vs 80-85 mmHg in septic shock, Asfar et al. "
          "NEJM 2014) - not an arbitrary slider range. The controller honors "
          "whatever target you set; the AI advisory models were trained across "
          "this same range so they stay meaningful at any value within it.")
)
speed_choice = st.sidebar.radio("Simulation Speed", list(SPEED_MAP.keys()), index=1, horizontal=True)

col_start, col_pause = st.sidebar.columns(2)
start_clicked = col_start.button("Start", width='stretch')
pause_clicked = col_pause.button("Pause", width='stretch')

col_reset, col_new = st.sidebar.columns(2)
reset_clicked = col_reset.button("Reset Patient", width='stretch')
new_random_clicked = col_new.button("New Random Patient", width='stretch')

st.sidebar.markdown("---")
st.sidebar.caption(
    "The controller always drives the pump and brings MAP to target. "
    "The AI only proposes a parallel recommendation from noisy sensor "
    "readings - it never actuates the pump directly."
)

# -----------------------------------------------------
# Init / control handling
# -----------------------------------------------------
if "patient" not in st.session_state:
    st.session_state.seed = 1
    new_patient(profile_choice, st.session_state.seed, target_map_choice)

if reset_clicked:
    new_patient(profile_choice, st.session_state.seed, target_map_choice)

if new_random_clicked:
    st.session_state.seed = random.randint(0, 1_000_000)
    new_patient(profile_choice, st.session_state.seed, target_map_choice)

if start_clicked:
    st.session_state.running = True
    log_event("Simulation started")

if pause_clicked:
    st.session_state.running = False
    log_event("Simulation paused")

clf, reg, risk_clf = load_models()
patient = st.session_state.patient

# -----------------------------------------------------
# Main layout
# -----------------------------------------------------
st.title("Smart AI Infusion Pump - Digital Twin (V2)")
st.caption(f"Profile: {patient.profile}  |  SAPS II: {patient.saps_ii}  |  Status: "
           f"{'RUNNING' if st.session_state.running else 'PAUSED'}")

if st.session_state.history:
    latest = st.session_state.history[-1]

    # --- ICU monitor cards ---
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    label, color = bp_status(latest["MAP"], latest["TargetBP"], TARGET_DEADBAND)
    render_card(c1, "MAP", f"{latest['MAP']} mmHg", label, color)
    label, color = hr_status(latest["HR"])
    render_card(c2, "HR", f"{latest['HR']} bpm", label, color)
    label, color = spo2_status(latest["SpO2"])
    render_card(c3, "SpO2", f"{latest['SpO2']} %", label, color)
    label, color = flow_status(latest["ControllerFlow"])
    render_card(c4, "Flow Rate", f"{latest['ControllerFlow']} mL/hr", label, color)
    label, color = mode_status(latest["AI_Action"])
    render_card(c5, "AI Mode", latest["AI_Action"], label, color)
    label, color = mode_status(latest["ControllerMode"])
    render_card(c6, "Controller", latest["ControllerMode"], label, color)

    st.markdown("")

    # --- Target status strip ---
    hist_df = pd.DataFrame(st.session_state.history)
    pct_in_target = hist_df["InTargetRange"].mean() * 100
    reached_ticks = hist_df[hist_df["InTargetRange"]]
    first_reached = int(reached_ticks["Time"].iloc[0]) if not reached_ticks.empty else None

    colT1, colT2, colT3 = st.columns(3)
    if latest["InTargetRange"]:
        colT1.success(f"AT TARGET (within +/-{TARGET_DEADBAND} mmHg of {latest['TargetBP']})")
    else:
        gap = round(latest['TargetBP'] - latest['MAP'], 1)
        colT1.warning(f"NOT YET AT TARGET ({abs(gap)} mmHg {'below' if gap > 0 else 'above'} target)")
    colT2.metric("Time in target so far", f"{pct_in_target:.0f}%")
    colT3.metric("First reached target at", f"{first_reached}s" if first_reached is not None else "Not yet")

    # --- Risk monitor (forward-looking, not a current-tick imitation) ---
    st.markdown("### AI Risk Monitor (independent early-warning model)")
    st.caption(
        f"Predicts probability of deterioration (MAP<{EMERGENCY_MAP_MMHG:.0f} or SpO2<90) "
        f"in the next {LOOKAHEAD} seconds, using rolling trend history. This is NOT what "
        "the controller does - the controller only reacts to the current tick and cannot "
        "forecast. Disagreement here is genuine forward-looking signal, not sensor-noise "
        "disagreement."
    )
    risk = latest["RiskScore"]
    colR1, colR2 = st.columns([1, 3])
    with colR1:
        st.metric("Risk Score", f"{risk*100:.0f}%")
    with colR2:
        if risk >= RISK_ALERT_THRESHOLD:
            st.error(f"ELEVATED RISK: {risk*100:.0f}% probability of deterioration in next {LOOKAHEAD}s")
        elif risk >= 0.2:
            st.warning(f"Watch: {risk*100:.0f}% probability of deterioration in next {LOOKAHEAD}s")
        else:
            st.success(f"Low risk ({risk*100:.0f}%)")

    # --- Live charts ---
    st.markdown("### Mean Arterial Pressure (MAP)")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    target = latest["TargetBP"]
    ax.axhspan(target - TARGET_DEADBAND, target + TARGET_DEADBAND, color="green", alpha=0.15)
    ax.plot(hist_df["Time"], hist_df["MAP"], label="True MAP", color="tab:blue")
    ax.plot(hist_df["Time"], hist_df["MeasuredMAP"], label="Measured MAP",
            color="tab:orange", linestyle="--", alpha=0.6)
    ax.axhline(target, color="black", linestyle=":", label=f"Target ({target:.0f} mmHg)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("mmHg"); ax.legend(loc="upper right")
    st.pyplot(fig)
    plt.close(fig)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("### Heart Rate")
        st.line_chart(hist_df.set_index("Time")[["HR"]])
    with chart_col2:
        st.markdown("### Oxygen Saturation")
        st.line_chart(hist_df.set_index("Time")[["SpO2"]])

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        st.markdown("### Infusion Flow Rate (Controller vs AI)")
        st.line_chart(hist_df.set_index("Time")[["ControllerFlow", "AI_SuggestedFlow"]])
    with chart_col4:
        st.markdown("### Plasma Norepinephrine Concentration (Cp)")
        st.line_chart(hist_df.set_index("Time")[["PlasmaConcentration"]])

    st.markdown("### Deterioration Risk Score Over Time")
    # RiskScore is a probability, always in [0,1] by definition - fixing
    # the y-domain avoids relying on Vega-Lite's auto-scale, which was
    # observed to occasionally produce a garbled axis (ticks scaled by
    # ~1e23) on this specific chart when its data range is tiny and
    # changing every rerun. A fixed domain is also better UX here
    # regardless: a consistent 0-1 scale over time, not one that keeps
    # rescaling to whatever narrow range the last few ticks happen to span.
    risk_chart = (
        alt.Chart(hist_df.reset_index())
        .mark_line(color="#ef4444")
        .encode(
            x=alt.X("Time:Q", title="Time (s)"),
            y=alt.Y("RiskScore:Q", title="Risk Score", scale=alt.Scale(domain=[0, 1])),
        )
    )
    st.altair_chart(risk_chart, width='stretch')

    # --- Bottom panels ---
    colAI, colCtrl, colHemo, colLog = st.columns(4)

    with colAI:
        st.markdown("#### AI (advisory)")
        st.write(f"**Predicted Action:** {latest['AI_Action']}")
        st.write(f"**Predicted Flow:** {latest['AI_SuggestedFlow']} mL/hr")
        st.write(f"**Flow Divergence:** {latest['FlowDivergence']} mL/hr")
        if latest["ActionAgrees"]:
            st.success("Agrees with Controller")
        else:
            st.warning("Disagrees with Controller")
        if latest["Alert"]:
            st.caption(latest["Alert"])

    with colCtrl:
        st.markdown("#### Controller (sole actuator)")
        st.write(f"**Mode:** {latest['ControllerMode']}")
        st.write(f"**Flow Rate:** {latest['ControllerFlow']} mL/hr")
        st.write(f"**Reason:** {latest['ClinicalExplanation']}")
        st.write(f"**Confidence:** {latest['Confidence']}")
        st.write(f"**Stability Score:** {latest['StabilityScore']}")

    with colHemo:
        st.markdown("#### Patient & Hemodynamics (hidden from AI)")
        st.write(f"**Age / Weight:** {patient.age:.0f}y / {patient.weight_kg:.0f} kg")
        st.write(f"**Profile:** {patient.profile} (SAPS II {patient.saps_ii})")
        st.write(f"**SVR:** {latest['SVR']} dyn·s/cm⁵ | **CO:** {latest['CO']} L/min")
        st.write(f"**Drug Effect (E):** {latest['DrugEffect']} | **Plasma Cp:** {latest['PlasmaConcentration']} mcg/L")
        st.write(f"**Clearance:** {patient.clearance_l_min:.2f} L/min | **Vd:** {patient.vd_l:.1f} L")
        st.write(f"**Time stable at target:** {latest['TimeInTargetMin']:.1f} min")
        if latest["HRAdvisory"]:
            st.warning("HR elevated despite adequate MAP — consider non-shock causes "
                       "(pain, agitation, fever). Not addressable by NE dosing.")

    with colLog:
        st.markdown("#### Event Log")
        if st.session_state.event_log:
            for entry in reversed(st.session_state.event_log):
                st.text(entry)
        else:
            st.caption("No events yet.")

    with st.expander("Raw history table"):
        st.dataframe(hist_df, width='stretch')

else:
    st.info("Click 'New Random Patient' in the sidebar, then 'Start' to begin.")

# -----------------------------------------------------
# Auto-refresh loop - advances the simulation ~1x/second,
# ticks-per-refresh scaled by the chosen speed multiplier.
# -----------------------------------------------------
if st.session_state.running:
    speed = SPEED_MAP[speed_choice]
    st.session_state.tick_accumulator += speed
    prev_tick = st.session_state.history[-1] if st.session_state.history else None

    while st.session_state.tick_accumulator >= 1.0:
        result = step_simulation(st.session_state.patient, clf, reg, risk_clf)
        detect_events(prev_tick, result)
        st.session_state.history.append(result)
        prev_tick = result
        st.session_state.tick_accumulator -= 1.0

    time.sleep(1.0)
    st.rerun()
