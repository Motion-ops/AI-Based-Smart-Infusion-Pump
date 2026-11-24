"""
===========================================================
NATIVE_APP.PY
Smart AI Infusion Pump - Native Desktop Monitor (PyQt6 + pyqtgraph)

A true native rewrite of dashboard.py. Same simulation core (digital
twin, expert controller, trained AI models, risk model) - only the UI
layer changes. No browser, no web engine: real Qt widgets throughout,
with pyqtgraph driving the live waveform charts, closer to how actual
bedside monitor / infusion pump software is built.

Safety architecture (unchanged from dashboard.py):
    expert_controller is the ONLY thing that ever sets patient.flow_rate.
    The AI classifier/regressor and the risk model are advisory only -
    they are displayed for comparison and never touch actuation.

Optional hardware-in-the-loop mode (PumpLink, sidebar "Physical Pump"
panel): when connected to a real ESP32 running firmware/pump_firmware/
pump_firmware.ino, the exact same controller-computed rate that drives
the on-screen charts is also streamed to the real motor - over wired
USB-Serial OR wireless WiFi (pick with the Wired/Wireless toggle),
using the identical text protocol (SET_RATE:/STOP/SET_WEIGHT_DOSE:/
SET_VTBI_PROGRAM:) both esp32_bridge.py and tools/wifi_bridge_test.py
validated. A "Manual Pump Dosing" panel below it can also send a
weight-based dose or a Rate+VTBI program straight to the physical
pump, independent of the on-screen simulated patient. Purely simulated
when not connected - nothing changes for anyone just running the
digital twin on-screen.

Run with:
    pip install PyQt6 pyqtgraph pyserial
    python native_app.py
===========================================================
"""

import sys
import os
import json
import random
import socket
import threading
import time
import uuid
import winsound
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import pyqtgraph as pg
import serial
import serial.tools.list_ports

try:
    from Adafruit_IO import Client as AdafruitIOClient
except ImportError:
    AdafruitIOClient = None   # ADAFRUIT_IO_ENABLED is guarded below either way
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QRadioButton, QButtonGroup, QDoubleSpinBox, QFrame,
    QGroupBox, QTextEdit, QTableWidget, QTableWidgetItem, QTabWidget,
    QHeaderView, QSizePolicy, QScrollArea, QCheckBox, QComboBox, QMessageBox,
    QLineEdit,
)

import constants as c
from digital_twin import VirtualPatient, TARGET_MAP_MMHG
from patient_generator import PROFILE_WEIGHTS
from expert_controller import (
    expert_controller, TARGET_DEADBAND, EMERGENCY_MAP_MMHG, MAX_STEP_MCG_KG_MIN,
)
from pharmacokinetics import mass_rate_to_flow_ml_hr, flow_rate_to_mass_rate
from ai_consistency_monitor import evaluate_ai_recommendation
import clinical_alerts
from generate_dataset import (
    FEATURE_COLUMNS, RISK_FEATURE_COLUMNS, WINDOW, LOOKAHEAD,
    BP_NOISE_SD, HR_NOISE_SD, SPO2_NOISE_SD, _rolling_slope,
)

# Second clinical mode - hypertensive emergency / sodium nitroprusside.
# Aliased on import (not just at call sites) since several names collide
# with the septic-shock side's own (different-content) versions of the
# same concept - WINDOW/BP_NOISE_SD/HR_NOISE_SD/SPO2_NOISE_SD/
# _rolling_slope/LOOKAHEAD are identical in both modules so those are
# safely reused from the septic import above without duplicating them.
import hypertensive_constants as hc
from hypertensive_twin import VirtualHypertensivePatient
from hypertensive_expert_controller import (
    hypertensive_expert_controller, MAX_STEP_MCG_KG_MIN as HT_MAX_STEP_MCG_KG_MIN,
)
from nitroprusside_pharmacokinetics import (
    mass_rate_to_flow_ml_hr as ht_mass_rate_to_flow_ml_hr,
    flow_rate_to_mass_rate as ht_flow_rate_to_mass_rate,
)
from generate_hypertensive_dataset import (
    FEATURE_COLUMNS as HT_FEATURE_COLUMNS,
    RISK_FEATURE_COLUMNS as HT_RISK_FEATURE_COLUMNS,
)
from hypertensive_constants import PROFILE_WEIGHTS as HT_PROFILE_WEIGHTS

# -----------------------------------------------------
# Palette / constants
# -----------------------------------------------------
GREEN, YELLOW, RED, GREY = "#22c55e", "#eab308", "#ef4444", "#9aa1ac"
BG = "#0d0f14"
PANEL = "#151822"
PANEL2 = "#1b1f2b"
BORDER = "#2a2f3d"
TEXT = "#e6e8ec"
TEXT_DIM = "#8b93a1"
ACCENT = "#3b82f6"

SPEED_MAP = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0, "5x": 5.0}
MAX_EVENT_LOG = 30
RISK_ALERT_THRESHOLD = 0.5
MAX_HISTORY_ROWS_SHOWN = 250
TICK_TIMER_MS = 150  # UI/simulation timer resolution

# ---------------- Adafruit IO (optional remote telemetry) ----------------
# [ADDED 2026-10-10] One-way, outbound-only telemetry mirror so the
# system's live vitals can be VIEWED from genuinely far away - true
# internet access, not just the same local WiFi/hotspot the wireless
# pump control and clinician_dashboard.py both require (e.g. a
# supervisor checking in from off-site). Deliberately view-only: this
# never subscribes to anything or accepts commands back, so it cannot
# become a second, remote way to change the dose rate - that stays
# confined to the local wired/wireless control channel, same reasoning
# the phone dashboard is view-only.
#
# Publishes from THIS app, not the ESP32 firmware - the clinically
# meaningful values (MAP/HR/SpO2/risk) only ever exist in this Python
# simulation, never on the pump itself, and keeping this off the
# battery-powered ESP32 avoids adding yet more WiFi current draw to a
# board that already had a real brownout problem from WiFi alone.
#
# Get a free account + API key at https://io.adafruit.com (Account ->
# My Key). Adafruit IO auto-creates feeds on first publish if they
# don't already exist, so creating them ahead of time is optional.
ADAFRUIT_IO_ENABLED = False   # set True once real credentials are filled in below
ADAFRUIT_IO_USERNAME = "CHANGE_ME"
ADAFRUIT_IO_KEY = "CHANGE_ME"
ADAFRUIT_IO_PUBLISH_INTERVAL_SEC = 15.0   # keeps total points/min under the free tier's 30/min cap
ADAFRUIT_IO_FEEDS = {
    "MAP": "pump-map",
    "HR": "pump-hr",
    "SpO2": "pump-spo2",
    "ControllerFlow": "pump-flow",
    "RiskScore": "pump-risk",
}

# ---------------- Remote monitoring dashboard (Firebase) ----------------
# [UPDATED 2026-10-20] The dashboard was rebuilt by the developer on
# Firebase/Firestore (live at infusion-d.web.app) - see
# Infusion_Monitor_Handoff.pdf and remote_dashboard/README.md's own
# note on what this superseded. Same off-by-default,
# fill-in-credentials-later pattern as ADAFRUIT_IO_ENABLED above, and
# the same monitoring-only scope decision as every other remote-access
# feature in this project (clinician_dashboard.py, Adafruit IO) - this
# never sends anything back to the pump, it only ever publishes.
#
# [BLOCKED 2026-10-20] REMOTE_DASHBOARD_DEVICE_PASSWORD is intentionally
# read from an environment variable, never hardcoded - set it once with
# `setx PUMP_DEVICE_PASSWORD "..."` (new terminal afterward) per the
# handoff doc. The actual telemetry_publisher.py matching this new
# constructor signature (Firebase Auth sign-in + Firestore writes) has
# NOT been received from the developer yet - the remote_dashboard/
# folder still has the OLD Supabase-based version, which this
# constructor call does not match. Enabling this before that file
# arrives will raise a TypeError at startup (caught below -> publisher
# simply stays off, same fail-safe behavior as a missing password).
REMOTE_DASHBOARD_ENABLED = True   # set True once gateway/telemetry_publisher.py (Firebase version) is in place
REMOTE_DASHBOARD_FIREBASE_API_KEY = "AIzaSyDyiA-AvfOmUX8BTNRKVjpGO8veivKeSJc"
REMOTE_DASHBOARD_FIREBASE_PROJECT_ID = "c-xpr25"   # e.g. "r-data-a6366"
REMOTE_DASHBOARD_FIRESTORE_DATABASE = "infusion"
REMOTE_DASHBOARD_DEVICE_ID = "prototype-01"           # must match the Settings page exactly
REMOTE_DASHBOARD_DEVICE_EMAIL = "prototype-01@devices.local"           # e.g. "prototype-01@devices.local"
def _resolve_pump_device_password():
    pw = os.environ.get("PUMP_DEVICE_PASSWORD", "")
    if not pw and sys.platform == "win32":
        try:
            import winreg
            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    with winreg.OpenKey(hive, r"Environment") as k:
                        val, _ = winreg.QueryValueEx(k, "PUMP_DEVICE_PASSWORD")
                        if val:
                            return val
                except OSError:
                    pass
        except Exception:
            pass
    return pw

REMOTE_DASHBOARD_DEVICE_PASSWORD = _resolve_pump_device_password()
REMOTE_DASHBOARD_PUBLISH_INTERVAL_SEC = 2.0   # independent of ADAFRUIT_IO's own interval/cap

try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote_dashboard"))
    from telemetry_publisher import RemoteTelemetryPublisher, build_telemetry_payload, parse_status_line
except ImportError:
    RemoteTelemetryPublisher = None   # REMOTE_DASHBOARD_ENABLED is guarded below either way


def resource_path(relative_path):
    """Resolve a data file path that works both when run as a normal
    script and when frozen into a PyInstaller bundle (onedir/onefile),
    where bundled data lives under sys._MEIPASS instead of this file's
    own directory."""
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def writable_output_dir(subfolder):
    """Resolve a directory to WRITE output into (session reports, etc.) -
    the opposite problem from resource_path() above. A frozen PyInstaller
    build's _MEIPASS is a temporary extraction folder that can vanish
    between runs, so output must go next to the actual .exe instead; in
    normal script mode, next to this file."""
    base_path = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) \
        else os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_path, subfolder)
    os.makedirs(path, exist_ok=True)
    return path


def load_model_for_live_prediction(relative_path):
    """Loads a trained RandomForest model and forces n_jobs=1 for
    inference. These models were trained with n_jobs=-1 (parallelize
    across all cores while fitting 500 trees on hundreds of thousands
    of rows - genuinely useful there), but that same setting stays
    attached to the pickled model and also governs .predict()/
    .predict_proba(). The live dashboard only ever predicts ONE row at
    a time, every tick - parallelizing a single-row prediction across
    worker processes is pure overhead (process coordination cost with
    no parallel work to actually split up), and it was silently doing
    this on every tick the whole time. Found by running an extended
    (24 simulated hour) background test of the actual engine, which
    surfaced both the slowdown and floods of an unrelated joblib/loky
    warning that a plain warnings.filterwarnings() in the calling
    script couldn't suppress, because those parallel workers are
    separate OS processes with their own warnings state."""
    model = joblib.load(resource_path(relative_path))
    model.n_jobs = 1
    return model


# -----------------------------------------------------
# Status helpers - identical clinical thresholds to dashboard.py,
# for display only, not diagnostic claims.
# -----------------------------------------------------
def bp_status(bp, target, deadband, at_target_label="Normal"):
    # at_target_label defaults to septic-mode wording. For hypertensive
    # mode, "target" is actually the current staged FLOOR - which for
    # Urgency/Emergency profiles can sit at 133-180 mmHg early in
    # treatment - so printing "Normal" there is technically correct by
    # this function's own "at the current safe reference" logic, but
    # genuinely misleading to read, since that's nowhere near normal
    # blood pressure. The <50 mmHg "Critical" check is left shared/
    # unconditional on purpose - severe hypotension is genuinely
    # dangerous for a patient in EITHER mode, not a septic-only concept.
    if bp < EMERGENCY_MAP_MMHG:
        return "Critical", RED
    if abs(bp - target) <= deadband:
        return at_target_label, GREEN
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


# -----------------------------------------------------
# Physical pump link - same wire protocol esp32_bridge.py already
# validated (SET_RATE:/STOP/GET_STATUS/SET_WEIGHT_DOSE:/
# SET_VTBI_PROGRAM:). This lets the simulation drive the REAL motor:
# expert_controller still only ever decides the rate from a simulated
# patient, exactly as before - this just adds a second destination for
# that number besides the on-screen charts.
#
# [ADDED 2026-10-09] Two transports, one object. Wired (serial) is
# unchanged from before; wireless (TCP, talking to pump_firmware.ino's
# WiFiServer control channel) is new - both present the exact same
# public interface (connect*/connected/send*/poll_lines), so the rest
# of the app (both sim engines, the timer loop) never needs to know
# which one is active. Deliberately a single mode attribute rather
# than two separate classes - there's a standing project precedent
# (see shared_pump below) of exactly one PumpLink instance shared by
# both engines, and two sibling classes would risk reintroducing the
# same per-engine-mismatch bug that precedent was built to avoid.
# Deliberately non-blocking on both transports (serial timeout=0, wifi
# socket.setblocking(False)) so it's safe to call every UI timer tick
# without stalling the GUI thread.
# -----------------------------------------------------
class PumpLink:
    def __init__(self):
        self.ser = None
        self.sock = None
        self.mode = None   # "serial" | "wifi" | None (disconnected)
        self._wifi_recv_buffer = b""

    @staticmethod
    def available_ports():
        return [p.device for p in serial.tools.list_ports.comports()]

    def connect(self, port, baud=115200):
        self.ser = serial.Serial(port, baud, timeout=0)
        # [FIX, 2026-10-01] Opening a serial port toggles DTR/RTS, which
        # on most ESP32 dev boards (via the standard CP2102/CH340
        # auto-reset circuit wired to EN/GPIO0) triggers an actual board
        # reset - the firmware then spends up to ~1-2s back in setup()
        # (LCD/I2C/Preferences init) before it's listening again. Without
        # this pause, every SET_RATE sent immediately after connect()
        # was sent into that reboot window and silently lost - including
        # the ones needed to trigger the firmware's own "any PC command
        # during mode-select auto-selects Autonomous" handshake
        # (handleModeSelection() in pump_firmware.ino), which is the
        # only way this app ever gets the board out of its boot-time
        # "SELECT MODE" screen. The UI briefly freezes for this - a
        # deliberate, rare, user-initiated click, not a per-tick cost.
        time.sleep(2.0)
        self.mode = "serial"

    def connect_wifi(self, host, port=5000, timeout=5.0):
        """Wireless counterpart to connect() - talks to pump_firmware.ino's
        WiFiServer control channel (TCP_CONTROL_PORT). The one brief
        blocking connect() call is acceptable for the same reason
        connect()'s 2.0s sleep above is: rare, user-initiated, not a
        per-tick cost. Non-blocking immediately afterward, same as the
        serial path's timeout=0."""
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.setblocking(False)
        self.sock = sock
        self._wifi_recv_buffer = b""
        self.mode = "wifi"

    def disconnect(self):
        if self.connected:
            try:
                self.send("STOP")
            except Exception:
                pass
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        self.mode = None

    @property
    def connected(self):
        if self.mode == "serial":
            return self.ser is not None and self.ser.is_open
        if self.mode == "wifi":
            return self.sock is not None
        return False

    def send(self, line):
        if not self.connected:
            return
        data = (line + "\n").encode("utf-8")
        try:
            if self.mode == "serial":
                self.ser.write(data)
            else:
                self.sock.sendall(data)
        except Exception:
            # Cable pulled / WiFi link dropped mid-run, etc. - drop the
            # link rather than crash the UI; the pump's own watchdog
            # stops the motor within WATCHDOG_TIMEOUT_MS once commands
            # stop arriving, regardless of which transport was in use.
            self.disconnect()

    def send_rate(self, ml_hr):
        self.send(f"SET_RATE:{ml_hr:.2f}")

    def send_weight_dose(self, weight_kg, dose_value, unit):
        """unit is 'MLHR' or 'UGKGMIN' - matches pump_firmware.ino's
        SET_WEIGHT_DOSE: command exactly, available over either
        transport since both dispatch through the same firmware-side
        handleSerialLine()."""
        self.send(f"SET_WEIGHT_DOSE:{weight_kg:.1f},{dose_value:.3f},{unit}")

    def send_vtbi_program(self, rate_ml_hr, vtbi_ml):
        """Matches pump_firmware.ino's SET_VTBI_PROGRAM: command - starts
        a Rate+VTBI run as one atomic command, instead of the keypad's
        two-step rate-then-VTBI entry."""
        self.send(f"SET_VTBI_PROGRAM:{rate_ml_hr:.2f},{vtbi_ml:.2f}")

    def poll_lines(self):
        """Drain whatever the pump has already sent. Never blocks on
        either transport - serial timeout=0 and the wifi socket's
        non-blocking mode both make reads return immediately with
        whatever's already buffered."""
        if not self.connected:
            return []
        if self.mode == "serial":
            return self._poll_serial_lines()
        return self._poll_wifi_lines()

    def _poll_serial_lines(self):
        lines = []
        try:
            while self.ser.in_waiting:
                raw = self.ser.readline().decode("utf-8", errors="replace").strip()
                if raw:
                    lines.append(raw)
        except Exception:
            self.disconnect()
        return lines

    def _poll_wifi_lines(self):
        """TCP has no readline() - accumulate raw bytes ourselves and
        split on newline. BlockingIOError just means "nothing new yet"
        on a non-blocking socket, not an error."""
        lines = []
        try:
            while True:
                chunk = self.sock.recv(4096)
                if not chunk:
                    self.disconnect()   # peer closed the connection cleanly
                    break
                self._wifi_recv_buffer += chunk
        except BlockingIOError:
            pass
        except Exception:
            self.disconnect()
            return lines

        while b"\n" in self._wifi_recv_buffer:
            raw, self._wifi_recv_buffer = self._wifi_recv_buffer.split(b"\n", 1)
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                lines.append(line)
        return lines


# -----------------------------------------------------
# Simulation engine - the non-UI equivalent of dashboard.py's
# st.session_state. Owns the patient, history, rolling windows, and
# per-tick feature construction (mirrors generate_dataset.py exactly).
# -----------------------------------------------------
class SimEngine:
    def __init__(self, clf, reg, risk_clf, pump):
        self.clf = clf
        self.reg = reg
        self.risk_clf = risk_clf

        self.patient = None
        self.history = []
        self.event_log = []
        self.full_event_log = []   # uncapped, for session-report export - see log_event()
        self.prev_measured_map = None
        self.running = False
        self.tick_accumulator = 0.0
        self.seed = 1
        self.map_window = deque(maxlen=WINDOW)
        self.hr_window = deque(maxlen=WINDOW)
        self.spo2_window = deque(maxlen=WINDOW)
        # Shared across BOTH engines (see MainWindow.__init__) - there is
        # only one real physical pump/USB connection regardless of which
        # simulated condition is currently selected. Each engine
        # previously created its own separate PumpLink, so connecting
        # while in one mode silently did nothing once the dashboard was
        # switched to the other mode - the newly-active engine's own
        # PumpLink had never been connected, even though the UI still
        # showed "Connected" from before.
        self.pump = pump

        # Exposed so shared UI code (_update_cards) can read the right
        # max flow / dose-conversion for whichever engine is currently
        # active, instead of hardcoding the septic-shock ones.
        self.max_flow_ml_hr = c.MAX_FLOW_ML_HR
        self.condition_label = "Hypotensive (Septic Shock)"

    def flow_rate_to_mass_rate(self, flow_ml_hr):
        return flow_rate_to_mass_rate(flow_ml_hr)

    def new_patient(self, profile_choice, seed, target_map):
        profile = None if profile_choice == "Random" else profile_choice
        self.patient = VirtualPatient(0, profile=profile, seed=seed, target_map=target_map)
        self.history = []
        self.event_log = []
        self.full_event_log = []
        self.prev_measured_map = None
        self.running = False
        self.tick_accumulator = 0.0
        self.map_window = deque(maxlen=WINDOW)
        self.hr_window = deque(maxlen=WINDOW)
        self.spo2_window = deque(maxlen=WINDOW)
        self.log_event(f"Patient created (profile={self.patient.profile})")

    def log_event(self, message):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"{ts}  {message}"
        self.event_log.append(entry)
        self.event_log = self.event_log[-MAX_EVENT_LOG:]
        # Session-report export needs the FULL history, not just the
        # last MAX_EVENT_LOG shown on screen - the live view is
        # deliberately capped to stay readable, the export shouldn't be.
        self.full_event_log.append(entry)

    @staticmethod
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
            "MeasuredOxygenDeficit": oxy_deficit,
        }
        return pd.DataFrame([row])[FEATURE_COLUMNS], row

    @staticmethod
    def build_risk_feature_row(base_row, map_window, hr_window, spo2_window):
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

    def step(self):
        """Advance the patient one tick. expert_controller is the sole
        actuator; both AI models are advisory only."""
        patient = self.patient
        patient.update()
        expert_decision = expert_controller(patient)

        measured_map = float(np.clip(patient.map + np.random.normal(0, BP_NOISE_SD), 20, 200))
        measured_hr = float(np.clip(patient.hr + np.random.normal(0, HR_NOISE_SD), 30, 220))
        measured_spo2 = float(np.clip(patient.spo2 + np.random.normal(0, SPO2_NOISE_SD), 70, 100))

        prev_measured_map = self.prev_measured_map
        if prev_measured_map is None:
            prev_measured_map = measured_map
        X, base_row = self.build_feature_row(patient, measured_map, measured_hr, measured_spo2, prev_measured_map)
        self.prev_measured_map = measured_map

        self.map_window.append(measured_map)
        self.hr_window.append(measured_hr)
        self.spo2_window.append(measured_spo2)

        risk_X = self.build_risk_feature_row(base_row, self.map_window, self.hr_window, self.spo2_window)
        risk_prob = float(self.risk_clf.predict_proba(risk_X)[0, 1])

        ai_action = self.clf.predict(X)[0]
        # ai_flow_regressor.pkl now predicts a DELTA (RecommendedFlow -
        # FlowRate), not the absolute rate - see train_ai.py's module
        # docstring "Why delta, not absolute flow". Reconstructed and
        # clamped exactly like the hypertensive engine already does
        # below, for the same reason (an uncapped reconstruction can
        # go negative after a large real controller correction).
        ai_delta = float(self.reg.predict(X)[0])
        ai_flow = float(np.clip(patient.flow_rate + ai_delta, 0.0, self.max_flow_ml_hr))

        flow_disagreement_threshold = mass_rate_to_flow_ml_hr(MAX_STEP_MCG_KG_MIN * patient.weight_kg)
        comparison = evaluate_ai_recommendation(expert_decision, ai_action, ai_flow, flow_disagreement_threshold)

        # The pump only ever obeys the controller.
        patient.flow_rate = expert_decision["RecommendedFlow"]

        # If a real pump is connected, it obeys the exact same number -
        # this is what makes it hardware-in-the-loop rather than pure
        # simulation. Same SET_RATE:/STOP protocol esp32_bridge.py uses.
        if self.pump.connected:
            self.pump.send_rate(patient.flow_rate)

        in_target = abs(patient.map - patient.target_bp) <= TARGET_DEADBAND

        result = {
            "Time": len(self.history),
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
            # whose MAP is adequate but HR remains persistently high -
            # worth clinician review (pain, agitation, fever, etc.),
            # not something norepinephrine dosing can or should address.
            "HRAdvisory": in_target and patient.hr > c.HR_ADVISORY_THRESHOLD_BPM,
            "TimeInTargetMin": round(patient.time_in_target_min, 1),
        }
        return result

    def detect_events(self, prev, latest):
        if prev is None:
            return
        if latest["ControllerMode"] != prev["ControllerMode"]:
            if latest["ControllerMode"] == "Emergency":
                self.log_event("Controller escalated to EMERGENCY")
            elif prev["ControllerMode"] == "Emergency":
                self.log_event("Controller exited Emergency mode")
            else:
                self.log_event(f"Controller mode changed to {latest['ControllerMode']}")

        if latest["InTargetRange"] and not prev["InTargetRange"]:
            self.log_event("MAP entered target range")
        elif not latest["InTargetRange"] and prev["InTargetRange"]:
            self.log_event("MAP left target range")

        if not latest["ActionAgrees"] and prev["ActionAgrees"]:
            self.log_event(f"AI disagreed with controller ({latest['Alert']})")
        elif latest["ActionAgrees"] and not prev["ActionAgrees"]:
            self.log_event("AI agreement with controller restored")

        if latest["RiskScore"] >= RISK_ALERT_THRESHOLD > prev["RiskScore"]:
            self.log_event(f"AI RISK ALERT: {latest['RiskScore']*100:.0f}% probability of "
                            f"deterioration in next {LOOKAHEAD}s (controller has no equivalent forecast)")
        elif prev["RiskScore"] >= RISK_ALERT_THRESHOLD > latest["RiskScore"]:
            self.log_event(f"AI risk alert cleared ({latest['RiskScore']*100:.0f}%)")

        if latest.get("InfusionComplete") and not prev.get("InfusionComplete"):
            self.log_event("VTBI reached - infusion complete")


# -----------------------------------------------------
# Hypertensive Emergency / Sodium Nitroprusside engine - sibling of
# SimEngine above, producing the SAME result-dict shape (same key
# names) so every UI update method (_update_cards, _update_charts,
# _update_risk_box, etc.) works unmodified regardless of which engine
# is currently active. Where a septic-shock concept has no equivalent
# here (SAPS_II, ShockIndex, VTBI/InfusionComplete, illness-recovery
# TimeInTargetMin, HR-advisory overlay), the key is still present with
# an honest placeholder rather than omitted, so shared code never needs
# a special case.
#
# "TargetBP" is reused to carry patient.map_floor_mmhg (the staged
# safety floor) rather than a fixed goal - the shared UI's own "MAP >=
# TargetBP is good" polarity happens to be exactly correct for both
# directions: septic-shock wants to RAISE MAP to reach/exceed a target,
# hypertensive-emergency wants to LOWER MAP but never drop below its
# floor - both are "MAP >= reference value is safe", so no inversion
# is needed anywhere in the shared display code.
# -----------------------------------------------------
class HypertensiveSimEngine:
    def __init__(self, clf, reg, risk_clf, pump):
        self.clf = clf
        self.reg = reg
        self.risk_clf = risk_clf

        self.patient = None
        self.history = []
        self.event_log = []
        self.full_event_log = []   # uncapped, for session-report export - see log_event()
        self.prev_measured_map = None
        self.running = False
        self.tick_accumulator = 0.0
        self.seed = 1
        self.map_window = deque(maxlen=WINDOW)
        self.hr_window = deque(maxlen=WINDOW)
        # Shared with SimEngine - see that class's __init__ comment for
        # why this must NOT be a separate PumpLink per engine.
        self.pump = pump

        self.max_flow_ml_hr = hc.MAX_FLOW_ML_HR
        self.condition_label = "Hypertensive (Emergency)"

    def flow_rate_to_mass_rate(self, flow_ml_hr):
        return ht_flow_rate_to_mass_rate(flow_ml_hr)

    def new_patient(self, profile_choice, seed, target_map=None):
        # target_map is accepted (not required) purely so callers can
        # pass the same three positional args regardless of which
        # engine is active - this mode has no user-settable target, it
        # comes entirely from the patient's own presenting MAP.
        profile = None if profile_choice == "Random" else profile_choice
        self.patient = VirtualHypertensivePatient(0, profile=profile, seed=seed)
        self.history = []
        self.event_log = []
        self.full_event_log = []
        self.prev_measured_map = None
        self.running = False
        self.tick_accumulator = 0.0
        self.map_window = deque(maxlen=WINDOW)
        self.hr_window = deque(maxlen=WINDOW)
        self.log_event(f"Patient created (profile={self.patient.profile})")

    def log_event(self, message):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"{ts}  {message}"
        self.event_log.append(entry)
        self.event_log = self.event_log[-MAX_EVENT_LOG:]
        self.full_event_log.append(entry)

    @staticmethod
    def build_feature_row(patient, measured_map, measured_hr, measured_spo2, prev_measured_map):
        bp_error = measured_map - patient.map_floor_mmhg
        bp_trend = measured_map - prev_measured_map
        cumulative_mcg_per_kg = patient.cumulative_nitroprusside_mcg / patient.weight_kg

        row = {
            "MeasuredMAP": measured_map, "MeasuredHR": measured_hr, "MeasuredSpO2": measured_spo2,
            "FlowRate": patient.flow_rate, "WeightKg": patient.weight_kg, "Age": patient.age,
            "ElapsedHours": patient.hours_since_start, "MAP_Floor": patient.map_floor_mmhg,
            "MeasuredBP_Error": bp_error, "MeasuredBP_Trend": bp_trend,
            "CumulativeMcgPerKg": cumulative_mcg_per_kg,
            "CyanideRateRisk": int(patient.cyanide_rate_risk_active),
            "CyanideBufferExceeded": int(patient.cyanide_buffer_exceeded),
        }
        return pd.DataFrame([row])[HT_FEATURE_COLUMNS], row

    @staticmethod
    def build_risk_feature_row(base_row, map_window, hr_window):
        risk_row = dict(base_row)
        risk_row["RollMeanMAP"] = float(np.mean(map_window))
        risk_row["RollStdMAP"] = float(np.std(map_window))
        risk_row["RollSlopeMAP"] = _rolling_slope(map_window)
        risk_row["RollMeanHR"] = float(np.mean(hr_window))
        risk_row["RollSlopeHR"] = _rolling_slope(hr_window)
        return pd.DataFrame([risk_row])[HT_RISK_FEATURE_COLUMNS]

    def step(self):
        """Advance the patient one tick. hypertensive_expert_controller
        is the sole actuator; both AI models are advisory only - exact
        same safety architecture as the septic-shock side."""
        patient = self.patient
        patient.update()
        expert_decision = hypertensive_expert_controller(patient)

        measured_map = float(np.clip(patient.map + np.random.normal(0, BP_NOISE_SD), 20, 250))
        measured_hr = float(np.clip(patient.hr + np.random.normal(0, HR_NOISE_SD), 30, 220))
        measured_spo2 = float(np.clip(patient.spo2 + np.random.normal(0, SPO2_NOISE_SD), 70, 100))

        prev_measured_map = self.prev_measured_map
        if prev_measured_map is None:
            prev_measured_map = measured_map
        X, base_row = self.build_feature_row(patient, measured_map, measured_hr, measured_spo2, prev_measured_map)
        self.prev_measured_map = measured_map

        self.map_window.append(measured_map)
        self.hr_window.append(measured_hr)

        risk_X = self.build_risk_feature_row(base_row, self.map_window, self.hr_window)
        risk_prob = float(self.risk_clf.predict_proba(risk_X)[0, 1])

        ai_action = self.clf.predict(X)[0]
        # The regressor predicts the CHANGE in flow, not the absolute
        # value (see train_hypertensive_ai.py for why) - reconstruct
        # the absolute suggestion the same way that script does. Clamp
        # to the physically valid range - right after a large real
        # correction (e.g. the controller's own Toxicity-Cap dropping
        # the rate hard), a correctly-predicted large negative delta
        # can otherwise reconstruct into a nonsensical negative
        # "suggested flow rate", which is what an extended real-usage
        # test actually caught (AI_Flow=-62.5 mL/hr at hour 4).
        ai_delta = float(self.reg.predict(X)[0])
        ai_flow = float(np.clip(patient.flow_rate + ai_delta, 0.0, self.max_flow_ml_hr))

        flow_disagreement_threshold = ht_mass_rate_to_flow_ml_hr(HT_MAX_STEP_MCG_KG_MIN * patient.weight_kg)
        comparison = evaluate_ai_recommendation(expert_decision, ai_action, ai_flow, flow_disagreement_threshold)

        # The pump only ever obeys the controller.
        patient.flow_rate = expert_decision["RecommendedFlow"]

        if self.pump.connected:
            self.pump.send_rate(patient.flow_rate)

        # See class docstring for why "MAP >= floor" is the correct
        # "safe" polarity here, matching the septic side's "MAP >=
        # target" exactly despite the opposite treatment direction.
        in_target = patient.map >= patient.map_floor_mmhg

        result = {
            "Time": len(self.history),
            "MAP": round(patient.map, 2),
            "MeasuredMAP": round(measured_map, 2),
            "TargetBP": round(patient.map_floor_mmhg, 2),
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
            # No equivalent scalar confidence/stability score is computed
            # by hypertensive_expert_controller.py - shown honestly as
            # "N/A" rather than fabricating a number, since the display
            # slot accepts any string.
            "StabilityScore": "N/A",
            "Confidence": "N/A",
            "ShockIndex": 0.0,  # not a hypertensive-mode concept; kept for dict-shape parity only
            "PlasmaConcentration": round(patient.plasma_concentration, 3),
            "DrugEffect": round(patient.drug_effect, 3),
            "SVR": round(patient.svr, 1),
            "CO": round(patient.co, 2),
            "RiskScore": risk_prob,
            "InfusionComplete": False,  # no VTBI concept for this mode
            "HRAdvisory": False,        # reflex tachycardia here is an EXPECTED
                                        # treatment side effect, not an unexplained
                                        # overlay worth flagging the way it is on
                                        # the septic side
            "TimeInTargetMin": 0.0,     # no illness-recovery/weaning dynamics
                                        # modeled for this condition yet
            "CyanideRateRisk": expert_decision["CyanideRateRisk"],
            "CyanideBufferExceeded": expert_decision["CyanideBufferExceeded"],
        }
        return result

    def detect_events(self, prev, latest):
        if prev is None:
            return
        if latest["ControllerMode"] != prev["ControllerMode"]:
            if latest["ControllerMode"] == "Emergency":
                self.log_event("Controller flagged malignant-range MAP for review")
            elif prev["ControllerMode"] == "Emergency":
                self.log_event("MAP back below malignant-range threshold")
            elif latest["ControllerMode"] == "Toxicity-Cap":
                self.log_event("CYANIDE SAFETY: rate capped per FDA rate-risk threshold")
            elif prev["ControllerMode"] == "Toxicity-Cap":
                self.log_event("Cyanide rate-risk cap lifted")
            elif latest["ControllerMode"] == "Toxicity-Stop":
                self.log_event("CYANIDE SAFETY: infusion stopped - total buffer capacity exceeded")
            else:
                self.log_event(f"Controller mode changed to {latest['ControllerMode']}")

        if latest["InTargetRange"] and not prev["InTargetRange"]:
            self.log_event("MAP recovered back above the staged safety floor")
        elif not latest["InTargetRange"] and prev["InTargetRange"]:
            self.log_event("MAP BREACHED below the staged safety floor")

        if not latest["ActionAgrees"] and prev["ActionAgrees"]:
            self.log_event(f"AI disagreed with controller ({latest['Alert']})")
        elif latest["ActionAgrees"] and not prev["ActionAgrees"]:
            self.log_event("AI agreement with controller restored")

        if latest["RiskScore"] >= RISK_ALERT_THRESHOLD > prev["RiskScore"]:
            self.log_event(f"AI RISK ALERT: {latest['RiskScore']*100:.0f}% probability of "
                            f"breaching the safety floor in next {LOOKAHEAD}s")
        elif prev["RiskScore"] >= RISK_ALERT_THRESHOLD > latest["RiskScore"]:
            self.log_event(f"AI risk alert cleared ({latest['RiskScore']*100:.0f}%)")


# -----------------------------------------------------
# UI building blocks
# -----------------------------------------------------
class VitalCard(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)

        self.title_label = QLabel(title.upper())
        self.title_label.setObjectName("cardTitle")
        self.value_label = QLabel("--")
        self.value_label.setObjectName("cardValue")
        self.status_label = QLabel("")
        self.status_label.setObjectName("cardStatus")

        for w in (self.title_label, self.value_label, self.status_label):
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.status_label)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def update_value(self, value, status_label, color):
        self.value_label.setText(str(value))
        self.status_label.setText(f"● {status_label}")
        self.status_label.setStyleSheet(f"color: {color};")
        # Border highlight so a genuinely critical vital is visible at a
        # glance, not just from the small status-dot color. RED (truly
        # dangerous) only - YELLOW covers routine/non-alarming states too
        # (e.g. mode_status()'s "Increase"/"Decrease" for perfectly
        # normal titration), so bordering on YELLOW would highlight
        # ordinary operation as if it were a problem. GREEN/GREY/YELLOW
        # fall back to the app-wide #card stylesheet rule instead of an
        # inline override.
        if color == RED:
            self.setStyleSheet(
                f"QFrame#card {{ background-color: {PANEL}; "
                f"border: 2px solid {RED}; border-radius: 10px; }}"
            )
        else:
            self.setStyleSheet("")


class MetricBox(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setObjectName("metric")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricTitle")
        self.value_label = QLabel("--")
        self.value_label.setObjectName("metricValue")
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, text, color=None):
        self.value_label.setText(text)
        if color:
            self.value_label.setStyleSheet(f"color: {color};")


def make_plot(title, y_label, y_range=None):
    pw = pg.PlotWidget(title=title)
    pw.setBackground(PANEL)
    pw.showGrid(x=True, y=True, alpha=0.15)
    pw.setLabel("left", y_label)
    pw.setLabel("bottom", "Time (s)")
    pw.getAxis("left").setPen(pg.mkPen(TEXT_DIM))
    pw.getAxis("bottom").setPen(pg.mkPen(TEXT_DIM))
    pw.addLegend(offset=(10, 5))
    if y_range is not None:
        pw.setYRange(*y_range, padding=0)
    pw.setMinimumHeight(220)
    return pw


def wrap_scrollable(widget):
    """Wrap a tab's content widget in a QScrollArea so content that
    doesn't fit the window height scrolls instead of being crushed by
    the layout to whatever space remains."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(widget)
    return scroll


# -----------------------------------------------------
# Main window
# -----------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart AI Infusion Pump — Digital Twin Monitor")
        # [FIX] Was a flat self.resize(1500, 950) - on a screen narrower
        # than 1500px (e.g. a 1366-wide laptop display), that put the
        # window's own right edge off-screen, cutting off whatever text
        # sat near it (the EMERGENCY banner, the Clinical Alert box) -
        # not a text-wrapping bug, the window itself didn't fit. Clamp
        # to the actual available screen size so the whole window - and
        # everything inside it - is always fully visible.
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1500, screen.width() - 40), min(950, screen.height() - 40))

        # ONE shared physical pump connection for both engines - there is
        # only one real ESP32/USB connection regardless of which
        # simulated condition is active. Previously each engine created
        # its own separate PumpLink, so connecting while in one mode did
        # nothing once switched to the other mode (see each engine's
        # __init__ comment).
        shared_pump = PumpLink()

        # [ADDED 2026-10-10] Fire-and-forget UDP bridge to
        # clinician_dashboard.py (a separate process - see its own
        # module docstring). This app is the single source of truth
        # for live vitals; the dashboard never talks to the pump or
        # this app directly, it only ever receives whatever gets sent
        # here, once per tick, in _on_timer(). A plain UDP datagram to
        # localhost essentially never blocks and needs no new thread -
        # if clinician_dashboard.py isn't even running, these sends
        # just go nowhere, silently, which is the correct behavior
        # (this feature is fully optional or a plain-simulation session
        # must not change at all when it's absent).
        self._dashboard_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._dashboard_addr = ("127.0.0.1", 9999)

        # [ADDED 2026-10-10] Adafruit IO - optional, off by default (see
        # ADAFRUIT_IO_ENABLED's own comment). Client construction itself
        # doesn't make a network call, so doing it here is safe/cheap;
        # actual publishing is throttled and backgrounded - see
        # _maybe_publish_to_adafruit().
        self._adafruit_client = None
        if ADAFRUIT_IO_ENABLED and AdafruitIOClient is not None:
            try:
                self._adafruit_client = AdafruitIOClient(ADAFRUIT_IO_USERNAME, ADAFRUIT_IO_KEY)
            except Exception as exc:
                print(f"Adafruit IO client init failed: {exc}")
                self._adafruit_client = None
        self._last_adafruit_publish_time = 0.0

        # Remote dashboard publisher - construction never does network
        # I/O (see telemetry_publisher.py's own class docstring), so
        # this is safe/cheap to always create; REMOTE_DASHBOARD_ENABLED
        # (and a non-empty device password) just controls whether it's
        # ever actually sent anywhere.
        self._remote_dashboard = None
        self._remote_dashboard_session_id = None
        self._last_remote_dashboard_publish_time = 0.0
        self._last_pump_status = None   # most recent parsed GET_STATUS reply, if any
        self._active_alert_titles = set()   # clinical_alerts state, for onset/clear event detection
        if RemoteTelemetryPublisher is not None:
            try:
                self._remote_dashboard = RemoteTelemetryPublisher(
                    api_key=REMOTE_DASHBOARD_FIREBASE_API_KEY,
                    project_id=REMOTE_DASHBOARD_FIREBASE_PROJECT_ID,
                    device_id=REMOTE_DASHBOARD_DEVICE_ID,
                    device_email=REMOTE_DASHBOARD_DEVICE_EMAIL,
                    device_password=REMOTE_DASHBOARD_DEVICE_PASSWORD,
                    database=REMOTE_DASHBOARD_FIRESTORE_DATABASE,
                    enabled=REMOTE_DASHBOARD_ENABLED and bool(REMOTE_DASHBOARD_DEVICE_PASSWORD),
                )
            except Exception as exc:
                print(f"Remote dashboard publisher initialization failed: {exc}")
                self._remote_dashboard = None

        clf = load_model_for_live_prediction("models/ai_action_classifier.pkl")
        reg = load_model_for_live_prediction("models/ai_flow_regressor.pkl")
        risk_clf = load_model_for_live_prediction("models/risk_classifier.pkl")
        self.septic_engine = SimEngine(clf, reg, risk_clf, shared_pump)

        ht_clf = load_model_for_live_prediction("models/hypertensive_action_classifier.pkl")
        ht_reg = load_model_for_live_prediction("models/hypertensive_flow_regressor.pkl")
        ht_risk_clf = load_model_for_live_prediction("models/hypertensive_risk_classifier.pkl")
        self.hypertensive_engine = HypertensiveSimEngine(ht_clf, ht_reg, ht_risk_clf, shared_pump)

        # Both engines are loaded up front (not lazily on first switch)
        # so toggling the condition selector is instant - no model-load
        # pause mid-session. Only one is ever "active" (self.engine) and
        # ticking at a time; the other just sits idle holding its models.
        self.engine = self.septic_engine

        self._build_ui()

        self.beep_timer = QTimer(self)
        self.beep_timer.timeout.connect(self._on_beep_tick)
        self._beep_freq, self._beep_dur = 880, 50

        self.engine.new_patient("Random", self.engine.seed, TARGET_MAP_MMHG)
        self.refresh_all()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(TICK_TIMER_MS)

    # ---------------- UI construction ----------------
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setCentralWidget(central)

        sidebar_scroll = wrap_scrollable(self._build_sidebar())
        sidebar_scroll.setFixedWidth(280)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setObjectName("sidebarScroll")
        root.addWidget(sidebar_scroll)
        root.addWidget(self._build_main_area(), stretch=1)

        self.setStyleSheet(self._qss())

    def _build_sidebar(self):
        panel = QFrame()
        panel.setObjectName("sidebar")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        title = QLabel("CONTROLS")
        title.setObjectName("sidebarTitle")
        layout.addWidget(title)

        # Patient Condition - which clinical mode/twin/controller/AI
        # models are active. Switching rebuilds the Clinical Profile
        # options below and enables/disables the Target MAP control,
        # then starts a fresh matching patient.
        condition_box = QGroupBox("Patient Condition")
        cbl = QVBoxLayout(condition_box)
        self.condition_group = QButtonGroup(self)
        rb_septic = QRadioButton("Hypotensive (Septic Shock)")
        rb_septic.setChecked(True)
        rb_hyper = QRadioButton("Hypertensive (Emergency)")
        self.condition_group.addButton(rb_septic, 0)
        self.condition_group.addButton(rb_hyper, 1)
        cbl.addWidget(rb_septic)
        cbl.addWidget(rb_hyper)
        layout.addWidget(condition_box)
        self.condition_group.idClicked.connect(self._on_condition_changed)

        # Profile - options depend on the active condition, so this is
        # rebuilt by _rebuild_profile_options() rather than populated
        # once here. profile_box/pbl are kept as instance attributes so
        # that method can clear and repopulate them.
        self.profile_box = QGroupBox("Clinical Profile")
        self.profile_layout = QVBoxLayout(self.profile_box)
        self.profile_group = QButtonGroup(self)
        layout.addWidget(self.profile_box)
        self._rebuild_profile_options(PROFILE_WEIGHTS.keys())

        # Target MAP - only meaningful for the septic-shock side (a
        # clinician-adjustable goal). The hypertensive side has no
        # user-settable target - it's the staged safety floor, derived
        # automatically from each patient's own presenting MAP - so
        # this control is simply disabled (not hidden, to avoid layout
        # jumps) whenever that condition is active.
        self.target_box = QGroupBox("Target MAP (mmHg)")
        tbl = QVBoxLayout(self.target_box)
        target_lo, target_hi = c.TARGET_MAP_RANGE_MMHG
        self.target_spin = QDoubleSpinBox()
        self.target_spin.setRange(target_lo, target_hi)
        self.target_spin.setValue(TARGET_MAP_MMHG)
        self.target_spin.setSingleStep(1.0)
        self.target_spin.setDecimals(0)
        tbl.addWidget(self.target_spin)
        self.target_hint = QLabel(f"SEPSISPAM-studied range ({target_lo:.0f}–{target_hi:.0f} mmHg).\n"
                                   "Applied on next new/reset patient.")
        self.target_hint.setObjectName("hint")
        self.target_hint.setWordWrap(True)
        tbl.addWidget(self.target_hint)
        layout.addWidget(self.target_box)

        # Speed
        speed_box = QGroupBox("Simulation Speed")
        sbl = QHBoxLayout(speed_box)
        self.speed_group = QButtonGroup(self)
        for i, name in enumerate(SPEED_MAP.keys()):
            rb = QRadioButton(name)
            if name == "1x":
                rb.setChecked(True)
            self.speed_group.addButton(rb, i)
            sbl.addWidget(rb)
        layout.addWidget(speed_box)

        # Physical pump link - optional. When connected, the exact same
        # controller-computed rate that drives the on-screen charts is
        # also streamed to the real ESP32 pump over Serial OR WiFi,
        # making this hardware-in-the-loop instead of pure simulation.
        pump_box = QGroupBox("Physical Pump")
        pbl2 = QVBoxLayout(pump_box)

        transport_row = QHBoxLayout()
        self.pump_transport_group = QButtonGroup(self)
        self.pump_wired_radio = QRadioButton("Wired")
        self.pump_wireless_radio = QRadioButton("Wireless")
        self.pump_wired_radio.setChecked(True)
        self.pump_transport_group.addButton(self.pump_wired_radio, 0)
        self.pump_transport_group.addButton(self.pump_wireless_radio, 1)
        transport_row.addWidget(self.pump_wired_radio)
        transport_row.addWidget(self.pump_wireless_radio)
        pbl2.addLayout(transport_row)

        port_row = QHBoxLayout()
        self.pump_port_combo = QComboBox()
        self.pump_refresh_btn = QPushButton("⟳")
        self.pump_refresh_btn.setFixedWidth(32)
        port_row.addWidget(self.pump_port_combo, stretch=1)
        port_row.addWidget(self.pump_refresh_btn)
        pbl2.addLayout(port_row)

        self.pump_host_edit = QLineEdit()
        self.pump_host_edit.setPlaceholderText("pump IP:port, e.g. 192.168.1.50:5000")
        self.pump_host_edit.setVisible(False)
        pbl2.addWidget(self.pump_host_edit)

        self.pump_connect_btn = QPushButton("Connect")
        pbl2.addWidget(self.pump_connect_btn)
        self.pump_status_label = QLabel("Disconnected — running in pure simulation")
        self.pump_status_label.setObjectName("hint")
        self.pump_status_label.setWordWrap(True)
        pbl2.addWidget(self.pump_status_label)
        layout.addWidget(pump_box)

        # Manual dosing sent DIRECTLY to the physical pump's
        # SET_WEIGHT_DOSE:/SET_VTBI_PROGRAM: commands (pump_firmware.ino)
        # - deliberately separate from the simulated VirtualPatient's own
        # weight/age fields elsewhere in this UI, which only ever affect
        # the on-screen digital twin, never the real motor. Only
        # meaningful once actually connected (wired or wireless) -
        # disabled otherwise.
        dose_box = QGroupBox("Manual Pump Dosing (wired or wireless)")
        dbl = QVBoxLayout(dose_box)

        dbl.addWidget(QLabel("Weight-based dose:"))
        wd_row = QHBoxLayout()
        self.wd_weight_spin = QDoubleSpinBox()
        self.wd_weight_spin.setRange(0.1, 300.0)
        self.wd_weight_spin.setValue(70.0)
        self.wd_weight_spin.setSuffix(" kg")
        self.wd_dose_spin = QDoubleSpinBox()
        self.wd_dose_spin.setRange(0.0, 500.0)
        self.wd_dose_spin.setDecimals(3)
        self.wd_dose_spin.setValue(0.05)
        self.wd_unit_combo = QComboBox()
        self.wd_unit_combo.addItems(["UGKGMIN", "MLHR"])
        self.wd_apply_btn = QPushButton("Apply")
        wd_row.addWidget(self.wd_weight_spin)
        wd_row.addWidget(self.wd_dose_spin)
        wd_row.addWidget(self.wd_unit_combo)
        wd_row.addWidget(self.wd_apply_btn)
        dbl.addLayout(wd_row)

        dbl.addWidget(QLabel("Rate + VTBI:"))
        vtbi_row = QHBoxLayout()
        self.vtbi_rate_spin = QDoubleSpinBox()
        self.vtbi_rate_spin.setRange(0.0, 190.0)
        self.vtbi_rate_spin.setValue(50.0)
        self.vtbi_rate_spin.setSuffix(" mL/hr")
        self.vtbi_volume_spin = QDoubleSpinBox()
        self.vtbi_volume_spin.setRange(0.1, 2000.0)
        self.vtbi_volume_spin.setValue(100.0)
        self.vtbi_volume_spin.setSuffix(" mL")
        self.vtbi_apply_btn = QPushButton("Apply")
        vtbi_row.addWidget(self.vtbi_rate_spin)
        vtbi_row.addWidget(self.vtbi_volume_spin)
        vtbi_row.addWidget(self.vtbi_apply_btn)
        dbl.addLayout(vtbi_row)

        dose_box.setEnabled(False)
        self.pump_dose_box = dose_box
        layout.addWidget(dose_box)

        self.pump_refresh_btn.clicked.connect(self._refresh_pump_ports)
        self.pump_connect_btn.clicked.connect(self._on_pump_connect_toggle)
        self.pump_wired_radio.toggled.connect(self._on_pump_transport_changed)
        self.wd_apply_btn.clicked.connect(self._on_weight_dose_apply)
        self.vtbi_apply_btn.clicked.connect(self._on_vtbi_apply)
        self._refresh_pump_ports()
        self._on_pump_transport_changed()

        # Buttons
        btn_row1 = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("primaryBtn")
        self.pause_btn = QPushButton("Pause")
        btn_row1.addWidget(self.start_btn)
        btn_row1.addWidget(self.pause_btn)
        layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.reset_btn = QPushButton("Reset Patient")
        self.new_btn = QPushButton("New Random Patient")
        btn_row2.addWidget(self.reset_btn)
        btn_row2.addWidget(self.new_btn)
        layout.addLayout(btn_row2)

        self.start_btn.clicked.connect(self._on_start)
        self.pause_btn.clicked.connect(self._on_pause)
        self.reset_btn.clicked.connect(self._on_reset)
        self.new_btn.clicked.connect(self._on_new_random)

        self.save_report_btn = QPushButton("Save Session Report (.txt)")
        self.save_report_btn.clicked.connect(self._on_save_session_report)
        layout.addWidget(self.save_report_btn)

        self.sound_checkbox = QCheckBox("Monitor beep (HR-paced)")
        self.sound_checkbox.setChecked(True)
        layout.addWidget(self.sound_checkbox)

        layout.addStretch(1)

        caption = QLabel(
            "The controller always drives the pump and brings MAP to "
            "target. The AI only proposes a parallel recommendation from "
            "noisy sensor readings — it never actuates the pump directly."
        )
        caption.setObjectName("hint")
        caption.setWordWrap(True)
        layout.addWidget(caption)

        return panel

    def _build_main_area(self):
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(16, 12, 16, 12)
        v.setSpacing(10)

        header = QVBoxLayout()
        title_row = QHBoxLayout()
        title = QLabel("Smart AI Infusion Pump — Digital Twin")
        title.setObjectName("mainTitle")
        disclaimer = QLabel("RESEARCH / SIMULATION PROTOTYPE — NOT FOR CLINICAL USE")
        disclaimer.setObjectName("disclaimerBadge")
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(disclaimer)
        self.status_caption = QLabel("")
        self.status_caption.setObjectName("hint")
        header.addLayout(title_row)
        header.addWidget(self.status_caption)
        v.addLayout(header)

        self.tabs = QTabWidget()
        v.addWidget(self.tabs, stretch=1)

        self.tabs.addTab(wrap_scrollable(self._build_monitor_tab()), "Live Monitor")
        self.tabs.addTab(wrap_scrollable(self._build_trends_tab()), "Decisions && Trends")

        return wrap

    def _build_monitor_tab(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setSpacing(12)

        # ICU cards
        cards_row = QHBoxLayout()
        cards_row.setSpacing(10)
        self.card_map = VitalCard("MAP")
        self.card_hr = VitalCard("HR")
        self.card_spo2 = VitalCard("SpO2")
        self.card_flow = VitalCard("Flow Rate")
        self.card_ai_mode = VitalCard("AI Mode")
        self.card_ctrl_mode = VitalCard("Controller")
        for card in (self.card_map, self.card_hr, self.card_spo2,
                     self.card_flow, self.card_ai_mode, self.card_ctrl_mode):
            cards_row.addWidget(card)
        v.addLayout(cards_row)

        # Target status strip
        strip = QHBoxLayout()
        strip.setSpacing(10)
        self.target_status_label = QLabel("")
        self.target_status_label.setObjectName("targetStrip")
        self.target_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # [FIX] Without word-wrap, this label's full single-line
        # EMERGENCY/status sentence forced its own natural (very wide)
        # text width as a hard minimum - which in turn forced the whole
        # scrollable page wider than the viewport, so everything to the
        # right (including unrelated widgets like the Clinical Alert
        # box, which already wraps on its own) ended up needing a
        # horizontal scroll to see, rather than actually overflowing on
        # its own.
        self.target_status_label.setWordWrap(True)
        self.metric_pct_target = MetricBox("Time in target so far")
        self.metric_first_reached = MetricBox("First reached target at")
        strip.addWidget(self.target_status_label, stretch=2)
        strip.addWidget(self.metric_pct_target, stretch=1)
        strip.addWidget(self.metric_first_reached, stretch=1)
        v.addLayout(strip)

        # Clinical Alert panel - structured advisories from
        # clinical_alerts.py (severity + consequence + recommendation
        # per abnormal vital). Hidden entirely when there is nothing to
        # show, same "don't occupy space with an empty/green state" look
        # as the rest of this screen.
        self.clinical_alert_box = QFrame()
        self.clinical_alert_box.setObjectName("panel")
        cab_layout = QVBoxLayout(self.clinical_alert_box)
        self.clinical_alert_title = QLabel("")
        self.clinical_alert_title.setObjectName("panelTitle")
        self.clinical_alert_body = QLabel("")
        self.clinical_alert_body.setObjectName("clinicalAlertBody")
        self.clinical_alert_body.setWordWrap(True)
        self.clinical_alert_body.setTextFormat(Qt.TextFormat.RichText)
        cab_layout.addWidget(self.clinical_alert_title)
        cab_layout.addWidget(self.clinical_alert_body)
        self.clinical_alert_box.setVisible(False)
        v.addWidget(self.clinical_alert_box)

        # Risk monitor
        risk_box = QFrame()
        risk_box.setObjectName("panel")
        rbl = QHBoxLayout(risk_box)
        risk_title_col = QVBoxLayout()
        risk_title = QLabel("AI Risk Monitor — independent early-warning model")
        risk_title.setObjectName("panelTitle")
        self.risk_caption = QLabel(self._risk_caption_text())
        self.risk_caption.setObjectName("hint")
        self.risk_caption.setWordWrap(True)
        risk_title_col.addWidget(risk_title)
        risk_title_col.addWidget(self.risk_caption)
        self.risk_value_label = QLabel("--%")
        self.risk_value_label.setObjectName("riskValue")
        self.risk_status_label = QLabel("")
        self.risk_status_label.setObjectName("riskStatus")
        risk_val_col = QVBoxLayout()
        risk_val_col.addWidget(self.risk_value_label)
        risk_val_col.addWidget(self.risk_status_label)
        rbl.addLayout(risk_title_col, stretch=3)
        rbl.addLayout(risk_val_col, stretch=1)
        v.addWidget(risk_box)

        # Charts grid
        grid = QGridLayout()
        grid.setSpacing(10)

        self.plot_map = make_plot("Mean Arterial Pressure (MAP)", "mmHg")
        self.curve_map_true = self.plot_map.plot(pen=pg.mkPen("#60a5fa", width=2), name="True MAP")
        self.curve_map_meas = self.plot_map.plot(
            pen=pg.mkPen("#fb923c", width=1, style=Qt.PenStyle.DashLine), name="Measured MAP")
        self.target_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#e5e7eb", style=Qt.PenStyle.DotLine))
        self.target_band = pg.LinearRegionItem(
            values=(0, 0), orientation="horizontal", movable=False,
            brush=pg.mkBrush(34, 197, 94, 40), pen=pg.mkPen(None))
        self.target_band.setZValue(-10)
        self.plot_map.addItem(self.target_band)
        self.plot_map.addItem(self.target_line)

        self.plot_hr = make_plot("Heart Rate", "bpm")
        self.curve_hr = self.plot_hr.plot(pen=pg.mkPen("#60a5fa", width=2), name="HR")

        self.plot_spo2 = make_plot("Oxygen Saturation", "%")
        self.curve_spo2 = self.plot_spo2.plot(pen=pg.mkPen("#34d399", width=2), name="SpO2")

        self.plot_flow = make_plot("Infusion Flow Rate", "mL/hr")
        self.curve_flow_ctrl = self.plot_flow.plot(pen=pg.mkPen("#60a5fa", width=2), name="Controller")
        self.curve_flow_ai = self.plot_flow.plot(
            pen=pg.mkPen("#f59e0b", width=1, style=Qt.PenStyle.DashLine), name="AI Suggested")

        self.plot_plasma = make_plot("Plasma Norepinephrine Concentration (Cp)", "mcg/L")
        self.curve_plasma = self.plot_plasma.plot(pen=pg.mkPen("#a78bfa", width=2), name="Cp")

        self.plot_risk = make_plot("Deterioration Risk Score", "Probability", y_range=(0, 1))
        self.curve_risk = self.plot_risk.plot(pen=pg.mkPen("#ef4444", width=2), name="Risk")

        grid.addWidget(self.plot_map, 0, 0, 1, 2)
        grid.addWidget(self.plot_hr, 1, 0)
        grid.addWidget(self.plot_spo2, 1, 1)
        grid.addWidget(self.plot_flow, 2, 0)
        grid.addWidget(self.plot_plasma, 2, 1)
        grid.addWidget(self.plot_risk, 3, 0, 1, 2)

        v.addLayout(grid, stretch=1)
        return page

    def _build_trends_tab(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setSpacing(10)

        panels_row = QHBoxLayout()
        panels_row.setSpacing(10)

        self.ai_panel = self._make_text_panel("AI (advisory)")
        self.ctrl_panel = self._make_text_panel("Controller (sole actuator)")
        self.hemo_panel = self._make_text_panel("Patient && Hemodynamics (hidden from AI)")

        panels_row.addWidget(self.ai_panel["frame"])
        panels_row.addWidget(self.ctrl_panel["frame"])
        panels_row.addWidget(self.hemo_panel["frame"])
        v.addLayout(panels_row)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)

        log_frame = QFrame()
        log_frame.setObjectName("panel")
        log_layout = QVBoxLayout(log_frame)
        log_title = QLabel("Event Log")
        log_title.setObjectName("panelTitle")
        self.event_log_view = QTextEdit()
        self.event_log_view.setReadOnly(True)
        self.event_log_view.setObjectName("eventLog")
        log_layout.addWidget(log_title)
        log_layout.addWidget(self.event_log_view)

        table_frame = QFrame()
        table_frame.setObjectName("panel")
        table_layout = QVBoxLayout(table_frame)
        table_title = QLabel("Recent History")
        table_title.setObjectName("panelTitle")
        self.history_table = QTableWidget()
        columns = ["Time", "MAP", "MeasMAP", "HR", "SpO2",
                   "Ctrl Mode", "Ctrl Flow", "AI Action", "AI Flow", "Risk"]
        self.history_table.setColumnCount(len(columns))
        self.history_table.setHorizontalHeaderLabels(columns)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.verticalHeader().setVisible(False)
        table_layout.addWidget(table_title)
        table_layout.addWidget(self.history_table)

        bottom_row.addWidget(log_frame, stretch=1)
        bottom_row.addWidget(table_frame, stretch=2)
        v.addLayout(bottom_row, stretch=1)

        return page

    def _make_text_panel(self, title):
        frame = QFrame()
        frame.setObjectName("panel")
        layout = QVBoxLayout(frame)
        title_label = QLabel(title)
        title_label.setObjectName("panelTitle")
        body = QLabel("")
        body.setObjectName("panelBody")
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(title_label)
        layout.addWidget(body, stretch=1)
        return {"frame": frame, "body": body}

    # ---------------- Button handlers ----------------
    def _current_profile_choice(self):
        return self.profile_group.checkedButton().text()

    def _current_speed(self):
        return SPEED_MAP[self.speed_group.checkedButton().text()]

    def _rebuild_profile_options(self, profile_names):
        """Clears and repopulates the Clinical Profile radio group -
        called on startup and every time the Patient Condition changes,
        since the two modes have entirely different profile sets
        (Healthy/Mild/Moderate/Severe vs. Healthy/Stage2/Urgency/
        Emergency)."""
        for btn in self.profile_group.buttons():
            self.profile_group.removeButton(btn)
        while self.profile_layout.count():
            item = self.profile_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for i, name in enumerate(["Random"] + list(profile_names)):
            rb = QRadioButton(name)
            if name == "Random":
                rb.setChecked(True)
            self.profile_group.addButton(rb, i)
            self.profile_layout.addWidget(rb)

    def _risk_caption_text(self):
        # The risk model's actual target label is condition-specific
        # (WillDeteriorate vs WillBreachFloor - see generate_dataset.py
        # vs generate_hypertensive_dataset.py) - this caption must match
        # whichever engine is currently active, not be hardcoded to the
        # septic-shock definition. Missing this exact mismatch is what
        # an examiner comparing the live dashboard against a methodology
        # section's risk-label definition (3.9) would catch immediately.
        if self.engine is self.septic_engine:
            return (
                f"Probability of deterioration (MAP<{EMERGENCY_MAP_MMHG:.0f} or SpO2<90) in the "
                f"next {LOOKAHEAD}s, from rolling trend history. The controller has no equivalent forecast."
            )
        return (
            f"Probability of breaching the staged safety floor, or triggering the cyanide "
            f"rate-risk threshold, in the next {LOOKAHEAD}s, from rolling trend history. "
            f"The controller has no equivalent forecast."
        )

    def _on_condition_changed(self, checked_id):
        self.engine.pump.send("STOP")
        self.engine.running = False
        self.beep_timer.stop()

        if checked_id == 0:
            self.engine = self.septic_engine
            self._rebuild_profile_options(PROFILE_WEIGHTS.keys())
            self.target_spin.setEnabled(True)
            target_lo, target_hi = c.TARGET_MAP_RANGE_MMHG
            self.target_hint.setText(f"SEPSISPAM-studied range ({target_lo:.0f}–{target_hi:.0f} mmHg).\n"
                                      "Applied on next new/reset patient.")
        else:
            self.engine = self.hypertensive_engine
            self._rebuild_profile_options(HT_PROFILE_WEIGHTS.keys())
            self.target_spin.setEnabled(False)
            self.target_hint.setText(
                "Not user-adjustable in this mode - the safe target is a "
                "staged floor computed automatically from each patient's "
                "own presenting MAP (see Hemodynamics panel)."
            )

        self.risk_caption.setText(self._risk_caption_text())

        self.engine.seed = random.randint(0, 1_000_000)
        self.engine.new_patient("Random", self.engine.seed, self.target_spin.value())
        self.refresh_all()

    def _start_remote_dashboard_session(self):
        """[ADDED 2026-10-20] Per the handoff doc's "Where the calls
        go" table: a new patient is a new monitoring session. Ends
        whatever session was previously open first, so New Random
        Patient mid-run doesn't silently orphan the old one in History."""
        self._end_remote_dashboard_session()
        self._remote_dashboard_session_id = str(uuid.uuid4())

    def _end_remote_dashboard_session(self):
        if self._remote_dashboard is not None and self._remote_dashboard_session_id is not None:
            try:
                self._remote_dashboard.end_session(self._remote_dashboard_session_id)
            except Exception as exc:
                print(f"[remote_dashboard] end_session failed: {exc}")
        self._remote_dashboard_session_id = None

    def _on_start(self):
        self.engine.running = True
        self.engine.log_event("Simulation started")

    def _on_pause(self):
        self.engine.running = False
        self.engine.log_event("Simulation paused")
        self.beep_timer.stop()
        # The sim stops updating flow_rate, but without this the real
        # motor would just keep running at its last commanded rate
        # until the pump's own 3s watchdog eventually caught it.
        self.engine.pump.send("STOP")
        self._end_remote_dashboard_session()

    def _on_reset(self):
        self.engine.pump.send("STOP")
        self.engine.new_patient(self._current_profile_choice(), self.engine.seed, self.target_spin.value())
        self.beep_timer.stop()
        self._active_alert_titles = set()
        self.refresh_all()
        self._start_remote_dashboard_session()

    def _on_new_random(self):
        self.engine.pump.send("STOP")
        self.engine.seed = random.randint(0, 1_000_000)
        self.engine.new_patient(self._current_profile_choice(), self.engine.seed, self.target_spin.value())
        self.beep_timer.stop()
        self._active_alert_titles = set()
        self.refresh_all()
        self._start_remote_dashboard_session()

    def _on_save_session_report(self):
        """Writes a plain-text record of the current run - setup, final
        state, and the FULL event log (not just the last 30 shown on
        screen). There is no other persistent record of a session
        anywhere in this app otherwise - closing the window loses
        everything, which is a real gap for anyone wanting to document
        an actual test run."""
        if not self.engine.history:
            self.engine.log_event("Save Session Report: nothing to save yet - start a run first.")
            self.refresh_all()
            QMessageBox.warning(
                self, "Nothing to save",
                "No simulation data yet - click Start (or New Random Patient) "
                "and let it run for at least a tick before saving a report."
            )
            return

        latest = self.engine.history[-1]
        p = self.engine.patient

        out_dir = writable_output_dir("session_logs")
        ts_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"session_{ts_filename}.txt")

        ref_label = "Target" if self.engine is self.septic_engine else "Floor"

        lines = [
            "Smart AI Infusion Pump - Session Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Condition: {self.engine.condition_label}",
            f"Profile: {p.profile}  |  Seed: {self.engine.seed}",
            f"Age / Weight: {p.age}y / {p.weight_kg:.1f} kg",
            f"Duration: {len(self.engine.history)} ticks "
            f"({len(self.engine.history)} simulated seconds)",
            "",
            "--- Final State ---",
            f"MAP: {latest['MAP']} mmHg  (Measured: {latest['MeasuredMAP']} mmHg)",
            f"HR: {latest['HR']} bpm",
            f"SpO2: {latest['SpO2']} %",
            f"{ref_label}: {latest['TargetBP']} mmHg",
            f"Controller Mode: {latest['ControllerMode']}  |  Flow: {latest['ControllerFlow']} mL/hr",
            f"AI Action: {latest['AI_Action']}  |  AI Suggested Flow: {latest['AI_SuggestedFlow']} mL/hr",
            f"Risk Score: {latest['RiskScore']*100:.0f}%",
            "",
            f"--- Full Event Log ({len(self.engine.full_event_log)} entries) ---",
        ]
        lines.extend(self.engine.full_event_log or ["(no events recorded)"])

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except OSError as e:
            # A silent failure here (permissions, read-only install
            # location, etc.) would look identical to "it saved
            # somewhere I can't find" from the user's side - surface it
            # loudly instead of leaving that ambiguous.
            self.engine.log_event(f"Session report FAILED to save: {e}")
            self.refresh_all()
            QMessageBox.critical(
                self, "Save failed",
                f"Could not write the session report:\n\n{e}\n\n"
                f"Attempted path:\n{path}"
            )
            return

        self.engine.log_event(f"Session report saved: {path}")
        self.refresh_all()
        QMessageBox.information(
            self, "Session report saved",
            f"Saved to:\n\n{path}"
        )

    # ---------------- Physical pump link ----------------
    def _refresh_pump_ports(self):
        current = self.pump_port_combo.currentText()
        self.pump_port_combo.clear()
        ports = PumpLink.available_ports()
        self.pump_port_combo.addItems(ports)
        if current in ports:
            self.pump_port_combo.setCurrentText(current)

    def _on_pump_transport_changed(self):
        wireless = self.pump_wireless_radio.isChecked()
        self.pump_port_combo.setVisible(not wireless)
        self.pump_refresh_btn.setVisible(not wireless)
        self.pump_host_edit.setVisible(wireless)

    def _on_pump_connect_toggle(self):
        if self.engine.pump.connected:
            self.engine.pump.disconnect()
            self.pump_connect_btn.setText("Connect")
            self.pump_status_label.setText("Disconnected — running in pure simulation")
            self.pump_status_label.setStyleSheet(f"color: {TEXT_DIM};")
            self.pump_dose_box.setEnabled(False)
            self.engine.log_event("Physical pump disconnected")
            return

        if self.pump_wireless_radio.isChecked():

            target = self.pump_host_edit.text().strip()
            if not target:
                self.pump_status_label.setText("Enter the pump's IP:port, e.g. 192.168.1.50:5000")
                self.pump_status_label.setStyleSheet(f"color: {YELLOW};")
                return

            host, _, port_str = target.partition(":")
            try:
                port = int(port_str) if port_str else 5000
            except ValueError:
                self.pump_status_label.setText(f"Invalid port in '{target}'")
                self.pump_status_label.setStyleSheet(f"color: {RED};")
                return

            try:
                self.engine.pump.connect_wifi(host, port)
            except Exception as exc:
                self.pump_status_label.setText(f"Connect failed: {exc}")
                self.pump_status_label.setStyleSheet(f"color: {RED};")
                return

            self.pump_connect_btn.setText("Disconnect")
            self.pump_status_label.setText(f"LIVE (wireless) — driving the real pump on {host}:{port}")
            self.pump_status_label.setStyleSheet(f"color: {RED}; font-weight: 700;")
            self.pump_dose_box.setEnabled(True)
            self.engine.log_event(
                f"Physical pump connected wirelessly on {host}:{port} — motor will actually move"
            )
            return

        port = self.pump_port_combo.currentText()
        if not port:
            self.pump_status_label.setText("No serial port found — plug in the pump and hit ⟳")
            self.pump_status_label.setStyleSheet(f"color: {YELLOW};")
            return

        try:
            self.engine.pump.connect(port)
        except Exception as exc:
            self.pump_status_label.setText(f"Connect failed: {exc}")
            self.pump_status_label.setStyleSheet(f"color: {RED};")
            return

        self.pump_connect_btn.setText("Disconnect")
        self.pump_status_label.setText(f"LIVE — driving the real pump on {port}")
        self.pump_status_label.setStyleSheet(f"color: {RED}; font-weight: 700;")
        self.pump_dose_box.setEnabled(True)
        self.engine.log_event(f"Physical pump connected on {port} — motor will actually move")

    def _on_weight_dose_apply(self):
        if not self.engine.pump.connected:
            return
        weight_kg = self.wd_weight_spin.value()
        dose_value = self.wd_dose_spin.value()
        unit = self.wd_unit_combo.currentText()
        self.engine.pump.send_weight_dose(weight_kg, dose_value, unit)
        self.engine.log_event(
            f"Sent weight-dose to physical pump: {weight_kg:.1f} kg, {dose_value:.3f} {unit}"
        )

    def _on_vtbi_apply(self):
        if not self.engine.pump.connected:
            return
        rate = self.vtbi_rate_spin.value()
        vtbi = self.vtbi_volume_spin.value()
        self.engine.pump.send_vtbi_program(rate, vtbi)
        self.engine.log_event(
            f"Sent Rate+VTBI to physical pump: {rate:.2f} mL/hr, VTBI {vtbi:.2f} mL"
        )

    # ---------------- Simulation timer ----------------
    def _on_timer(self):
        # Drain the pump's own unsolicited lines (E-STOP, watchdog
        # stops, limit hits, and command rejections - e.g. a manual
        # dose command sent while the firmware is in MANUAL/keypad
        # mode) regardless of run state, so a physical safety event or
        # a silently-ignored command shows up even if the sim itself
        # is paused. [FIX, 2026-10-09] REJECTED was missing from this
        # filter - a rejected SET_RATE/SET_WEIGHT_DOSE/SET_VTBI_PROGRAM
        # would arrive from the pump but never be shown anywhere,
        # looking indistinguishable from a command that silently did
        # nothing.
        for line in self.engine.pump.poll_lines():
            if any(tag in line for tag in ("E-STOP", "WATCHDOG", "LIMIT", "REJECTED")):
                self.engine.log_event(f"PUMP: {line}")
                # [ADDED 2026-10-20] Per the handoff doc: real hardware
                # safety events (not simulated ones) get their own
                # publish_event call with data_source="hardware" -
                # these came straight off the physical pump's own
                # unsolicited lines, not the digital twin.
                if self._remote_dashboard is not None:
                    severity = "critical" if "E-STOP" in line or "WATCHDOG" in line else "warning"
                    self._remote_dashboard.publish_event(
                        REMOTE_DASHBOARD_DEVICE_ID, "hardware_safety", line,
                        severity=severity, session_id=self._remote_dashboard_session_id,
                        event_source="hardware", data_source="hardware",
                    )
            if RemoteTelemetryPublisher is not None:
                # Captures the pump's own E-STOP/limit-switch/battery
                # status whenever a GET_STATUS reply happens to arrive
                # (requested periodically below) - this is the ONLY
                # place this gets parsed; poll_lines() otherwise just
                # passes raw text through unexamined.
                parsed_status = parse_status_line(line)
                if parsed_status is not None:
                    self._last_pump_status = parsed_status

        if self.engine.running:
            speed = self._current_speed()
            self.engine.tick_accumulator += speed * (TICK_TIMER_MS / 1000.0)
            prev_tick = self.engine.history[-1] if self.engine.history else None
            ticked = False
            while self.engine.tick_accumulator >= 1.0:
                result = self.engine.step()
                self.engine.detect_events(prev_tick, result)
                self.engine.history.append(result)
                prev_tick = result
                self.engine.tick_accumulator -= 1.0
                ticked = True
            if ticked:
                self.refresh_all()
                # Once per _on_timer() call (not once per simulated
                # tick inside the while loop above) - at high speed
                # multipliers several ticks can happen per 150ms wall-
                # clock interval, and the dashboard only needs the
                # latest state, not every intermediate one.
                self._send_dashboard_snapshot(prev_tick)
        else:
            self._update_status_caption()

    def _send_dashboard_snapshot(self, result):
        """Fire-and-forget UDP push to clinician_dashboard.py, if it's
        running - see self._dashboard_sock's own comment in __init__.
        Never raises into the timer loop: a UDP send to localhost with
        nothing listening just returns normally (ICMP port-unreachable
        is not surfaced as a Python exception here), and any other
        failure is swallowed the same way poll_lines()/send() already
        self-heal elsewhere in this file - this is a nice-to-have
        telemetry mirror, never something a simulation run depends on."""
        try:
            snapshot = dict(result)
            snapshot["Condition"] = self.engine.condition_label
            self._dashboard_sock.sendto(
                json.dumps(snapshot).encode("utf-8"), self._dashboard_addr
            )
        except Exception:
            pass

        self._maybe_publish_to_adafruit(result)
        self._maybe_publish_to_remote_dashboard(result)

    def _maybe_publish_to_adafruit(self, snapshot):
        """Throttled, backgrounded publish to Adafruit IO - see
        ADAFRUIT_IO_ENABLED's own comment for why this exists and why
        it's view-only. Does nothing at all unless explicitly enabled
        with real credentials."""
        if self._adafruit_client is None:
            return

        now = time.monotonic()
        if now - self._last_adafruit_publish_time < ADAFRUIT_IO_PUBLISH_INTERVAL_SEC:
            return
        self._last_adafruit_publish_time = now

        values = {
            feed: snapshot[key]
            for key, feed in ADAFRUIT_IO_FEEDS.items()
            if key in snapshot and snapshot[key] is not None
        }
        if not values:
            return

        # Adafruit IO's send() is a blocking HTTP request - never do
        # this on the GUI thread (same reasoning winsound.Beep gets its
        # own thread elsewhere in this file). A slow/failed publish
        # must never stall the simulation or freeze the UI.
        threading.Thread(
            target=self._publish_to_adafruit_worker, args=(values,), daemon=True
        ).start()

    def _publish_to_adafruit_worker(self, values):
        for feed, value in values.items():
            try:
                self._adafruit_client.send(feed, value)
            except Exception as exc:
                print(f"Adafruit IO publish failed ({feed}): {exc}")

    def _maybe_publish_to_remote_dashboard(self, snapshot):
        """Throttled publish to the Supabase-backed remote dashboard -
        see REMOTE_DASHBOARD_ENABLED's own comment. Does nothing unless
        explicitly enabled with real credentials. The actual HTTP send
        happens entirely inside RemoteTelemetryPublisher's own
        background thread/queue (telemetry_publisher.py) - everything
        called from here is non-blocking by construction."""
        if self._remote_dashboard is None:
            return

        now = time.monotonic()
        if now - self._last_remote_dashboard_publish_time < REMOTE_DASHBOARD_PUBLISH_INTERVAL_SEC:
            return
        self._last_remote_dashboard_publish_time = now

        # Safety net: the app has a patient loaded at startup (before
        # any New Random Patient/Reset click), so if REMOTE_DASHBOARD_
        # ENABLED is on from the very first tick, no session_id has
        # been minted yet. Start one now rather than publish with
        # session_id=None, which Session History couldn't group.
        if self._remote_dashboard_session_id is None:
            self._remote_dashboard_session_id = str(uuid.uuid4())

        # Ask the pump for fresh hardware status (E-STOP/limits/
        # battery) - the REPLY arrives asynchronously and is captured
        # into self._last_pump_status by the poll_lines() loop above,
        # so this tick's payload uses whatever the MOST RECENT prior
        # reply was, not this request's answer (one interval of lag,
        # acceptable at a 2s+ cadence). GET_STATUS does not touch the
        # firmware's SET_RATE watchdog timer, so this is safe to send
        # on its own schedule regardless of dosing activity.
        if self.engine.pump.connected:
            self.engine.pump.send("GET_STATUS")

        payload = build_telemetry_payload(
            device_id=REMOTE_DASHBOARD_DEVICE_ID,
            session_id=self._remote_dashboard_session_id,
            sim_result=snapshot,
            status=self._last_pump_status,
            data_source="simulation",
            infusion_mode="NE" if self.engine is self.septic_engine else "SNP",
        )
        self._remote_dashboard.publish_snapshot(payload)

    # ---------------- Rendering ----------------
    def refresh_all(self):
        self._update_status_caption()
        if not self.engine.history:
            return
        hist = self.engine.history
        latest = hist[-1]
        self._update_cards(latest)
        self._update_target_strip(hist, latest)
        self._update_clinical_alerts(latest)
        self._update_risk_box(latest)
        self._update_charts(hist, latest)
        self._update_trend_panels(latest)
        self._update_event_log()
        self._update_history_table(hist)
        self._update_heartbeat_sound(latest)

    def _update_heartbeat_sound(self, latest):
        """Paces a short beep to the patient's own heart rate, like a
        bedside monitor's QRS tone - faster/higher-pitched during
        Emergency mode or an elevated risk score, same as a real
        monitor's alarm state."""
        if not self.sound_checkbox.isChecked() or not self.engine.running:
            self.beep_timer.stop()
            return

        critical = latest["ControllerMode"] == "Emergency" or latest["RiskScore"] >= RISK_ALERT_THRESHOLD
        if critical:
            new_freq, new_dur = 1568, 90  # higher pitch, ICU-alarm-like
            interval = 350
        else:
            new_freq, new_dur = 880, 50
            hr = max(latest["HR"], 30.0)
            interval = int(np.clip(60000.0 / hr, 300, 1500))

        tone_changed = (new_freq, new_dur) != (self._beep_freq, self._beep_dur)
        self._beep_freq, self._beep_dur = new_freq, new_dur

        # HR drifts by fractions of a beat almost every tick, which
        # recomputes a slightly different interval almost every
        # refresh. Restarting the timer on ANY difference resets its
        # countdown before it ever reaches zero, so the beep rarely
        # actually fires - only restart on a meaningful change (or a
        # normal/critical tone switch), so the timer survives long
        # enough to trigger on a regular, audible cadence.
        interval_changed = abs(interval - self.beep_timer.interval()) > 25
        if not self.beep_timer.isActive() or interval_changed or tone_changed:
            self.beep_timer.start(interval)

    def _on_beep_tick(self):
        # winsound.Beep blocks for its duration; run off the GUI thread
        # so a beep never stalls chart updates or button clicks.
        threading.Thread(target=winsound.Beep, args=(self._beep_freq, self._beep_dur), daemon=True).start()

    def _update_status_caption(self):
        p = self.engine.patient
        status = "RUNNING" if self.engine.running else "PAUSED"
        severity = f"SAPS II: {p.saps_ii}" if p.saps_ii is not None else f"Condition: {self.engine.condition_label}"
        self.status_caption.setText(f"Profile: {p.profile}  |  {severity}  |  Status: {status}")

    def _update_cards(self, latest):
        at_target_label = "Normal" if self.engine is self.septic_engine else "At Floor"
        label, color = bp_status(latest["MAP"], latest["TargetBP"], TARGET_DEADBAND, at_target_label)
        self.card_map.update_value(f"{latest['MAP']} mmHg", label, color)
        label, color = hr_status(latest["HR"])
        self.card_hr.update_value(f"{latest['HR']} bpm", label, color)
        label, color = spo2_status(latest["SpO2"])
        self.card_spo2.update_value(f"{latest['SpO2']} %", label, color)
        label, color = flow_status(latest["ControllerFlow"], self.engine.max_flow_ml_hr)
        weight_kg = self.engine.patient.weight_kg
        dose_mcg_kg_min = self.engine.flow_rate_to_mass_rate(latest["ControllerFlow"]) / weight_kg
        self.card_flow.update_value(
            f"{latest['ControllerFlow']} mL/hr<br>"
            f"<span style='font-size:13px;font-weight:500;color:{TEXT_DIM};'>{dose_mcg_kg_min:.3f} µg/kg/min</span>",
            label, color
        )
        label, color = mode_status(latest["AI_Action"])
        self.card_ai_mode.update_value(latest["AI_Action"], label, color)
        label, color = mode_status(latest["ControllerMode"])
        self.card_ctrl_mode.update_value(latest["ControllerMode"], label, color)

    def _update_target_strip(self, hist, latest):
        in_target_flags = [h["InTargetRange"] for h in hist]
        pct_in_target = 100.0 * sum(in_target_flags) / len(in_target_flags)
        first_reached = next((h["Time"] for h in hist if h["InTargetRange"]), None)

        # The reference value's own label/number - shown explicitly
        # rather than only implied by the gap, since for the
        # hypertensive mode this number changes every tick (the staged
        # safety floor descending) and was previously only visible
        # indirectly - a real gap noticed running the app for real.
        ref_label = "Target" if self.engine is self.septic_engine else "Floor"
        ref_value = latest["TargetBP"]

        # Keep the (disabled, for hypertensive mode) Target MAP spinbox
        # in sync with whatever reference value is actually active, so
        # it always shows a real live number instead of a stale initial
        # one - harmless for septic mode, where this just re-affirms the
        # same fixed value every tick.
        self.target_spin.blockSignals(True)
        self.target_spin.setValue(ref_value)
        self.target_spin.blockSignals(False)

        # Two separate questions, shown together rather than collapsed into
        # one label: (1) has the clinical goal (MAP >= target, the actual
        # SSC-style minimum) been met, and (2) is the controller still
        # actively adjusting or has it settled. A patient at 69 mmHg
        # against a 65 mmHg target HAS met the clinical goal even though
        # the controller's own tighter deadband may still be trimming the
        # dose down - conflating the two reads as alarmist/wrong to a
        # clinician and was the exact complaint that led to this split.
        if latest["ControllerMode"] == "Emergency":
            self.target_status_label.setText(
                f"EMERGENCY — {latest['ClinicalExplanation']}  ({ref_label}: {ref_value:.0f} mmHg)")
            self.target_status_label.setStyleSheet(f"background:{RED}22; color:{RED}; border:1px solid {RED};")
        elif latest["MAP"] >= latest["TargetBP"]:
            settled = latest["ControllerMode"] == "Maintain"
            descriptor = "Stable" if settled else "Fine-tuning dose"
            self.target_status_label.setText(
                f"{ref_label.upper()} MET — {descriptor}  ({ref_label}: {ref_value:.0f} mmHg)")
            self.target_status_label.setStyleSheet(f"background:{GREEN}22; color:{GREEN}; border:1px solid {GREEN};")
        else:
            gap = round(latest["TargetBP"] - latest["MAP"], 1)
            self.target_status_label.setText(
                f"BELOW {ref_label.upper()} ({gap} mmHg below)  ({ref_label}: {ref_value:.0f} mmHg)")
            self.target_status_label.setStyleSheet(f"background:{YELLOW}22; color:{YELLOW}; border:1px solid {YELLOW};")

        self.metric_pct_target.set_value(f"{pct_in_target:.0f}%")
        self.metric_first_reached.set_value(f"{first_reached}s" if first_reached is not None else "Not yet")

    def _update_clinical_alerts(self, latest):
        """Structured clinical advisories (clinical_alerts.py) - display
        only, same as every other advisory layer in this app; never
        touches dosing. Publishes onset/clear transitions to the remote
        dashboard the same way hardware safety events already are (see
        _on_timer's own publish_event call), tagged data_source=
        "simulation" since these come from the digital twin, not the
        physical pump."""
        is_septic = self.engine is self.septic_engine
        alerts = clinical_alerts.get_alerts(is_septic, latest)

        if not alerts:
            self.clinical_alert_box.setVisible(False)
            if self._active_alert_titles:
                self.engine.log_event("Clinical alerts cleared")
                if self._remote_dashboard is not None:
                    self._remote_dashboard.publish_event(
                        REMOTE_DASHBOARD_DEVICE_ID, "clinical_alert", "All clinical advisories cleared",
                        severity="info", session_id=self._remote_dashboard_session_id,
                        event_source="clinical_alert_engine", data_source="simulation",
                    )
            self._active_alert_titles = set()
            return

        self.clinical_alert_box.setVisible(True)
        n_critical = sum(1 for a in alerts if a["severity"] == clinical_alerts.CRITICAL)
        worst_color = RED if n_critical else YELLOW
        plural = "ADVISORY" if len(alerts) == 1 else "ADVISORIES"
        self.clinical_alert_title.setText(
            f"⚠ CLINICAL ALERT: {len(alerts)} ACTIVE {plural} (CRITICAL: {n_critical})")
        self.clinical_alert_title.setStyleSheet(f"color: {worst_color};")
        self.clinical_alert_box.setStyleSheet(
            f"QFrame#panel {{ background-color: {PANEL}; "
            f"border: 2px solid {worst_color}; border-radius: 10px; }}"
        )

        rows = []
        for a in alerts:
            critical = a["severity"] == clinical_alerts.CRITICAL
            tag_color = RED if critical else YELLOW
            tag = "CRITICAL" if critical else "WARNING"
            rows.append(
                f"<p style='margin:6px 0;'>"
                f"<span style='color:{tag_color}; font-weight:bold;'>● [{tag}] {a['title']}</span><br>"
                f"<b>Potential Consequence if Unchecked:</b> {a['consequence']}<br>"
                f"<span style='color:{GREEN};'><b>Actionable Clinical Recommendation:</b> "
                f"{a['recommendation']}</span></p>"
            )
        self.clinical_alert_body.setText("".join(rows))

        current_titles = {a["title"] for a in alerts}
        new_titles = current_titles - self._active_alert_titles
        cleared_titles = self._active_alert_titles - current_titles

        for a in alerts:
            if a["title"] in new_titles:
                self.engine.log_event(f"CLINICAL ALERT ({a['severity'].upper()}): {a['title']}")
                if self._remote_dashboard is not None:
                    # [FIX] Was sending only a["title"] - the dashboard
                    # never got the consequence/recommendation text that
                    # makes the alert actionable there, only its bare
                    # name. Include all three fields in the message, same
                    # content shown on the desktop panel.
                    full_message = (
                        f"{a['title']} | Potential Consequence if Unchecked: {a['consequence']} "
                        f"| Actionable Clinical Recommendation: {a['recommendation']}"
                    )
                    self._remote_dashboard.publish_event(
                        REMOTE_DASHBOARD_DEVICE_ID, "clinical_alert", full_message,
                        severity=a["severity"], session_id=self._remote_dashboard_session_id,
                        event_source="clinical_alert_engine", data_source="simulation",
                    )
        for title in cleared_titles:
            self.engine.log_event(f"Clinical alert resolved: {title}")
            if self._remote_dashboard is not None:
                self._remote_dashboard.publish_event(
                    REMOTE_DASHBOARD_DEVICE_ID, "clinical_alert", f"Resolved: {title}",
                    severity="info", session_id=self._remote_dashboard_session_id,
                    event_source="clinical_alert_engine", data_source="simulation",
                )

        self._active_alert_titles = current_titles

    def _update_risk_box(self, latest):
        risk = latest["RiskScore"]
        self.risk_value_label.setText(f"{risk*100:.0f}%")
        if risk >= RISK_ALERT_THRESHOLD:
            self.risk_status_label.setText(f"ELEVATED RISK — {risk*100:.0f}% in next {LOOKAHEAD}s")
            color = RED
        elif risk >= 0.2:
            self.risk_status_label.setText(f"Watch — {risk*100:.0f}% in next {LOOKAHEAD}s")
            color = YELLOW
        else:
            self.risk_status_label.setText(f"Low risk ({risk*100:.0f}%)")
            color = GREEN
        self.risk_value_label.setStyleSheet(f"color: {color};")
        self.risk_status_label.setStyleSheet(f"color: {color};")

    def _update_charts(self, hist, latest):
        t = np.array([h["Time"] for h in hist])
        target = latest["TargetBP"]
        self.target_line.setPos(target)
        self.target_band.setRegion((target - TARGET_DEADBAND, target + TARGET_DEADBAND))

        self.curve_map_true.setData(t, np.array([h["MAP"] for h in hist]))
        self.curve_map_meas.setData(t, np.array([h["MeasuredMAP"] for h in hist]))
        self.curve_hr.setData(t, np.array([h["HR"] for h in hist]))
        self.curve_spo2.setData(t, np.array([h["SpO2"] for h in hist]))
        self.curve_flow_ctrl.setData(t, np.array([h["ControllerFlow"] for h in hist]))
        self.curve_flow_ai.setData(t, np.array([h["AI_SuggestedFlow"] for h in hist]))
        self.curve_plasma.setData(t, np.array([h["PlasmaConcentration"] for h in hist]))
        self.curve_risk.setData(t, np.array([h["RiskScore"] for h in hist]))

    def _update_trend_panels(self, latest):
        p = self.engine.patient
        agree_line = "Agrees with Controller" if latest["ActionAgrees"] else "Disagrees with Controller"
        alert_line = f"\n{latest['Alert']}" if latest["Alert"] else ""
        self.ai_panel["body"].setText(
            f"Predicted Action: {latest['AI_Action']}\n"
            f"Predicted Flow: {latest['AI_SuggestedFlow']} mL/hr\n"
            f"Flow Divergence: {latest['FlowDivergence']} mL/hr\n\n"
            f"{agree_line}{alert_line}"
        )

        self.ctrl_panel["body"].setText(
            f"Mode: {latest['ControllerMode']}\n"
            f"Flow Rate: {latest['ControllerFlow']} mL/hr\n"
            f"Reason: {latest['ClinicalExplanation']}\n"
            f"Confidence: {latest['Confidence']}\n"
            f"Stability Score: {latest['StabilityScore']}"
        )

        hr_advisory_line = (
            "\n⚠ HR elevated despite adequate MAP — consider non-shock causes "
            "(pain, agitation, fever). Not addressable by NE dosing."
            if latest["HRAdvisory"] else ""
        )
        severity_tag = f"SAPS II {p.saps_ii}" if p.saps_ii is not None else self.engine.condition_label
        self.hemo_panel["body"].setText(
            f"Age / Weight: {p.age}y / {p.weight_kg:.0f} kg\n"
            f"Profile: {p.profile} ({severity_tag})\n"
            f"SVR: {latest['SVR']} dyn·s/cm⁵ | CO: {latest['CO']} L/min\n"
            f"Drug Effect (E): {latest['DrugEffect']} | Plasma Cp: {latest['PlasmaConcentration']} mcg/L\n"
            f"Clearance: {p.clearance_l_min:.2f} L/min | Vd: {p.vd_l:.1f} L\n"
            f"Time stable at target: {latest['TimeInTargetMin']:.1f} min"
            f"{hr_advisory_line}"
        )

    def _update_event_log(self):
        lines = list(reversed(self.engine.event_log)) or ["No events yet."]
        self.event_log_view.setPlainText("\n".join(lines))

    def _update_history_table(self, hist):
        rows = hist[-MAX_HISTORY_ROWS_SHOWN:]
        self.history_table.setRowCount(len(rows))
        for r, h in enumerate(rows):
            values = [
                h["Time"], h["MAP"], h["MeasuredMAP"], h["HR"], h["SpO2"],
                h["ControllerMode"], h["ControllerFlow"], h["AI_Action"],
                h["AI_SuggestedFlow"], f"{h['RiskScore']*100:.0f}%",
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.history_table.setItem(r, col, item)
        self.history_table.scrollToBottom()

    # ---------------- Lifecycle ----------------
    def closeEvent(self, event):
        # Never leave the real motor running just because the window closed.
        self.engine.pump.disconnect()
        # [ADDED 2026-10-20] Lets the dashboard show Offline immediately
        # instead of waiting for the stale-threshold timeout to guess.
        self._end_remote_dashboard_session()
        if self._remote_dashboard is not None:
            self._remote_dashboard.stop()
        super().closeEvent(event)

    # ---------------- Theme ----------------
    def _qss(self):
        return f"""
        QWidget {{ background-color: {BG}; color: {TEXT}; font-family: 'Segoe UI'; font-size: 13px; }}
        #sidebar {{ background-color: {PANEL}; border-right: 1px solid {BORDER}; }}
        #sidebarTitle {{ font-size: 15px; font-weight: 700; color: {TEXT}; letter-spacing: 1px; }}
        #mainTitle {{ font-size: 20px; font-weight: 700; }}
        #disclaimerBadge {{ color: {YELLOW}; font-size: 11px; font-weight: 700; letter-spacing: 0.5px;
                             border: 1px solid {YELLOW}; border-radius: 4px; padding: 4px 8px; }}
        #hint {{ color: {TEXT_DIM}; font-size: 11px; }}
        QGroupBox {{ border: 1px solid {BORDER}; border-radius: 8px; margin-top: 8px; padding-top: 10px; font-weight: 600; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 8px; color: {TEXT_DIM}; }}
        QPushButton {{ background-color: {PANEL2}; border: 1px solid {BORDER}; border-radius: 6px; padding: 8px; font-weight: 600; }}
        QPushButton:hover {{ border-color: {ACCENT}; }}
        QPushButton#primaryBtn {{ background-color: {ACCENT}; border: none; color: white; }}
        QPushButton#primaryBtn:hover {{ background-color: #2563eb; }}
        QDoubleSpinBox {{ background-color: {PANEL2}; border: 1px solid {BORDER}; border-radius: 6px; padding: 4px; }}
        QRadioButton {{ padding: 2px; }}
        #card {{ background-color: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; }}
        #cardTitle {{ color: {TEXT_DIM}; font-size: 11px; font-weight: 700; letter-spacing: 1px; }}
        #cardValue {{ font-size: 22px; font-weight: 700; }}
        #cardStatus {{ font-size: 12px; }}
        #metric {{ background-color: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; }}
        #metricTitle {{ color: {TEXT_DIM}; font-size: 11px; }}
        #metricValue {{ font-size: 18px; font-weight: 700; }}
        #targetStrip {{ border-radius: 10px; font-weight: 700; font-size: 13px; padding: 8px; }}
        #panel {{ background-color: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; }}
        #panelTitle {{ font-weight: 700; font-size: 13px; padding: 4px 0; }}
        #panelBody {{ color: {TEXT}; font-size: 12px; }}
        #riskValue {{ font-size: 28px; font-weight: 800; }}
        #riskStatus {{ font-size: 12px; font-weight: 600; }}
        QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; top: -1px; }}
        QTabBar::tab {{ background: {PANEL}; padding: 8px 16px; border: 1px solid {BORDER}; border-bottom: none;
                         border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 2px; }}
        QTabBar::tab:selected {{ background: {PANEL2}; color: {ACCENT}; font-weight: 700; }}
        QTextEdit#eventLog {{ background-color: {PANEL2}; border: 1px solid {BORDER}; border-radius: 6px; font-family: Consolas; font-size: 12px; }}
        QTableWidget {{ background-color: {PANEL2}; border: 1px solid {BORDER}; border-radius: 6px; gridline-color: {BORDER}; }}
        QHeaderView::section {{ background-color: {PANEL}; color: {TEXT_DIM}; border: none; padding: 4px; font-weight: 700; }}
        QScrollArea {{ border: none; background-color: {BG}; }}
        QScrollBar:vertical {{ background: {BG}; width: 12px; margin: 0; }}
        QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 6px; min-height: 24px; }}
        QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QCheckBox {{ padding: 2px; }}
        """


def main():
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
