"""
===========================================================
TELEMETRY_PUBLISHER.PY  (Firebase edition)
Smart AI Infusion Pump - Remote Monitoring Dashboard (M4)

Drop-in replacement for the Supabase version that lives beside
native_app.py. parse_status_line(), build_telemetry_payload() and
the RemoteTelemetryPublisher method names are unchanged; only the
constructor arguments differ (Firebase settings instead of an
ingest URL + token).

How data reaches the cloud:
  - The gateway signs in as its OWN Firebase Auth user (a "device
    account", email/password), never an admin or service-account key
    (spec Section 9: least privilege, no privileged keys in the
    desktop distributable).
  - Every snapshot/event is one atomic Firestore commit that appends
    the record AND stamps a heartbeat on devices/{id}. firestore.rules
    only accepts the record when the heartbeat is in the same commit,
    which is how schema validation, server receipt time and the
    minimum spacing between writes are enforced server-side.
  - Each record carries a record_id used as its document id with an
    "must not exist" precondition, so a retried/reconnected upload
    can never create a duplicate (spec Section 10.1).

Runtime behaviour is the same as before (spec Section 10):
  - bounded queue + one daemon worker thread; publish_*() never
    blocks the GUI or the control loop,
  - 3 s HTTP timeout, bounded exponential backoff with jitter,
  - queue full -> drop the OLDEST item,
  - failures logged to the console with a category, never secrets.
  - Nothing here ever reads from the cloud or changes a pump setting.
===========================================================
"""

import json
import os
import queue
import random
import threading
import time
import uuid
from datetime import datetime, timezone

import requests

SCHEMA_VERSION = "1.0"

HTTP_TIMEOUT_SEC = 3.0
RETRY_DELAYS_SEC = [1, 2, 4, 8]
QUEUE_MAX_SIZE = 50
# firestore.rules rejects heartbeats closer than 250 ms apart.
MIN_SEND_SPACING_SEC = 0.35
# Refresh the 1-hour ID token this long before it expires.
TOKEN_REFRESH_MARGIN_SEC = 300

TIMESTAMP_FIELDS = {"recorded_at", "source_timestamp", "occurred_at", "started_at", "last_recorded_at", "ended_at"}


def _utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def parse_status_line(line):
    """Parses the firmware's GET_STATUS reply line (pump_firmware.ino's
    printStatusLineTo()) into a dict with real types.

    Input example:
        "STATUS,mode=AUTONOMOUS,rate=12.50,running=1,estop=0,home=0,
         end=0,battery_v=11.80,battery_pct=64,vtbi_active=0,
         vtbi_target_ml=0.00,vtbi_delivered_ml=0.00,cal_active=0,
         cal_steps_done=0,cal_steps_target=0,retract_active=0,
         retract_steps_done=0,sound_enabled=1,wifi_enabled=1,
         position=1234"

    Returns None if the line isn't a STATUS line at all.
    """
    if not line.startswith("STATUS,"):
        return None

    fields = {}
    for part in line[len("STATUS,"):].split(","):
        if "=" not in part:
            continue
        key, _, value = part.partition("=")
        fields[key] = value

    def _bool(key):
        return fields.get(key) == "1"

    def _float(key):
        try:
            return float(fields[key])
        except (KeyError, ValueError):
            return None

    def _int(key):
        try:
            return int(fields[key])
        except (KeyError, ValueError):
            return None

    return {
        "mode": fields.get("mode"),
        "rate": _float("rate"),
        "running": _bool("running"),
        "estop_active": _bool("estop"),
        "home_limit_active": _bool("home"),
        "end_limit_active": _bool("end"),
        "battery_voltage_v": _float("battery_v"),
        "battery_pct": _float("battery_pct"),
        "vtbi_active": _bool("vtbi_active"),
        "vtbi_target_ml": _float("vtbi_target_ml"),
        "vtbi_delivered_ml": _float("vtbi_delivered_ml"),
        "sound_enabled": _bool("sound_enabled"),
        "wifi_enabled": _bool("wifi_enabled"),
        "position": _int("position"),
    }


def build_telemetry_payload(device_id, session_id, sim_result=None, status=None,
                            data_source="simulation", infusion_mode=None,
                            model_version=None):
    """Maps native_app.py's per-tick SimEngine/HypertensiveSimEngine
    result dict, plus an optional parsed GET_STATUS dict, into the
    versioned telemetry contract (spec Section 7) that firestore.rules
    validates. Pure function - no I/O.
    """
    now = _utc_now_iso()

    payload = {
        "schema_version": SCHEMA_VERSION,
        "device_id": device_id,
        "session_id": session_id,
        "recorded_at": now,
        "data_source": data_source,
        "device_status": "unknown",
        "infusion_mode": infusion_mode,
        "model_version": model_version,
    }

    if status is not None:
        payload["device_status"] = "running" if status.get("running") else "stopped"
        payload["estop_active"] = status.get("estop_active")
        payload["home_limit_active"] = status.get("home_limit_active")
        payload["end_limit_active"] = status.get("end_limit_active")
        payload["battery_voltage_v"] = status.get("battery_voltage_v")
        payload["battery_pct"] = status.get("battery_pct")
        payload["commanded_flow_ml_hr"] = status.get("rate")
        payload["estimated_delivered_volume_ml"] = status.get("vtbi_delivered_ml")
        payload["syringe_position_steps"] = status.get("position")
        payload["controller_status"] = status.get("mode")
        # measured_flow_ml_hr / delivered_volume_ml deliberately left
        # unset - there is no independently-measured flow sensor in
        # this project. Never populate them from an estimate.

    if sim_result is not None:
        payload["map_mmhg"] = sim_result.get("MAP")
        payload["heart_rate_bpm"] = sim_result.get("HR")
        payload["spo2_pct"] = sim_result.get("SpO2")
        # TargetBP (septic) vs MAP_Floor (hypertensive) - prefer
        # whichever is actually present.
        payload["target_or_floor_mmhg"] = sim_result.get("TargetBP", sim_result.get("MAP_Floor"))
        payload["ai_action"] = sim_result.get("AI_Action")
        payload["ai_suggested_flow_ml_hr"] = sim_result.get("AI_SuggestedFlow")
        payload["ai_risk_score"] = sim_result.get("RiskScore")
        payload["cyanide_rate_risk"] = sim_result.get("CyanideRateRisk")
        payload["cyanide_buffer_exceeded"] = sim_result.get("CyanideBufferExceeded")
        # ControllerFlow is the real actuator output when no physical
        # pump status is available to override it.
        if payload.get("commanded_flow_ml_hr") is None:
            payload["commanded_flow_ml_hr"] = sim_result.get("ControllerFlow")
        if payload["device_status"] == "unknown":
            payload["device_status"] = "running"

    return payload


# -----------------------------------------------------------
# Firestore REST encoding
# -----------------------------------------------------------

def _to_value(key, value):
    if value is None:
        return {"nullValue": None}
    if isinstance(value, bool):
        return {"booleanValue": value}
    if key in TIMESTAMP_FIELDS and isinstance(value, str):
        return {"timestampValue": value}
    if isinstance(value, int):
        # Stored as double so charts/rules treat every reading alike.
        return {"doubleValue": float(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


def _fields(data):
    return {k: _to_value(k, v) for k, v in data.items()}


class _Rejected(Exception):
    """Firestore refused the commit for a reason retrying won't fix."""


class _Duplicate(Exception):
    """The record (or session) already exists - already delivered."""


class RemoteTelemetryPublisher:
    """One instance per app run. Call publish_snapshot()/publish_event()
    from the GUI thread as often as you like; call stop() on app close.

        publisher = RemoteTelemetryPublisher(
            api_key=REMOTE_DASHBOARD_FIREBASE_API_KEY,
            project_id=REMOTE_DASHBOARD_FIREBASE_PROJECT_ID,
            device_id=REMOTE_DASHBOARD_DEVICE_ID,
            device_email=REMOTE_DASHBOARD_DEVICE_EMAIL,
            device_password=REMOTE_DASHBOARD_DEVICE_PASSWORD,
            enabled=REMOTE_DASHBOARD_ENABLED,
        )
    """

    def __init__(self, api_key, project_id, device_id, device_email, device_password,
                 enabled=True, database="(default)"):
        self.api_key = api_key
        self.project_id = project_id
        self.device_id = device_id
        self.device_email = device_email
        self.device_password = device_password
        self.enabled = enabled
        self._docs_root = f"projects/{project_id}/databases/{database}/documents"
        # Standard Firebase emulator variables - local testing only.
        fs_host = os.environ.get("FIRESTORE_EMULATOR_HOST")
        auth_host = os.environ.get("FIREBASE_AUTH_EMULATOR_HOST")
        fs_base = f"http://{fs_host}" if fs_host else "https://firestore.googleapis.com"
        self._auth_base = f"http://{auth_host}/" if auth_host else "https://"
        self._commit_url = f"{fs_base}/v1/{self._docs_root}:commit"

        self._id_token = None
        self._refresh_token = None
        self._token_expires_at = 0.0
        self._last_send_at = 0.0
        self._last_recorded_at = None          # newest recorded_at we have delivered
        self._known_sessions = set()            # session docs we know exist
        self._session_last = {}                 # session_id -> newest recorded_at

        self._session = requests.Session()
        self._queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self._stop_event = threading.Event()
        self._thread = None
        if self.enabled:
            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()

    # ---------- public API (called from the GUI thread) ----------

    def publish_snapshot(self, payload):
        self._enqueue(("telemetry", dict(payload)))

    def publish_event(self, device_id, event_type, message, severity="info",
                      session_id=None, event_source="gateway", data_source="simulation"):
        self._enqueue(("event", {
            "schema_version": SCHEMA_VERSION,
            "device_id": device_id,
            "session_id": session_id,
            "occurred_at": _utc_now_iso(),
            "data_source": data_source,
            "event_type": event_type,
            "severity": severity,
            "source": event_source,
            "message": message,
        }))

    def end_session(self, session_id):
        """Stamps ended_at on the session (e.g. when the run is stopped)."""
        if session_id:
            self._enqueue(("end_session", {"session_id": session_id, "ended_at": _utc_now_iso()}))

    def stop(self):
        """Marks the gateway explicitly offline (best effort), then stops."""
        self._stop_event.set()
        if self._thread is None:
            return
        self._thread.join(timeout=2.0)
        try:
            self._commit([self._gateway_offline_write()])
        except Exception as exc:  # never raise into the app on shutdown
            print(f"[telemetry_publisher] could not mark gateway offline: {type(exc).__name__}")

    # ---------- queue ----------

    def _enqueue(self, item):
        if not self.enabled:
            return
        kind, data = item
        data.setdefault("record_id", str(uuid.uuid4()))
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Drop the OLDEST item - a current reading beats a backlog.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                pass

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._send_with_retry(item)

    # ---------- auth (device account) ----------

    def _ensure_token(self, force=False):
        if not force and self._id_token and time.time() < self._token_expires_at - TOKEN_REFRESH_MARGIN_SEC:
            return
        if self._refresh_token and not force:
            resp = self._session.post(
                f"{self._auth_base}securetoken.googleapis.com/v1/token?key={self.api_key}",
                data={"grant_type": "refresh_token", "refresh_token": self._refresh_token},
                timeout=HTTP_TIMEOUT_SEC,
            )
            if resp.status_code == 200:
                body = resp.json()
                self._id_token = body["id_token"]
                self._refresh_token = body["refresh_token"]
                self._token_expires_at = time.time() + int(body.get("expires_in", 3600))
                return
        resp = self._session.post(
            f"{self._auth_base}identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={self.api_key}",
            json={"email": self.device_email, "password": self.device_password, "returnSecureToken": True},
            timeout=HTTP_TIMEOUT_SEC,
        )
        if resp.status_code != 200:
            reason = resp.json().get("error", {}).get("message", resp.status_code) if resp.content else resp.status_code
            raise _Rejected(f"device sign-in failed ({reason})")
        body = resp.json()
        self._id_token = body["idToken"]
        self._refresh_token = body["refreshToken"]
        self._token_expires_at = time.time() + int(body.get("expiresIn", 3600))

    # ---------- write builders ----------

    def _doc(self, *parts):
        return f"{self._docs_root}/" + "/".join(parts)

    def _heartbeat_write(self, data_source, recorded_at):
        fields = {"gateway_state": "online", "last_data_source": data_source}
        if recorded_at and (self._last_recorded_at is None or recorded_at >= self._last_recorded_at):
            fields["last_recorded_at"] = recorded_at
        return {
            "update": {"name": self._doc("devices", self.device_id), "fields": _fields(fields)},
            "updateMask": {"fieldPaths": list(fields.keys())},
            "updateTransforms": [{"fieldPath": "last_ingest_at", "setToServerValue": "REQUEST_TIME"}],
            "currentDocument": {"exists": True},
        }

    def _gateway_offline_write(self):
        return {
            "update": {"name": self._doc("devices", self.device_id), "fields": _fields({"gateway_state": "offline"})},
            "updateMask": {"fieldPaths": ["gateway_state"]},
            "currentDocument": {"exists": True},
        }

    def _session_write(self, session_id, data_source, mode, recorded_at, include_create):
        if include_create:
            return {
                "update": {"name": self._doc("devices", self.device_id, "sessions", session_id), "fields": _fields({
                    "device_id": self.device_id, "data_source": data_source, "mode": mode,
                    "started_at": recorded_at, "last_recorded_at": recorded_at,
                })},
                "currentDocument": {"exists": False},
            }
        prev = self._session_last.get(session_id)
        if prev is not None and recorded_at < prev:
            return None  # an older buffered record; don't move the session backwards
        fields = {"last_recorded_at": recorded_at}
        if mode:
            fields["mode"] = mode
        return {
            "update": {"name": self._doc("devices", self.device_id, "sessions", session_id), "fields": _fields(fields)},
            "updateMask": {"fieldPaths": list(fields.keys())},
            "currentDocument": {"exists": True},
        }

    def _build_writes(self, kind, data, include_session_create):
        data = dict(data)
        record_id = data.pop("record_id")
        if kind == "end_session":
            return [{
                "update": {"name": self._doc("devices", self.device_id, "sessions", data["session_id"]),
                           "fields": _fields({"ended_at": data["ended_at"]})},
                "updateMask": {"fieldPaths": ["ended_at"]},
                "currentDocument": {"exists": True},
            }]

        collection = "telemetry" if kind == "telemetry" else "events"
        when = data.get("recorded_at") if kind == "telemetry" else data.get("occurred_at")
        writes = [
            self._heartbeat_write(data.get("data_source"), when if kind == "telemetry" else None),
            {
                "update": {"name": self._doc("devices", self.device_id, collection, record_id), "fields": _fields(data)},
                "updateTransforms": [{"fieldPath": "received_at", "setToServerValue": "REQUEST_TIME"}],
                "currentDocument": {"exists": False},
            },
        ]
        session_id = data.get("session_id")
        if kind == "telemetry" and session_id:
            create = include_session_create and session_id not in self._known_sessions
            sw = self._session_write(session_id, data.get("data_source"), data.get("infusion_mode"), when, create)
            if sw is not None:
                writes.append(sw)
        return writes

    # ---------- sending ----------

    def _commit(self, writes, allow_retry_auth=True):
        self._ensure_token()
        wait = MIN_SEND_SPACING_SEC - (time.time() - self._last_send_at)
        if wait > 0:
            time.sleep(wait)
        self._last_send_at = time.time()
        resp = self._session.post(
            self._commit_url,
            data=json.dumps({"writes": writes}),
            headers={"Authorization": f"Bearer {self._id_token}", "Content-Type": "application/json"},
            timeout=HTTP_TIMEOUT_SEC,
        )
        if resp.status_code == 200:
            return
        status = ""
        try:
            status = resp.json().get("error", {}).get("status", "")
        except ValueError:
            pass
        if resp.status_code == 401 and allow_retry_auth:
            self._ensure_token(force=True)
            return self._commit(writes, allow_retry_auth=False)
        if status in ("ALREADY_EXISTS", "FAILED_PRECONDITION") or resp.status_code == 409:
            raise _Duplicate(status)
        if resp.status_code in (400, 403, 404):
            raise _Rejected(f"{resp.status_code} {status}: {resp.text[:200]}")
        # 429 / 5xx - worth retrying.
        raise requests.RequestException(f"HTTP {resp.status_code} {status}")

    def _send_with_retry(self, item):
        kind, data = item
        include_session_create = True
        for attempt, delay in enumerate([0] + RETRY_DELAYS_SEC):
            if delay:
                time.sleep(delay + random.uniform(0, 0.5))
            if self._stop_event.is_set():
                return
            writes = self._build_writes(kind, data, include_session_create)
            try:
                self._commit(writes)
                self._after_success(kind, data)
                return
            except _Duplicate:
                session_id = data.get("session_id")
                if kind == "telemetry" and session_id and session_id not in self._known_sessions:
                    # The session doc already existed (e.g. app restart) -
                    # resend once without trying to create it.
                    self._known_sessions.add(session_id)
                    include_session_create = False
                    try:
                        self._commit(self._build_writes(kind, data, False))
                        self._after_success(kind, data)
                    except _Duplicate:
                        pass  # record itself already delivered
                    except (_Rejected, requests.RequestException) as exc:
                        print(f"[telemetry_publisher] send failed after session retry: {exc}")
                return  # already delivered - idempotent, nothing to do
            except _Rejected as exc:
                print(f"[telemetry_publisher] rejected ({kind}): {exc}")
                return
            except requests.RequestException as exc:
                print(f"[telemetry_publisher] send failed (attempt {attempt + 1}, {type(exc).__name__}): {exc}")
        print(f"[telemetry_publisher] giving up on one {kind} record after all retries")

    def _after_success(self, kind, data):
        if kind == "telemetry":
            when = data.get("recorded_at")
            if when and (self._last_recorded_at is None or when > self._last_recorded_at):
                self._last_recorded_at = when
            session_id = data.get("session_id")
            if session_id:
                self._known_sessions.add(session_id)
                if when and when > self._session_last.get(session_id, ""):
                    self._session_last[session_id] = when


if __name__ == "__main__":
    # Offline self-test - no network needed.
    line = ("STATUS,mode=AUTONOMOUS,rate=12.50,running=1,estop=0,home=0,"
            "end=0,battery_v=11.80,battery_pct=64,vtbi_active=0,"
            "vtbi_target_ml=0.00,vtbi_delivered_ml=3.25,cal_active=0,"
            "cal_steps_done=0,cal_steps_target=0,retract_active=0,"
            "retract_steps_done=0,sound_enabled=1,wifi_enabled=1,position=1234")
    status = parse_status_line(line)
    assert status["estop_active"] is False
    assert status["rate"] == 12.50
    assert status["position"] == 1234

    sim_result = {
        "MAP": 71.2, "HR": 88.0, "SpO2": 97.0, "TargetBP": 65.0,
        "ControllerFlow": 12.5, "AI_Action": "Maintain",
        "AI_SuggestedFlow": 12.3, "RiskScore": 0.04,
    }
    payload = build_telemetry_payload("prototype-01", "s-1", sim_result=sim_result, status=status,
                                      data_source="simulation", infusion_mode="NE")
    assert payload["commanded_flow_ml_hr"] == 12.50  # hardware status wins
    assert "measured_flow_ml_hr" not in payload  # never fabricated

    pub = RemoteTelemetryPublisher("k", "demo-project", "prototype-01", "d@x", "p", enabled=False)
    payload["record_id"] = "r-1"
    writes = pub._build_writes("telemetry", payload, include_session_create=True)
    assert writes[0]["update"]["name"].endswith("/devices/prototype-01")
    assert writes[1]["update"]["name"].endswith("/devices/prototype-01/telemetry/r-1")
    assert writes[1]["update"]["fields"]["recorded_at"].keys() == {"timestampValue"}
    assert writes[1]["update"]["fields"]["map_mmhg"] == {"doubleValue": 71.2}
    assert writes[1]["currentDocument"] == {"exists": False}
    assert writes[2]["currentDocument"] == {"exists": False}  # session create
    assert "record_id" not in writes[1]["update"]["fields"]
    print("All offline self-tests passed.")
