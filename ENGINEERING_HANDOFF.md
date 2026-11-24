# Engineering Handoff — Smart AI Infusion Pump

Briefing for a new Claude Code session continuing engineering work on this project. For report-writing/citations/results material, use `CONVERSATION_HANDOFF.md` instead. Current as of 2026-10-21.

## What this project is

A final-year project: a mechanistic digital-twin simulation of two clinical emergencies (septic shock / norepinephrine, hypertensive emergency / sodium nitroprusside), each with a deterministic PD (not PID) expert controller as the sole actuator and three advisory-only ML models, a real ESP32-S3 syringe-pump prototype controllable over wired USB-serial or wireless WiFi, a PyQt6 desktop app (`native_app.py`) tying simulation + hardware + telemetry together, and a full cloud remote-monitoring dashboard built by a second developer on Firebase/Firestore, live at **infusion-d.web.app**.

## Current state — verified working

**Firmware** (`firmware/pump_firmware/pump_firmware.ino`):
- Wired (`Serial0`) and wireless (WiFi TCP) control, both through the same `handleSerialLine()` parser. Confirmed on real hardware.
- New commands giving wired/wireless parity with keypad dosing: `SET_WEIGHT_DOSE:<kg>,<value>,<MLHR|UGKGMIN>` and `SET_VTBI_PROGRAM:<rate>,<vtbi_ml>`. Deliberately don't set `manualModeActive=true` (would exempt them from the watchdog).
- A battery-only reboot loop was root-caused (WiFi TX current spikes browning out the regulator) and fixed: `WiFi.setTxPower(WIFI_POWER_8_5dBm)` plus a persisted runtime WiFi on/off setting (`bool wifiEnabled`, NVS-backed, keypad Settings menu `B` → `2`) so WiFi can be disabled for battery-only runs without reflashing.
- Not hardware-validated: `RETRACT_SPEED_ML_HR=250` (raised at user request, untested at that speed on the real motor) and the battery voltage calibration factor (still a no-op `1.0`, needs a real multimeter reading).

**Desktop app** (`native_app.py`):
- Wired/wireless pump control UI + Manual Pump Dosing UI (weight-dose, Rate+VTBI).
- `clinician_dashboard.py` — separate local-network-only dashboard, still functional, superseded in practice by the Firebase one.
- Adafruit IO integration exists but is unconfigured (`ADAFRUIT_IO_USERNAME`/`KEY` still placeholders) — optional, not blocking anything.
- Firebase remote dashboard integration — confirmed working end-to-end from source (`python native_app.py`): real simulated vitals, real hardware status (E-STOP/limits/battery) when a pump is connected, and session lifecycle all verified landing in Firestore and rendering live on the dashboard.

**Packaging**: `SmartInfusionAI.spec` was fixed to bundle `remote_dashboard/telemetry_publisher.py` (it's imported via a runtime `sys.path.insert()`, which PyInstaller's analyzer doesn't follow without `pathex=['remote_dashboard']` + an explicit `hiddenimports` entry). Rebuilt `.exe` at `dist/SmartInfusionAI/SmartInfusionAI.exe` (2026-10-20 11:14), build completed clean.

## Open items

- Dashboard access/roles (team only / clinician viewer / other) — owner's decision, not engineering.
- Old Supabase artifacts in `remote_dashboard/` (`supabase_schema.sql`, `ingest_function/`, `frontend/`) — superseded by Firebase, not yet deleted.
- Network-pull/reconnect test (dashboard should go Stale within ~6s, then resume on reconnect) — not yet run.
- Firmware's on-device `wifiEnabled` state is unknown without checking the physical board.
- `model_version`/`ai_prediction_horizon_s`/`ai_advisory`/real `controller_status` fields — supported by `build_telemetry_payload()` but not yet populated by `native_app.py`'s call site. Low priority.

## File map (recent additions)

| File | Purpose |
|---|---|
| `CONTROLLER_AND_AI_MECHANISM.md` | Both PD controllers' math/gains, both AI models' architecture. |
| `TELEMETRY_FIELD_MAP.md` | What data is real vs. simulated — accurate, but written during the Supabase phase; pipeline-specific parts are stale, use `remote_dashboard/README.md` instead. |
| `remote_dashboard/README.md` | Current dashboard gateway integration state. |
| `remote_dashboard/telemetry_publisher.py` | The real Firebase-edition file from the dashboard developer — device-account auth, Firestore REST writes, retry/backoff. Don't regress to the old Supabase version. |
| `remote_dashboard/simulate_device.py` | Synthetic-data test tool, no pump/native_app.py needed (needs `remote_dashboard/.env`). |
| `SmartInfusionAI.spec` | Includes `remote_dashboard` in `pathex`/`hiddenimports` — keep this when editing the spec later, or the packaged `.exe` silently loses the remote-dashboard feature. |
