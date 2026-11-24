# Telemetry Field Map — Audit for Remote Dashboard (M0)

Written for whoever builds `AI_Syringe_Infusion_Pump_Remote_Dashboard_Developer_Specification.pdf`'s dashboard. This directly answers that spec's **Section 15, "Decisions to confirm during the code audit"**, and its **M0: Audit** milestone ("Field map documents each available variable, source and units") — so M0 is effectively done; start at M1.

Scope note, confirmed with the project owner: **this release is monitoring-only.** The dashboard displays E-STOP/limit-switch/all other status live — it never sends anything back to the pump. That matches the spec's own non-negotiable principle (`Section 2`) and its acceptance checklist item on having no control path. Build accordingly — no command/write path to the ESP32 or `native_app.py` is needed for v1.

---

## Answers to Section 15's audit questions

**Does the desktop app receive ESP32 telemetry over USB serial, Wi-Fi, or both?**
Both. `native_app.py`'s `PumpLink` class supports a wired serial transport (`connect()`) and a wireless TCP transport (`connect_wifi()`) to the same firmware, selected by a Wired/Wireless toggle in the sidebar. Both use the identical text-line protocol — see below.

**What exact telemetry fields are already available in the firmware/parser?**
The firmware's `GET_STATUS` command (`pump_firmware.ino`, `printStatusLineTo()`) returns one comma-separated line with these fields — this is the complete, current set of what the hardware itself knows:

| Field | Type | Meaning |
|---|---|---|
| `mode` | enum string | `MANUAL` \| `AUTONOMOUS` \| `UNSELECTED` |
| `rate` | float, mL/hr | Commanded flow rate |
| `running` | 0/1 | Is the motor currently supposed to be turning |
| `estop` | 0/1 | **E-STOP latched** — this is your E-STOP status field |
| `home` | 0/1 | **HOME limit switch active** |
| `end` | 0/1 | **END limit switch active** |
| `battery_v` | float, volts | Smoothed battery voltage (EMA-filtered) |
| `battery_pct` | int, % | Derived from a voltage-to-percent curve, not independently calibrated — see `constants.py`'s own calibration note before treating this as precise |
| `vtbi_active` | 0/1 | Is a Rate+VTBI or Volume+Time program currently running |
| `vtbi_target_ml` | float | VTBI target volume |
| `vtbi_delivered_ml` | float | **Estimated** delivered volume (rate × elapsed time accounting — not an independent flow sensor, see below) |
| `cal_active`, `cal_steps_done`, `cal_steps_target` | — | Calibration jog state, bench-use only |
| `retract_active`, `retract_steps_done` | — | Syringe-reload jog state |
| `sound_enabled` | 0/1 | Keypad-sound mute setting |
| `wifi_enabled` | 0/1 | Persisted WiFi on/off setting (added after a real battery-brownout fix — see `pump_firmware.ino`'s own comments) |
| `position` | int, steps | Raw stepper position |

This is a flat line, not JSON (`STATUS,mode=...,rate=...,...`) — `native_app.py` doesn't currently parse this into a dict anywhere; it's only used for raw diagnostic display. **Whoever builds the publisher will need to add a small parser for this line**, or better, add a new firmware/app-side path that emits it as structured JSON directly, matching the spec's telemetry contract.

The **wired/wireless control protocol** itself (separate from `GET_STATUS`) is: `SET_RATE:<mL/hr>`, `SET_WEIGHT_DOSE:<kg>,<value>,<MLHR|UGKGMIN>`, `SET_VTBI_PROGRAM:<rate>,<vtbi_ml>`, `STOP`, `CAL:<steps>`, `RETRACT` — **none of these are needed for the monitoring dashboard**, listed here only so it's clear what exists vs. what this release intentionally doesn't touch.

**Are MAP and heart rate live sensor inputs, manually entered values, or digital-twin simulation outputs?**
**Always digital-twin simulation outputs.** There is no real patient and no real MAP/HR/SpO2 sensor anywhere in this project — `native_app.py` runs a mechanistic physiology simulation (`digital_twin.py` / `hypertensive_twin.py`) and that's the sole source of every vital sign. **`data_source` must always be `"simulation"` for these fields, never `"hardware"`**, per the spec's own truthful-labeling principle. The only fields that come from real hardware are the `GET_STATUS` ones in the table above (battery voltage, limit switches, E-STOP, commanded rate, stepper position) — those genuinely are live from the physical pump when connected.

**Where do the Random Forest models run, and what exact output names/scales do they produce?**
They run inside `native_app.py` itself (Python, on the laptop — `joblib.load()`'d `.pkl` files under `models/`), not on the ESP32. Per-tick output fields (from `SimEngine.step()` / `HypertensiveSimEngine.step()`):

| Field | Type | Meaning |
|---|---|---|
| `AI_Action` | string | Imitation classifier's predicted mode label (`Increase`/`Decrease`/`Maintain`/`Emergency`) — advisory only |
| `AI_SuggestedFlow` | float, mL/hr | Imitation regressor's reconstructed absolute flow (`FlowRate + predicted_delta`, clamped `[0, max_flow]`) — advisory only |
| `RiskScore` | float, **0–1** | Risk classifier's `predict_proba()[:,1]` — genuinely a 0–1 probability, safe to document as such per the spec's `ai_risk_score` field rule |
| `ActionAgrees` | bool | Whether `AI_Action` matches the real controller's chosen mode |
| `Alert` | string/null | Set when the AI disagrees with the controller — display-only |

**The thing actually driving the pump is never the AI** — `ControllerMode` / `ControllerFlow` (the deterministic PD controller's own output, see `CONTROLLER_AND_AI_MECHANISM.md`) is what's sent to `send_rate()`. Map `commanded_flow_ml_hr` in the spec's schema to `ControllerFlow`, not `AI_SuggestedFlow`.

**Is delivered volume measured independently or estimated?**
**Estimated**, on both the firmware (`deliveredVolumeML`, accumulated as `rate × elapsed_time`) and simulation side — there is no independent flow sensor anywhere in this build. Always populate the spec's `estimated_delivered_volume_ml` field, never `delivered_volume_ml` (which the spec reserves for an independently-measured value).

**How is the current desktop event log stored?**
In-memory only (`self.engine.history` — a Python list, lost on close) plus an optional one-shot "Save Session Report (.txt)" export. **There is no existing database** — the spec's `telemetry`/`device_events` tables would be genuinely new persistence, not a migration of something that already exists.

**What is the intended authorised-user model?**
Not yet decided — this is a real open question for the project owner, not something I can answer from the code. Worth settling before `M2: Supabase foundation` (RLS policies need to know this).

---

## Full simulation/vitals field reference (septic mode)

From `SimEngine.step()` (`native_app.py`), one dict per simulated tick — all `data_source: "simulation"`:

`Time`, `MAP`, `MeasuredMAP` (noisy — this is what a real monitor would show, use this one for display, not `MAP`), `TargetBP`, `InTargetRange`, `HR`, `SpO2`, `ControllerMode`, `ControllerFlow`, `ClinicalExplanation`, `AI_Action`, `AI_SuggestedFlow`, `FlowDivergence`, `ActionAgrees`, `Alert`, `StabilityScore`, `Confidence`, `ShockIndex`, `PlasmaConcentration`, `DrugEffect`, `SVR`, `CO`, `RiskScore`, `InfusionComplete`, `HRAdvisory`, `TimeInTargetMin`.

## Full simulation/vitals field reference (hypertensive mode)

From `HypertensiveSimEngine.step()` — **mostly the same shape, with real differences, not just a renamed copy**:
- `TargetBP` does not exist here — the equivalent is **`MAP_Floor`** (a *moving* value, not a fixed setpoint — see `CONTROLLER_AND_AI_MECHANISM.md` §2.1). A frontend component expecting one fixed `TargetBP` field across both modes will break on this mode.
- `StabilityScore`/`Confidence` are the literal string `"N/A"` (no equivalent computed on this side) — not a number, handle as a string or map to `null`.
- `ShockIndex` is always `0.0` here — not a real hypertensive-mode concept, kept only for dict-shape parity. Don't display it for this mode.
- `InfusionComplete` is always `False` (no VTBI concept on this side), `HRAdvisory` always `False`, `TimeInTargetMin` always `0.0` — all kept for shape parity, not meaningful here.
- Two fields that only exist on this side: **`CyanideRateRisk`** (bool) and **`CyanideBufferExceeded`** (bool) — the deterministic toxicity safety overlay's state (`CONTROLLER_AND_AI_MECHANISM.md` §2.6). Worth surfacing on the dashboard as their own status indicator, same tier as E-STOP/limit switches.

**Practical implication for the frontend**: don't assume one fixed schema across both therapy modes. Either branch on `infusion_mode` (`"NE"` vs `"SNP"`) when mapping fields, or have the Python publisher normalize both into one superset shape with `null` for whichever mode's fields don't apply — the latter is cleaner and matches the spec's own "fail visibly, don't fabricate" principle better than silently sending `0.0`/`"N/A"` placeholders to the cloud as if they were real values.

---

## What this means for the spec's data flow

```
ESP32 (GET_STATUS: estop/home/end/battery/rate/position/...)
   │
   ▼
native_app.py  ──┬── PumpLink (wired or wireless) ── hardware fields above
                 └── SimEngine / HypertensiveSimEngine ── simulation vitals + AI fields above
   │
   ▼  (NEW — doesn't exist yet, this is M4's actual work)
background publisher → ingestion API → Supabase → dashboard
```

Nothing in the existing codebase currently sends anything to a remote server — `clinician_dashboard.py` (a separate, local-network-only tool built earlier, see its own docstring) is the closest existing precedent, but it pushes raw UDP to `localhost` only and has no cloud/auth/persistence layer at all. **M4's publisher is genuinely new work**, not an extension of that file, though reading it first will save time since it already solved "how do I get a snapshot out of `_on_timer()` without blocking the GUI."
