# How the Expert Controller and the AI Models Actually Work

A step-by-step technical reference for the Methodology/Results chapters: the exact control law, every gain and threshold, the unit-conversion chain, and how the deterioration-prediction AI differs from the dosing AI. All values below are read directly from the current code (`expert_controller.py`, `hypertensive_expert_controller.py`, `pharmacokinetics.py`, `generate_dataset.py`, `train_risk_model.py`, `train_ai.py`) as of 2026-10-12, not from memory or an earlier draft.

---

## 0. The one-line answer: which parameter are we controlling?

**MAP (mean arterial pressure) is the controlled variable. Flow rate is the manipulated variable.**

In standard control-theory language:
- **Process variable (what we actually care about and want to reach/hold a target)**: MAP, in mmHg. This is the patient's blood pressure — the clinical outcome.
- **Setpoint**: a target MAP value (septic side: a fixed clinician-chosen target, default 65 mmHg; hypertensive side: a *moving* safety floor that changes over time — see Part 2 below).
- **Manipulated variable (what the controller actually changes to drive the process variable toward the setpoint)**: the pump's flow rate, in mL/hr.
- **Error**: `target − MAP` (or `MAP − floor` on the hypertensive side — the sign is flipped because the clinical goal is opposite, see Part 2).

So the controller never "controls flow rate" in the sense of flow rate being the goal — flow rate is the *lever*. MAP is the *goal*. The controller's entire job, every tick, is: "look at the current error between MAP and its target, and decide how much to move the flow-rate lever to shrink that error" — this is a textbook closed-loop feedback control problem, and the two controllers in this project (`expert_controller.py`, `hypertensive_expert_controller.py`) are both proportional-derivative (PD) controllers implementing exactly that loop.

```
          target MAP (setpoint)
               │
               ▼
      ┌─────────────────┐        flow rate (mL/hr)        ┌──────────────┐
error │  PD control law  │ ───────────────────────────────▶│  patient     │
──────▶  (this file,     │        (manipulated variable)   │  physiology  │
 MAP  │   Parts 1 & 2)   │                                  │  (digital    │
      └─────────────────┘                                  │   twin)      │
               ▲                                             └──────┬───────┘
               │                     measured MAP (feedback)        │
               └─────────────────────────────────────────────────────┘
```

---

## 1. Septic shock controller (`expert_controller.py`) — step by step

### 1.1 Inputs read every tick

```python
bp      = patient.bp          # true MAP, mmHg — NOT the noisy measured value (see Part 4)
hr      = patient.hr
spo2    = patient.spo2
flow    = patient.flow_rate   # the CURRENT commanded rate, mL/hr — this tick's starting point
target  = patient.target_bp   # setpoint, mmHg (default 65, clinician-adjustable 65–85)
weight_kg = patient.weight_kg
```

**Important, and worth stating explicitly in a viva**: the controller reads the simulation's *true* MAP, not the noisy sensor-simulated `MeasuredMAP` the AI models see. This is a deliberate, verified design choice (confirmed directly against the code this session) — it means the controller is not fighting sensor noise, while the AI models' performance numbers *are* earned against noisy inputs. That asymmetry is why the imitation classifier's 86.7% accuracy (Part 3) is a meaningful result and not a trivial one.

### 1.2 The error and the trend

```python
bp_error = target - bp                                   # positive = still below target, needs MORE drug
bp_trend = patient.smoothed_bp - patient.previous_smoothed_bp   # mmHg change over the last tick, SMOOTHED
```

`bp_trend` uses a smoothed MAP (an exponential moving average updated every tick), not the raw single-tick difference — this avoids the derivative term reacting to one noisy sample instead of a genuine trend. This is the same rationale a real PID implementation uses a filtered derivative for.

### 1.3 The three-tier decision structure

The controller is **not** a single PD formula applied uniformly — it has three tiers, each with a distinct clinical justification:

**Tier A — genuine emergency** (`bp < EMERGENCY_MAP_MMHG = 50.0`):
```python
flow = min(flow + step_to_flow(MAX_STEP_MCG_KG_MIN), max_flow)
```
Escalate by the single largest step the protocol allows, clamped at the pump ceiling. No PD math here deliberately — below 50 mmHg is organ-hypoperfusion territory, and there is no such thing as "correcting hypotension too fast" with norepinephrine, so the controller doesn't taper, it just pushes to the safe maximum.

**Tier B — significantly overshooting** (`bp > target + deadband + BIG_OVERSHOOT_MARGIN_MMHG`, i.e. more than 18 mmHg over target):
```python
flow = max(flow - step_to_flow(MAX_STEP_MCG_KG_MIN), min_flow)
```
Cut hard, same magnitude step, opposite direction.

**Tier C — everything else (the actual PD law, the tier that matters for ~95%+ of ticks)**:
```python
error      = target - bp          # positive = below target
derivative = -bp_trend            # positive = MAP still falling (moving away from target if below it)
raw_step_mcg_kg_min = KP * error + KD * derivative
step_mcg_kg_min = clip(raw_step_mcg_kg_min, -MAX_STEP_MCG_KG_MIN, +MAX_STEP_MCG_KG_MIN)
flow = clip(flow + step_to_flow(step_mcg_kg_min), min_flow, max_flow)
```

This is the actual proportional-derivative control law. Two terms:
- **Proportional term** (`KP * error`): the further MAP is from target, the bigger the step — this is what makes the controller converge at all.
- **Derivative term** (`KD * derivative`): reacts to the *rate of change*, not just the current gap — this is what lets the controller start tapering the dose *before* MAP actually crosses the target, rather than waiting to overshoot first and correcting afterward. This is the anticipatory part of "PD."

### 1.4 The gains — exact values and why

| Symbol | Value | Meaning | Status |
|---|---|---|---|
| `KP` | `0.002` mcg/kg/min per mmHg of error | Proportional gain | `[ASSUMPTION]` — tuned so that ~25 mmHg of error alone saturates the step at `MAX_STEP_MCG_KG_MIN`. No literature value exists for a PD gain on this exact drug/unit combination (nothing in the norepinephrine titration literature is expressed this way) — this is an engineering tuning choice, stated as such. |
| `KD` | `0.15` mcg/kg/min per (mmHg/tick) of trend | Derivative gain | `[DERIVED/RETUNED]` — **this one has a real empirical story worth including in your Methodology.** The original value (carried over from an earlier version at `0.025`, a 12.5:1 ratio to `KP`) was tuned for a *coarser* control tick rate. At this simulation's actual per-second tick rate, `bp_trend` (mmHg per ONE second) is tiny even during a fast MAP rise, so the old `KD` gave the anticipatory/braking term almost no real weight relative to the proportional term — verified empirically: the controller kept commanding "increase" right up to and past the target before the error term itself flipped sign, producing measured overshoot of several mmHg (peak MAP 65–73 vs. a target of 65, across multiple seeds/profiles). Raising `KD` 6× (to 0.15) brought peak overshoot down to ~0–2 mmHg on the same test seeds, with no new undershoot or mode-chattering introduced. **This retuning, with the before/after numbers, is itself a legitimate empirical result for your report** — it demonstrates the derivative term is load-bearing, not decorative. |
| `MAX_STEP_MCG_KG_MIN` | `0.05` mcg/kg/min per tick | Per-tick step ceiling (saturates both P and D terms combined) | `[CITED, low end]` — matches the standard ICU norepinephrine titration increment cited in the literature. Applying it every simulated *second* rather than every 5–15 minutes (the real clinical cadence) is an explicit, stated time-compression simplification of this simulation, not a claim about real bedside practice. |
| `EMERGENCY_MAP_MMHG` | `50.0` mmHg | Absolute "genuinely critical" threshold, triggers Tier A | `[ASSUMPTION]` — reasonable clinical staging below the 65 mmHg target, not independently cited to a specific guideline value. |
| `SAFE_MARGIN_BELOW_TARGET_MMHG` | `5.0` mmHg | Used only in the wording of the clinical explanation string ("below safe range" vs "below target") | `[ASSUMPTION]`, cosmetic/explanatory, doesn't change the dose math. |
| `BIG_OVERSHOOT_MARGIN_MMHG` | `15.0` mmHg | Distance above `target + deadband` that triggers Tier B | `[ASSUMPTION]` — operationally motivated hard-tier boundary. |
| `TARGET_DEADBAND` | `3.0` mmHg | Width of the "Maintain" zone around target (`|bp - target| <= 3`) | `[ASSUMPTION]` — hysteresis width, prevents the displayed *mode label* (Increase/Decrease/Maintain) from chattering on sub-mmHg noise. Does **not** gate the PD math itself — the PD law runs continuously in Tier C regardless of whether you're inside or outside the deadband; the deadband only changes which label/explanation string gets attached to the same computed step. |

### 1.5 From mcg/kg/min to mL/hr — the unit-conversion chain

The PD law above computes everything in **mcg/kg/min** (clinical dosing units), never in mL/hr directly — this matters because it's what lets the exact same step size mean the same clinical thing regardless of a patient's weight. The conversion to the pump's actual unit happens in exactly one place, applied twice (once to convert the PD step, once inside `applyRate()`-equivalent clamping):

```python
def step_to_flow(step_mcg_kg_min):
    return mass_rate_to_flow_ml_hr(step_mcg_kg_min * weight_kg)

# pharmacokinetics.py:
def mass_rate_to_flow_ml_hr(mass_rate_mcg_min):
    return mass_rate_mcg_min * 60.0 / DRUG_CONCENTRATION_MCG_ML
```

So the full chain for one PD step is:
```
step_mcg_kg_min  ×  weight_kg               → mcg/min        (patient-specific mass rate)
mcg/min  ×  60  ÷  DRUG_CONCENTRATION_MCG_ML → mL/hr          (pump flow rate)
```
`DRUG_CONCENTRATION_MCG_ML = 16.0` is the standard commercial norepinephrine premix concentration (4 mg in 250 mL D5W) — `[CITED]`, cross-validated against multiple independent sources (see `REFERENCES.md` item 10).

### 1.6 The hypoxia overlay (label only, never touches flow)

```python
if spo2 < 90 and mode != "Emergency":
    mode = "Emergency"   # relabels for alerting only
    # flow is NOT modified here
```
Worth stating precisely in a report because it's a common point of confusion: low SpO2 does **not** change the dose. Norepinephrine cannot fix oxygenation — it's a pressor, not a respiratory drug — so this overlay exists purely to flag the patient prominently for clinical review. The actual flow is governed entirely by the three MAP-driven tiers above, with SpO2 never entering the dosing math at all.

---

## 2. Hypertensive emergency controller (`hypertensive_expert_controller.py`) — what's genuinely different, and why

This is **not** the same controller with a sign flipped — the clinical requirement is structurally different, and the code reflects that.

### 2.1 The setpoint is a moving floor, not a fixed target

On the septic side, `target` is fixed for the whole episode. On the hypertensive side, the controller tracks `patient.map_floor_mmhg` — a value that itself changes over simulated time, computed by a separate staged-reduction schedule (`map_reduction_floor_mmhg()` in `hypertensive_twin.py`): no more than ~25% reduction in hour 1, further reduction toward a ~160/100 mmHg-equivalent band by hour 6, gradual normalization by hour 24–48. This schedule is derived from the cited NHLBI/ACC-AHA staged blood-pressure-reduction guidance for hypertensive emergencies (`REFERENCES.md` item 12). The controller's job is **never** "push MAP toward the floor as fast as possible" — pushing MAP *below* the floor is itself a safety violation (risks cerebral/coronary/renal hypoperfusion from overwhelming autoregulation), so undershooting the floor is treated with the same severity as a critical-high reading, not merely as "slightly off target."

### 2.2 The error sign is flipped, for a real clinical reason

```python
bp_error = bp - floor   # positive = still ABOVE the current safe floor (needs MORE drug)
```
On the septic side, positive error meant "too low, give more drug." Here, positive error means "too high, give more drug" — because nitroprusside *lowers* MAP, the direction of "needs more drug" is reversed relative to norepinephrine. This is the one place the sign genuinely has to flip, and it flows through correctly into the PD law below.

### 2.3 No fast-ramp emergency tier — deliberately

This is the most clinically important structural difference, and worth a full sentence in your report: **there is no equivalent of the septic controller's Tier A (escalate hard regardless of PD math)**. The FDA label's own protocol for nitroprusside is "start low, titrate every few minutes," with no emergency exception — because the drug's near-instant onset/offset (plasma half-life ~2 minutes) means a fast ramp is actively dangerous at *any* starting MAP, not just safe-at-low-MAP the way norepinephrine's escalation is. This was verified empirically during development, not just reasoned about: an earlier, larger step size (proportioned the same way relative to nitroprusside's max dose as the septic controller's step is to its own max dose) caused the controller to breach the safety floor by 19–48 mmHg within the same simulated minute, in exactly the seeded tests used to validate it. The step size was cut by more than 10× in response (see `MAX_STEP_MCG_KG_MIN` below) and re-verified to track the floor within its own deadband instead of blowing through it.

### 2.4 The PD law itself (Tier structure: only two tiers, not three)

**Hard decrease tier** — only triggers when *genuinely undershooting* the floor:
```python
if bp < floor - deadband - FLOOR_BREACH_MARGIN_MMHG:   # more than 8mmHg below floor
    flow = max(flow - step_to_flow(MAX_STEP_MCG_KG_MIN), min_flow)
```

**PD law** — everything else:
```python
error      = bp - floor        # positive = still needs more drug
derivative = bp_trend          # positive = MAP still rising -> lean toward more drug
raw_step_mcg_kg_min = KP * error + KD * derivative
step_mcg_kg_min = clip(raw_step_mcg_kg_min, -MAX_STEP_MCG_KG_MIN, +MAX_STEP_MCG_KG_MIN)
flow = clip(flow + step_to_flow(step_mcg_kg_min), min_flow, max_flow)
```
Structurally identical PD math to the septic side (same proportional + derivative shape, same `step_to_flow()` unit conversion), just against a moving floor instead of a fixed target, and with the sign convention from §2.2.

### 2.5 The gains — exact values and why they're smaller than the septic side

| Symbol | Value | Septic equivalent | Why smaller |
|---|---|---|---|
| `KP` | `0.0015` mcg/kg/min per mmHg | `0.002` | `[ASSUMPTION, UNVALIDATED]` — ported from the septic controller's *structure*, not independently retuned against measured overshoot the way `KD` below was. Flagged explicitly as a follow-up: a proper seeded-sweep validation (same kind `KD` received on the septic side) hasn't been done for this gain yet. |
| `KD` | `0.02` mcg/kg/min per (mmHg/tick) | `0.15` | Same "ported, not independently validated" caveat as `KP`. |
| `MAX_STEP_MCG_KG_MIN` | `0.01` mcg/kg/min per tick | `0.05` | `[ASSUMPTION, EMPIRICALLY CONSTRAINED]` — this is the one gain on this side with a real empirical justification: cut >10× from an initial value after it was shown to breach the safety floor by 19–48 mmHg (§2.3). Unlike the septic side's step size, there's no clean literature figure for a per-tick nitroprusside titration increment to anchor to — the FDA label only says "titrate every few minutes" with no numeric step — so this value is explicitly provisional, constrained by what empirically kept the floor-tracking safe rather than derived from a cited source. |
| `FLOOR_BREACH_MARGIN_MMHG` | `5.0` mmHg | — | `[ASSUMPTION]` — same role/magnitude convention as the septic side's `BIG_OVERSHOOT_MARGIN_MMHG`. Combined with `TARGET_DEADBAND` (3.0, shared with the septic side), the actual "real breach" threshold used everywhere in this project (the controller's own hard tier, the `WillBreachFloor` AI label, and the Results-chapter breach statistics) is **more than 8 mmHg below the floor** — a looser "any `MAP < floor`" check over-counts harmless sub-mmHg control noise as breaches (flags ~29.6% of all ticks), so this stricter definition is the one to use/cite consistently.

**Overall takeaway worth stating in your report**: the hypertensive controller is intentionally *more conservative* than the septic one — smaller step size, no fast-ramp tier at all — because the clinical risk profile of "correcting too fast" is real for this drug and absent for the septic drug. The gain values reflect that asymmetry, not an oversight.

### 2.6 The cyanide toxicity overlay — a second, independent safety layer

Applied **after** the PD law, and can only ever *reduce* flow, never increase it:
```python
if patient.cyanide_buffer_exceeded:          # cumulative dose > FDA label's ~10 mg/kg buffering capacity
    flow = min_flow                           # full stop
elif patient.cyanide_rate_risk_active:        # cumulative dose past a lower threshold, AND current rate still high
    flow = min(flow, rate_cap_flow)            # capped, not stopped
```
This is deliberately a hard, deterministic constraint, not something the PD law or any AI model ever learns or influences — matching this project's stated design philosophy that toxicity management is a deterministic safety constraint, never an AI-learned behavior, same as the firmware's own watchdog/limit-switch layer sitting outside the PD/control logic entirely.

---

## 3. The AI dosing-imitation models (advisory only — never touch the pump)

These are **not** a second control law. They exist to answer a different question: "could a model learn to reproduce what the expert controller is already doing, from only the information a bedside monitor could realistically provide?" They are trained *after* the controller runs, on the controller's own output, and their predictions are displayed for comparison only — `expert_controller`/`hypertensive_expert_controller` remain the sole thing that ever sets `patient.flow_rate`, in both the simulation and the real pump firmware.

**Two separate models, two separate targets:**

- **Action classifier** (`RandomForestClassifier`): predicts the controller's *mode label* (`Increase`/`Decrease`/`Maintain`/`Emergency`) from the same noisy `Measured*` features the risk model uses (Part 4) — i.e., it tries to imitate *which tier/direction* the controller chose.
- **Flow regressor** (`RandomForestRegressor`): predicts a **delta**, `RecommendedFlow - FlowRate` (the *change* the controller made that tick), not the absolute flow rate itself. This was a deliberate redesign — an earlier version predicting absolute flow directly produced a model that could trivially "cheat" by learning something close to a copy of the previous tick's flow rate (since flow rate changes only a little per second), giving an artificially good-looking score without actually learning the controller's logic. Predicting the delta removes that shortcut; the absolute flow shown in the UI is then reconstructed as `FlowRate + predicted_delta`, explicitly clamped to `[0, max_flow]` since an unclamped delta prediction can occasionally go negative right after a large real correction.

Both models use validation-selected `max_depth` (grid search, scored on a held-out validation split, same discipline as the risk model in Part 4) — this is the same "validation set, not guessed hyperparameters" standard applied consistently across every trained model in this project.

---

## 4. The AI deterioration-prediction model (`risk_classifier.pkl`) — step by step

This is the model your question is really asking about by name ("how the AI predicts deterioration"), and it is architecturally a completely different task from Part 3 above — worth stating clearly, since it's easy for a reader to conflate "the AI" as one thing when there are three separate models doing three separate jobs.

### 4.1 The prediction task, precisely

> Given only the most recent rolling window of *noisy, bedside-realistic* measurements, will this patient's condition deteriorate within the next `LOOKAHEAD` simulated seconds?

```python
WINDOW    = 30   # ticks (seconds) of rolling history used to BUILD the input features
LOOKAHEAD = 30   # ticks (seconds) into the future checked to BUILD the label
```

### 4.2 How the label is actually constructed (this is the ground truth the model is trained against)

```python
# Septic side (generate_dataset.py):
will_deteriorate = any(m < EMERGENCY_MAP_MMHG for m in future_map) or any(s < 90 for s in future_spo2)
RISK_LABEL_COLUMN = "WillDeteriorate"
```
For every single simulated tick, the dataset generator looks **forward** `LOOKAHEAD` ticks into that same patient's own already-simulated future trajectory, and asks: does true MAP ever fall below 50 mmHg, or true SpO2 ever fall below 90%, at any point in that window? If yes, this tick is labeled `1` (a deterioration event is coming); if no, `0`. This is only possible to construct because the dataset generator has the whole simulated trajectory available ahead of time — the model itself, at prediction time, only ever sees the past/present rolling window, never the future the label was built from.

The hypertensive side uses the equivalent construction with its own moving floor (`WillBreachFloor`): MAP falling more than `FLOOR_BREACH_MARGIN_MMHG` below its own *future* floor at that future tick, OR the cyanide rate-risk flag turning on, within the same lookahead window.

### 4.3 What the model is actually allowed to see (features)

```python
FEATURE_COLUMNS = [
    "MeasuredMAP", "MeasuredHR", "MeasuredSpO2", "FlowRate",
    "WeightKg", "Age", "TargetBP",
    "MeasuredBP_Error", "MeasuredBP_Trend", "MeasuredShockIndex",
    "MeasuredPerfusionIndex", "MeasuredOxygenDeficit",
]
ROLLING_FEATURE_COLUMNS = [
    "RollMeanMAP", "RollStdMAP", "RollSlopeMAP",
    "RollMeanSpO2", "RollStdSpO2", "RollSlopeSpO2",
    "RollMeanHR", "RollSlopeHR",
]
RISK_FEATURE_COLUMNS = FEATURE_COLUMNS + ROLLING_FEATURE_COLUMNS   # 20 features total
```
Everything here is either a direct noisy sensor reading (`Measured*`), a simple derived clinical index computed from those noisy readings (shock index, perfusion index, oxygen deficit, BP error/trend), or a rolling statistic (mean/std/slope) over the last `WINDOW=30` ticks of that same noisy data. Explicitly **excluded**: `Profile`, `SAPS_II`, `ClearanceLMin`, `VdL`, `SVR`, `CO`, `DrugEffect`, `PlasmaConcentration` — these are simulation-internal "ground truth" values a real bedside monitor could never measure, and the code comments flag them explicitly as hidden ground truth, not model inputs. This is the thing to emphasize if asked "how do you know this isn't cheating" — the model is trained under the same observability constraint a real clinical deployment would face.

### 4.4 The rolling features — what they actually capture

`RollSlopeMAP` (and the HR/SpO2 equivalents) are the genuinely interesting features here: a linear slope fitted over the last 30 seconds of noisy MAP, i.e. a direct, computed measure of *trend*, independent of the controller's own trend calculation in Part 1/2. `RollStdMAP` captures how noisy/unstable the signal currently is. Together with the means, these rolling features are what let the model distinguish "MAP is low but stable" from "MAP is low and actively falling" — the latter being the actual deterioration signature the lookahead label is built to capture.

### 4.5 The model itself and how it was trained

```python
RandomForestClassifier(
    n_estimators=500,          # 150 during the validation search, 500 for the final fit
    max_depth=<validation-selected, grid [10, 15, 20]>,
    class_weight="balanced",    # compensates for deterioration being a rare event in the data
    random_state=42,
    n_jobs=-1,
)
```
- **Validation-based hyperparameter selection**: `max_depth` is chosen by fitting each candidate on the training set and scoring ROC-AUC on a completely separate validation set, picking whichever depth scores best — not guessed, not fixed by convention.
- **Final refit**: once `max_depth` is chosen, the final model is refit on **train + validation combined** (validation has already done its one job — selecting the hyperparameter — so folding it back in uses those patients' data for the final fit rather than discarding it). The **test set is never touched** until final evaluation.
- **`class_weight="balanced"`**: deterioration events are intentionally rare in the simulated population (the septic-side positive rate is only 2.9% on the final test set — see the earlier citation-audit work in this same project), so this reweights the loss function to stop the model from trivially predicting "never deteriorates" and still scoring high accuracy.

### 4.6 Why accuracy alone is the wrong metric here (and what's reported instead)

Because the positive class is rare, the model is evaluated primarily on:
- **Recall** — of the patients who genuinely do deteriorate, how many does the model actually catch? This is the clinically critical number: a missed deterioration event is the costly failure mode, far more than a false alarm.
- **ROC-AUC** — how well the model separates low-risk from high-risk patients across all possible decision thresholds, independent of where exactly the threshold is set.
- **Precision** — of the alerts the model raises, how many are real? Relevant for alarm-fatigue arguments in your Discussion.

Per-profile recall is also broken out separately (grouped by `Profile`), so you can show the model isn't just catching the "easy" severe cases while missing subtler ones in milder profiles.

---

## 5. How all of this fits together — the full picture

```
                         ┌─────────────────────────────────────────┐
                         │         DIGITAL TWIN PHYSIOLOGY          │
                         │   (true MAP, HR, SpO2, SVR, CO, ...)     │
                         └───────────────┬───────────────────────────┘
                                         │
                ┌────────────────────────┼─────────────────────────────┐
                │                        │                             │
         true MAP (feedback)      noisy MeasuredMAP/HR/SpO2      noisy MeasuredMAP/HR/SpO2
                │                        │                             │
                ▼                        ▼                             ▼
   ┌─────────────────────┐   ┌───────────────────────┐   ┌───────────────────────────┐
   │  EXPERT CONTROLLER   │   │  AI IMITATION MODELS   │   │  AI RISK/DETERIORATION     │
   │  (Parts 1-2 above)   │   │  (Part 3 above)        │   │  MODEL (Part 4 above)      │
   │                      │   │                        │   │                            │
   │  PD law, deterministic│   │  Predicts the          │   │  Predicts "will MAP<50 or  │
   │  — the ONLY thing     │   │  controller's OWN mode │   │  SpO2<90 happen in the     │
   │  that ever sets       │   │  + flow delta, for     │   │  next 30 ticks?" — a       │
   │  patient.flow_rate    │   │  DISPLAY/comparison    │   │  forward-looking, label    │
   │  or drives the real   │   │  only                  │   │  built from the FUTURE     │
   │  pump motor           │   │                        │   │  trajectory, not a copy    │
   └──────────┬───────────┘   └────────────────────────┘   │  of the controller's logic │
              │                                              └────────────────────────────┘
              ▼
       flow rate (mL/hr)
      sent to the real pump
      (SET_RATE: / SET_WEIGHT_DOSE: / SET_VTBI_PROGRAM:)
```

The one sentence that ties the whole architecture together, and the one most worth having ready for a viva question: **the AI never actuates anything** — the imitation models are graded on how well they *predict what the deterministic controller already decided*, and the risk model is graded on how well it *predicts a future physiological event defined independently of the controller's own logic*. Both are advisory/monitoring layers sitting beside a control loop whose actuator is pure, auditable, deterministic PD control — which is also precisely why this project can report a clean mathematical control law with exact gains (this document) as well as AI performance metrics (Parts 3-4, and `CONVERSATION_HANDOFF.md` §8.1/8.2/8.5): they are being held to different, appropriate standards because they do different jobs.
