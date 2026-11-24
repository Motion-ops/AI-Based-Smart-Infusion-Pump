# Methodology Source Data — answers to "Everything I need from you"

This answers, item by item, the checklist the other Claude session asked for, sourced directly from the current project code (paths/line references given so you or that session can verify). Every `[code]`-tagged item is answered below with its actual value. Items that are **not in the code** (physical hardware facts, photos, physical test status) are clearly marked **NOT AVAILABLE — needs your input** rather than guessed.

**New (2026-09-29): controller performance stats and confusion matrices, computed directly from the held-out test sets, are now in `CONVERSATION_HANDOFF.md` Part 8.5** — per-profile time-in-band/time-to-target/peak-overshoot/never-reached-target (septic) and floor-breach-rate/time-near-floor/peak-HR (hypertensive), plus raw confusion-matrix counts for all four classifiers (both action classifiers, both risk classifiers). Not duplicated here to avoid drift between two copies - go to that document for these numbers. One correction embedded there worth flagging: a naive "MAP < floor" check way over-counts harmless sub-mmHg control noise as "breaches" (29.6% of ticks) - the real, clinically/model-relevant breach definition requires MAP more than 8 mmHg below the floor (matching the controller's own safety-tier logic and the `WillBreachFloor` label's construction), which gives a much more sensible 0.53% overall / 8 of 200 patients.

**One real discrepancy remains open — read this before anything else, since it may affect a number already in your draft report:**

1. **Sodium nitroprusside's `NITROPRUSSIDE_MAX_DOSE_DURATION_MIN = 10.0` (minutes) constant exists in `hypertensive_constants.py` but is not actually enforced anywhere in `hypertensive_expert_controller.py`.** It's a cited FDA-label value (max-rate dosing should never exceed 10 minutes) that was recorded but never wired into a timer check. If your methodology describes this as an implemented safety behaviour, that would currently be inaccurate — either flag it as a stated-but-unenforced limitation, or ask me to implement the actual timer if you want it working before submission.

**Resolved (2026-09-27): the septic-shock (norepinephrine) side now has a proper patient-level 70/10/20 train/validation/test split, mirroring the hypertensive side's methodology exactly.** The earlier "60 unseen patients, not 300" discrepancy noted here previously is superseded: the dataset is now 300 total simulated patients (210 train / 30 validation / 60 test, 600 ticks each, independently seeded, zero patient-level leakage confirmed via `(seed, patient_id)` comparison), and `train_ai.py`/`train_risk_model.py` now select `max_depth` by scoring a small grid on the held-out validation set (accuracy for the action classifier, MAE for the flow regressor, ROC-AUC for the risk model) before a single untouched evaluation on test — previously these used fixed, untuned hyperparameters (`max_depth=15` by convention, not validated). Test stayed at 60 patients deliberately so any already-written test-set figures elsewhere in your report stay valid; if you had heard "300 unseen test patients" from an earlier source, this is very likely what that referred to (a 300-patient total dataset with a real split), just not implemented until now. See generate_dataset.py's `__main__` block and train_ai.py/train_risk_model.py for the actual code.

**Baroreflex HR fix (2026-09-27, same day, after the above)**: a real implementation bug was found and fixed, separate from the validation-set/delta-target work above. `constants.py` already documented that unbounded baroreflex deviation had previously caused simulated HR above 145 bpm in ~20% of a 60-patient sweep, and a cap (`BAROREFLEX_MAX_DEVIATION_MMHG=10`) was added to fix it — but `digital_twin.py` only ever applied that cap once MAP was already at/near target. During the acute pre-target phase it stayed fully uncapped, so an unlucky combination of sampled `hr_set`/`map_set` (both drawn independently from `HR_NORMAL_BPM=(60,100)` / `MAP_NORMAL_MMHG=(70,100)`, occasionally landing both near the high end) could drive HR past 130 bpm within the first couple of simulated minutes, well before MAP had recovered — reproduced directly in testing (a `Severe`-profile patient with `hr_set=97.7`, `map_set=97.0` hit HR=126.7 at the moment MAP first reached target). Fixed with a two-tier cap: a new `BAROREFLEX_MAX_DEVIATION_ACUTE_MMHG=25` applied during the acute/pre-target phase (still allows a strong, fast compensatory response — HR genuinely helps raise MAP via `compute_cardiac_output`, so this must not be too tight), tightening to the original `BAROREFLEX_MAX_DEVIATION_MMHG=10` once at/near target. Re-verified on the same worst-case patient: HR at target-crossing dropped from 126.7 to 114.9 bpm and no longer keeps climbing past that point. Because this changes how `MeasuredHR` is generated, the dataset was regenerated and all three septic models retrained a second time (same 210/30/60 split, same seeds).

**SpO2 recovery fix (2026-09-28, next day)**: user reported SpO2 freezing at 89.4% indefinitely in a live session even after MAP had stabilized. Root cause found in `physiology.py`'s `update_spo2` (shared by both clinical modes): it used a 3-way split - decline below 65-3=62 mmHg, recover above 65+3=68 mmHg, and hold perfectly STEADY in between (62-68 mmHg). That "steady" band was exactly the septic-shock clinical target zone (`expert_controller.py` holds MAP at 65±`TARGET_DEADBAND`=3 once treatment succeeds) - so a successfully treated patient, held right at target as intended, got SpO2 permanently frozen wherever it happened to be, only ever resuming recovery if MAP later drifted above 68 via the separate, much slower illness-recovery mechanism. Reproduced directly: a test patient held at MAP=65.0-67.9 mmHg had SpO2 frozen at 87.73% for 900+ simulated seconds. This directly contradicts what an earlier note in this document said (that continued SpO2 decline was "not a bug" - that assessment covered the genuinely-refractory case correctly, but missed this separate freeze-at-target case). Fixed by collapsing the 3-way split to 2-way: reaching or exceeding the perfusion threshold (with a deadband kept only on the lower/decline side, to avoid noise-flicker right at that one boundary) now starts recovery immediately. Re-verified on the same test patient: SpO2 now reaches 100% within seconds of MAP crossing into target, instead of freezing at 87.73%. This function is shared with the hypertensive twin, but hypertensive patients rarely linger in the narrow 62-68 mmHg band (they present far above it), so that side's dataset was not regenerated. The septic dataset was regenerated and all three models retrained a third time.

**Updated metrics after all three fixes (validation-set methodology + delta-target regressor + baroreflex HR cap + SpO2 recovery fix; real numbers, not projected, current as of 2026-09-28):**
- Action classifier: validation-selected `max_depth=10` (val accuracy 0.8528), held-out test accuracy **86.7%** (macro F1 0.84; per-profile accuracy Healthy 90.4%, Mild 86.9%, Moderate 81.3%, Severe 82.3%).
- Flow regressor (delta target): validation-selected `max_depth=6`, `FlowRate` importance 64.9% (`MeasuredMAP` second at 19.9%). **Headline (delta-scale, honest) metrics**: R²=0.762, MAE=0.1467 mL/hr — beats the naive "predict zero change" baseline of 0.2442 mL/hr. **Don't headline** the reconstructed absolute-flow figures (R²=0.9999, MAE=0.1467 mL/hr, RMSE=0.6938 mL/hr) - they're dominated by the current-flow input, not by learned signal; report them only as a secondary, clinically-readable view alongside the delta numbers.
- Risk model (`WillDeteriorate`): validation-selected `max_depth=20` (val ROC-AUC 0.9998), held-out test accuracy 99.7%, precision 0.938, recall 0.961, F1 0.949, ROC-AUC **>0.999** (raw value 0.9998 - report as ">0.999" rather than a rounded "1.000", which would overclaim a literally-perfect classifier). Positive class rate: train 7.5%, val 5.0%, test 2.9% (this rate keeps dropping across each retrain as the physiology fixes make treatment succeed more often and more completely - a real, physiology-driven shift, not noise).

(Earlier metrics reported the same week — after the validation-set fix alone, then after the baroreflex fix — are both superseded by the numbers above. Use these for your report.)

(The immediately-preceding numbers reported earlier the same day — classifier 85.6%/depth 20, regressor delta R²=0.726/depth 10, risk model 98.9%/precision 0.884 — were from the retrain done *between* the validation-set fix and the baroreflex fix, using the since-corrected HR dynamics' predecessor. They are superseded by the numbers above; use these for your report.)

---

## 1. Clinical and mode settings (3.1, 3.7)

**Norepinephrine (NE) mode:**
| Setting | Value | Source |
|---|---|---|
| Target MAP | 65.0 mmHg default (Surviving Sepsis Campaign) | `digital_twin.py:51`, `TARGET_MAP_MMHG` |
| Clinician-adjustable target range | 65.0–85.0 mmHg | `constants.py`, `TARGET_MAP_RANGE_MMHG` (SEPSISPAM-studied range) |
| Acceptable band ("in target") | ±3.0 mmHg around target | `expert_controller.py:39`, `TARGET_DEADBAND = 3.0` |
| Critical/emergency low threshold | 50.0 mmHg (absolute, regardless of target) | `expert_controller.py:43`, `EMERGENCY_MAP_MMHG = 50.0` — `[ASSUMPTION]`, not independently cited like the 65 mmHg default itself |
| Minimum dose | 0.0 mL/hr | `constants.py`, `MIN_FLOW_ML_HR` |
| Maximum dose | 190.0 mL/hr (derived from the top row of Medinfo Galway's Table 3: 0.5 mcg/kg/min at 100 kg = 188 mL/hr, rounded up — *not* an explicitly stated "3 mg/hr" hard cap; see REFERENCES.md, corrected 2026-10-05) | `constants.py`, `MAX_FLOW_ML_HR` |
| Dosing range in clinical units | 0.01–3.0 mcg/kg/min (both bounds SCCM-cited; *corrected 2026-10-05 — upper bound was previously an untraceable 3.3*) | `constants.py`, `NOREPINEPHRINE_DOSE_RANGE_MCG_KG_MIN` |

**Sodium nitroprusside (SNP) mode:**
| Setting | Value | Source |
|---|---|---|
| Reduction target | ≤25% MAP drop in hour 1; then toward ~120–126.7 mmHg (≈160/100–110 mmHg equivalent) by hour 6; gradual normalization to 70 mmHg by hour 48 | `hypertensive_constants.py`, `STAGE1_MAX_REDUCTION_FRACTION=0.25`/`STAGE1_DURATION_HOURS=1.0`, `STAGE2_TARGET_MAP_BAND_MMHG=(120.0,126.7)`/`STAGE2_END_HOURS=6.0`, `STAGE3_END_HOURS=48.0` |
| "Hypotension stop" threshold | **There is no single separate stop value** — the safety mechanism *is* the staged floor itself: the controller must never push MAP below `map_floor_mmhg` at the current elapsed time. This is a moving target, not a fixed number. |
| Maximum dose | 10.0 mcg/kg/min; 210.0 mL/hr equivalent (derived for a representative 70 kg patient) | `hypertensive_constants.py`, `NITROPRUSSIDE_MAX_DOSE_MCG_KG_MIN`, `MAX_FLOW_ML_HR` |
| Time limit at max dose | **10.0 minutes, cited but NOT enforced — see discrepancy #2 above.** | `hypertensive_constants.py`, `NITROPRUSSIDE_MAX_DOSE_DURATION_MIN = 10.0` |
| What actually IS enforced instead | A cyanide rate-risk cap (forces `Toxicity-Cap` mode) once cumulative dose exceeds 500 mcg/kg AND current rate exceeds 2 mcg/kg/min; a full stop (`Toxicity-Stop`) once cumulative dose exceeds the 10 mg/kg total buffering capacity | `hypertensive_expert_controller.py`, `hypertensive_twin.py` cyanide tracking |

**Drug concentrations assumed in the syringe:**
- Norepinephrine: **16.0 mcg/mL** (4 mg in 250 mL D5W premix) — `constants.py`, `DRUG_CONCENTRATION_MCG_ML`
- Sodium nitroprusside: **200.0 mcg/mL** (50 mg in 250 mL D5W, the more concentrated standard ICU premix option) — `hypertensive_constants.py`, `DRUG_CONCENTRATION_MCG_ML`

**Titration step size and controller tick interval:**
| | NE controller | SNP controller |
|---|---|---|
| Max step per tick | 0.05 mcg/kg/min | 0.01 mcg/kg/min |
| Proportional gain (KP) | 0.002 mcg/kg/min per mmHg error | 0.0015 mcg/kg/min per mmHg error |
| Derivative gain (KD) | 0.15 mcg/kg/min per (mmHg/tick) trend | 0.02 mcg/kg/min per (mmHg/tick) trend |
| Controller tick interval | 1 second (both simulations use `dt_seconds=1.0`) | 1 second |

**NE emergency action, MAP < 50 mmHg (`EMERGENCY_MAP_MMHG`)**: mode is set to `"Emergency"` and flow escalates by the full max step every tick, clamped at the ceiling — `flow = min(flow + step_to_flow(MAX_STEP_MCG_KG_MIN), max_flow)` (`expert_controller.py:157-160`). This is a hard escalation, not the gentler proportional-derivative law used in the normal operating band.

**SpO2 dynamics (`physiology.py`, `update_spo2`, shared by both twins)**: when MAP < 62 mmHg (65 − the 3 mmHg `SPO2_PERFUSION_DEADBAND_MMHG`), SpO2 declines at a random rate drawn from `uniform(0.03, 0.08)` percentage points per second. When MAP > 68 mmHg, it recovers at `uniform(0.01, 0.05)` percentage points per second. Between 62-68 mmHg it holds steady (the deadband exists specifically to stop sub-1-mmHg noise right at the 65 mmHg threshold from flipping the decline/recover decision every tick).

**`WillBreachFloor` margin (hypertensive risk-model label, `generate_hypertensive_dataset.py`)**: uses `FLOOR_BREACH_MARGIN_MMHG = 5.0` (defined in `hypertensive_expert_controller.py`) — the label fires only if a future tick's MAP falls more than 5 mmHg below that same future tick's own floor, not on any breach at all. The same constant also defines a "gentle correction" zone in the controller itself: between the floor and floor−5mmHg, the controller nudges MAP back up softly rather than reacting as if a full breach occurred.

Note: both controllers apply their step **every simulated second**, which is explicitly a compressed/accelerated version of real clinical titration cadence (real bedside titration happens every few minutes) — already flagged in the code as a stated time-compression simplification, not a claim that real nurses titrate every second.

**Where thresholds come from**: every one of the values above traces to a specific guideline or paper in `REFERENCES.md` (in this same project folder) — that file is Vancouver-numbered and ready to hand to the other Claude directly for the Chapter Two reference list. In short: the 65 mmHg NE target is Surviving Sepsis Campaign; the 65–85 mmHg adjustable range is the SEPSISPAM trial; the 190 mL/hr NE cap is derived from the top row of a hospital IV guideline's (Medinfo Galway) own peripheral-dosing table, not a stated absolute cap (corrected 2026-10-05 — see REFERENCES.md); every SNP number (dosing, cyanide thresholds, half-life) is the FDA NITROPRESS label; the staged reduction schedule is NHLBI guidance as restated in MSD Manual/StatPearls.

---

## 2. Hardware as actually built (3.4, 3.5)

**None of this is derivable from code — all of it needs to come from you:**
- Exact motor datasheet/model number
- Syringe brand and size (confirm still 20 mL)
- Battery type, voltage, mAh
- Adapter rating
- Whether a BMS is actually fitted
- Measured syringe inner diameter (caliper reading)
- A4988 current-limit setting / measured Vref (if not yet set, ask me and I'll give you the trim-pot procedure: `Vref = Imax × 8 × Rsense`)
- EasyEDA schematic export
- Photos (perfboard front/back, assembled pump, CAD model)
- Which mechanical parts are physically fitted vs. design-only (bearings, guide rods, heat-set inserts, light cover) — **limit switches and E-stop ARE physically fitted and confirmed working** (see firmware safety section below), so those two specifically are not design-only
- Fuse rating and polarity-protection method

**What I *can* give you — the firmware's assumed (not measured) syringe/mechanism geometry**, used in the volume-per-step calculation:
| Parameter | Value | Source |
|---|---|---|
| Assumed syringe bore diameter | 19.13 mm | `pump_firmware.ino:172`, `SYRINGE_BORE_DIAMETER_MM` — **this is a design assumption, not the caliper-measured value 3.4.6 needs** |
| Lead screw pitch | 8.0 mm/turn | `pump_firmware.ino:173` |
| Assumed microstepping | 1/16 (3200 steps/turn) | `pump_firmware.ino:174` — **flagged in the firmware's own comments as unverified**, since the MS1/MS2/MS3 microstepping-select pins on the A4988 are not wired to the ESP32 |

**ESP32-S3 pin assignments (fully confirmed from firmware):**
| Function | GPIO |
|---|---|
| Keypad rows (1–4) | 7, 6, 5, 4 |
| Keypad columns (1–4) | 18, 17, 16, 15 |
| Stepper STEP | 2 |
| Stepper DIR | 21 |
| Stepper EN (enable) | 42 |
| LCD I2C SDA | 8 |
| LCD I2C SCL | 9 |
| E-Stop | 1 |
| Home limit switch | 10 |
| End limit switch | 11 |
| Buzzer | 12 |
| Green status LED | 13 |
| Red status LED | 14 |
| Battery voltage ADC | 3 |

(Note: STEP/DIR/EN were deliberately moved off GPIO19/20, which are this board's native-USB D-/D+ lines — using them caused real conflicts during bring-up.)

---

## 3. Digital twin (3.6)

**Norepinephrine / septic shock:**
| Parameter | Value | Source |
|---|---|---|
| Volume of distribution (Vd) | 8.8 L, fixed population value (not weight-scaled) | `constants.py`, `VD_NOREPINEPHRINE_L` |
| Clearance | Interpolated from SAPS-II severity: 3.0 L/min at SAPS 20 → 0.9 L/min at SAPS 60, then age-adjusted and given 25% CV log-normal inter-individual variability | `constants.py`/`patient_generator.py` |
| Emax | 1.2 | `constants.py`, `EMAX_ASSUMED` |
| EC50 | 6.0 mcg/L | `constants.py`, `EC50_ASSUMED` |
| Hill coefficient (γ) | 1.5 | `constants.py`, `HILL_COEFFICIENT_ASSUMED` |
| Baseline SVR by severity | Healthy 900–1200, Mild 700–800, Moderate 600–700, Severe 400–500 dyn·s/cm5 | `constants.py`, `SVR_BY_PROFILE_DYN_S_CM5` |
| Baroreflex gains | k=0.15 (HR relaxation rate), G=0.30 (MAP-deviation response gain), deviation capped at ±25 mmHg (acute/pre-target phase) or ±10 mmHg (once at/near target) - see "Baroreflex HR fix" below | `constants.py`, `BARORELFEX_K`, `BAROREFLEX_G`, `BAROREFLEX_MAX_DEVIATION_ACUTE_MMHG`, `BAROREFLEX_MAX_DEVIATION_MMHG` |

**Sodium nitroprusside / hypertensive emergency:**
| Parameter | Value | Source |
|---|---|---|
| Volume of distribution | Weight-scaled: 0.20 × weight_kg (ECF approximation) | `hypertensive_constants.py`, `ECF_FRACTION_OF_BODY_WEIGHT` |
| Clearance | Derived from half-life: ke=ln(2)/2min, CL=ke×Vd, with 25% CV inter-individual variability | `hypertensive_constants.py` |
| Emax | −0.70 (negative = vasodilation) | `hypertensive_constants.py`, `EMAX_ASSUMED` |
| EC50 | 20.0 mcg/L | `hypertensive_constants.py`, `EC50_ASSUMED_MCG_L` |
| Hill coefficient (γ) | 1.5 | `hypertensive_constants.py` |
| Baseline SVR by severity | Healthy 900–1200, Stage2 1300–1600, Urgency 1600–2000, Emergency 2000–2600 dyn·s/cm5 | `hypertensive_constants.py`, `SVR_BY_PROFILE_DYN_S_CM5` |
| Baroreflex gains | Same k/G as above (shared, drug-agnostic module) | `physiology.py` (reused unmodified) |

**Population ranges (shared by both twins):**
- Age: 22–85 years (uniform)
- Weight: 45–120 kg (uniform)
- Height: 150–195 cm (uniform)
- Severity distribution — NE: Healthy 40% / Mild 30% / Moderate 20% / Severe 10%. SNP: Healthy 30% / Stage2 30% / Urgency 25% / Emergency 15%.

**Time step and integration method**: both twins use `dt_seconds = 1.0` (1-second ticks) and explicit **Euler integration** for every ODE (plasma concentration, heart rate) — stated in the code as computationally simple and adequate at this small a step size, not a higher-order method.

**SpO2 computation**: a heuristic, not a modeled gas-exchange system — SpO2 declines gradually when MAP falls below ~65 mmHg (a 3 mmHg neutral deadband prevents noise-driven flip-flopping right at the threshold), recovers gradually once MAP is adequate again. Explicitly documented as a simplified approximation, not a measured physiological relationship. (`physiology.py`, `update_spo2`)

---

## 4. Dataset and AI (3.8, 3.9)

**Septic shock (norepinephrine):**
- **210 train / 30 validation / 60 test** patients (300 total), 600 ticks (10 simulated minutes) each, independently seeded, leakage-checked (confirmed zero patient-identity overlap across splits) — see "Resolved (2026-09-27)" note above.
- Hyperparameters (`max_depth`) are selected by scoring a small grid on the validation set, then evaluated on test exactly once — `[10, 15, 20]` for the action classifier and risk model, `[6, 10, 14]` for the flow regressor (shallower, since its delta target needs less depth budget — see item below) — same methodology as the hypertensive side, below.

**Hypertensive emergency (sodium nitroprusside):**
- **700 train / 100 validation / 200 test** patients, 7200 ticks (2 simulated hours) each, independently seeded, leakage-checked (confirmed zero patient-identity overlap across splits).
- Hyperparameters (`max_depth`) were selected by scoring a small grid on the validation set, then evaluated on test exactly once.

**Feature lists:**
- NE imitation-model features: `MeasuredMAP, MeasuredHR, MeasuredSpO2, FlowRate, WeightKg, Age, TargetBP, MeasuredBP_Error, MeasuredBP_Trend, MeasuredShockIndex, MeasuredPerfusionIndex, MeasuredOxygenDeficit` (label: `RecommendedFlow`; classification label: `AI_Action`)
- NE risk-model features: the above + `RollMeanMAP, RollStdMAP, RollSlopeMAP, RollMeanSpO2, RollStdSpO2, RollSlopeSpO2, RollMeanHR, RollSlopeHR` (label: `WillDeteriorate`)
- SNP imitation-model features: `MeasuredMAP, MeasuredHR, MeasuredSpO2, FlowRate, WeightKg, Age, ElapsedHours, MAP_Floor, MeasuredBP_Error, MeasuredBP_Trend, CumulativeMcgPerKg, CyanideRateRisk, CyanideBufferExceeded` (label: `FlowDelta` for the regressor — see note below; `AI_Action` for the classifier)
- SNP risk-model features: the above + `RollMeanMAP, RollStdMAP, RollSlopeMAP, RollMeanHR, RollSlopeHR` (label: `WillBreachFloor`)

**Deliberately excluded from all feature sets** (internal simulation ground truth, not real-world observable): SVR, cardiac output, plasma drug concentration, drug effect, clinical severity Profile, SAPS-II.

**Algorithms and final hyperparameters:**
- All six models: `RandomForestClassifier`/`RandomForestRegressor` (scikit-learn), `class_weight="balanced"` on all classification tasks.
- NE side: `n_estimators=500, max_depth=15` (fixed, not searched) for all three models.
- SNP side: `max_depth` selected via validation grid — classifier grid `[10,15,20]` (selected 20), regressor grid `[6,10,14]` (selected 14), risk-model grid `[10,15,20]` (selected 20, by validation ROC-AUC). Final models refit at `n_estimators=500` on subsampled-train + full validation.
- **Important methodological note for your methodology section**: the SNP flow regressor predicts `FlowDelta` (the *change* in flow, `RecommendedFlow − FlowRate`), not the absolute flow rate — an earlier absolute-value version scored a misleadingly perfect R²=1.000 by essentially just copying the current flow rate as its own answer. The delta-target version is the one actually used; the displayed "AI suggested flow" is reconstructed as `FlowRate + predicted_delta`, clamped to `[0, max_flow]`.

**One model per mode, not a shared model** — this was asked before and is now settled: there are **six separate trained model files** (three per condition: action classifier, flow regressor, risk classifier), not one shared model with a mode indicator as an input feature. Each condition has its own feature set, its own label definitions, and its own training data, because the physiology and the safety logic are genuinely different between the two conditions (e.g., SNP's `MAP_Floor`/`CumulativeMcgPerKg` features have no NE equivalent).

**Deterioration/risk label definitions, 30-second horizon:**
- NE — `WillDeteriorate`: within the next 30 ticks (`LOOKAHEAD=30`, i.e. 30 seconds at 1s/tick), does MAP fall below 50 mmHg (`EMERGENCY_MAP_MMHG`) OR SpO2 fall below 90%? (OR of two conditions)
- SNP — `WillBreachFloor`: within the next 30 ticks, does MAP fall below its own future safety floor (with a margin), OR does the cyanide rate-risk flag trip? (OR of two conditions)
- Both labels are built by looking into each simulated patient's own *actual future* trajectory during dataset generation — impossible for the real-time controller to know, which is what makes this a genuinely independent prediction task rather than imitation.

---

## 5. Dashboard and firmware (3.10, 3.11)

**Dashboard**: two applications exist, not one —
1. `dashboard.py` — a **Streamlit** browser dashboard (septic-shock mode only, the original/earlier one).
2. `native_app.py` — a **PyQt6 + pyqtgraph** native Windows desktop application (the current, actively-developed one), covering **both** clinical modes with a live condition selector, plus optional real-hardware-in-the-loop via the ESP32.

A screenshot is not something I can produce for you — you'll need to run `python native_app.py` (or the packaged `SmartInfusionAI.exe`) and capture one yourself.

**Serial command format**: plain-text, newline-terminated commands over USB serial:
- `SET_RATE:<mL/hr>` — e.g. `SET_RATE:12.5`
- `STOP`
- `GET_STATUS`
- `CAL:<steps>` — calibration jog, runs an exact step count at a fixed slow speed
- **Baud rate: 115200**
- **Watchdog timeout: 3000 ms (3 seconds)** — if no `SET_RATE` arrives within this window (and the pump isn't in manual/keypad mode, which has no PC to "go quiet"), the firmware auto-stops the motor.

---

## 6. Testing (3.12)

**Actually run and confirmed (real hardware), as of this project's current state:**
- Motor direction verified correct (clockwise/push) after a hardware fix (coil-lead swap).
- Volumetric accuracy check: commanded 10.00 mL via Rate+VTBI at 190 mL/hr; measured ~9.8 mL on the syringe's own graduation marking (~2% shortfall, accepted, not corrected).
- Autonomous (hardware-in-the-loop) mode: a full simulated Emergency→Decrease→Maintain control cycle was streamed to and executed by the real pump, with correct clamping at the 190 mL/hr maximum.
- Basic functional checks: keypad, LCD, buzzer, status LEDs, battery voltage reading, E-stop (disables driver, latches until reset), communication watchdog, physical limit switches (block motion at each end of travel) — all confirmed working during hardware bring-up.
- The `CAL:<steps>` raw step-count calibration command **exists in the firmware but has not actually been run/exercised** — the volumetric check above used the Rate+VTBI method instead.

**NOT yet run / not confirmed — be honest about these in your methodology as planned, not completed, testing:**
- **No gravimetric (mass-based, scale) testing has been done at any point in this project.** The only volumetric check performed was reading the syringe's own printed graduation marks, not weighing delivered fluid. If your report describes gravimetric testing, that has not happened — you would need an actual precision scale (0.01 g resolution is standard for this kind of small-volume infusion accuracy testing) to do it.
- No repeated-trial statistical characterization of the ~2% volumetric shortfall (only a single measurement was taken; the project's own notes explicitly flag this as needing 2-3 repeat trials before treating the shortfall as a real systematic bias rather than reading-precision noise).
- No formal, long-duration (multi-hour+) physical hardware endurance test — the multi-hour verification that *was* done (confirming a vascular-tone-recovery/baroreflex-adaptation fix in the hypertensive twin) was a **pure software simulation test**, not a physical hardware run.
- No independent verification of the A4988 current-limit/Vref setting against the motor's actual rated current (this requires the physical measurement listed in Part 2 above).

---

## 7. Detailed model internals (follow-up batch, 2026-09-27)

1. **Age-adjusted NE clearance** (`patient_generator.py:137-150`): if age ≤ 25, `age_factor=1.0`. Else `age_factor = 1.0 - 0.25 × min(1.0, (age-25)/(67-25))`, floored at `AGE_CLEARANCE_FLOOR=0.70`. Final clearance = SAPS-II-interpolated value × age_factor × log-normal inter-individual noise (`CLEARANCE_INTER_INDIVIDUAL_CV=0.25`).
2. **Baseline stroke volume / CVP**: SV is *derived*, not sampled - `sv0_ml = co0_l_min × 1000 / hr_set_bpm`, to keep MAP/SVR/CO/SV internally consistent. `CVP_TYPICAL_MMHG = 4.0` mmHg, fixed (not dynamic).
3. **Fluid-balance model: NOT implemented.** `bv0_fraction` is a static per-profile constant (Healthy 1.0, Mild 0.95, Moderate 0.85, Severe 0.75) - never changes during a run.
4. **Baroreflex setpoints (septic)**: `hr_set` ~ uniform(60,100) bpm; `map_set` ~ uniform(70,100) mmHg - both sampled once per patient, independently.
5. **Hypertensive vascular-tone recovery + baroreceptor adaptation** (`hypertensive_twin.py`, added 2026-09-18): once MAP≥floor continuously for `HTN_RECOVERY_GRACE_HOURS=0.5h`, `svr0 += (1050 - svr0) × (dt_hours/6.0)` (τ=6h, resets on breach). Independently and unconditionally, `map_set += (MAP - map_set) × (dt_hours/24.0)` (τ=24h, always running).
6. **SpO2 rates**: drawn fresh EVERY tick (`uniform(0.03,0.08)` decline / `uniform(0.01,0.05)` recovery, %/second), not once per patient. Upper limit hard-clipped at 100%.
7. **Measurement noise**: Gaussian (`np.random.normal(0, sd)`) added every tick - MAP sd=3.0 mmHg, HR sd=2.0 bpm, SpO2 sd=1.0%.
8. **SAPS II**: fixed point values, not ranges - Healthy=20, Mild=20, Moderate=40, Severe=60.
9. **NE trend - two different signals, don't conflate them**: the controller's own decision uses the EMA-*smoothed* trend (`smoothed_bp - previous_smoothed_bp`). The AI's `MeasuredBP_Trend` feature is a RAW one-tick difference of the noisy measured value (`measured_map - prev_measured_map`) - not smoothed.
10. **Inside the ±3 mmHg deadband**: dose is NOT held constant - the continuous PD law keeps running every tick; it's only the displayed Mode that reads "Maintain," with small proactive nudges still possible.
11. **Staged floor**: a continuous piecewise-linear RAMP, not a step, within and across all three phases (`map_reduction_floor_mmhg()` in `hypertensive_twin.py`) - phase boundaries connect with no discontinuity.
12. **Nitroprusside control law**: same structural form as the septic PD law (`step = KP×error + KD×derivative`, clamped to ±MAX_STEP), but with NO fast-escalation tier at all - deliberate, since the FDA label has no emergency exception for this drug.
13. **MAP >5 mmHg below floor**: hard override - `mode="Decrease"`, flow cut by the full max step every tick, bypassing the normal PD law entirely.
14. **Toxicity-Cap rate**: capped exactly at `CYANIDE_RATE_RISK_THRESHOLD_MCG_KG_MIN = 2.0` mcg/kg/min.
15. **AI_Action classes**: Septic - `Maintain, Increase, Decrease, Emergency` (4). Hypertensive - `Maintain, Increase, Decrease, Emergency, Toxicity-Cap` (5, all observed in data); `Toxicity-Stop` exists in the controller but has never occurred in any generated dataset.
16. **Subsampling fraction**: `min(1.0, 150000 / actual_train_row_count)`, recomputed dynamically - ≈3.0% for the 5.04M-row hypertensive training set.
17. **Feature definitions** (from noisy measured values): `MeasuredShockIndex = MeasuredHR / max(MeasuredMAP,1)`; `MeasuredPerfusionIndex = MeasuredMAP × MeasuredSpO2 / 100`; `MeasuredOxygenDeficit = 100 - MeasuredSpO2`.
18. **Rolling window length**: 30 ticks (`WINDOW=30`), both modes.
19. **Last ~29 ticks of each run**: Python slicing just returns a shorter lookahead window near the end (never errors) - recorded honestly per-row via `LookaheadTicksAvailable`. The negative label on those final rows is weaker evidence than a full-window negative.
20. **Positive deterioration-label rate**: Septic `WillDeteriorate` (regenerated 2026-09-27, fixed seeds 1/999, deterministic/reproducible): train 9.0% positive, test 6.3% positive. Hypertensive `WillBreachFloor`: train 1.98%, val 1.20%, test 2.12% (from the actual training log). Note the septic rate is noticeably higher than the hypertensive one - the hypertensive side needed the deliberate titration-lag injection mechanism (see Part on dataset generation) specifically because a well-behaved run alone produced ~0% positives, whereas the septic side's ordinary controller behavior alone already produces a meaningfully imbalanced-but-present positive rate.

---

## 8. Second follow-up batch (2026-09-27)

1. **Baseline CO**: derived, not sampled - `co0_l_min = (map0_target - CVP_TYPICAL_MMHG) × 80.0 / svr0`, clipped to [1.5, 12.0] L/min. Identical formula in both twins - keeps MAP/SVR/CO mutually consistent.
2. **`bv0_fraction` multiplies stroke volume**: `SV = sv0_ml × blood_volume_fraction` (Frank-Starling/preload proxy). Only meaningful on the septic side - hypertensive twin fixes this at 1.0 always.
3. **Setpoint ranges - only partially shared.** HR: same range both twins (`uniform(60,100)`). MAP: NOT shared - septic draws `map_set ~ uniform(70,100)`; hypertensive sets `map_set = presenting_map_mmhg` directly (the patient's own value, not a population range) - this IS the baroreflex-resetting mechanism, not an oversight.
4. **α = 0.3** (`smoothed_bp += 0.3×(bp-smoothed_bp)`), identical in both twins. Error term uses RAW `patient.map`, not smoothed; only the trend/derivative term uses `smoothed_bp`. Same in both controllers.
5. **Floor at t=0** = `presenting_map_mmhg` exactly. At hour 6 it reaches the MIDPOINT of the 120-126.7 mmHg band (123.35 mmHg), not either endpoint - code computes `sum(STAGE2_TARGET_MAP_BAND_MMHG)/2.0`.
6. **"Emergency" in the SNP controller**: fires when `MAP > 133.33 mmHg` (`EMERGENCY_MAP_MMHG_HIGH`, derived from ACC/AHA 180/110 via the MAP formula) AND mode isn't already "Decrease" - applied AFTER flow is already computed, only overwrites the Mode label/explanation, never changes flow. That's why it can appear in the data despite "no fast-escalation tier" - it's a label overlay, not a control branch.
7. **Titration-lag mechanics - three distinct parts**: (a) decided once per patient at run start, 30% probability (`TITRATION_LAG_FRACTION`); (b) a single delay drawn once per lagged patient, uniform 30-180s, held fixed for their whole run via a delay queue; (c) NOT just a delayed safe policy - lagged patients get a substituted, less-safe policy (`_naive_titration_flow()`): a 15x larger step (0.15 vs 0.01 mcg/kg/min) plus a reintroduced "ramp toward max in the malignant range" branch deliberately removed from the real controller earlier for being dangerous. `AI_Action`/`RecommendedFlow` labels always come from the real controller regardless of subgroup - only the physiology is affected.

**References (by content, not your list's numbering):**
- Normal SVR range: generic "standard adult haemodynamic reference ranges" (SVR 800-1200 dyn.s/cm5 etc.) - not one single primary source, consistent across ICU/nursing tables and StatPearls.
- Baroreflex model: Ringwood JV, Malpas SC - structural citation for the first-order ODE form only; no numeric gain constant found in this literature.
- Baroreflex resetting in hypertension: general physiological concept, not one single cited paper.

---

## 9. Third follow-up batch (2026-09-27)

1. **Depth-selection scoring measure**: classifier - validation accuracy (highest wins). Regressor - validation delta-MAE (lowest wins). Risk model - validation ROC-AUC (highest wins). Three different metrics, one per model.
2. **All other RandomForest settings are scikit-learn defaults.** Only `n_estimators`, `max_depth`, `class_weight="balanced"` (classifier only), `random_state=42`, `n_jobs=-1` are ever explicitly set, across all four constructor calls (search + final, classifier + regressor). `min_samples_split`/`min_samples_leaf`/`max_features`/`criterion`/`bootstrap` etc. all left at default.
3. **Risk-percentage thresholds**: <20% = "Low risk" (green); 20-49% = "Watch" (yellow); >=50% = "ELEVATED RISK" (red, `RISK_ALERT_THRESHOLD=0.5`).
4. **Decisions & Trends tab**: three side-by-side text panels (AI advisory / Controller sole-actuator / Patient & Hemodynamics-hidden-from-AI), plus an Event Log (scrolling, timestamped) and a Recent History table (10 columns: Time, MAP, MeasMAP, HR, SpO2, Ctrl Mode, Ctrl Flow, AI Action, AI Flow, Risk).
5. **Session log: NOT persisted to any file.** No `open()`/`to_csv()`/file-write anywhere in `native_app.py` for the event log or history table - both are in-memory only (event log: last 30 entries, `MAX_EVENT_LOG`; history table: last 250 ticks, `MAX_HISTORY_ROWS_SHOWN`). Nothing survives closing the app. If the methodology describes persistent session logging, that does not currently exist.
6. **Carried over, reconfirmed**: the controller's error term uses the TRUE/raw `patient.map`, never the noisy `Measured*` value - the noise-injected values are a completely separate computation that only ever reaches the AI models and the display. The controller and the AI do NOT see the same input signal.

---

*Compiled from the current state of the actual project files on 2026-09-23, extended 2026-09-27. Anything above not explicitly marked "NOT AVAILABLE" was pulled directly from the code, with file/line references given so you can spot-check it yourself.*
