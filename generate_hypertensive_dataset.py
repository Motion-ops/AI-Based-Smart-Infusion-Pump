"""
===============================================================================
GENERATE_HYPERTENSIVE_DATASET.PY
Smart AI Infusion Pump - Hypertensive Emergency / Sodium Nitroprusside

Purpose
-------
Sibling of generate_dataset.py for the second clinical mode. Same two
prediction tasks, same imitation-plus-independent-lookahead-label
design, produced in one generation pass:

1. Infusion Advisory Model
   Imitates hypertensive_expert_controller.py: predicts the
   recommended nitroprusside rate from observable bedside measurements.

2. Early Risk Prediction Model
   Predicts "WillBreachFloor" - whether MAP is about to fall below the
   staged safety floor, or the cyanide rate-risk flag is about to
   trip, within a future lookahead window. The hypertensive-mode
   analogue of the septic side's "WillDeteriorate" - same OR-of-two-
   conditions pattern, mirrored deliberately.

Why some patients get a simulated titration lag
------------------------------------------------
hypertensive_expert_controller.py was deliberately tuned (see its own
module docstring) to never breach the staged safety floor - which
means a dataset generated purely from its decisions applied
instantaneously has ZERO floor-breach events: nothing for the risk
model to learn to anticipate. This isn't a flaw in the controller, it's
a mismatch between "how a robot applies its own decisions" (instantly)
and "how bedside titration actually happens in practice" (a clinician
notices a monitor value, decides to adjust the pump, and does so - not
instantaneously). That real response delay is exactly the gap a
genuine early-warning system earns its value from.

[ASSUMPTION] A minority of simulated patients (TITRATION_LAG_FRACTION)
get BOTH a fixed per-patient response delay (TITRATION_LAG_RANGE_SECONDS)
AND have their ACTUALLY-APPLIED rate driven by a deliberately naive
titration policy (_naive_titration_flow(), below) instead of the real
controller - modeling realistic imperfect human titration, not a
literature-measured value.

An earlier version of this file tried delay alone, and separately tried
just raising the real controller's max-step-per-tick ceiling - NEITHER
alone produced a single floor breach in testing. Reason: the real
controller's proportional-derivative law naturally tapers its own step
size as MAP approaches the floor (the KP*error term shrinks on its
own), regardless of what the ceiling is - a bigger ceiling only speeds
up the initial approach when the error is still large, it doesn't stop
the same law from slowing down near the floor. The ORIGINAL unsafe
behavior this project found and fixed (see
hypertensive_expert_controller.py's docstring) wasn't "steps too big",
it was a floor-BLIND branch that kept ramping toward max dose purely
because MAP was still in the malignant range, with no awareness of how
close the floor already was. _naive_titration_flow() deliberately
reproduces exactly that removed branch - it's what a genuinely careless
manual titration (someone chasing "the number is still too high, give
more") looks like, and it's the only thing that actually generates real
breach events. The real controller and this naive one otherwise share
the same PD tapering logic away from the two hard branches, so the
naive policy isn't a strawman - it degrades in exactly the one place
the real controller was specifically fixed to not degrade.

The imitation/flow model is UNAFFECTED by any of this - RecommendedFlow
always reflects the real, current controller's own decision (what it
WOULD recommend given the true measurements), never the naive policy's
output, so that model only ever learns the correct expert policy.
Lag identity/duration is recorded per row for post-hoc analysis only
(TitrationLagSeconds) - it is NEVER included as a model input feature,
same reasoning constants.py gives for excluding Profile/SAPS_II from
FEATURE_COLUMNS: a real deployed system has no way to know in advance
which patient's care team will respond slower.

Dataset split - PATIENT-LEVEL, not row-level
----------------------------------------------
Each simulated patient produces hundreds of rows (one per tick).
Splitting rows randomly would let the model see part of a patient's
trajectory in training and be "tested" on another part of the SAME
patient - this is data leakage: the model has partial knowledge of
that exact patient's response curve, so test performance would
overstate genuine generalization to a never-seen patient.

The fix used throughout this project (both here and in the existing
septic-shock generate_dataset.py, which already assigns entire
patients to train vs. test via independent seeded RNG streams) is to
split at the PATIENT level: every tick belonging to a given patient_id
goes into exactly one of the three sets. This file makes that split
explicit and three-way (train/validation/test), per the standard
70/10/20-of-total methodology (not 80/10/10 - that's "80% development,
further split 700/100 into train/validation" restated, which is where
this project's 70/10/20 numbers actually come from):

    1000 simulated patients
              |
    +---------+---------+
    |         |         |
   700       100       200
  Training Validation  Test

Training  -> teaches the models (train_hypertensive_ai.py / risk model)
Validation-> tune hyperparameters / compare models, never touched by fit()
Test      -> touched exactly once, for the final reported numbers

All feature-selection rules (hidden ground truth vs. legitimate
bedside-observable inputs) follow the same philosophy documented in
generate_dataset.py.
===============================================================================
"""

import random
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hypertensive_constants as hc
from hypertensive_twin import VirtualHypertensivePatient
from hypertensive_expert_controller import (
    hypertensive_expert_controller, FLOOR_BREACH_MARGIN_MMHG,
    EMERGENCY_MAP_MMHG_HIGH, TARGET_DEADBAND, KP, KD,
)
from nitroprusside_pharmacokinetics import mass_rate_to_flow_ml_hr


# Columns legitimately observable by a real pump/bedside monitor.
# MAP_Floor and ElapsedHours are included even though they're DERIVED
# quantities, not raw sensor readings - unlike SVR/CO/plasma
# concentration (true hidden internal physiology), both are things a
# real pump could compute itself: ElapsedHours is just a clock since
# infusion start, and MAP_Floor is a deterministic function of it plus
# the patient's OWN presenting MAP (recorded at treatment start, not a
# hidden simulation variable). CumulativeMcgPerKg is likewise legitimate
# - it's arithmetic on what the pump itself has already delivered, not
# something requiring a lab test to know.
FEATURE_COLUMNS = [
    "MeasuredMAP", "MeasuredHR", "MeasuredSpO2", "FlowRate",
    "WeightKg", "Age", "ElapsedHours", "MAP_Floor",
    "MeasuredBP_Error", "MeasuredBP_Trend",
    "CumulativeMcgPerKg", "CyanideRateRisk", "CyanideBufferExceeded",
]
LABEL_COLUMN = "RecommendedFlow"

ROLLING_FEATURE_COLUMNS = [
    "RollMeanMAP", "RollStdMAP", "RollSlopeMAP",
    "RollMeanHR", "RollSlopeHR",
]
RISK_FEATURE_COLUMNS = FEATURE_COLUMNS + ROLLING_FEATURE_COLUMNS
RISK_LABEL_COLUMN = "WillBreachFloor"

WINDOW = 30
LOOKAHEAD = 30

# Same generic monitor-noise magnitudes as generate_dataset.py - the
# noise model is drug-agnostic, only the physiology it's applied to differs.
BP_NOISE_SD = 3.0
HR_NOISE_SD = 2.0
SPO2_NOISE_SD = 1.0

# [ASSUMPTION] See module docstring "Why some patients get a simulated
# titration lag" - not a literature-measured human-response-time
# figure, calibrated only to actually produce breach events given how
# fast nitroprusside acts (its ~2 min half-life - see
# hypertensive_constants.py) and how conservatively the controller
# steps (MAX_STEP_MCG_KG_MIN=0.01).
TITRATION_LAG_FRACTION = 0.30
TITRATION_LAG_RANGE_SECONDS = (30, 180)

# [ASSUMPTION] Step size used by _naive_titration_flow() (below) for
# the same minority of lagged patients. Deliberately the SAME
# 0.15 mcg/kg/min that hypertensive_expert_controller.py's own
# docstring documents as causing real 19-48 mmHg floor breaches when it
# was the DEFAULT step, before the floor-blind ramp branch that made it
# dangerous was found and removed - reusing that exact magnitude, WITH
# that same removed branch, is what makes this a genuine reproduction
# of a real failure mode this project already found, not an arbitrarily
# invented one.
IMPERFECT_TITRATION_MAX_STEP_MCG_KG_MIN = 0.15


def _naive_titration_flow(patient, max_step_mcg_kg_min):
    """
    Deliberately reproduces the floor-BLIND ramp behavior
    hypertensive_expert_controller.py's own "Control law" comment
    documents finding and removing: ramps toward max whenever MAP is
    in the ACC/AHA malignant range, with no regard for how close MAP
    already is to the safety floor. This is what a genuinely careless
    manual titration ("the number is still too high, give more",
    without checking how fast it's already falling) looks like.

    Used ONLY to generate realistic imperfect/naive titration for a
    minority of simulated patients in THIS dataset - never used as
    this project's actual control policy (hypertensive_expert_controller
    remains that), and never used as an imitation-learning label (see
    the module docstring's "Why some patients get a simulated titration
    lag" section - AI_Action/RecommendedFlow always come from the real
    controller, regardless of what this function drives physiologically).
    """

    bp = patient.map
    flow = patient.flow_rate
    weight_kg = patient.weight_kg
    floor = patient.map_floor_mmhg

    def step_to_flow(step_mcg_kg_min):
        return mass_rate_to_flow_ml_hr(step_mcg_kg_min * weight_kg)

    if bp > EMERGENCY_MAP_MMHG_HIGH:
        return min(flow + step_to_flow(max_step_mcg_kg_min), hc.MAX_FLOW_ML_HR)

    if bp < floor - TARGET_DEADBAND - FLOOR_BREACH_MARGIN_MMHG:
        return max(flow - step_to_flow(max_step_mcg_kg_min), hc.MIN_FLOW_ML_HR)

    # Same PD taper as the real controller elsewhere - only the two
    # branches above are the reintroduced unsafe part.
    error = bp - floor
    derivative = patient.smoothed_bp - patient.previous_smoothed_bp
    raw_step = KP * error + KD * derivative
    step = max(-max_step_mcg_kg_min, min(raw_step, max_step_mcg_kg_min))
    return float(np.clip(flow + step_to_flow(step), hc.MIN_FLOW_ML_HR, hc.MAX_FLOW_ML_HR))


def _profile_assignment(num_patients, seed):
    names = list(hc.PROFILE_WEIGHTS.keys())
    weights = [hc.PROFILE_WEIGHTS[n] for n in names]
    total_weight = sum(weights)
    counts = [round(num_patients * w / total_weight) for w in weights]
    diff = num_patients - sum(counts)
    counts[0] += diff
    assignment = []
    for name, cnt in zip(names, counts):
        assignment.extend([name] * cnt)
    rng = random.Random(seed)
    rng.shuffle(assignment)
    return assignment


def _rolling_slope(buffer):
    n = len(buffer)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = np.asarray(buffer, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def generate_hypertensive_dataset(num_patients, sim_time, seed, output_file):
    random.seed(seed)
    np.random.seed(seed)

    profile_assignment = _profile_assignment(num_patients, seed)

    rows = []
    print(f"Generating {num_patients} virtual hypertensive-emergency patients "
          f"(seed={seed}) -> {output_file}")

    for patient_id in range(num_patients):
        patient = VirtualHypertensivePatient(
            patient_id,
            profile=profile_assignment[patient_id],
            seed=seed * 100_003 + patient_id,  # deterministic-but-distinct per patient,
                                                # same convention as generate_dataset.py
        )

        # Titration lag - see module docstring "Why some patients get a
        # simulated titration lag". Uses the plain `random` module
        # (seeded once via random.seed(seed) at the top of this
        # function), which nothing else in this loop draws from - the
        # twin's own physiological RNG and np.random's measurement
        # noise are both separate streams, so this doesn't perturb
        # either.
        has_titration_lag = random.random() < TITRATION_LAG_FRACTION
        if has_titration_lag:
            lag_seconds = random.uniform(*TITRATION_LAG_RANGE_SECONDS)
        else:
            lag_seconds = 0.0
        lag_ticks = int(round(lag_seconds / patient.dt_seconds))
        # A fixed-length queue IS a fixed-delay line: once full,
        # index [0] is always exactly lag_ticks old. maxlen=1 (the
        # no-lag case) degenerates to "apply immediately", identical
        # to this project's original undelayed behavior.
        pending_flow_queue = deque(maxlen=lag_ticks + 1)

        prev_measured_map = None
        map_window = deque(maxlen=WINDOW)
        hr_window = deque(maxlen=WINDOW)

        patient_rows = []

        for second in range(sim_time):
            patient.update()

            active_flow = patient.flow_rate  # rate BEFORE this tick's decision - no leakage

            # ALWAYS the ideal/default-step decision - this is what
            # becomes AI_Action/RecommendedFlow below, so the imitation
            # model trains on the correct expert policy regardless of
            # which subgroup this patient is in (see module docstring).
            decision = hypertensive_expert_controller(patient)

            measured_map = float(np.clip(patient.map + np.random.normal(0, BP_NOISE_SD), 20, 250))
            measured_hr = float(np.clip(patient.hr + np.random.normal(0, HR_NOISE_SD), 30, 220))
            measured_spo2 = float(np.clip(patient.spo2 + np.random.normal(0, SPO2_NOISE_SD), 70, 100))

            if prev_measured_map is None:
                prev_measured_map = measured_map

            measured_bp_error = measured_map - patient.map_floor_mmhg
            measured_bp_trend = measured_map - prev_measured_map
            prev_measured_map = measured_map

            map_window.append(measured_map)
            hr_window.append(measured_hr)

            roll_mean_map = float(np.mean(map_window))
            roll_std_map = float(np.std(map_window))
            roll_slope_map = _rolling_slope(map_window)
            roll_mean_hr = float(np.mean(hr_window))
            roll_slope_hr = _rolling_slope(hr_window)

            cumulative_mcg_per_kg = patient.cumulative_nitroprusside_mcg / patient.weight_kg

            patient_rows.append({
                "PatientID": patient.patient_id,
                "Time": second,
                "Profile": patient.profile,

                # Hidden ground-truth - NOT valid AI input features.
                # TitrationLagSeconds especially - a real deployed
                # system has no way to know in advance which patient's
                # care team will respond slower (see module docstring).
                "TitrationLagSeconds": round(lag_seconds, 1),
                "ClearanceLMin": round(patient.clearance_l_min, 3),
                "VdL": round(patient.vd_l, 3),
                "SVR": round(patient.svr, 1),
                "CO": round(patient.co, 2),
                "DrugEffect": round(patient.drug_effect, 3),
                "PlasmaConcentration": round(patient.plasma_concentration, 3),

                # TRUE simulated hemodynamics - ground truth / diagnostics only.
                "MAP": round(patient.map, 2),
                "HR": round(patient.hr, 2),
                "SpO2": round(patient.spo2, 2),

                # NOISY sensor readings + legitimately-observable derived
                # quantities - valid AI input features.
                "MeasuredMAP": round(measured_map, 2),
                "MeasuredHR": round(measured_hr, 2),
                "MeasuredSpO2": round(measured_spo2, 2),
                "MeasuredBP_Error": round(measured_bp_error, 2),
                "MeasuredBP_Trend": round(measured_bp_trend, 2),
                "ElapsedHours": round(patient.hours_since_start, 4),
                "MAP_Floor": round(patient.map_floor_mmhg, 2),
                "CumulativeMcgPerKg": round(cumulative_mcg_per_kg, 2),
                "CyanideRateRisk": int(patient.cyanide_rate_risk_active),
                "CyanideBufferExceeded": int(patient.cyanide_buffer_exceeded),

                "RollMeanMAP": round(roll_mean_map, 2),
                "RollStdMAP": round(roll_std_map, 2),
                "RollSlopeMAP": round(roll_slope_map, 3),
                "RollMeanHR": round(roll_mean_hr, 2),
                "RollSlopeHR": round(roll_slope_hr, 3),
                "WindowSize": len(map_window),

                "FlowRate": round(active_flow, 2),
                "WeightKg": round(patient.weight_kg, 1),
                "Age": patient.age,
                "PresentingMAP": round(patient.presenting_map_mmhg, 2),

                "AI_Action": decision["Mode"],
                "ClinicalExplanation": decision["ClinicalExplanation"],
                "RecommendedFlow": decision["RecommendedFlow"],
            })

            # What actually reaches the pump next tick is a SEPARATE
            # computation from the ideal `decision` above, for patients
            # with a titration lag - the naive floor-blind policy
            # instead of the real controller, plus a queued delay (see
            # the titration-lag setup above and the module docstring).
            # This keeps the applied physiology fully independent of
            # the ideal decision recorded as this row's label.
            if has_titration_lag:
                applied_flow_this_tick = _naive_titration_flow(
                    patient, IMPERFECT_TITRATION_MAX_STEP_MCG_KG_MIN
                )
            else:
                applied_flow_this_tick = decision["RecommendedFlow"]

            pending_flow_queue.append(applied_flow_this_tick)
            patient.flow_rate = pending_flow_queue[0]

        # ---- Second pass: look ahead into this patient's OWN future
        # trajectory, exactly matching generate_dataset.py's pattern.
        # WillBreachFloor combines two conditions, same OR-of-two-risks
        # structure as the septic side's WillDeteriorate (MAP<emergency
        # OR spo2<90): here, MAP dropping unsafely below its OWN
        # future floor at that future tick, OR the cyanide rate-risk
        # flag turning on - both are genuinely different "something bad
        # is about to happen" categories worth an early warning on.
        true_map_seq = [r["MAP"] for r in patient_rows]
        true_floor_seq = [r["MAP_Floor"] for r in patient_rows]
        true_cyanide_seq = [r["CyanideRateRisk"] for r in patient_rows]

        for i, r in enumerate(patient_rows):
            future_slice = slice(i + 1, i + 1 + LOOKAHEAD)
            future_map = true_map_seq[future_slice]
            future_floor = true_floor_seq[future_slice]
            future_cyanide = true_cyanide_seq[future_slice]

            will_breach = any(
                m < (f - FLOOR_BREACH_MARGIN_MMHG)
                for m, f in zip(future_map, future_floor)
            ) or any(future_cyanide)

            r[RISK_LABEL_COLUMN] = int(will_breach)
            r["LookaheadTicksAvailable"] = len(future_map)

        rows.extend(patient_rows)

    dataset = pd.DataFrame(rows)
    dataset.to_csv(output_file, index=False)

    print(dataset.shape)
    print("Profile distribution (per patient):")
    print(dataset.groupby("PatientID")["Profile"].first().value_counts())
    print("AI_Action distribution:")
    print(dataset["AI_Action"].value_counts(normalize=True).round(3))
    print(f"{RISK_LABEL_COLUMN} distribution:")
    print(dataset[RISK_LABEL_COLUMN].value_counts(normalize=True).round(3))
    print("FlowRate == RecommendedFlow (should be False):",
          (dataset["FlowRate"] == dataset["RecommendedFlow"]).all())
    print("MAP range:", dataset["MAP"].min(), "-", dataset["MAP"].max())

    return dataset


def plot_patient(dataset, patient_id, title_suffix=""):
    sample = dataset[dataset.PatientID == patient_id]

    plt.figure(figsize=(10, 5))
    plt.plot(sample["Time"], sample["MAP"], label="MAP")
    plt.plot(sample["Time"], sample["MAP_Floor"], linestyle="--", label="Safety Floor")
    plt.xlabel("Time (s)")
    plt.ylabel("MAP (mmHg)")
    profile = sample["Profile"].iloc[0]
    plt.title(f"Patient {patient_id} ({profile}) MAP vs. Staged Floor {title_suffix}")
    plt.legend()
    plt.grid()

    plt.figure(figsize=(10, 5))
    plt.plot(sample["Time"], sample["FlowRate"], label="Flow Rate")
    plt.xlabel("Time (s)")
    plt.ylabel("mL/hr")
    plt.title(f"Patient {patient_id} ({profile}) Nitroprusside Flow Rate {title_suffix}")
    plt.grid()


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Patient-level 700/100/200 split (of 1000 total) - see module
    # docstring for why this is 70/10/20-of-TOTAL, not 80/10/10, and
    # why the split happens at the whole-patient level via independent
    # seeded RNG streams rather than by shuffling individual rows.
    #
    # sim_time is longer than the septic side's 600s (10 simulated
    # minutes) because the staged safety schedule plays out over HOURS,
    # not minutes. An initial 3600s (1 simulated hour) run only ever
    # covered Stage 1 (0-1h) of the three-stage floor schedule (see
    # hypertensive_constants.py) - Stage 2 (1-6h, target ~160/100-110
    # mmHg equivalent) and Stage 3 (6-48h, gradual normalization) were
    # NEVER observed by the model at all, despite MAP_Floor being one
    # of its most-relied-on features. That same 1-hour run also
    # revealed cumulative nitroprusside exposure topped out at 268.6
    # mcg/kg across all 700 patients - nowhere near the 500 mcg/kg
    # cyanide rate-risk threshold, not because doses were too low
    # (23.6% of rows already exceeded the 2 mcg/kg/min rate threshold)
    # but simply because there wasn't enough simulated TIME for
    # cumulative exposure to accumulate. 7200s (2 simulated hours) is
    # the smallest extension that fixes both: it opens a full hour of
    # genuine Stage 2 coverage, and extrapolating from the measured
    # 268.6 mcg/kg/hour worst case, the sickest lagged patients should
    # now cross ~537 mcg/kg - just past the real threshold, without
    # inflating row count enough to reintroduce the training-time
    # problem this file's training script already had to fix once
    # (see train_hypertensive_ai.py's TRAIN_SUBSAMPLE_FRACTION note).
    SIM_TIME_SECONDS = 7200

    train_df = generate_hypertensive_dataset(
        num_patients=700, sim_time=SIM_TIME_SECONDS, seed=1,
        output_file="data/hypertensive_train_patients.csv"
    )

    val_df = generate_hypertensive_dataset(
        num_patients=100, sim_time=SIM_TIME_SECONDS, seed=777,
        output_file="data/hypertensive_val_patients.csv"
    )

    test_df = generate_hypertensive_dataset(
        num_patients=200, sim_time=SIM_TIME_SECONDS, seed=999,
        output_file="data/hypertensive_test_patients.csv"
    )

    # Sanity check: confirm no patient ID appears in more than one
    # split's underlying identity - patient_id ALONE can collide across
    # files (each file numbers its own patients starting at 0), so the
    # real leakage check is on the (seed, patient_id) pair actually
    # used to construct each VirtualHypertensivePatient, which is what
    # genuinely determines whether two rows came from the same
    # simulated individual.
    train_keys = {(1, pid) for pid in train_df["PatientID"].unique()}
    val_keys = {(777, pid) for pid in val_df["PatientID"].unique()}
    test_keys = {(999, pid) for pid in test_df["PatientID"].unique()}
    overlap = (train_keys & val_keys) | (train_keys & test_keys) | (val_keys & test_keys)
    print(f"\nPatient-level leakage check (should be empty set): {overlap}")

    for profile in ["Stage2", "Emergency"]:
        match = train_df[train_df.Profile == profile]
        if not match.empty:
            plot_patient(train_df, match["PatientID"].iloc[0], title_suffix="(train)")
            map_fignum, flow_fignum = plt.get_fignums()[-2:]
            plt.figure(map_fignum).savefig(f"figures/hypertensive_example_{profile.lower()}_map.png", dpi=150)
            plt.figure(flow_fignum).savefig(f"figures/hypertensive_example_{profile.lower()}_flow.png", dpi=150)
    plt.close("all")

    print("\nTrain/validation/test hypertensive datasets generated successfully.")
    print(f"Action/flow model features: {FEATURE_COLUMNS}")
    print(f"Risk model features: {RISK_FEATURE_COLUMNS}")
    print(f"Risk label: {RISK_LABEL_COLUMN} (lookahead={LOOKAHEAD} ticks)")
    print("Profile/ClearanceLMin/VdL/SVR/CO/DrugEffect/PlasmaConcentration/MAP/HR/SpO2 "
          "are hidden ground truth - do not use as model inputs (use the Measured* columns instead).")
