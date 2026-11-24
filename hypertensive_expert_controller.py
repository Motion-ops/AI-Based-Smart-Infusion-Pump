"""
===============================================================================
HYPERTENSIVE_EXPERT_CONTROLLER.PY
Smart AI Infusion Pump - Hypertensive Emergency Digital Twin

Purpose
-------
Sibling of expert_controller.py for sodium nitroprusside. Same overall
shape (weight-based dosing internally, converted to mL/hr only at the
end; a hard emergency tier plus a continuous PD law; a deterministic
safety overlay applied last), but the CONTROL PROBLEM ITSELF is
different in a way that matters clinically, not just a sign flip:

    Septic shock (expert_controller.py): the target (65 mmHg) is FIXED.
    "MAP too low" always means "give more drug, as fast as safely
    possible" - there is no such thing as correcting hypotension too
    fast in this model.

    Hypertensive emergency (this file): the target is a MOVING FLOOR
    (patient.map_floor_mmhg, computed by hypertensive_twin.py per the
    cited NHLBI staged-reduction protocol - see hypertensive_constants.py).
    "MAP too high" does NOT mean "correct it as fast as possible" -
    precipitous reduction risks cerebral/coronary/renal hypoperfusion by
    overwhelming vascular autoregulation. Undershooting BELOW the floor
    is therefore treated as a safety event of its own, symmetric in
    severity to the "genuinely critical MAP" tier, not merely "slightly
    off target."

A second, independent deterministic safety layer (NOT part of the PD
law, applied after it) caps or stops the infusion based on the twin's
cyanide exposure flags - matching the project's standing design
decision that toxicity management is a deterministic constraint, never
an AI-learned behavior.
===============================================================================
"""

import numpy as np

import hypertensive_constants as hc
from expert_controller import TARGET_DEADBAND
from nitroprusside_pharmacokinetics import mass_rate_to_flow_ml_hr

# [DERIVED] Absolute "genuinely critical, still rising" MAP ceiling,
# converted from the cited ACC/AHA hypertensive-crisis BP threshold
# (>180/110 mmHg) via hypertensive_constants.map_from_sbp_dbp() - the
# same conversion already used to build the Urgency/Emergency profile
# bands. Crossing this, rega1rdless of where the staged floor currently
# sits, warrants the hard/fast tier - a genuinely malignant, still
# life-threatening pressure should not wait on the gentler staged pace.
EMERGENCY_MAP_MMHG_HIGH = hc.map_from_sbp_dbp(hc.CRISIS_SBP_MMHG, hc.CRISIS_DBP_LOW_MMHG)

# [ASSUMPTION, EMPIRICALLY CONSTRAINED] No literature figure gives a
# specific mcg/kg/min per-titration-cycle increment for nitroprusside
# the way septic-shock norepinephrine literature does
# (expert_controller.MAX_STEP_MCG_KG_MIN is [CITED, low end] for that
# reason) - the FDA label only says "titrate every few minutes" without
# a numeric step.
#
# An initial value proportioned the SAME way relative to nitroprusside's
# max dose (10 mcg/kg/min) as the septic controller's step is to ITS
# max dose (3.0 mcg/kg/min, corrected 2026-10-05 from a since-fixed
# untraceable 3.3 - see constants.NOREPINEPHRINE_DOSE_RANGE_MCG_KG_MIN)
# - i.e. ~1.67% of max per tick, 0.15 - was
# tried first and FAILED an empirical safety check: at 1 tick/second,
# that step lets the controller reach near-max dose within ~60 seconds
# if it keeps saturating, which is safe for norepinephrine (there is no
# such thing as correcting hypotension too fast) but is NOT safe here,
# because nitroprusside's near-instant onset (~2 min half-life) means
# that dose is enough to crash MAP tens of mmHg through the staged
# safety floor within the same minute - verified directly: the
# Urgency/Emergency profiles breached the floor by 19-48 mmHg during
# testing at that step size. Cut by >10x to 0.01, which was re-verified
# (see the project's smoke test) to track the floor within its own
# deadband instead of blowing through it. This value should be treated
# as provisional, not literature-derived, pending the kind of broader
# seeded-sweep validation expert_controller.py's own KD received.
MAX_STEP_MCG_KG_MIN = 0.01

# [ASSUMPTION, UNVALIDATED] Proportional/derivative gains. Ported from
# the STRUCTURE of expert_controller.py's KP/KD, but NOT empirically
# retuned against measured overshoot the way that module's KD was
# (see its docstring - that took a real empirical sweep across seeds).
# This is flagged explicitly as a follow-up: before trusting this
# controller's overshoot/undershoot behavior quantitatively, run the
# same kind of seeded-sweep verification against map_floor_mmhg that
# the septic controller's KD change was validated with.
KP = 0.0015   # mcg/kg/min per mmHg of error
KD = 0.02     # mcg/kg/min per (mmHg/tick) of trend

# [ASSUMPTION] mmHg below the floor (i.e. how far UNDER-shooting the
# safe floor) that triggers the hard "correcting too fast" safety tier.
# Same role/magnitude convention as expert_controller.BIG_OVERSHOOT_MARGIN_MMHG.
FLOOR_BREACH_MARGIN_MMHG = 5.0


def hypertensive_expert_controller(patient, max_flow=hc.MAX_FLOW_ML_HR,
                                    min_flow=hc.MIN_FLOW_ML_HR,
                                    deadband=TARGET_DEADBAND,
                                    max_step_mcg_kg_min=MAX_STEP_MCG_KG_MIN):
    """
    max_step_mcg_kg_min is exposed (rather than hardcoded to the module
    constant) so dataset generation can simulate a LESS careful titrator
    for a subset of patients - a larger step per decision, on top of
    generate_hypertensive_dataset.py's separate response-delay
    mechanism - both model realistic imperfect human titration, not the
    idealized instant/tiny-step behavior this module's own default
    represents. Ordinary use (the real controller, e.g. via
    esp32_bridge.py) should never override this.
    """

    step_noise_threshold = 0.02 * max_step_mcg_kg_min

    bp = patient.map
    flow = patient.flow_rate
    weight_kg = patient.weight_kg
    floor = patient.map_floor_mmhg  # the moving target - see module docstring

    bp_error = bp - floor  # positive = still above the current safe floor (needs more drug)
    bp_trend = patient.smoothed_bp - patient.previous_smoothed_bp

    def step_to_flow(step_mcg_kg_min):
        return mass_rate_to_flow_ml_hr(step_mcg_kg_min * weight_kg)

    # ------------------------------
    # Control law
    #
    # NOTE, deliberately different from expert_controller.py: there is
    # NO fast-ramp-up tier for "MAP is critically high" here. That
    # pattern is correct for norepinephrine (escalate hard, hypotension
    # can't be corrected too fast) but is actively WRONG for
    # nitroprusside - the FDA label's own protocol is "start low,
    # titrate every few minutes" with no stated emergency exception,
    # precisely because its potency/speed make a fast ramp dangerous
    # regardless of how high MAP currently is (verified empirically -
    # see MAX_STEP_MCG_KG_MIN above). Severity is instead expressed
    # entirely through the staged floor (which severity-graded
    # `presenting_map_mmhg` sets), not through a faster rate of
    # approach. EMERGENCY_MAP_MMHG_HIGH is applied ONLY as a label
    # overlay at the end of this function (same pattern as
    # expert_controller.py's spo2<90 hypoxia overlay) - it flags the
    # situation for review without ever changing `flow` directly.
    # ------------------------------
    if bp < floor - deadband - FLOOR_BREACH_MARGIN_MMHG:
        # Correcting faster than the cited safe-reduction pace allows -
        # this is the clinically important branch this controller exists
        # to enforce (see module docstring). Cut the rate hard, same
        # urgency register as the "Emergency" tier above, just triggered
        # by the opposite direction of error.
        mode = "Decrease"
        clinical_explanation = (
            f"MAP fell below the safe staged floor ({floor:.1f}) - "
            f"reducing quickly to avoid hypoperfusion from over-correction."
        )
        flow = max(flow - step_to_flow(max_step_mcg_kg_min), min_flow)

    else:
        error = bp - floor           # positive = still needs more drug
        derivative = bp_trend        # positive = MAP still rising -> lean toward more drug
        raw_step_mcg_kg_min = KP * error + KD * derivative
        step_mcg_kg_min = max(-max_step_mcg_kg_min, min(raw_step_mcg_kg_min, max_step_mcg_kg_min))
        flow = float(np.clip(flow + step_to_flow(step_mcg_kg_min), min_flow, max_flow))

        if abs(bp - floor) <= deadband:
            mode = "Maintain"
            if step_mcg_kg_min > step_noise_threshold:
                clinical_explanation = f"At the safe floor, proactive increase (trend={bp_trend:+.1f})."
            elif step_mcg_kg_min < -step_noise_threshold:
                clinical_explanation = f"At the safe floor, proactive decrease (trend={bp_trend:+.1f})."
            else:
                clinical_explanation = "MAP stable at the current staged floor."

        elif bp > floor + deadband:
            if step_mcg_kg_min >= 0:
                mode = "Increase"
                clinical_explanation = f"Above the current safe floor, PD increase (error={error:.1f})."
            else:
                mode = "Decrease"
                clinical_explanation = (
                    f"Above the floor but falling fast - anticipatory decrease "
                    f"to avoid breaching it (trend={bp_trend:+.1f})."
                )

        else:  # bp < floor - deadband, within FLOOR_BREACH_MARGIN_MMHG - gentle correction back up
            if step_mcg_kg_min <= 0:
                mode = "Decrease"
                clinical_explanation = f"Slightly below floor, PD decrease (error={-error:.1f})."
            else:
                mode = "Increase"
                clinical_explanation = (
                    f"Slightly below the floor but rising fast - anticipatory increase "
                    f"(trend={bp_trend:+.1f})."
                )

    # Malignant-range alert overlay - flags for review only, never
    # touches flow (see the "Control law" comment above for why this
    # drug gets no fast-ramp exception the way norepinephrine does).
    if bp > EMERGENCY_MAP_MMHG_HIGH and mode not in ("Decrease",):
        clinical_explanation = (
            f"Malignant-range MAP (>{EMERGENCY_MAP_MMHG_HIGH:.0f}, "
            f"ACC/AHA hypertensive-crisis threshold) - flagged for review. {clinical_explanation}"
        )
        mode = "Emergency"

    # ------------------------------
    # Deterministic cyanide-toxicity safety overlay - applied LAST,
    # after the PD law, and can only ever REDUCE flow, never increase
    # it. This is intentionally not folded into the PD math above: it
    # is a hard constraint on what's SAFE to deliver, independent of
    # what the haemodynamic control law would otherwise choose - same
    # "AI/controller proposes, deterministic safety layer disposes"
    # separation already used for the pump firmware's own watchdog/
    # limit-switch layer.
    # ------------------------------
    if patient.cyanide_buffer_exceeded:
        mode = "Toxicity-Stop"
        clinical_explanation = (
            "Cumulative nitroprusside exceeds the body's methemoglobin "
            "buffering capacity (FDA label: ~10 mg/kg) - stopping infusion."
        )
        flow = min_flow

    elif patient.cyanide_rate_risk_active:
        rate_cap_flow = step_to_flow(hc.CYANIDE_RATE_RISK_THRESHOLD_MCG_KG_MIN)
        if flow > rate_cap_flow:
            mode = "Toxicity-Cap"
            clinical_explanation = (
                f"Cumulative dose past {hc.CYANIDE_RATE_RISK_CUMULATIVE_MCG_KG:.0f} mcg/kg - "
                f"capping rate at {hc.CYANIDE_RATE_RISK_THRESHOLD_MCG_KG_MIN:.0f} mcg/kg/min "
                f"per the FDA label's cyanide-generation threshold."
            )
            flow = rate_cap_flow

    return {
        "Mode": mode,
        "ClinicalExplanation": clinical_explanation,
        "RecommendedFlow": round(flow, 2),
        "MAP_Floor": round(floor, 2),
        "BP_Error": round(bp_error, 2),
        "BP_Trend": round(bp_trend, 2),
        "CyanideRateRisk": patient.cyanide_rate_risk_active,
        "CyanideBufferExceeded": patient.cyanide_buffer_exceeded,
    }
