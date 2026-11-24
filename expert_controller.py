"""
===============================================================================
EXPERT_CONTROLLER.PY
Smart AI Infusion Pump - Mechanistic Digital Twin

Purpose
-------
This module implements the rule-based expert controller responsible for
adjusting the norepinephrine infusion rate throughout the digital twin
simulation.

The controller continuously evaluates the patient's haemodynamic state
using observable physiological measurements and determines an
appropriate infusion rate that maintains the target mean arterial
pressure (MAP) while avoiding unnecessary oscillations or overshoot.

Control philosophy
------------------
Clinical dosing decisions are computed in weight-based units
(mcg/kg/min), consistent with standard norepinephrine titration
practice. The calculated dose adjustment is converted to the equivalent
pump flow rate (mL/hr) only at the final stage before application.

The controller combines threshold-based emergency rules with a
proportional-derivative control strategy to provide both rapid response
during severe hypotension and smooth dose adjustment near the target
pressure.

All modelling assumptions and parameter justifications are documented
in constants.py.
===============================================================================
"""

import numpy as np

import constants as c
from pharmacokinetics import mass_rate_to_flow_ml_hr

TARGET_DEADBAND = 3.0  # mmHg - shared definition of "within target range"

# [ASSUMPTION] see module docstring - not independently cited like the
# 65 mmHg Surviving Sepsis Campaign default target itself.
EMERGENCY_MAP_MMHG = 50.0             # absolute - dangerous at any target
SAFE_MARGIN_BELOW_TARGET_MMHG = 5.0   # relative - "close to goal" at any target

# [CITED, low end] standard ICU norepinephrine titration increment -
# see module docstring for the time-compression caveat on applying it
# every tick rather than every 5-15 minutes.
MAX_STEP_MCG_KG_MIN = 0.05

# [ASSUMPTION] tuned so ~25 mmHg of error saturates the step at
# MAX_STEP_MCG_KG_MIN.
KP = 0.002   # mcg/kg/min per mmHg of error

# [DERIVED/RETUNED] KD was originally carried over from V1 unchanged
# (12.5x KP) when V1 used a coarser control tick. At this model's
# per-second tick rate, bp_trend (mmHg per SINGLE SECOND) is tiny even
# during a fast MAP rise, so the old KD=0.025 gave the anticipatory
# braking term almost no weight relative to the proportional term -
# the controller kept commanding "increase" right up to (and past) the
# target before the error term flipped sign, producing several mmHg of
# real, measured overshoot (verified empirically: peak MAP 65-73 vs a
# target of 65 across multiple seeds/profiles at the old gain). Raising
# KD 6x brings peak overshoot down to ~0-2 mmHg across the same seeds
# without introducing undershoot or mode chattering - the controller
# now measurably starts tapering the dose before MAP reaches target,
# not just after crossing it.
KD = 0.15   # mcg/kg/min per (mmHg/tick) of trend

# [ASSUMPTION] mmHg above target+deadband that triggers the hard
# "significantly overshooting" tier, same role as V1's
# big_overshoot_margin, re-scaled for the new target.
BIG_OVERSHOOT_MARGIN_MMHG = 15.0

# [ASSUMPTION, user-requested compromise] While still below target,
# soften the flow CEILING (not the step size - MAX_STEP_MCG_KG_MIN is
# unchanged) as MAP gets within this many mmHg of target, landing on a
# lower ceiling right at target instead of still being clamped toward
# the full max_flow in the final stretch. Rationale: drug already
# committed to the bloodstream keeps acting for a while after a dose
# change (PK lag), so a lower ceiling near target leaves more margin
# against that residual push before the PD law's own taper (KP/KD) has
# fully caught up. Does not apply to EMERGENCY_MAP_MMHG - genuinely
# critical hypotension still gets the full max_flow ceiling.
APPROACH_DECEL_MARGIN_MMHG = 30.0
APPROACH_FLOW_CAP_ML_HR = 150.0


def _approach_flow_ceiling(bp, target, max_flow):
    """Flow ceiling to use while bp is below target: the full max_flow
    once target - bp >= APPROACH_DECEL_MARGIN_MMHG, tapering linearly
    down to APPROACH_FLOW_CAP_ML_HR right at target. At/above target
    there is nothing to cap - the ordinary PD taper already reduces
    flow from there."""
    if bp >= target:
        return max_flow
    distance = target - bp
    if distance >= APPROACH_DECEL_MARGIN_MMHG:
        return max_flow
    fraction = distance / APPROACH_DECEL_MARGIN_MMHG   # 1.0 at the far edge, 0.0 at target
    return APPROACH_FLOW_CAP_ML_HR + fraction * (max_flow - APPROACH_FLOW_CAP_ML_HR)

# "Basically zero" cutoff for wording the Maintain-zone explanation -
# same 2% of MAX_STEP proportion V1 used (0.02 out of MAX_STEP=1.0).
_STEP_NOISE_THRESHOLD = 0.02 * MAX_STEP_MCG_KG_MIN


def _confidence(bp, spo2, target, deadband):
    safe_map = target - SAFE_MARGIN_BELOW_TARGET_MMHG
    boundaries = [EMERGENCY_MAP_MMHG, safe_map, target - deadband, target + deadband]
    dist_to_boundary = min(abs(bp - b) for b in boundaries)
    bp_confidence = min(dist_to_boundary / 10.0, 1.0) * 100

    spo2_margin = max(0.0, (spo2 - 90) / 5.0)
    spo2_confidence = min(spo2_margin, 1.0) * 100

    return round(min(bp_confidence, spo2_confidence), 1)


def _stability_score(bp_error, bp_trend, shock_index, oxygen_deficit):
    penalty = (
        abs(bp_error) * 1.5
        + abs(bp_trend) * 8.0
        + max(0.0, shock_index - 1.0) * 40.0
        + oxygen_deficit * 1.0
    )
    return round(max(0.0, 100.0 - penalty), 1)


def expert_controller(patient, max_flow=c.MAX_FLOW_ML_HR, min_flow=c.MIN_FLOW_ML_HR,
                       deadband=TARGET_DEADBAND):

    bp = patient.bp  # MAP, mmHg (alias maintained by digital_twin.py)
    hr = patient.hr
    spo2 = patient.spo2
    flow = patient.flow_rate
    target = patient.target_bp
    weight_kg = patient.weight_kg

    bp_error = target - bp
    # Use the SMOOTHED trend for control decisions, not a raw one-tick
    # difference - same rationale as V1 (avoid reacting to short-term
    # physiological noise instead of genuine drug-driven trend).
    bp_trend = patient.smoothed_bp - patient.previous_smoothed_bp

    # ------------------------------
    # Feature Engineering
    # ------------------------------
    shock_index = hr / max(bp, 1)
    oxygen_deficit = 100 - spo2
    perfusion = (bp * spo2) / 100

    def step_to_flow(step_mcg_kg_min):
        """Convert a dosing step in mcg/kg/min (this patient's weight)
        into the equivalent mL/hr flow-rate change."""
        return mass_rate_to_flow_ml_hr(step_mcg_kg_min * weight_kg)

    # ------------------------------
    # AI Decision Rules
    # ------------------------------
    # Same two-hard-tier + continuous-PD structure as V1 - only the
    # units feeding `flow` changed (mcg/kg/min -> mL/hr via
    # step_to_flow(), instead of a raw mL/hr PD output).
    #
    # NOTE on hypoxia: an earlier revision escalated flow to max
    # whenever EITHER bp<EMERGENCY_MAP_MMHG OR spo2<90. That was wrong:
    # a real shock trajectory can have SpO2 lag well behind MAP
    # recovery (it drifts back up slowly - see physiology.py), so MAP
    # could already be at/above target while spo2<90 kept forcing max
    # dose indefinitely - continuing to infuse after the goal is met,
    # which this drug cannot even fix (it raises MAP, not oxygenation),
    # and which risks dangerous overshoot. A first fix added a second,
    # separate "taper" branch gated on bp vs target-deadband, but that
    # introduced a NEW hard on/off boundary with no hysteresis, which
    # chattered (flow oscillating between max and zero) whenever MAP
    # sat right at that boundary.
    #
    # The correct fix needs no new boundary at all: only escalate hard
    # when bp is GENUINELY critically low. Once bp >= EMERGENCY_MAP_MMHG,
    # dosing always runs through the same continuous PD law used
    # everywhere else in this function - which already tapers smoothly
    # as MAP approaches/exceeds target, with no discontinuity to
    # chatter around. spo2<90 only overrides the MODE label afterward
    # (for alerting), never the computed flow.
    if bp < EMERGENCY_MAP_MMHG:
        mode = "Emergency"
        clinical_explanation = f"Critical hypotension (MAP<{EMERGENCY_MAP_MMHG:.0f})."
        flow = min(flow + step_to_flow(MAX_STEP_MCG_KG_MIN), max_flow)

    elif bp > target + deadband + BIG_OVERSHOOT_MARGIN_MMHG:
        mode = "Decrease"
        clinical_explanation = "Significantly above target - reducing flow quickly."
        flow = max(flow - step_to_flow(MAX_STEP_MCG_KG_MIN), min_flow)

    else:
        # Anticipatory PD law - see V1's expert_controller.py for the
        # original overshoot bug this structure fixes: a strong trend
        # overrides current position, so the controller starts cutting
        # flow before BP even crosses target if it's climbing fast
        # (and vice versa), rather than waiting for it to cross first.
        error = target - bp          # positive = below target
        derivative = -bp_trend       # positive = still moving away from target
        raw_step_mcg_kg_min = KP * error + KD * derivative
        step_mcg_kg_min = max(-MAX_STEP_MCG_KG_MIN, min(raw_step_mcg_kg_min, MAX_STEP_MCG_KG_MIN))
        approach_ceiling = _approach_flow_ceiling(bp, target, max_flow)
        flow = float(np.clip(flow + step_to_flow(step_mcg_kg_min), min_flow, approach_ceiling))

        if abs(bp - target) <= deadband:
            mode = "Maintain"
            if step_mcg_kg_min > _STEP_NOISE_THRESHOLD:
                clinical_explanation = f"Within deadband, proactive increase (trend={bp_trend:+.1f})."
            elif step_mcg_kg_min < -_STEP_NOISE_THRESHOLD:
                clinical_explanation = f"Within deadband, proactive decrease (trend={bp_trend:+.1f})."
            else:
                clinical_explanation = "Blood pressure stable within target deadband."

        elif bp < target - deadband:
            if step_mcg_kg_min >= 0:
                mode = "Increase"
                zone = "safe range" if bp < target - SAFE_MARGIN_BELOW_TARGET_MMHG else "target"
                clinical_explanation = f"Below {zone}, PD increase (error={error:.1f}, trend={bp_trend:+.1f})."
            else:
                mode = "Decrease"
                clinical_explanation = (f"Below target but rising fast - anticipatory decrease "
                                         f"to prevent overshoot (trend={bp_trend:+.1f}).")

        else:  # bp > target + deadband, within BIG_OVERSHOOT_MARGIN_MMHG
            if step_mcg_kg_min <= 0:
                mode = "Decrease"
                clinical_explanation = f"Above target, PD decrease (error={-error:.1f}, trend={bp_trend:+.1f})."
            else:
                mode = "Increase"
                clinical_explanation = (f"Above target but falling fast - anticipatory increase "
                                         f"to prevent undershoot (trend={bp_trend:+.1f}).")

    # Hypoxia alert overlay: more norepinephrine cannot fix low SpO2,
    # so dosing above is governed entirely by MAP (as it should be) -
    # but a hypoxic patient still needs to be flagged prominently,
    # regardless of what the MAP-driven mode/dose happens to be. This
    # never touches `flow`.
    if spo2 < 90 and mode != "Emergency":
        clinical_explanation = f"SpO2 critical (<90) - flagged for review. {clinical_explanation}"
        mode = "Emergency"

    confidence = _confidence(bp, spo2, target, deadband)
    stability = _stability_score(bp_error, bp_trend, shock_index, oxygen_deficit)

    return {
        "Mode": mode,
        "ClinicalExplanation": clinical_explanation,
        "RecommendedFlow": round(flow, 2),
        "ShockIndex": round(shock_index, 2),
        "OxygenDeficit": round(oxygen_deficit, 2),
        "PerfusionIndex": round(perfusion, 2),
        "BP_Error": round(bp_error, 2),
        "BP_Trend": round(bp_trend, 2),
        "Confidence": confidence,
        "StabilityScore": stability
    }
