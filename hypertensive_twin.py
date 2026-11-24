"""
===============================================================================
HYPERTENSIVE_TWIN.PY
Smart AI Infusion Pump - Hypertensive Emergency Digital Twin (Sodium Nitroprusside)

Orchestrates the SECOND clinical mode's chain, mirroring digital_twin.py's
structure exactly:

    Demographics -> PK (plasma concentration)         [nitroprusside-specific]
                 -> PD (drug effect, NEGATIVE)         [nitroprusside-specific]
                 -> Cardiovascular (SVR, SV, CO, MAP)  [REUSED, unchanged]
                 -> Baroreflex (HR)                    [REUSED, unchanged]
                 -> SpO2                               [REUSED, unchanged]
                 -> Staged MAP-reduction safety floor  [new to this mode]
                 -> Cyanide cumulative-exposure monitor [new to this mode,
                                                          deterministic - NOT
                                                          an AI-learned feature]

This file is the FOUNDATION step agreed before building the hypertensive
expert controller or generating AI training data (see project design
discussion): a physiologically-grounded twin that responds correctly to
an externally-supplied flow_rate, exactly like digital_twin.VirtualPatient
does. It does not decide dosing itself - same separation of concerns as
the septic-shock side (AI/controller decides, twin simulates the body).

Why cardiovascular.py and physiology.py needed ZERO changes
-------------------------------------------------------------
compute_svr() is SVR = SVR0 * (1 + E) - never assumed E >= 0, so a
negative E (from nitroprusside_pharmacodynamics.py) already reduces SVR
correctly. update_heart_rate() and update_spo2() only care about MAP
relative to a setpoint/threshold, not which drug produced that MAP. The
entire cardiovascular/physiology engine is drug-agnostic by construction -
only PK (concentration kinetics) and PD (concentration -> effect) needed
drug-specific siblings.
===============================================================================
"""

import random
import numpy as np

import constants as c
import hypertensive_constants as hc
from hypertensive_patient_generator import generate_patient_params
from hypertensive_constants import PROFILE_WEIGHTS
from nitroprusside_pharmacokinetics import update_plasma_concentration
from nitroprusside_pharmacodynamics import compute_drug_effect
from cardiovascular import compute_svr, compute_stroke_volume, compute_cardiac_output, compute_map
from physiology import update_heart_rate, update_spo2


def _choose_profile(rng):
    names = list(PROFILE_WEIGHTS.keys())
    weights = [PROFILE_WEIGHTS[n] for n in names]
    return rng.choices(names, weights=weights)[0]


def map_reduction_floor_mmhg(elapsed_hours, presenting_map_mmhg):
    """
    The staged safety floor: the LOWEST MAP considered safe to reduce
    to at a given elapsed time since treatment started. A controller
    should treat this as "don't push MAP below this yet", NOT as a
    target to aim for directly - the floor loosens over time,
    representing progressively more aggressive safe correction as the
    cited NHLBI-derived staged protocol allows (see
    hypertensive_constants.py for the full citation).

    Three phases:
      0-1h    : floor glides from the presenting MAP down to
                presenting_map * (1 - 25%) - "no more than ~25% in hour 1".
      1-6h    : floor glides further down to the ~160/100-110 mmHg
                equivalent MAP band (~120-127 mmHg).
      6-48h   : floor glides the rest of the way to the normal MAP
                range floor - "gradual normalization over 24-48h".
      >48h    : floor holds at the normal MAP range floor.
    """

    stage1_end_floor = presenting_map_mmhg * (1.0 - hc.STAGE1_MAX_REDUCTION_FRACTION)
    stage2_end_floor = sum(hc.STAGE2_TARGET_MAP_BAND_MMHG) / 2.0
    stage3_end_floor = c.MAP_NORMAL_MMHG[0]

    if elapsed_hours <= hc.STAGE1_DURATION_HOURS:
        frac = elapsed_hours / hc.STAGE1_DURATION_HOURS
        return presenting_map_mmhg + frac * (stage1_end_floor - presenting_map_mmhg)

    if elapsed_hours <= hc.STAGE2_END_HOURS:
        frac = (elapsed_hours - hc.STAGE1_DURATION_HOURS) / (
            hc.STAGE2_END_HOURS - hc.STAGE1_DURATION_HOURS
        )
        return stage1_end_floor + frac * (stage2_end_floor - stage1_end_floor)

    if elapsed_hours <= hc.STAGE3_END_HOURS:
        frac = (elapsed_hours - hc.STAGE2_END_HOURS) / (
            hc.STAGE3_END_HOURS - hc.STAGE2_END_HOURS
        )
        return stage2_end_floor + frac * (stage3_end_floor - stage2_end_floor)

    return stage3_end_floor


class VirtualHypertensivePatient:

    def __init__(self, patient_id, dt_seconds=1.0, profile=None, seed=None):
        self.patient_id = patient_id
        self.dt_seconds = dt_seconds
        self.dt_min = dt_seconds / 60.0

        rng = random.Random(seed) if seed is not None else random.Random()

        if profile is None:
            profile = _choose_profile(rng)

        params = generate_patient_params(profile, rng)

        demo = params["demographics"]
        self.age = demo.age
        self.sex = demo.sex
        self.weight_kg = demo.weight_kg
        self.height_cm = demo.height_cm
        self.bmi = demo.bmi

        self.profile = profile
        self.saps_ii = None  # SAPS II is a septic-shock-specific severity score - no
                              # equivalent here. Kept as an attribute (not omitted) so
                              # dashboard code shared with the septic-shock twin doesn't
                              # need a special case to display patient info generically.

        # PK parameters (nitroprusside-specific: Vd is weight-scaled,
        # unlike norepinephrine's fixed population Vd - see
        # hypertensive_patient_generator.py)
        self.clearance_l_min = params["clearance_l_min"]
        self.vd_l = params["vd_l"]
        self.plasma_concentration = 0.0  # Cp, mcg/L - starts drug-naive

        # CV baselines
        self.svr0 = params["svr0_dyn_s_cm5"]
        self.sv0_ml = params["sv0_ml"]
        self.hr_set = params["hr_set_bpm"]
        self.bv0_fraction = params["bv0_fraction"]
        self.blood_volume_fraction = self.bv0_fraction
        self.map_set = params["map_set_mmhg"]

        # The MAP this patient PRESENTED at, before any treatment -
        # anchors the staged reduction-floor schedule for this whole
        # episode. Does not change after initialization.
        self.presenting_map_mmhg = params["presenting_map_mmhg"]
        self.hours_since_start = 0.0
        self.map_floor_mmhg = self.presenting_map_mmhg

        # Illness recovery (vascular tone normalization) - see
        # hypertensive_constants.py. svr0 drifts toward
        # HTN_RECOVERY_SVR0_CEILING_DYN_S_CM5 once MAP has held at/above
        # the (descending) floor long enough; any breach pauses recovery.
        self.time_safely_controlled_hours = 0.0

        # Initial state at zero drug
        self.drug_effect = 0.0
        self.svr = self.svr0
        self.hr = self.hr_set
        self.sv = compute_stroke_volume(self.sv0_ml, self.blood_volume_fraction)
        self.co = compute_cardiac_output(self.hr, self.sv)
        self.map = compute_map(self.co, self.svr)
        self.previous_map = self.map

        # Backward-compatible aliases (bp == map), same convention as
        # digital_twin.VirtualPatient, for eventual dashboard/pipeline reuse
        self.bp = self.map
        self.previous_bp = self.previous_map
        self.smoothed_bp = self.map
        self.previous_smoothed_bp = self.map

        self.spo2 = float(np.clip(rng.uniform(94, 98), 70, 100))

        # Pump state
        self.flow_rate = 0.0  # mL/hr, externally driven - same contract
                               # as digital_twin.VirtualPatient.flow_rate

        # ---------------------------------------------------------
        # Cyanide cumulative-exposure tracking - DETERMINISTIC safety
        # accounting only (see hypertensive_constants.py citations).
        # This class only TRACKS and REPORTS these flags; it does not
        # self-limit flow_rate - actuation belongs to the (future)
        # controller, same separation as the septic-shock side.
        # ---------------------------------------------------------
        self.cumulative_nitroprusside_mcg = 0.0
        self.cyanide_buffer_exceeded = False   # total dose > CYANIDE_BUFFER_CAPACITY_MG_KG
        self.cyanide_rate_risk_active = False  # cumulative + current-rate combined risk

    # =====================================================

    def _update_cyanide_tracking(self, mass_rate_mcg_min):
        self.cumulative_nitroprusside_mcg += mass_rate_mcg_min * self.dt_min

        cumulative_mcg_per_kg = self.cumulative_nitroprusside_mcg / self.weight_kg
        current_rate_mcg_kg_min = mass_rate_mcg_min / self.weight_kg

        self.cyanide_buffer_exceeded = (
            cumulative_mcg_per_kg >= hc.CYANIDE_BUFFER_CAPACITY_MG_KG * 1000.0
        )
        self.cyanide_rate_risk_active = (
            cumulative_mcg_per_kg > hc.CYANIDE_RATE_RISK_CUMULATIVE_MCG_KG
            and current_rate_mcg_kg_min > hc.CYANIDE_RATE_RISK_THRESHOLD_MCG_KG_MIN
        )

    # =====================================================

    def update(self):
        from nitroprusside_pharmacokinetics import flow_rate_to_mass_rate

        # PK
        self.plasma_concentration = update_plasma_concentration(
            self.plasma_concentration, self.flow_rate,
            self.vd_l, self.clearance_l_min, self.dt_min
        )

        # Cyanide accounting - tracked off the same mass rate the PK
        # step just used, before PD/CV run, so it reflects exactly what
        # was actually infused this tick.
        self._update_cyanide_tracking(flow_rate_to_mass_rate(self.flow_rate))

        # PD (NEGATIVE effect - vasodilation)
        self.drug_effect = compute_drug_effect(self.plasma_concentration)

        # Cardiovascular - REUSED, unchanged functions. compute_svr's
        # SVR0*(1+E) with a negative E is what actually produces the
        # opposite-direction (falling SVR/MAP) response.
        self.svr = compute_svr(self.svr0, self.drug_effect)
        self.sv = compute_stroke_volume(self.sv0_ml, self.blood_volume_fraction)
        self.co = compute_cardiac_output(self.hr, self.sv)
        self.previous_map = self.map
        self.map = compute_map(self.co, self.svr)
        self.map = float(np.clip(self.map, 20, 250))  # physiological floor/ceiling
                                                        # (raised ceiling vs the septic
                                                        # twin's 200, since this
                                                        # population presents higher)

        # Staged safety floor - updates every tick as the episode clock
        # advances. This is information for the (future) controller to
        # respect, not something this twin enforces on itself.
        self.hours_since_start += self.dt_min / 60.0
        self.map_floor_mmhg = map_reduction_floor_mmhg(
            self.hours_since_start, self.presenting_map_mmhg
        )

        # Illness recovery (vascular tone normalization) - see
        # hypertensive_constants.py. Mirrors digital_twin.py's own
        # mechanism for the opposite condition: without this, svr0 never
        # improves, so a controller correctly chasing the ever-stricter
        # descending floor has to administer an ever-increasing dose
        # forever, with no plateau - exactly what was observed running
        # this for real over several hours. "Safely controlled" here
        # means MAP >= floor (this mode has no fixed target/deadband to
        # check against); any breach resets the clock, pausing recovery.
        if self.map >= self.map_floor_mmhg:
            self.time_safely_controlled_hours += self.dt_min / 60.0
        else:
            self.time_safely_controlled_hours = 0.0

        if self.time_safely_controlled_hours > hc.HTN_RECOVERY_GRACE_HOURS:
            self.svr0 += (
                (hc.HTN_RECOVERY_SVR0_CEILING_DYN_S_CM5 - self.svr0)
                * (self.dt_min / 60.0 / hc.HTN_RECOVERY_TAU_HOURS)
            )

        # Baroreceptor resetting - see hypertensive_constants.py. Without
        # this, map_set stays pinned at the ORIGINAL presenting MAP
        # forever while treatment pushes real MAP down for up to 48
        # hours, so the baroreflex gap (and therefore reflex tachycardia)
        # only ever GROWS, unbounded - exactly what was observed as HR
        # climbing past 160 bpm and still rising after hours of real use.
        # Drifts continuously toward the CURRENT MAP (not gated by safe-
        # control status, unlike svr0 above - real baroreceptor
        # adaptation responds to sustained pressure regardless of
        # whether treatment has fully succeeded yet), with a time
        # constant deliberately much slower than the acute correction
        # schedule so early, acute reflex tachycardia is preserved.
        self.map_set += (self.map - self.map_set) * (
            self.dt_min / 60.0 / hc.BARORECEPTOR_RESET_TAU_HOURS
        )

        # Baroreflex (HR). Still no hard deviation cap (unlike the
        # septic twin) - acute reflex tachycardia as MAP is brought down
        # is a real, well-documented effect of vasodilator therapy, not
        # an artifact to suppress. What changes the LONG-run behavior is
        # map_set no longer being frozen forever (immediately above) -
        # the reflex now has a chance to genuinely settle as baroreceptor
        # resetting catches up, rather than escalating indefinitely.
        self.hr = update_heart_rate(
            self.hr, self.hr_set, self.map, self.map_set, self.dt_min
        )

        # SpO2 - REUSED, unchanged. Hypertensive emergency does not
        # typically impair perfusion the way hypotension does, so this
        # will realistically stay near its recovered/normal band
        # throughout for most patients - included for pipeline/dashboard
        # parity (same attributes available in both modes), not because
        # SpO2 is a clinically central signal here.
        self.spo2 = update_spo2(self.spo2, self.map, self.dt_min)

        # Backward-compatible aliases + smoothed trend (identical
        # convention to digital_twin.VirtualPatient)
        self.bp = self.map
        self.previous_bp = self.previous_map
        self.previous_smoothed_bp = self.smoothed_bp
        self.smoothed_bp += 0.3 * (self.bp - self.smoothed_bp)
