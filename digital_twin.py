"""
===========================================================
DIGITAL_TWIN.PY
Smart AI Infusion Pump - Mechanistic Digital Twin V2

Orchestrates the full chain, each stage in its own cited module:

    Demographics -> PK (plasma concentration)
                 -> PD (drug effect)
                 -> Cardiovascular (SVR, SV, CO, MAP)
                 -> Baroreflex (HR)
                 -> SpO2

This REPLACES the V1 digital_twin.py's ad-hoc "vascular tone" gain
model with the literature-grounded chain built in pharmacokinetics.py,
pharmacodynamics.py, cardiovascular.py, and physiology.py. See
constants.py for full citations on every numeric value used here.

Target MAP is now the Surviving Sepsis Campaign's actual published
resuscitation target (>=65 mmHg) rather than the earlier, uncited
100-110 mmHg range - this was a known gap flagged during literature
review and is fixed here.

Backward-compatibility note: this class exposes .bp / .previous_bp /
.smoothed_bp aliases (all equal to MAP) so it can eventually be dropped
into the existing controller/dashboard pipeline, which was built around
"bp" terminology. Wiring this into expert_controller.py, dashboard.py,
and dataset generation is the next step, NOT done in this file - the
realistic flow-rate range alone (0-190 mL/hr vs the old 0-5 mL/hr)
means the controller's step sizes need retuning before that
integration, and that deserves its own verification pass.
===========================================================
"""

import random
import numpy as np

import constants as c
from patient_generator import generate_patient_params, PROFILE_WEIGHTS
from pharmacokinetics import update_plasma_concentration
from pharmacodynamics import compute_drug_effect
from cardiovascular import compute_svr, compute_stroke_volume, compute_cardiac_output, compute_map
from physiology import update_heart_rate, update_spo2
from expert_controller import TARGET_DEADBAND

# [CITED] Surviving Sepsis Campaign target MAP for vasopressor
# resuscitation: >=65 mmHg. This is THE standard clinical target for
# norepinephrine titration in septic shock - see Surviving Sepsis
# Campaign International Guidelines (2021 update, and consistent across
# prior versions).
TARGET_MAP_MMHG = 65.0


def _choose_profile(rng):
    names = list(PROFILE_WEIGHTS.keys())
    weights = [PROFILE_WEIGHTS[n] for n in names]
    return rng.choices(names, weights=weights)[0]


class VirtualPatient:

    def __init__(self, patient_id, dt_seconds=1.0, profile=None,
                 seed=None, vtbi=None, target_map=None):
        self.patient_id = patient_id
        self.dt_seconds = dt_seconds
        self.dt_min = dt_seconds / 60.0

        rng = random.Random(seed) if seed is not None else random.Random()

        if profile is None:
            profile = _choose_profile(rng)

        params = generate_patient_params(profile, rng)

        # Demographics (stored/reported; NOT wired into adult PK per
        # Beloeil et al. - see constants.py - EXCEPT age, which drives
        # clearance via the Esler et al. mechanism, already applied
        # inside generate_patient_params).
        demo = params["demographics"]
        self.age = demo.age
        self.sex = demo.sex
        self.weight_kg = demo.weight_kg
        self.height_cm = demo.height_cm
        self.bmi = demo.bmi

        self.profile = profile
        self.saps_ii = params["saps_ii"]

        # PK parameters
        self.clearance_l_min = params["clearance_l_min"]
        self.vd_l = params["vd_l"]
        self.plasma_concentration = 0.0  # Cp, mcg/L - starts drug-naive

        # CV baselines
        self.svr0 = params["svr0_dyn_s_cm5"]
        self.sv0_ml = params["sv0_ml"]
        self.hr_set = params["hr_set_bpm"]
        self.bv0_fraction = params["bv0_fraction"]
        self.blood_volume_fraction = self.bv0_fraction  # ~constant; no fluids modeled
        self.map_set = params["map_set_mmhg"]

        # Illness recovery (weaning) state - see constants.py. svr0
        # drifts toward svr0_ceiling once MAP has held at target long
        # enough; time_in_target_min resets on any relapse out of the
        # deadband, pausing recovery.
        self.svr0_ceiling = c.ILLNESS_RECOVERY_SVR0_CEILING_DYN_S_CM5
        self.time_in_target_min = 0.0

        # Target MAP - defaults to the cited Surviving Sepsis Campaign
        # value (see module docstring), but a caller may pass any value
        # within the SEPSISPAM-studied range (constants.TARGET_MAP_RANGE_MMHG)
        # to individualize the goal, e.g. for a chronically hypertensive
        # patient. expert_controller.py reads this dynamically per patient,
        # so any value here is honored correctly by actuation; only the
        # trained AI advisory models need per-target training coverage,
        # which generate_dataset.py provides by sampling TargetBP across
        # this same range.
        self.target_bp = target_map if target_map is not None else TARGET_MAP_MMHG

        # Initial state at zero drug - severity naturally falls out of
        # the SVR0/CO relationship, no artificial "start low" needed.
        self.drug_effect = 0.0
        self.svr = self.svr0
        self.hr = self.hr_set
        self.sv = compute_stroke_volume(self.sv0_ml, self.blood_volume_fraction)
        self.co = compute_cardiac_output(self.hr, self.sv)
        self.map = compute_map(self.co, self.svr)
        self.previous_map = self.map

        # Backward-compatible aliases (bp == map) for eventual pipeline reuse
        self.bp = self.map
        self.previous_bp = self.previous_map
        self.smoothed_bp = self.map
        self.previous_smoothed_bp = self.map

        self.spo2 = float(np.clip(rng.uniform(94, 98), 70, 100))

        # Pump state
        self.flow_rate = 0.0  # mL/hr - infusion starts at zero, titrated up

        # VTBI/KVO tracking (unchanged concept from V1 - real infusion
        # pump convention, see constants.py)
        self.vtbi = vtbi if vtbi is not None else 1e9
        self.delivered_volume = 0.0
        self.infusion_complete = False

    # =====================================================

    def update_delivered_volume(self):
        if self.infusion_complete:
            return
        self.delivered_volume += self.flow_rate * (self.dt_seconds / 3600.0)
        if self.delivered_volume >= self.vtbi:
            self.infusion_complete = True

    # =====================================================

    def update(self):
        # PK
        self.plasma_concentration = update_plasma_concentration(
            self.plasma_concentration, self.flow_rate,
            self.vd_l, self.clearance_l_min, self.dt_min
        )

        # PD
        self.drug_effect = compute_drug_effect(self.plasma_concentration)

        # Cardiovascular
        self.svr = compute_svr(self.svr0, self.drug_effect)
        self.sv = compute_stroke_volume(self.sv0_ml, self.blood_volume_fraction)
        self.co = compute_cardiac_output(self.hr, self.sv)
        self.previous_map = self.map
        self.map = compute_map(self.co, self.svr)
        self.map = float(np.clip(self.map, 20, 200))  # physiological floor/ceiling

        # Illness recovery (weaning) - see constants.py. Affects NEXT
        # tick's svr0, not this one, consistent with previous_map above.
        # One-sided check (MAP >= target - deadband), not a symmetric
        # band: as svr0 recovers, MAP naturally drifts ABOVE target
        # once flow reaches 0 (there is no way to pull it back down -
        # norepinephrine only raises MAP) - that is the correct,
        # desired sign of recovery, not instability, and must not reset
        # the clock. Only a drop back toward/below the lower bound
        # (genuine hypotension recurring) should pause recovery.
        if self.map >= self.target_bp - TARGET_DEADBAND:
            self.time_in_target_min += self.dt_min
        else:
            self.time_in_target_min = 0.0

        if self.time_in_target_min > c.ILLNESS_RECOVERY_GRACE_MIN:
            self.svr0 += (
                (self.svr0_ceiling - self.svr0)
                * (self.dt_min / c.ILLNESS_RECOVERY_TAU_MIN)
            )

        # Baroreflex (HR). Always capped now - see constants.py's
        # "[FIX, 2026-09-27]" note. A tighter cap engages once MAP is
        # already at/near target (curbing the unrealistic "still
        # elevated forever" long tail); a looser but still-bounded cap
        # applies during genuine acute hypotension, preserving a strong,
        # fast compensatory HR rise (which helps restore cardiac output
        # in a real emergency) without letting an unlucky hr_set/map_set
        # combination compound into a clinically implausible spike.
        hr_cap = (
            c.BAROREFLEX_MAX_DEVIATION_MMHG
            if self.map >= self.target_bp - TARGET_DEADBAND
            else c.BAROREFLEX_MAX_DEVIATION_ACUTE_MMHG
        )
        self.hr = update_heart_rate(
            self.hr, self.hr_set, self.map, self.map_set, self.dt_min,
            max_deviation_mmhg=hr_cap
        )

        # SpO2
        self.spo2 = update_spo2(self.spo2, self.map, self.dt_min)

        # VTBI tracking
        self.update_delivered_volume()

        # Backward-compatible aliases + smoothed trend (EMA, alpha=0.3 -
        # same rationale as V1: averages out fast noise for control
        # decisions while raw values stay available for safety checks)
        self.bp = self.map
        self.previous_bp = self.previous_map
        self.previous_smoothed_bp = self.smoothed_bp
        self.smoothed_bp += 0.3 * (self.bp - self.smoothed_bp)
