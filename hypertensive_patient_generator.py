"""
===============================================================================
HYPERTENSIVE_PATIENT_GENERATOR.PY
Smart AI Infusion Pump - Hypertensive Emergency Digital Twin

Purpose
-------
Sibling of patient_generator.py for the hypertensive-emergency /
sodium-nitroprusside mode. Reuses PatientDemographics UNCHANGED (age,
sex, weight, height, BMI are generic - nothing about them is
norepinephrine- or septic-shock-specific), but replaces the
severity-profile logic entirely: this population is vasoconstricted
(elevated SVR, elevated MAP0) rather than vasodilated, and clearance
scales with a hypertensive patient's own weight (via Vd = ECF fraction
* weight) rather than with illness-severity score, since nitroprusside
has no equivalent of norepinephrine's SAPS-II-linked clearance data.

All parameter sources documented in hypertensive_constants.py.
===============================================================================
"""

import numpy as np

import constants as c
import hypertensive_constants as hc
from patient_generator import PatientDemographics


def generate_patient_params(profile, rng):
    """
    Generates individualized PK and cardiovascular parameters for a
    single virtual hypertensive-emergency patient. Mirrors
    patient_generator.generate_patient_params()'s structure and
    "sample the physiologically-regulated quantity (MAP0), derive the
    rest" pattern.
    """

    demo = PatientDemographics(rng)

    # -------------------------------------------------------------
    # Pharmacokinetics - Vd from extracellular-fluid approximation
    # (weight-dependent, unlike norepinephrine's fixed population Vd -
    # see hypertensive_constants.py for why this drug's Vd IS
    # weight-scaled per the FDA label's own description).
    # -------------------------------------------------------------
    vd_l = hc.ECF_FRACTION_OF_BODY_WEIGHT * demo.weight_kg

    ke_per_min = hc.NITROPRUSSIDE_KE_PER_MIN
    iiv = np.exp(rng.gauss(0, hc.CLEARANCE_INTER_INDIVIDUAL_CV))
    clearance_l_min = max(ke_per_min * vd_l * iiv, 0.1)

    # -------------------------------------------------------------
    # Baseline cardiovascular state - elevated SVR/MAP0 by profile,
    # same "sample MAP0, derive CO0, derive SV0" consistency pattern
    # the septic-shock generator uses (see patient_generator.py).
    # -------------------------------------------------------------
    svr_lo, svr_hi = hc.SVR_BY_PROFILE_DYN_S_CM5[profile]
    svr0 = rng.uniform(svr_lo, svr_hi)

    hr_set_bpm = rng.uniform(*c.HR_NORMAL_BPM)

    map0_lo, map0_hi = hc.MAP0_BY_PROFILE_MMHG[profile]
    map0_target = rng.uniform(map0_lo, map0_hi)

    co0_l_min = (map0_target - c.CVP_TYPICAL_MMHG) * 80.0 / svr0
    co0_l_min = float(np.clip(co0_l_min, 1.5, 12.0))

    sv0_ml = co0_l_min * 1000.0 / hr_set_bpm

    # [ASSUMPTION] Hypertensive emergency is not a volume-depletion
    # state the way septic shock is - no blood-volume reduction is
    # modeled here (unlike patient_generator.py's bv0_fraction, which
    # falls with septic severity). Held at 1.0 for every profile.
    bv0_fraction = 1.0

    # [ASSUMPTION, physiologically motivated] Baroreflex resetting in
    # chronic hypertension is a well-established concept (the
    # baroreceptors adapt to operate around the chronically elevated
    # pressure rather than the population-normal range) - modeled here
    # by setting each patient's defended map_set to their OWN
    # presenting MAP0, not the population-normal range
    # patient_generator.py uses for septic patients. This is what makes
    # reflex tachycardia emerge naturally as MAP is lowered by
    # treatment (via physiology.update_heart_rate(), completely
    # unchanged) - a real, well-documented side effect of rapid
    # vasodilator-driven BP reduction - without any extra code.
    map_set_mmhg = map0_target

    return {
        "demographics": demo,
        "profile": profile,
        "clearance_l_min": clearance_l_min,
        "vd_l": vd_l,
        "svr0_dyn_s_cm5": svr0,
        "sv0_ml": sv0_ml,
        "hr_set_bpm": hr_set_bpm,
        "bv0_fraction": bv0_fraction,
        "map_set_mmhg": map_set_mmhg,
        "presenting_map_mmhg": map0_target,
    }
