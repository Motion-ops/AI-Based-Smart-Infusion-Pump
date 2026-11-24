"""
===============================================================================
PATIENT_GENERATOR.PY
Smart AI Infusion Pump - Mechanistic Digital Twin

Purpose
-------
This module generates virtual patients for the digital twin simulation.
Each patient is assigned demographic characteristics together with
individualized pharmacokinetic (PK) and cardiovascular (CV) parameters
that reflect their clinical condition.

The generated parameters serve as the initial physiological state of the
patient before any norepinephrine infusion is administered.

Model philosophy
----------------
Rather than assigning identical physiological parameters to every
patient, inter-patient variability is introduced so that each simulated
patient responds differently to treatment, similar to what is observed
in real intensive care unit (ICU) populations.

Patient-specific parameters are derived from published population
pharmacokinetic relationships where available, while clearly identified
assumptions are used only when supporting literature was unavailable.

All parameter sources and assumptions are fully documented in
constants.py.
===============================================================================
"""

import random
import numpy as np

import constants as c


PROFILE_WEIGHTS = {
    "Healthy": 40,
    "Mild": 30,
    "Moderate": 20,
    "Severe": 10,
}


def _assign_profiles(num_patients, seed):
    """
    Assigns clinical severity profiles to the virtual population.

    Instead of relying on purely random sampling, a quota-based allocation
    is used to preserve the intended distribution of patient severity
    levels. This approach prevents small simulation populations from
    containing very few or no severely ill patients, thereby ensuring that
    all clinically relevant profiles remain represented.
    """
    names = list(PROFILE_WEIGHTS.keys())
    weights = [PROFILE_WEIGHTS[n] for n in names]
    total = sum(weights)
    counts = [round(num_patients * w / total) for w in weights]
    diff = num_patients - sum(counts)
    counts[0] += diff
    assignment = []
    for name, count in zip(names, counts):
        assignment.extend([name] * count)
    rng = random.Random(seed)
    rng.shuffle(assignment)
    return assignment


class PatientDemographics:
    """
    Stores the demographic characteristics of a virtual patient.

    These variables describe the simulated patient and are included because
    they represent standard demographic information collected in clinical
    practice and population pharmacokinetic datasets.

    For the norepinephrine pharmacokinetic model adopted in this project,
    published evidence indicates that body weight, height, and sex do not
    significantly influence clearance or volume of distribution in the adult
    ICU population considered.

    Age is the only demographic characteristic incorporated into the
    clearance calculation, based on the published age-dependent reduction in
    norepinephrine clearance described in the supporting literature (see
    constants.py).
    """

    def __init__(self, rng):
        self.age = rng.randint(22, 85)  # whole years, as age is normally recorded
        self.sex = rng.choice(["M", "F"])

        # Adult weight and height are sampled from physiologically plausible
        # ranges to generate realistic virtual ICU patients.
        self.weight_kg = rng.uniform(45, 120)
        self.height_cm = rng.uniform(150, 195)
        self.bmi = self.weight_kg / ((self.height_cm / 100) ** 2)


def generate_patient_params(profile, rng):
    """
    Generates the individualized pharmacokinetic and cardiovascular
    parameters for a single virtual patient.

    The generated parameters define the patient's baseline physiological
    state before drug administration and are subsequently used throughout
    the digital twin simulation.
    """

    demo = PatientDemographics(rng)

    # -------------------------------------------------------------------------
    # Pharmacokinetic parameter generation
    #
    # Drug clearance is determined primarily by illness severity and is then
    # adjusted for the age-related reduction in norepinephrine clearance.
    # -------------------------------------------------------------------------

    saps = c.SAPS_II_BY_PROFILE[profile]

    # Clearance is obtained by linearly interpolating between the published
    # anchor values relating SAPS II score to norepinephrine clearance.
    # This allows patients with intermediate illness severity to receive
    # physiologically consistent clearance values.
    frac = (saps - c.SAPS_LOW_ANCHOR) / (
        c.SAPS_HIGH_ANCHOR - c.SAPS_LOW_ANCHOR
    )
    cl_severity = (
        c.CL_AT_SAPS_20_L_MIN
        + frac * (c.CL_AT_SAPS_60_L_MIN - c.CL_AT_SAPS_20_L_MIN)
    )

    # Clearance is further adjusted to account for the reduction in
    # norepinephrine clearance observed with increasing age.
    # The reduction is applied only within the published age range and is
    # prevented from decreasing indefinitely by enforcing a lower limit.
    if demo.age <= c.AGE_REFERENCE_YOUNG:
        age_factor = 1.0
    else:
        span = c.AGE_REFERENCE_OLD - c.AGE_REFERENCE_YOUNG
        reduction_at_67 = c.CLEARANCE_REDUCTION_BY_OLD_AGE
        age_factor = (
            1.0
            - reduction_at_67
            * min(
                1.0,
                (demo.age - c.AGE_REFERENCE_YOUNG) / span,
            )
        )
        age_factor = max(age_factor, c.AGE_CLEARANCE_FLOOR)

    # A log-normal inter-individual variability term is applied so that
    # patients with identical clinical profiles still exhibit realistic
    # biological variation in drug clearance.
    iiv = np.exp(rng.gauss(0, c.CLEARANCE_INTER_INDIVIDUAL_CV))

    clearance_l_min = cl_severity * age_factor * iiv

    # Clearance cannot be negative or unrealistically small; therefore,
    # a physiological lower bound is enforced.
    clearance_l_min = max(clearance_l_min, 0.1)

    # Volume of distribution is treated as a population constant because
    # the adopted pharmacokinetic model does not identify body weight as a
    # significant predictor for this parameter.
    vd_l = c.VD_NOREPINEPHRINE_L

    # -------------------------------------------------------------------------
    # Baseline cardiovascular state
    #
    # The following parameters define the patient's cardiovascular condition
    # before norepinephrine infusion begins.
    # -------------------------------------------------------------------------

    svr_lo, svr_hi = c.SVR_BY_PROFILE_DYN_S_CM5[profile]
    svr0 = rng.uniform(svr_lo, svr_hi)

    # Initially sampled only for comparison. Stroke volume is later derived
    # from cardiac output and heart rate to maintain physiological
    # consistency between cardiovascular variables.
    sv0_ml_ignored = rng.uniform(*c.SV_NORMAL_ML)

    hr_set_bpm = rng.uniform(*c.HR_NORMAL_BPM)

    # Baseline mean arterial pressure (MAP) is sampled directly from the
    # physiological range corresponding to the patient's clinical profile.
    #
    # Cardiac output is then calculated from the rearranged cardiovascular
    # relationship:
    #
    #     MAP = CVP + (SVR × CO) / 80
    #
    # Deriving cardiac output in this manner guarantees consistency between
    # MAP, systemic vascular resistance, and cardiac output, preventing
    # unrealistic baseline haemodynamics.
    map0_lo, map0_hi = c.MAP0_BY_PROFILE_MMHG[profile]
    map0_target = rng.uniform(map0_lo, map0_hi)

    co0_l_min = (map0_target - c.CVP_TYPICAL_MMHG) * 80.0 / svr0
    co0_l_min = float(np.clip(co0_l_min, 1.5, 12.0))

    # Stroke volume is calculated from cardiac output and heart rate:
    #
    #     CO = HR × SV / 1000
    #
    # Deriving stroke volume instead of sampling it independently ensures
    # that all baseline cardiovascular variables remain physiologically
    # consistent.
    sv0_ml = co0_l_min * 1000.0 / hr_set_bpm

    # Relative circulating blood volume.
    #
    # Healthy patients begin with normal blood volume, whereas increasing
    # illness severity is associated with a modest reduction. The reduction
    # is intentionally smaller than the reduction in systemic vascular
    # resistance because vasodilatory (septic) shock is characterized
    # primarily by vascular dilation rather than absolute blood loss.
    bv0_fraction = {
        "Healthy": 1.0,
        "Mild": 0.95,
        "Moderate": 0.85,
        "Severe": 0.75,
    }[profile]

    # Each patient is assigned an individual baroreflex MAP setpoint,
    # representing the blood pressure that their cardiovascular control
    # system attempts to maintain under normal physiological conditions.
    #
    # During shock, the difference between this defended setpoint and the
    # patient's reduced MAP provides the stimulus for compensatory
    # tachycardia, thereby reproducing the normal physiological baroreflex
    # response.
    map_set_mmhg = rng.uniform(*c.MAP_NORMAL_MMHG)

    return {
        "demographics": demo,
        "profile": profile,
        "saps_ii": saps,
        "clearance_l_min": clearance_l_min,
        "vd_l": vd_l,
        "svr0_dyn_s_cm5": svr0,
        "sv0_ml": sv0_ml,
        "hr_set_bpm": hr_set_bpm,
        "bv0_fraction": bv0_fraction,
        "map_set_mmhg": map_set_mmhg,
    }