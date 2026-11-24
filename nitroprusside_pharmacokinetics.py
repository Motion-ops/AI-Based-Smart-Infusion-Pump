"""
===============================================================================
NITROPRUSSIDE_PHARMACOKINETICS.PY
Smart AI Infusion Pump - Hypertensive Emergency Digital Twin

Purpose
-------
Sibling of pharmacokinetics.py for sodium nitroprusside (SNP). Same
one-compartment IV infusion model and Euler integration approach as the
norepinephrine module - the underlying mathematics is drug-agnostic,
only the numeric parameters (concentration, Vd, clearance) differ. Kept
as a separate module rather than parameterizing pharmacokinetics.py
directly so the two drugs' constants can never be accidentally
cross-wired.

Model
-----
    dCp/dt = R/Vd - ke*Cp,   ke = CL/Vd

Same equation as pharmacokinetics.py. What's actually different for
nitroprusside is the SCALE of ke: its ~2-minute circulatory half-life
(see hypertensive_constants.NITROPRUSSIDE_HALFLIFE_MIN) gives it a
somewhat larger ke than norepinephrine's.

[CORRECTED 2026-10-04] This comment previously claimed ke was "roughly
two orders of magnitude larger" than norepinephrine's - that was never
actually checked against this sim's own numbers and doesn't hold up:
directly simulating both twins and comparing ke=clearance_l_min/vd_l
for representative patients gives ke_NE~0.14-0.21/min and
ke_SNP~0.25-0.47/min - a ~1.7-2.7x difference, not 100x. (NE's clearance
here is SAPS-II/age-derived, not simply "2.5 min half-life" as some
literature states for healthy volunteers - that's most of why the ratio
is smaller than the half-life numbers alone would suggest.) Measuring
time-to-90%-of-steady-state-effect directly from a cold start confirms
SNP is still genuinely faster-onset (~5 min vs ~11-13 min for NE across
3 test patients each) - the qualitative "near-instant vs. slower" claim
holds, just not by two orders of magnitude. No special "fast drug"
mechanism is needed either way - the standard one-compartment equation
produces this difference from Vd/clearance alone.
===============================================================================
"""

import hypertensive_constants as hc


def flow_rate_to_mass_rate(flow_ml_hr):
    """
    Converts pump flow rate (mL/hr) to nitroprusside mass delivery rate
    (mcg/min), using the SNP-specific premix concentration - NOT
    constants.DRUG_CONCENTRATION_MCG_ML, which is norepinephrine's.
    """
    return flow_ml_hr * hc.DRUG_CONCENTRATION_MCG_ML / 60.0


def mass_rate_to_flow_ml_hr(mass_rate_mcg_min):
    """Inverse of flow_rate_to_mass_rate() - see pharmacokinetics.py for
    the identical norepinephrine-side rationale."""
    return mass_rate_mcg_min * 60.0 / hc.DRUG_CONCENTRATION_MCG_ML


def update_plasma_concentration(cp_current, flow_ml_hr,
                                vd_l, clearance_l_min, dt_min):
    """
    Identical Euler-integrated one-compartment update to
    pharmacokinetics.update_plasma_concentration(), calling THIS
    module's flow_rate_to_mass_rate() so the SNP concentration constant
    is used instead of norepinephrine's. See that module's docstring
    for the full derivation - not repeated here to avoid drift between
    two copies of the same explanation.
    """

    ke = clearance_l_min / vd_l
    r_mcg_min = flow_rate_to_mass_rate(flow_ml_hr)
    dcp_dt = (r_mcg_min / vd_l) - (ke * cp_current)
    cp_new = cp_current + dcp_dt * dt_min

    return max(cp_new, 0.0)
