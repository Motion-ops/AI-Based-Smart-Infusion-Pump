"""
===============================================================================
NITROPRUSSIDE_PHARMACODYNAMICS.PY
Smart AI Infusion Pump - Hypertensive Emergency Digital Twin

Purpose
-------
Sibling of pharmacodynamics.py. Converts nitroprusside plasma
concentration into a normalized drug effect using the same Emax (Hill)
structure as the norepinephrine side - but the returned effect is
NEGATIVE, representing vasodilation (SVR falling) rather than
vasoconstriction (SVR rising).

cardiovascular.compute_svr() is reused UNCHANGED from the
septic-shock side:

    SVR = SVR0 * (1 + E)

That equation is already drug-agnostic - it was never hardcoded to
assume E >= 0. A negative E (this module) reduces SVR; a positive E
(pharmacodynamics.py) raises it. This is the one piece of architecture
that made "one shared cardiovascular engine, two opposite-direction
drugs" actually work without touching cardiovascular.py at all.
===============================================================================
"""

import hypertensive_constants as hc


def compute_drug_effect(cp):
    """
    Converts plasma concentration (Cp, mcg/L) into a normalized,
    NEGATIVE pharmacodynamic effect via the Emax (Hill) equation, for
    use as the `drug_effect` argument to cardiovascular.compute_svr().

    Returns
    -------
    float
        Dimensionless drug effect in [EMAX_ASSUMED, 0] - i.e. zero or
        negative, never positive. Passing this into
        SVR = SVR0 * (1 + E) can only reduce SVR, matching
        nitroprusside's vasodilatory mechanism.
    """

    if cp <= 0:
        return 0.0

    cp_g = cp ** hc.HILL_COEFFICIENT_ASSUMED
    ec50_g = hc.EC50_ASSUMED_MCG_L ** hc.HILL_COEFFICIENT_ASSUMED

    # Same Hill fraction (0 to 1) as the norepinephrine side, but scaled
    # by a NEGATIVE Emax - this is the only structural difference.
    return hc.EMAX_ASSUMED * cp_g / (ec50_g + cp_g)
