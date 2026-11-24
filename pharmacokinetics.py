"""
===============================================================================
PHARMACOKINETICS.PY
Smart AI Infusion Pump - Mechanistic Digital Twin

Purpose
-------
This module simulates the pharmacokinetics (PK) of intravenously infused
norepinephrine. Pharmacokinetics describes how the drug moves through the
body after administration, specifically how its plasma concentration changes
over time due to continuous infusion and elimination.

The plasma concentration predicted by this module serves as the input to the
pharmacodynamic model (pharmacodynamics.py), which converts drug
concentration into the corresponding physiological effect.

Model implemented
-----------------
A standard one-compartment intravenous infusion model is employed:

        dCp/dt = R/Vd − ke·Cp

where:

    Cp  = Plasma concentration (mcg/L)
    R   = Drug infusion rate (mcg/min)
    Vd  = Apparent volume of distribution (L)
    ke  = Elimination rate constant (1/min)

The elimination rate constant is computed as:

        ke = CL / Vd

where:

    CL = Drug clearance (L/min)

This model assumes that immediately after entering the bloodstream,
norepinephrine distributes uniformly throughout a single well-mixed
compartment while being continuously eliminated at a rate proportional
to its current concentration.

Model justification
-------------------
The one-compartment intravenous infusion model is a well-established
pharmacokinetic model and is widely used for drugs administered by
continuous infusion. It provides an appropriate balance between
physiological realism and computational simplicity, making it suitable
for real-time digital twin simulations.

The mathematical formulation follows the standard pharmacokinetic model
described by Rowland & Tozer, while the parameterization adopted in this
project is based on published norepinephrine pharmacokinetic literature
(see constants.py).

Unit consistency
----------------
To prevent unit inconsistencies throughout the simulation, all variables
used in this module follow a consistent unit convention:

    Flow rate (pump)          : mL/hr
    Drug input rate (R)       : mcg/min
    Volume of distribution    : L
    Clearance                 : L/min
    Plasma concentration (Cp) : mcg/L

Since infusion pumps deliver fluid volumetrically (mL/hr), whereas the
pharmacokinetic equations require drug mass per unit time (mcg/min),
conversion functions are provided to translate between these two
representations using the known premixed drug concentration.
===============================================================================
"""

import constants as c


def flow_rate_to_mass_rate(flow_ml_hr):
    """
    Converts the infusion pump flow rate into the corresponding drug
    mass delivery rate.

    Parameters
    ----------
    flow_ml_hr : float
        Pump flow rate in mL/hr.

    Returns
    -------
    float
        Drug input rate in mcg/min.

    Explanation
    -----------
    The infusion pump controls the delivery of fluid in millilitres per
    hour, whereas the pharmacokinetic model requires the amount of drug
    entering the bloodstream per minute.

    This conversion is performed using the known premixed drug
    concentration defined in constants.py.

    Conversion:

        Drug input rate = Flow rate × Drug concentration ÷ 60
    """
    return flow_ml_hr * c.DRUG_CONCENTRATION_MCG_ML / 60.0


def mass_rate_to_flow_ml_hr(mass_rate_mcg_min):
    """
    Converts a desired drug mass delivery rate back into the equivalent
    infusion pump flow rate.

    Parameters
    ----------
    mass_rate_mcg_min : float
        Desired drug delivery rate in mcg/min.

    Returns
    -------
    float
        Equivalent pump flow rate in mL/hr.

    Explanation
    -----------
    Clinically, norepinephrine doses are prescribed and titrated in
    terms of drug mass per unit time (or mcg/kg/min).

    However, an infusion pump cannot directly administer drug mass.
    Instead, it controls the volume of fluid delivered per hour.

    This inverse conversion allows the controller to calculate dosing
    decisions using pharmacological units and then convert those
    decisions into the flow rate required by the pump hardware.
    """
    return mass_rate_mcg_min * 60.0 / c.DRUG_CONCENTRATION_MCG_ML


def update_plasma_concentration(cp_current, flow_ml_hr,
                                vd_l, clearance_l_min, dt_min):
    """
    Updates plasma concentration over one simulation time step using
    Euler's numerical integration method.

    Parameters
    ----------
    cp_current : float
        Current plasma concentration (mcg/L).

    flow_ml_hr : float
        Current infusion pump flow rate (mL/hr).

    vd_l : float
        Apparent volume of distribution (L).

    clearance_l_min : float
        Drug clearance (L/min).

    dt_min : float
        Simulation time step expressed in minutes.

    Returns
    -------
    float
        Updated plasma concentration (mcg/L).

    Explanation
    -----------
    The rate of change of plasma concentration consists of two opposing
    physiological processes:

    • Drug entering the bloodstream through continuous infusion.

    • Drug being removed from circulation through metabolic and renal
      clearance.

    Euler's method is used to numerically integrate the differential
    equation over each simulation time step:

        Cp(new) = Cp(current) + (dCp/dt × dt)

    Although more sophisticated numerical solvers exist, Euler
    integration is computationally efficient and sufficiently accurate
    for the small simulation time steps used throughout this project.

    Note
    ----
    The overall digital twin operates using one-second simulation
    intervals. Since this pharmacokinetic model uses minutes as its
    base time unit, the calling function converts:

        dt_min = dt_seconds / 60

    before invoking this function, ensuring complete unit consistency.
    """

    # Compute the elimination rate constant from the relationship:
    # ke = Clearance / Volume of Distribution
    ke = clearance_l_min / vd_l

    # Convert the pump flow rate into the drug mass entering the body
    # each minute.
    r_mcg_min = flow_rate_to_mass_rate(flow_ml_hr)

    # Compute the instantaneous rate of change of plasma concentration.
    # The first term represents drug input, while the second term
    # represents first-order elimination.
    dcp_dt = (r_mcg_min / vd_l) - (ke * cp_current)

    # Advance the solution forward by one numerical integration step.
    cp_new = cp_current + dcp_dt * dt_min

    # Plasma concentration cannot be negative in reality, so a lower
    # physical bound of zero is enforced.
    return max(cp_new, 0.0)