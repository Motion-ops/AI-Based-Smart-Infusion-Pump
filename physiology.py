"""
===============================================================================
PHYSIOLOGY.PY
Smart AI Infusion Pump - Mechanistic Digital Twin

Purpose
-------
This module simulates the physiological response of a virtual patient
that is not directly controlled by the pharmacokinetic or
cardiovascular models.

Two physiological processes are represented:

1. Baroreflex regulation of heart rate.
2. Peripheral oxygen saturation (SpO₂) dynamics.

These physiological responses allow the digital twin to reproduce the
body's natural compensatory mechanisms during hypotension and recovery
following norepinephrine therapy.

Baroreflex model
----------------
Heart rate is governed using a first-order baroreflex model:

    dHR/dt = k(HRset − HR) − G(MAP − MAPset)

where:

    HR       = Current heart rate
    HRset    = Patient-specific resting heart rate
    MAP      = Current mean arterial pressure
    MAPset   = Individual baroreflex pressure setpoint
    k        = Heart-rate recovery constant
    G        = Baroreflex gain

This mathematical structure is widely used in cardiovascular control
modelling to represent autonomic regulation of heart rate.

The gain constants used in this implementation are stated assumptions
because directly applicable literature values for this simplified model
were not available. Their derivation and justification are documented
in constants.py.

SpO₂ model
----------
Unlike heart rate, oxygen saturation is not the primary controlled
variable in this project.

Rather than implementing a detailed oxygen transport model involving
oxygen delivery, oxygen consumption, pulmonary gas exchange, and tissue
perfusion, a simplified physiological approximation is adopted.

Reduced mean arterial pressure leads to impaired tissue perfusion,
causing SpO₂ to decline gradually, while restoration of adequate blood
pressure allows oxygen saturation to recover progressively toward
normal values.

The rates of decline and recovery are therefore heuristic parameters
chosen to reproduce realistic physiological behaviour rather than
representing experimentally measured oxygen transport kinetics.

All assumptions are documented in constants.py.
===============================================================================
"""

import random
import numpy as np

import constants as c


def update_heart_rate(hr_current, hr_set, map_current, map_set, dt_min,
                       max_deviation_mmhg=None):
    """
    Updates the patient's heart rate using a first-order baroreflex
    model.

    Parameters
    ----------
    hr_current : float
        Current heart rate (beats/min).

    hr_set : float
        Patient-specific resting heart rate.

    map_current : float
        Current mean arterial pressure (mmHg).

    map_set : float
        Patient-specific baroreflex MAP setpoint (mmHg).

    dt_min : float
        Simulation time step expressed in minutes.

    Returns
    -------
    float
        Updated heart rate (beats/min).

    Explanation
    -----------
    The baroreflex continuously attempts to restore cardiovascular
    homeostasis.

    When arterial pressure falls below the patient's defended MAP
    setpoint, the autonomic nervous system increases heart rate to help
    maintain cardiac output.

    Conversely, when arterial pressure rises above the defended
    setpoint, heart rate gradually decreases toward its resting value.

    Euler's numerical integration method is used to advance the
    baroreflex equation over each simulation time step.
    """

    # Compute the rate of change of heart rate according to the
    # first-order baroreflex model. The pressure-deviation term is
    # optionally capped (saturates) rather than scaled linearly without
    # limit - see constants.py for why: an occasional patient combining
    # a high hr_set with a high map_set otherwise produces unrealistic
    # sustained tachycardia (>145 bpm) purely from those two
    # independent, each individually-normal, samples compounding
    # together. The cap is opt-in (max_deviation_mmhg=None by default)
    # because it must NOT blunt the compensatory response during a
    # genuine acute hypotensive emergency, where a strong, fast HR rise
    # is exactly what helps restore cardiac output - digital_twin.py
    # only supplies a cap once the patient is already at/near target,
    # targeting the specific long-tail "still elevated forever" case
    # without slowing real emergency recovery.
    map_deviation = map_current - map_set
    if max_deviation_mmhg is not None:
        map_deviation = float(np.clip(map_deviation, -max_deviation_mmhg, max_deviation_mmhg))
    dhr_dt = (
        c.BARORELFEX_K * (hr_set - hr_current)
        - c.BAROREFLEX_G * map_deviation
    )

    # Advance the solution forward by one simulation step.
    hr_new = hr_current + dhr_dt * dt_min

    # Heart rate is constrained within physiologically plausible limits.
    return float(np.clip(hr_new, 30, 220))


def update_spo2(spo2_current, map_current, dt_min):
    """
    Updates peripheral oxygen saturation (SpO₂).

    Parameters
    ----------
    spo2_current : float
        Current oxygen saturation (%).

    map_current : float
        Current mean arterial pressure (mmHg).

    dt_min : float
        Simulation time step expressed in minutes.

    Returns
    -------
    float
        Updated oxygen saturation (%).

    Explanation
    -----------
    This model provides a simplified approximation of oxygen saturation
    behaviour rather than a detailed physiological model.

    When mean arterial pressure falls below approximately 65 mmHg,
    tissue perfusion becomes inadequate, causing oxygen saturation to
    decline gradually.

    Once adequate arterial pressure is restored, oxygen saturation
    slowly recovers toward normal physiological values.

    The recovery and deterioration rates are heuristic parameters
    selected to produce realistic simulation behaviour and should not
    be interpreted as experimentally measured physiological constants.
    """

    # [FIX, 2026-09-28] This used to be a 3-way split: decline below
    # 65-deadband, recover above 65+deadband, and hold perfectly STEADY
    # in between (62-68 mmHg) to avoid noise-driven flip-flopping right
    # at the threshold. That "steady" band was the entire septic-shock
    # clinical target zone (65 +/- TARGET_DEADBAND=3, the exact range
    # expert_controller.py holds MAP in once treatment succeeds) - so a
    # SUCCESSFULLY treated patient, held right at target as intended,
    # would get its SpO2 permanently frozen wherever it happened to be
    # (e.g. stuck at 89.4% indefinitely) instead of recovering, only
    # ever resuming recovery if MAP later drifted above 68 via the
    # separate, much slower illness-recovery mechanism - confirmed
    # directly: a test patient held at MAP=65.0-67.9 for 900+ simulated
    # seconds with SpO2 frozen the entire time. Collapsed to a 2-way
    # split - reaching or exceeding the perfusion threshold (with a
    # deadband kept only on the lower/decline side, to avoid flapping
    # right at that one boundary) now correctly starts recovery
    # immediately, matching the docstring's own stated intent ("once
    # adequate arterial pressure is restored, oxygen saturation slowly
    # recovers").
    if map_current < 65 - c.SPO2_PERFUSION_DEADBAND_MMHG:
        # Reduced arterial pressure results in inadequate tissue
        # perfusion, producing a gradual decline in oxygen saturation.
        spo2_new = (
            spo2_current
            - random.uniform(0.03, 0.08) * dt_min * 60
        )
    else:
        # Adequate perfusion allows oxygen saturation to progressively
        # recover toward normal physiological values.
        spo2_new = (
            spo2_current
            + random.uniform(0.01, 0.05) * dt_min * 60
        )

    # Oxygen saturation is constrained within physiologically realistic
    # limits.
    return float(np.clip(spo2_new, 70, 100))