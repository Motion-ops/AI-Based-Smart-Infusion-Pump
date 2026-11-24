"""
===========================================================
CADENCE_OVERSHOOT_EXPERIMENT.PY
Smart AI Infusion Pump - Standalone Report Figure

Purpose
-------
This is a ONE-OFF illustrative experiment, deliberately kept separate
from generate_dataset.py / train_ai.py / train_risk_model.py. It does
not change the main simulation pipeline, does not touch any trained
model, and its output is not consumed anywhere else in the project -
it exists solely to produce a supporting figure/table for the project
report's discussion of titration cadence.

What it demonstrates
---------------------
The trained system (dashboard.py, native_app.py, generate_dataset.py)
re-evaluates and re-doses every simulated SECOND - a documented "time
compression" simplification relative to real ICU practice, where a
clinician typically only adjusts the norepinephrine flow rate every
2-5 minutes (see expert_controller.py's MAX_STEP_MCG_KG_MIN comment).

This script asks: what would happen if the SAME per-decision dose step
(taken from real per-2-5-minute clinical titration literature) were
applied at the REALISTIC decision frequency instead of every second?

It runs the identical patient physiology and the identical
expert_controller() logic - nothing about the controller itself is
modified - varying only how often a new dosing decision is computed.
Between decisions, the pump holds its last commanded flow rate
constant (a zero-order hold), exactly like a real infusion pump does
between clinician adjustments.

Expected finding
-----------------
Deciding infrequently with a fixed-size step causes materially larger
overshoot than deciding every second, because more drug has already
been committed to the bloodstream (and is still being absorbed/
distributed) by the time the next decision arrives - this is the same
"jump the rate, panic two minutes later" dynamic described in the
clinical literature on why continuous small titration is preferred
over infrequent large adjustments.

Caveat (documented, not hidden)
--------------------------------
expert_controller()'s derivative term reads patient.smoothed_bp /
patient.previous_smoothed_bp, which update every simulated SECOND
regardless of decision cadence. So even at a 2-5 minute decision
interval, the anticipatory (KD) term here still "sees" a 1-second-old
trend - more reactive than a real clinician re-checking a monitor only
every few minutes would be. If anything, this makes the infrequent-
decision arms of this experiment MORE conservative (less prone to
overshoot) than a truly faithful once-per-N-minutes clinician model
would be - so the overshoot gap this script measures is a
lower-bound estimate of the real effect, not an inflated one.

Run with:
    python cadence_overshoot_experiment.py
Output:
    figures/cadence_overshoot_<profile>_seed<seed>.png
===========================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from digital_twin import VirtualPatient
from expert_controller import expert_controller, TARGET_DEADBAND

TARGET_MAP = 65.0
TOTAL_TICKS = 1800  # 30 simulated minutes
CADENCES_SECONDS = {
    "Every 1s (current model)": 1,
    "Every 2 min (clinical)": 120,
    "Every 5 min (clinical)": 300,
}

# (profile, seed) pairs chosen from prior diagnostic runs as patients
# that genuinely need - and successfully respond to - norepinephrine
# under the current per-second controller (excludes the two unrelated
# "floor"/"ceiling" cases documented separately: a Healthy/Mild patient
# whose baroreflex keeps MAP above target with zero drug, and a
# catecholamine-refractory Severe patient capped at max flow).
TEST_PATIENTS = [
    ("Moderate", 1),
    ("Severe", 100),
]


def simulate(profile, seed, cadence_ticks):
    """Advance physiology every tick; only recompute the controller's
    decision every `cadence_ticks` ticks. Between decisions the pump
    holds its last commanded flow rate constant (zero-order hold),
    exactly as a real infusion pump behaves between adjustments."""
    patient = VirtualPatient(0, profile=profile, seed=seed, target_map=TARGET_MAP)
    map_trace = np.empty(TOTAL_TICKS)
    flow_trace = np.empty(TOTAL_TICKS)

    for t in range(TOTAL_TICKS):
        patient.update()
        if t % cadence_ticks == 0:
            decision = expert_controller(patient)
            patient.flow_rate = decision["RecommendedFlow"]
        map_trace[t] = patient.map
        flow_trace[t] = patient.flow_rate

    return map_trace, flow_trace


def main():
    os.makedirs("figures", exist_ok=True)
    summary_rows = []

    for profile, seed in TEST_PATIENTS:
        fig, (ax_map, ax_flow) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        t_axis = np.arange(TOTAL_TICKS)

        for label, cadence in CADENCES_SECONDS.items():
            map_trace, flow_trace = simulate(profile, seed, cadence)
            peak = map_trace.max()
            overshoot = peak - TARGET_MAP
            settle_final = map_trace[-1]
            summary_rows.append((profile, seed, label, peak, overshoot, settle_final))

            ax_map.plot(t_axis, map_trace, label=f"{label} (overshoot {overshoot:+.1f} mmHg)")
            ax_flow.step(t_axis, flow_trace, where="post", label=label)

        ax_map.axhline(TARGET_MAP, color="black", linestyle=":", linewidth=1, label=f"Target ({TARGET_MAP:.0f})")
        ax_map.axhspan(TARGET_MAP - TARGET_DEADBAND, TARGET_MAP + TARGET_DEADBAND, color="green", alpha=0.1)
        ax_map.set_ylabel("MAP (mmHg)")
        ax_map.set_title(f"Effect of decision cadence on overshoot — {profile} patient (seed={seed})")
        ax_map.legend(loc="lower right", fontsize=8)

        ax_flow.set_ylabel("Flow rate (mL/hr)")
        ax_flow.set_xlabel("Time (s)")
        ax_flow.legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        out_path = f"figures/cadence_overshoot_{profile}_seed{seed}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

    print("\n=== Summary: peak MAP and overshoot by decision cadence ===")
    print(f"{'Profile':10s} {'Seed':5s} {'Cadence':26s} {'Peak MAP':>9s} {'Overshoot':>10s} {'Final MAP':>10s}")
    for profile, seed, label, peak, overshoot, final in summary_rows:
        print(f"{profile:10s} {seed:<5d} {label:26s} {peak:9.1f} {overshoot:+10.1f} {final:10.1f}")


if __name__ == "__main__":
    main()
