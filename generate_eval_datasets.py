"""
===============================================================================
GENERATE_EVAL_DATASETS.PY
Smart AI Infusion Pump - progress seminar evaluation data

Purpose
-------
Generates TWO new datasets for the progress seminar. Neither one replaces
or modifies your existing files, and NEITHER REQUIRES RETRAINING.

  SET A - data/test_patients_large.csv
          300 patients x 600 ticks, seed 2026
          Same episode length as training, so it is a like-for-like
          comparison. Replaces the 60-patient test set for reporting
          accuracy. Fixes the small-sample problem: ~30 Severe patients
          instead of 6.

  SET B - data/test_patients_long.csv
          50 patients x 3600 ticks, seed 3141
          One simulated hour per patient, long enough for the illness-
          recovery mechanism to run its full course. Used for PLOTTING
          the weaning arc, not for reporting accuracy - the models were
          trained on 600-tick episodes and have never seen recovery-phase
          states.

Your existing train_patients.csv and test_patients.csv are NOT touched,
so every number currently in your report stays reproducible.

-------------------------------------------------------------------------------
RECOMMENDED (but optional) ONE-TIME EDIT TO generate_dataset.py
-------------------------------------------------------------------------------
Set B is meant to show the Recovery Index in action, but the dataset does
not currently record it. SVR is logged; the baseline vascular tone that
drives recovery is not.

In generate_dataset.py, inside patient_rows.append({...}), find this line:

        "SVR": round(patient.svr, 1),

and add these two lines directly beneath it:

        "SVR0": round(getattr(patient, "svr0", float("nan")), 1),
        "TimeStableAtTarget": getattr(patient, "time_stable_at_target", float("nan")),

Written with getattr() defaults so that if either attribute is named
differently on your VirtualPatient, the run still completes and the column
is simply NaN - it cannot crash the generation. Check the printed
diagnostics at the end of this script: it reports whether SVR0 arrived
with real values or as NaN.

If you skip this edit, everything still runs. You just will not be able to
plot the Recovery Index itself, only its downstream effects (flow taper,
falling HR).

-------------------------------------------------------------------------------
HOW TO RUN
-------------------------------------------------------------------------------
Place in your project root (the folder with generate_dataset.py) and run:

    C:\\Python314\\python.exe generate_eval_datasets.py

Options:
    --skip-long        generate Set A only
    --skip-large       generate Set B only
    --stride N         write every Nth tick of Set B (physics still runs
                       every tick; this only thins what is SAVED).
                       Use --stride 5 if the CSV is unwieldy.
    --yes              skip the confirmation prompt
===============================================================================
"""

import argparse
import os
import sys
import time

import pandas as pd

try:
    from generate_dataset import generate_dataset, WINDOW, LOOKAHEAD
except ImportError as e:
    sys.exit(
        f"Could not import generate_dataset.py ({e}).\n"
        "Run this script from your project root - the folder that contains "
        "generate_dataset.py, digital_twin.py and expert_controller.py."
    )

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
LARGE = dict(
    name="SET A - large evaluation set",
    num_patients=300,
    sim_time=600,
    seed=2026,                      # not 1 (train) and not 999 (old test)
    output_file=os.path.join("data", "test_patients_large.csv"),
)

LONG = dict(
    name="SET B - long-episode weaning set",
    num_patients=50,
    sim_time=3600,                  # one simulated hour
    seed=3141,
    output_file=os.path.join("data", "test_patients_long.csv"),
)

CALIBRATION_PATIENTS = 5
EXISTING = [
    os.path.join("data", "train_patients.csv"),
    os.path.join("data", "test_patients.csv"),
]


def human_time(seconds):
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    if seconds < 5400:
        return f"{seconds / 60:.1f} minutes"
    return f"{seconds / 3600:.1f} hours"


def calibrate():
    """Time a tiny run so we can estimate the real ones before committing."""
    print("\nCalibrating on a short run...")
    tmp = os.path.join("data", "_calibration_tmp.csv")
    t0 = time.perf_counter()
    generate_dataset(
        num_patients=CALIBRATION_PATIENTS,
        sim_time=600,
        seed=424242,
        output_file=tmp,
    )
    elapsed = time.perf_counter() - t0
    try:
        os.remove(tmp)
    except OSError:
        pass
    per_tick = elapsed / (CALIBRATION_PATIENTS * 600)
    print(f"\nCalibration took {human_time(elapsed)} "
          f"({per_tick * 1000:.3f} ms per patient-tick)")
    return per_tick


def estimate(per_tick, cfg):
    return per_tick * cfg["num_patients"] * cfg["sim_time"]


def run(cfg, stride=1):
    print("\n" + "=" * 70)
    print(cfg["name"])
    print("=" * 70)
    print(f"  patients : {cfg['num_patients']}")
    print(f"  ticks    : {cfg['sim_time']}  ({cfg['sim_time'] / 60:.0f} simulated minutes each)")
    print(f"  seed     : {cfg['seed']}")
    print(f"  output   : {cfg['output_file']}")
    print()

    t0 = time.perf_counter()
    df = generate_dataset(
        num_patients=cfg["num_patients"],
        sim_time=cfg["sim_time"],
        seed=cfg["seed"],
        output_file=cfg["output_file"],
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  Completed in {human_time(elapsed)}")

    # Optional thinning of the SAVED rows. The simulation itself always
    # ran every tick - this only reduces file size for plotting.
    if stride > 1:
        before = len(df)
        df = df[df["Time"] % stride == 0].copy()
        df.to_csv(cfg["output_file"], index=False)
        print(f"  Thinned for storage: {before:,} -> {len(df):,} rows "
              f"(every {stride}th tick; physics ran on all of them)")

    size_mb = os.path.getsize(cfg["output_file"]) / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")
    return df


def diagnostics(df, cfg, is_long):
    print("\n  ---- checks ----")

    patients = df.groupby("PatientID")["Profile"].first().value_counts()
    print("  Patients per profile:")
    for prof, n in patients.items():
        print(f"      {str(prof):<10} {n:>4}")

    sev = int(patients.get("Severe", 0))
    if not is_long:
        verdict = "good" if sev >= 20 else "still low"
        print(f"  Severe patients: {sev}  ({verdict} - the old test set had 6)")

    rate = df["WillDeteriorate"].mean() * 100
    print(f"  Deterioration rate: {rate:.2f}% of rows")

    leak_free = not (df["FlowRate"] == df["RecommendedFlow"]).all()
    print(f"  FlowRate != RecommendedFlow (no target leakage): {leak_free}")
    if not leak_free:
        print("      ** WARNING: these columns are identical. Investigate before reporting. **")

    if "SVR0" in df.columns:
        valid = df["SVR0"].notna().sum()
        if valid == 0:
            print("  SVR0 column present but ALL NaN - the attribute name on "
                  "VirtualPatient is probably not 'svr0'. Check and re-edit.")
        else:
            lo, hi = df["SVR0"].min(), df["SVR0"].max()
            print(f"  SVR0 recorded: {lo:.0f} -> {hi:.0f} dyn.s/cm5 across the set  (good)")
    else:
        print("  SVR0 not recorded - apply the optional edit in this file's "
              "header if you want to plot the Recovery Index itself.")

    if is_long:
        # Did anything actually wean? Compare early vs late flow per patient.
        early = df[df["Time"] < 300].groupby("PatientID")["FlowRate"].mean()
        late = df[df["Time"] > df["Time"].max() - 300].groupby("PatientID")["FlowRate"].mean()
        joined = pd.concat([early.rename("early"), late.rename("late")], axis=1).dropna()
        weaned = (joined["late"] < joined["early"] * 0.25).sum()
        zeroed = (joined["late"] <= 0.01).sum()
        print(f"  Patients whose flow fell below 25% of its early average: "
              f"{weaned}/{len(joined)}")
        print(f"  Patients fully weaned to zero flow: {zeroed}/{len(joined)}")

        hr_early = df[df["Time"] < 300].groupby("PatientID")["HR"].mean().mean()
        hr_late = df[df["Time"] > df["Time"].max() - 300].groupby("PatientID")["HR"].mean().mean()
        print(f"  Mean HR, first 5 min -> last 5 min: {hr_early:.1f} -> {hr_late:.1f} bpm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-large", action="store_true")
    ap.add_argument("--skip-long", action="store_true")
    ap.add_argument("--stride", type=int, default=1,
                    help="save every Nth tick of Set B (default 1 = save all)")
    ap.add_argument("--yes", action="store_true", help="skip confirmation")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)

    print("=" * 70)
    print("SMART INFUSION AI - EVALUATION DATASET GENERATION")
    print("=" * 70)
    print(f"\nRolling window: {WINDOW} ticks   Risk lookahead: {LOOKAHEAD} ticks")

    for path in EXISTING:
        if os.path.exists(path):
            print(f"  Existing file left untouched: {path}")

    per_tick = calibrate()

    jobs = []
    if not args.skip_large:
        jobs.append((LARGE, False))
    if not args.skip_long:
        jobs.append((LONG, True))
    if not jobs:
        sys.exit("Nothing to do - you skipped both sets.")

    print("\nEstimated run times:")
    total = 0.0
    for cfg, _ in jobs:
        est = estimate(per_tick, cfg)
        total += est
        print(f"  {cfg['name']:<36} ~{human_time(est)}")
    print(f"  {'TOTAL':<36} ~{human_time(total)}")

    if not args.yes:
        reply = input("\nProceed? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            sys.exit("Cancelled. Nothing was written.")

    results = []
    for cfg, is_long in jobs:
        df = run(cfg, stride=args.stride if is_long else 1)
        diagnostics(df, cfg, is_long)
        results.append((cfg, df))

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    for cfg, df in results:
        print(f"  {cfg['output_file']}   {len(df):,} rows")
    print("\nNothing was retrained. Your models and packaged app are unchanged.")
    print("\nNext step: re-score the models against the LARGE set by editing")
    print("  TEST_CSV in report_metrics_v2.py to point at test_patients_large.csv,")
    print("  then run it and paste the output into the chat.")


if __name__ == "__main__":
    main()
