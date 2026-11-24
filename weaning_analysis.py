"""
===============================================================================
WEANING_ANALYSIS.PY
Smart AI Infusion Pump - corrected weaning / recovery analysis

Why this exists
---------------
The earlier weaning check averaged heart rate across ALL 50 patients in the
long-episode set. Only 34 of them ever received drug. The 16 who never
needed any were mostly Healthy patients whose HR RISES toward their own
baroreflex setpoint - which cancelled out the genuine HR decline in the
patients who actually weaned, producing a misleading flat result
(86.4 -> 85.6 bpm).

This script fixes that by:

  1. Restricting every recovery claim to patients who ACTUALLY RECEIVED
     DRUG (mean flow above a threshold during the first 5 minutes).
  2. Splitting all results BY CLINICAL PROFILE, so Healthy patients who
     never needed treatment cannot dilute the treated ones.
  3. Reporting per-patient HR change rather than a population mean, so a
     few outliers cannot dominate.
  4. Reporting time-to-wean: how long until flow first reached zero.

Read-only. Nothing is retrained or modified.

HOW TO RUN
----------
From your project root:

    C:\\Python314\\python.exe weaning_analysis.py

Optional:
    --csv PATH      analyse a different file
    --window N      seconds used for the early/late comparison (default 300)
===============================================================================
"""

import argparse
import json
import os
import sys

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    sys.exit(f"Missing package: {e}. Install with: pip install pandas numpy")

DEFAULT_CSV = os.path.join("data", "test_patients_long.csv")
DOSED_THRESHOLD = 0.1      # mL/hr - above this early average counts as "treated"
PROFILES = ["Healthy", "Mild", "Moderate", "Severe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--window", type=int, default=300)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"Could not find {args.csv}. Run from your project root.")

    df = pd.read_csv(args.csv)
    tmax = int(df["Time"].max())
    w = args.window

    print("=" * 72)
    print("WEANING ANALYSIS - dosed patients only, split by profile")
    print("=" * 72)
    print(f"\n  File     : {args.csv}")
    print(f"  Rows     : {len(df):,}")
    print(f"  Patients : {df['PatientID'].nunique()}")
    print(f"  Duration : {tmax + 1} ticks ({(tmax + 1) / 60:.0f} simulated minutes)")
    print(f"  Comparison window: first {w}s vs last {w}s")

    early_mask = df["Time"] < w
    late_mask = df["Time"] > tmax - w

    # ---- per-patient summary table ----
    per = pd.DataFrame({
        "Profile": df.groupby("PatientID")["Profile"].first(),
        "flow_early": df[early_mask].groupby("PatientID")["FlowRate"].mean(),
        "flow_late": df[late_mask].groupby("PatientID")["FlowRate"].mean(),
        "flow_peak": df.groupby("PatientID")["FlowRate"].max(),
        "hr_early": df[early_mask].groupby("PatientID")["HR"].mean(),
        "hr_late": df[late_mask].groupby("PatientID")["HR"].mean(),
        "map_late": df[late_mask].groupby("PatientID")["MAP"].mean(),
        "target": df.groupby("PatientID")["TargetBP"].first(),
    }).dropna(subset=["flow_early", "flow_late"])

    per["hr_change"] = per["hr_late"] - per["hr_early"]
    per["dosed"] = per["flow_early"] > DOSED_THRESHOLD

    # time at which flow first hit zero and stayed there
    def first_zero(g):
        z = g.loc[g["FlowRate"] <= 0.01, "Time"]
        return int(z.iloc[0]) if len(z) else np.nan

    tz = df.groupby("PatientID")[["Time", "FlowRate"]].apply(first_zero)
    per["first_zero_s"] = tz

    dosed = per[per["dosed"]]
    untreated = per[~per["dosed"]]

    print(f"\n  Received drug early : {len(dosed)} patients")
    print(f"  Never dosed         : {len(untreated)} patients "
          f"(excluded from all recovery claims below)")

    if len(untreated):
        print("\n  Why the earlier HR number was flat - the untreated group:")
        print(f"    mean HR change: {untreated['hr_change'].mean():+.1f} bpm "
              f"(these RISE toward their own setpoint)")
        counts = untreated["Profile"].value_counts()
        print("    profiles: " + ", ".join(f"{p} x{int(n)}" for p, n in counts.items()))

    results = {"file": args.csv, "patients": int(df["PatientID"].nunique()),
               "ticks": int(tmax + 1), "dosed": int(len(dosed)),
               "never_dosed": int(len(untreated))}

    # ---- headline, dosed patients only ----
    print("\n" + "-" * 72)
    print("DOSED PATIENTS - headline numbers")
    print("-" * 72)

    weaned = int((dosed["flow_late"] < dosed["flow_early"] * 0.25).sum())
    zeroed = int((dosed["flow_late"] <= 0.01).sum())
    hr_mean = float(dosed["hr_change"].mean())
    hr_med = float(dosed["hr_change"].median())
    hr_fell = int((dosed["hr_change"] < 0).sum())

    print(f"\n  Flow below 25% of early rate : {weaned}/{len(dosed)} "
          f"({weaned / max(len(dosed),1) * 100:.0f}%)")
    print(f"  Fully weaned to zero         : {zeroed}/{len(dosed)} "
          f"({zeroed / max(len(dosed),1) * 100:.0f}%)")
    print(f"\n  HR change (mean)             : {hr_mean:+.1f} bpm")
    print(f"  HR change (median)           : {hr_med:+.1f} bpm")
    print(f"  Patients whose HR fell       : {hr_fell}/{len(dosed)}")

    tw = dosed["first_zero_s"].dropna()
    if len(tw):
        print(f"\n  Time to first zero flow      : "
              f"median {tw.median() / 60:.1f} min, "
              f"range {tw.min() / 60:.1f}-{tw.max() / 60:.1f} min")
        results["time_to_zero_median_min"] = round(float(tw.median()) / 60, 1)

    results.update({
        "weaned_below_25pct": weaned,
        "fully_weaned": zeroed,
        "hr_change_mean": round(hr_mean, 1),
        "hr_change_median": round(hr_med, 1),
        "hr_fell_count": hr_fell,
    })

    # ---- by profile ----
    print("\n" + "-" * 72)
    print("BY CLINICAL PROFILE  (dosed patients only)")
    print("-" * 72)
    print(f"\n  {'Profile':<10}{'n':>4}{'flow early':>12}{'flow late':>11}"
          f"{'HR early':>10}{'HR late':>9}{'HR change':>11}{'weaned':>9}")
    by_profile = {}
    for prof in PROFILES:
        g = dosed[dosed["Profile"] == prof]
        if g.empty:
            continue
        z = int((g["flow_late"] <= 0.01).sum())
        print(f"  {prof:<10}{len(g):>4}{g['flow_early'].mean():>12.1f}"
              f"{g['flow_late'].mean():>11.1f}{g['hr_early'].mean():>10.1f}"
              f"{g['hr_late'].mean():>9.1f}{g['hr_change'].mean():>+11.1f}"
              f"{z:>6}/{len(g):<3}")
        by_profile[prof] = {
            "n": int(len(g)),
            "flow_early": round(float(g["flow_early"].mean()), 2),
            "flow_late": round(float(g["flow_late"].mean()), 2),
            "hr_early": round(float(g["hr_early"].mean()), 1),
            "hr_late": round(float(g["hr_late"].mean()), 1),
            "hr_change": round(float(g["hr_change"].mean()), 1),
            "fully_weaned": z,
        }
    results["by_profile"] = by_profile

    # ---- the strongest single case, for the slide ----
    print("\n" + "-" * 72)
    print("BEST ILLUSTRATIVE PATIENT  (largest HR decline among dosed)")
    print("-" * 72)
    if len(dosed):
        best = dosed["hr_change"].idxmin()
        b = dosed.loc[best]
        print(f"\n  PatientID {best}  ({b['Profile']})")
        print(f"    flow : {b['flow_early']:.1f} -> {b['flow_late']:.1f} mL/hr "
              f"(peak {b['flow_peak']:.1f})")
        print(f"    HR   : {b['hr_early']:.1f} -> {b['hr_late']:.1f} bpm "
              f"({b['hr_change']:+.1f})")
        print(f"    MAP at end: {b['map_late']:.1f} mmHg (target {b['target']:.0f})")
        results["example_patient"] = {
            "id": int(best), "profile": str(b["Profile"]),
            "flow_early": round(float(b["flow_early"]), 1),
            "flow_late": round(float(b["flow_late"]), 1),
            "hr_early": round(float(b["hr_early"]), 1),
            "hr_late": round(float(b["hr_late"]), 1),
        }
        # time series for this patient, thinned - useful for plotting the slide
        s = df[df["PatientID"] == best][["Time", "FlowRate", "HR", "MAP"]]
        s = s[s["Time"] % 60 == 0]
        results["example_series"] = {
            "time_s": [int(v) for v in s["Time"]],
            "flow": [round(float(v), 2) for v in s["FlowRate"]],
            "hr": [round(float(v), 1) for v in s["HR"]],
            "map": [round(float(v), 1) for v in s["MAP"]],
        }

    # ---- mean trajectory across dosed patients, for the slide chart ----
    dosed_ids = list(dosed.index)
    traj = df[df["PatientID"].isin(dosed_ids)]
    traj = traj[traj["Time"] % 60 == 0].groupby("Time")[["FlowRate", "HR", "MAP"]].mean()
    results["mean_trajectory"] = {
        "time_s": [int(v) for v in traj.index],
        "flow": [round(float(v), 2) for v in traj["FlowRate"]],
        "hr": [round(float(v), 1) for v in traj["HR"]],
        "map": [round(float(v), 1) for v in traj["MAP"]],
    }
    print("\n  Mean trajectory across dosed patients captured at 60s intervals "
          f"({len(traj)} points) - included in the JSON for plotting.")

    print("\n" + "=" * 72)
    print("COPY EVERYTHING BELOW INTO THE CHAT")
    print("=" * 72)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
