"""
===============================================================================
CONSTANTS.PY
Smart AI Infusion Pump - Mechanistic Digital Twin

Purpose
-------
This module centralizes all numerical parameters used throughout the
digital twin, including pharmacokinetic, pharmacodynamic,
cardiovascular, physiological, and infusion pump constants.

Documentation philosophy
------------------------
Every numerical value is explicitly classified according to its origin
to ensure complete transparency and traceability throughout the
simulation.

Three classifications are used:

    [CITED]
        Numerical values obtained directly from published literature,
        clinical guidelines, drug labels, or other authoritative
        references.

    [DERIVED]
        Values calculated from published data using established
        physiological or pharmacokinetic relationships.

    [ASSUMPTION]
        Values for which no directly applicable literature was
        available. These parameters represent either standard modelling
        conventions or physiologically reasonable assumptions adopted
        to maintain stable and realistic simulation behaviour.

This classification allows every parameter used in the digital twin to
be traced back to either published evidence or an explicitly stated
modelling assumption, thereby supporting transparency, reproducibility,
and scientific defensibility.
===============================================================================
"""

# ===========================================================
# PHARMACOKINETICS (one-compartment model)
# ===========================================================

# [CITED] Volume of distribution, 8.8 L.
# Source 1: Pfizer Levophed (norepinephrine bitartrate) FDA drug label,
#   "Clinical Pharmacology" section: "The volume of distribution is 8.8 L."
# Source 2 (cross-validation): Beloeil H, Mazoit JX, Benhamou D,
#   Duranteau J. "Norepinephrine kinetics and dynamics in septic shock
#   and trauma patients." Br J Anaesth. 2005;95(6):782-788.
#   PMID 16227334. Reported CL and T1/2 at two SAPS II severities;
#   back-calculating Vd = T1/2 * CL / ln(2) from BOTH reported points
#   gives ~8.7-8.8 L, independently matching the FDA label value.
# Beloeil et al. explicitly tested and found body weight did NOT
# significantly affect Vd in their adult ICU cohort - so this is used
# as a fixed population value, NOT weight-scaled per patient.
VD_NOREPINEPHRINE_L = 8.8

# [CITED] Clearance range and its relationship to illness severity.
# Source: Beloeil et al. 2005 (above). One-compartment NONMEM model,
#   23 adult ICU patients (12 septic shock, 11 trauma) requiring
#   norepinephrine. Direct quote: "CL and T1/2 varied from 3 litre
#   min-1 and 2 min, respectively, when SAPS II=20 to 0.9 litre min-1
#   and 6.8 min when SAPS II=60." Group, gender, and body weight did
#   NOT significantly influence CL or Vd - only SAPS II (illness
#   severity) did.
# NOTE: the paper's exact CL/T1/2 figures at each SAPS II anchor were
# not independently re-confirmed against the full-text PDF during the
# most recent literature pass (only the abstract-level relationship -
# CL negatively related to SAPS II, no weight/gender/group effect - was
# re-verified). Cross-validated indirectly: the FDA Levophed label's
# independently reported average clearance (3.1 L/min) sits very close
# to the CL_AT_SAPS_20 anchor below. Re-check against the primary PDF
# before citing these two numbers in anything published.
CL_AT_SAPS_20_L_MIN = 3.0   # [CITED, re-verify against full text] least-severe end of studied range
CL_AT_SAPS_60_L_MIN = 0.9   # [CITED, re-verify against full text] most-severe end of studied range
SAPS_LOW_ANCHOR = 20
SAPS_HIGH_ANCHOR = 60

# [ASSUMPTION] Representative SAPS II values mapped to this project's
# four clinical Profiles. Beloeil's cohort was ALL vasopressor-requiring
# ICU patients (SAPS II 20-60 observed range) - there is no drug-specific
# data below SAPS II~20, so "Healthy" is anchored at the same low end
# as "Mild" rather than extrapolated below the studied range. Mild/
# Moderate/Severe are spread across Beloeil's actual observed range
# (interpolation, not extrapolation).
SAPS_II_BY_PROFILE = {
    "Healthy": 20,   # [ASSUMPTION] anchored at Beloeil's low-end boundary
    "Mild": 20,      # [CITED anchor] matches Beloeil's least-severe point
    "Moderate": 40,  # [ASSUMPTION] midpoint interpolation within Beloeil's range
    "Severe": 60,    # [CITED anchor] matches Beloeil's most-severe point
}

# [CITED, mechanism] + [ASSUMPTION, magnitude applied here]
# Aging reduces PLASMA norepinephrine clearance via reduced neuronal
# reuptake (NOT the same clearance pathway/population as Beloeil above -
# see caveat below).
# Source 1: Esler M, Kaye D, Lambert G, et al. "Aging effects on human
#   sympathetic neuronal function." PMID 7840332. Healthy men aged
#   20-30 vs 60-75: plasma NE clearance 22% lower in older men (P<0.05).
#   [RE-VERIFIED] confirmed directly against the PubMed abstract: "Mean
#   plasma norepinephrine concentration was 66% higher in older men,
#   attributable to 22% lower norepinephrine plasma clearance (P<0.05)."
# Source 2: Esler M et al. Am J Physiol Regul Integr Comp Physiol.
#   2002. Same paradigm, follow-up analysis: 27.5% lower clearance in
#   older men (P<0.05).
# CAVEAT (stated explicitly, not hidden): both sources measured
# ENDOGENOUS norepinephrine clearance in HEALTHY subjects, not
# exogenously infused norepinephrine in critically ill ICU patients
# (Beloeil's population). The underlying clearance mechanism (hepatic/
# neuronal degradation) should apply to norepinephrine regardless of
# origin, so extrapolating is reasonable, but this specific combination
# (age effect x ICU population) has not been directly measured in one
# study. Beloeil tested age as a covariate and did not find significance
# in his 23-patient cohort - plausibly because illness severity
# variance dominated a sample too small to also resolve a secondary
# ~25% age effect.
AGE_REFERENCE_YOUNG = 25   # [DERIVED] approx. mean of Esler's 20-30 group
AGE_REFERENCE_OLD = 67     # [DERIVED] approx. mean of Esler's 60-75 group
CLEARANCE_REDUCTION_BY_OLD_AGE = 0.25  # [ASSUMPTION] midpoint of cited 22-27.5% range
AGE_CLEARANCE_FLOOR = 0.70  # [ASSUMPTION] don't extrapolate below this past age 67

# [ASSUMPTION] Standard population-PK convention: lognormal
# inter-individual variability around a typical value. This specific
# magnitude (25% CV) is NOT reported in Beloeil et al. - it is a
# generic PopPK convention (see Ette & Williams, "Pharmacometrics: The
# Science of Quantitative Pharmacology"), applied here because some
# patient-to-patient spread is expected and none was available to cite
# directly for this drug.
CLEARANCE_INTER_INDIVIDUAL_CV = 0.25

# ===========================================================
# CARDIOVASCULAR BASELINES
# ===========================================================

# [CITED] Normal adult resting values, consistent across multiple
# clinical hemodynamic reference sources (ICU handbook parameter
# tables; StatPearls "Physiology, Cardiac Output" and "Physiology,
# Stroke Volume").
CO_NORMAL_L_MIN = (4.0, 8.0)      # [CITED]
CO_NORMAL_TYPICAL_L_MIN = 5.0      # [CITED] "classic average value"
SV_NORMAL_ML = (60, 100)           # [CITED]
SVR_NORMAL_DYN_S_CM5 = (800, 1200)  # [CITED] multiple consistent sources

# [CITED] SVR by severity profile.
# Healthy: normal range, as above.
# Mild/Moderate/Severe shock: anchored to real septic-shock SVR data.
#   - Landry DW, Levin HR, Gallant EM, Ashton RC Jr, Seo S, D'Alessandro D,
#     Oz MC, Oliver JA. "Vasopressin Deficiency Contributes to the
#     Vasodilation of Septic Shock." Circulation. 1997;95(5):1122-1125.
#     PMID 9054839. [RE-VERIFIED 2026-09-22 against PubMed/AHA Journals -
#     corrected from a prior version of this comment, which had the
#     author initial wrong ("Landry AT") and cited the wrong PMID
#     (9679728).] Septic shock diagnostic criterion used: SVR < 800
#     dyn.s/cm5. Their septic shock cohort's baseline (pre-vasopressin)
#     SVR was ~644 dyn.s/cm5 (from "increased from 644 to 1187... P<.001").
#     [NOT RE-VERIFIED] the specific 644/1187 SVR figures themselves were
#     not re-confirmed against the full text (only available past the
#     abstract) - re-check before citing externally.
#   - Melo J, Peters JI. "Low systemic vascular resistance:
#     differential diagnosis and outcome." Crit Care. 1999;3(3):71-77.
#     DOI 10.1186/cc343. [RE-VERIFIED 2026-09-22 against PubMed/BioMed
#     Central - corrected from a prior version of this comment, which
#     misattributed this exact paper (same journal/volume/issue/pages)
#     to "Groeneveld ABJ et al."] Sepsis-syndrome group SVR: 445 +/- 168
#     dyn.s/cm5, associated with high mortality (50%) - used as the
#     Severe anchor. [RE-VERIFIED] confirmed directly against the
#     abstract: "mean SVR of 445 +/- 168 dynes x s/cm5" in the
#     sepsis-syndrome group.
SVR_BY_PROFILE_DYN_S_CM5 = {
    "Healthy": (900, 1200),    # [CITED] normal range
    "Mild": (700, 800),        # [CITED] just below Landry's <800 shock threshold
    "Moderate": (600, 700),    # [CITED] near Landry's ~644 septic shock baseline
    "Severe": (400, 500),      # [CITED] near Melo & Peters' 445+/-168 severe/mortality anchor
}

# [Healthy: CITED] [Mild/Moderate/Severe: ASSUMPTION] Target baseline
# (pre-drug) MAP by profile, used in patient_generator.py to DERIVE CO0
# from a sampled MAP0 and the already profile-cited svr0, rather than
# sampling CO0 independently.
#
# BUG this fixes: with CO0 previously drawn independently from the full
# CO_NORMAL_L_MIN range, MAP = CVP + SVR*CO/80 could land far outside
# the clinically expected band purely from combinatorial variance -
# verified empirically across 500 seeded "Healthy" patients, baseline
# MAP ranged 50.3-121.9 mmHg (17.4% below the 65 mmHg shock threshold,
# 26.6% outside MAP_NORMAL_MMHG) despite profile="Healthy" implying no
# shock. Real autoregulation keeps MAP itself, not CO or SVR
# individually, within a tight homeostatic band, so MAP0 is now the
# variable sampled directly (CO0 becomes the derived one) - same
# "sample the physiologically-regulated quantity, derive the rest"
# pattern already used for CO0/SV0/HR above.
#
# Sourcing:
#   Healthy: [CITED] reuses MAP_NORMAL_MMHG directly - a healthy
#     profile should start within the normal clinical MAP range.
#   Mild/Moderate/Severe: [ASSUMPTION] upper bound anchored at the
#     [CITED] 65 mmHg Sepsis-3 / Surviving Sepsis Campaign septic-shock
#     threshold (a patient needing vasopressor support is by definition
#     below this pre-treatment) - see TARGET_MAP_MMHG in digital_twin.py
#     for the same citation. No literature source grading septic shock
#     severity tiers by a specific MAP band was found (only the SVR
#     citations above are per-tier), so the subdivision BELOW 65 mmHg
#     across Mild/Moderate/Severe is a proportional, undocumented
#     assumption - flagged here rather than presented as cited.
MAP0_BY_PROFILE_MMHG = {
    "Healthy": (70, 100),   # [CITED] = MAP_NORMAL_MMHG
    "Mild": (55, 64),       # [ASSUMPTION] just below the 65 mmHg shock threshold
    "Moderate": (45, 55),   # [ASSUMPTION] proportional step down from Mild
    "Severe": (30, 45),     # [ASSUMPTION] proportional step down from Moderate
}

# [ASSUMPTION] Central venous pressure, used as a small constant offset
# in MAP = CVP + SVR*CO/80. Real CVP normal range is ~2-8 mmHg (widely
# reported clinical reference range); held fixed here rather than
# modeled dynamically - a documented simplification.
CVP_TYPICAL_MMHG = 4.0

# ===========================================================
# CLINICAL MAP TARGET RANGE (for a clinician-adjustable target)
# ===========================================================

# [CITED] Real, trial-studied range of MAP targets used in septic shock
# resuscitation, spanning both arms of the SEPSISPAM RCT.
# Source: Asfar P, Meziani F, Ruckly S, et al. "High versus Low
# Blood-Pressure Target in Patients with Septic Shock." N Engl J Med.
# 2014;370(17):1583-1593. PMID 24635770. Multicenter RCT, n=776,
# randomized to MAP target 80-85 mmHg ("high") vs 65-70 mmHg ("low").
# No significant 28-/90-day mortality difference overall; the higher
# target reduced need for renal-replacement therapy specifically in
# patients with pre-existing chronic hypertension, at the cost of more
# new atrial fibrillation and higher catecholamine exposure.
# This is the basis for letting a clinician pick a target within a
# real, studied range (e.g. individualizing higher for a chronically
# hypertensive patient) rather than treating 65 mmHg (this project's
# default, per the Surviving Sepsis Campaign - see TARGET_MAP_MMHG in
# digital_twin.py) as the only valid value. Values outside this range
# have not been studied in this trial and are not supported here.
TARGET_MAP_RANGE_MMHG = (65.0, 85.0)

# ===========================================================
# BAROREFLEX
# ===========================================================

# [CITED, structure only] First-order (or first-order-delay) ODE
# modeling of the baroreflex is standard in cardiovascular modeling
# literature - e.g. Ringwood JV, Malpas SC, baroreflex modeling papers;
# see also the general form dHR/dt = k(HRset-HR) - G(MAP-MAPset) used
# in multiple cardiovascular control papers found during literature
# review.
# [GAP, explicitly not filled] A single clean numeric gain constant in
# directly usable units (bpm per mmHg, per-tick) was not found -
# published baroreflex sensitivity values use inconsistent units
# (commonly ms/mmHg via RR-interval methods) and vary by study/method.
# k and G below are ASSUMPTIONS, tuned for stable simulation behavior.
BARORELFEX_K = 0.15    # [ASSUMPTION] HR relaxation rate toward set point
BAROREFLEX_G = 0.30    # [ASSUMPTION] HR response gain to MAP deviation

# [ASSUMPTION] hr_set and map_set are sampled independently per patient
# (see patient_generator.py), each from a real normal reference range -
# but occasionally the combination lands both near the high end at
# once (e.g. hr_set~98, map_set~97), and since the treatment target
# (65 mmHg) sits well below either range, the RAW (map_current-map_set)
# gap can exceed 30 mmHg. Applied linearly through BAROREFLEX_G, this
# produced simulated sustained HR above 145 bpm in ~20% of a 60-patient
# sweep - clinically extreme, and not something a real baroreflex does:
# the reflex response saturates at large deviations rather than scaling
# linearly forever. Capping the deviation term the baroreflex "sees"
# keeps individual variability in hr_set/map_set (both still real,
# still drive HR independently) while preventing this specific
# co-occurrence from compounding into an unrealistic spike.
#
# [FIX, 2026-09-27] This cap was only ever wired into digital_twin.py
# for the "already at/near target" case - during the acute pre-target
# phase it stayed fully uncapped (max_deviation_mmhg=None), which
# defeats the purpose documented above: with map_set/hr_set both
# sampled up to ~100 and presenting MAP as low as ~20-30 during severe
# shock, the raw deviation can exceed 70-80 mmHg, driving HR past 130
# bpm within the first couple of simulated minutes - well before MAP
# has recovered, and well past what this constant was meant to prevent.
# Two-tier fix: a generous but still-bounded cap during the acute
# phase (large enough that genuine emergency compensatory tachycardia -
# which mechanistically helps raise CO/MAP via compute_cardiac_output -
# is still preserved, just not unlimited), tightening to the original,
# stricter cap once MAP is at/near target (see digital_twin.py).
BAROREFLEX_MAX_DEVIATION_ACUTE_MMHG = 25.0
BAROREFLEX_MAX_DEVIATION_MMHG = 10.0

# ===========================================================
# ILLNESS RECOVERY (WEANING) DYNAMICS
# ===========================================================

# [ASSUMPTION] Without this section, a patient's baseline vascular tone
# (svr0) is a fixed constant for the entire simulation - the illness
# itself never resolves, so the drug requirement to hold MAP at target
# never decreases, and the baroreflex gap driving compensatory
# tachycardia (defended map_set vs the 65 mmHg treatment target, which
# is deliberately BELOW normal - see TARGET_MAP_RANGE_MMHG) never
# closes either. Real septic shock resolves under supportive care over
# ~24-72+ hours; this compresses that resolution to a simulateable
# timescale, using the same "time-compression" precedent already
# applied to the controller's titration cadence (see MAX_STEP_MCG_KG_MIN
# in expert_controller.py).
#
# Mechanism: once MAP has held within the controller's own target
# deadband continuously for ILLNESS_RECOVERY_GRACE_MIN (representing
# initial hemodynamic stabilization), svr0 drifts exponentially toward
# ILLNESS_RECOVERY_SVR0_CEILING_DYN_S_CM5 with time constant
# ILLNESS_RECOVERY_TAU_MIN. Any excursion outside the deadband resets
# the stabilization clock, pausing recovery - "if the underlying
# illness is not controlled, the patient still needs support."
# As svr0 recovers, LESS drug is needed to hold target - the existing
# PD controller tapers flow down on its own with no special-cased
# weaning logic, and the shrinking gap between MAP and map_set lets HR
# decay back toward hr_set as a genuine consequence, not a patch.
ILLNESS_RECOVERY_GRACE_MIN = 5.0    # [ASSUMPTION] stabilization period before recovery begins
ILLNESS_RECOVERY_TAU_MIN = 30.0     # [ASSUMPTION] recovery time constant - "can't be rushed"

# [DERIVED] Recovery ceiling = midpoint of the Healthy SVR range
# (SVR_BY_PROFILE_DYN_S_CM5["Healthy"] = (900, 1200) above), i.e. shock
# resolves toward normal vascular tone, not merely toward "less severe
# shock."
ILLNESS_RECOVERY_SVR0_CEILING_DYN_S_CM5 = 1050.0

# [ASSUMPTION] Display-only advisory threshold: HR persisting above
# this value despite MAP already being at target is flagged for
# clinician review (e.g. pain, agitation, fever, other non-shock causes
# of tachycardia) - this NEVER feeds back into dosing; it is purely
# informational, matching this project's existing safety architecture
# where only expert_controller.py actuates the pump.
HR_ADVISORY_THRESHOLD_BPM = 110.0

# ===========================================================
# SpO2 PERFUSION-ADEQUACY TOLERANCE
# ===========================================================

# [ASSUMPTION] physiology.py's SpO2 model switches between "declining"
# and "recovering" based on whether MAP is above/below the 65 mmHg
# perfusion-adequacy threshold (same cited value as the Surviving
# Sepsis Campaign target). A hard cutoff with zero tolerance means a
# patient whose controller-driven MAP settles at e.g. 64.97 (clinically
# indistinguishable from, and within the controller's own deadband of,
# 65) gets permanently stuck on the "declining" branch, ratcheting SpO2
# down to its floor and never recovering, purely from sub-1-mmHg
# numerical noise with no clinical meaning. This tolerance band creates
# a neutral zone around the threshold (MAP within it holds SpO2 steady
# rather than forcing a decline/recovery decision), matching the same
# magnitude as the controller's own TARGET_DEADBAND for consistency.
SPO2_PERFUSION_DEADBAND_MMHG = 3.0

# ===========================================================
# NORMAL VITAL SIGN REFERENCE RANGES
# ===========================================================
# [CITED] Standard adult clinical reference ranges, consistent across
# general clinical references (ICU/nursing hemodynamic parameter
# tables cited above; standard vital-sign normal ranges).
HR_NORMAL_BPM = (60, 100)
MAP_NORMAL_MMHG = (70, 100)
SPO2_NORMAL_PCT = (95, 100)

# ===========================================================
# CLINICAL NOREPINEPHRINE DOSING RANGE
# ===========================================================
# [CITED] Standard clinical dosing range for norepinephrine infusion,
# both bounds from SCCM's Surviving Sepsis Campaign guidelines (see
# REFERENCES.md #8).
#
# [CORRECTED 2026-10-05] Upper bound was previously 3.3 mcg/kg/min -
# that figure could not be traced to a specific source despite another
# literature pass (a live search turned up only a vague, unnamed
# "clinical context" mention, not a citable primary source the way
# every other [CITED] constant in this file is verified - studies DO
# report maximum doses USED as high as 0.2-5 mcg/kg/min, but a study's
# observed maximum isn't the same thing as a guideline's recommended
# upper bound, which is what this constant represents). Switched to
# SCCM's own clean, actually-stated upper bound of 3.0 instead of
# continuing to carry an unverifiable number.
NOREPINEPHRINE_DOSE_RANGE_MCG_KG_MIN = (0.01, 3.0)

# [CITED] Standard premix concentration: 4 mg norepinephrine in 250 mL
# D5W = 16 mcg/mL. Cross-validated across a Baxter commercial premix
# product, multiple hospital IV guidelines (Medinfo Galway, RAH
# Peripheral Noradrenaline Guideline), and clinical trial protocols
# (e.g. NCT06977659, CENSER2/NCT07343206) - all independently use this
# exact concentration. [RE-VERIFIED] confirmed the 4 mg/250 mL -> 16
# mcg/mL preparation directly against a commercially available premix
# product listing. This is what makes flow rate (mL/hr, what the pump
# actually displays) convertible into a real mass dose (mcg/min, what
# the PK equation needs).
DRUG_CONCENTRATION_MCG_ML = 16.0

# [DERIVED] Realistic pump flow-rate range at the above concentration.
# Lower bound: matches SCCM's cited minimum (0.01 mcg/kg/min) for a
# 70 kg patient -> 0.7 mcg/min -> 2.6 mL/hr at 16 mcg/mL.
# Upper bound: [CORRECTED 2026-10-05] NOT an explicitly stated clinical
# hard cap - the Medinfo Galway IV guideline (medinfogalway.ie/node/2824)
# states no such "3 mg/hour" ceiling anywhere in its text. 187.5 mL/hr
# is derived from the TOP ROW of that guideline's own Table 3 (peripheral
# dosing): 0.5 mcg/kg/min at a 100 kg patient = 188 mL/hr (confirmed
# directly against the primary PDF) -> 3000 mcg/hr = "3 mg/hour"
# equivalent, which is where the earlier wording came from.
# NOTE: this REPLACES the earlier (V1) 0-5 mL/hr range, which was not
# calibrated against a real drug concentration and is now known to be
# unrealistic - at 16 mcg/mL, 5 mL/hr only delivers ~1.3 mcg/min,
# below even the low end of typical maintenance dosing (2-4 mcg/min).
MIN_FLOW_ML_HR = 0.0
MAX_FLOW_ML_HR = 190.0  # rounded from the derived 187.5 mL/hr cap

# ===========================================================
# PHARMACODYNAMICS (Emax model)
# ===========================================================

# [CITED, structure only] Beloeil et al. 2005 (above) ALSO fit a
# classic Emax/Hill model to MAP vs norepinephrine plasma concentration
# in the same patients: MAP = MAP0 + MAPmax * C^gamma / (EC50^gamma + C^gamma)
# This confirms the Emax model IS the literature-standard approach for
# this exact drug/population.
#
# [GAP, explicitly not filled] The specific EC50/MAPmax/gamma numeric
# values from that model were NOT obtainable - the full text is
# paywalled, and a search-snippet fragment showing "EC50 = 70 ug/mL"
# is almost certainly a text-extraction error (that unit is implausible
# for norepinephrine plasma levels, which run in ng/mL-to-low-mcg/mL).
# Rather than use an untrustworthy number, EC50/Emax below are
# ASSUMPTIONS - but now calibrated against a real, derived Cp range
# rather than picked blind:
#   Using Cp_ss = R/CL (steady-state one-compartment PK):
#   - Typical maintenance dose (2-4 mcg/min, Drugs.com) at CL=3 L/min
#     (healthiest) -> Cp ~0.7-1.3 mcg/L
#   - Typical initial dose (8-12 mcg/min) at CL=0.9 L/min (sickest)
#     -> Cp ~8.9-13.3 mcg/L
#   - Extreme high dose near the MAX_FLOW_ML_HR ceiling (~50 mcg/min) at CL=0.9
#     -> Cp ~55.6 mcg/L
#   EC50 is set in the middle of the "typical therapeutic" zone so
#   ordinary clinical doses produce a meaningful but non-saturated
#   effect, with saturation approached only at high-end dosing -
#   the expected qualitative shape of an Emax curve.
EC50_ASSUMED = 6.0      # [ASSUMPTION] mcg/L, calibrated per above reasoning
EMAX_ASSUMED = 1.2      # [ASSUMPTION] maximum fractional increase in SVR
HILL_COEFFICIENT_ASSUMED = 1.5  # [ASSUMPTION] moderate sigmoidicity, typical
                                  # of Hill coefficients in Emax PK/PD literature (1-3 range)

# ===========================================================
# VTBI / KVO (unchanged from V1 - real infusion pump convention)
# ===========================================================
# [CITED] Keep Vein Open rate convention, standard across commercial
# infusion pumps (Alaris, B. Braun, etc. device manuals/clinical
# practice).
KVO_RATE_ML_HR = 2.0  # within the standard 1-5 mL/hr clinical KVO range
