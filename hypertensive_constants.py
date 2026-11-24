"""
===============================================================================
HYPERTENSIVE_CONSTANTS.PY
Smart AI Infusion Pump - Hypertensive Emergency Digital Twin (Sodium Nitroprusside)

Purpose
-------
Sibling of constants.py for the SECOND clinical mode of this project:
hypertensive emergency, treated with sodium nitroprusside (SNP) instead
of norepinephrine. This file is deliberately separate from constants.py
rather than merged into it - the two drugs/conditions share almost no
literature-cited numeric values, and keeping them apart means neither
mode's citations can be silently confused with the other's during
review.

Same documentation philosophy as constants.py: every value is tagged
[CITED], [DERIVED], or [ASSUMPTION].

Scope note
----------
"Hypertensive emergency" here means the classic ICU/ED scenario this
literature actually describes: markedly elevated BP (historically
SBP/DBP >180/110-120 mmHg) requiring a titrated IV agent, NOT ordinary
outpatient chronic hypertension - sodium nitroprusside is never used for
routine BP management. This distinction is deliberate (see the
project's own design discussion) and should be reflected in any UI
labelling ("Hypertensive Emergency Mode", not "Hypertension Mode").
===============================================================================
"""

# ===========================================================
# BP STAGING (for severity profiles / baseline MAP0 generation)
# ===========================================================

# [CITED] Whelton PK, Carey RM, Aronow WS, et al. 2017 ACC/AHA/AAPA/ABC/
# ACPM/AGS/APhA/ASH/ASPC/NMA/PCNA Guideline for the Prevention, Detection,
# Evaluation, and Management of High Blood Pressure in Adults.
# Hypertension. 2018;71(6):e13-e115. PMID 29133356.
# [CORRECTED 2026-09-29 - previously cited as Circulation. 2018;138(17):
# e484-e594, re-verified against PubMed/AHA Journals and found wrong:
# that page range matches neither real publication of this guideline,
# and Circulation only carries a separate, shorter Executive Summary
# (2018;138(17):e426-e483) of the same guideline, not the full document
# actually used for the detail cited throughout this file.]
# Stage 2 hypertension: SBP >=140 or DBP >=90 mmHg.
STAGE2_SBP_MIN_MMHG = 140.0
STAGE2_DBP_MIN_MMHG = 90.0

# [CITED] Same guideline (and consistently re-stated in later AHA
# scientific statements, e.g. Whelton PK et al., and StatPearls
# "Hypertensive Emergency"/"Hypertensive Crisis"). Historical threshold
# for a hypertensive crisis: SBP/DBP >180/110-120 mmHg. "Hypertensive
# urgency" = this threshold WITHOUT new/worsening target-organ damage;
# "hypertensive emergency" = this threshold WITH target-organ damage.
# This project cannot simulate target-organ damage directly, so the
# distinction is approximated as a severity escalation (see
# MAP0_BY_PROFILE_MMHG below) rather than a true organ-damage flag -
# flagged explicitly as a modeling simplification, not presented as
# clinically equivalent.
CRISIS_SBP_MMHG = 180.0
CRISIS_DBP_LOW_MMHG = 110.0
CRISIS_DBP_HIGH_MMHG = 120.0

# [DERIVED] Standard MAP approximation from SBP/DBP: MAP = DBP + (SBP-DBP)/3.
# Used only to translate the SBP/DBP literature thresholds above into
# the MAP terms this simulation actually runs on (same "M AP is the
# regulated/controlled quantity" convention as the septic-shock twin).
def map_from_sbp_dbp(sbp_mmhg, dbp_mmhg):
    return dbp_mmhg + (sbp_mmhg - dbp_mmhg) / 3.0


# [DERIVED] Baseline (pre-treatment, presenting) MAP by severity
# profile, converted from the SBP/DBP bands above via map_from_sbp_dbp().
# Healthy reuses constants.MAP_NORMAL_MMHG directly (same normal range,
# not re-cited here to avoid duplicating that citation).
#   Stage2 (~140-179/90-109, i.e. hypertensive but below crisis level):
#     midpoint ~160/100 -> MAP ~120; banded around that.
#   Urgency (crisis-level BP, no organ damage): >=180/110 -> MAP >=133.3.
#   Emergency (crisis-level BP + organ damage): same BP threshold as
#     Urgency per the guideline - literature does NOT report a higher
#     BP band for "emergency" vs "urgency" (they're defined by organ
#     damage, not magnitude). The escalation used here is an
#     [ASSUMPTION]: proportionally higher MAP to give the simulation a
#     distinguishable, more severe case, exactly analogous to how the
#     septic-shock twin's own Mild/Moderate/Severe MAP0 subdivision
#     below the cited 65 mmHg threshold is flagged as an
#     undocumented-but-proportional assumption in constants.py.
MAP0_BY_PROFILE_MMHG = {
    "Healthy": (70.0, 100.0),     # [CITED] = constants.MAP_NORMAL_MMHG
    "Stage2": (106.0, 133.0),     # [DERIVED] ~140-179/90-109 band via map_from_sbp_dbp
    "Urgency": (133.3, 150.0),    # [CITED lower bound] >=180/110 crisis threshold
    "Emergency": (150.0, 180.0),  # [ASSUMPTION] proportional escalation above Urgency
}

PROFILE_WEIGHTS = {
    "Healthy": 30,
    "Stage2": 30,
    "Urgency": 25,
    "Emergency": 15,
}

# [ASSUMPTION] SVR by profile. Hypertensive emergency is predominantly
# an elevated-SVR (vasoconstricted) state with relatively preserved
# cardiac output, the pathophysiological mirror of septic shock's
# low-SVR/vasodilated state (see cardiovascular.py - the same
# MAP = CVP + SVR*CO/80 relationship applies in both directions).
# Unlike the septic-shock SVR tiers (Landry/Melo & Peters, cited in
# constants.py), there is no equivalent widely-cited invasive
# hemodynamic profiling literature specifically for hypertensive
# emergency patients (they are rarely instrumented with the
# thermodilution/arterial catheters those septic-shock studies used) -
# these bands are therefore explicit assumptions, proportionally
# escalated above the normal range, NOT literature-anchored the way the
# septic tiers are. Flagged honestly rather than presented as cited.
SVR_BY_PROFILE_DYN_S_CM5 = {
    "Healthy": (900.0, 1200.0),   # [CITED] = constants.SVR_NORMAL_DYN_S_CM5 range
    "Stage2": (1300.0, 1600.0),   # [ASSUMPTION] proportional elevation
    "Urgency": (1600.0, 2000.0),  # [ASSUMPTION] proportional elevation
    "Emergency": (2000.0, 2600.0),  # [ASSUMPTION] proportional elevation
}

# ===========================================================
# STAGED MAP-REDUCTION TARGET (the controller's actual setpoint
# schedule - NOT a fixed target like the septic-shock side's 65 mmHg)
# ===========================================================

# [CITED, primary source added 2026-09-29] Staged reduction protocol
# for hypertensive emergencies: reduce MAP by NO MORE than ~20-25%
# within the first hour, then toward a BP of roughly 160/100-110 mmHg
# over the next 2-6 hours, with gradual normalization over 24-48 hours.
# Precipitous correction risks cerebral, coronary, or renal
# hypoperfusion by overwhelming vascular autoregulation - this is the
# central safety fact that makes this controller genuinely different
# from the septic-shock one: "MAP is too high" does NOT mean "drive it
# down as fast as possible."
#
# Primary source: Whelton PK, Carey RM, Aronow WS, et al. 2017 ACC/AHA
# guideline (see STAGE2_SBP_MIN_MMHG's citation above for the full
# reference), Section 11.2 "Hypertensive Crises - Emergencies and
# Urgencies" - this exact staged schedule is stated there, confirmed
# via web search 2026-09-29. An earlier version of this comment cited
# NHLBI/MSD Manual/StatPearls only, without an ACC/AHA cross-reference -
# those remain valid secondary/corroborating sources (all state the
# identical numbers), but the primary citation for this project's
# report should be the ACC/AHA guideline already cited above, for
# consistency with the guideline cited for the severity-profile
# thresholds just above.
STAGE1_MAX_REDUCTION_FRACTION = 0.25   # [CITED] max fractional MAP drop allowed in hour 1
STAGE1_DURATION_HOURS = 1.0            # [CITED]

# [DERIVED] "~160/100-110 mmHg" converted to MAP via map_from_sbp_dbp():
#   160/100 -> MAP 120.0
#   160/110 -> MAP 126.7
STAGE2_TARGET_MAP_BAND_MMHG = (120.0, 126.7)  # [DERIVED] see above
STAGE2_END_HOURS = 6.0   # [CITED] "2-6 hours" - end of this phase

STAGE3_END_HOURS = 48.0  # [CITED] "24-48 hours" - gradual normalization complete

# Source (both searches, cross-consistent):
#   - NHLBI hypertensive-emergency staged BP reduction guidance, as
#     summarized in MSD Manual Professional Edition, "Hypertensive
#     Emergencies" (Cardiovascular Disorders > Hypertension section).
#   - StatPearls (NCBI Bookshelf) "Hypertensive Emergency", NBK470371.

# ===========================================================
# SODIUM NITROPRUSSIDE PHARMACOKINETICS
# ===========================================================

# [CITED] FDA NITROPRESS (sodium nitroprusside) prescribing information,
# accessdata.fda.gov, label id 071961s013, 2014 revision.
# "The half-life of nitroprusside... is about 2 minutes." (circulatory
# half-life of the parent drug itself, prior to its very rapid
# intraerythrocytic conversion to cyanide/thiocyanate - NOT the
# thiocyanate elimination half-life, see below).
NITROPRUSSIDE_HALFLIFE_MIN = 2.0  # [CITED]

# [DERIVED] ke = ln(2) / t_half, standard first-order elimination
# relationship (same relationship constants.py uses in the other
# direction, ke = CL/Vd, for norepinephrine).
import math as _math
NITROPRUSSIDE_KE_PER_MIN = _math.log(2) / NITROPRUSSIDE_HALFLIFE_MIN  # [DERIVED] ~0.3466 /min

# [CITED, qualitative] + [DERIVED, numeric] Same FDA label: "Infused
# sodium nitroprusside is rapidly distributed to a volume that is
# approximately coextensive with the extracellular space." The label
# gives no numeric Vd, so the standard physiological approximation for
# extracellular fluid volume (~20% of body weight - a standard
# textbook value, e.g. Guyton & Hall "Textbook of Medical Physiology")
# is applied per-patient rather than as a fixed population constant
# (unlike norepinephrine's Vd, which Beloeil et al. found NOT
# weight-dependent - nitroprusside's IS, by the label's own
# description of its distribution volume).
ECF_FRACTION_OF_BODY_WEIGHT = 0.20  # [CITED, generic physiology] standard ECF approximation

# [DERIVED] CL = ke * Vd. Computed per-patient in
# hypertensive_patient_generator.py once Vd is known (Vd depends on
# each patient's sampled weight), rather than stored as a single
# population constant here.

# [ASSUMPTION] Population inter-individual variability on clearance -
# same generic PopPK convention constants.py uses for norepinephrine
# (CLEARANCE_INTER_INDIVIDUAL_CV = 0.25), applied here for the same
# reason: no directly-cited inter-patient variability figure exists for
# nitroprusside clearance specifically.
CLEARANCE_INTER_INDIVIDUAL_CV = 0.25

# [CITED] FDA NITROPRESS label: reconstitution/premix concentration.
# "The solution... must be further diluted in 250-1000 mL of sterile 5%
# dextrose injection", from a 50 mg vial, yielding 50-200 mcg/mL
# depending on dilution volume. This project uses the more concentrated
# standard ICU premix (50 mg / 250 mL D5W = 200 mcg/mL), the same
# "smallest bag / most concentrated standard premix" convention already
# used for norepinephrine's own 4 mg/250 mL premix in constants.py.
DRUG_CONCENTRATION_MCG_ML = 200.0  # [CITED]

# ===========================================================
# SODIUM NITROPRUSSIDE DOSING RANGE
# ===========================================================

# [CITED] FDA NITROPRESS label, dosing section.
NITROPRUSSIDE_STARTING_DOSE_MCG_KG_MIN = 0.3   # [CITED] initial infusion rate
NITROPRUSSIDE_USUAL_DOSE_RANGE_MCG_KG_MIN = (0.5, 10.0)  # [CITED] "usual dose rate"
NITROPRUSSIDE_MAX_DOSE_MCG_KG_MIN = 10.0        # [CITED] maximum recommended rate
NITROPRUSSIDE_MAX_DOSE_DURATION_MIN = 10.0      # [CITED] max rate must never last longer than this

# [CITED] FDA NITROPRESS label: prolonged-infusion ceilings for renal
# function (distinct from the short-term max-rate limit above - these
# apply to infusions run longer than the acute titration window).
NITROPRUSSIDE_PROLONGED_DOSE_NORMAL_RENAL_MCG_KG_MIN = 3.0  # [CITED]
NITROPRUSSIDE_PROLONGED_DOSE_ANURIC_MCG_KG_MIN = 1.0        # [CITED]

# ===========================================================
# PHARMACODYNAMICS (Emax model, VASODILATION - negative effect on SVR)
# ===========================================================

# [GAP, explicitly not filled] As with norepinephrine's EC50/Emax/Hill
# in constants.py, no published Emax/Hill fit of nitroprusside plasma
# concentration vs SVR/MAP was found in the accessible literature (SNP
# is dosed by direct hemodynamic titration in practice, not by target
# plasma level, so this kind of PK/PD fit is simply not clinically
# needed and rarely studied). EC50 below is therefore an [ASSUMPTION],
# calibrated the same way constants.EC50_ASSUMED was: using steady-state
# Cp_ss = R/CL for a representative 70 kg patient across the CITED dose
# range above, so ordinary clinical doses land in the meaningful part of
# the curve rather than being picked blind.
#   Using this module's NITROPRUSSIDE_KE_PER_MIN and a representative
#   Vd = 0.20 L/kg * 70 kg = 14 L (so CL = ke*Vd ~ 4.85 L/min):
#   - Starting dose (0.3 mcg/kg/min -> 21 mcg/min) -> Cp_ss ~4.3 mcg/L
#   - Upper "usual" dose (4 mcg/kg/min -> 280 mcg/min, the top of the
#     0.5-4 mcg/kg/min range multiple clinical sources describe as
#     typical before needing the rarely-used higher end) -> Cp_ss ~57.7 mcg/L
#   - Max dose (10 mcg/kg/min -> 700 mcg/min) -> Cp_ss ~144.3 mcg/L
#   EC50 is set within the "typical usual dosing" band so ordinary
#   titration produces a meaningful, sub-saturated effect, with
#   saturation approached only near the max-dose end - same qualitative
#   shape/rationale as the norepinephrine side.
EC50_ASSUMED_MCG_L = 20.0  # [ASSUMPTION] mcg/L, calibrated per above reasoning

# [ASSUMPTION] Maximum fractional REDUCTION in SVR at full effect (note
# the sign: this is subtracted, whereas norepinephrine's EMAX_ASSUMED is
# added - see nitroprusside_pharmacodynamics.py). -0.7 means SVR falls
# to 30% of baseline at maximal effect, reflecting SNP's potency as a
# direct arteriolar/venous vasodilator (mechanism: NO donor -> cGMP ->
# smooth muscle relaxation - well-established pharmacology, magnitude
# of the SVR fraction itself is the assumption).
EMAX_ASSUMED = -0.70  # [ASSUMPTION]

HILL_COEFFICIENT_ASSUMED = 1.5  # [ASSUMPTION] same generic value/rationale as constants.py

# [DERIVED] Realistic pump flow-rate range at DRUG_CONCENTRATION_MCG_ML
# above, for a representative 70 kg patient (used only to sanity-check
# the range presented to the UI/controller - the firmware/pump itself
# clamps on mL/hr directly, same architecture as the norepinephrine side).
#   Starting dose: 0.3 mcg/kg/min * 70 kg = 21 mcg/min -> 6.3 mL/hr
#   Max dose: 10 mcg/kg/min * 70 kg = 700 mcg/min -> 210 mL/hr
MIN_FLOW_ML_HR = 0.0
MAX_FLOW_ML_HR = 210.0  # [DERIVED] at 200 mcg/mL, 70kg patient, max dose rate

# ===========================================================
# CYANIDE TOXICITY - DETERMINISTIC SAFETY MONITOR
# (explicitly NOT an AI-learned feature - see project design notes:
#  AI = advisory, deterministic controller = final authority, hardware
#  safety limits = independent protection. This belongs in the middle
#  tier, same as the septic-shock side's watchdog/limit-switch logic.)
# ===========================================================

# [CITED] FDA NITROPRESS label: "a patient... can safely receive about
# 10 mg/kg of sodium nitroprusside" - i.e. total cumulative buffering
# capacity of the body's methemoglobin pool before cyanide toxicity
# becomes likely.
CYANIDE_BUFFER_CAPACITY_MG_KG = 10.0  # [CITED]

# [CITED] FDA NITROPRESS label: "when more than 500 mcg/kg of sodium
# nitroprusside is administered faster than 2 mcg/kg/min, cyanide is
# generated faster than the unaided patient can eliminate it." Two
# conditions must BOTH hold for this specific rate-dependent risk (as
# opposed to the absolute buffer-capacity ceiling above): cumulative
# dose past 500 mcg/kg, AND current rate above 2 mcg/kg/min.
CYANIDE_RATE_RISK_THRESHOLD_MCG_KG_MIN = 2.0     # [CITED]
CYANIDE_RATE_RISK_CUMULATIVE_MCG_KG = 500.0      # [CITED]

# [CITED] FDA NITROPRESS label: thiocyanate (the downstream metabolite
# actually responsible for LONGER-term toxicity once cyanide is
# converted) has an accumulation half-time of "3-4 days" - orders of
# magnitude slower than nitroprusside's own ~2-minute circulatory
# half-life. This is why prolonged infusions carry cumulative risk even
# at rates that look "safe" tick-to-tick: thiocyanate simply doesn't
# clear within the timescale of a single ICU shift, let alone this
# simulation's compressed timescale.
THIOCYANATE_HALFLIFE_DAYS = 3.5  # [CITED] midpoint of stated 3-4 day range

# ===========================================================
# ILLNESS RECOVERY (vascular tone normalization) - found missing during
# real extended dashboard use, not part of the original foundation
# ===========================================================
# [ASSUMPTION] Without this, svr0 is a fixed constant for the ENTIRE
# simulation (unlike the septic-shock twin, which has an equivalent
# mechanism in constants.py/digital_twin.py). Combined with the staged
# safety floor descending for up to 48 hours regardless, a fixed svr0
# means the controller must administer an ever-increasing dose just to
# keep MAP tracking an ever-stricter floor against vascular resistance
# that never gets any easier to overcome - observed directly during
# real multi-hour dashboard use as a flow rate that "just kept
# increasing" toward the maximum with no plateau. Real hypertensive
# emergencies DO resolve with treatment + time (the acute vasospasm/
# vasoconstrictive trigger settles), the same way septic shock's
# vasodilation resolves - this was simply never modeled.
#
# Mechanism: once MAP has held AT OR ABOVE the current safety floor
# (i.e. safely controlled, not breaching) continuously for
# HTN_RECOVERY_GRACE_HOURS, svr0 drifts exponentially toward
# HTN_RECOVERY_SVR0_CEILING_DYN_S_CM5 with time constant
# HTN_RECOVERY_TAU_HOURS. Any breach below the floor resets the
# stabilization clock, pausing recovery - "if the emergency isn't
# actually controlled, the underlying vasoconstriction hasn't
# resolved" - the same "if not controlled, still needs support" logic
# constants.py already documents for the septic-shock mechanism this
# mirrors.
HTN_RECOVERY_GRACE_HOURS = 0.5   # [ASSUMPTION] half an hour of safe control before recovery begins
HTN_RECOVERY_TAU_HOURS = 6.0     # [ASSUMPTION] recovery time constant

# [DERIVED] Same "normal vascular tone" target as the septic-shock
# twin's own recovery ceiling (constants.ILLNESS_RECOVERY_SVR0_CEILING_
# DYN_S_CM5 = 1050.0, the midpoint of the cited Healthy SVR range) -
# both conditions resolve toward the same normal physiological state,
# just approached from opposite directions.
HTN_RECOVERY_SVR0_CEILING_DYN_S_CM5 = 1050.0

# ===========================================================
# BARORECEPTOR RESETTING (map_set drift) - found missing during the
# same real extended dashboard session
# ===========================================================
# [ASSUMPTION] map_set (the baroreflex's defended pressure setpoint) is
# initialized to each patient's own presenting MAP - see
# hypertensive_patient_generator.py - but was never updated afterward.
# Combined with the staged floor pushing MAP down for up to 48 hours,
# a FIXED map_set means the (map_set - MAP) gap the baroreflex reacts
# to keeps GROWING rather than shrinking the longer treatment
# continues, producing unbounded, ever-worsening reflex tachycardia
# with no physiological limit - observed directly as HR climbing past
# 160 bpm and still rising after hours of real dashboard use. Real
# baroreceptors DO reset toward whatever pressure has been sustained
# recently (this is the same well-established "baroreflex resetting"
# concept already cited as the reason map_set starts at the presenting
# MAP in the first place) - it was only ever modeled as a one-time
# initial condition, never as an ongoing adaptation.
#
# Mechanism: map_set drifts continuously (not gated by safe-control
# status, unlike svr0 above - real baroreceptor adaptation responds to
# sustained pressure levels regardless of whether treatment has fully
# succeeded yet) toward the CURRENT actual MAP, with a time constant
# deliberately much slower than the acute BP-correction schedule -
# real baroreceptor resetting takes on the order of days, compressed
# here to roughly the same simulation timescale as the illness-recovery
# mechanism above so early acute tachycardia is preserved while
# indefinite escalation over many hours is not.
BARORECEPTOR_RESET_TAU_HOURS = 24.0  # [ASSUMPTION]
