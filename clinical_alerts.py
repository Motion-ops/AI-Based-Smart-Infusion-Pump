"""
===============================================================================
CLINICAL_ALERTS.PY
Smart AI Infusion Pump - Structured Clinical Advisory Panel

Purpose
-------
Produces the "Clinical Alert" advisories shown in native_app.py's Live
Monitor tab: for each abnormal vital currently present, a short
statement of (a) what is wrong, (b) the potential consequence if left
unchecked, and (c) an actionable recommendation - covering both clinical
modes (septic shock / hypotension and hypertensive emergency) and both
abnormal directions of MAP, plus SpO2 and HR.

Display-only, exactly like this project's other advisory layers (the AI
classifier/regressor, the AI risk model, HRAdvisory) - this module never
touches patient.flow_rate or any other simulation state. The
deterministic expert_controller()/hypertensive_expert_controller()
functions remain the sole actuators; this module only interprets their
output and the patient's vitals for the clinician-facing display, and
(via native_app.py's own publish_event() wiring) the remote dashboard's
alarm/event feed.

Every threshold used here already exists elsewhere in this project,
cited there - this module deliberately does not introduce any new,
uncited numeric threshold. See each function's own comment for the
specific source. In particular, the hypertensive "malignant-range MAP"
alert reuses hypertensive_expert_controller.EMERGENCY_MAP_MMHG_HIGH
(~133 mmHg, derived from the cited ACC/AHA >=180/110 crisis threshold)
rather than a separate, uncited round number.
===============================================================================
"""

from expert_controller import TARGET_DEADBAND, EMERGENCY_MAP_MMHG as SEPTIC_EMERGENCY_MAP_MMHG
from hypertensive_expert_controller import EMERGENCY_MAP_MMHG_HIGH, FLOOR_BREACH_MARGIN_MMHG

CRITICAL = "critical"
WARNING = "warning"


# ===========================================================
# SHARED (both modes) - SpO2, HR
# ===========================================================

def _spo2_alerts(spo2):
    """Hypoxia is hypoxia regardless of which drug/condition is being
    managed. Thresholds match native_app.py's own spo2_status() (<90
    Critical, <95 Low), itself consistent with constants.SPO2_NORMAL_PCT
    = (95, 100)."""
    if spo2 < 90:
        return [{
            "severity": CRITICAL,
            "title": f"Critical Hypoxia (SpO2 <90%, measured {spo2:.0f}%)",
            "consequence": "Tissue hypoxia with risk of cardiac arrhythmia and "
                            "end-organ injury if sustained.",
            "recommendation": "Increase supplemental oxygen/FiO2 immediately; "
                               "assess airway patency and ventilation; consider "
                               "an arterial blood gas.",
        }]
    if spo2 < 95:
        return [{
            "severity": WARNING,
            "title": f"Mild Hypoxia / Suboptimal Saturation (SpO2 {spo2:.0f}%)",
            "consequence": "Developing ventilation-perfusion mismatch or shallow "
                            "respiratory effort.",
            "recommendation": "Titrate supplemental oxygen to maintain SpO2 >=95%; "
                               "encourage deep breathing/spirometry.",
        }]
    return []


def _hr_alerts(hr, hr_advisory_flag):
    """Critical bradycardia/tachycardia thresholds match native_app.py's
    own hr_status() (<50 or >140 Critical) - the same numbers already
    color-coded on the HR tile, just explained here instead of only
    colored. The milder "persistently elevated despite target reached"
    case reuses the engine's own HRAdvisory flag rather than re-deriving
    that judgment call here - SimEngine.step() already excludes expected
    shock-compensatory tachycardia from that flag (only true for septic
    mode; hypertensive mode's HRAdvisory is always False by design, see
    its own comment), so this function does not need to repeat that
    reasoning."""
    if hr < 50:
        return [{
            "severity": CRITICAL,
            "title": f"Severe Bradycardia (HR <50 bpm, measured {hr:.0f})",
            "consequence": "Reduced cardiac output and tissue perfusion; risk of "
                            "progression to hemodynamic instability.",
            "recommendation": "Assess for reversible causes (hypoxia, drug effect, "
                               "conduction abnormality); consider atropine or pacing "
                               "per the ACLS bradycardia algorithm if symptomatic.",
        }]
    if hr > 140:
        return [{
            "severity": CRITICAL,
            "title": f"Severe Tachycardia (HR >140 bpm, measured {hr:.0f})",
            "consequence": "Increased myocardial oxygen demand and reduced diastolic "
                            "filling time; risk of arrhythmia or ischemia.",
            "recommendation": "Evaluate for pain, agitation, fever, hypovolemia, or "
                               "arrhythmia; obtain an ECG; treat the underlying cause "
                               "rather than the rate directly.",
        }]
    if hr_advisory_flag:
        return [{
            "severity": WARNING,
            "title": f"Persistently Elevated HR Despite Hemodynamic Control ({hr:.0f} bpm)",
            "consequence": "May indicate an unaddressed cause (pain, agitation, fever) "
                            "separate from the already-treated pressure target.",
            "recommendation": "Clinical review for pain/agitation/fever/volume status - "
                               "not addressable by further norepinephrine titration.",
        }]
    return []


# ===========================================================
# SEPTIC SHOCK (norepinephrine) - low MAP is the danger
# ===========================================================

def septic_alerts(latest):
    """Thresholds reuse expert_controller.TARGET_DEADBAND and the
    Surviving Sepsis Campaign's own 65 mmHg resuscitation target
    (already latest["TargetBP"], see constants.py/digital_twin.py) -
    any MAP genuinely below target is flagged CRITICAL, matching this
    drug's own controller logic (expert_controller.py: "there is no
    such thing as correcting hypotension too fast")."""
    alerts = []
    map_val = latest["MAP"]
    target = latest["TargetBP"]

    if map_val < target - TARGET_DEADBAND:
        alerts.append({
            "severity": CRITICAL,
            "title": f"Hypotensive Shock / Sub-Target Perfusion (MAP {map_val:.0f} mmHg)",
            "consequence": f"Prolonged mean arterial pressure below {target:.0f} mmHg is "
                            "strongly correlated with organ dysfunction and lactic acidosis "
                            "(Surviving Sepsis Campaign).",
            "recommendation": f"Titrate norepinephrine upward toward target {target:.0f} mmHg; "
                               "reassess volume status and consider source control per the "
                               "sepsis bundle.",
        })

    alerts.extend(_hr_alerts(latest["HR"], latest["HRAdvisory"]))
    alerts.extend(_spo2_alerts(latest["SpO2"]))
    return alerts


# ===========================================================
# HYPERTENSIVE EMERGENCY (sodium nitroprusside) - high MAP is the
# danger, but over-correcting BELOW the staged floor is an equally
# real, separately modeled danger (hypertensive_expert_controller.py's
# own module docstring).
# ===========================================================

def hypertensive_alerts(latest):
    alerts = []
    map_val = latest["MAP"]
    floor = latest["TargetBP"]  # the hypertensive engine stores the current staged floor here

    if map_val > EMERGENCY_MAP_MMHG_HIGH:
        alerts.append({
            "severity": CRITICAL,
            "title": f"Severe Hypertensive Emergency (MAP {map_val:.0f} mmHg)",
            "consequence": "Imminent threat of target-organ damage: intracranial "
                            "hemorrhage, aortic dissection, or acute pulmonary edema.",
            "recommendation": "Titrate sodium nitroprusside to reduce MAP by 20-25% over "
                               "the first hour per the staged reduction protocol; continuous "
                               "arterial line monitoring.",
        })
    elif map_val < floor - TARGET_DEADBAND - FLOOR_BREACH_MARGIN_MMHG:
        alerts.append({
            "severity": CRITICAL,
            "title": f"Over-Correction Below Safe Staged Floor (MAP {map_val:.0f}, "
                     f"floor {floor:.0f} mmHg)",
            "consequence": "Reducing MAP faster than the staged protocol allows risks "
                            "cerebral, coronary, or renal hypoperfusion by overwhelming "
                            "vascular autoregulation.",
            "recommendation": "Reduce or temporarily hold the nitroprusside infusion; "
                               "resume the staged reduction pace once MAP returns to the "
                               "current floor.",
        })

    if latest.get("CyanideBufferExceeded"):
        alerts.append({
            "severity": CRITICAL,
            "title": "Cyanide Toxicity - Cumulative Buffer Capacity Exceeded",
            "consequence": "Methemoglobin buffering capacity exceeded (FDA label: "
                            "~10 mg/kg cumulative) - risk of cyanide toxicity "
                            "(metabolic acidosis, altered mental status).",
            "recommendation": "Infusion stopped automatically; consider sodium "
                               "thiosulfate/hydroxocobalamin per cyanide toxicity "
                               "protocol and switch to an alternative antihypertensive.",
        })
    elif latest.get("CyanideRateRisk"):
        alerts.append({
            "severity": WARNING,
            "title": "Elevated Cyanide Generation Rate Risk",
            "consequence": "Cumulative dose approaching the FDA label's cyanide-"
                            "generation threshold.",
            "recommendation": "Infusion rate has been automatically capped; monitor "
                               "for early cyanide toxicity signs and consider a "
                               "co-infused antidote per protocol if the cap persists.",
        })

    alerts.extend(_hr_alerts(latest["HR"], latest["HRAdvisory"]))
    alerts.extend(_spo2_alerts(latest["SpO2"]))
    return alerts


def get_alerts(is_septic, latest):
    """Single entry point native_app.py calls each tick."""
    return septic_alerts(latest) if is_septic else hypertensive_alerts(latest)
