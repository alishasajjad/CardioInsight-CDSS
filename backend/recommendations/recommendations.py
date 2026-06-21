"""Clinical recommendation engine — rule-based next actions by risk tier."""
from __future__ import annotations

from backend.config import RISK_LOW_THRESHOLD, RISK_MEDIUM_THRESHOLD


def classify_risk(probability: float) -> str:
    if probability < RISK_LOW_THRESHOLD:
        return "low"
    if probability < RISK_MEDIUM_THRESHOLD:
        return "medium"
    return "high"


def _base_actions(risk_level: str) -> dict:
    if risk_level == "low":
        return {
            "tier": "Low Risk",
            "color": "#27ae60",
            "summary": "Preventive lifestyle focus; routine wellness monitoring.",
            "actions": [
                "Maintain heart-healthy diet (fruits, vegetables, whole grains, lean protein).",
                "Aim for 150 minutes/week moderate aerobic activity.",
                "Monitor blood pressure periodically at home or pharmacy.",
                "Avoid tobacco; limit alcohol per national guidelines.",
                "Schedule routine annual physical with primary care provider.",
            ],
            "clinical_follow_up": [
                "Repeat lipid profile every 4–6 years if prior results normal (per general guidelines).",
            ],
            "urgency": "Routine",
        }
    if risk_level == "medium":
        return {
            "tier": "Moderate Risk",
            "color": "#f39c12",
            "summary": "Enhanced monitoring and modifiable risk factor management recommended.",
            "actions": [
                "Track blood pressure and cholesterol more frequently (every 3–6 months).",
                "Increase physical activity with provider guidance if sedentary.",
                "Reduce sodium and saturated fat; consider DASH-style eating pattern.",
                "Manage stress, sleep, and weight (BMI/waist circumference).",
                "Discuss family history and symptoms with a healthcare provider.",
            ],
            "clinical_follow_up": [
                "Lipid profile within 3–6 months.",
                "Blood pressure evaluation and home monitoring log.",
                "Fasting glucose / HbA1c if metabolic risk factors present.",
                "Consider resting ECG if not recently performed.",
            ],
            "urgency": "Prompt outpatient follow-up (2–4 weeks)",
        }
    return {
        "tier": "High Risk",
        "color": "#c0392b",
        "summary": "Urgent clinical evaluation recommended; do not rely on this app alone.",
        "actions": [
            "Seek cardiologist or primary care consultation promptly.",
            "Obtain comprehensive cardiovascular evaluation.",
            "Strict adherence to prescribed medications if already under care.",
            "If experiencing chest pain, shortness of breath, or syncope — call emergency services.",
        ],
        "clinical_follow_up": [
            "Cardiologist consultation",
            "12-lead ECG",
            "Exercise stress test or pharmacologic stress imaging (provider decision)",
            "Lipid profile (fasting)",
            "Blood pressure evaluation (clinic + ambulatory if indicated)",
            "Echocardiogram if murmur, heart failure signs, or structural disease suspected",
            "Coronary risk stratification per ACC/AHA framework (clinical judgment)",
        ],
        "urgency": "Urgent — schedule within days; emergency if symptomatic",
    }


def generate_recommendations(
    ensemble_prob: float,
    inputs: dict,
    shap_contributors: list | None = None,
) -> dict:
    """Rule-based recommendations augmented with patient-specific factor notes."""
    risk_level = classify_risk(ensemble_prob)
    rec = _base_actions(risk_level)

    factor_notes = []
    if inputs.get("exercise_angina"):
        factor_notes.append("Exercise-induced angina reported — prioritize clinical stress testing discussion.")
    if inputs.get("chest_pain_type") == 3:
        factor_notes.append("Asymptomatic presentation pattern — still warrants clinical correlation.")
    if inputs.get("resting_bp", 0) >= 140:
        factor_notes.append("Elevated resting BP — hypertension management and repeat measurements.")
    if inputs.get("cholesterol", 0) >= 240:
        factor_notes.append("Elevated cholesterol — lipid management and dietary counseling.")
    if inputs.get("major_vessels", 0) >= 2:
        factor_notes.append("Multiple major vessels flagged — high priority for specialist review.")

    if shap_contributors:
        top = shap_contributors[:3]
        factor_notes.append(
            "Top model drivers: " + ", ".join(f"{t['feature']} ({t['shap']:+.3f})" for t in top)
        )

    rec["risk_level"] = risk_level
    rec["risk_probability"] = ensemble_prob
    rec["patient_specific_notes"] = factor_notes
    rec["disclaimer"] = (
        "These recommendations are educational decision-support suggestions only. "
        "They do not constitute medical diagnosis or treatment. Always follow guidance from "
        "qualified healthcare professionals."
    )
    return rec
