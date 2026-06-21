"""
Groq-powered health assistant — prediction-aware conversational AI.
"""
from __future__ import annotations

import os
from typing import Any

from backend.config import MEDICAL_DISCLAIMER
from backend.logging_config import get_logger

logger = get_logger(__name__)

# Supported Groq chat models
GROQ_MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B (best quality)",
    "llama-3.1-8b-instant": "Llama 3.1 8B (faster)",
}

DEFAULT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a supportive AI Health Assistant integrated into an educational heart disease \
risk prediction dashboard. You help users understand ML model outputs — you are NOT a doctor.

RULES (always follow):
1. State clearly that your guidance is educational only and NOT a substitute for professional medical advice, \
diagnosis, or treatment. Encourage users to consult qualified healthcare providers for any health decisions.
2. Base explanations on the PATIENT CONTEXT provided (model prediction, risk probability, SHAP factors). \
Do not invent lab values or diagnoses not in the context.
3. Explain predictions in plain language: what the risk % means, major contributing factors from SHAP, \
and general lifestyle recommendations (diet, exercise, stress, smoking cessation) when appropriate.
4. Be empathetic, concise, and structured. Use bullet points for clarity when listing factors or tips.
5. If asked about medications, specific treatments, or emergencies (chest pain, etc.), defer to emergency \
services or a physician — do not prescribe.
6. You may answer follow-up questions about the same patient's results using the context and chat history.
"""

DISCLAIMER = f"**Medical disclaimer:** {MEDICAL_DISCLAIMER}"


def get_api_key() -> str | None:
    """Resolve Groq API key: per-session (Streamlit) > environment > Streamlit secrets.

    Per-session is checked first so a key entered in the UI stays scoped to that
    browser session and is never written to the shared process environment.
    """
    try:
        import streamlit as st

        sess = st.session_state.get("groq_api_key")
        if sess and str(sess).strip():
            return str(sess).strip()
    except Exception:
        pass
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st

        if hasattr(st, "secrets") and st.secrets.get("GROQ_API_KEY"):
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    return None


def format_patient_context(ctx: dict[str, Any]) -> str:
    """Build structured context string for the LLM from prediction session."""
    lines = [
        "=== PATIENT ML PREDICTION CONTEXT ===",
        f"Model: {ctx.get('model_name', 'XGBoost')}",
        f"Prediction: {'Heart disease indicated' if ctx.get('prediction') == 1 else 'No heart disease indicated'}",
        f"Risk probability (disease class): {ctx.get('risk_probability', 0) * 100:.1f}%",
        f"Model confidence: {ctx.get('confidence', 0) * 100:.1f}%",
    ]
    ens = ctx.get("ensemble") or {}
    if ens.get("individual"):
        lines.append("Per-model probabilities:")
        for name, info in ens["individual"].items():
            lines.append(f"  - {name}: {info.get('probability', 0) * 100:.1f}% (weight={info.get('weight', 0):.2f})")
    if ens.get("ensemble_probability") is not None:
        lines.append(f"Ensemble probability: {ens['ensemble_probability'] * 100:.1f}%")
    rec = ctx.get("recommendations") or {}
    if rec.get("tier"):
        lines.append(f"Risk tier: {rec.get('tier')} — {rec.get('summary', '')}")
    lines.append("")
    lines.append("Clinical inputs:")
    for label, value in ctx.get("human_readable_inputs", {}).items():
        lines.append(f"  - {label}: {value}")

    shap_top = ctx.get("shap_contributors", [])
    if shap_top:
        lines.append("")
        lines.append("Top SHAP contributors (positive SHAP increases predicted risk):")
        for item in shap_top:
            direction = "increases risk" if item["shap"] > 0 else "decreases risk"
            lines.append(f"  - {item['feature']}: SHAP={item['shap']:.4f} ({direction})")

    lines.append("=== END CONTEXT ===")
    return "\n".join(lines)


def build_initial_assistant_message(ctx: dict[str, Any]) -> str:
    """Prompt the model to generate the first explanation after prediction."""
    return (
        f"{format_patient_context(ctx)}\n\n"
        "Using ONLY the context above, provide a patient-friendly response with these sections:\n"
        "1. **Prediction Summary** — explain the result and risk percentage in simple terms.\n"
        "2. **Key Risk Factors** — list the main contributing factors from SHAP/clinical values.\n"
        "3. **Lifestyle Recommendations** — 3–5 general, evidence-informed wellness tips (not prescriptions).\n"
        "4. **Reminder** — one sentence that this is educational, not medical advice.\n"
        "Keep the total response under 400 words."
    )


def chat_completion(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.4,
    max_tokens: int = 1024,
) -> str:
    """Call Groq chat API and return assistant reply text."""
    from groq import Groq

    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it in the Health Assistant settings, your "
            "environment, or .streamlit/secrets.toml."
        )

    client = Groq(api_key=api_key, timeout=30.0)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        logger.warning("Groq request failed: %s", exc)
        raise RuntimeError("The AI service is temporarily unavailable. Please try again.") from exc

    if not getattr(response, "choices", None):
        return ""
    return response.choices[0].message.content or ""


def generate_prediction_explanation(
    ctx: dict[str, Any],
    model: str = DEFAULT_MODEL,
) -> str:
    """First-turn explanation after ML prediction."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_initial_assistant_message(ctx)},
    ]
    return chat_completion(messages, model=model)


def chat_with_context(
    user_message: str,
    ctx: dict[str, Any],
    history: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
) -> str:
    """Follow-up chat turn with prediction context injected."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append(
        {
            "role": "user",
            "content": (
                f"{format_patient_context(ctx)}\n\n"
                "The above is the patient's current ML assessment. "
                "Answer the user's question using this context when relevant."
            ),
        },
    )
    messages.append(
        {
            "role": "assistant",
            "content": "I have reviewed the patient's prediction context and am ready to help with educational questions.",
        },
    )
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    return chat_completion(messages, model=model)
