import re

QUALITATIVE_FIELDS = {
    "GeneralCondition", "MentalStatus", "Mood", "Wellbeing",
    "Bonding", "Edema", "Urination", "Notes",
}


def build_chain(llm, prompt):
    return prompt | llm


def process_patient(chain, patient_dict: dict, numeric_risk: str, numeric_findings: list[str]) -> dict:
    """
    Sends qualitative fields + Python numeric assessment to the LLM.
    LLM can only escalate beyond numeric_risk, never lower it.
    Returns a dict with 'risk_level' and 'reasoning'.
    """
    latest = patient_dict.get("latest", {})

    qual_lines = [
        f"{k}: {v}" for k, v in latest.items()
        if k in QUALITATIVE_FIELDS and v is not None
    ]
    qualitative_data = "\n".join(qual_lines) if qual_lines else "None recorded."
    findings_str = "\n".join(f"- {f}" for f in numeric_findings) if numeric_findings else "- No numeric risk factors found."

    result = chain.invoke({
        "numeric_risk": numeric_risk,
        "numeric_findings": findings_str,
        "qualitative_data": qualitative_data,
    })

    text = result.content if hasattr(result, "content") else str(result)
    risk_level = _extract(text, "RISK_LEVEL").upper().strip()
    reasoning = _extract(text, "REASONING")

    # Enforce that LLM cannot lower the risk
    from pipeline.classifier import RISK_ORDER
    if RISK_ORDER.get(risk_level, 0) < RISK_ORDER[numeric_risk]:
        risk_level = numeric_risk

    return {"risk_level": risk_level, "reasoning": reasoning}


def _extract(text: str, field: str) -> str:
    match = re.search(rf"{field}:\s*(.+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else "UNKNOWN"
