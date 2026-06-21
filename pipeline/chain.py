import re


def build_chain(llm, prompt):
    return prompt | llm


def process_patient(chain, patient_dict: dict) -> dict:
    """
    Runs the chain for a single patient.
    patient_dict must have 'patient_id' and 'summary' keys (produced by loader).
    Returns a dict with 'risk_level' and 'reasoning'.
    """
    result = chain.invoke({"patient_data": patient_dict["summary"]})
    text = result.content if hasattr(result, "content") else str(result)

    risk_level = _extract(text, "RISK_LEVEL").upper()
    # Normalize multi-word value
    if "INSUFFICIENT" in risk_level:
        risk_level = "INSUFFICIENT DATA"
    reasoning = _extract(text, "REASONING")

    return {"risk_level": risk_level, "reasoning": reasoning}


def _extract(text: str, field: str) -> str:
    match = re.search(rf"{field}:\s*(.+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else "UNKNOWN"
