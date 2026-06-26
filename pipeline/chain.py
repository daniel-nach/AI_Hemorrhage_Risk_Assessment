import re


def build_chain(llm, prompt):
    return prompt | llm


def process_patient(chain, patient_dict: dict) -> dict:
    """
    Sends the full visit history to the LLM and returns risk_level + reasoning.
    """
    result = chain.invoke({"patient_data": patient_dict["summary"]})
    text = result.content if hasattr(result, "content") else str(result)

    risk_level = _extract(text, "RISK_LEVEL").upper().strip()
    reasoning = _extract(text, "REASONING")

    return {"risk_level": risk_level, "reasoning": reasoning}


def _extract(text: str, field: str) -> str:
    match = re.search(rf"{field}:\s*(.+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else "UNKNOWN"
