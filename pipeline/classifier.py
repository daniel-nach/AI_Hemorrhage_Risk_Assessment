RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def evaluate_numeric_risk(latest: dict) -> tuple[str, list[str]]:
    """
    Applies clinical thresholds to numeric fields.
    Returns (risk_level, list of finding strings).
    Python handles this so the LLM never has to parse numbers.
    """
    risk = "LOW"
    findings = []

    def escalate(to: str, reason: str):
        nonlocal risk
        findings.append(reason)
        if RISK_ORDER[to] > RISK_ORDER[risk]:
            risk = to

    hb = _num(latest.get("Hb"))
    if hb is not None:
        if hb < 7.0:
            escalate("HIGH", f"Hb {hb} < 7.0 (severe anemia)")
        elif hb < 10.0:
            escalate("MEDIUM", f"Hb {hb} in 7.0-9.9 (moderate anemia)")

    sysbp = _num(latest.get("SysBP"))
    diabp = _num(latest.get("DiaBP"))
    if sysbp is not None:
        if sysbp < 90:
            escalate("HIGH", f"SysBP {sysbp} < 90 (hypotension/possible shock)")
        elif sysbp >= 160:
            escalate("HIGH", f"SysBP {sysbp} >= 160 (severe hypertension)")
        elif sysbp >= 140:
            escalate("MEDIUM", f"SysBP {sysbp} in 140-159 (hypertension)")
    if diabp is not None:
        if diabp >= 110:
            escalate("HIGH", f"DiaBP {diabp} >= 110 (severe hypertension)")
        elif diabp >= 90:
            escalate("MEDIUM", f"DiaBP {diabp} in 90-109 (hypertension)")

    pulse = _num(latest.get("Pulse"))
    if pulse is not None:
        if pulse > 110:
            escalate("HIGH", f"Pulse {pulse} > 110 (tachycardia)")

    platelet = _num(latest.get("Platelet"))
    if platelet is not None:
        if platelet < 100_000:
            escalate("HIGH", f"Platelet {platelet} < 100,000")
        elif platelet < 150_000:
            escalate("MEDIUM", f"Platelet {platelet} in 100,000-150,000")

    up = str(latest.get("UrineProtein", "")).strip()
    if up and up.lower() not in ("nan", ""):
        if up in ("3+", "4+"):
            escalate("HIGH", f"UrineProtein {up} (severe proteinuria/preeclampsia)")
        elif up == "2+":
            escalate("MEDIUM", f"UrineProtein {up} (significant proteinuria)")

    bl = _num(latest.get("BloodLoss"))
    if bl is not None and bl > 0:
        escalate("MEDIUM", f"BloodLoss {bl} reported")

    return risk, findings


def _num(val):
    """Safely converts a value to float, returns None if not possible."""
    if val is None:
        return None
    try:
        f = float(val)
        import math
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None
