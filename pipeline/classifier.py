"""
Deterministic classification of clinical values and trends.

Thresholds live HERE (not in the prompt) so the code does the mechanical
range-checking and the LLM only does clinical judgment. Each value is labelled
as normal / BORDERLINE / SEVERE with a short description; the raw number is
always kept alongside the label so the LLM can still reason about combinations
(e.g. obesity + an upper-normal BP).

To tune thresholds later, edit only this file.
"""


def _f(value):
    """Safely convert to float, or None."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x != x:  # NaN
        return None
    return x


# --- per-field classifiers: return an annotation string, or None for plain-normal ---
# Threshold sources are noted per function; full references in CITATIONS.md.
# NOTE: the "normal, upper/lower end" edge bands are a project design choice (used
# by the aggravating-factor logic), NOT clinical categories from any guideline.

def anemia_severity(hb):
    """
    WHO anaemia-in-pregnancy severity from hemoglobin (g/dL). SINGLE SOURCE OF
    TRUTH for anemia banding -- both this classifier and the Markov model
    (markov.current_anemia_state) call this, so they can never disagree.
    source: WHO -- anemia <11; mild 10.0-10.9, moderate 7.0-9.9, severe <7.0.
    Returns 'severe' / 'moderate' / 'mild' / None (None = normal or no value).
    """
    x = _f(hb)
    if x is None:
        return None
    if x < 7:
        return "severe"
    if x < 10:      # WHO moderate: 7.0-9.9
        return "moderate"
    if x < 11:      # WHO mild: 10.0-10.9
        return "mild"
    return None     # >= 11: normal for pregnancy


def _hb(v):
    # WHO bands via anemia_severity(). moderate/severe -> SEVERE tier (real
    # hemorrhage risk factor); mild -> BORDERLINE; 11.0-11.5 flagged as a near-
    # threshold edge for the aggravating-factor logic only.
    sev = anemia_severity(v)
    if sev == "severe":
        return "SEVERE - severe anemia"
    if sev == "moderate":
        return "SEVERE - moderate anemia"
    if sev == "mild":
        return "BORDERLINE - mild anemia"
    x = _f(v)
    if x is not None and x < 11.5:
        return "normal, lower end"
    return None


def _sysbp(v):
    # source: ACOG Practice Bulletin 222 -- hypertension >=140, severe >=160.
    # (SysBP <90 hypotension and the 130-139 "upper end" edge are project choices.)
    x = _f(v)
    if x is None:
        return None
    if x < 90:
        return "SEVERE - hypotension / possible shock"
    if x >= 160:
        return "SEVERE - severe hypertension"
    if x >= 140:
        return "BORDERLINE - hypertension"
    if x >= 130:
        return "normal, upper end"
    if x <= 95:
        return "normal, lower end"
    return None


def _diabp(v):
    # source: ACOG Practice Bulletin 222 -- hypertension >=90, severe >=110.
    x = _f(v)
    if x is None:
        return None
    if x >= 110:
        return "SEVERE - severe hypertension"
    if x >= 90:
        return "BORDERLINE - hypertension"
    if x >= 85:
        return "normal, upper end"
    if x < 50:
        return "BORDERLINE - low diastolic"
    if x < 60:
        return "normal, lower end"
    return None


def _pulse(v):
    # source: standard clinical vitals (tachycardia >100-110; pregnancy resting HR
    # runs higher). Not from a single guideline; see CITATIONS.md.
    x = _f(v)
    if x is None:
        return None
    if x > 110:
        return "SEVERE - tachycardia"
    if x >= 101:
        return "BORDERLINE - mild tachycardia"
    if x >= 95:
        return "normal, upper end"
    if x < 50:
        return "SEVERE - bradycardia"
    if x < 60:
        return "normal, lower end"
    return None


def _o2(v):
    # source: standard clinical vitals (SpO2 >=95 normal, <92 concerning).
    x = _f(v)
    if x is None:
        return None
    if x < 92:
        return "SEVERE - hypoxemia"
    if x < 95:
        return "BORDERLINE - low oxygen saturation"
    if x < 97:
        return "normal, lower end"
    return None


def _platelet(v):
    # source: standard thresholds -- thrombocytopenia <150k, severe <100k (also a
    # HELLP-syndrome criterion).
    x = _f(v)
    if x is None:
        return None
    if x < 1000:  # value given in x10^3/uL (e.g. 150 == 150,000)
        x *= 1000
    if x < 100000:
        return "SEVERE - thrombocytopenia"
    if x < 150000:
        return "BORDERLINE - low platelets"
    return None


def _urine_protein(v):
    # source: ACOG Practice Bulletin 222 -- dipstick 2+ proteinuria threshold;
    # 3+/4+ heavy proteinuria (preeclampsia marker).
    s = str(v).strip().lower()
    if s in ("3+", "4+"):
        return "SEVERE - heavy proteinuria (preeclampsia marker)"
    if s == "2+":
        return "BORDERLINE - significant proteinuria"
    return None


def _temp(v):
    # source: standard clinical vitals -- fever >=100.4 F (38 C).
    x = _f(v)
    if x is None:
        return None
    if x >= 102:
        return "SEVERE - high fever (infection risk)"
    if x >= 100.4:
        return "BORDERLINE - fever (infection risk)"
    if x < 96:
        return "BORDERLINE - hypothermia"
    if x >= 99.6:
        return "normal, upper end"
    return None


def _resp(v):
    # source: standard clinical vitals -- normal respiratory rate 12-20/min.
    x = _f(v)
    if x is None:
        return None
    if x > 24:
        return "SEVERE - tachypnea"
    if x >= 21:
        return "BORDERLINE - mildly elevated respiratory rate"
    if x < 10:
        return "SEVERE - bradypnea"
    return None


def _blood_loss(v):
    # source: PPH is defined as >=500 mL (vaginal) / >=1000 mL blood loss
    # (ACOG / CMQCC). Threshold here approximate; see CITATIONS.md.
    x = _f(v)
    if x is None or x <= 0:
        return None
    if x > 500:
        return "SEVERE - significant blood loss"
    return "BORDERLINE - blood loss reported"


_CLASSIFIERS = {
    "Hb": _hb,
    "SysBP": _sysbp,
    "DiaBP": _diabp,
    "Pulse": _pulse,
    "O2Sat": _o2,
    "Platelet": _platelet,
    "UrineProtein": _urine_protein,
    "Temp": _temp,
    "Resp": _resp,
    "BloodLoss": _blood_loss,
}


def classify(field: str, value) -> str | None:
    """Return a short label for a value, or None if it's plainly normal / unclassified."""
    fn = _CLASSIFIERS.get(field)
    return fn(value) if fn else None


def severity(field: str, value) -> str | None:
    """
    Coarse severity tier for triage: 'severe', 'borderline', 'edge' (normal but
    near a threshold), or None (plainly normal / unclassified).
    """
    label = classify(field, value)
    if not label:
        return None
    if label.startswith("SEVERE"):
        return "severe"
    if label.startswith("BORDERLINE"):
        return "borderline"
    if label.startswith("normal"):  # "normal, upper/lower end"
        return "edge"
    return None


# --- trend computation across visits ---

# Numeric stats worth tracking over time.
TREND_FIELDS = [
    "SysBP", "DiaBP", "Pulse", "O2Sat", "Hb",
    "Platelet", "Weight", "FundalHeight", "Temp", "Resp",
]


def trend_direction(values: list[float]) -> str:
    """Classify a chronological sequence of numbers as a direction word."""
    if len(values) < 2:
        return "single reading"
    change = values[-1] - values[0]
    rel = abs(change) / abs(values[0]) if values[0] else 0
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    ups = sum(1 for d in diffs if d > 0)
    downs = sum(1 for d in diffs if d < 0)
    if rel < 0.05:
        return "stable"
    if ups and downs:
        return "fluctuating"
    return "rising" if change > 0 else "falling"
