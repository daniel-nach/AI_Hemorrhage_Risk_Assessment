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

def _hb(v):
    x = _f(v)
    if x is None:
        return None
    if x < 7:
        return "SEVERE - severe anemia"
    if x < 9:
        return "SEVERE - moderate anemia"
    if x < 10:
        return "BORDERLINE - mild anemia"
    if x < 11:
        return "normal, lower end (borderline-mild anemia)"
    return None


def _sysbp(v):
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
    s = str(v).strip().lower()
    if s in ("3+", "4+"):
        return "SEVERE - heavy proteinuria (preeclampsia marker)"
    if s == "2+":
        return "BORDERLINE - significant proteinuria"
    return None


def _temp(v):
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
