import pandas as pd

from pipeline.classifier import classify, severity, trend_direction, TREND_FIELDS, _f

# Pure identifiers / administrative fields with no clinical value — excluded.
# Everything else in the file (including Notes and postnatal fields) is passed to the AI.
EXCLUDE_COLUMNS = {
    "FacilityName", "PatientName", "CRID", "ANCNo", "PNCNo",
    "EncounterFacility", "VisitTime", "VitalDateTime",
}

# Patient-level fields handled separately (not repeated per visit).
PATIENT_LEVEL_COLUMNS = {"PatientNo", "DOB", "Sex"}

# If none of these are present across any visit, we cannot assess risk.
CRITICAL_FIELDS = {"Hb", "SysBP", "DiaBP", "Pulse", "UrineProtein"}

# Free-text / qualitative fields the code cannot classify. If any are populated,
# the patient is sent to the LLM rather than auto-classified.
QUALITATIVE_FIELDS = {
    "Notes", "GeneralCondition", "Mood", "Wellbeing", "Bonding",
    "MentalStatus", "Perineum", "UterineFundus", "Urination", "Edema",
}


def load_patients(file_path: str) -> list[dict]:
    """
    Reads the xlsx, groups by patient, and returns one dict per patient with:
      - a computed patient-level facts block (age, BMI, sex)
      - the full visit history including every recorded field and free-text notes
    """
    df = pd.read_excel(file_path)
    df = df[[c for c in df.columns if c not in EXCLUDE_COLUMNS]]

    if "VisitDate" in df.columns:
        df["VisitDate"] = pd.to_datetime(df["VisitDate"], errors="coerce")
        df = df.sort_values("VisitDate")
    if "DOB" in df.columns:
        df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")

    patients = []
    for patient_no, group in df.groupby("PatientNo"):
        facts = _computed_facts(group)
        history = _build_visit_history(group)
        trends = _build_trends(group)
        summary = f"{facts}\n\nVISIT HISTORY:\n{history}"
        if trends:
            summary += f"\n\n{trends}"

        has_data = any(
            col in group.columns and group[col].notna().any()
            for col in CRITICAL_FIELDS
        )
        auto_risk, auto_reason = (None, None) if not has_data else _triage(group)
        patients.append({
            "patient_id": patient_no,
            "summary": summary,
            "insufficient_data": not has_data,
            "auto_risk": auto_risk,      # 'LOW' / 'HIGH' decided in code, or None -> needs LLM
            "auto_reason": auto_reason,
        })

    return patients


def _triage(group: pd.DataFrame):
    """
    Decide high-confidence cases in code so we don't spend an LLM call on them.
    Returns ('LOW', reason), ('HIGH', reason), or (None, None) meaning 'ask the LLM'.
    """
    # Per-visit severe counts and overall severity presence.
    max_severe_in_a_visit = 0
    has_severe = has_borderline = has_edge = False

    for _, row in group.iterrows():
        severe_this_visit = 0
        for field in group.columns:
            sev = severity(field, row.get(field))
            if sev == "severe":
                severe_this_visit += 1
                has_severe = True
            elif sev == "borderline":
                has_borderline = True
            elif sev == "edge":
                has_edge = True
        max_severe_in_a_visit = max(max_severe_in_a_visit, severe_this_visit)

    # Auto-HIGH: two or more severe values at a single visit is unambiguous.
    if max_severe_in_a_visit >= 2:
        return "HIGH", "two or more severe values recorded at a single visit"

    # Anything with a severe or borderline value needs the LLM's judgment (single
    # severe = medium floor that may or may not escalate; borderline may be softened
    # by a trend or escalated by an aggravating factor).
    if has_severe or has_borderline:
        return None, None

    # Free-text/qualitative content -> the code can't judge it; defer to the LLM.
    if _has_qualitative(group):
        return None, None

    # A directional trend on a tracked stat is a judgment call -> defer to the LLM.
    if _has_directional_trend(group):
        return None, None

    # A near-threshold value combined with an aggravating factor can justify MEDIUM,
    # so that combination goes to the LLM. Edge value alone (no aggravating factor)
    # or aggravating factor alone (no edge value) stays low-confidence LOW.
    if has_edge and _aggravating_present(group):
        return None, None

    return "LOW", "all values within normal range; no risk factors or worrying trends"


def _has_qualitative(group: pd.DataFrame) -> bool:
    for col in QUALITATIVE_FIELDS:
        if col in group.columns and group[col].notna().any():
            # treat non-empty strings as content
            for v in group[col].dropna():
                if str(v).strip() not in ("", "nan"):
                    return True
    return False


def _has_directional_trend(group: pd.DataFrame) -> bool:
    for col in TREND_FIELDS:
        if col not in group.columns:
            continue
        nums = [n for n in (_f(v) for v in group[col].tolist()) if n is not None]
        if len(nums) >= 2 and trend_direction(nums) in ("rising", "falling"):
            return True
    return False


def _aggravating_present(group: pd.DataFrame) -> bool:
    age = _compute_age(group)
    if age is not None and (age >= 40 or age <= 17):
        return True

    bmi = _bmi_value(group)
    if bmi is not None and (bmi >= 30 or bmi < 18.5):
        return True

    for col in ("BloodGlucoseLevel", "BedSideGlucose"):
        val = _f(_latest(group, col))
        if val is not None and val > 140:  # elevated random glucose
            return True

    ug = _latest(group, "UrineGlucose")
    if ug is not None and str(ug).strip() not in ("", "nan", "0", "0.0", "Negative", "negative"):
        return True

    return False


def _latest(group: pd.DataFrame, col: str):
    """Most recent non-null value of a column across a patient's visits."""
    if col in group.columns:
        series = group[col].dropna()
        if not series.empty:
            return series.iloc[-1]
    return None


def _computed_facts(group: pd.DataFrame) -> str:
    """Values the system computes for the AI so it doesn't have to do arithmetic."""
    lines = ["PATIENT-LEVEL FACTS (computed by the system):"]

    sex = _latest(group, "Sex")
    if sex is not None:
        lines.append(f"  Sex: {sex}")

    age = _compute_age(group)
    if age is not None:
        lines.append(f"  Age: {age} years")

    bmi_line = _compute_bmi(group)
    if bmi_line:
        lines.append(bmi_line)

    if len(lines) == 1:
        lines.append("  (none could be computed)")
    return "\n".join(lines)


def _compute_age(group: pd.DataFrame):
    dob = _latest(group, "DOB")
    if dob is None or pd.isna(dob):
        return None
    # Age at the most recent visit, or today if no visit date.
    ref = _latest(group, "VisitDate")
    if ref is None or pd.isna(ref):
        ref = pd.Timestamp.now()
    years = int((ref - dob).days // 365.25)
    return years if 0 < years < 120 else None


def _bmi_value(group: pd.DataFrame):
    """BMI as a float, or None if height/weight unavailable."""
    h_cm = _f(_latest(group, "Height"))
    w_kg = _f(_latest(group, "Weight"))
    if h_cm is None or w_kg is None or h_cm <= 0 or w_kg <= 0:
        return None
    return w_kg / ((h_cm / 100) ** 2)


def _compute_bmi(group: pd.DataFrame):
    bmi = _bmi_value(group)
    if bmi is None:
        return None
    h_cm = _f(_latest(group, "Height"))
    w_kg = _f(_latest(group, "Weight"))
    return f"  BMI: {bmi:.1f} (Height {h_cm:g} cm, Weight {w_kg:g} kg) -> {_bmi_category(bmi)}"


def _bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal weight"
    if bmi < 30:
        return "overweight"
    return "obese"


def _build_visit_history(visits: pd.DataFrame) -> str:
    lines = []
    visit_cols = [
        c for c in visits.columns
        if c not in PATIENT_LEVEL_COLUMNS and c != "VisitDate"
    ]
    prev_date = None

    for i, (_, row) in enumerate(visits.iterrows(), start=1):
        date = row.get("VisitDate")

        if prev_date is not None and pd.notna(date) and pd.notna(prev_date):
            gap = (date - prev_date).days
            if gap == 0:
                header = f"Visit {i} (same day as previous visit):"
            else:
                header = f"Visit {i} ({gap} days after previous visit):"
        else:
            header = f"Visit {i}:"

        lines.append(header)

        any_data = False
        for col in visit_cols:
            val = row.get(col)
            if val is not None and pd.notna(val) and str(val).strip() not in ("", "nan"):
                label = classify(col, val)
                if label:
                    lines.append(f"  {col}: {val} [{label}]")
                else:
                    lines.append(f"  {col}: {val}")
                any_data = True

        if not any_data:
            lines.append("  (no data recorded)")

        if pd.notna(date):
            prev_date = date

    return "\n".join(lines)


def _build_trends(visits: pd.DataFrame) -> str:
    """Consolidated chronological sequence + direction per numeric stat with >= 2 readings."""
    lines = []
    for col in TREND_FIELDS:
        if col not in visits.columns:
            continue
        nums = [_f(v) for v in visits[col].tolist()]
        nums = [n for n in nums if n is not None]
        if len(nums) < 2:
            continue
        seq = " -> ".join(f"{n:g}" for n in nums)
        lines.append(f"  {col}: {seq} ({trend_direction(nums)})")

    if not lines:
        return ""
    return "TRENDS (values across visits, earliest to latest):\n" + "\n".join(lines)
