import pandas as pd

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
        summary = f"{facts}\n\nVISIT HISTORY:\n{history}"

        has_data = any(
            col in group.columns and group[col].notna().any()
            for col in CRITICAL_FIELDS
        )
        patients.append({
            "patient_id": patient_no,
            "summary": summary,
            "insufficient_data": not has_data,
        })

    return patients


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


def _compute_bmi(group: pd.DataFrame):
    height = _latest(group, "Height")  # cm
    weight = _latest(group, "Weight")  # kg
    try:
        h_cm = float(height)
        w_kg = float(weight)
    except (TypeError, ValueError):
        return None
    if h_cm <= 0 or w_kg <= 0:
        return None
    bmi = w_kg / ((h_cm / 100) ** 2)
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
                lines.append(f"  {col}: {val}")
                any_data = True

        if not any_data:
            lines.append("  (no data recorded)")

        if pd.notna(date):
            prev_date = date

    return "\n".join(lines)
