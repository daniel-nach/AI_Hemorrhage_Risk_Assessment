import pandas as pd

CLINICAL_COLUMNS = [
    "PatientNo", "DOB", "Sex", "VisitDate", "VisitType",
    "Temp", "SysBP", "DiaBP", "Pulse", "Resp", "O2Sat",
    "Height", "Weight", "Hb", "MAP", "Platelet",
    "FHR", "FundalHeight", "Presentation", "Dilatation",
    "Effacement", "Station", "Membrane", "AmnioticFluid",
    "UrineProtein", "UrineGlucose", "Urineketone",
    "BloodGlucoseLevel", "BedSideGlucose",
    "BloodLoss", "Edema", "GeneralCondition",
    "Urination", "MentalStatus", "Notes",
]

# If none of these are present across any visit, we cannot assess risk
CRITICAL_FIELDS = {"Hb", "SysBP", "DiaBP", "Pulse", "UrineProtein"}


def load_patients(file_path: str) -> list[dict]:
    """
    Reads the xlsx, groups by patient, and returns one dict per patient
    containing their full visit history sorted chronologically.
    """
    df = pd.read_excel(file_path)
    cols = [c for c in CLINICAL_COLUMNS if c in df.columns]
    df = df[cols]

    if "VisitDate" in df.columns:
        df["VisitDate"] = pd.to_datetime(df["VisitDate"], errors="coerce")
        df = df.sort_values("VisitDate")

    patients = []
    for patient_no, group in df.groupby("PatientNo"):
        summary = _build_visit_history(group)
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


def _build_visit_history(visits: pd.DataFrame) -> str:
    lines = []
    visit_cols = [c for c in visits.columns if c not in ("PatientNo", "VisitDate")]
    prev_date = None

    for i, (_, row) in enumerate(visits.iterrows(), start=1):
        date = row.get("VisitDate")
        date_str = date.strftime("%Y-%m-%d") if pd.notna(date) else None

        if date_str and prev_date and pd.notna(date) and pd.notna(prev_date):
            gap = (date - prev_date).days
            if gap == 0:
                header = f"Visit {i} ({date_str}, same day as previous visit):"
            else:
                header = f"Visit {i} ({date_str}, {gap} days after previous visit):"
        else:
            header = f"Visit {i} ({date_str or 'unknown date'}):"

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
