import pandas as pd
import numpy as np

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

# Stats where trend direction matters clinically
TREND_COLUMNS = ["Hb", "SysBP", "DiaBP", "Pulse", "O2Sat", "Platelet", "Weight", "FundalHeight"]

# For these stats, a rising value is clinically worse
HIGHER_IS_WORSE = {"SysBP", "DiaBP", "Pulse"}
# For these stats, a falling value is clinically worse
LOWER_IS_WORSE  = {"Hb", "O2Sat", "Platelet"}


def load_patients(file_path: str) -> list[dict]:
    """
    Reads the xlsx, groups by patient, and returns one aggregated dict per patient
    that includes all visit data and trend information for key stats.
    """
    df = pd.read_excel(file_path)
    cols = [c for c in CLINICAL_COLUMNS if c in df.columns]
    df = df[cols]

    if "VisitDate" in df.columns:
        df["VisitDate"] = pd.to_datetime(df["VisitDate"], errors="coerce")

    patients = []
    for patient_no, group in df.groupby("PatientNo"):
        group = group.sort_values("VisitDate") if "VisitDate" in group.columns else group
        summary = _build_patient_summary(patient_no, group)
        patients.append({"patient_id": patient_no, "summary": summary})

    return patients


def _build_patient_summary(patient_no: str, visits: pd.DataFrame) -> str:
    lines = []
    n = len(visits)

    # Visit timeline
    if "VisitDate" in visits.columns:
        dates = visits["VisitDate"].dropna()
        if len(dates) >= 2:
            span = (dates.iloc[-1] - dates.iloc[0]).days
            date_strs = [d.strftime("%Y-%m-%d") for d in dates]
            lines.append(f"Visits: {n} visit(s) on {', '.join(date_strs)} (span: {span} days)")
        elif len(dates) == 1:
            lines.append(f"Visits: 1 visit on {dates.iloc[0].strftime('%Y-%m-%d')}")
        else:
            lines.append(f"Visits: {n} visit(s), dates unknown")
    else:
        lines.append(f"Visits: {n} visit(s)")

    # Static fields (take first non-null value)
    for col in ["DOB", "Sex", "VisitType"]:
        if col in visits.columns:
            val = visits[col].dropna().iloc[0] if not visits[col].dropna().empty else None
            if val is not None:
                lines.append(f"{col}: {val}")

    # Trend fields — show each recorded value with its date and flag direction
    for col in TREND_COLUMNS:
        if col not in visits.columns:
            continue
        records = _dated_values(visits, col)
        if not records:
            continue
        if len(records) == 1:
            lines.append(f"{col}: {records[0][1]} (on {records[0][0]})")
        else:
            values = [v for _, v in records]
            trend = _trend_label(values, col)
            entries = ", ".join(f"{v} ({d})" for d, v in records)
            lines.append(f"{col}: {entries} -> {trend}")

    # Other clinical fields — list all non-null values
    other_cols = [c for c in visits.columns
                  if c not in TREND_COLUMNS + ["PatientNo", "DOB", "Sex", "VisitDate", "VisitType"]]
    for col in other_cols:
        records = _dated_values(visits, col)
        if not records:
            continue
        if len(records) == 1:
            lines.append(f"{col}: {records[0][1]} (on {records[0][0]})")
        else:
            entries = ", ".join(f"{v} ({d})" for d, v in records)
            lines.append(f"{col}: {entries}")

    return "\n".join(lines)


def _dated_values(visits: pd.DataFrame, col: str) -> list[tuple]:
    """Returns list of (date_str, value) for non-null entries in a column."""
    result = []
    for _, row in visits.iterrows():
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        date = row.get("VisitDate")
        date_str = date.strftime("%Y-%m-%d") if pd.notna(date) else "unknown date"
        result.append((date_str, val))
    return result


def _trend_label(values: list, col: str = "") -> str:
    try:
        nums = [float(v) for v in values]
    except (ValueError, TypeError):
        return "changed"
    delta = nums[-1] - nums[0]
    pct = abs(delta) / nums[0] * 100 if nums[0] != 0 else 0
    if pct < 5:
        return "stable"
    if col in HIGHER_IS_WORSE:
        return "WORSENING" if delta > 0 else "improving"
    if col in LOWER_IS_WORSE:
        return "WORSENING" if delta < 0 else "improving"
    # For unlabeled stats just describe direction
    return "increasing" if delta > 0 else "decreasing"
