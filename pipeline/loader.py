import pandas as pd

# If none of these are present, we can't assess risk
CRITICAL_FIELDS = {"Hb", "SysBP", "DiaBP", "Pulse", "UrineProtein"}

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


def load_patients(file_path: str) -> list[dict]:
    """
    Reads the xlsx, groups by patient, and returns one dict per patient
    using the latest non-null value for each clinical field.
    """
    df = pd.read_excel(file_path)
    cols = [c for c in CLINICAL_COLUMNS if c in df.columns]
    df = df[cols]

    if "VisitDate" in df.columns:
        df["VisitDate"] = pd.to_datetime(df["VisitDate"], errors="coerce")
        df = df.sort_values("VisitDate")

    patients = []
    for patient_no, group in df.groupby("PatientNo"):
        # For each column, take the last non-null value across all visits
        latest = {}
        for col in group.columns:
            if col == "PatientNo":
                continue
            series = group[col].dropna()
            if not series.empty:
                latest[col] = series.iloc[-1]

        insufficient = not any(k in latest for k in CRITICAL_FIELDS)
        patients.append({
            "patient_id": patient_no,
            "latest": latest,
            "insufficient_data": insufficient,
        })

    return patients
