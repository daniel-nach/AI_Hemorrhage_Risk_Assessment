import pandas as pd

# Columns relevant to hemorrhage risk — skip PII and administrative fields
CLINICAL_COLUMNS = [
    "PatientNo", "DOB", "Sex", "VisitType",
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
    Reads the professor's xlsx file and returns a list of dicts — one per patient visit.
    Only passes clinical columns to the LLM; strips names and facility identifiers.
    """
    df = pd.read_excel(file_path)

    # Keep only columns that exist in this file
    cols = [c for c in CLINICAL_COLUMNS if c in df.columns]
    df = df[cols]

    # Drop rows with no clinical data at all
    clinical_data_cols = [c for c in cols if c != "PatientNo"]
    df = df.dropna(how="all", subset=clinical_data_cols)

    return df.to_dict(orient="records")
