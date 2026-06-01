import pandas as pd


def load_patients(csv_path: str) -> list[dict]:
    """
    Reads the patient CSV and returns a list of dicts — one per patient.
    Columns depend on what Prof. Iacobelli provides; this works with any CSV.
    """
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


def load_guidelines(file_path: str) -> str:
    """
    Reads the guidelines file as a plain string.
    Supports .txt files. For PDF, see ARCHITECTURE.md — Future Enhancements.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
