import os
import sys
import pandas as pd

from config import PATIENTS_FILE, OUTPUT_CSV
from pipeline.loader import load_patients
from pipeline.prompt import build_prompt
from pipeline.llm import get_llm
from pipeline.chain import build_chain, process_patient


# Leave empty to run all patients, or list specific IDs to run only those.
PATIENT_IDS = []


def main(patients_file: str = None, patient_ids: list[str] = None, limit: int = None):
    patients_file = patients_file or PATIENTS_FILE
    patient_ids = patient_ids if patient_ids is not None else PATIENT_IDS

    print(f"Loading patient data from {patients_file} ...")
    patients = load_patients(patients_file)

    if patient_ids:
        patients = [p for p in patients if p["patient_id"] in patient_ids]
        found = {p["patient_id"] for p in patients}
        missing = [pid for pid in patient_ids if pid not in found]
        if missing:
            print(f"Warning: no patient found for ID(s): {', '.join(missing)}")
        if not patients:
            print("No matching patients to process.")
            return

    if limit:
        patients = patients[:limit]

    # How many patients does the code resolve on its own vs. need the LLM?
    auto = [p for p in patients if p["insufficient_data"] or p["auto_risk"]]
    llm_needed = [p for p in patients if not p["insufficient_data"] and not p["auto_risk"]]
    print(f"{len(patients)} patients: {len(auto)} decided in code, {len(llm_needed)} need the LLM.")

    chain = None  # built lazily, only if an LLM call is actually required
    results = []

    try:
        for i, patient in enumerate(patients):
            patient_id = patient.get("patient_id", f"row_{i}")

            if patient["insufficient_data"]:
                results.append(_row(patient_id, "INSUFFICIENT DATA",
                                    "No critical fields (Hb, SysBP, DiaBP, Pulse, UrineProtein) "
                                    "recorded across any visit."))
                continue

            if patient["auto_risk"]:
                results.append(_row(patient_id, patient["auto_risk"],
                                    f"Decided in code: {patient['auto_reason']}."))
                continue

            if chain is None:
                print("Setting up LangChain pipeline...")
                chain = build_chain(get_llm(backend="groq"), build_prompt())

            print(f"  -> LLM: {patient_id} ({i + 1}/{len(patients)})")
            assessment = process_patient(chain, patient)
            results.append(_row(patient_id, assessment["risk_level"], assessment["reasoning"]))
    except Exception as e:
        print(f"\nStopped at patient {i + 1}/{len(patients)}: {e}")
        if not results:
            raise
        print(f"Saving {len(results)} completed results...")

    os.makedirs("output", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Results saved to {OUTPUT_CSV} ({len(results)}/{len(patients)} patients)")

    counts = df["hemorrhage_risk"].value_counts().to_dict()
    print("\n--- Risk breakdown ---")
    for level, n in counts.items():
        print(f"  {level}: {n}")


def _row(patient_id, risk, reasoning):
    return {"patient_id": patient_id, "hemorrhage_risk": risk, "reasoning": reasoning}


if __name__ == "__main__":
    # CLI args (any order): a .xlsx/.csv path sets the data file, a bare integer
    # caps how many patients to process (handy for testing), anything else is a
    # patient ID filter. Examples:
    #   python main.py
    #   python main.py data/new_data.xlsx 50
    #   python main.py KCK0033218 KCK0033295
    file_arg, limit_arg, ids = None, None, []
    for arg in sys.argv[1:]:
        if arg.lower().endswith((".xlsx", ".csv")):
            file_arg = arg
        elif arg.isdigit():
            limit_arg = int(arg)
        else:
            ids.append(arg)
    main(patients_file=file_arg, patient_ids=ids or None, limit=limit_arg)
