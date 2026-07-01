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


def main(patient_ids: list[str] = None):
    patient_ids = patient_ids if patient_ids is not None else PATIENT_IDS

    print("Loading patient data...")
    patients = load_patients(PATIENTS_FILE)

    if patient_ids:
        patients = [p for p in patients if p["patient_id"] in patient_ids]
        found = {p["patient_id"] for p in patients}
        missing = [pid for pid in patient_ids if pid not in found]
        if missing:
            print(f"Warning: no patient found for ID(s): {', '.join(missing)}")
        if not patients:
            print("No matching patients to process.")
            return

    print("Setting up LangChain pipeline...")
    llm = get_llm(backend="groq")
    prompt = build_prompt()
    chain = build_chain(llm, prompt)

    print(f"Processing {len(patients)} patients...")
    results = []

    try:
        for i, patient in enumerate(patients):
            patient_id = patient.get("patient_id", f"row_{i}")
            print(f"  -> {patient_id} ({i + 1}/{len(patients)})")

            if patient["insufficient_data"]:
                results.append({
                    "patient_id": patient_id,
                    "hemorrhage_risk": "INSUFFICIENT DATA",
                    "reasoning": "No critical fields (Hb, SysBP, DiaBP, Pulse, UrineProtein) recorded across any visit.",
                })
                continue

            assessment = process_patient(chain, patient)

            results.append({
                "patient_id": patient_id,
                "hemorrhage_risk": assessment["risk_level"],
                "reasoning": assessment["reasoning"],
            })
    except Exception as e:
        print(f"\nStopped at patient {i + 1}/{len(patients)}: {e}")
        if not results:
            raise
        print(f"Saving {len(results)} completed results...")

    os.makedirs("output", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Results saved to {OUTPUT_CSV} ({len(results)}/{len(patients)} patients)")

    print("\n--- Summary ---")
    for r in results:
        print(f"  {r['patient_id']}: {r['hemorrhage_risk']}")


if __name__ == "__main__":
    # IDs passed on the command line override PATIENT_IDS above.
    cli_ids = sys.argv[1:]
    main(patient_ids=cli_ids if cli_ids else None)
