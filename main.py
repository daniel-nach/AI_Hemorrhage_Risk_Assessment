import os
import sys
import pandas as pd

from config import PATIENTS_FILE, OUTPUT_CSV
from pipeline.loader import load_patients
from pipeline.prompt import build_prompt
from pipeline.llm import get_llm
from pipeline.chain import build_chain, process_patient


def main(patient_id_filter: str = None):
    print("Loading patient data...")
    patients = load_patients(PATIENTS_FILE)

    if patient_id_filter:
        patients = [p for p in patients if p["patient_id"] == patient_id_filter]
        if not patients:
            print(f"No patient found with ID '{patient_id_filter}'")
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
    filter_id = sys.argv[1] if len(sys.argv) > 1 else None
    main(patient_id_filter=filter_id)
