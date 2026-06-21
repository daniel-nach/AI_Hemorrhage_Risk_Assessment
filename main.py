import os
import pandas as pd

from config import PATIENTS_FILE, OUTPUT_CSV
from pipeline.loader import load_patients
from pipeline.prompt import build_prompt
from pipeline.llm import get_llm
from pipeline.chain import build_chain, process_patient


def main():
    print("Loading patient data...")
    patients = load_patients(PATIENTS_FILE)

    print("Setting up LangChain pipeline...")
    llm = get_llm(backend="groq")
    prompt = build_prompt()
    chain = build_chain(llm, prompt)

    print(f"Processing {len(patients)} patient visits...")
    results = []

    for i, patient in enumerate(patients):
        patient_id = patient.get("PatientNo", f"row_{i}")
        print(f"  -> {patient_id} ({i + 1}/{len(patients)})")

        assessment = process_patient(chain, patient)

        results.append({
            "patient_id": patient_id,
            "hemorrhage_risk": assessment["risk_level"],
            "reasoning": assessment["reasoning"],
        })

    os.makedirs("output", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Results saved to {OUTPUT_CSV}")

    print("\n--- Summary ---")
    for r in results:
        print(f"  {r['patient_id']}: {r['hemorrhage_risk']}")


if __name__ == "__main__":
    main()
