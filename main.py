import os
import pandas as pd

from config import PATIENTS_CSV, GUIDELINES_FILE, OUTPUT_CSV
from pipeline.loader import load_patients, load_guidelines
from pipeline.prompt import build_prompt
from pipeline.llm import get_llm
from pipeline.chain import build_chain, process_patient


def main():
    print("Loading data...")
    patients = load_patients(PATIENTS_CSV)
    guidelines = load_guidelines(GUIDELINES_FILE)

    print("Setting up LangChain pipeline...")
    llm = get_llm(backend="groq")  # change to "ollama" if running locally
    prompt = build_prompt()
    chain = build_chain(llm, prompt)

    print(f"Processing {len(patients)} patients...")
    results = []

    for i, patient in enumerate(patients):
        patient_id = patient.get("patient_id", f"row_{i}")
        print(f"  → Patient {patient_id} ({i + 1}/{len(patients)})")

        recommendation = process_patient(chain, patient, guidelines)

        results.append({
            "patient_id": patient_id,
            "recommendation": recommendation,
        })

    os.makedirs("output", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
