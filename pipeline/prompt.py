from langchain_core.prompts import PromptTemplate

TEMPLATE = """You are a clinical decision support assistant specializing in maternal health and obstetric care.

Assess the obstetric hemorrhage risk for the following patient using the thresholds below.
Apply each threshold exactly as written. Do not use general reasoning to override a threshold.

--- CLINICAL THRESHOLDS ---
Hemoglobin (Hb):
  - Hb < 7.0 g/dL          -> HIGH risk factor
  - 7.0 <= Hb < 10.0        -> MEDIUM risk factor
  - Hb >= 10.0              -> no risk factor

Blood Pressure:
  - SysBP >= 160 OR DiaBP >= 110  -> HIGH risk factor
  - SysBP 140-159 OR DiaBP 90-109 -> MEDIUM risk factor
  - SysBP < 90                    -> HIGH risk factor (possible shock)
  - Otherwise                     -> no risk factor

Urine Protein:
  - 3+ or 4+  -> HIGH risk factor
  - 2+        -> MEDIUM risk factor
  - Otherwise -> no risk factor

Platelet count:
  - < 100,000       -> HIGH risk factor
  - 100,000-150,000 -> MEDIUM risk factor
  - Otherwise       -> no risk factor

Pulse:
  - > 110 bpm -> HIGH risk factor
  - Otherwise -> no risk factor

Blood Loss:
  - Any reported value -> MEDIUM risk factor minimum

--- RISK LEVEL RULES ---
- HIGH:             at least one HIGH risk factor is present
- MEDIUM:           no HIGH risk factors, but at least one MEDIUM risk factor
- LOW:              all available values are within normal ranges
- INSUFFICIENT DATA: Hb, SysBP, DiaBP, Pulse, and UrineProtein are ALL missing

--- PATIENT DATA (most recent recorded value per field) ---
{patient_data}

Respond in this exact format:

RISK_LEVEL: <HIGH, MEDIUM, LOW, or INSUFFICIENT DATA>
REASONING: <List which specific values triggered which risk factors. Be brief and cite the numbers.>"""


def build_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["patient_data"],
        template=TEMPLATE,
    )
