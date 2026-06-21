from langchain.prompts import PromptTemplate

TEMPLATE = """You are a clinical decision support assistant specializing in maternal health and obstetric care.

Using your medical knowledge, assess the hemorrhage risk for the following obstetric patient visit.

Patient data:
{patient_data}

Based on factors such as hemoglobin (Hb), platelet count, blood pressure, blood loss, pulse, \
oxygen saturation, edema, urine protein, and general condition, assess the risk of obstetric hemorrhage.

Respond in this exact format:

RISK_LEVEL: <HIGH, MEDIUM, or LOW>
REASONING: <A concise explanation citing the specific values from the patient data that drive this risk level>"""


def build_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["patient_data"],
        template=TEMPLATE,
    )
