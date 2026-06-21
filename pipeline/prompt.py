from langchain.prompts import PromptTemplate

TEMPLATE = """You are a clinical decision support assistant specializing in Parkinson's disease.

Using your medical knowledge, assess the hemorrhage risk for the following patient.

Patient data:
{patient_data}

Respond in this exact format:

RISK_LEVEL: <HIGH, MEDIUM, or LOW>
REASONING: <A concise explanation of the key factors driving this risk level, citing specific values from the patient data>"""


def build_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["patient_data"],
        template=TEMPLATE,
    )
