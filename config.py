import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "llama3-70b-8192"

PATIENTS_FILE = "data/sample_WONDER_Data.xlsx"
OUTPUT_CSV = "output/recommendations.csv"
