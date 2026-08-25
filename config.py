import os
from dotenv import load_dotenv

# Project root = the folder this config.py lives in, so paths work regardless of
# the current working directory you launch the script from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "openai/gpt-oss-20b"

PATIENTS_FILE = os.path.join(BASE_DIR, "data", "mock_data_new.xlsx")
OUTPUT_CSV = os.path.join(BASE_DIR, "output", "recommendations.csv")
