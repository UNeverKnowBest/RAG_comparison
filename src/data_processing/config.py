import os
from pathlib import Path

from dotenv import load_dotenv

SEED = 42
SAMPLE_SIZE = 1000
DATASET_NAME = "framolfese/2WikiMultihopQA"

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dataset"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
KG_DIR = DATA_DIR / "kg"
CORPUS_DIR = DATA_DIR / "corpus"
PASSAGES_DIR = CORPUS_DIR / "passages.jsonl"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
REPORT_DIR = RESULTS_DIR / "reports"
CHROMA_DIR = DATA_DIR / "chroma"

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = "qwen3.5:9b"
TEMPERATURE = 0
EMBEDDING_NAME = "qwen3-embedding:4b"
