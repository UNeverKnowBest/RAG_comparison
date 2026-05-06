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
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
REPORT_DIR = RESULTS_DIR / "reports"

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = "qwen3.5:9b"
TEMPERATURE = 0.0
EMBEDDING_NAME = "qwen3-embedding:8b"

NATIONALITY_NORMALIZE = {
    "american": "American",
    "america": "American",
    "u.s.": "American",
    "u.s.a": "American",
    "us": "American",
    "usa": "American",
    "united states": "American",
    "united states of america": "American",
    "british": "British",
    "uk": "British",
    "u.k.": "British",
    "united kingdom": "British",
    "england": "British",
    "english": "British",
    "indian": "Indian",
    "india": "Indian",
    "canadian": "Canadian",
    "canada": "Canadian",
    "chinese": "Chinese",
    "china": "Chinese",
    "french": "French",
    "france": "French",
    "german": "German",
    "germany": "German",
    "japanese": "Japanese",
    "japan": "Japanese",
    "nippon": "Japanese",
    "italian": "Italian",
    "italy": "Italian",
    "korean": "Korean",
    "south korea": "Korean",
    "south korean": "Korean",
    "australian": "Australian",
    "australia": "Australian",
    "spanish": "Spanish",
    "spain": "Spanish",
    "russian": "Russian",
    "russia": "Russian",
    "irish": "Irish",
    "ireland": "Irish",
    "swiss": "Swiss",
    "switzerland": "Swiss",
    "dutch": "Dutch",
    "netherlands": "Dutch",
    "austrian": "Austrian",
    "austria": "Austrian",
    "norwegian": "Norwegian",
    "norway": "Norwegian",
    "swedish": "Swedish",
    "sweden": "Swedish",
    "danish": "Danish",
    "denmark": "Danish",
    "iranian": "Iranian",
    "iran": "Iranian",
    "czech": "Czech",
    "czechia": "Czech",
    "pakistani": "Pakistani",
    "pakistan": "Pakistani",
}

NATIONALITY_RELATIONS = {
    "COUNTRY_OF_CITIZENSHIP",
    "COUNTRY",
    "COUNTRY_OF_ORIGIN",
}
