import os

from dotenv import load_dotenv

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
MODEL_NAME = "qwen3.5:9b"
TEMPERATURE = 0.0
EMBEDDING_NAME = "qwen3-embedding:8b"
