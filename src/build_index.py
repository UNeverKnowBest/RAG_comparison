import json

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm

from src.data_processing.config import CHROMA_DIR, EMBEDDING_NAME, PASSAGES_DIR


def build_vector_store(file_path, persist_dir):
    docs = []

    with open(file_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Processing JSONL"):
            if line.strip():
                data = json.loads(line)
                doc = Document(
                    page_content=data["text"],
                    metadata={
                        "title": data["title"],
                        "passage_id": data["passage_id"],
                        "question_id": data.get("question_id", ""),
                        "is_supporting": data.get("is_supporting", False),
                    },
                )
                docs.append(doc)

    embeddings = OllamaEmbeddings(model=EMBEDDING_NAME)

    vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_dir)

    batch_size = 50
    for i in tqdm(range(0, len(docs), batch_size), desc="Indexing Documents"):
        vectorstore.add_documents(docs[i : i + batch_size])


if __name__ == "__main__":
    build_vector_store(PASSAGES_DIR, CHROMA_DIR)
