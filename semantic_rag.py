from datasets import load_dataset
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EMBEDDING_NAME, MODEL_NAME, TEMPERATURE

dataset = load_dataset("framolfese/2WikiMultihopQA", split="train")
docs = []
for data in dataset:
    id = data["id"]
    titles = data["context"]["title"]
    sentences_list = data["context"]["sentences"]
    for title, sentence in zip(titles, sentences_list):
        content = " ".join(sentence)
        docs.append(
            Document(
                page_content=content,
                metadata={"title": title, "id": id},
            )
        )
embeddings = OllamaEmbeddings(model=EMBEDDING_NAME)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="2WikiMultihopQA",
    persist_directory="./dataset/chromadb",
)
