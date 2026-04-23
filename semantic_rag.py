from typing import List

from datasets import load_dataset
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

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
retriever = vector_store.as_retriever(search_kwargs={"k": 6})
llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)
prompt = ChatPromptTemplate.from_template(
    """
        You are a Question Answering assisant power by a RAG, you need
        to answer the multi-hop questions using the given Wikipedia context.
        You need to reson step by step and finally output the only short final
        answer(a name, date, country, yes/no, etc). If the context is insufficient,
        please output unkonwn.

        Context:{context}
        Question:{question}
        Answer  """
)
chain = prompt | llm | StrOutputParser()


class GraphState(TypedDict):
    question: str
    answer: str
    context: List[str]


def retrieve(state: GraphState):
    question = state["question"]
    documents = retriever.invoke(question)
    context = [page.page_content for page in documents]
    return {"context": context}


def generate(state: GraphState):
    question = state["question"]
    context = "\n".join(state["context"])
    response = chain.invoke({"context": context, "question": question})
    return {"answer": response}


workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()

if __name__ == "__main__":
    test_question = "Are director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?"
    result = app.invoke({"question": test_question})
    print(f"Question : {test_question}\n")
    print(f"Context : {result['context']}\n")
    print(f"Answer : {result['answer']}\n")
