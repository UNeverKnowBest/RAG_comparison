from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.data_processing.config import (
    BASE_URL,
    CHROMA_DIR,
    EMBEDDING_NAME,
    MODEL_NAME,
    TEMPERATURE,
)

embeddings = OllamaEmbeddings(model=EMBEDDING_NAME, base_url=BASE_URL)
vector_store = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)

llm = ChatOllama(
    base_url=BASE_URL,
    model=MODEL_NAME,
    num_ctx=4096,
    temperature=TEMPERATURE,
    top_p=0.95,
)

prompt = ChatPromptTemplate.from_template(
    """You are a precise question answering system. Use the Wikipedia passages below to answer the question.

Rules:
Output only the final answer (a name, date, place, yes/no, number, etc.)
Do not include reasoning, explanations, or any other text
Keep the answer as short as possible.
If you do not know the answer, output exactly: unknown

Context:
{context}

Question: {question}

Answer:"""
)

chain = prompt | llm | StrOutputParser()


class GraphState(TypedDict):
    question: str
    answer: str
    context: List[Document]


def create_app(k: int = 20):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    def retrieve(state: GraphState):
        return {"context": retriever.invoke(state["question"])}

    def generate(state: GraphState):
        context_text = "\n\n".join(
            [f"[{d.metadata['title']}]: {d.page_content}" for d in state["context"]]
        )
        response = chain.invoke(
            {"context": context_text, "question": state["question"]}
        )
        return {"answer": response}

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


app = create_app()


if __name__ == "__main__":
    test_question = (
        "Who is Charles Bretagne Marie De La Trémoille's paternal grandfather?"
    )

    result = app.invoke({"question": test_question})
