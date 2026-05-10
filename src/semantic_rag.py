from typing import List

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

vector_store = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 10})

embeddings = OllamaEmbeddings(
    model=EMBEDDING_NAME,
)

llm = ChatOllama(
    base_url=BASE_URL,
    model=MODEL_NAME,
    num_ctx=8192,
    temperature=TEMPERATURE,
    top_p=0.95,
)

prompt = ChatPromptTemplate.from_template(
    """

        You are a Question Answering assisant power by a RAG, you need

        to answer the multi-hop questions using the given Wikipedia context.

        You need to reason step by step and finally output the only short final

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

    return {"context": documents}


def generate(state: GraphState):

    question = state["question"]

    context_text = "\n\n".join(
        [f"[{d.metadata['title']}]: {d.page_content}" for d in state["context"]]
    )

    response = chain.invoke({"context": context_text, "question": question})

    return {"answer": response}


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)

workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")

workflow.add_edge("retrieve", "generate")

workflow.add_edge("generate", END)

app = workflow.compile()


if __name__ == "__main__":
    test_question = (
        "Who is Charles Bretagne Marie De La Trémoille's paternal grandfather?"
    )

    result = app.invoke({"question": test_question})

    print(f"Question : {test_question}\n")

    print(f"Context : {result['context']}\n")

    print(f"Answer : {result['answer']}\n")
