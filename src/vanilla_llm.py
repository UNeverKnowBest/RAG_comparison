from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.data_processing.config import BASE_URL, MODEL_NAME, TEMPERATURE

llm = ChatOllama(
    base_url=BASE_URL,
    model=MODEL_NAME,
    num_ctx=4096,
    temperature=TEMPERATURE,
    top_p=0.95,
)

prompt = ChatPromptTemplate.from_template(
    """You are a precise question answering system. Answer the question using only your internal knowledge.

Rules:ss
Output only the final answer (a name, date, place, yes/no, number, etc.)
Do not include reasoning, explanations, or any other text
Keep the answer as short as possible
If you do not know the answer, output exactly: unknown

Question: {question}

Answer:"""
)

chain = prompt | llm | StrOutputParser()


class GraphState(TypedDict):
    question: str
    answer: str


def generate(state: GraphState):
    response = chain.invoke({"question": state["question"]})
    return {"answer": response}


workflow = StateGraph(GraphState)
workflow.add_node("generate", generate)
workflow.add_edge(START, "generate")
workflow.add_edge("generate", END)

app = workflow.compile()


if __name__ == "__main__":
    test_question = (
        "Who is Charles Bretagne Marie De La Trémoille's paternal grandfather?"
    )
    result = app.invoke({"question": test_question})
