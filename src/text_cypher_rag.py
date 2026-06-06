from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.graph_rag import (
    answer_chain,
    clean_cypher,
    cypher_chain,
    driver,
    schema_description,
)


class T2CState(TypedDict):
    question: str
    schema: str
    cypher_query: str
    db_results: List[Dict[str, Any]]
    answer: str
    error: Optional[str]


def generate_cypher(state: T2CState):
    raw = cypher_chain.invoke(
        {"schema": state["schema"], "question": state["question"]}
    )
    print(f"[text2cypher]\n{raw}")
    return {"cypher_query": clean_cypher(raw)}


def execute_query(state: T2CState):
    results = []
    error = None
    try:
        with driver.session() as session:
            results = session.run(state["cypher_query"]).data()
    except Exception as e:
        error = str(e)
    return {"db_results": results, "error": error}


def generate_answer(state: T2CState):
    if state.get("error") and not state.get("db_results"):
        return {"answer": "unknown"}
    response = answer_chain.invoke(
        {"question": state["question"], "results": state["db_results"]}
    )
    return {"answer": response.strip()}


workflow = StateGraph(T2CState)
workflow.add_node("generate_cypher", generate_cypher)
workflow.add_node("execute_query", execute_query)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "generate_cypher")
workflow.add_edge("generate_cypher", "execute_query")
workflow.add_edge("execute_query", "generate_answer")
workflow.add_edge("generate_answer", END)

app = workflow.compile()


if __name__ == "__main__":
    test_question = "Where was the director of film Ronnie Rocket born?"
    try:
        result = app.invoke({"question": test_question, "schema": schema_description})
    finally:
        driver.close()
