from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from config import (
    MODEL_NAME,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    TEMPERATURE,
)

llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)
graph = Neo4jGraph(url=NEO4J_URI, password=NEO4J_PASSWORD, username=NEO4J_USERNAME)


class GraphState(TypedDict):
    question: str
    query: str
    query_result: str
    answer: str


def generate_cypher(state: GraphState):
    question = state["question"]
    schema = graph.schema
    prompt = ChatPromptTemplate.from_template(
        """
        You are a Cypher expert with deep knowledge of Neo4j, you need
        to convert the multi-hop questions into a Cypher query.
        You need to reason step by step and finally output the only final
        Cypher query.
        Important: Return only the Cypher query.
        Do not include any Markdown formatting or explanations.

        Graph Schema:{schema}
        Question:{question}
        Answer  """
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"schema": schema, "question": question})
    return {"query": response}


def query(state: GraphState):
    query = state["query"]
    result = graph.query(query)
    if not result:
        results = "No relevant data found"
    else:
        results = str(result)
    return {"query_result": results}


def generate_anwer(state: GraphState):
    question = state["question"]
    query_result = state["query_result"]
    prompt = ChatPromptTemplate.from_template(
        """
        You are a Question Answering assisant power by a Knowledge Graph,

        Question:{question}
        Retrieved data from Knowledge Graph: {query_result}

        Based on the data above, answer the user's question concisely and clearly.
        You need to reason step by step and finally output the only short final
        answer(a name, date, country, yes/no, etc). If the context is insufficient,
        please output unkonwn.
        Answer  """
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question, "query_result": query_result})
    return {"answer": response}


workflow = StateGraph(GraphState)
workflow.add_node("generate_cypher", generate_cypher)
workflow.add_node("query", query)
workflow.add_node("generate_anwer", generate_anwer)
workflow.add_edge(START, "generate_cypher")
workflow.add_edge("generate_cypher", "query")
workflow.add_edge("query", "generate_anwer")
workflow.add_edge("generate_anwer", END)
app = workflow.compile()

if __name__ == "__main__":
    test_question = "Are director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?"
    result = app.invoke({"question": test_question})
    print(f"Question: {test_question}\n")
    print(f"Cypher:{result['query']}\n")
    print(f"Query Result:{result['query_result']}\n")
    print(f"Answer: {result['answer']}\n")
