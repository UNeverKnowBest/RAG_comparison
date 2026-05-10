import re
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from neo4j import GraphDatabase
from typing_extensions import TypedDict

from src.data_processing.config import (
    BASE_URL,
    MODEL_NAME,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    TEMPERATURE,
)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

llm = ChatOllama(
    base_url=BASE_URL,
    model=MODEL_NAME,
    num_ctx=8192,
    temperature=TEMPERATURE,
)

schema_description = """Knowledge graph schema:

All nodes have label :Entity with properties: name (string), mentions (string).

Relationship types (all between :Entity nodes):

PEOPLE-TO-COUNTRY:
- (person)-[:COUNTRY_OF_CITIZENSHIP]->(country)

WORKS-TO-COUNTRY:
- (film/song/work)-[:COUNTRY_OF_ORIGIN]->(country)

PLACES-TO-COUNTRY:
- (place/organization)-[:COUNTRY]->(country)

FILM/WORK ATTRIBUTES:
- (film)-[:DIRECTOR]->(person)
- (film/song)-[:COMPOSER]->(person)
- (song)-[:PERFORMER]->(person)
- (work)-[:PRODUCER]->(person)
- (work)-[:PUBLICATION_DATE]->(date)

PERSON ATTRIBUTES:
- (person)-[:DATE_OF_BIRTH]->(date)
- (person)-[:DATE_OF_DEATH]->(date)
- (person)-[:PLACE_OF_BIRTH]->(place)
- (person)-[:PLACE_OF_DEATH]->(place)
- (person)-[:PLACE_OF_BURIAL]->(place)
- (person)-[:CAUSE_OF_DEATH]->(cause)
- (person)-[:EDUCATED_AT]->(institution)
- (person)-[:EMPLOYER]->(organization)
- (person)-[:AWARD_RECEIVED]->(award)

FAMILY:
- (person)-[:FATHER]->(person)
- (person)-[:MOTHER]->(person)
- (person)-[:SPOUSE]->(person)
- (person)-[:CHILD]->(person)
- (person)-[:SIBLING]->(person)

OTHER:
- (organization)-[:FOUNDED_BY]->(person)
- (entity)-[:HAS_PART]->(entity)
- (entity)-[:INCEPTION]->(date)"""


cypher_prompt = ChatPromptTemplate.from_template("""You are a Neo4j expert.
{schema}

RULES:
1. Always Use toLower(n.name) CONTAINS toLower('keyword') for search.
3. Limit results to 30
4. Output ONLY the Cypher query, no markdown fences, no explanation
5. The query MUST start with MATCH and end with RETURN

EXAMPLES:
Question: Who is the mother of the director of film Polish-Russian War?
Cypher: MATCH (f:Entity)-[:DIRECTOR]->(d:Entity)-[:MOTHER]->(m:Entity)
WHERE toLower(f.name) CONTAINS toLower('Polish-Russian War')
RETURN m.name AS answer LIMIT 5

Question: When did John V, Prince Of Anhalt-Zerbst's father die?
Cypher: MATCH (p:Entity)-[:FATHER]->(f:Entity)
WHERE toLower(p.name) CONTAINS toLower('John V, Prince Of Anhalt-Zerbst')
RETURN f.`date_of_death` AS answer LIMIT 5

Question: Who is Charles Bretagne Marie De La Trémoille's paternal grandfather?
Cypher: MATCH (p:Entity)-[:FATHER]->(father:Entity)-[:FATHER]->(gf:Entity)
WHERE toLower(p.name) CONTAINS toLower('Charles Bretagne Marie De La Trémoille')
RETURN gf.name AS paternal_grandfather LIMIT 5

Question: Where was the director of film Ronnie Rocket born?
Cypher: MATCH (f:Entity)-[:DIRECTOR]->(d:Entity)
WHERE toLower(f.name) CONTAINS toLower('Ronnie Rocket')
RETURN d.place_of_birth AS answer LIMIT 5

Question: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
Cypher: MATCH (f1:Entity)-[:PUBLICATION_DATE]->(d1), (f2:Entity)-[:PUBLICATION_DATE]->(d2)
WHERE toLower(f1.name) CONTAINS toLower('Blind Shaft')
  AND toLower(f2.name) CONTAINS toLower('The Mask Of Fu Manchu')
RETURN f1.name AS film1, d1 AS date1, f2.name AS film2, d2 AS date2 LIMIT 10

Question: What is the award that the director of film Wearing Velvet Slippers Under A Golden Umbrella won?
Cypher: MATCH (f:Entity)-[:DIRECTOR]->(d:Entity)-[:AWARD_RECEIVED]->(a:Entity)
WHERE toLower(f.name) CONTAINS toLower('Wearing Velvet Slippers Under A Golden Umbrella')
RETURN a.name AS answer LIMIT 5

Now generate Cypher for the following question:
Question: {question}
Cypher:""")


answer_prompt = ChatPromptTemplate.from_template("""Answer the question based ONLY on the following query results.
Question: {question}
Results: {results}
Instructions: Short, direct answer. For yes/no, use only "yes"/"no". If no results, answer "unknown".
Answer:""")

cypher_chain = cypher_prompt | llm | StrOutputParser()
answer_chain = answer_prompt | llm | StrOutputParser()


class GraphState(TypedDict):
    question: str
    schema: str
    cypher_query: str
    db_results: List[Dict[str, Any]]
    formatted_results: str
    answer: str
    error: Optional[str]


def clean_cypher(text: str) -> str:

    if not text or not isinstance(text, str):
        return ""

    text = text.strip()

    text = re.sub(
        r"^```(?:cypher|Cypher|query|Query)?\s*\n?", "", text, flags=re.IGNORECASE
    )
    text = re.sub(r"\s*```\s*$", "", text)

    text = re.sub(
        r"^(?:Cypher:|Query:|Here is the Cypher:|The Cypher query is:)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^(?:Sure, here|Okay, the|The following).*?query is:?\s*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"^\s+", "", text, flags=re.MULTILINE)

    return text.strip()


def generate_cypher(state: GraphState):
    raw_cypher = cypher_chain.invoke(
        {"schema": state["schema"], "question": state["question"]}
    )
    print(raw_cypher)
    return {"cypher_query": clean_cypher(raw_cypher)}


def execute_query(state: GraphState):
    query = state["cypher_query"]
    results = []
    error = None
    try:
        with driver.session() as session:
            results = session.run(query).data()
    except Exception as e:
        error = str(e)

    return {"db_results": results, "error": error}


def generate_answer(state: GraphState):
    if state.get("error") and not state.get("db_results"):
        return {"answer": "unknown"}
    response = answer_chain.invoke(
        {"question": state["question"], "results": state["db_results"]}
    )
    return {"answer": response.strip()}


workflow = StateGraph(GraphState)
workflow.add_node("generate_cypher", generate_cypher)
workflow.add_node("execute_query", execute_query)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "generate_cypher")
workflow.add_edge("generate_cypher", "execute_query")
workflow.add_edge("execute_query", "generate_answer")
workflow.add_edge("generate_answer", END)

app = workflow.compile()

if __name__ == "__main__":
    test_question = (
        "Where was the place of burial of Albert Frederick, Duke Of Prussia's mother?"
    )
    try:
        result = app.invoke({"question": test_question, "schema": schema_description})
        print(f"Question : {result['question']}\n")
        print(f"Cypher : {result['cypher_query']}\n")
        print(f"Answer : {result['answer']}")
    finally:
        driver.close()
