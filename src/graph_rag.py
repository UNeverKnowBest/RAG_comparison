import json
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
    num_ctx=4096,
    temperature=TEMPERATURE,
    top_p=0.95,
)

schema_description = """Knowledge graph schema:
We have all the relationships in our graph below:

people from one country:
(person)-[:COUNTRY_OF_CITIZENSHIP]->(country)

work from one country:
(work)-[:COUNTRY_OF_ORIGIN]->(country)

this place is in one country:
(place)-[:COUNTRY]->(country)

work relation schema:
(film)-[:DIRECTOR]->(person)
(song)-[:COMPOSER]->(person)
(song)-[:PERFORMER]->(person)
(work)-[:PRODUCER]->(person)
(work)-[:PUBLICATION_DATE]->(date)

person related schema:
(person)-[:DATE_OF_BIRTH]->(date)
(person)-[:DATE_OF_DEATH]->(date)
(person)-[:PLACE_OF_BIRTH]->(place)
(person)-[:PLACE_OF_DEATH]->(place)
(person)-[:PLACE_OF_BURIAL]->(place)
(person)-[:CAUSE_OF_DEATH]->(cause)
(person)-[:EDUCATED_AT]->(institution)
(person)-[:EMPLOYER]->(organization)
(person)-[:AWARD_RECEIVED]->(award)

family related schema:
(person)-[:FATHER]->(person)
(person)-[:MOTHER]->(person)
(person)-[:SPOUSE]->(person)
(person)-[:CHILD]->(person)
(person)-[:SIBLING]->(person)

other relations:
(organization)-[:FOUNDED_BY]->(person)
(entity)-[:HAS_PART]->(entity)
(entity)-[:INCEPTION]->(date)
"""


cypher_prompt = ChatPromptTemplate.from_template("""
You are a Neo4j expert writing Cypher queries. You need to write the Cypher queries based on the question.
{schema}

Output only the Cypher query with no explanation.
The query must start with MATCH and end with RETURN. Limit results to 5.
You need to first identify the intent of the question, extract the entity from the questions and then generate the Cypher based on the intent and entities.
You have the intent below:
compositional: traverse a relation path starting from one named entity:
Comparison: compare two named entities on the same attribute:
Bridge comparison: bridge through a relation to reach two intermediate entities then compare them:
Inference: infer new relation based on the exist relation from the schema.
Use toLower(n.name) CONTAINS toLower('entity')

For example:
Question: Who is the mother of the director of film Polish-Russian War?
Intent:compositional entity:Polish-Russian War
Cypher: MATCH (f:Entity)-[:DIRECTOR]->(d:Entity)-[:MOTHER]->(m:Entity)
WHERE toLower(f.name) CONTAINS toLower('Polish-Russian War')
RETURN m.name AS answer LIMIT 5

Question: Who is Charles Bretagne Marie De La Trémoille's paternal grandfather?
Intent:inference entity:Charles Bretagne Marie De La Trémoille
Cypher: MATCH (p:Entity)-[:FATHER]->(father:Entity)-[:FATHER]->(gf:Entity)
WHERE toLower(p.name) CONTAINS toLower('Charles Bretagne Marie De La Trémoille')
RETURN gf.name AS answer LIMIT 5


Question: Are director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?
Intent: comparison entity:Move entity2:Méditerranée
Cypher: MATCH (f1:Entity)-[:DIRECTOR]->(d1:Entity)-[:COUNTRY_OF_CITIZENSHIP]->(c1:Entity),
       (f2:Entity)-[:DIRECTOR]->(d2:Entity)-[:COUNTRY_OF_CITIZENSHIP]->(c2:Entity)
WHERE toLower(f1.name) CONTAINS toLower('Move')
  AND toLower(f2.name) CONTAINS toLower('Méditerranée')
RETURN d1.name AS director1, c1.name AS country1, d2.name AS director2, c2.name AS country2 LIMIT 10


Now generate Cypher for:
Question: {question}
Cypher:""")


answer_prompt = ChatPromptTemplate.from_template("""
Answer the question using only the query results below.
Give a short direct answer. For yes or no questions say only "yes" or "no".
If the results are empty, say "unknown".

Question: {question}
Results: {results}
Answer:""")

cypher_chain = cypher_prompt | llm | StrOutputParser()
answer_chain = answer_prompt | llm | StrOutputParser()


VALID_RELATIONS = frozenset(
    {
        "MOTHER",
        "FATHER",
        "SPOUSE",
        "CHILD",
        "SIBLING",
        "DIRECTOR",
        "COMPOSER",
        "PERFORMER",
        "PRODUCER",
        "DATE_OF_BIRTH",
        "DATE_OF_DEATH",
        "PLACE_OF_BIRTH",
        "PLACE_OF_DEATH",
        "PLACE_OF_BURIAL",
        "CAUSE_OF_DEATH",
        "COUNTRY_OF_CITIZENSHIP",
        "EDUCATED_AT",
        "EMPLOYER",
        "AWARD_RECEIVED",
        "PUBLICATION_DATE",
        "COUNTRY_OF_ORIGIN",
        "COUNTRY",
        "FOUNDED_BY",
        "INCEPTION",
        "HAS_PART",
    }
)

intent_prompt = ChatPromptTemplate.from_template(
    """You are a query planner for a Neo4j knowledge graph.
You need to extract the entity of the questions and recognize the intent of the query.

We only have the relation type below, so don't output any other relation types.
Copy entity names from the question and strip the hints like "(film)" or "(song)".

You have the relation types below:
Family: MOTHER, FATHER, SPOUSE, CHILD, SIBLING
Film: DIRECTOR, COMPOSER, PERFORMER, PRODUCER
Person dates: DATE_OF_BIRTH, DATE_OF_DEATH
Person places: PLACE_OF_BIRTH, PLACE_OF_DEATH, PLACE_OF_BURIAL, CAUSE_OF_DEATH
Person related: COUNTRY_OF_CITIZENSHIP, EDUCATED_AT, EMPLOYER, AWARD_RECEIVED
Work related: PUBLICATION_DATE, COUNTRY_OF_ORIGIN
Other: COUNTRY, FOUNDED_BY, INCEPTION, HAS_PART

Output exactly one of these four JSON formats:

compositional: traverse a relation path starting from one named entity:
{{"query_type":"compositional","entity":"<name>","relation_chain":["REL1","REL2"]}}

Comparison: compare two named entities on the same attribute:
{{"query_type":"comparison","entity1":"<name1>","entity2":"<name2>","comparison_relation":"REL"}}

Bridge comparison — bridge through a relation to reach two intermediate entities then compare them:
{{"query_type":"bridge_comparison","entity1":"<name1>","entity2":"<name2>","bridge_relation":"REL","comparison_relation":"REL"}}

Unknown: question cannot be mapped to the schema:
{{"query_type":"unknown"}}

There are some question need you to infer the relation.
For example:
"born in" or "birth place" maps to PLACE_OF_BIRTH.
"born on" or "birth date" maps to DATE_OF_BIRTH.
"died on" or "death date" maps to DATE_OF_DEATH.
"paternal grandfather" maps to ["FATHER","FATHER"].
"maternal grandfather" maps to ["MOTHER","FATHER"].
"mother-in-law" maps to ["SPOUSE","MOTHER"].
"father-in-law" maps to ["SPOUSE","FATHER"].

Output ONLY the JSON object with no markdown and no explanation.

Question: Who is the mother of the director of film Polish-Russian War?
{{"query_type":"compositional","entity":"Polish-Russian War","relation_chain":["DIRECTOR","MOTHER"]}}

Question: Do the movies Bloody Birthday and The Beckoning Silence originate from the same country?
{{"query_type":"comparison","entity1":"Bloody Birthday","entity2":"The Beckoning Silence","comparison_relation":"COUNTRY_OF_ORIGIN"}}

Question: Are director of film Move and director of film Méditerranée from the same country?
{{"query_type":"bridge_comparison","entity1":"Move","entity2":"Méditerranée","bridge_relation":"DIRECTOR","comparison_relation":"COUNTRY_OF_CITIZENSHIP"}}

Question: {question}
JSON:"""
)

intent_chain = intent_prompt | llm | StrOutputParser()


def _parse_intent_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"query_type": "unknown"}


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def build_cypher_from_intent(intent: dict) -> Optional[str]:
    qt = intent.get("query_type", "unknown")

    if qt == "compositional":
        anchor = intent.get("entity", "").strip()
        chain = [r.upper() for r in intent.get("relation_chain", [])]
        if not anchor or not (1 <= len(chain) <= 4):
            return None
        if not all(r in VALID_RELATIONS for r in chain):
            return None
        pattern = "(n0:Entity)"
        for i, rel in enumerate(chain):
            pattern += f"-[:{rel}]->(n{i + 1}:Entity)"
        return (
            f"MATCH {pattern}\n"
            f"WHERE toLower(n0.name) CONTAINS toLower('{_escape(anchor)}')\n"
            f"RETURN n{len(chain)}.name AS answer LIMIT 5"
        )

    if qt == "comparison":
        e1 = intent.get("entity1", "").strip()
        e2 = intent.get("entity2", "").strip()
        rel = intent.get("comparison_relation", "").upper()
        if not e1 or not e2 or rel not in VALID_RELATIONS:
            return None
        return (
            f"MATCH (e1:Entity)-[:{rel}]->(v1:Entity), (e2:Entity)-[:{rel}]->(v2:Entity)\n"
            f"WHERE toLower(e1.name) CONTAINS toLower('{_escape(e1)}')\n"
            f"  AND toLower(e2.name) CONTAINS toLower('{_escape(e2)}')\n"
            f"RETURN e1.name AS entity1, v1.name AS value1, "
            f"e2.name AS entity2, v2.name AS value2 LIMIT 10"
        )

    if qt == "bridge_comparison":
        e1 = intent.get("entity1", "").strip()
        e2 = intent.get("entity2", "").strip()
        bridge = intent.get("bridge_relation", "").upper()
        cmp_rel = intent.get("comparison_relation", "").upper()
        if (
            not e1
            or not e2
            or bridge not in VALID_RELATIONS
            or cmp_rel not in VALID_RELATIONS
        ):
            return None
        return (
            f"MATCH (e1:Entity)-[:{bridge}]->(b1:Entity)-[:{cmp_rel}]->(v1:Entity),\n"
            f"       (e2:Entity)-[:{bridge}]->(b2:Entity)-[:{cmp_rel}]->(v2:Entity)\n"
            f"WHERE toLower(e1.name) CONTAINS toLower('{_escape(e1)}')\n"
            f"  AND toLower(e2.name) CONTAINS toLower('{_escape(e2)}')\n"
            f"RETURN e1.name AS entity1, b1.name AS bridge1, v1.name AS value1, "
            f"e2.name AS entity2, b2.name AS bridge2, v2.name AS value2 LIMIT 10"
        )

    return None


class GraphState(TypedDict):
    question: str
    schema: str
    cypher_query: str
    result: List[Dict[str, Any]]
    formatted_results: str
    answer: str
    error: Optional[str]
    query_mode: Optional[str]


def clean_cypher(text: str) -> str:

    if not text or not isinstance(text, str):
        return ""

    text = text.strip()
    text = re.sub(
        r"^```(?:cypher|Cypher|query|Query)?\s*\n?", "", text, flags=re.IGNORECASE
    )
    return text.strip()


def generate_cypher(state: GraphState):
    question = state["question"]

    raw_intent = intent_chain.invoke({"question": question})
    intent = _parse_intent_json(raw_intent)
    template_cypher = build_cypher_from_intent(intent)
    if template_cypher:
        return {"cypher_query": template_cypher, "query_mode": "template"}

    raw_cypher = cypher_chain.invoke({"schema": state["schema"], "question": question})
    return {"cypher_query": clean_cypher(raw_cypher), "query_mode": "llm_fallback"}


def execute_query(state: GraphState):
    query = state["cypher_query"]
    results = []
    error = None
    try:
        with driver.session() as session:
            results = session.run(query).data()
    except Exception as e:
        error = str(e)

    return {"result": results, "error": error}


def generate_answer(state: GraphState):
    if state.get("error") and not state.get("result"):
        return {"answer": "unknown"}
    response = answer_chain.invoke(
        {"question": state["question"], "results": state["result"]}
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
    finally:
        driver.close()
