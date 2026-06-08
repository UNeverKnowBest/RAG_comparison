import json
import random

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

from src.evaluation import exact_match
from src.graph_rag import (
    _parse_intent_json,
    answer_chain,
    build_cypher_from_intent,
    clean_cypher,
    cypher_chain,
    driver,
    intent_chain,
    llm,
    schema_description,
)

DATA_PATH = "dataset/processed/validation_full.jsonl"
SAMPLE_SIZE = 300

random.seed(42)

cypher_prompt_with_intent = ChatPromptTemplate.from_template("""
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
Intent and entities based on the quetsion: {intent}
Cypher:""")
cypher_chain_with_intent = cypher_prompt_with_intent | llm | StrOutputParser()


def load_data(path, n):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data[:n]


def run_query(cypher):
    try:
        with driver.session() as session:
            return session.run(cypher).data()
    except Exception:
        return []


def get_answer(question, results):
    if not results:
        return "unknown"
    answer = answer_chain.invoke({"question": question, "results": results})
    return answer.strip()


def random_routing(question):
    raw_intent = intent_chain.invoke({"question": question})
    intent = _parse_intent_json(raw_intent)
    intent["query_type"] = random.choice(
        ["compositional", "comparison", "bridge_comparison"]
    )
    cypher = build_cypher_from_intent(intent)
    if cypher is None:
        raw_cypher = cypher_chain.invoke(
            {"schema": schema_description, "question": question}
        )
        cypher = clean_cypher(raw_cypher)
    results = run_query(cypher)
    return get_answer(question, results)


def no_template(question):
    raw_intent = intent_chain.invoke({"question": question})
    intent = _parse_intent_json(raw_intent)
    raw_cypher = cypher_chain_with_intent.invoke(
        {"schema": schema_description, "question": question, "intent": intent}
    )
    cypher = clean_cypher(raw_cypher)
    results = run_query(cypher)
    return get_answer(question, results)


def no_fallback(question):
    raw_intent = intent_chain.invoke({"question": question})
    intent = _parse_intent_json(raw_intent)
    cypher = build_cypher_from_intent(intent)
    if cypher is None:
        return "unknown"
    results = run_query(cypher)
    return get_answer(question, results)


def run_variant(name, generate, dataset):
    correct = 0
    for item in tqdm(dataset, desc=name):
        question = item.get("question", "")
        gold = item.get("answer", "")
        try:
            pred = generate(question)
        except Exception:
            pred = "unknown"
        correct += exact_match(pred, gold)
    em = correct / len(dataset) * 100
    print(f"{name}: EM = {em:.2f}%")
    return em


def main():
    dataset = load_data(DATA_PATH, SAMPLE_SIZE)
    variants = {
        "- Intent Routing": random_routing,
        "- Template": no_template,
        "- Fallback": no_fallback,
    }
    try:
        for name, generate in variants.items():
            run_variant(name, generate, dataset)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
