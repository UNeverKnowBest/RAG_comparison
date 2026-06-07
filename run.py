import argparse
import json
import time

from tqdm import tqdm

from src.evaluation import evaluate, exact_match, f1_score, print_report
from src.graph_rag import app as graph_app
from src.graph_rag import schema_description
from src.semantic_rag import app as semantic_app


def load_dataset(file_path, sample_size=None):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if sample_size and sample_size > 0:
        data = data[:sample_size]
    return data


def estimate_tokens(text):
    if not isinstance(text, str):
        text = str(text)
    return int(len(text.split()) * 1.3)


def main(args):
    dataset = load_dataset(args.data_path, args.sample)

    semantic_preds = []
    graph_preds = []

    try:
        for item in tqdm(dataset, desc="Processing Questions"):
            q_id = item.get("id", "")
            question = item.get("question", "")
            gold_answer = item.get("answer", "")
            q_type = item.get("type", "unknown")

            _sf_titles = item.get("supporting_facts", {}).get("title", [])
            gold_docs = list(dict.fromkeys(_sf_titles))

            gold_entities = []
            _seen_ent: set = set()
            for _triple in item.get("evidences", []):
                if len(_triple) == 3:
                    for _ent in [str(_triple[0]).strip(), str(_triple[2]).strip()]:
                        if _ent and _ent not in _seen_ent:
                            _seen_ent.add(_ent)
                            gold_entities.append(_ent)

            start_time = time.time()
            try:
                sem_result = semantic_app.invoke({"question": question})  # type: ignore[arg-type]
                sem_latency = (time.time() - start_time) * 1000

                sem_retrieved = []
                if "context" in sem_result:
                    sem_retrieved = [
                        doc.metadata.get("title", "")
                        for doc in sem_result["context"]
                        if hasattr(doc, "metadata") and "title" in doc.metadata
                    ]

                sem_answer = sem_result.get("answer", "unknown")
                sem_context_docs = sem_result.get("context", [])
                sem_context_text = " ".join(
                    doc.page_content
                    for doc in sem_context_docs
                    if hasattr(doc, "page_content")
                )
                gold_in_context = gold_answer.lower() in sem_context_text.lower()
                sem_tokens = (
                    estimate_tokens(question)
                    + estimate_tokens(sem_context_text)
                    + estimate_tokens(sem_answer)
                )

                semantic_preds.append(
                    {
                        "id": q_id,
                        "question": question,
                        "type": q_type,
                        "gold_answer": gold_answer,
                        "predicted_answer": sem_answer,
                        "em": exact_match(sem_answer, gold_answer),
                        "f1": round(f1_score(sem_answer, gold_answer), 4),
                        "gold_in_context": gold_in_context,
                        "retrieved_context": [
                            {
                                "title": doc.metadata.get("title", ""),
                                "content": doc.page_content,
                            }
                            for doc in sem_context_docs
                            if hasattr(doc, "metadata") and hasattr(doc, "page_content")
                        ],
                        "retrieved_items": sem_retrieved,
                        "gold_items": gold_docs,
                        "latency_ms": sem_latency,
                        "token_count": sem_tokens,
                    }
                )
                with open("semantic_predictions.json", "w", encoding="utf-8") as f:
                    json.dump(semantic_preds, f, indent=2, ensure_ascii=False)

            except Exception as e:
                print(f"\n[Error] Semantic RAG failed, ID {q_id}: {e}")

            start_time = time.time()
            try:
                graph_result = graph_app.invoke(  # type: ignore[arg-type]
                    {"question": question, "schema": schema_description}
                )
                graph_latency = (time.time() - start_time) * 1000

                graph_retrieved = []
                for record in graph_result.get("db_results", []):
                    for _, val in record.items():
                        if isinstance(val, str):
                            graph_retrieved.append(val)

                graph_tokens = (
                    estimate_tokens(question)
                    + estimate_tokens(graph_result.get("cypher_query", ""))
                    + estimate_tokens(graph_result.get("answer", ""))
                )

                graph_context_text = " ".join(
                    str(v)
                    for record in graph_result.get("db_results", [])
                    for v in record.values()
                )
                graph_gold_in_context = (
                    gold_answer.lower() in graph_context_text.lower()
                )

                graph_preds.append(
                    {
                        "id": q_id,
                        "question": question,
                        "type": q_type,
                        "gold_answer": gold_answer,
                        "predicted_answer": graph_result.get("answer", "unknown"),
                        "gold_in_context": graph_gold_in_context,
                        "retrieved_items": graph_retrieved,
                        "gold_items": gold_entities,
                        "latency_ms": graph_latency,
                        "token_count": graph_tokens,
                    }
                )
                with open("graph_predictions.json", "w", encoding="utf-8") as f:
                    json.dump(graph_preds, f, indent=2, ensure_ascii=False)

            except Exception as e:
                print(f"\n[Error] Graph RAG failed, ID {q_id}: {e}")

    except KeyboardInterrupt:
        print(
            f"\n[Interrupted] Saving {len(semantic_preds)} semantic and {len(graph_preds)} graph results so far..."
        )
        return

    if not semantic_preds and not graph_preds:
        print("No predictions collected, skipping evaluation.")
        return

    sem_eval = evaluate(semantic_preds, "Semantic RAG", retrieval_unit="doc")
    gph_eval = evaluate(graph_preds, "Graph RAG", retrieval_unit="entity")

    print_report(sem_eval)
    print_report(gph_eval)

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {"semantic": sem_eval, "graph": gph_eval}, f, indent=2, ensure_ascii=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG Pipelines")
    parser.add_argument(
        "--data_path", type=str, default="dataset/processed/validation_full.jsonl"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=300,
        help="Number of samples to evaluate.",
    )

    args = parser.parse_args()

    if args.sample == 0:
        args.sample = None

    main(args)
