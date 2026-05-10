import argparse
import json
import time

from tqdm import tqdm

from src.evaluation import evaluate, print_report
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
    print(f"Starting evaluation on {len(dataset)} samples...")

    semantic_preds = []
    graph_preds = []

    for item in tqdm(dataset, desc="Processing Questions"):
        q_id = item.get("id", "")
        question = item.get("question", "")
        gold_answer = item.get("answer", "")
        q_type = item.get("type", "unknown")

        gold_items = item.get("supporting_facts", {}).get("title", [])

        start_time = time.time()
        try:
            sem_result = semantic_app.invoke({"question": question})
            sem_latency = (time.time() - start_time) * 1000

            sem_retrieved = []
            if "context" in sem_result:
                sem_retrieved = [
                    doc.metadata.get("title", "")
                    for doc in sem_result["context"]
                    if hasattr(doc, "metadata") and "title" in doc.metadata
                ]

            context_text = " ".join(
                [
                    doc.page_content
                    for doc in sem_result.get("context", [])
                    if hasattr(doc, "page_content")
                ]
            )
            sem_tokens = (
                estimate_tokens(question)
                + estimate_tokens(context_text)
                + estimate_tokens(sem_result.get("answer", ""))
            )

            semantic_preds.append(
                {
                    "id": q_id,
                    "question": question,
                    "type": q_type,
                    "gold_answer": gold_answer,
                    "predicted_answer": sem_result.get("answer", "unknown"),
                    "retrieved_items": sem_retrieved,
                    "gold_items": gold_items,
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
            graph_result = graph_app.invoke(
                {"question": question, "schema": schema_description}
            )
            graph_latency = (time.time() - start_time) * 1000

            graph_retrieved = []
            for record in graph_result.get("db_results", []):
                for key, val in record.items():
                    if isinstance(val, str):
                        graph_retrieved.append(val)

            graph_tokens = (
                estimate_tokens(question)
                + estimate_tokens(graph_result.get("cypher_query", ""))
                + estimate_tokens(graph_result.get("answer", ""))
            )

            graph_preds.append(
                {
                    "id": q_id,
                    "question": question,
                    "type": q_type,
                    "gold_answer": gold_answer,
                    "predicted_answer": graph_result.get("answer", "unknown"),
                    "retrieved_items": graph_retrieved,
                    "gold_items": gold_items,
                    "latency_ms": graph_latency,
                    "token_count": graph_tokens,
                }
            )
            with open("graph_predictions.json", "w", encoding="utf-8") as f:
                json.dump(graph_preds, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"\n[Error] Graph RAG failed, ID {q_id}: {e}")

    sem_eval = evaluate(semantic_preds, "Semantic RAG")
    gph_eval = evaluate(graph_preds, "Graph RAG")

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
        default=100,
        help="Number of samples to evaluate.",
    )

    args = parser.parse_args()

    if args.sample == 0:
        args.sample = None

    main(args)
