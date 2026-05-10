import json
import re
import string
from collections import Counter, defaultdict


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def recall_at_k(retrieved, gold, k):
    top_k = set(retrieved[:k])
    if not gold:
        return 0.0
    return len(top_k & set(gold)) / len(gold)


def hit_at_k(retrieved, gold, k):
    return int(bool(set(retrieved[:k]) & set(gold)))


def evaluate(predictions, system_name="Semantic RAG"):
    results = []
    for p in predictions:
        results.append(
            {
                "id": p["id"],
                "type": p["type"],
                "em": exact_match(p["predicted_answer"], p["gold_answer"]),
                "f1": f1_score(p["predicted_answer"], p["gold_answer"]),
                "recall_2": recall_at_k(p["retrieved_items"], p["gold_items"], 2),
                "recall_5": recall_at_k(p["retrieved_items"], p["gold_items"], 5),
                "hit_5": hit_at_k(p["retrieved_items"], p["gold_items"], 5),
                "latency_ms": p["latency_ms"],
                "tokens": p["token_count"],
            }
        )

    n = len(results)
    overall = {
        "system": system_name,
        "n_samples": n,
        "EM": sum(r["em"] for r in results) / n * 100,
        "F1": sum(r["f1"] for r in results) / n * 100,
        "Recall@2": sum(r["recall_2"] for r in results) / n * 100,
        "Recall@5": sum(r["recall_5"] for r in results) / n * 100,
        "Hit@5": sum(r["hit_5"] for r in results) / n * 100,
        "AvgLatency_ms": sum(r["latency_ms"] for r in results) / n,
        "AvgTokens": sum(r["tokens"] for r in results) / n,
    }

    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)
    type_metrics = {}
    for t, items in by_type.items():
        type_metrics[t] = {
            "n": len(items),
            "EM": sum(r["em"] for r in items) / len(items) * 100,
            "F1": sum(r["f1"] for r in items) / len(items) * 100,
            "Recall@5": sum(r["recall_5"] for r in items) / len(items) * 100,
        }

    return {"overall": overall, "by_type": type_metrics, "details": results}


def print_report(eval_result):
    o = eval_result["overall"]
    print(f"\n{'=' * 60}")
    print(f"  Evaluation Report: {o['system']}")
    print(f"{'=' * 60}")
    print(f"  Samples: {o['n_samples']}")
    print(f"  EM:           {o['EM']:.2f}%")
    print(f"  F1:           {o['F1']:.2f}%")
    print(f"  Recall@2:     {o['Recall@2']:.2f}%")
    print(f"  Recall@5:     {o['Recall@5']:.2f}%")
    print(f"  Hit@5:        {o['Hit@5']:.2f}%")
    print(f"  Avg Latency:  {o['AvgLatency_ms']:.1f} ms")
    print(f"  Avg Tokens:   {o['AvgTokens']:.0f}")

    print(f"\n  By Question Type:")
    print(f"  {'Type':<22} {'N':>5} {'EM':>7} {'F1':>7} {'R@5':>7}")
    for t, m in eval_result["by_type"].items():
        print(
            f"  {t:<22} {m['n']:>5} {m['EM']:>6.1f}% {m['F1']:>6.1f}% {m['Recall@5']:>6.1f}%"
        )


if __name__ == "__main__":
    with open("semantic_predictions.json") as f:
        sem_preds = json.load(f)
    with open("graph_predictions.json") as f:
        gph_preds = json.load(f)

    sem_eval = evaluate(sem_preds, "Semantic RAG")
    gph_eval = evaluate(gph_preds, "Graph RAG")

    print_report(sem_eval)
    print_report(gph_eval)

    with open("evaluation_results.json", "w") as f:
        json.dump({"semantic": sem_eval, "graph": gph_eval}, f, indent=2)
