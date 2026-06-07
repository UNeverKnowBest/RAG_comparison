import json
import math
import re
import string
from collections import Counter, defaultdict

from scipy import stats


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


def precision_at_k(retrieved, gold, k):
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & set(gold)) / k


def hit_at_k(retrieved, gold, k):
    return int(bool(set(retrieved[:k]) & set(gold)))


def evaluate(predictions, system_name="Semantic RAG", retrieval_unit="doc"):
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
                "precision_2": precision_at_k(p["retrieved_items"], p["gold_items"], 2),
                "precision_5": precision_at_k(p["retrieved_items"], p["gold_items"], 5),
                "hit_5": hit_at_k(p["retrieved_items"], p["gold_items"], 5),
                "answer_coverage": int(p.get("gold_in_context", False)),
                "latency_ms": p["latency_ms"],
                "tokens": p["token_count"],
            }
        )

    n = len(results)
    latencies = [r["latency_ms"] for r in results]
    avg_lat = sum(latencies) / n
    std_lat = math.sqrt(sum((x - avg_lat) ** 2 for x in latencies) / max(n - 1, 1))
    overall = {
        "system": system_name,
        "retrieval_unit": retrieval_unit,
        "n_samples": n,
        "EM": sum(r["em"] for r in results) / n * 100,
        "F1": sum(r["f1"] for r in results) / n * 100,
        "Recall@2": sum(r["recall_2"] for r in results) / n * 100,
        "Recall@5": sum(r["recall_5"] for r in results) / n * 100,
        "Precision@2": sum(r["precision_2"] for r in results) / n * 100,
        "Precision@5": sum(r["precision_5"] for r in results) / n * 100,
        "Hit@5": sum(r["hit_5"] for r in results) / n * 100,
        "AnswerCoverage": sum(r["answer_coverage"] for r in results) / n * 100,
        "AvgLatency_ms": avg_lat,
        "StdLatency_ms": std_lat,
        "AvgTokens_est": sum(r["tokens"] for r in results) / n,
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
            "Precision@5": sum(r["precision_5"] for r in items) / len(items) * 100,
            "AnswerCoverage": sum(r["answer_coverage"] for r in items)
            / len(items)
            * 100,
        }

    return {
        "overall": overall,
        "by_type": type_metrics,
        "details": results,
    }


def significance_test(preds_a, preds_b):
    a_map = {p["id"]: p for p in preds_a}
    b_map = {p["id"]: p for p in preds_b}
    common = sorted(set(a_map) & set(b_map))

    metric_fns = {
        "EM": lambda p: exact_match(p["predicted_answer"], p["gold_answer"]),
        "F1": lambda p: f1_score(p["predicted_answer"], p["gold_answer"]),
    }

    tests = {}
    for name, fn in metric_fns.items():
        a_scores = [fn(a_map[i]) for i in common]
        b_scores = [fn(b_map[i]) for i in common]
        if all(a == b for a, b in zip(a_scores, b_scores)):
            tests[name] = {"statistic": 0.0, "p_value": 1.0, "significant": False}
        else:
            try:
                stat, p = stats.wilcoxon(a_scores, b_scores, zero_method="pratt")
                tests[name] = {
                    "statistic": float(stat),
                    "p_value": float(p),
                    "significant": bool(p < 0.05),
                }
            except ValueError as e:
                tests[name] = {"error": str(e)}

    return {"n_paired": len(common), "tests": tests}


def print_report(eval_result):
    o = eval_result["overall"]
    unit = o.get("retrieval_unit", "?")

    print(f"  Evaluation Report: {o['system']} (retrieval_unit={unit})")
    print(f"Samples:          {o['n_samples']}")
    print(f"EM:               {o['EM']:.2f}%")
    print(f"F1:               {o['F1']:.2f}%")
    print(
        f"AnswerCoverage:   {o.get('AnswerCoverage', float('nan')):.2f}%  [cross-system comparable]"
    )
    print(f"{'Recall@2':<18}{o.get('Recall@2', float('nan')):.2f}%")
    print(f"{'Recall@5':<18}{o.get('Recall@5', float('nan')):.2f}%")
    print(f"{'Precision@2':<18}{o.get('Precision@2', float('nan')):.2f}%")
    print(f"{'Precision@5':<18}{o.get('Precision@5', float('nan')):.2f}%")
    print(f"{'Hit@5':<18}{o.get('Hit@5', float('nan')):.2f}%")
    print(
        f"Avg Latency:      {o['AvgLatency_ms']:.1f} ± {o.get('StdLatency_ms', 0):.1f} ms"
    )
    print(f"Avg Tokens (est): {o.get('AvgTokens_est', o.get('AvgTokens', 0)):.0f}")

    print("\nBy Question Type:")
    print(
        f"  {'Type':<22} {'N':>5} {'EM':>7} {'F1':>7} {'R@5':>8} {'P@5':>8} {'Coverage':>9}"
    )
    for t, m in eval_result["by_type"].items():
        print(
            f"  {t:<22} {m['n']:>5} {m['EM']:>6.1f}% {m['F1']:>6.1f}% "
            f"{m.get('Recall@5', float('nan')):>7.1f}% {m.get('Precision@5', float('nan')):>7.1f}% "
            f"{m.get('AnswerCoverage', float('nan')):>8.1f}%"
        )


def print_significance(sig_result, name_a="Semantic RAG", name_b="Graph RAG"):
    print(f"\n{'=' * 60}")
    print(f"  Significance Test: {name_a} vs {name_b}")
    print(f"  Wilcoxon , n={sig_result['n_paired']}")
    print(f"{'=' * 60}")
    for metric, result in sig_result["tests"].items():
        if "error" in result:
            print(f"  {metric:<8}: ERROR — {result['error']}")
        else:
            sig_str = "* p<0.05" if result["significant"] else "  n.s."
            print(
                f"  {metric:<8}: stat={result['statistic']:.2f}, "
                f"p={result['p_value']:.4f}  {sig_str}"
            )


if __name__ == "__main__":
    with open("semantic_predictions.json") as f:
        sem_preds = json.load(f)
    with open("graph_predictions.json") as f:
        gph_preds = json.load(f)

    sem_eval = evaluate(sem_preds, "Semantic RAG", retrieval_unit="doc")
    gph_eval = evaluate(gph_preds, "Graph RAG", retrieval_unit="entity")

    print_report(sem_eval)
    print_report(gph_eval)

    sig = significance_test(sem_preds, gph_preds)
    print_significance(sig)

    with open("evaluation_results.json", "w") as f:
        json.dump(
            {
                "semantic": sem_eval,
                "graph": gph_eval,
                "significance_test": sig,
            },
            f,
            indent=2,
        )
