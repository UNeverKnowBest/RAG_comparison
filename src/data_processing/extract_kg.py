import json
import re
from collections import defaultdict

import pandas as pd
from config import KG_DIR, NATIONALITY_NORMALIZE, NATIONALITY_RELATIONS, PROCESSED_DIR


def clean_name(name):
    if not name:
        return ""
    name = re.sub(r"\s+", " ", str(name)).strip()
    return name


def normalize_rel(rel):
    if not rel:
        return "RELATED_TO"
    rel = re.sub(r"[^A-Z0-9]+", "_", str(rel).upper()).strip("_")
    return rel or "RELATED_TO"


def extract_kg_data(samples):
    entities = defaultdict(lambda: {"name": "", "mentions": set()})
    relations = []
    rel_counter = defaultdict(int)
    stats = {"skipped": 0, "norm_count": 0}

    for item in samples:
        for step, triple in enumerate(item.get("evidences", [])):
            if len(triple) != 3:
                stats["skipped"] += 1
                continue

            s_raw, r_raw, o_raw = triple
            r_norm = normalize_rel(r_raw)

            is_nat = r_norm in NATIONALITY_RELATIONS
            s_clean = clean_name(s_raw)
            o_clean = clean_name(o_raw)

            if not s_clean or not o_clean:
                stats["skipped"] += 1
                continue

            if is_nat and o_clean != o_raw.strip():
                stats["norm_count"] += 1

            for clean, raw in [(s_clean, s_raw), (o_clean, o_raw)]:
                entities[clean]["name"] = clean
                entities[clean]["mentions"].add(raw)

            relations.append(
                {
                    "head": s_clean,
                    "tail": o_clean,
                    "type": r_norm,
                    "qid": item["id"],
                    "step": step,
                    "q_type": item["type"],
                }
            )
            rel_counter[r_norm] += 1

    print(f"Entities: {len(entities)}, Relations: {len(relations)}")
    return entities, relations, rel_counter


def main():
    input_path = PROCESSED_DIR / "validation_full.jsonl"
    with open(input_path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    entities, relations, rel_counter = extract_kg_data(samples)
    pd.DataFrame(
        [
            {
                "name": e["name"],
                "mentions": "|".join(sorted(e["mentions"])),
                "n": len(e["mentions"]),
            }
            for e in entities.values()
        ]
    ).to_csv(KG_DIR / "nodes.csv", index=False)

    pd.DataFrame(relations).to_csv(KG_DIR / "relations.csv", index=False)
    pd.DataFrame(
        [
            {"rel": r, "count": c}
            for r, c in sorted(rel_counter.items(), key=lambda x: -x[1])
        ]
    ).to_csv(KG_DIR / "relation_statistics.csv", index=False)


if __name__ == "__main__":
    main()
