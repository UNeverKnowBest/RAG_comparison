import json

from config import PROCESSED_DIR, RAW_DIR


def main():
    val_path = RAW_DIR / "validation.jsonl"
    with open(val_path, encoding="utf-8") as f:
        all_validation = [json.loads(line) for line in f]
    print(f"Loaded {len(all_validation)} validation samples")

    valid_samples = [item for item in all_validation if item.get("evidences")]
    print(f"Final dataset size: {len(valid_samples)}")
    out_path = PROCESSED_DIR / "validation_full.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in valid_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
