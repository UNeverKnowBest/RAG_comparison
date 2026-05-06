import json

from config import DATASET_NAME, RAW_DIR
from datasets import load_dataset


def main():
    print(f"Downloading {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)
    for split in ds:
        path = RAW_DIR / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for data in ds[split]:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        print("Saved")


if __name__ == "__main__":
    main()
