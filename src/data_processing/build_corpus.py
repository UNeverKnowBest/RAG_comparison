import hashlib
import json

from config import CORPUS_DIR, PROCESSED_DIR


def make_id(*parts):
    key = "::".join(map(str, parts))
    return hashlib.md5(key.encode()).hexdigest()[:12]


def build_corpus(samples):
    passages, sentences = [], []
    for item in samples:
        qid = item["id"]
        ctx = item.get("context", {})
        sf = item.get("supporting_facts", {})
        sf_titles = set(sf.get("title", []))
        sf_pairs = set(zip(sf.get("title", []), sf.get("sent_id", [])))

        for p_idx, (title, sents) in enumerate(
            zip(ctx.get("title", []), ctx.get("sentences", []))
        ):
            p_id = make_id(qid, title, p_idx)
            passages.append(
                {
                    "passage_id": p_id,
                    "question_id": qid,
                    "title": title,
                    "text": " ".join(sents),
                    "is_supporting": title in sf_titles,
                }
            )

            for s_idx, sent in enumerate(sents):
                sentences.append(
                    {
                        "sentence_id": make_id(qid, title, p_idx, s_idx),
                        "passage_id": p_id,
                        "question_id": qid,
                        "title": title,
                        "sent_idx": s_idx,
                        "text": sent,
                        "is_supporting": (title, s_idx) in sf_pairs,
                    }
                )
    return passages, sentences


def write_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in items)


def main():
    input_path = PROCESSED_DIR / "validation_full.jsonl"
    with open(input_path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    passages, sentences = build_corpus(samples)
    write_jsonl(passages, CORPUS_DIR / "passages.jsonl")
    write_jsonl(sentences, CORPUS_DIR / "sentences.jsonl")


if __name__ == "__main__":
    main()
