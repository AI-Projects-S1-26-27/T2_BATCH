"""Run baselines B1-B3 on BASIL stories and save their outputs.

    python -m baselines.run                        # B1 + B2 + B3 on the 20 annotated stories
    python -m baselines.run --systems B1 B2        # no API calls
    python -m baselines.run --split dev --show 38f7cbb7-5d6a-4c89-bcbd-8e164144172a

Outputs: data/baselines/B1.json, B2.json, B3.json   {story_id: [difference records]}
"""

import argparse
import csv
import json
import os

from src.load import SOURCES, load_stories, stories_by_id

OUT_DIR = os.path.join("data", "baselines")


def load_b1_model():
    from baselines.b1_tfidf import TfidfModel
    with open(os.path.join("data", "processed", "headlines.csv"), encoding="utf-8") as f:
        corpus = [r["headline"] for r in csv.DictReader(f)]
    return TfidfModel(corpus)


def select(stories, split):
    if split == "all":
        return list(stories)
    with open(os.path.join("data", "splits.json")) as f:
        sp = json.load(f)
    ids = sp["dev"] + sp["test"] if split == "annotated" else sp[split]
    return [s for s in stories if s in set(ids)]


def main():
    ap = argparse.ArgumentParser(description="Run baselines B1-B3")
    ap.add_argument("--systems", nargs="+", default=["B1", "B2", "B3"], choices=["B1", "B2", "B3"])
    ap.add_argument("--split", default="annotated", choices=["annotated", "dev", "test", "all"])
    ap.add_argument("--show", metavar="STORY_ID", help="print one story's outputs")
    args = ap.parse_args()

    stories = stories_by_id(load_stories())
    ids = [args.show] if args.show else select(stories, args.split)
    runners = {}
    if "B1" in args.systems:
        from baselines import b1_tfidf
        model = load_b1_model()
        runners["B1"] = lambda h, l: (b1_tfidf.compare(h, model), [])
    if "B2" in args.systems:
        from baselines import b2_sbert
        from src.align import SBERTEmbedder
        emb = SBERTEmbedder()
        runners["B2"] = lambda h, l: (b2_sbert.compare(h, emb), [])
    if "B3" in args.systems:
        from baselines.b3_llm import B3
        b3 = B3()
        runners["B3"] = b3.compare

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, run in runners.items():
        path = os.path.join(OUT_DIR, f"{name}.json")
        results = {}
        if os.path.exists(path) and not args.show:
            with open(path, encoding="utf-8") as f:
                results = json.load(f)
        failed = 0
        for n, sid in enumerate(ids, 1):
            arts = stories[sid]["articles"]
            heads = {s: arts[s]["headline"] for s in SOURCES}
            leads = {s: [x["text"] for x in arts[s]["sentences"]] for s in SOURCES}
            recs, errors = run(heads, leads)
            results[sid] = recs
            if name == "B3":
                b3.save()
            if errors:
                failed += 1
                print(f"  [{name}] {sid[:8]} errors: {errors}")
            if args.show:
                print(f"\n=== {name} ===")
                for s in SOURCES:
                    print(f"{s}: {heads[s]}")
                for r in recs:
                    print(f"  [{r['dimension']}] {r['from']}->{r['to']}: {r['evidence'][0]['text']!r}"
                          f"  {r['explanation']}")
            else:
                print(f"[{name}] {n:2d}/{len(ids)} {sid[:8]}: {len(recs)} differences")
        if not args.show:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)
            print(f"[{name}] saved {sum(len(v) for v in results.values())} differences -> {path}"
                  + (f"  ({failed} stories with errors: re-run to retry)" if failed else ""))


if __name__ == "__main__":
    main()
