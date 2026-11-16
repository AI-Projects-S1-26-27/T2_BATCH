"""Evaluation (proposal, Section 6): score our system, the baselines and the ablation
against the gold annotations.

Unit of scoring: one headline pair x one dimension (D1-D4). A system "detects" a
difference on dimension d for pair {A, B} if it outputs at least one record with that
dimension between A and B (either direction). Compared with the gold y/n label this
gives TP / FP / FN, hence precision, recall and F1 per dimension, plus micro-averages.

Evidence overlap (proposal: "compared with the annotated differences by text overlap"):
for every true positive, do the words the system cites share at least one content word
with the words the annotator copied for that dimension?

Systems
    OURS      data/differences/differences.json   (python -m src.engine)
    B1/B2/B3  data/baselines/B1.json ...            (python -m baselines.run)
    ABLATION  computed here: S2 extraction + S3 alignment, NO rules. A dimension is flagged
              whenever the two headlines' extracted fields differ:
              D1 item IDs, D2 (actor, item) agent pairs, D3 (target, polarity) evals,
              D4 outcome types. It drops the rules' conditions (R2: B must report the
              action; R3: B must mention the target; R4: both stress an outcome and the
              ontology must not relate the types), so it isolates what the rules add.

Run:
    python -m eval.evaluate                  # test split (the single test run)
    python -m eval.evaluate --split dev      # dev split (safe for tuning)
    python -m eval.evaluate --make-rating-sheet       # blind evidence-support sheet
    python -m eval.evaluate --score-ratings           # after the sheet is filled
Outputs: eval/results/results_<split>.json, results_<split>.md, pairs_<split>.csv
"""

import argparse
import csv
import json
import os
import random
import re

DIMS = ["D1_info", "D2_actor", "D3_eval", "D4_outcome"]
DIM_NAMES = {"D1_info": "Information selection", "D2_actor": "Actor/action framing",
             "D3_eval": "Evaluative wording", "D4_outcome": "Consequence emphasis"}
GOLD_WORDS = {"D1_info": ["D1_words_A", "D1_words_B"], "D2_actor": ["D2_words_A", "D2_words_B"],
              "D3_eval": ["D3_words_A", "D3_words_B"], "D4_outcome": ["D4_words_A", "D4_words_B"]}
SYSTEM_FILES = {"OURS": os.path.join("data", "differences", "differences.json"),
                "B1": os.path.join("data", "baselines", "B1.json"),
                "B2": os.path.join("data", "baselines", "B2.json"),
                "B3": os.path.join("data", "baselines", "B3.json")}
OUT_DIR = os.path.join("eval", "results")
_STOP = set("a an the of to in on for and or by with as at from is are was were be his her its "
            "their he she it they this that".split())


# ------------------------------------------------------------------ loading

def load_gold(split, gold_dir=os.path.join("eval", "gold")):
    path = os.path.join(gold_dir, f"gold_{split}.csv")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    gold = {}
    for r in rows:
        key = (r["story_id"], frozenset((r["outlet_A"], r["outlet_B"])))
        gold[key] = r
    return gold


def load_system(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ablation_records(extractions, alignments):
    """No-rules variant: flag a dimension when the extracted field sets differ."""
    from src.facts import build_facts
    out = {}
    for sid, ext in extractions.items():
        if sid not in alignments:
            continue
        kb = build_facts(ext, alignments[sid])
        ev = kb["evidence"]
        by = {}
        for a in kb["atoms"]:
            by.setdefault(a[0], []).append(a)
        heads = kb["headlines"]

        def fields(h):
            return {"D1_info": {a[2] for a in by.get("Reports", []) if a[1] == h},
                    "D2_actor": {(a[2], a[3]) for a in by.get("Agent", []) if a[1] == h},
                    "D3_eval": {(a[2], a[3]) for a in by.get("Eval", []) if a[1] == h},
                    "D4_outcome": {a[3] for a in by.get("Outcome", []) if a[1] == h and a[3] != "Other"}}

        recs = []
        for i, a in enumerate(heads):
            for b in heads[i + 1:]:
                fa, fb = fields(a), fields(b)
                for d in DIMS:
                    for src, extra in ((a, fa[d] - fb[d]), (b, fb[d] - fa[d])):
                        if not extra:
                            continue
                        cited = [e for k, e in ev.items() if e["source"] == src and
                                 ((d == "D1_info" and k.startswith("Reports(")) or
                                  (d == "D2_actor" and k.startswith("Agent(")) or
                                  (d == "D3_eval" and k.startswith("Eval(")) or
                                  (d == "D4_outcome" and k.startswith("Outcome(")))]
                        recs.append({"system": "ABLATION", "dimension": d, "from": src,
                                     "to": b if src == a else a, "evidence": cited})
        out[sid] = recs
    return out


# ------------------------------------------------------------------ scoring

def content_words(text):
    return {re.sub(r"(ing|ed|es|s)$", "", w) or w
            for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in _STOP}


def predictions(records, gold):
    """{(story, pair, dim): [records]} for pairs in gold."""
    pred = {}
    for sid, recs in (records or {}).items():
        for r in recs:
            key = (sid, frozenset((r["from"], r["to"])))
            if key in gold and r.get("dimension") in DIMS:
                pred.setdefault(key + (r["dimension"],), []).append(r)
    return pred


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}


def score(records, gold):
    pred = predictions(records, gold)
    res, pair_rows = {}, []
    totals = [0, 0, 0]
    ev_hits, ev_total = 0, 0
    for d in DIMS:
        tp = fp = fn = 0
        for key, g in gold.items():
            truth = g[d] == "y"
            recs = pred.get(key + (d,), [])
            hit = bool(recs)
            if hit and truth:
                tp += 1
                gw = content_words(" ".join(g.get(c, "") for c in GOLD_WORDS[d]))
                cw = content_words(" ".join(e.get("text") or "" for r in recs for e in r.get("evidence", [])))
                ev_total += 1
                ev_hits += bool(gw & cw)
            elif hit:
                fp += 1
            elif truth:
                fn += 1
            pair_rows.append({"story_id": key[0], "pair": "-".join(sorted(key[1])), "dimension": d,
                              "gold": "y" if truth else "n", "predicted": "y" if hit else "n",
                              "n_records": len(recs)})
        res[d] = prf(tp, fp, fn)
        for k, v in zip(range(3), (tp, fp, fn)):
            totals[k] += v
    res["micro"] = prf(*totals)
    res["macro_f1"] = round(sum(res[d]["f1"] for d in DIMS) / len(DIMS), 3)
    res["evidence_overlap"] = round(ev_hits / ev_total, 3) if ev_total else None
    res["n_records"] = sum(len(v) for k, v in (records or {}).items()
                           if any(k == g[0] for g in gold))
    return res, pair_rows


def b3_citation_rate(records, gold):
    ids = {k[0] for k in gold}
    rs = [r for sid, v in (records or {}).items() if sid in ids for r in v]
    return round(sum(r.get("evidence_found", True) for r in rs) / len(rs), 3) if rs else None


# ------------------------------------------------------------------ report

def markdown(results, split, n_pairs):
    systems = list(results)
    lines = [f"# Results on the {split} split ({n_pairs} headline pairs)", "",
             "F1 per dimension (precision / recall in brackets):", "",
             "| Dimension | " + " | ".join(systems) + " |",
             "|---|" + "---|" * len(systems)]
    for d in DIMS:
        cells = []
        for s in systems:
            m = results[s][d]
            cells.append(f"{m['f1']:.2f} ({m['precision']:.2f} / {m['recall']:.2f})")
        lines.append(f"| {DIM_NAMES[d]} | " + " | ".join(cells) + " |")
    for label, f in (("Micro F1", lambda m: f"{m['micro']['f1']:.2f}"),
                     ("Macro F1", lambda m: f"{m['macro_f1']:.2f}"),
                     ("Evidence overlap with gold words (TPs)",
                      lambda m: "-" if m["evidence_overlap"] is None else f"{m['evidence_overlap']:.0%}"),
                     ("Differences output", lambda m: str(m["n_records"]))):
        lines.append(f"| **{label}** | " + " | ".join(f(results[s]) for s in systems) + " |")
    gold_pos = {d: results[systems[0]][d]["tp"] + results[systems[0]][d]["fn"] for d in DIMS}
    lines += ["", "Gold positives per dimension: " +
              ", ".join(f"{DIM_NAMES[d]} {gold_pos[d]}" for d in DIMS) + ".",
              "B1/B2 can only output D1/D3 (see baselines/common.py), so their D2/D4 scores are 0 by design."]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ blind evidence-support rating

def make_rating_sheet(gold, systems, path, per_system=40, seed=407):
    """Shuffled, system-blind sheet: each row = one claimed difference + its cited words."""
    from src.load import load_stories, stories_by_id
    stories = stories_by_id(load_stories())
    rng = random.Random(seed)
    ids = {k[0] for k in gold}
    rows = []
    for name, recs in systems.items():
        pool = [(sid, r) for sid, v in (recs or {}).items() if sid in ids for r in v]
        rng.shuffle(pool)
        for sid, r in pool[:per_system]:
            arts = stories[sid]["articles"]
            claim = r.get("explanation") or ""
            if not claim and name == "OURS":
                from src.engine import explain
                claim = explain(r)
            rows.append({"system": name, "story_id": sid,
                         "headline_1": f"{r['from']}: {arts[r['from']]['headline']}",
                         "headline_2": f"{r['to']}: {arts[r['to']]['headline']}",
                         "claimed_difference": claim,
                         "cited_words": "; ".join(e.get("text") or "" for e in r.get("evidence", []))})
    rng.shuffle(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    key_path = path.replace(".csv", "_KEY.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f, \
            open(key_path, "w", newline="", encoding="utf-8") as k:
        w = csv.writer(f)
        kw = csv.writer(k)
        w.writerow(["item", "headline_1", "headline_2", "claimed_difference", "cited_words",
                    "supported (yes/partial/no)"])
        kw.writerow(["item", "system", "story_id"])
        for n, r in enumerate(rows, 1):
            w.writerow([n, r["headline_1"], r["headline_2"], r["claimed_difference"], r["cited_words"], ""])
            kw.writerow([n, r["system"], r["story_id"]])
    print(f"Wrote {len(rows)} items -> {path}\n(key with system names: {key_path}; raters must not open it)")


def score_ratings(path):
    key_path = path.replace(".csv", "_KEY.csv")
    with open(path, encoding="utf-8-sig") as f:
        rated = {r["item"]: r["supported (yes/partial/no)"].strip().lower() for r in csv.DictReader(f)}
    with open(key_path, encoding="utf-8") as f:
        key = {r["item"]: r["system"] for r in csv.DictReader(f)}
    out = {}
    for item, sysname in key.items():
        v = rated.get(item, "")
        if v in ("yes", "partial", "no"):
            out.setdefault(sysname, {"yes": 0, "partial": 0, "no": 0})[v] += 1
    print("Evidence-support rate (yes / partial / no):")
    for s, c in out.items():
        n = sum(c.values())
        print(f"  {s:8s} n={n:3d}  yes {c['yes'] / n:.0%}  partial {c['partial'] / n:.0%}  no {c['no'] / n:.0%}")
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description="Score all systems against the gold annotations")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--make-rating-sheet", action="store_true")
    ap.add_argument("--score-ratings", action="store_true")
    args = ap.parse_args()
    rating_path = os.path.join(OUT_DIR, f"evidence_rating_{args.split}.csv")
    if args.score_ratings:
        score_ratings(rating_path)
        return

    gold = load_gold(args.split)
    systems = {name: load_system(p) for name, p in SYSTEM_FILES.items()}
    for name, recs in systems.items():
        if recs is None:
            print(f"(skipping {name}: {SYSTEM_FILES[name]} not found)")
    systems = {k: v for k, v in systems.items() if v is not None}

    from src.align import load_alignment
    with open(os.path.join("data", "extractions", "extractions.json"), encoding="utf-8") as f:
        systems["ABLATION"] = ablation_records(json.load(f), load_alignment())

    if args.make_rating_sheet:
        make_rating_sheet(gold, {k: v for k, v in systems.items() if k != "ABLATION"}, rating_path)
        return

    results, all_pairs = {}, []
    for name, recs in systems.items():
        res, pairs = score(recs, gold)
        if name == "B3":
            res["b3_citations_found"] = b3_citation_rate(recs, gold)
        results[name] = res
        all_pairs += [{"system": name, **p} for p in pairs]

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"results_{args.split}.json"), "w") as f:
        json.dump(results, f, indent=1)
    md = markdown(results, args.split, len(gold))
    if "B3" in results and results["B3"].get("b3_citations_found") is not None:
        md += f"B3 citations found verbatim in the headline: {results['B3']['b3_citations_found']:.0%}.\n"
    with open(os.path.join(OUT_DIR, f"results_{args.split}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(OUT_DIR, f"pairs_{args.split}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_pairs[0]))
        w.writeheader()
        w.writerows(all_pairs)
    print(md)
    print(f"Saved to {OUT_DIR}/results_{args.split}.json, .md and pairs_{args.split}.csv")


if __name__ == "__main__":
    main()
