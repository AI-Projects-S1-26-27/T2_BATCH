"""Collect the filled annotation workbooks into gold files and compute Cohen's kappa.

Reads    eval/annotation/{dev,test}/*.xlsx  (as created by scripts/make_annotation_sheets.py)
Writes   eval/gold/gold_dev.csv, eval/gold/gold_test.csv    one row per headline pair
         eval/gold/kappa.json                                agreement on the 5 double-annotated stories
         eval/gold/disagreements.csv                         for M8 to adjudicate

Gold = the first annotator's labels. Where M8 has adjudicated a disagreement (filled the
"final" column of disagreements.csv), that decision replaces the first annotator's label.

Run:    python -m eval.merge_annotations            (progress report + merge)
        python -m eval.merge_annotations --split dev   (dev only: safe for M2-M5 to run)
"""

import argparse
import csv
import json
import os
import re

DIMS = ["D1_info", "D2_actor", "D3_eval", "D4_outcome"]
WORD_COLS = {"D1_info": ["D1_words_A", "D1_words_B"],
             "D2_actor": ["D2_words_A", "D2_words_B"],
             "D3_eval": ["D3_words_A", "D3_words_B", "D3_polarity"],
             "D4_outcome": ["D4_type_A", "D4_type_B", "D4_words_A", "D4_words_B"]}
LABEL_COLS = DIMS + [c for d in DIMS for c in WORD_COLS[d]] + ["notes"]
BASE_COLS = ["row_id", "story_id", "outlet_A", "headline_A", "outlet_B", "headline_B"]

ANN_DIR = os.path.join("eval", "annotation")
GOLD_DIR = os.path.join("eval", "gold")


def _clean(v):
    return re.sub(r"\s+", " ", str(v)).strip() if v is not None else ""


def read_sheet(path):
    """Rows of one annotator file (.xlsx or .csv) as dicts."""
    if path.endswith(".csv"):
        with open(path, encoding="utf-8-sig") as f:
            return [{k: _clean(v) for k, v in r.items()} for r in csv.DictReader(f)]
    from openpyxl import load_workbook
    ws = load_workbook(path, read_only=True, data_only=True)["Annotate"]
    rows = list(ws.iter_rows(values_only=True))
    head = [_clean(h) for h in rows[0]]
    return [{h: _clean(v) for h, v in zip(head, r)} for r in rows[1:] if any(r)]


def check_row(r):
    """Return a list of problems with one annotated row (empty = complete and valid)."""
    problems = []
    for d in DIMS:
        v = r.get(d, "").lower()
        r[d] = v
        if v not in ("y", "n"):
            problems.append(f"{d} is {v or 'blank'} (need y or n)")
        elif v == "y":
            words = [r.get(c, "") for c in WORD_COLS[d] if "_words_" in c]
            if not any(words):
                problems.append(f"{d}=y but no words copied")
            for c in WORD_COLS[d]:
                if "_words_" in c and r.get(c):
                    side = c[-1]
                    for part in re.split(r"\s*[;|]\s*", r[c]):
                        if part and part.lower() not in r.get(f"headline_{side}", "").lower():
                            problems.append(f"{c}: '{part}' not found in headline {side}")
    return problems


def cohen_kappa(a, b):
    """Cohen's kappa for two equal-length lists of labels."""
    n = len(a)
    if n == 0:
        return None
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    if pe == 1:
        return 1.0 if po == 1 else 0.0
    return (po - pe) / (1 - pe)


def load_all(ann_dir, splits):
    with open(os.path.join(ann_dir, "assignment.json")) as f:
        plan = json.load(f)
    primary, second, report = {}, {}, []
    for name, info in plan.items():
        if info["split"] not in splits:
            continue
        folder = os.path.join(ann_dir, info["split"])
        path = next((os.path.join(folder, f"{info['code']}_{name}{ext}") for ext in (".xlsx", ".csv")
                     if os.path.exists(os.path.join(folder, f"{info['code']}_{name}{ext}"))), None)
        if path is None:
            report.append((name, 0, 0, ["file missing"]))
            continue
        rows = read_sheet(path)
        done, problems = 0, []
        for r in rows:
            p = check_row(r)
            if not p:
                done += 1
            problems += [f"{r.get('row_id')}: {x}" for x in p]
            r["annotator"] = name
            r["split"] = info["split"]
            if r["story_id"] in info["primary"]:
                primary[r["row_id"]] = r
            elif r["story_id"] in info["second"]:
                second[r["row_id"]] = r
        report.append((name, done, len(rows), problems))
    return primary, second, report


def load_adjudication(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return {(r["row_id"], r["dimension"]): r["final"].strip().lower()
                for r in csv.DictReader(f) if r.get("final", "").strip().lower() in ("y", "n")}


def main():
    ap = argparse.ArgumentParser(description="Merge annotations into gold files and compute kappa")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--ann-dir", default=ANN_DIR)
    ap.add_argument("--gold-dir", default=GOLD_DIR)
    args = ap.parse_args()
    splits = ["dev", "test"] if args.split == "all" else [args.split]

    primary, second, report = load_all(args.ann_dir, splits)

    print("Progress (complete rows / total):")
    for name, done, total, problems in report:
        print(f"  {name:10s} {done:3d}/{total:<3d}" + ("" if not problems else f"  {len(problems)} issue(s)"))
        for p in problems[:8]:
            print(f"      - {p}")
        if len(problems) > 8:
            print(f"      ... and {len(problems) - 8} more")

    os.makedirs(args.gold_dir, exist_ok=True)
    adjud_path = os.path.join(args.gold_dir, "disagreements.csv")
    adjud = load_adjudication(adjud_path)

    # gold files
    for split in splits:
        rows = sorted((r for r in primary.values() if r["split"] == split), key=lambda r: r["row_id"])
        out = os.path.join(args.gold_dir, f"gold_{split}.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=BASE_COLS + LABEL_COLS + ["annotator", "adjudicated"])
            w.writeheader()
            for r in rows:
                row = {k: r.get(k, "") for k in BASE_COLS + LABEL_COLS + ["annotator"]}
                changed = []
                for d in DIMS:
                    if (r["row_id"], d) in adjud:
                        if adjud[(r["row_id"], d)] != row[d]:
                            changed.append(d)
                        row[d] = adjud[(r["row_id"], d)]
                row["adjudicated"] = ",".join(changed)
                w.writerow(row)
        print(f"\nWrote {len(rows)} gold pairs -> {out}")

    # kappa on double-annotated rows (both complete)
    pairs = [(primary[k], second[k]) for k in second if k in primary
             and not check_row(dict(primary[k])) and not check_row(dict(second[k]))]
    if not pairs:
        print("\nKappa: no complete double-annotated rows yet.")
        return
    kappa = {"rows": len(pairs)}
    print(f"\nCohen's kappa on {len(pairs)} double-annotated pairs:")
    all_a, all_b = [], []
    for d in DIMS:
        a, b = [p[d] for p, _ in pairs], [s[d] for _, s in pairs]
        all_a += a
        all_b += b
        k = cohen_kappa(a, b)
        agree = sum(x == y for x, y in zip(a, b)) / len(a)
        kappa[d] = {"kappa": None if k is None else round(k, 3), "raw_agreement": round(agree, 3)}
        print(f"  {d:11s} kappa={k:.2f}  raw agreement={agree:.0%}")
    k = cohen_kappa(all_a, all_b)
    kappa["overall"] = {"kappa": round(k, 3),
                        "raw_agreement": round(sum(x == y for x, y in zip(all_a, all_b)) / len(all_a), 3)}
    print(f"  {'overall':11s} kappa={k:.2f}")
    with open(os.path.join(args.gold_dir, "kappa.json"), "w") as f:
        json.dump(kappa, f, indent=1)

    # disagreements for adjudication (keep M8's existing decisions)
    with open(adjud_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "dimension", "headline_A", "headline_B",
                    "first_annotator", "first_label", "second_annotator", "second_label", "final"])
        n = 0
        for p, s in pairs:
            for d in DIMS:
                if p[d] != s[d]:
                    w.writerow([p["row_id"], d, p["headline_A"], p["headline_B"], p["annotator"], p[d],
                                s["annotator"], s[d], adjud.get((p["row_id"], d), "")])
                    n += 1
    print(f"\n{n} disagreements -> {adjud_path} (M8: fill 'final' with y/n, then re-run)")


if __name__ == "__main__":
    main()
