"""Tests for eval/evaluate.py.  Run:  python -m tests.test_evaluate"""

import csv
import os
import tempfile

import eval.evaluate as E


def _gold():
    rows = {}
    for sid, pair, labels in [("s1", ("fox", "nyt"), "yynn"), ("s1", ("fox", "hpo"), "nnyn"),
                              ("s2", ("nyt", "hpo"), "ynny")]:
        r = {"story_id": sid, "outlet_A": pair[0], "outlet_B": pair[1],
             "D1_words_A": "tear gas", "D3_words_A": "slams"}
        r.update({d: v for d, v in zip(E.DIMS, labels)})
        rows[(sid, frozenset(pair))] = r
    return rows


def rec(d, a, b, words=""):
    return {"dimension": d, "from": a, "to": b, "evidence": [{"text": words}]}


def test_prf_and_score():
    assert E.prf(2, 1, 1) == {"tp": 2, "fp": 1, "fn": 1, "precision": 0.667, "recall": 0.667, "f1": 0.667}
    records = {"s1": [rec("D1_info", "nyt", "fox", "fired tear gas"),      # TP (either direction)
                      rec("D3_eval", "fox", "hpo", "calm"),                 # TP, words do not overlap
                      rec("D4_outcome", "fox", "nyt")],                     # FP
               "s2": [rec("D1_info", "hpo", "nyt", "x")],                   # TP
               "s9": [rec("D1_info", "fox", "nyt")]}                        # story not in gold: ignored
    res, rows = E.score(records, _gold())
    assert res["D1_info"]["tp"] == 2 and res["D1_info"]["fn"] == 0
    assert res["D2_actor"]["fn"] == 1 and res["D2_actor"]["tp"] == 0        # missed
    assert res["D4_outcome"]["fp"] == 1 and res["D4_outcome"]["fn"] == 1
    assert res["micro"]["tp"] == 3 and len(rows) == 12
    assert res["evidence_overlap"] == round(1 / 3, 3)                       # only "tear gas" overlaps


def test_ablation_on_repo_data():
    import json
    from src.align import load_alignment
    with open(os.path.join("data", "extractions", "extractions.json"), encoding="utf-8") as f:
        ext = json.load(f)
    out = E.ablation_records(ext, load_alignment())
    recs = [r for v in out.values() for r in v]
    assert recs and {r["dimension"] for r in recs} <= set(E.DIMS)
    assert all(r["from"] != r["to"] for r in recs)


def test_rating_round_trip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rate.csv")
        sid = "38f7cbb7-5d6a-4c89-bcbd-8e164144172a"
        gold = {(sid, frozenset(("fox", "nyt"))): {}}
        systems = {"OURS": {sid: [dict(rec("D3_eval", "nyt", "fox", "Used"), explanation="tone")]},
                   "B1": {sid: [dict(rec("D1_info", "fox", "nyt", "Urge"), explanation="word")]}}
        E.make_rating_sheet(gold, systems, path)
        with open(path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2 and "system" not in rows[0]                   # blind
        for r in rows:
            r["supported (yes/partial/no)"] = "yes"
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        out = E.score_ratings(path)
        assert out["OURS"]["yes"] == 1 and out["B1"]["yes"] == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All evaluation checks passed.")
