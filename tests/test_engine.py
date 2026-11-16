"""Tests for Stage S4 (facts + ontology) and S5 (forward chaining, rules R1-R4).
No model or API needed.  Run:  python -m tests.test_engine
"""

import csv
import os
import subprocess
import sys
import tempfile

from src.engine import (RULES, Rule, article_support, check_stratification, derive_differences,
                        explain, forward_chain)
from src.facts import atom_key, build_facts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ engine mechanics

def test_transitive_chaining_is_multi_step():
    facts = {("IsA", "A", "B"), ("IsA", "B", "C"), ("IsA", "C", "D")}
    known, trace = forward_chain(facts, [RULES[0]])
    assert ("IsA", "A", "D") in known                       # needs two rounds of the same rule
    d = trace[("IsA", "A", "D")][0]
    assert d["rule"] == "T_IsA_transitive"
    assert any(m in trace for m in d["matched"])            # built from a derived fact


def test_negation_as_failure():
    r = Rule("R", ("Only", "?a", "?i"), [("Pair", "?a", "?b"), ("Reports", "?a", "?i")],
             neg=[("Reports", "?b", "?i")], stratum=1)
    facts = {("Pair", "x", "y"), ("Reports", "x", "i1"), ("Reports", "x", "i2"), ("Reports", "y", "i1")}
    known, trace = forward_chain(facts, [r])
    assert ("Only", "x", "i2") in known and ("Only", "x", "i1") not in known
    assert trace[("Only", "x", "i2")][0]["absent"] == [("Reports", "y", "i2")]


def test_stratification_checks():
    bad = [Rule("P", ("Q", "?x"), [("B", "?x")]),
           Rule("N", ("Z", "?x"), [("B", "?x")], neg=[("Q", "?x")], stratum=0)]
    try:
        check_stratification(bad)
        raise AssertionError("negating a predicate of the same stratum must fail")
    except ValueError as e:
        assert "stratum" in str(e)
    unsafe = [Rule("U", ("Z", "?x"), [("B", "?x")], neg=[("Q", "?y")], stratum=1)]
    try:
        check_stratification(unsafe)
        raise AssertionError("unbound variable in negation must fail")
    except ValueError as e:
        assert "not bound" in str(e)
    check_stratification(RULES)                              # the real rule set is valid


# ------------------------------------------------------------------ a complete story

def _ext(actors, items, agents=(), evals=(), outcomes=()):
    def span(t):
        return {"span": t, "start": 0, "end": len(t)}
    return {"ok": True, "fields": {
        "actors": [{"name": n, "type": t, **span(n)} for n, t in actors],
        "items": [{"text": t, **span(t)} for t in items],
        "agents": [{"actor": a, "action": i, **span(f"{a} {i}")} for a, i in agents],
        "evals": [{"target": x, "polarity": p, "word": w, **span(w)} for x, p, w in evals],
        "outcomes": [{"text": t, "type": ty, **span(t)} for t, ty in outcomes]}}


EXT = {
    "fox": _ext([("police", "LawEnforcement"), ("protesters", "Activist")],
                ["fire tear gas", "city faces curfew"], agents=[("police", "fire tear gas")],
                evals=[("protesters", "positive", "peaceful")], outcomes=[("curfew", "Safety")]),
    "nyt": _ext([("protesters", "Activist")], ["use tear gas", "protest turns violent"],
                evals=[("protesters", "negative", "violent")], outcomes=[("injuries", "Health")]),
    "hpo": _ext([("police", "LawEnforcement"), ("protesters", "Activist")], ["fire tear gas"],
                agents=[("police", "fire tear gas")], outcomes=[("costs city millions", "Economic")]),
}
IDS = {"actors": {"police": "a1", "protesters": "a2"},
       "items": {"fire tear gas": "i1", "use tear gas": "i1", "city faces curfew": "i2",
                 "protest turns violent": "i3"},
       "outcomes": {"curfew": "o1", "injuries": "o2", "costs city millions": "o3"}}
KEY = {"actors": "name", "items": "text", "outcomes": "text"}
ALIGN = {"mentions": {kind: [{"src": s, "index": n, "id": IDS[kind][m[KEY[kind]]]}
                             for s in EXT for n, m in enumerate(EXT[s]["fields"][kind])]
                      for kind in IDS}}
ARTICLES = {"fox": ["Police used tear gas downtown.", "Several people were hurt."],
            "nyt": ["A curfew begins tonight across the city.", "Officers used tear gas."],
            "hpo": ["The protest was large."]}


def _diffs():
    kb = build_facts(EXT, ALIGN)
    return kb, derive_differences(kb, article_texts=ARTICLES)


def test_facts():
    kb, _ = _diffs()
    atoms = set(kb["atoms"])
    assert ("Reports", "nyt", "i1") in atoms and ("Agent", "fox", "a1", "i1") in atoms
    assert ("Eval", "fox", "a2", "positive", "peaceful") in atoms
    assert ("Outcome", "nyt", "o2", "Health") in atoms and ("InstanceOf", "a1", "LawEnforcement") in atoms
    assert ("Pair", "fox", "nyt") in atoms and ("Pair", "nyt", "fox") in atoms
    assert kb["evidence"][atom_key(("Reports", "fox", "i2"))]["text"] == "city faces curfew"
    assert kb["own_labels"]["nyt:i1"] == "use tear gas"


def test_rules_r1_to_r4():
    _, diffs = _diffs()
    got = {(d["rule"], d["from"], d["to"]) for d in diffs}
    # R1: curfew only in fox, violence only in nyt
    assert ("R1_InfoOnlyIn", "fox", "nyt") in got and ("R1_InfoOnlyIn", "nyt", "fox") in got
    r1 = next(d for d in diffs if d["rule"] == "R1_InfoOnlyIn" and d["item"] == "city faces curfew"
              and d["to"] == "nyt")
    assert r1["status"] == "headline_omission" and "curfew" in r1["article_evidence"]["sentence"]
    # R2: fox and hpo name police as firing tear gas; nyt reports it with no actor
    assert ("R2_ActorShift", "fox", "nyt") in got and ("R2_ActorShift", "hpo", "nyt") in got
    assert ("R2_ActorShift", "fox", "hpo") not in got                  # same actor: no shift
    r2 = next(d for d in diffs if d["rule"] == "R2_ActorShift" and d["from"] == "fox")
    assert r2["other_actors"] == [] and r2["other_action"] == "use tear gas"
    # R3: opposite tone about protesters, both directions; hpo mentions them neutrally
    assert {("R3_EvalShift", "fox", "nyt"), ("R3_EvalShift", "nyt", "fox"),
            ("R3_EvalShift", "fox", "hpo")} <= got
    # R4: Safety vs Economic and Health vs Economic fire once each;
    #     Health vs Safety is blocked by the ontology (Health IsA Safety)
    r4 = [frozenset((d["left_type"], d["right_type"])) for d in diffs if d["rule"] == "R4_OutcomeShift"]
    assert sorted(r4, key=sorted) == sorted([frozenset(("Health", "Economic")),
                                             frozenset(("Safety", "Economic"))], key=sorted), r4


def test_trace_uses_derived_facts():
    _, diffs = _diffs()
    r3 = next(d for d in diffs if d["rule"] == "R3_EvalShift" and d["to"] == "hpo")
    assert "Mentions(hpo, a2)" in r3["trace"][0]["matched"]           # derived in stratum 0
    assert "HasEval(hpo, a2, positive)" in r3["trace"][0]["absent"]
    assert r3["evidence"][0]["text"] == "peaceful"
    assert "peaceful" in explain(r3)


def test_article_support_fallback():
    s = article_support("city faces curfew", ["A curfew begins tonight across the city."])
    assert s["found"] and s["method"] == "word-overlap"
    assert not article_support("stock market rally", ["A curfew begins tonight."])["found"]


def test_export_review_runs_on_repo_data():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "review.csv")
        subprocess.run([sys.executable, "scripts/export_review.py", "--split", "dev", "--no-sbert",
                        "--output", out],
                       cwd=ROOT, check=True, capture_output=True, text=True)
        with open(out, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert rows and {r["rule"][:2] for r in rows} <= {"R1", "R2", "R3", "R4"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All S4/S5 checks passed.")
