"""Stage S5: forward-chaining engine + framing rules R1-R4 (proposal, Section 5.3).

The engine is generic: rules are data (Horn clauses), facts are ground atoms
(tuples such as ("Reports", "fox", "i1")), and variables are strings starting with "?".

    Rule("R1_InfoOnlyIn",
         head=("InfoOnlyIn", "?a", "?b", "?i"),
         body=[("Pair", "?a", "?b"), ("Reports", "?a", "?i")],
         neg=[("Reports", "?b", "?i")])
    reads:  Pair(a,b) AND Reports(a,i) AND NOT Reports(b,i)  =>  InfoOnlyIn(a,b,i)

Forward chaining: repeatedly apply every rule to the known facts and add the new
conclusions, until no rule can add anything (a fixed point).

Negation ("NOT") is negation-as-failure under the closed-world assumption: NOT Reports(b,i)
means "Reports(b,i) is not in the knowledge base". This is only sound if every fact the
negation looks at is already derived, so rules are run in STRATA:
    stratum 0  helper facts: IsA transitivity, Mentions, HasEval     (no negation)
    stratum 1  framing rules R1-R4, which negate stratum-0 / base predicates
check_stratification() verifies this ordering.

Every derived fact keeps a trace: the rule, the facts it matched, and the facts that
were checked to be absent. That trace is the explanation shown to the user (S6).
"""

import re
from dataclasses import dataclass, field

from src.facts import atom_key


# ------------------------------------------------------------------ generic engine

@dataclass
class Rule:
    name: str
    head: tuple
    body: list                       # positive literals, matched left to right
    neg: list = field(default_factory=list)      # literals that must NOT be facts
    tests: list = field(default_factory=list)    # built-in tests, e.g. ("neq", "?t", "?u")
    stratum: int = 0
    text: str = ""                   # human-readable form for explanations


def is_var(x):
    return isinstance(x, str) and x.startswith("?")


def substitute(lit, env):
    return tuple(env.get(x, x) if is_var(x) else x for x in lit)


def unify(pattern, fact, env):
    """Extend env so that pattern matches fact, or return None."""
    if len(pattern) != len(fact) or pattern[0] != fact[0]:
        return None
    env = dict(env)
    for p, f in zip(pattern[1:], fact[1:]):
        if is_var(p):
            if p in env and env[p] != f:
                return None
            env[p] = f
        elif p != f:
            return None
    return env


BUILTINS = {"neq": lambda a, b: a != b}


def match_body(body, index, env=None):
    """All variable bindings that satisfy every positive literal (backtracking join)."""
    env = env or {}
    if not body:
        yield env
        return
    first, rest = body[0], body[1:]
    for fact in index.get(first[0], ()):
        e = unify(first, fact, env)
        if e is not None:
            yield from match_body(rest, index, e)


def check_stratification(rules):
    """A rule may only negate predicates that are complete before its stratum starts."""
    heads = {}
    for r in rules:
        heads.setdefault(r.head[0], set()).add(r.stratum)
    for r in rules:
        for lit in r.neg:
            later = [s for s in heads.get(lit[0], ()) if s >= r.stratum]
            if later:
                raise ValueError(f"{r.name}: negates {lit[0]}, which is still being derived "
                                 f"in stratum {max(later)}; move it to an earlier stratum")
        for v in {x for lit in r.neg + r.tests + [r.head] for x in lit if is_var(x)}:
            if not any(v in lit for lit in r.body):
                raise ValueError(f"{r.name}: variable {v} is not bound by a positive literal")


def forward_chain(facts, rules):
    """Return (all_facts, trace). trace[atom] = list of derivations, each
    {"rule", "matched": [atoms], "absent": [atoms], "bindings"}."""
    check_stratification(rules)
    known = set(facts)
    index = {}
    for f in known:
        index.setdefault(f[0], set()).add(f)
    trace = {}

    for stratum in sorted({r.stratum for r in rules}):
        layer = [r for r in rules if r.stratum == stratum]
        changed = True
        while changed:                                   # iterate this stratum to a fixed point
            changed = False
            for rule in layer:
                new = []
                for env in match_body(rule.body, index):
                    if any(substitute(n, env) in known for n in rule.neg):
                        continue
                    if not all(BUILTINS[t[0]](*substitute(t, env)[1:]) for t in rule.tests):
                        continue
                    head = substitute(rule.head, env)
                    derivation = {"rule": rule.name,
                                  "matched": [substitute(b, env) for b in rule.body],
                                  "absent": [substitute(n, env) for n in rule.neg],
                                  "bindings": {k[1:]: v for k, v in env.items()}}
                    derivs = trace.setdefault(head, [])
                    if derivation not in derivs:
                        derivs.append(derivation)
                    if head not in known:
                        new.append(head)
                for head in new:
                    if head not in known:
                        known.add(head)
                        index.setdefault(head[0], set()).add(head)
                        changed = True
    return known, trace


# ------------------------------------------------------------------ framing rules

RULES = [
    # stratum 0: helper facts (positive rules only)
    Rule("T_IsA_transitive", ("IsA", "?c", "?e"), [("IsA", "?c", "?d"), ("IsA", "?d", "?e")],
         text="IsA(c,d) AND IsA(d,e) => IsA(c,e)"),
    Rule("M_actor", ("Mentions", "?h", "?x"), [("Actor", "?h", "?x")],
         text="Actor(h,x) => Mentions(h,x)"),
    Rule("M_item", ("Mentions", "?h", "?i"), [("Reports", "?h", "?i")],
         text="Reports(h,i) => Mentions(h,i)"),
    Rule("H_eval", ("HasEval", "?h", "?x", "?p"), [("Eval", "?h", "?x", "?p", "?w")],
         text="Eval(h,x,p,w) => HasEval(h,x,p)"),

    # stratum 1: the four framing rules, each applied in both directions via Pair(a,b)
    Rule("R1_InfoOnlyIn", ("InfoOnlyIn", "?a", "?b", "?i"),
         [("Pair", "?a", "?b"), ("Reports", "?a", "?i")],
         neg=[("Reports", "?b", "?i")], stratum=1,
         text="Reports(A,i) AND NOT Reports(B,i) => InfoOnlyIn(A,B,i)"),
    Rule("R2_ActorShift", ("ActorShift", "?a", "?b", "?x", "?i"),
         [("Pair", "?a", "?b"), ("Agent", "?a", "?x", "?i"), ("Reports", "?b", "?i")],
         neg=[("Agent", "?b", "?x", "?i")], stratum=1,
         text="Agent(A,x,i) AND Reports(B,i) AND NOT Agent(B,x,i) => ActorShift(A,B,i)"),
    Rule("R3_EvalShift", ("EvalShift", "?a", "?b", "?x", "?p", "?w"),
         [("Pair", "?a", "?b"), ("Eval", "?a", "?x", "?p", "?w"), ("Mentions", "?b", "?x")],
         neg=[("HasEval", "?b", "?x", "?p")], stratum=1,
         text="Eval(A,x,p,w) AND Mentions(B,x) AND NOT Eval(B,x,p,_) => EvalShift(A,B,x,w)"),
    Rule("R4_OutcomeShift", ("OutcomeShift", "?a", "?b", "?t", "?u"),
         [("Pair", "?a", "?b"), ("Outcome", "?a", "?k", "?t"), ("Outcome", "?b", "?m", "?u"),
          ("Specific", "?t"), ("Specific", "?u")],
         neg=[("IsA", "?t", "?u"), ("IsA", "?u", "?t")], tests=[("neq", "?t", "?u")], stratum=1,
         text="Outcome(A,k,t) AND Outcome(B,k',t') AND t != t' AND NOT IsA(t,t') "
              "AND NOT IsA(t',t) => OutcomeShift(A,B,t,t')"),
]
RULE_TEXT = {r.name: r.text for r in RULES}
DIFFERENCE_PREDICATES = {"InfoOnlyIn": "R1_InfoOnlyIn", "ActorShift": "R2_ActorShift",
                         "EvalShift": "R3_EvalShift", "OutcomeShift": "R4_OutcomeShift"}
DIMENSION = {"R1_InfoOnlyIn": "D1_info", "R2_ActorShift": "D2_actor",
             "R3_EvalShift": "D3_eval", "R4_OutcomeShift": "D4_outcome"}


# ------------------------------------------------------------------ article provenance for R1

ARTICLE_THRESHOLD = 0.55      # Sentence-BERT similarity for "the other article says it too"
_STOP = set("a an the of to in on for and or by with as at from is are was were be been "
            "his her its their he she it they this that".split())


def _content_words(text):
    return {re.sub(r"(ing|ed|es|s)$", "", w) or w
            for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOP}


def split_sentences(text):
    if isinstance(text, (list, tuple)):
        return [s for s in text if s and s.strip()]
    return [s for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def article_support(item_text, sentences, embedder=None, threshold=ARTICLE_THRESHOLD):
    """Best-matching sentence of the other article for an information item.
    With an embedder: cosine similarity. Without: share of the item's content words found."""
    if not sentences or not item_text:
        return None
    if embedder is not None:
        import numpy as np
        vecs = embedder.encode([item_text] + list(sentences))
        q, s = vecs[0], vecs[1:]
        sims = s @ q / (np.linalg.norm(s, axis=1) * np.linalg.norm(q) + 1e-12)
        best = int(np.argmax(sims))
        score = float(sims[best])
        method = "sentence-bert"
    else:
        words = _content_words(item_text)
        if not words:
            return None
        scores = [len(words & _content_words(s)) / len(words) for s in sentences]
        best = max(range(len(scores)), key=scores.__getitem__)
        score, method, threshold = scores[best], "word-overlap", 0.6
    return {"sentence": sentences[best], "sentence_index": best, "score": round(score, 3),
            "method": method, "found": score >= threshold}


# ------------------------------------------------------------------ public API

def derive_differences(kb, headlines=None, article_texts=None, ontology=None, embedder=None,
                       rules=None):
    """Run forward chaining on one story's KB and turn derived difference atoms into
    explanation records (one per difference; symmetric R4 results reported once).

    kb             output of src.facts.build_facts
    article_texts  {source: article text or list of sentences}, used for R1 provenance
    ontology       optional extra (child, parent) pairs added as IsA facts
    """
    headlines = headlines or {}
    article_texts = article_texts or {}
    facts = set(kb["atoms"]) | {("IsA", c, d) for c, d in (ontology or [])}
    known, trace = forward_chain(facts, rules or RULES)
    ev, own = kb["evidence"], kb.get("own_labels", {})

    def lab_in(h, x):
        """How headline h words id x (falls back to the story-level label)."""
        return own.get(f"{h}:{x}") or kb["labels"].get(x, x)

    def evidence_of(atoms):
        out = []
        for a in atoms:
            e = ev.get(atom_key(a))
            if e and e not in out:
                out.append(e)
        return out

    results, seen_r4 = [], set()
    order = {"InfoOnlyIn": 1, "ActorShift": 2, "EvalShift": 3, "OutcomeShift": 4}
    for head in sorted((f for f in known if f[0] in DIFFERENCE_PREDICATES),
                       key=lambda f: (order[f[0]], f[1:])):
        rule = DIFFERENCE_PREDICATES[head[0]]
        derivs = trace[head]
        matched = [m for d in derivs for m in d["matched"] if m[0] != "Pair"]
        rec = {"rule": rule, "dimension": DIMENSION[rule], "from": head[1], "to": head[2],
               "facts": [atom_key(head)], "evidence": evidence_of(matched),
               "trace": [{"rule": d["rule"], "rule_text": RULE_TEXT[d["rule"]],
                          "matched": [atom_key(m) for m in d["matched"]],
                          "absent": [atom_key(n) for n in d["absent"]]} for d in derivs]}

        if rule == "R1_InfoOnlyIn":
            i = head[3]
            rec["item"] = lab_in(head[1], i)
            sents = split_sentences(article_texts.get(head[2]))
            sup = article_support(rec["item"], sents, embedder)
            rec["status"] = ("article_not_available" if not sents else
                             "headline_omission" if sup and sup["found"] else "not_found_in_article")
            rec["article_evidence"] = ({"source": head[2], **sup} if sup and sup["found"] else None)
        elif rule == "R2_ActorShift":
            x, i = head[3], head[4]
            rec["actor"], rec["action"] = lab_in(head[1], x), lab_in(head[1], i)
            rec["other_action"] = lab_in(head[2], i)
            others = sorted({f[2] for f in known if f[0] == "Agent" and f[1] == head[2] and f[3] == i})
            rec["other_actors"] = [lab_in(head[2], o) for o in others]
            rec["evidence"] += evidence_of([("Agent", head[2], o, i) for o in others]
                                           + [("Reports", head[2], i)])
        elif rule == "R3_EvalShift":
            x, p, w = head[3], head[4], head[5]
            rec["actor"], rec["polarity"], rec["word"] = lab_in(head[1], x), p, w
            rec["other_words"] = sorted({f[4] for f in known if f[0] == "Eval" and f[1] == head[2]
                                         and f[2] == x})
        elif rule == "R4_OutcomeShift":
            key = (frozenset(head[1:3]), frozenset(head[3:5]))
            if key in seen_r4:
                continue                         # same difference seen from the other side
            seen_r4.add(key)
            t, u = head[3], head[4]
            rec["left_type"], rec["right_type"] = t, u
            left = [m for d in derivs for m in d["matched"] if m[0] == "Outcome" and m[1] == head[1]]
            right = [m for d in derivs for m in d["matched"] if m[0] == "Outcome" and m[1] == head[2]]
            rec["left_outcome"] = "; ".join(sorted({lab_in(head[1], m[2]) for m in left}))
            rec["right_outcome"] = "; ".join(sorted({lab_in(head[2], m[2]) for m in right}))
        results.append(rec)
    return results


def explain(rec, headlines=None):
    """One-paragraph plain-English explanation of a difference record."""
    h = headlines or {}
    a, b = rec["from"], rec["to"]
    if rec["rule"] == "R1_InfoOnlyIn":
        s = f"{a.upper()} reports \"{rec['item']}\"; {b.upper()}'s headline does not."
        if rec.get("status") == "headline_omission":
            s += f" {b.upper()}'s article does mention it: \"{rec['article_evidence']['sentence'][:160]}\""
    elif rec["rule"] == "R2_ActorShift":
        other = ", ".join(rec["other_actors"]) or "no one (actor not named)"
        s = (f"{a.upper()} shows {rec['actor']} doing \"{rec['action']}\"; "
             f"{b.upper()} reports the same action (\"{rec.get('other_action', rec['action'])}\") "
             f"but credits {other}.")
    elif rec["rule"] == "R3_EvalShift":
        s = (f"{a.upper()} uses the {rec['polarity']} word \"{rec['word']}\" about {rec['actor']}; "
             f"{b.upper()} mentions {rec['actor']} without that tone.")
    else:
        s = (f"{a.upper()} stresses a {rec['left_type']} outcome ({rec['left_outcome']}); "
             f"{b.upper()} stresses a {rec['right_type']} outcome ({rec['right_outcome']}).")
    return s


def run_story(story_id, extractions, alignments, stories, embedder=None):
    from src.facts import build_facts
    story = stories[story_id]
    arts = story["articles"]
    headlines = {s: arts[s]["headline"] for s in arts}
    texts = {s: [x["text"] for x in arts[s]["sentences"]] for s in arts}
    kb = build_facts(extractions[story_id], alignments[story_id])
    return kb, derive_differences(kb, headlines, texts, embedder=embedder), headlines


def main():
    import argparse
    import json
    import os
    from src.align import load_alignment
    from src.load import load_stories, stories_by_id

    ap = argparse.ArgumentParser(description="Stage S5: forward chaining with rules R1-R4")
    ap.add_argument("--story-id", help="explain one story in detail")
    ap.add_argument("--no-sbert", action="store_true",
                    help="use word overlap instead of Sentence-BERT for R1 article checks")
    ap.add_argument("--out", default=os.path.join("data", "differences", "differences.json"))
    args = ap.parse_args()

    with open(os.path.join("data", "extractions", "extractions.json"), encoding="utf-8") as f:
        ext = json.load(f)
    al = load_alignment()
    stories = stories_by_id(load_stories())
    emb = None
    if not args.no_sbert:
        try:
            from src.align import SBERTEmbedder
            emb = SBERTEmbedder()
        except Exception as e:                   # missing package: fall back, but say so
            print(f"(Sentence-BERT unavailable: {e}; using word overlap for R1 article checks)")

    ids = [args.story_id] if args.story_id else [s for s in ext if s in al]
    out = {}
    for sid in ids:
        kb, diffs, heads = run_story(sid, ext, al, stories, emb)
        out[sid] = diffs
        if args.story_id:
            for s, t in heads.items():
                print(f"{s}: {t}")
            print(f"\n{len(diffs)} differences:")
            for d in diffs:
                print(f"\n[{d['rule']}] {explain(d)}")
                for tr in d["trace"][:1]:
                    print(f"   rule:    {tr['rule_text']}")
                    print(f"   matched: {', '.join(m for m in tr['matched'] if not m.startswith('Pair'))}")
                    if tr["absent"]:
                        print(f"   absent:  {', '.join(tr['absent'])}")
        else:
            counts = {}
            for d in diffs:
                counts[d["rule"][:2]] = counts.get(d["rule"][:2], 0) + 1
            print(f"{sid[:8]}  {len(diffs):3d} differences  {counts}")
    if not args.story_id:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"\nSaved {sum(len(v) for v in out.values())} differences for {len(out)} stories -> {args.out}")


if __name__ == "__main__":
    main()
