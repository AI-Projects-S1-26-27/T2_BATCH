"""Story-level analysis for the demo: ties the pair-by-pair differences together.

Extra rules (run with the same forward-chaining engine as R1-R4):

  stratum 0   ReportedByOther(h,i) <= Pair(h,b) AND Reports(b,i)
  stratum 1   Exclusive(h,i)       <= Reports(h,i) AND NOT ReportedByOther(h,i)
              Omits(h,i)           <= Headline(h) AND ReportedByOther(h,i) AND NOT Reports(h,i)
              Missing(i)           <= Headline(h) AND Mentions(g,i) AND Reports(g,i) AND NOT Reports(h,i)
  stratum 2   Common(i)            <= Reports(h,i) AND NOT Missing(i)

Exclusive = information only one outlet gives; Omits = information the others give and this
one leaves out; Common = information every outlet reports (the shared core of the event).
Together with the Agent / Eval / Outcome facts and the derived differences, they give each
outlet a "framing profile" and a few plain-language takeaways. Everything is computed from
facts, so every statement can be traced back to headline words (no LLM involved here).
"""

from src.engine import RULES, Rule, forward_chain

SUMMARY_RULES = [
    Rule("S_reported_by_other", ("ReportedByOther", "?h", "?i"),
         [("Pair", "?h", "?b"), ("Reports", "?b", "?i")],
         text="Pair(h,b) AND Reports(b,i) => ReportedByOther(h,i)"),
    Rule("S_exclusive", ("Exclusive", "?h", "?i"), [("Reports", "?h", "?i")],
         neg=[("ReportedByOther", "?h", "?i")], stratum=1,
         text="Reports(h,i) AND NOT ReportedByOther(h,i) => Exclusive(h,i)"),
    Rule("S_omits", ("Omits", "?h", "?i"), [("Headline", "?h"), ("ReportedByOther", "?h", "?i")],
         neg=[("Reports", "?h", "?i")], stratum=1,
         text="ReportedByOther(h,i) AND NOT Reports(h,i) => Omits(h,i)"),
    Rule("S_missing", ("Missing", "?i"), [("Headline", "?h"), ("Reports", "?g", "?i")],
         neg=[("Reports", "?h", "?i")], stratum=1,
         text="Headline(h) AND Reports(g,i) AND NOT Reports(h,i) => Missing(i)"),
    Rule("S_common", ("Common", "?i"), [("Reports", "?h", "?i")],
         neg=[("Missing", "?i")], stratum=2,
         text="Reports(h,i) AND NOT Missing(i) => Common(i)"),
]

DIMS = ["D1_info", "D2_actor", "D3_eval", "D4_outcome"]


def build_summary(result, name=lambda s: s.upper()):
    """result: output of src.pipeline.analyze. name: source key -> display name.
    Returns {"common", "profiles", "matrix", "takeaways", "n_differences"} or None."""
    kb = result.get("kb")
    if not kb:
        return None
    heads = kb["headlines"]
    own, labels = kb.get("own_labels", {}), kb["labels"]

    def lab(h, x):
        return own.get(f"{h}:{x}") or labels.get(x, x)

    known, trace = forward_chain(set(kb["atoms"]), RULES + SUMMARY_RULES)
    facts = {}
    for f in known:
        facts.setdefault(f[0], []).append(f)

    common = sorted({f[1] for f in facts.get("Common", [])})
    profiles = {}
    for h in heads:
        profiles[h] = {
            "exclusive": sorted(lab(h, f[2]) for f in facts.get("Exclusive", []) if f[1] == h),
            "omits": sorted({lab(next(g for g in heads if ("Reports", g, f[2]) in known), f[2])
                             for f in facts.get("Omits", []) if f[1] == h}),
            "acting": sorted({f"{lab(h, f[2])} → {lab(h, f[3])}" for f in facts.get("Agent", []) if f[1] == h}),
            "loaded": sorted({(f[4], f[3]) for f in facts.get("Eval", []) if f[1] == h}),
            "outcomes": sorted({(f[3], lab(h, f[2])) for f in facts.get("Outcome", []) if f[1] == h}),
        }

    diffs = result.get("differences", [])
    matrix = {h: {d: sum(1 for r in diffs if r["from"] == h and r["dimension"] == d) for d in DIMS}
              for h in heads}

    take = []
    if common:
        take.append(f"All {len(heads)} headlines agree on the core event: "
                    + "; ".join(f"“{lab(heads[0], i)}”" for i in common) + ".")
    elif len(heads) > 1:
        take.append("No single piece of information is reported by every headline: "
                    "each outlet picks a different angle.")
    for h in heads:
        ex = profiles[h]["exclusive"]
        if ex and len(heads) > 2:
            take.append(f"Only {name(h)} reports " + ", ".join(f"“{x}”" for x in ex) + ".")
        elif ex:
            take.append(f"{name(h)} adds " + ", ".join(f"“{x}”" for x in ex)
                        + f", which the other headline leaves out.")
    # who is shown acting, per shared action (same item ID), straight from Agent facts
    reporters = {}
    for f in facts.get("Reports", []):
        reporters.setdefault(f[2], []).append(f[1])
    for i, hs in sorted(reporters.items()):
        if len(hs) < 2:
            continue
        who = {h: tuple(sorted(lab(h, f[2]) for f in facts.get("Agent", []) if f[1] == h and f[3] == i))
               for h in hs}
        if len(set(who.values())) < 2:
            continue                                    # everyone names the same actor(s)
        groups = {}
        for h in heads:
            if h in who:
                groups.setdefault(who[h], []).append(name(h))
        parts = [f"{' and '.join(v)}: {', '.join(k) if k else 'actor not named'}"
                 for k, v in groups.items()]
        take.append(f"Who is shown acting in “{lab(hs[0] if heads[0] not in hs else heads[0], i)}” "
                    "differs: " + "; ".join(parts) + ".")
    tone = [(h, w, p) for h in heads for w, p in profiles[h]["loaded"]]
    if tone:
        take.append("Loaded wording: " + "; ".join(f"{name(h)} uses “{w}” ({p})" for h, w, p in tone) + ".")
    types = {h: {t for t, _ in profiles[h]["outcomes"] if t != "Other"} for h in heads}
    if any(r["dimension"] == "D4_outcome" for r in diffs):
        take.append("Consequences stressed differ: " + "; ".join(
            f"{name(h)}: {', '.join(sorted(t))}" for h, t in types.items() if t) + ".")
    if diffs:
        counts = {d: sum(r["dimension"] == d for r in diffs) for d in DIMS}
        top = max(counts, key=counts.get)
        names_ = {"D1_info": "which information is included", "D2_actor": "who is shown acting",
                  "D3_eval": "loaded wording", "D4_outcome": "which consequences are stressed"}
        take.append(f"Overall, most differences ({counts[top]} of {len(diffs)}) concern {names_[top]}.")
    else:
        take.append("No framing differences were derived: the headlines report the same facts "
                    "in comparable terms (as far as the extracted facts show).")

    return {"common": [lab(heads[0], i) for i in common], "profiles": profiles, "matrix": matrix,
            "takeaways": take, "n_differences": len(diffs)}
