"""End-to-end pipeline for any set of same-event headlines (used by the demo, app.py).

    result = analyze({"fox": "...", "nyt": "...", "you": "..."}, tools)

Runs S2 extraction -> S3 alignment -> S4 facts -> S5 forward chaining, plus the three
baselines, and NEVER raises: every problem becomes a readable message in
result["errors"] / result["message"] (proposal goal G3: fail gracefully).

`tools` bundles the heavy objects so the app can load them once:
    Tools(extractor, embedder, b1_model, b3=None, thresholds=None)
"""

from dataclasses import dataclass, field
from itertools import combinations

from src.load import source_order


@dataclass
class Tools:
    extractor: object                 # src.extract.Extractor
    embedder: object                  # src.align.SBERTEmbedder (or any .encode())
    b1_model: object = None           # baselines.b1_tfidf.TfidfModel
    b3: object = None                 # baselines.b3_llm.B3 (None = skip B3)
    thresholds: dict = field(default_factory=dict)


def clean_name(name, used):
    """Turn a user-typed outlet name into a short source key ('The Guardian' -> 'guardian')."""
    import re
    base = re.sub(r"[^a-z0-9]+", "", (name or "").lower().replace("the ", "")) or "headline"
    key, n = base[:24], 2
    while key in used:
        key, n = f"{base[:22]}{n}", n + 1
    return key


def analyze(headlines, tools, articles=None, extractions=None, run_baselines=True):
    """headlines:   {source: headline text}
    articles:    {source: list of sentences} (optional, used for R1 provenance and B3)
    extractions: {source: cached S2 result} (optional; missing ones are extracted live)"""
    from src.align import align_story
    from src.engine import derive_differences
    from src.facts import build_facts

    heads = {s: " ".join((t or "").split()) for s, t in headlines.items()}
    heads = {s: heads[s] for s in source_order(heads) if heads[s]}
    articles = articles or {}
    out = {"ok": False, "headlines": heads, "errors": [], "message": "", "extractions": {},
           "alignment": None, "kb": None, "differences": [], "baselines": {},
           "pairs": [tuple(p) for p in combinations(list(heads), 2)]}
    if len(heads) < 2:
        out["message"] = "Please give at least two headlines about the same event."
        return out

    # S2: extraction (cached or live)
    for src, text in heads.items():
        cached = (extractions or {}).get(src)
        if cached and cached.get("ok") and cached.get("headline", text) == text:
            res = cached
        else:
            try:
                res = tools.extractor.extract(text)
            except Exception as e:                    # e.g. no API key configured
                res = {"ok": False, "error": f"Extraction unavailable: {e}"}
        out["extractions"][src] = res
        if not res.get("ok"):
            out["errors"].append(f"{src.upper()}: {res.get('error') or 'extraction failed'}")
    try:
        tools.extractor.save()
    except Exception:
        pass
    usable = [s for s, r in out["extractions"].items() if r.get("ok")]
    if len(usable) < 2:
        out["message"] = ("Could not extract facts from enough headlines to compare them. "
                          "The rule-based comparison needs at least two; baseline results are "
                          "shown where available.")
    else:
        # S3-S5
        try:
            al = align_story(out["extractions"], tools.embedder, tools.thresholds or None)
            kb = build_facts(out["extractions"], al)
            diffs = derive_differences(kb, heads, articles, embedder=tools.embedder)
            out.update(alignment=al, kb=kb, differences=diffs, ok=True)
            if not diffs:
                out["message"] = "No framing differences were derived under rules R1-R4."
        except Exception as e:
            out["message"] = f"The reasoning step failed: {type(e).__name__}: {e}"

    if run_baselines:
        out["baselines"] = run_all_baselines(heads, articles, tools, out["errors"])
    return out


def run_all_baselines(heads, articles, tools, errors):
    res = {}
    try:
        if tools.b1_model is not None:
            from baselines import b1_tfidf
            res["B1"] = b1_tfidf.compare(heads, tools.b1_model)
    except Exception as e:
        errors.append(f"B1 failed: {e}")
    try:
        from baselines import b2_sbert
        res["B2"] = b2_sbert.compare(heads, tools.embedder,
                                     (tools.thresholds or {}).get("items"))
    except Exception as e:
        errors.append(f"B2 failed: {e}")
    if tools.b3 is not None:
        try:
            recs, errs = tools.b3.compare(heads, articles)
            tools.b3.save()
            res["B3"] = recs
            errors += [f"B3: {x}" for x in errs]
        except Exception as e:
            errors.append(f"B3 unavailable: {e}")
    return res


def by_pair(records, pairs):
    """Group difference records by unordered headline pair, in the order of `pairs`."""
    out = {frozenset(p): [] for p in pairs}
    for r in records or []:
        k = frozenset((r["from"], r["to"]))
        if k in out:
            out[k].append(r)
    return out
