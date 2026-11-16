"""Tests for retrieval and the end-to-end demo pipeline (no API key or model download).
Run:  python -m tests.test_pipeline
"""

import zlib

import numpy as np

from src.load import load_stories
from src.pipeline import Tools, analyze, by_pair, clean_name
from src.retrieve import Retriever


class HashEmbedder:
    """Bag-of-words vectors hashed into 512 slots: stand-in for Sentence-BERT."""
    def encode(self, texts):
        m = np.zeros((len(texts), 512))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                m[i, zlib.crc32(w.strip(".,'\"").encode()) % 512] += 1
        return m


class FakeExtractor:
    """Returns canned S2 results; anything unknown fails like a real API error."""
    DATA = {
        "Police fire tear gas on peaceful protesters": {
            "actors": [("police", "LawEnforcement"), ("protesters", "Activist")],
            "items": ["fire tear gas"], "agents": [("police", "fire tear gas")],
            "evals": [("protesters", "positive", "peaceful")], "outcomes": []},
        "Tear gas fired as protest turns violent": {
            "actors": [("protest", "Activist")],
            "items": ["tear gas fired", "protest turns violent"], "agents": [],
            "evals": [("protest turns violent", "negative", "violent")], "outcomes": []},
    }

    def extract(self, headline):
        d = self.DATA.get(headline)
        if d is None:
            return {"ok": False, "headline": headline, "error": "Extraction failed: HTTP 401: invalid key"}

        def sp(t):
            s = headline.lower().find(t.lower())
            return {"span": headline[s:s + len(t)], "start": s, "end": s + len(t)}
        return {"ok": True, "headline": headline, "fields": {
            "actors": [{"name": n, "type": t, **sp(n)} for n, t in d["actors"]],
            "items": [{"text": t, **sp(t.split()[-1])} for t in d["items"]],
            "agents": [{"actor": a, "action": i, **sp(i.split()[-1])} for a, i in d["agents"]],
            "evals": [{"target": x, "polarity": p, "word": w, **sp(w)} for x, p, w in d["evals"]],
            "outcomes": []}}

    def save(self):
        pass


def tools():
    from baselines.run import load_b1_model
    return Tools(extractor=FakeExtractor(), embedder=HashEmbedder(), b1_model=load_b1_model(),
                 thresholds={"items": 0.5, "actors": 0.75, "outcomes": 0.7})


A = "Police fire tear gas on peaceful protesters"
B = "Tear gas fired as protest turns violent"


def test_retrieval():
    r = Retriever(load_stories())
    hit = r.search("Senate overrides Obama veto of 9/11 victims bill")
    assert hit["match"] and hit["results"][0]["story_id"].startswith("1b96a5db")
    assert {"obama", "veto"} <= set(hit["results"][0]["shared_words"])
    miss = r.search("Apple unveils new iPhone with a better camera")
    assert not miss["match"]
    assert not r.search("")["match"]


def test_pipeline_success():
    res = analyze({"fox": A, "you": B}, tools(), articles={"fox": ["Police used tear gas."]})
    assert res["ok"] and not res["errors"], res["errors"]
    assert res["pairs"] == [("fox", "you")]                       # fox first, then user source
    rules = {d["rule"] for d in res["differences"]}
    assert "R2_ActorShift" in rules and "R1_InfoOnlyIn" in rules
    assert {"B1", "B2"} <= set(res["baselines"])
    g = by_pair(res["differences"], res["pairs"])
    assert len(g[frozenset(("fox", "you"))]) == len(res["differences"])
    r1_you = [d for d in res["differences"] if d["rule"] == "R1_InfoOnlyIn" and d["to"] == "you"]
    assert all(d["status"] == "article_not_available" for d in r1_you)


def test_pipeline_failures():
    one = analyze({"fox": A}, tools())
    assert not one["ok"] and "at least two" in one["message"]
    bad = analyze({"fox": A, "nyt": "Unknown headline"}, tools())
    assert not bad["ok"] and "Could not extract" in bad["message"]
    assert any("401" in e for e in bad["errors"])
    assert "B1" in bad["baselines"]                               # baselines still shown

    class Broken(FakeExtractor):
        def extract(self, h):
            raise RuntimeError("no key")
    t = tools()
    t.extractor = Broken()
    res = analyze({"a": A, "b": B}, t)                            # never raises
    assert not res["ok"] and all("unavailable" in e for e in res["errors"][:2])


def test_cached_extraction_used():
    ext = {"fox": FakeExtractor().extract(A), "nyt": FakeExtractor().extract(B)}
    t = tools()
    t.extractor = None                                            # would crash if called
    res = analyze({"fox": A, "nyt": B}, t, extractions=ext, run_baselines=False)
    assert res["ok"] and res["differences"]


def test_clean_name():
    used = set()
    k1 = clean_name("The Guardian", used)
    used.add(k1)
    assert k1 == "guardian" and clean_name("Guardian", used) != "guardian"
    assert clean_name("", set()) == "headline"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All pipeline checks passed.")
