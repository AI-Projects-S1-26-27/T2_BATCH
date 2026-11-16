"""Tests for baselines B1-B3 (no model download or API key needed).
Run:  python -m tests.test_baselines
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from baselines import b1_tfidf, b2_sbert
from baselines.b3_llm import B3, parse
from baselines.common import chunks, is_evaluative

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = {"fox": "Police fire tear gas on peaceful protesters",
     "nyt": "Police fired tear gas as protest turns violent",
     "hpo": "Tear gas used on protesters"}


def test_b1():
    model = b1_tfidf.TfidfModel(list(H.values()) + ["Senate passes budget", "Police arrest man"])
    recs = b1_tfidf.compare(H, model)
    flagged = {(r["from"], r["to"], r["evidence"][0]["text"]) for r in recs}
    assert ("fox", "nyt", "peaceful protesters") in flagged
    assert not any(t.lower().startswith("fire") for a, b, t in flagged if (a, b) == ("fox", "nyt"))  # fire~fired
    for r in recs:                                           # offsets point at the words
        e = r["evidence"][0]
        assert H[r["from"]][e["start"]:e["end"]] == e["text"]
    dims = {r["evidence"][0]["text"]: r["dimension"] for r in recs if r["from"] == "nyt"}
    assert any("violent" in k and v == "D3_eval" for k, v in dims.items()), dims
    assert {r["dimension"] for r in recs} <= {"D1_info", "D3_eval"}


class WordEmbedder:
    """Bag-of-stems vectors (hashed into 256 slots): stand-in for Sentence-BERT."""
    def encode(self, texts):
        import zlib
        from baselines.common import stem, tokens
        m = np.zeros((len(texts), 256))
        for i, t in enumerate(texts):
            for w, _, _ in tokens(t):
                m[i, zlib.crc32(stem(w).encode()) % 256] = 1
        return m


def test_b2():
    recs = b2_sbert.compare(H, WordEmbedder(), threshold=0.5)
    flagged = {(r["from"], r["to"], r["evidence"][0]["text"]) for r in recs}
    assert ("nyt", "fox", "protest turns violent") in flagged
    assert ("fox", "nyt", "tear gas") not in flagged                   # same phrase in both
    assert all(r["system"] == "B2" and r["score"] < 0.5 for r in recs)


def test_chunks_and_lexicon():
    assert [c[0] for c in chunks("White House Used Bill Clinton to Ask Sestak")] == \
        ["White House Used", "Bill Clinton", "Ask Sestak"]
    assert is_evaluative("Obama slams GOP") and not is_evaluative("Senate passes bill")


class FakeLLM:
    model = "fake"

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def chat_json(self, system, user):
        self.calls += 1
        assert "Headline A" in user and "Article A" in user
        return self.reply, ""


def test_b3_parse_and_cache():
    reply = {"differences": [
        {"dimension": "evaluative", "headline": "A", "words": "peaceful", "explanation": "tone"},
        {"dimension": "information", "headline": "A", "words": "protest turns violent",
         "explanation": "cited the wrong side"},                       # actually in B
        {"dimension": "actor", "headline": "B", "words": "brutal cops", "explanation": "invented"},
        {"dimension": "vibes", "headline": "A", "words": "Police", "explanation": "bad dimension"}]}
    recs = parse(reply, "fox", "nyt", H)
    assert [r["dimension"] for r in recs] == ["D3_eval", "D1_info", "D2_actor"]
    assert recs[0]["evidence_found"] and recs[0]["from"] == "fox"
    assert recs[1]["from"] == "nyt" and recs[1]["evidence_found"]      # moved to the right side
    assert not recs[2]["evidence_found"]                               # kept, marked unsupported
    with tempfile.TemporaryDirectory() as d:
        fake = FakeLLM(reply)
        b3 = B3(client=fake, cache_path=os.path.join(d, "c.json"))
        out, errors = b3.compare(H, {s: ["Lead sentence."] for s in H})
        assert not errors and fake.calls == 3                          # one call per unordered pair
        b3.compare(H, {s: ["Lead sentence."] for s in H})
        assert fake.calls == 3                                         # cached


def test_run_b1_on_repo():
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, "-m", "baselines.run", "--systems", "B1", "--split", "dev"],
                       cwd=ROOT, check=True, capture_output=True, text=True)
        with open(os.path.join(ROOT, "data", "baselines", "B1.json"), encoding="utf-8") as f:
            res = json.load(f)
        with open(os.path.join(ROOT, "data", "splits.json")) as f:
            assert set(json.load(f)["dev"]) <= set(res)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All baseline checks passed.")
