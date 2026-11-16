"""B3 - Direct LLM baseline: the same LLM as extraction (src/llm.py) compares two
headlines directly and lists framing differences with cited words.

Question it answers (proposal 5.4): do structure and explicit rules add value over
direct prompting?

The model sees both headlines plus the first sentences of both articles, and returns
differences along the same four dimensions as our system. The cited words are checked
against the headlines: "evidence_found" records whether the citation is real, which is
what the evidence-support comparison looks at. Results are cached like extraction.
"""

import hashlib
import json
import os

from baselines.common import record
from src.extract import find_span

PROMPT_VERSION = "b3-v1"
CACHE_PATH = os.path.join("data", "baselines", "b3_cache.json")
LEAD_SENTENCES = 4
DIMENSIONS = {"information": "D1_info", "actor": "D2_actor", "evaluative": "D3_eval",
              "consequence": "D4_outcome"}

SYSTEM_PROMPT = """You compare two news headlines (A and B) that report the SAME event, and list
how their FRAMING differs. Use only these four dimensions:

"information": one headline includes a piece of information the other leaves out.
"actor":       the same action is attributed to a different actor, or one headline hides the actor.
"evaluative":  one headline uses loaded or judgmental wording the other does not.
"consequence": the headlines stress different types of consequences (economic, legal, safety, ...).

Return ONLY a JSON object: {"differences": [ ... ]}, each item:
{"dimension": one of "information" | "actor" | "evaluative" | "consequence",
 "headline": "A" or "B"   (the headline that contains the cited words),
 "words": exact words copied from that headline,
 "explanation": one short sentence}
List only real framing differences; paraphrases of the same fact are not differences.
Return {"differences": []} if there are none. The article excerpts are context only:
always cite words from the HEADLINES."""


def _lead(sentences, n=LEAD_SENTENCES):
    return " ".join(s.strip() for s in (sentences or [])[:n])


def _prompt(ha, hb, la, lb):
    return (f'Headline A: "{ha}"\nHeadline B: "{hb}"\n\n'
            f"Article A (start): {la or '(not available)'}\n\n"
            f"Article B (start): {lb or '(not available)'}")


class B3:
    def __init__(self, client=None, cache_path=CACHE_PATH):
        self._client = client
        self.cache_path = cache_path
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                self.cache = json.load(f)

    @property
    def client(self):
        if self._client is None:
            from src.llm import LLMClient
            self._client = LLMClient()
        return self._client

    def _model(self):
        if self._client is not None:
            return self._client.model
        from src.llm import DEFAULT_MODEL, load_dotenv
        load_dotenv()
        return os.environ.get("LLM_MODEL") or DEFAULT_MODEL

    def save(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.cache_path)

    def ask(self, ha, hb, la="", lb=""):
        """Raw model answer for one unordered pair (cached). Returns (data, error)."""
        prompt = _prompt(ha, hb, la, lb)
        key = hashlib.sha1(f"{PROMPT_VERSION}|{self._model()}|{prompt}".encode()).hexdigest()
        if key in self.cache:
            c = self.cache[key]
            return c["data"], c["error"]
        from src.llm import LLMError
        try:
            data, _ = self.client.chat_json(SYSTEM_PROMPT, prompt)
            err = None if isinstance(data, dict) else "not a JSON object"
        except LLMError as e:
            data, err = None, str(e)
        if err is None:
            self.cache[key] = {"data": data, "error": None}
        return data, err

    def compare(self, headlines, leads=None):
        """headlines/leads: {source: text / list of sentences}. One call per unordered pair."""
        leads = leads or {}
        from src.load import source_order
        srcs = [s for s in source_order(headlines) if headlines.get(s)]
        out, errors = [], []
        for i, a in enumerate(srcs):
            for b in srcs[i + 1:]:
                data, err = self.ask(headlines[a], headlines[b], _lead(leads.get(a)), _lead(leads.get(b)))
                if err:
                    errors.append(f"{a}-{b}: {err}")
                    continue
                out += parse(data, a, b, headlines)
        return out, errors


def parse(data, a, b, headlines):
    """Turn the model's JSON into difference records. Unknown dimensions are dropped;
    citations not found in the headline are KEPT but marked evidence_found=False."""
    recs = []
    for d in (data or {}).get("differences", []) if isinstance(data, dict) else []:
        if not isinstance(d, dict):
            continue
        dim = DIMENSIONS.get(str(d.get("dimension", "")).strip().lower())
        if dim is None:
            continue
        side = str(d.get("headline", "")).strip().upper()
        src, other = (a, b) if side != "B" else (b, a)
        words = str(d.get("words", "")).strip()
        off = find_span(headlines[src], words)
        if off is None:                              # model may have cited the other headline
            alt = find_span(headlines[other], words)
            if alt is not None:
                src, other, off = other, src, alt
        r = record("B3", src, other, headlines[src][off[0]:off[1]] if off else words,
                   off[0] if off else None, off[1] if off else None,
                   str(d.get("explanation", "")).strip(), dimension=dim)
        r["evidence_found"] = off is not None
        recs.append(r)
    return recs
