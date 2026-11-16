"""Shared helpers for the three baselines (M7).

Every baseline returns a list of difference records in the same shape as the main system
(src/engine.py), so eval/ can score all systems the same way:

    {"system": "B1", "dimension": "D1_info" | "D2_actor" | "D3_eval" | "D4_outcome",
     "from": "fox", "to": "nyt",            # the headline that HAS the flagged words
     "evidence": [{"source", "text", "start", "end"}],
     "explanation": "...", "score": float}

B1 and B2 only find "words/phrases in one headline with no counterpart in the other".
They have no notion of actors, tone or outcomes, so for scoring each flagged unit is
given a dimension by one transparent rule:
    contains a word from the small evaluative lexicon below  -> D3_eval
    otherwise                                                 -> D1_info
They never predict D2 or D4 (they cannot tell who acts or what kind of consequence).
"""

import re
from itertools import permutations

from src.load import source_order

STOPWORDS = set("""a an the and or but if of to in on at by for with from as into over after before
about against between during under up down out off than then so not no nor is are was were be been
being am do does did has have had having will would can could should may might must shall this that
these those it its he she they them his her their our we you your i me my who whom which what when
where why how all any both each few more most other some such only own same too very s t just
says said say amid via vs new""".split())

# Small hand-made lexicon of evaluative / loaded headline words (stems), used only to map
# B1/B2 output to D3. Kept short and generic on purpose; documented in the report.
EVALUATIVE = set("""slam blast rip attack bash lash slap scold mock taunt snub smear trash torch
outrage outrag furious fury rage angry anger fume seeth livid shock shocking stun stunn
scandal disgrace shame shameful chaos chaotic crisis disaster debacl fiasco mess meltdown
radical extreme extremist militant hardlin fringe controversial divisive toxic
lie liar false bogus fake sham hoax rigged corrupt cronies
defy defiant cave caves capitul surrender retreat flip flop backtrack dodg evade
hail prais champion landmark historic victory triumph bold brave heroic peaceful
celebrat welcom applaud thank laud tout boost
threat threaten danger dangerous alarm alarming fear fearmonger panic warn
brutal violent vicious deadly bloody killer savage
slash gut hike soar plunge tumble crash surge spike
blame accuse allege deny insist admit concede exploit""".split())


def stem(word):
    w = word.lower().strip("'’")
    for suf in ("ing", "ed", "es", "s"):
        if len(w) - len(suf) >= 2 and w.endswith(suf) and not w.endswith("eed"):
            w = w[: -len(suf)]
            break
    return w[:-1] if len(w) > 2 and w.endswith("e") else w      # fire / fired -> fir


def tokens(text):
    """Words with their character offsets: [(word, start, end)]."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9'’.\-]*[A-Za-z0-9]|[A-Za-z0-9]", text)]


def content_tokens(text):
    return [(w, s, e) for w, s, e in tokens(text) if w.lower() not in STOPWORDS]


def is_evaluative(text):
    return any(stem(w) in EVALUATIVE or w.lower() in EVALUATIVE for w, _, _ in tokens(text))


def group_spans(tok_list, text, max_gap=1):
    """Merge flagged tokens that are next to each other (at most max_gap words apart in the
    headline) into phrases: [(phrase_text, start, end)]."""
    all_tok = tokens(text)
    pos = {(s, e): k for k, (_, s, e) in enumerate(all_tok)}
    idx = sorted(pos[(s, e)] for _, s, e in tok_list if (s, e) in pos)
    groups, cur = [], []
    for k in idx:
        if cur and k - cur[-1] > max_gap:
            groups.append(cur)
            cur = []
        cur.append(k)
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        s, e = all_tok[g[0]][1], all_tok[g[-1]][2]
        out.append((text[s:e], s, e))
    return out


def chunks(text, max_words=3):
    """Phrase units for B2: runs of content words between stopwords/punctuation (RAKE-style),
    split evenly into pieces of at most max_words words."""
    all_tok = tokens(text)
    runs, cur = [], []
    for k, (w, s, e) in enumerate(all_tok):
        gap = cur and re.search(r"[,;:!?|\u2014\u2013\"]|\s-\s", text[all_tok[cur[-1]][2]:s])
        if w.lower() in STOPWORDS or gap:
            if cur:
                runs.append(cur)
            cur = [] if w.lower() in STOPWORDS else [k]
        else:
            cur.append(k)
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        n_pieces = -(-len(r) // max_words)              # ceil: split long runs evenly
        size = -(-len(r) // n_pieces)
        for i in range(0, len(r), size):
            piece = r[i:i + size]
            s, e = all_tok[piece[0]][1], all_tok[piece[-1]][2]
            out.append((text[s:e], s, e))
    return out


def record(system, a, b, text, start, end, explanation, score=None, dimension=None):
    return {"system": system,
            "dimension": dimension or ("D3_eval" if is_evaluative(text) else "D1_info"),
            "from": a, "to": b,
            "evidence": [{"source": a, "text": text, "start": start, "end": end}],
            "explanation": explanation, "score": None if score is None else round(float(score), 3)}


def directed_pairs(headlines):
    """All ordered pairs of available headlines, in the fixed source order."""
    srcs = [s for s in source_order(headlines) if headlines.get(s)]
    return list(permutations(srcs, 2))
