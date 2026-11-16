"""B1 - Lexical baseline: TF-IDF word overlap between headlines.

Question it answers (proposal 5.4): can word overlap alone find framing differences?

For headlines A and B: every content word of A whose stem does not occur in B is
"unmatched". Its TF-IDF weight (IDF learned from all 300 BASIL headlines) says how
distinctive it is; words below MIN_WEIGHT (very common words) are ignored. Adjacent
unmatched words are merged into one flagged phrase. Run in both directions.
"""

import math
from collections import Counter

from baselines.common import content_tokens, directed_pairs, group_spans, record, stem

MIN_WEIGHT = 1.5      # = skip words found in more than ~60% of BASIL headlines


class TfidfModel:
    def __init__(self, corpus):
        docs = [{stem(w) for w, _, _ in content_tokens(t)} for t in corpus]
        df = Counter(w for d in docs for w in d)
        n = len(docs)
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}
        self.default_idf = math.log(1 + n) + 1          # unseen word: maximally distinctive

    def weight(self, word, doc_tokens):
        tf = sum(stem(w) == stem(word) for w, _, _ in doc_tokens) / max(len(doc_tokens), 1)
        return tf * self.idf.get(stem(word), self.default_idf) * len(doc_tokens)   # = count * idf


def compare(headlines, model, min_weight=MIN_WEIGHT):
    """headlines: {source: text}. Returns difference records."""
    out = []
    for a, b in directed_pairs(headlines):
        ta, tb = content_tokens(headlines[a]), content_tokens(headlines[b])
        b_stems = {stem(w) for w, _, _ in tb}
        unmatched = [(w, s, e) for w, s, e in ta
                     if stem(w) not in b_stems and model.weight(w, ta) >= min_weight]
        for text, s, e in group_spans(unmatched, headlines[a]):
            score = max(model.weight(w, ta) for w, ws, we in unmatched if ws >= s and we <= e)
            out.append(record("B1", a, b, text, s, e,
                              f"\"{text}\" appears in {a.upper()} but no matching word appears in "
                              f"{b.upper()} (TF-IDF weight {score:.2f}).", score))
    return out
