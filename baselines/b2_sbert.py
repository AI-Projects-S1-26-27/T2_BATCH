"""B2 - Semantic baseline: Sentence-BERT similarity between headline phrases.

Question it answers (proposal 5.4): is semantic matching enough without structured
fields and rules?

Each headline is split into short phrases (runs of content words, at most 3 words).
A phrase of A is flagged when its best cosine similarity to any phrase of B is below
the threshold. Same model as S3 alignment (all-MiniLM-L6-v2) and, by default, the same
item threshold tuned on the dev stories, so B2 differs from our pipeline only in having
no structured fields and no rules (proposal: this isolates the effect of the rules).
"""

import numpy as np

from baselines.common import chunks, directed_pairs, record


def default_threshold():
    from src.align import load_thresholds
    return load_thresholds()["items"]


def compare(headlines, embedder, threshold=None):
    threshold = default_threshold() if threshold is None else threshold
    phrases = {s: chunks(t) for s, t in headlines.items() if t}
    vecs = {s: embedder.encode([p for p, _, _ in ph]) if ph else None for s, ph in phrases.items()}
    out = []
    for a, b in directed_pairs(headlines):
        if not phrases.get(a) or not phrases.get(b):
            continue
        va, vb = vecs[a], vecs[b]
        va = va / np.clip(np.linalg.norm(va, axis=1, keepdims=True), 1e-12, None)
        vb = vb / np.clip(np.linalg.norm(vb, axis=1, keepdims=True), 1e-12, None)
        sims = va @ vb.T
        for k, (text, s, e) in enumerate(phrases[a]):
            best = int(np.argmax(sims[k]))
            score = float(sims[k, best])
            if score < threshold:
                closest = phrases[b][best][0]
                out.append(record("B2", a, b, text, s, e,
                                  f"\"{text}\" in {a.upper()} has no close match in {b.upper()} "
                                  f"(closest: \"{closest}\", similarity {score:.2f} < {threshold}).",
                                  score))
    return out
