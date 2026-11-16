"""Same-event retrieval (S1, demo): match a new headline to the stored BASIL headlines.

TF-IDF (word unigrams + bigrams, stop words removed) over all 300 BASIL articles, each
represented by its headline plus its first LEAD_SENTENCES sentences (headlines alone are
too short: one shared word such as "new" dominated the score). A story's score is the
best cosine similarity between the query headline and any of its three articles.
A match also needs at least MIN_SHARED meaningful words in common with that story.
Otherwise the demo answers "no same-event match" instead of comparing unrelated
headlines (proposal: graceful failure case).

The cutoff is chosen on the dev + unused stories (never the 14 test stories):
    python -m src.retrieve --tune
For every headline we record (a) its score against its OWN story's other two headlines
(should be high) and (b) its best score against all OTHER stories (no true match: should
be low). The cutoff that best separates the two is saved to data/retrieval_params.json.
"""

import argparse
import json
import os
import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from src.load import SOURCES, load_stories

PARAMS_PATH = os.path.join("data", "retrieval_params.json")
DEFAULT_CUTOFF = 0.11
LEAD_SENTENCES = 4
MIN_SHARED = 2
STOP = sorted(set(ENGLISH_STOP_WORDS) | {"says", "said", "new", "news", "report", "reports",
                                          "year", "years", "day", "week", "time", "just"})


def content_words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2}


class Retriever:
    def __init__(self, stories, cutoff=None):
        self.stories = stories
        self.rows = [(s["story_id"], src,
                      s["articles"][src]["headline"] + " " +
                      " ".join(x["text"] for x in s["articles"][src]["sentences"][:LEAD_SENTENCES]))
                     for s in stories for src in SOURCES]
        self.words = {}
        for sid, _, text in self.rows:
            self.words.setdefault(sid, set()).update(content_words(text))
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=STOP,
                                   sublinear_tf=True, lowercase=True)
        self.matrix = self.vec.fit_transform([r[2] for r in self.rows])
        self.cutoff = load_cutoff() if cutoff is None else cutoff

    def scores(self, headline, exclude_story=None, exclude_row=None):
        q = self.vec.transform([headline])
        sims = (self.matrix @ q.T).toarray().ravel()
        best = {}
        for k, (sid, src, _) in enumerate(self.rows):
            if sid == exclude_story or k == exclude_row:
                continue
            if sims[k] > best.get(sid, (-1, None))[0]:
                best[sid] = (float(sims[k]), src)
        return sorted(((s, sid, src) for sid, (s, src) in best.items()), reverse=True)

    def search(self, headline, k=3):
        """Returns {"match": bool, "results": [{story_id, score, best_source, main_event, headlines}]}."""
        ranked = self.scores(headline)[:k]
        by_id = {s["story_id"]: s for s in self.stories}
        q = content_words(headline)
        results = [{"story_id": sid, "score": round(score, 3), "best_source": src,
                    "shared_words": sorted(q & self.words[sid]),
                    "main_event": by_id[sid].get("main_event", ""),
                    "headlines": {s: by_id[sid]["articles"][s]["headline"] for s in SOURCES}}
                   for score, sid, src in ranked]
        return {"match": bool(results) and results[0]["score"] >= self.cutoff
                and len(results[0]["shared_words"]) >= MIN_SHARED,
                "cutoff": self.cutoff, "results": results}


def load_cutoff():
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            return json.load(f)["cutoff"]
    return DEFAULT_CUTOFF


def tune(stories, test_ids):
    r = Retriever(stories, cutoff=0)
    pos, neg, top1 = [], [], 0
    for k, (sid, src, text) in enumerate(r.rows):
        if sid in test_ids:
            continue
        query = next(x["articles"][src]["headline"] for x in stories if x["story_id"] == sid)
        own = r.scores(query, exclude_row=k)
        if own and own[0][1] == sid:
            top1 += 1
        q = content_words(query)
        pos.append(max(s for s, x, _ in own if x == sid) if len(q & r.words[sid]) >= MIN_SHARED else 0.0)
        best_other = r.scores(query, exclude_story=sid)[0]
        neg.append(best_other[0] if len(q & r.words[best_other[1]]) >= MIN_SHARED else 0.0)
    pos, neg = np.array(pos), np.array(neg)
    best = max(np.arange(0.05, 0.6, 0.01),
               key=lambda c: ((pos >= c).mean() + (neg < c).mean()) / 2)
    best = round(float(best), 2)
    stats = {"cutoff": best, "headlines": len(pos),
             "top1_accuracy": round(top1 / len(pos), 3),
             "same_event_above_cutoff": round(float((pos >= best).mean()), 3),
             "no_match_below_cutoff": round(float((neg < best).mean()), 3),
             "tuned_on": "dev + unused stories (test excluded)"}
    with open(PARAMS_PATH, "w") as f:
        json.dump(stats, f, indent=1)
    return stats


def main():
    ap = argparse.ArgumentParser(description="Same-event retrieval")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--query", help="headline to search for")
    args = ap.parse_args()
    stories = load_stories()
    if args.tune:
        with open(os.path.join("data", "splits.json")) as f:
            test_ids = set(json.load(f)["test"])
        for k, v in tune(stories, test_ids).items():
            print(f"{k:26s} {v}")
    if args.query:
        res = Retriever(stories).search(args.query)
        print(f"match: {res['match']} (cutoff {res['cutoff']})")
        for x in res["results"]:
            print(f"  {x['score']:.3f}  {x['story_id'][:8]}  {x['headlines'][x['best_source']]}"
                  f"   shared: {', '.join(x['shared_words'])}")


if __name__ == "__main__":
    main()
