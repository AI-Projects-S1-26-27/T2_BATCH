"""Export S4/S5 candidate differences for human review.

Examples:
    python scripts/export_review.py
    python scripts/export_review.py --split dev
    python scripts/export_review.py --split test
    python scripts/export_review.py --split all
"""

import argparse
import csv
import json
import os
import sys

# Make imports work when this is run as: python scripts/export_review.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.align import load_alignment
from src.engine import derive_differences, explain
from src.facts import build_facts
from src.load import load_stories, stories_by_id


def json_cell(value):
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("dev", "test", "annotated", "all"),
        default="annotated",
        help="Which stories to export. Default: dev + test stories.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "data", "review", "candidate_differences.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--no-sbert", action="store_true",
                        help="use word overlap instead of Sentence-BERT for R1 article checks")
    args = parser.parse_args()

    embedder = None
    if not args.no_sbert:
        try:
            from src.align import SBERTEmbedder
            embedder = SBERTEmbedder()
        except Exception as e:
            print(f"(Sentence-BERT unavailable: {e}; using word overlap)")

    data_dir = os.path.join(ROOT, "data")
    with open(os.path.join(data_dir, "extractions", "extractions.json"),
              encoding="utf-8") as f:
        extractions = json.load(f)

    alignments = load_alignment(
        os.path.join(data_dir, "alignment", "alignment.json")
    )
    stories = stories_by_id(
        load_stories(os.path.join(data_dir, "processed", "stories.json"))
    )

    split_ids = None
    if args.split != "all":
        with open(os.path.join(data_dir, "splits.json"), encoding="utf-8") as f:
            split_data = json.load(f)
        if args.split == "annotated":
            split_ids = set(split_data["dev"]) | set(split_data["test"])
        else:
            split_ids = set(split_data[args.split])

    columns = [
        "story_id", "year", "rule", "dimension", "from_outlet", "to_outlet",
        "from_headline", "to_headline", "item", "actor", "action",
        "other_actors", "word", "polarity",
        "left_outcome", "left_type", "right_outcome", "right_type",
        "status", "explanation", "headline_evidence", "article_evidence", "rule_trace",
        "review_verdict", "review_note",
    ]

    rows_written = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for story_id, story_extractions in extractions.items():
            if split_ids is not None and story_id not in split_ids:
                continue
            if story_id not in stories or story_id not in alignments:
                continue

            story = stories[story_id]
            articles = story["articles"]
            headlines = {
                src: articles[src]["headline"]
                for src in ("fox", "nyt", "hpo")
            }
            article_texts = {
                src: [sentence["text"]
                      for sentence in articles[src].get("sentences", [])]
                for src in ("fox", "nyt", "hpo")
            }

            facts = build_facts(story_extractions, alignments[story_id])
            differences = derive_differences(facts, headlines, article_texts,
                                             embedder=embedder)

            for difference in differences:
                source = difference.get("from", "")
                target = difference.get("to", "")
                writer.writerow({
                    "story_id": story_id,
                    "year": story.get("year", ""),
                    "rule": difference.get("rule", ""),
                    "dimension": difference.get("dimension", ""),
                    "from_outlet": source,
                    "to_outlet": target,
                    "from_headline": headlines.get(source, ""),
                    "to_headline": headlines.get(target, ""),
                    "item": difference.get("item", ""),
                    "actor": difference.get("actor", ""),
                    "action": difference.get("action", ""),
                    "other_actors": json_cell(difference.get("other_actors")),
                    "word": difference.get("word", ""),
                    "polarity": difference.get("polarity", ""),
                    "left_outcome": difference.get("left_outcome", ""),
                    "left_type": difference.get("left_type", ""),
                    "right_outcome": difference.get("right_outcome", ""),
                    "right_type": difference.get("right_type", ""),
                    "status": difference.get("status", ""),
                    "explanation": explain(difference),
                    "headline_evidence": json_cell(difference.get("evidence")),
                    "article_evidence": json_cell(difference.get("article_evidence")),
                    "rule_trace": json_cell(difference.get("trace")),
                    "review_verdict": "",
                    "review_note": "",
                })
                rows_written += 1

    print(f"Exported {rows_written} candidate differences to: {args.output}")


if __name__ == "__main__":
    main()
