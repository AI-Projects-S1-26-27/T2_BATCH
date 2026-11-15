"""Stage S4: turn S2 extractions and S3 alignment into grounded facts.

Usage:
    facts = build_facts(extractions_for_one_story, alignment_for_one_story)

The returned dictionaries are JSON serializable. IDs are local to one story;
always call this once per story, not across the whole corpus.
"""

from __future__ import annotations

import re


SOURCES = ("fox", "nyt", "hpo")


def _norm(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _mentions_by_index(alignment, kind, source):
    """Map the extraction list index to the S3 shared ID."""
    return {
        int(m["index"]): m["id"]
        for m in alignment.get("mentions", {}).get(kind, [])
        if m.get("src") == source and "index" in m and "id" in m
    }


def _id_for_label(fields, index_map, field, label):
    wanted = _norm(label)
    for i, entry in enumerate(fields):
        if _norm(entry.get(field)) == wanted and i in index_map:
            return index_map[i]
    return None


def build_facts(extractions, alignment):
    """Build per-headline facts from {source: extraction_result} plus S3 output.

    Each evidence object preserves the exact quoted headline span and offsets
    emitted by S2. Entries without usable extraction/alignment are skipped.
    """
    facts = {"actors": [], "reports": [], "agents": [], "evals": [], "outcomes": []}
    for src in SOURCES:
        result = extractions.get(src) or {}
        if not result.get("ok") or not isinstance(result.get("fields"), dict):
            continue
        fields = result["fields"]
        actor_ids = _mentions_by_index(alignment, "actors", src)
        item_ids = _mentions_by_index(alignment, "items", src)
        outcome_ids = _mentions_by_index(alignment, "outcomes", src)

        for i, actor in enumerate(fields.get("actors", [])):
            if i in actor_ids:
                facts["actors"].append({"h": src, "x": actor_ids[i],
                    "name": actor.get("name"), "type": actor.get("type"),
                    "evidence": _evidence(src, actor)})

        for i, item in enumerate(fields.get("items", [])):
            if i in item_ids:
                facts["reports"].append({"h": src, "i": item_ids[i],
                    "text": item.get("text"), "evidence": _evidence(src, item)})

        for agent in fields.get("agents", []):
            actor_id = _id_for_label(fields.get("actors", []), actor_ids, "name", agent.get("actor"))
            item_id = _id_for_label(fields.get("items", []), item_ids, "text", agent.get("action"))
            if actor_id and item_id:
                facts["agents"].append({"h": src, "x": actor_id, "i": item_id,
                    "actor": agent.get("actor"), "action": agent.get("action"),
                    "evidence": _evidence(src, agent)})

        for ev in fields.get("evals", []):
            target_id = (_id_for_label(fields.get("actors", []), actor_ids, "name", ev.get("target"))
                         or _id_for_label(fields.get("items", []), item_ids, "text", ev.get("target")))
            facts["evals"].append({"h": src, "x": target_id,
                "target": ev.get("target"), "polarity": ev.get("polarity"),
                "word": ev.get("word"), "evidence": _evidence(src, ev)})

        for i, outcome in enumerate(fields.get("outcomes", [])):
            if i in outcome_ids:
                facts["outcomes"].append({"h": src, "k": outcome_ids[i],
                    "text": outcome.get("text"), "type": outcome.get("type"),
                    "evidence": _evidence(src, outcome)})

    return facts


def _evidence(source, entry):
    return {"source": source, "text": entry.get("span") or entry.get("word"),
            "start": entry.get("start"), "end": entry.get("end")}


def main():
    """Run S4 over the saved extraction and alignment outputs."""
    import json
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.align import load_alignment
    from src.load import load_stories

    extraction_path = os.path.join(root, "data", "extractions", "extractions.json")
    with open(extraction_path, encoding="utf-8") as f:
        extractions = json.load(f)
    alignments = load_alignment(os.path.join(root, "data", "alignment", "alignment.json"))
    print("S4: building grounded facts")
    for story_id, story_extractions in extractions.items():
        alignment = alignments.get(story_id)
        if not alignment:
            continue
        facts = build_facts(story_extractions, alignment)
        counts = ", ".join(f"{name}={len(rows)}" for name, rows in facts.items())
        print(f"{story_id}: {counts}")


if __name__ == "__main__":
    main()
