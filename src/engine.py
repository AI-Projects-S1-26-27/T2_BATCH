"""Stage S5: forward-chaining framing rules R1-R4.

Usage:
    differences = derive_differences(facts, headlines, article_texts, ontology)

headlines/article_texts are {source: text}. ontology maps a subtype to its
parent(s), for example {"Minister": ["GovernmentOfficial"]}. Output preserves
the rule, matched fact IDs, direction, and supporting evidence for S6.
"""

from __future__ import annotations

from itertools import combinations


SOURCES = ("fox", "nyt", "hpo")


def _by_headline(facts, key):
    result = {}
    for fact in facts.get(key, []):
        result.setdefault(fact.get("h"), []).append(fact)
    return result


def _article_quote(text, phrase):
    """Return a supporting article excerpt if the phrase occurs in the text."""
    if not text or not phrase:
        return None
    start = text.casefold().find(str(phrase).casefold())
    if start < 0:
        return None
    end = start + len(str(phrase))
    left, right = max(0, start - 100), min(len(text), end + 100)
    return {"text": text[left:right], "start": start, "end": end}


def _is_a(child, parent, ontology):
    """Transitive subtype check for the small S4 category hierarchy."""
    if child == parent:
        return True
    todo, seen = list(ontology.get(child, [])), set()
    while todo:
        current = todo.pop()
        if current == parent:
            return True
        if current not in seen:
            seen.add(current)
            todo.extend(ontology.get(current, []))
    return False


def derive_differences(facts, headlines=None, article_texts=None, ontology=None):
    """Apply the proposal's four rules to each pair of usable headlines.

    R1 uses article text to distinguish a headline-level omission from an
    item that was not found in the other article. R2 detects changed/hidden
    agents, R3 detects evaluative wording about a shared actor, and R4 detects
    different incomparable outcome categories.
    """
    headlines = headlines or {}
    article_texts = article_texts or {}
    ontology = ontology or {}
    groups = {name: _by_headline(facts, name) for name in
              ("actors", "reports", "agents", "evals", "outcomes")}
    usable = [s for s in SOURCES if s in headlines or any(s in g for g in groups.values())]
    differences = []
    seen = set()

    def emit(rule, a, b, key, evidence, **details):
        # A deterministic key prevents duplicate derivations during chaining.
        marker = (rule, a, b, key, tuple((e or {}).get("source") for e in evidence))
        if marker in seen:
            return False
        seen.add(marker)
        differences.append({"rule": rule, "from": a, "to": b,
            "facts": list(key), "evidence": [e for e in evidence if e], **details})
        return True

    # The rules have no recursive premises, but iterate to a fixed point so
    # derived explanations can be extended safely if new rules are added.
    changed = True
    while changed:
        before = len(differences)
        for a, b in combinations(usable, 2):
            reports_a, reports_b = groups["reports"].get(a, []), groups["reports"].get(b, [])
            ids_a = {f["i"] for f in reports_a}
            ids_b = {f["i"] for f in reports_b}

            # R1 — information included in one headline, absent from the other.
            for source, target, source_reports, target_ids in (
                (a, b, reports_a, ids_b), (b, a, reports_b, ids_a)):
                for report in source_reports:
                    if report["i"] in target_ids:
                        continue
                    quote = _article_quote(article_texts.get(target), report.get("text"))
                    emit("R1_InfoOnlyIn", source, target, (report["i"],),
                         [report.get("evidence")], item=report.get("text"),
                         status="headline_omission" if quote else "not_found_in_article",
                         article_evidence=({"source": target, **quote} if quote else None))

            # R2 — same action, but a different or hidden actor.
            agents_a, agents_b = groups["agents"].get(a, []), groups["agents"].get(b, [])
            for source, target, source_agents, target_agents in (
                (a, b, agents_a, agents_b), (b, a, agents_b, agents_a)):
                target_by_item = {}
                for ag in target_agents:
                    target_by_item.setdefault(ag["i"], []).append(ag)
                for ag in source_agents:
                    if ag["i"] not in {f["i"] for f in groups["reports"].get(target, [])}:
                        continue
                    counterparts = target_by_item.get(ag["i"], [])
                    if any(x["x"] == ag["x"] for x in counterparts):
                        continue
                    emit("R2_ActorShift", source, target, (ag["x"], ag["i"]),
                         [ag.get("evidence"), *[x.get("evidence") for x in counterparts]],
                         actor=ag.get("actor"), action=ag.get("action"),
                         other_actors=[x.get("actor") for x in counterparts])

            # R3 — shared actor, but evaluative wording/polarity is absent.
            targets_a = ({f["x"] for f in groups["actors"].get(a, [])}
                         | {f["i"] for f in groups["reports"].get(a, [])})
            targets_b = ({f["x"] for f in groups["actors"].get(b, [])}
                         | {f["i"] for f in groups["reports"].get(b, [])})
            for source, target, source_evals, target_entities, target_evals in (
                (a, b, groups["evals"].get(a, []), targets_b, groups["evals"].get(b, [])),
                (b, a, groups["evals"].get(b, []), targets_a, groups["evals"].get(a, []))):
                for ev in source_evals:
                    if not ev.get("x") or ev["x"] not in target_entities:
                        continue
                    if any(x.get("x") == ev["x"] and x.get("polarity") == ev.get("polarity")
                           for x in target_evals):
                        continue
                    emit("R3_EvalShift", source, target, (ev["x"], ev.get("polarity")),
                         [ev.get("evidence")], actor=ev.get("target"),
                         word=ev.get("word"), polarity=ev.get("polarity"))

            # R4 — headlines stress incomparable outcome categories.
            for left in groups["outcomes"].get(a, []):
                for right in groups["outcomes"].get(b, []):
                    lt, rt = left.get("type"), right.get("type")
                    if not lt or not rt or lt == rt or _is_a(lt, rt, ontology) or _is_a(rt, lt, ontology):
                        continue
                    emit("R4_OutcomeShift", a, b, (left["k"], right["k"]),
                         [left.get("evidence"), right.get("evidence")],
                         left_outcome=left.get("text"), left_type=lt,
                         right_outcome=right.get("text"), right_type=rt)
                    emit("R4_OutcomeShift", b, a, (right["k"], left["k"]),
                         [right.get("evidence"), left.get("evidence")],
                         left_outcome=right.get("text"), left_type=rt,
                         right_outcome=left.get("text"), right_type=lt)
        changed = len(differences) > before

    return differences


def main():
    """Run S4 and S5 over saved outputs and print derived differences."""
    import argparse
    import json
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.align import load_alignment
    from src.facts import build_facts
    from src.load import load_stories, stories_by_id

    parser = argparse.ArgumentParser(description="Run the S4-S5 framing rules")
    parser.add_argument("--story-id", help="run one story; defaults to all stories")
    parser.add_argument("--limit", type=int, default=5,
                        help="maximum differences printed per story (default: 5)")
    args = parser.parse_args()

    extraction_path = os.path.join(root, "data", "extractions", "extractions.json")
    with open(extraction_path, encoding="utf-8") as f:
        extractions = json.load(f)
    alignments = load_alignment(os.path.join(root, "data", "alignment", "alignment.json"))
    stories = stories_by_id(load_stories(os.path.join(root, "data", "processed", "stories.json")))
    story_ids = [args.story_id] if args.story_id else list(extractions)

    for story_id in story_ids:
        story = stories.get(story_id)
        alignment = alignments.get(story_id)
        if story is None or alignment is None or story_id not in extractions:
            print(f"{story_id}: missing story, extraction, or alignment data")
            continue
        articles = story["articles"]
        headlines = {src: articles[src]["headline"] for src in ("fox", "nyt", "hpo")}
        article_texts = {
            src: " ".join(s["text"] for s in articles[src].get("sentences", []))
            for src in ("fox", "nyt", "hpo")
        }
        facts = build_facts(extractions[story_id], alignment)
        differences = derive_differences(facts, headlines, article_texts)
        print(f"\n{story_id}: {len(differences)} derived differences")
        for difference in differences[:max(0, args.limit)]:
            print(" ", json.dumps(difference, ensure_ascii=False))


if __name__ == "__main__":
    main()
