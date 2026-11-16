"""Stage S4: ontology + logical facts (atoms) for one story.

Input:  S2 extractions {source: result} and the S3 alignment of the same story.
Output: a knowledge base (KB) dict:
    atoms     list of ground atoms, each a tuple: (predicate, arg1, arg2, ...)
    evidence  {atom_key: {"source", "text", "start", "end", "label"}}  where the atom came from
    labels    {id: readable label}   e.g. "i1" -> "fire tear gas", "a2" -> "police"
    own_labels {"fox:i1": "fire tear gas", "nyt:i1": "use tear gas"}  each headline's own wording

Predicates (proposal, Section 5.3):
    Headline(h)            a usable headline (h = fox | nyt | hpo)
    Pair(a, b)             two different usable headlines of the story (both orders)
    Actor(h, x)            headline h mentions actor x
    InstanceOf(x, c)       actor x belongs to ontology category c (from the S2 actor type)
    Reports(h, i)          headline h reports information item i
    Agent(h, x, i)         headline h shows actor x performing item i
    Eval(h, x, p, w)       headline h uses word w with polarity p about actor/item x
    Outcome(h, k, t)       headline h stresses consequence k of type t
    IsA(c, d)              ontology: category c is a sub-type of category d
    Specific(t)            outcome type t is informative (every type except "Other")

IDs (i1, a1, o1 ...) come from S3 alignment, so equivalent mentions in different
headlines share one ID. IDs are local to one story.
"""

from src.load import source_order

# ------------------------------------------------------------------ ontology
# Small, hand-written category hierarchy. Each pair (child, parent) becomes IsA(child, parent);
# the engine derives the transitive closure (e.g. Company -> Organization -> Institution -> Agent).
ACTOR_ONTOLOGY = [
    ("Politician", "PoliticalActor"), ("GovernmentOfficial", "PoliticalActor"),
    ("PoliticalParty", "PoliticalActor"), ("PoliticalActor", "Agent"),
    ("GovernmentBody", "Government"), ("Court", "Government"),
    ("LawEnforcement", "Government"), ("Military", "Government"),
    ("Government", "Institution"), ("Country", "Institution"),
    ("Company", "Organization"), ("Media", "Organization"),
    ("Organization", "Institution"), ("Institution", "Agent"),
    ("Activist", "CivilSociety"), ("Citizen", "CivilSociety"), ("CivilSociety", "Agent"),
    ("Person", "Agent"), ("Other", "Agent"),
]
# Outcome types (same names as src/extract.py OUTCOME_TYPES). R4 does not fire when one type
# is a sub-type of the other, e.g. a Health outcome vs a Safety outcome.
OUTCOME_ONTOLOGY = [
    ("Health", "Safety"), ("Safety", "Harm"),
    ("Reputational", "Political"), ("Political", "PublicAffairs"),
    ("Legal", "PublicAffairs"),
    ("Economic", "Material"), ("Social", "PublicAffairs"),
]
OUTCOME_TYPES = ["Economic", "Safety", "Legal", "Political", "Health", "Social", "Reputational", "Other"]


def ontology_atoms():
    atoms = [("IsA", c, d) for c, d in ACTOR_ONTOLOGY + OUTCOME_ONTOLOGY]
    atoms += [("Specific", t) for t in OUTCOME_TYPES if t != "Other"]
    return atoms


# ------------------------------------------------------------------ facts

def atom_key(atom):
    """Readable, hashable form of an atom, e.g. 'Reports(fox, i1)'."""
    return f"{atom[0]}({', '.join(str(a) for a in atom[1:])})"


def _ids(alignment, kind, src):
    """extraction-list index -> shared ID, for one headline and one kind."""
    return {m["index"]: m["id"] for m in alignment.get("mentions", {}).get(kind, [])
            if m.get("src") == src}


def _norm(s):
    return " ".join(str(s or "").lower().split())


def build_facts(extractions, alignment):
    """Build the KB of one story from its S2 extractions and S3 alignment."""
    atoms, evidence, labels, own = [], {}, {}, {}      # own[(h, id)] = that headline's wording

    def add(atom, src=None, entry=None, label=None):
        if atom in seen:
            return
        seen.add(atom)
        atoms.append(atom)
        if src and entry is not None:
            evidence[atom_key(atom)] = {
                "source": src, "text": entry.get("span") or entry.get("word"),
                "start": entry.get("start"), "end": entry.get("end"), "label": label}

    seen = set()
    usable = [s for s in source_order(extractions) if (extractions.get(s) or {}).get("ok")]
    for h in usable:
        add(("Headline", h))
    for a in usable:
        for b in usable:
            if a != b:
                add(("Pair", a, b))

    for h in usable:
        f = extractions[h]["fields"]
        actor_id, item_id, outcome_id = (_ids(alignment, k, h) for k in ("actors", "items", "outcomes"))
        name_to_actor = {_norm(a["name"]): actor_id[n] for n, a in enumerate(f["actors"]) if n in actor_id}
        text_to_item = {_norm(t["text"]): item_id[n] for n, t in enumerate(f["items"]) if n in item_id}

        for n, a in enumerate(f["actors"]):
            if n in actor_id:
                x = actor_id[n]
                labels.setdefault(x, a["name"])
                own[(h, x)] = a["name"]
                add(("Actor", h, x), h, a, a["name"])
                add(("InstanceOf", x, a.get("type") or "Other"))
        for n, it in enumerate(f["items"]):
            if n in item_id:
                labels.setdefault(item_id[n], it["text"])
                own[(h, item_id[n])] = it["text"]
                add(("Reports", h, item_id[n]), h, it, it["text"])
        for ag in f["agents"]:
            x, i = name_to_actor.get(_norm(ag["actor"])), text_to_item.get(_norm(ag["action"]))
            if x and i:
                add(("Agent", h, x, i), h, ag, f"{ag['actor']} -> {ag['action']}")
        for ev in f["evals"]:
            x = name_to_actor.get(_norm(ev["target"])) or text_to_item.get(_norm(ev["target"]))
            if x:                                  # evals about something not extracted are skipped
                add(("Eval", h, x, ev["polarity"], ev["word"]), h, ev, ev["word"])
        for n, oc in enumerate(f["outcomes"]):
            if n in outcome_id:
                labels.setdefault(outcome_id[n], oc["text"])
                own[(h, outcome_id[n])] = oc["text"]
                add(("Outcome", h, outcome_id[n], oc.get("type") or "Other"), h, oc, oc["text"])

    for atom in ontology_atoms():
        add(atom)
    return {"atoms": atoms, "evidence": evidence, "labels": labels, "headlines": usable,
            "own_labels": {f"{h}:{x}": t for (h, x), t in own.items()}}


def main():
    import argparse
    import json
    from src.align import load_alignment

    ap = argparse.ArgumentParser(description="Stage S4: print the logical facts of a story")
    ap.add_argument("--story-id", help="print all atoms of one story; default: counts for all")
    args = ap.parse_args()
    with open("data/extractions/extractions.json", encoding="utf-8") as f:
        ext = json.load(f)
    al = load_alignment()
    for sid in ([args.story_id] if args.story_id else ext):
        kb = build_facts(ext[sid], al[sid])
        story_atoms = [a for a in kb["atoms"] if a[0] not in ("IsA", "Specific")]
        if args.story_id:
            for a in story_atoms:
                lab = " ".join(f"[{kb['labels'][x]}]" for x in a[1:] if x in kb["labels"])
                print(f"  {atom_key(a)}  {lab}")
        else:
            counts = {}
            for a in story_atoms:
                counts[a[0]] = counts.get(a[0], 0) + 1
            print(sid[:8], counts)


if __name__ == "__main__":
    main()
