"""Create the gold-annotation workbooks for the 20 annotated stories (proposal, Section 5.1).

One Excel file per annotator. Each row is one headline pair (3 pairs per story); the
annotator marks which of the four framing differences are present and copies the
words that show them. Dev and test files go in separate folders: M2-M5 (pipeline
builders) must not open eval/annotation/test/ until the single test run.

Assignment (from the proposal):
    dev  (6 stories):  M3 Sameeksha 3, M4 Zainab 3
    test (14 stories): M1 Siddharth 3, M6 Sachin 3, M7 Nitya 4, M8 Megha 4
    kappa: 5 test stories get a second, independent annotator (never the first)

Run once:  python scripts/make_annotation_sheets.py
Needs:     pip install openpyxl
"""

import json
import os
import random
import sys

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.load import headline_pairs, load_stories, stories_by_id  # noqa: E402

OUT = os.path.join(ROOT, "eval", "annotation")
SEED = 407

DEV_ANNOTATORS = [("M3", "Sameeksha", 3), ("M4", "Zainab", 3)]
TEST_ANNOTATORS = [("M1", "Siddharth", 3), ("M6", "Sachin", 3), ("M7", "Nitya", 4), ("M8", "Megha", 4)]
# Who does the 5 second annotations: the two with the smallest load get two each.
KAPPA_SECOND = ["Siddharth", "Sachin", "Siddharth", "Sachin", "Megha"]

OUTCOME_TYPES = ["Economic", "Safety", "Legal", "Political", "Health", "Social", "Reputational", "Other"]

COLUMNS = [
    # (header, width, filled by annotator?)
    ("row_id", 14, False), ("story_id", 12, False), ("event", 40, False),
    ("outlet_A", 8, False), ("headline_A", 45, False),
    ("outlet_B", 8, False), ("headline_B", 45, False),
    ("D1_info", 9, True), ("D1_words_A", 22, True), ("D1_words_B", 22, True),
    ("D2_actor", 9, True), ("D2_words_A", 22, True), ("D2_words_B", 22, True),
    ("D3_eval", 9, True), ("D3_words_A", 22, True), ("D3_words_B", 22, True), ("D3_polarity", 11, True),
    ("D4_outcome", 11, True), ("D4_type_A", 13, True), ("D4_type_B", 13, True),
    ("D4_words_A", 22, True), ("D4_words_B", 22, True),
    ("notes", 30, True),
]

GUIDELINES = """\
GOLD ANNOTATION GUIDELINES: Same-event headline framing (CS F407, Team BATCH)

Each row compares two headlines (A and B) about the SAME event. For each of the four
dimensions, answer y (difference present) or n (absent), and copy the exact words from the
headline(s) that show it. Copy words exactly as written (copy-paste is best). Work alone:
do not discuss rows with other annotators and do not look at any system output.

Judge the HEADLINES. Use the "Article leads" sheet only to understand what the event is.

D1 INFORMATION SELECTION: one headline includes a piece of information that the other
   leaves out entirely (a fact, event, detail, number, person, quote or reason).
   - Same fact in different words is NOT a difference ("fire tear gas" vs "use tear gas" -> n).
   - Words: the included information, in the headline that has it (A or B column).
   e.g. A "Police fire tear gas as city faces curfew" / B "Police fire tear gas" -> y,
        D1_words_A = "city faces curfew".

D2 ACTOR / ACTION FRAMING: both headlines describe the same action, but they name a
   different actor as the one acting, or one hides the actor (passive voice, "X dies", "used").
   - Words: the actor + action in each headline.
   e.g. A "Police shoot man" / B "Man dies in shooting" -> y,
        D2_words_A = "Police shoot", D2_words_B = "Man dies in shooting".

D3 EVALUATIVE WORDING: one headline uses loaded / judgmental words about an actor or event
   that the other does not (or the other uses the opposite tone). Includes loaded verbs
   ("slams", "blasts", "caves"), adjectives ("radical", "peaceful", "furious") and word
   choices that change the impression ("asked" vs "used").
   - Neutral reporting verbs ("says", "announces") are not evaluative.
   - Words: the loaded word(s). D3_polarity: pos, neg or mixed (tone of the loaded word(s)).
   e.g. A "Obama slams GOP plan" / B "Obama criticizes GOP plan" -> y, D3_words_A = "slams", neg.

D4 CONSEQUENCE EMPHASIS: BOTH headlines mention consequences / stakes, but of different
   types. Types: Economic, Safety, Legal, Political, Health, Social, Reputational, Other.
   - If only ONE headline mentions a consequence, that is D1 (information), not D4.
   - Fill D4_type_A / D4_type_B (dropdowns) and the words.
   e.g. A "Shutdown to cost economy $6bn" / B "Shutdown puts air safety at risk" -> y,
        Economic vs Safety.

GENERAL
- A row can have several y's, or none.
- If unsure, choose your best reading and write why in "notes".
- Leave the grey columns unchanged. Do not add or delete rows.
- Save as .xlsx with the same file name and send it to Megha (M8), or push it to
  eval/annotation/ in the repo. Expected time: about 3-4 minutes per row.
"""


def assign(dev, test):
    rng = random.Random(SEED)
    dev, test = list(dev), list(test)
    rng.shuffle(dev)
    rng.shuffle(test)
    plan, k = {}, 0
    for code, name, n in DEV_ANNOTATORS:
        plan[name] = {"code": code, "split": "dev", "primary": dev[k:k + n], "second": []}
        k += n
    k = 0
    for code, name, n in TEST_ANNOTATORS:
        plan[name] = {"code": code, "split": "test", "primary": test[k:k + n], "second": []}
        k += n
    # 5 kappa stories: each given to a second annotator who is not its first annotator
    first_of = {sid: name for name, p in plan.items() if p["split"] == "test" for sid in p["primary"]}
    pool = [sid for sid in test]
    rng.shuffle(pool)
    for second in KAPPA_SECOND:
        sid = next(s for s in pool if first_of[s] != second
                   and s not in {x for p in plan.values() for x in p["second"]})
        plan[second]["second"].append(sid)
    return plan


def short_event(story):
    ev = story.get("main_event") or ""
    return ev if len(ev) <= 160 else ev[:157] + "..."


def write_workbook(path, name, info, stories):
    wb = Workbook()
    ws = wb.active
    ws.title = "Annotate"
    grey = PatternFill("solid", fgColor="EEEEEE")
    head = PatternFill("solid", fgColor="1F3864")
    wrap = Alignment(wrap_text=True, vertical="top")

    for c, (h, width, _) in enumerate(COLUMNS, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = head
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[cell.column_letter].width = width
    ws.freeze_panes = "H2"
    ws.column_dimensions["B"].hidden = True          # story_id: needed by the merge script only

    yn = DataValidation(type="list", formula1='"y,n"', allow_blank=True)
    pol = DataValidation(type="list", formula1='"pos,neg,mixed"', allow_blank=True)
    typ = DataValidation(type="list", formula1='"' + ",".join(OUTCOME_TYPES) + '"', allow_blank=True)
    for dv in (yn, pol, typ):
        ws.add_data_validation(dv)
    col = {h: i + 1 for i, (h, _, _) in enumerate(COLUMNS)}

    r = 2
    for sid in info["primary"] + info["second"]:
        story = stories[sid]
        for a, b in headline_pairs(story):
            row_id = f"{sid[:8]}-{a['source']}-{b['source']}"
            values = {"row_id": row_id, "story_id": sid, "event": short_event(story),
                      "outlet_A": a["source"], "headline_A": a["headline"],
                      "outlet_B": b["source"], "headline_B": b["headline"]}
            for h, (c) in col.items():
                cell = ws.cell(row=r, column=c, value=values.get(h))
                cell.alignment = wrap
                if not COLUMNS[c - 1][2]:
                    cell.fill = grey
            for h in ("D1_info", "D2_actor", "D3_eval", "D4_outcome"):
                yn.add(ws.cell(row=r, column=col[h]))
            pol.add(ws.cell(row=r, column=col["D3_polarity"]))
            typ.add(ws.cell(row=r, column=col["D4_type_A"]))
            typ.add(ws.cell(row=r, column=col["D4_type_B"]))
            ws.row_dimensions[r].height = 60
            r += 1

    g = wb.create_sheet("Guidelines")
    g.column_dimensions["A"].width = 110
    for i, line in enumerate(GUIDELINES.splitlines(), 1):
        g.cell(row=i, column=1, value=line).alignment = Alignment(wrap_text=True)
    g.cell(row=1, column=1).font = Font(bold=True)

    leads = wb.create_sheet("Article leads")
    for c, (h, w) in enumerate([("story_id", 12), ("outlet", 8), ("headline", 45),
                                ("first sentences of the article", 110)], 1):
        cell = leads.cell(row=1, column=c, value=h)
        cell.font = Font(bold=True)
        leads.column_dimensions[cell.column_letter].width = w
    lr = 2
    for sid in info["primary"] + info["second"]:
        for src, art in stories[sid]["articles"].items():
            lead = " ".join(s["text"].strip() for s in art["sentences"][:4])
            for c, v in enumerate([sid, src, art["headline"], lead], 1):
                leads.cell(row=lr, column=c, value=v).alignment = wrap
            lr += 1

    wb.save(path)
    return r - 2


def main():
    stories = stories_by_id(load_stories(os.path.join(ROOT, "data", "processed", "stories.json")))
    with open(os.path.join(ROOT, "data", "splits.json")) as f:
        split = json.load(f)
    plan = assign(split["dev"], split["test"])

    for sub in ("dev", "test"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    with open(os.path.join(OUT, "GUIDELINES.txt"), "w", encoding="utf-8") as f:
        f.write(GUIDELINES)
    with open(os.path.join(OUT, "assignment.json"), "w") as f:
        json.dump(plan, f, indent=1)

    for name, info in plan.items():
        path = os.path.join(OUT, info["split"], f"{info['code']}_{name}.xlsx")
        n = write_workbook(path, name, info, stories)
        extra = f" + {len(info['second'])} second-annotator" if info["second"] else ""
        print(f"{info['code']} {name:10s} {info['split']:4s} {len(info['primary'])} stories{extra}"
              f" = {n} rows -> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
