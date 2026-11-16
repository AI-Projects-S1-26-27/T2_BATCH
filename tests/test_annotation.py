"""Checks the annotation workflow with simulated annotators (no real labels needed).
Run:  python -m tests.test_annotation   (needs openpyxl)
"""

import csv
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

from openpyxl import load_workbook

from eval.merge_annotations import DIMS, check_row, cohen_kappa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_kappa():
    assert cohen_kappa(list("yynn"), list("yynn")) == 1.0
    assert abs(cohen_kappa(list("yyyynnnn"), list("yynnyynn"))) < 1e-9          # chance level
    assert abs(cohen_kappa(list("yyyyyynn"), list("yyyyynnn")) - 0.714) < 0.01


def test_check_row():
    r = {"headline_A": "Police fire tear gas", "headline_B": "Tear gas used",
         "D1_info": "N", "D2_actor": "y", "D2_words_A": "Police fire", "D2_words_B": "used",
         "D3_eval": "y", "D3_words_A": "brutal", "D4_outcome": ""}
    p = check_row(r)
    assert r["D1_info"] == "n"                                                 # case normalised
    assert any("brutal" in x for x in p) and any("D4_outcome" in x for x in p)
    assert not any("D2" in x for x in p)


def _fill(path, rng, complete=True):
    wb = load_workbook(path)
    ws = wb["Annotate"]
    head = [c.value for c in ws[1]]
    col = {h: i + 1 for i, h in enumerate(head)}
    for r in range(2, ws.max_row + 1):
        ha = ws.cell(row=r, column=col["headline_A"]).value
        for d in DIMS:
            if not complete and r == 2 and d == "D4_outcome":
                continue                                                       # leave one blank
            v = rng.choice("yn")
            ws.cell(row=r, column=col[d], value=v)
            if v == "y":
                words_col = {"D1_info": "D1_words_A", "D2_actor": "D2_words_A",
                             "D3_eval": "D3_words_A", "D4_outcome": "D4_words_A"}[d]
                ws.cell(row=r, column=col[words_col], value=ha.split()[0])
    wb.save(path)


def test_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        shutil.copytree(os.path.join(ROOT, "eval", "annotation"), os.path.join(d, "ann"))
        rng = random.Random(1)
        for sub in ("dev", "test"):
            for fn in os.listdir(os.path.join(d, "ann", sub)):
                _fill(os.path.join(d, "ann", sub, fn), rng, complete=fn != "M7_Nitya.xlsx")
        run = lambda: subprocess.run([sys.executable, "-m", "eval.merge_annotations",
                                      "--ann-dir", os.path.join(d, "ann"), "--gold-dir", os.path.join(d, "gold")],
                                     cwd=ROOT, capture_output=True, text=True, check=True).stdout
        out = run()
        assert "Nitya" in out and "11/12" in out, out                          # incomplete row reported
        with open(os.path.join(d, "gold", "gold_test.csv"), encoding="utf-8") as f:
            gold = list(csv.DictReader(f))
        with open(os.path.join(d, "gold", "gold_dev.csv"), encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == 18
        assert len(gold) == 42
        with open(os.path.join(d, "gold", "kappa.json")) as f:
            k = json.load(f)
        assert k["rows"] == 15 and "overall" in k

        # adjudication: flip the first disagreement and check gold changes
        dpath = os.path.join(d, "gold", "disagreements.csv")
        with open(dpath, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows, "random labels should disagree somewhere"
        first = rows[0]
        flipped = "y" if first["first_label"] == "n" else "n"
        first["final"] = flipped
        with open(dpath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(first))
            w.writeheader()
            w.writerows(rows)
        run()
        with open(os.path.join(d, "gold", "gold_test.csv"), encoding="utf-8") as f:
            g = {r["row_id"]: r for r in csv.DictReader(f)}
        assert g[first["row_id"]][first["dimension"]] == flipped
        assert first["dimension"] in g[first["row_id"]]["adjudicated"]
        with open(dpath, encoding="utf-8") as f:                               # decision kept on re-run
            assert any(r["final"] == flipped for r in csv.DictReader(f))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All annotation checks passed.")
