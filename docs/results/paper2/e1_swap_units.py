"""Paper 2, §11b H4: units of the name-swapped ClinGen docs (data/paper2/e1/swap_clingen_{u,k}.json) for e1_eval.py.
A swapped doc '<pmid>_s<i>' changes one gene's surface only (entity ids and pairs are unchanged), so its unit is the
original single-pair unit of that (doc, gene) with doc_id = the swapped id.

    PYTHONPATH=. .venv/bin/python docs/results/paper2/e1_swap_units.py   -> data/paper2/e1/swap_clingen_{u,k}_units.tsv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from units import read_units, write_units  # noqa: E402

E1 = ROOT / "data" / "paper2" / "e1"

if __name__ == "__main__":
    units = read_units(E1 / "clingen_units.tsv")
    by = {}
    for u in units:
        for g in u["gene_entrez"]:
            by.setdefault((u["doc_id"], g), []).append(u)
    rows = list(csv.DictReader(open(ROOT / "data/paper2/swap/mapping.tsv", encoding="utf-8"), delimiter="\t"))
    for c in "uk":
        ids = {d["pmid"] for d in json.loads((E1 / f"swap_clingen_{c}.json").read_text(encoding="utf-8"))}
        out = [{**u, "unit_id": f"{u['unit_id']}|{m['doc_id']}", "doc_id": m["doc_id"]}
               for m in rows if m["file"] == f"clingen_{c}" and m["doc_id"] in ids
               for u in by.get((m["orig_doc_id"], m["gene"]), [])]
        write_units(E1 / f"swap_clingen_{c}_units.tsv", out)
        print(c, len(ids), "docs,", len(out), "units")
