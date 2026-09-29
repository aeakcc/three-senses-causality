"""How the ClinGen units outside the single-candidate E1 set differ from those inside it (descriptive, post hoc;
asked for by the supervisor on 2026-09-28). Run from the repo root: python3 docs/results/paper2/excluded_units.py
Writes docs/results/paper2/excluded_units.md. A unit counts toward every coarse class it carries."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from units import candidate_pairs, read_units  # noqa: E402

DATA = Path("data/paper2")
GROUP = {"GEN_CLV": "genetic", "GEN_SEG": "genetic", "GEN_CC": "genetic", "EXP_A": "expression", "EXP_B": "expression",
         "EXP_U": "expression", "MS_ORG": "model_rescue", "MS_CELL": "model_rescue", "RES": "model_rescue",
         "FA_PAT": "functional", "FA_NONPAT": "functional", "BIO": "biochem", "PI": "biochem"}
COLS = ["genetic", "expression", "model_rescue", "functional", "biochem"]

docs = {d["pmid"]: d for d in json.load(open(DATA / "clingen_docs.json"))}
units = read_units(DATA / "clingen_units.tsv")
kept = {u["unit_id"] for u in read_units(DATA / "e1/clingen_units.tsv")}
groups = {"included (1 candidate)": [], "excluded, 0 candidates": [], "excluded, 2+ candidates": []}
for u in units:
    n = len(candidate_pairs(docs[u["pmid"]], u["gene_entrez"], u["disease_mesh"])[0])
    if u["unit_id"] in kept:
        assert n == 1, u["unit_id"]
        groups["included (1 candidate)"].append(u)
    else:
        assert n != 1, u["unit_id"]
        groups["excluded, 0 candidates" if n == 0 else "excluded, 2+ candidates"].append(u)

lines = ["# ClinGen units inside and outside the single-candidate E1 set", "",
         "| Group | Units | " + " | ".join(COLS) + " |", "|---|---|" + "---|" * len(COLS)]
for name, us in groups.items():
    c = Counter(g for u in us for g in {GROUP[x] for x in u["classes"] if x in GROUP})
    lines.append(f"| {name} | {len(us):,} | " + " | ".join(f"{100 * c[g] / len(us):.1f}%" for g in COLS) + " |")
Path("docs/results/paper2/excluded_units.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
