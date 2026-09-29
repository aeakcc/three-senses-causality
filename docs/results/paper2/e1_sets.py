"""Paper 2, §11b: the scoring sets of E1 / E2 (units with exactly one candidate pair, so a unit's score is one pair's
score and the max-over-candidates bias of docs/results/paper2/e1_eval.py cannot favour classes with more candidates).

    PYTHONPATH=. .venv/bin/python docs/results/paper2/e1_sets.py
      writes data/paper2/e1/{clingen,rgd}_{docs.json,units.tsv}, swap_clingen_{u,k}.json, manifest.json

RGD IEP and IAGP single-pair units are capped at CAP each (random.Random(0) over sorted unit ids; units that also carry
a rarer code are always kept). ClinGen swaps keep the counterfactuals of the E1 primary units (GEN_CLV, EXP_*).
Percha, BioTriplex and the CRED swaps are scored whole.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from units import candidate_pairs, read_units, write_units  # noqa: E402

P2, OUT = ROOT / "data" / "paper2", ROOT / "data" / "paper2" / "e1"
CAP = 3000
CAPPED = {"IEP", "IAGP"}
PRIMARY = {"GEN_CLV", "EXP_A", "EXP_B", "EXP_U"}


def single_pair(src: str) -> tuple[list[dict], dict[str, dict]]:
    docs = {d["pmid"]: d for d in json.loads((P2 / f"{src}_docs.json").read_text(encoding="utf-8"))}
    units = [u for u in read_units(P2 / f"{src}_units.tsv")
             if len(candidate_pairs(docs[u["doc_id"]], u["gene_entrez"], u["disease_mesh"])[0]) == 1]
    return units, docs


def cap_rgd(units: list[dict]) -> list[dict]:
    keep = [u for u in units if not set(u["classes"]) <= CAPPED]
    for c in sorted(CAPPED):
        pool = sorted((u for u in units if set(u["classes"]) <= CAPPED and c in u["classes"] and u not in keep),
                      key=lambda u: u["unit_id"])
        keep += random.Random(0).sample(pool, min(CAP, len(pool)))
    return sorted(keep, key=lambda u: u["unit_id"])


def write(name: str, docs: list[dict], manifest: dict) -> None:
    p = OUT / name
    p.write_text(json.dumps(docs, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest[name] = {"n": len(docs), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"rule": __doc__.split("\n\n")[2].strip(), "cap": CAP}
    for src in ("clingen", "rgd"):
        units, docs = single_pair(src)
        if src == "rgd":
            units = cap_rgd(units)
        write_units(OUT / f"{src}_units.tsv", units)
        manifest[f"{src}_units.tsv"] = {"n": len(units),
                                        "sha256": hashlib.sha256((OUT / f"{src}_units.tsv").read_bytes()).hexdigest()}
        write(f"{src}_docs.json", [docs[i] for i in sorted({u["doc_id"] for u in units})], manifest)
        if src == "clingen":
            prim = {(u["doc_id"], g) for u in units if set(u["classes"]) & PRIMARY for g in u["gene_entrez"]}
    rows = list(csv.DictReader(open(P2 / "swap" / "mapping.tsv", encoding="utf-8"), delimiter="\t"))
    for c in "uk":
        keep = {r["doc_id"] for r in rows if r["file"] == f"clingen_{c}" and (r["orig_doc_id"], r["gene"]) in prim}
        docs = [d for d in json.loads((P2 / "swap" / f"clingen_{c}.json").read_text(encoding="utf-8")) if d["pmid"] in keep]
        write(f"swap_clingen_{c}.json", docs, manifest)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v["n"] for k, v in manifest.items() if isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
