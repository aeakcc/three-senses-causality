"""Preregistration §10: CRED rows per split and the val-threshold rule of the confirmatory comparisons.

    PYTHONPATH=. .venv/bin/python docs/results/cred/cred_s10.py thresholds NAME=PRED.jsonl [NAME=PRED.jsonl ...]
      reads  data/CRED/cred_pairs.tsv and prediction files that cover the CRED val documents
      writes docs/results/cred/s10_thresholds.json (merged by NAME; SHA-256 of every prediction file)

Rows: a CRED key is (pmid, set of gene ids, disease id), so '20208' and '20208;20208' are one pair; exact duplicates
merge, a key with two labels is dropped (test PMID 27608955, D000686-20208). Score of a pair = p(Pos) + p(Neg) of its
candidate pair (max over the candidates of a composite gene id); an unreachable pair has score 0 (strict).
Threshold rule (§10 item 4): on reachable val pairs, the t maximising causal F1 for "score >= t", ties -> highest t.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "docs" / "results" / "posthoc"))
from causal_audit import best_threshold  # noqa: E402

CAUSAL = ("Positive_Correlation", "Negative_Correlation")
OUT = HERE / "s10_thresholds.json"


def load_rows(split: str, path: Path = ROOT / "data" / "CRED" / "cred_pairs.tsv") -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"):
        if r["split"] == split:
            groups.setdefault((r["pmid"], frozenset(r["gene_id"].split(";")), r["disease_id"]), []).append(r)
    return [rs[0] for rs in groups.values() if len({r["label"] for r in rs}) == 1]


def scores(rows: list[dict], pred_file: Path) -> np.ndarray:
    probs = {}
    with open(pred_file, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            probs[(p["pmid"], p["e1"], p["e2"])] = p["probs"][CAUSAL[0]] + p["probs"][CAUSAL[1]]
    return np.array([max(probs[(r["pmid"], *k.split("|"))] for k in r["pairs"].split(";")) if r["pairs"] else 0.0
                     for r in rows])


def val_threshold(pred_file: Path) -> dict:
    rows = [r for r in load_rows("val") if r["status"] == "reachable"]
    y = np.array([r["label"] == "1" for r in rows])
    t, f1 = best_threshold(y, scores(rows, pred_file))
    return {"threshold": t, "val_f1": f1, "n_pairs": len(rows), "n_causal": int(y.sum()), "pred_file": str(pred_file),
            "pred_sha256": hashlib.sha256(Path(pred_file).read_bytes()).hexdigest()}


def main(argv: list[str]) -> None:
    if not argv or argv[0] != "thresholds" or len(argv) < 2:
        sys.exit(__doc__)
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for arg in argv[1:]:
        name, pred = arg.split("=", 1)
        out[name] = val_threshold(Path(pred))
        print(name, {k: out[name][k] for k in ("threshold", "val_f1", "n_pairs", "n_causal")})
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
