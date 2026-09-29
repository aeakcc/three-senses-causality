"""Probability-averaging ensemble of N prediction files -> same metrics as evaluate.py.

CLI: ``python -m biocausal.ensemble runs/a/pred_test.jsonl runs/b/pred_test.jsonl [-o ens_pred.jsonl] [--metrics m.json]
[--unpairable TYPE ...]`` (gold relations without a candidate row, counted as fn: BioRED dev ``Bind``, test none).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from biocausal.data import AUX_CLASSES, LABELS, NOVELTY
from biocausal.evaluate import all_metrics, read_jsonl, write_jsonl


def ensemble_rows(row_sets: list[list[dict]]) -> list[dict]:
    """Average probs over aligned rows (same (pmid,e1,e2) sets required); pred = argmax.

    Novelty (``probs_novel`` of a novelty-head run): averaged the same way, ``pred_novel`` = argmax. If any member lacks
    ``probs_novel``, the ensemble rows carry no novelty keys, so no +NV metric is reported for a mixed ensemble.
    Aux-head ``probs_<k>`` lists (model.predict): averaged per class, dropped unless every member has them.
    """
    base = row_sets[0]
    keys = [(r["pmid"], r["e1"], r["e2"]) for r in base]
    out = []
    lookups = [{(r["pmid"], r["e1"], r["e2"]): r for r in rs} for rs in row_sets[1:]]
    for lk in lookups:
        if set(lk) != set(keys):
            raise ValueError("prediction files cover different pair sets")
    for r, k in zip(base, keys):
        probs = {l: (r["probs"][l] + sum(lk[k]["probs"][l] for lk in lookups)) / len(row_sets) for l in LABELS}
        row = {**r, "pred": max(LABELS, key=probs.__getitem__), "probs": probs}
        if "probs_novel" in r and all("probs_novel" in lk[k] for lk in lookups):
            pn = {n: (r["probs_novel"][n] + sum(lk[k]["probs_novel"][n] for lk in lookups)) / len(row_sets)
                  for n in NOVELTY}
            row.update(probs_novel=pn, pred_novel=max(NOVELTY, key=pn.__getitem__))
        else:
            for key in ("gold_novel", "pred_novel", "probs_novel"):
                row.pop(key, None)
        for pk in (f"probs_{h}" for h in AUX_CLASSES):   # aux heads (model.predict): [p per class], same rule
            if pk in r and all(pk in lk[k] for lk in lookups):
                row[pk] = [sum(v) / len(row_sets) for v in zip(r[pk], *(lk[k][pk] for lk in lookups))]
            else:
                row.pop(pk, None)
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", help="pred_*.jsonl files to average")
    ap.add_argument("-o", "--out", help="write the ensembled rows to this jsonl")
    ap.add_argument("--metrics", help="write metrics json here (default: print)")
    ap.add_argument("--unpairable", nargs="*", default=[], metavar="TYPE",
                    help="gold relation types without a candidate row, counted as fn (BioRED dev: Bind; test: none)")
    a = ap.parse_args(argv)
    rows = ensemble_rows([read_jsonl(p) for p in a.files])
    if a.out:
        write_jsonl(a.out, rows)
    m = {"files": a.files, **all_metrics(rows, a.unpairable)}
    if a.metrics:
        Path(a.metrics).write_text(json.dumps(m, indent=1))
    else:
        print(json.dumps(m, indent=1))
    return m


if __name__ == "__main__":
    main(sys.argv[1:])
