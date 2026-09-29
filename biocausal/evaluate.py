"""Pure metric functions over prediction rows (DESIGN.md §2, evaluate.py).

Row = {pmid, e1, e2, t1, t2, gold, pred, probs{label: p}} (one per candidate pair; pred_*.jsonl lines).

* micro P/R/F1 over (pair, type) triples — BioRED style, a prediction counts iff pair and type match
* +NV (``nv``): same, but novelty must match too — only for rows of a ``--nv-weight`` run, or of an ensemble whose
  members all are (ensemble.ensemble_rows drops the novelty keys otherwise; preregistration §7)
* pair-level (type-agnostic) P/R/F1
* per-type P/R/F1 over the 8 relation types + macro-F1 (rare types never dropped)
* 9x9 confusion matrix (rows = gold, cols = pred, order = LABELS)
* causal subset ({Gene, Variant} x Disease): causal / association / none, causal-class P/R/F1 + 3-way macro-F1
* ``unpairable``: types of gold relations with no candidate row (BioRED self-relations: Dev 1, Train 4, Test 0;
  one Gene-Variant relation in Train) — counted as false negatives so recall is over the official gold count

CLI: ``python -m biocausal.evaluate runs/<name>/pred_test.jsonl``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from biocausal.data import LABEL2ID, LABELS

CAUSAL_TYPES = frozenset({"Positive_Correlation", "Negative_Correlation"})
CAUSAL_CLASSES = ["none", "association", "causal"]
_DIS, _GV = "DiseaseOrPhenotypicFeature", {"GeneOrGeneProduct", "SequenceVariant"}


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def _key(r: dict) -> tuple[str, str, str]:
    return (r["pmid"], r["e1"], r["e2"])


def micro_metrics(rows: Iterable[dict], unpairable: Sequence[str] = ()) -> dict[str, float]:
    rows = list(rows)
    gold = {(_key(r), r["gold"]) for r in rows if r["gold"] != "None"}
    pred = {(_key(r), r["pred"]) for r in rows if r["pred"] != "None"}
    tp = len(gold & pred)
    return prf(tp, len(pred) - tp, len(gold) - tp + len(unpairable))


def pair_metrics(rows: Iterable[dict], unpairable: Sequence[str] = ()) -> dict[str, float]:
    rows = list(rows)
    gold = {_key(r) for r in rows if r["gold"] != "None"}
    pred = {_key(r) for r in rows if r["pred"] != "None"}
    tp = len(gold & pred)
    return prf(tp, len(pred) - tp, len(gold) - tp + len(unpairable))


def nv_metrics(rows: Iterable[dict], unpairable: Sequence[str] = ()) -> dict[str, float]:
    """+NV (preregistration §7 row 4): like micro, but a prediction counts only if pair, type AND novelty match.

    Needs rows from a novelty-head run (``gold_novel`` / ``pred_novel``); unpairable gold relations stay false negatives.
    """
    rows = list(rows)
    gold = {(_key(r), r["gold"], r["gold_novel"]) for r in rows if r["gold"] != "None"}
    pred = {(_key(r), r["pred"], r["pred_novel"]) for r in rows if r["pred"] != "None"}
    tp = len(gold & pred)
    return prf(tp, len(pred) - tp, len(gold) - tp + len(unpairable))


def per_type_metrics(rows: Iterable[dict], unpairable: Sequence[str] = ()) -> tuple[dict[str, dict], float]:
    rows = list(rows)
    out = {}
    for t in LABELS[1:]:
        tp = sum(r["gold"] == t == r["pred"] for r in rows)
        fp = sum(r["pred"] == t != r["gold"] for r in rows)
        fn = sum(r["gold"] == t != r["pred"] for r in rows) + unpairable.count(t)
        out[t] = {**prf(tp, fp, fn), "support": tp + fn}
    return out, sum(v["f1"] for v in out.values()) / len(out)


def confusion(rows: Iterable[dict]) -> dict[str, Any]:
    m = [[0] * len(LABELS) for _ in LABELS]
    for r in rows:
        m[LABEL2ID[r["gold"]]][LABEL2ID[r["pred"]]] += 1
    return {"labels": LABELS, "matrix": m}


def is_causal_pair(t1: str, t2: str) -> bool:
    return (t1 == _DIS and t2 in _GV) or (t2 == _DIS and t1 in _GV)


def causal_class(label: str) -> str:
    return "causal" if label in CAUSAL_TYPES else "association" if label == "Association" else "none"


def causal_metrics(rows: Iterable[dict]) -> dict[str, Any]:
    sub = [(causal_class(r["gold"]), causal_class(r["pred"])) for r in rows if is_causal_pair(r["t1"], r["t2"])]
    per = {}
    for c in CAUSAL_CLASSES:
        tp = sum(g == c == p for g, p in sub)
        per[c] = {**prf(tp, sum(p == c != g for g, p in sub), sum(g == c != p for g, p in sub)),
                  "support": sum(g == c for g, _ in sub)}
    return {"n_pairs": len(sub), "causal": per["causal"], "association": per["association"],
            "macro_f1": sum(v["f1"] for v in per.values()) / len(per), "per_class": per,
            "accuracy": sum(g == p for g, p in sub) / len(sub) if sub else 0.0}


def all_metrics(rows: Iterable[dict], unpairable: Sequence[str] = ()) -> dict[str, Any]:
    """Every metric of a prediction file; ``nv`` only for novelty-head rows (the other numbers never change)."""
    rows = list(rows)
    per_type, macro = per_type_metrics(rows, unpairable)
    out = {"n_pairs": len(rows), "n_gold_unpairable": len(unpairable), "micro": micro_metrics(rows, unpairable),
           "pair": pair_metrics(rows, unpairable), "per_type": per_type, "macro_f1": macro,
           "confusion": confusion(rows), "causal": causal_metrics(rows)}
    if rows and "pred_novel" in rows[0]:
        out["nv"] = nv_metrics(rows, unpairable)
    return out


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python -m biocausal.evaluate pred.jsonl [more.jsonl ...]  (each file scored separately)")
        return
    for p in argv:
        print(json.dumps({"file": p, **all_metrics(read_jsonl(p))}, indent=1))


if __name__ == "__main__":
    main()
