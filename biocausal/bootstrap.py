"""Paired document-level bootstrap CIs and p-values for RE metrics (preregistration §2, §6.4; improvement_plan K8).

Counts are kept PER DOCUMENT: tp / predicted / gold for micro (pair + type), pair-level, each of the 8 relation types
and the 3 causal-subset classes, with a document's unpairable gold relations added to its gold. A resample sums the
counts of the drawn documents, so a document drawn twice counts twice (the set-based ``evaluate.all_metrics`` would
count it once for micro/pair and twice for per-type, which is why it is not reused here). Numpy only.

    python -m biocausal.bootstrap --a runs/x/pred_dev.jsonl [--b runs/y/pred_dev.jsonl] --data data/BioRED/Dev.BioC.JSON
        [--n 10000] [--seed 0] [--out boot.json]

Several files after ``--a`` / ``--b`` are ensembled first (probability average, ``ensemble.ensemble_rows``).
``--data`` (the gold file) supplies the unpairable gold relations and the documents without candidate pairs, and each
side must predict exactly its candidate pairs; without
it the documents are those in the prediction rows and nothing is unpairable. CI = 2.5 / 97.5 percentiles; paired
one-sided p for "A better than B" = share of resamples with A - B <= 0. Every call first asserts that the full-sample
point estimates equal ``evaluate.all_metrics`` (the self-check).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from biocausal.data import LABELS, load_docs, unpairable_gold
from biocausal.ensemble import ensemble_rows
from biocausal.evaluate import CAUSAL_CLASSES, all_metrics, causal_class, is_causal_pair, read_jsonl

GROUPS = ["micro", "pair", *LABELS[1:], *(f"causal:{c}" for c in CAUSAL_CLASSES)]
_G = {g: i for i, g in enumerate(GROUPS)}
TYPES = [_G[t] for t in LABELS[1:]]
CAUSAL = [_G[f"causal:{c}"] for c in CAUSAL_CLASSES]


def doc_counts(rows: list[dict], unpairable: dict[str, list[str]], pmids: list[str]) -> np.ndarray:
    """[n_docs, len(GROUPS), 3] counts (tp, predicted, gold) per document, documents in ``pmids`` order."""
    idx = {p: i for i, p in enumerate(pmids)}
    c = np.zeros((len(pmids), len(GROUPS), 3))
    for r in rows:
        d, gold, pred = c[idx[r["pmid"]]], r["gold"], r["pred"]
        if gold != "None":
            d[[0, 1, _G[gold]], 2] += 1
        if pred != "None":
            d[[0, 1, _G[pred]], 1] += 1
        if gold != "None" and pred != "None":
            d[1, 0] += 1
            if gold == pred:
                d[[0, _G[gold]], 0] += 1
        if is_causal_pair(r["t1"], r["t2"]):
            cg, cp = _G["causal:" + causal_class(gold)], _G["causal:" + causal_class(pred)]
            d[cg, 2] += 1
            d[cp, 1] += 1
            d[cg, 0] += cg == cp
    for p, types in unpairable.items():
        for t in types:
            c[idx[p], [0, 1, _G[t]], 2] += 1
    return c


def metrics(summed: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics from summed counts [..., len(GROUPS), 3]; same float operations as ``evaluate.prf``."""
    tp, npred, ngold = summed[..., 0], summed[..., 1], summed[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(npred > 0, tp / npred, 0.0)
        r = np.where(ngold > 0, tp / ngold, 0.0)
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return {"micro_p": p[..., 0], "micro_r": r[..., 0], "micro_f1": f1[..., 0], "pair_f1": f1[..., 1],
            "macro_f1": sum(f1[..., j] for j in TYPES) / len(TYPES),          # sequential sum, as evaluate.py
            "causal_f1": f1[..., _G["causal:causal"]], "causal_macro_f1": sum(f1[..., j] for j in CAUSAL) / len(CAUSAL)}


def resample(counts: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics per resample; ``weights`` [B, n_docs] = how often each document is drawn."""
    n, g, k = counts.shape
    return metrics((weights @ counts.reshape(n, g * k)).reshape(-1, g, k))


def check_point(rows: list[dict], unpairable: dict[str, list[str]], counts: np.ndarray) -> None:
    """Self-check: full-sample point estimates equal evaluate.all_metrics."""
    ref = all_metrics(rows, [t for ts in unpairable.values() for t in ts])
    want = {"micro_p": ref["micro"]["p"], "micro_r": ref["micro"]["r"], "micro_f1": ref["micro"]["f1"],
            "pair_f1": ref["pair"]["f1"], "macro_f1": ref["macro_f1"], "causal_f1": ref["causal"]["causal"]["f1"],
            "causal_macro_f1": ref["causal"]["macro_f1"]}
    got = metrics(counts.sum(0))
    bad = {k: (float(got[k]), v) for k, v in want.items() if abs(float(got[k]) - v) > 1e-12}
    assert not bad, f"bootstrap counts disagree with evaluate.all_metrics: {bad}"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", nargs="+", required=True, metavar="PRED", help="pred jsonl (several: ensembled)")
    ap.add_argument("--b", nargs="*", default=[], metavar="PRED", help="paired comparison system (several: ensembled)")
    ap.add_argument("--data", help="gold BioC / Doc JSON: unpairable gold and pair-less documents")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write the result json here (default: print)")
    a = ap.parse_args(argv)
    sides = {"a": a.a, "b": a.b} if a.b else {"a": a.a}
    rows = {s: read_jsonl(files[0]) if len(files) == 1 else ensemble_rows([read_jsonl(p) for p in files])
            for s, files in sides.items()}
    if a.data:
        docs = load_docs(a.data)
        pmids, unpairable = [d["pmid"] for d in docs], {d["pmid"]: unpairable_gold(d) for d in docs}
        want = {(d["pmid"], p["e1"], p["e2"]) for d in docs for p in d["pairs"]}
        for s, rs in rows.items():   # a missing document would silently drop its gold from the false negatives
            assert len(rs) == len(want) and {(r["pmid"], r["e1"], r["e2"]) for r in rs} == want, \
                f"--{s} does not predict exactly the candidate pairs of --data"
    else:
        pmids, unpairable = sorted({r["pmid"] for rs in rows.values() for r in rs}), {}
    counts = {s: doc_counts(rs, unpairable, pmids) for s, rs in rows.items()}
    for s in rows:
        check_point(rows[s], unpairable, counts[s])
    w = np.random.default_rng(a.seed).multinomial(len(pmids), np.full(len(pmids), 1 / len(pmids)), size=a.n)
    point = {s: metrics(c.sum(0)) for s, c in counts.items()}
    boot = {s: resample(c, w) for s, c in counts.items()}
    out: dict = {"files": sides, "data": a.data, "n_docs": len(pmids), "n_resamples": a.n, "seed": a.seed}
    for s in rows:
        out[s] = {k: {"point": float(point[s][k]), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
                  for k, v in boot[s].items()}
    if "b" in rows:
        out["a_minus_b"] = {}
        for k in boot["a"]:
            d = boot["a"][k] - boot["b"][k]
            out["a_minus_b"][k] = {"point": float(point["a"][k] - point["b"][k]), "lo": float(np.percentile(d, 2.5)),
                                   "hi": float(np.percentile(d, 97.5)), "p_one_sided": float(np.mean(d <= 0))}
    text = json.dumps(out, indent=1)
    Path(a.out).write_text(text, encoding="utf-8") if a.out else print(text)
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
