"""CRED external validation of the causal proxy (preregistration §8.3): metrics, document bootstrap, baselines.

    PYTHONPATH=. .venv/bin/python docs/results/cred/cred_eval.py
      reads  data/CRED/cred_pairs.tsv (cred_build.py), docs/results/cred/ens_*/pred_cred.jsonl (biocausal.score)
      writes docs/results/cred/cred_results.json; prints the markdown tables of cred.md

Causal = argmax in {Positive_Correlation, Negative_Correlation}; score = p(Pos) + p(Neg); no threshold is tuned.
A CRED pair whose gene id is composite ('5595;5594') is causal if any component pair is, score = max over them.
Variants: ``reachable`` = only pairs with both entities among the PubTator3 entities; ``strict`` = all pairs, an
unreachable one predicted non-causal with score 0. Bootstrap: documents (PMIDs of the subset) drawn with replacement,
10 000 resamples, seed 0, the same draws for every system of a subset; CI = 2.5 / 97.5 percentiles.
Deterministic: no GPU, no unseeded randomness.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from biocausal.evaluate import read_jsonl

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SYSTEMS = {"ens_bc8all": "ensemble re_bl_base_bc8all s42-44", "ens_biolinkbert_base": "ensemble re_biolinkbert_base s42-46"}
CAUSAL = ("Positive_Correlation", "Negative_Correlation")
N_BOOT, SEED, CHUNK = 10_000, 0, 500
# Bansal et al., bioRxiv 10.1101/2024.09.17.613424 v2, Tables 2-3: CRED test file (615 rows), CRED's own entities
REPORTED = [
    {"system": "BioBERT+SVM (CRED paper, Table 2)", "note": "reported, trained in-domain on CRED",
     "p": 0.61, "r": 0.35, "f1": 0.45, "auprc": 0.47},
    {"system": "BioBERT+SVM trained on BioRED (CRED paper, Table 3, BioRED-CRED)",
     "note": "reported, trained on BioRED by the CRED authors", "p": 0.06, "r": 0.01, "f1": 0.01, "auprc": None},
]


def prf(tp, npred, ngold):
    """P/R/F1 from (weighted) counts, arrays allowed; 0 where undefined (as biocausal.evaluate.prf)."""
    tp, npred, ngold = (np.asarray(x, dtype=float) for x in (tp, npred, ngold))
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(npred > 0, tp / npred, 0.0)
        r = np.where(ngold > 0, tp / ngold, 0.0)
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return p, r, f1


def weighted_ap(score: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Average precision per row of sample weights ``w`` [B, n]; equals sklearn average_precision_score(y, score,
    sample_weight=w[b]) (step sum over distinct thresholds). 0 for a row without positive weight."""
    order = np.argsort(-score, kind="mergesort")
    s, yy, ww = score[order], y[order].astype(float), w[:, order]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]         # last index of each tied-score group
    tps = np.cumsum(ww * yy, axis=1)[:, last]
    fps = np.cumsum(ww * (1 - yy), axis=1)[:, last]
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tps + fps > 0, tps / (tps + fps), 0.0)
        rec = np.where(tps[:, -1:] > 0, tps / tps[:, -1:], 0.0)
    return np.sum(np.diff(rec, axis=1, prepend=0.0) * prec, axis=1)


def score_system(y, pred, score, doc_w) -> dict:
    """Point P/R/F1/AUPRC and bootstrap CIs of F1 and AUPRC; ``doc_w`` [B, n] = pair weights of the resamples."""
    y, pred = y.astype(bool), pred.astype(bool)
    p, r, f1 = prf((pred & y).sum(), pred.sum(), y.sum())
    ap = float(average_precision_score(y, score))
    assert abs(weighted_ap(score, y, np.ones((1, len(y))))[0] - ap) < 1e-12, "weighted_ap disagrees with sklearn"
    f1s, aps = [], []
    for i in range(0, len(doc_w), CHUNK):
        w = doc_w[i:i + CHUNK]
        f1s.append(prf(w @ (pred & y), w @ pred, w @ y)[2])
        aps.append(weighted_ap(score, y, w))
    f1s, aps = np.concatenate(f1s), np.concatenate(aps)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"p": float(p), "r": float(r), "f1": float(f1), "f1_ci": ci(f1s), "auprc": ap, "auprc_ci": ci(aps),
            "n_pred_causal": int(pred.sum())}


def load() -> tuple[list[dict], dict[str, dict]]:
    pairs = list(csv.DictReader(open(ROOT / "data/CRED/cred_pairs.tsv", encoding="utf-8"), delimiter="\t"))
    preds = {}
    for name in SYSTEMS:
        rows = {(r["pmid"], r["e1"], r["e2"]): r["probs"] for r in read_jsonl(HERE / name / "pred_cred.jsonl")}
        pred, score, label = [], [], []
        for c in pairs:
            probs = [rows[(c["pmid"], *k.split("|"))] for k in c["pairs"].split(";")] if c["pairs"] else []
            assert bool(probs) == (c["status"] == "reachable"), c
            pred.append(any(max(p, key=p.get) in CAUSAL for p in probs))
            score.append(max((p[CAUSAL[0]] + p[CAUSAL[1]] for p in probs), default=0.0))
            label.append(max(probs[0], key=probs[0].get) if len(probs) == 1 else None)
        preds[name] = {"pred": np.array(pred), "score": np.array(score), "argmax": label}
    pt3 = np.array([any(t in CAUSAL for t in c["pt3_type"].split(";")) and c["status"] == "reachable" for c in pairs])
    preds["pubtator3"] = {"pred": pt3, "score": pt3.astype(float)}
    preds["majority"] = {"pred": np.zeros(len(pairs), bool), "score": np.zeros(len(pairs))}
    return pairs, preds


def evaluate(pairs: list[dict], preds: dict[str, dict], n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    y = np.array([c["label"] == "1" for c in pairs])
    reach = np.array([c["status"] == "reachable" for c in pairs])
    same = np.array([c["same_sent"] == "1" for c in pairs])
    test = np.array([c["split"] == "test" for c in pairs])
    pmid = np.array([c["pmid"] for c in pairs])
    out: dict = {}
    for scope, s_mask in (("all", np.ones(len(pairs), bool)), ("test", test)):
        for group, g_mask in (("all", np.ones(len(pairs), bool)), ("same_sentence", same), ("cross_sentence", ~same)):
            for variant, v_mask in (("reachable", reach), ("strict", np.ones(len(pairs), bool))):
                m = s_mask & g_mask & v_mask
                docs, idx = np.unique(pmid[m], return_inverse=True)
                w_docs = np.random.default_rng(seed).multinomial(len(docs), np.full(len(docs), 1 / len(docs)), size=n_boot)
                w = w_docs[:, idx]
                res = {"n_pairs": int(m.sum()), "n_causal": int(y[m].sum()), "n_docs": len(docs),
                       "n_reachable": int((m & reach).sum()), "n_causal_reachable": int((m & reach & y).sum())}
                assert y[m].mean() < 0.5, "majority class is not non-causal"
                for name, p in preds.items():
                    res[name] = score_system(y[m], p["pred"][m], p["score"][m], w)
                out.setdefault(scope, {}).setdefault(group, {})[variant] = res
    return out


def main() -> None:
    pairs, preds = load()
    results = evaluate(pairs, preds)
    y = [c["label"] == "1" for c in pairs]
    argmax = {name: {lab: dict(Counter(a for a, yy in zip(preds[name]["argmax"], y) if a and yy == (lab == "causal")))
                     for lab in ("causal", "non_causal")} for name in SYSTEMS}
    status = Counter((c["split"] == "test", c["status"], c["label"]) for c in pairs)
    meta = {"preregistration": "docs/results/preregistration.md §8.3", "systems": SYSTEMS,
            "pred_files": {n: f"docs/results/cred/{n}/pred_cred.jsonl" for n in SYSTEMS},
            "pairs_file": "data/CRED/cred_pairs.tsv", "n_resamples": N_BOOT, "seed": SEED,
            "causal": "argmax in {Positive_Correlation, Negative_Correlation}", "score": "p(Pos)+p(Neg)",
            "pubtator3_baseline": "causal iff a PubTator3 relation of the pair has type Positive_ or Negative_Correlation; "
                                  "score = that 0/1 decision",
            "majority_baseline": "every pair non-causal; score constant, so AUPRC = prevalence",
            "status_counts": {f"{'test' if t else 'train+val'}|{s}|label={l}": n for (t, s, l), n in sorted(status.items())},
            "argmax_by_cred_label_single_gene_reachable": argmax}
    (HERE / "cred_results.json").write_text(json.dumps({"meta": meta, "results": results, "reported": REPORTED},
                                                       indent=1), encoding="utf-8")
    names = [*SYSTEMS, "pubtator3", "majority"]
    for scope in ("all", "test"):
        for variant in ("reachable", "strict"):
            print(f"\n{scope} / {variant}\n")
            print("| Подмножество | Пары (кауз.) | Док. | Система | P | R | F1 [95% CI] | AUPRC [95% CI] |")
            print("|---|---|---|---|---|---|---|---|")
            for group in ("all", "same_sentence", "cross_sentence"):
                r = results[scope][group][variant]
                for n in names:
                    s = r[n]
                    pct = lambda v: f"{100 * v:.1f}"
                    print(f"| {group} | {r['n_pairs']} ({r['n_causal']}) | {r['n_docs']} | {n} | {pct(s['p'])} | "
                          f"{pct(s['r'])} | {pct(s['f1'])} [{pct(s['f1_ci'][0])}–{pct(s['f1_ci'][1])}] | "
                          f"{pct(s['auprc'])} [{pct(s['auprc_ci'][0])}–{pct(s['auprc_ci'][1])}] |")
    print("\nargmax (single-gene reachable pairs, all CRED):", json.dumps(argmax))
    print("status:", json.dumps(meta["status_counts"]))


if __name__ == "__main__":
    main()
