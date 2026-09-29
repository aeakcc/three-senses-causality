"""Paper 2, E3 (preregistration §11a item 2): out-of-fold scores of the CPU arms on the 159-abstract CRED pool.
PL / P0 = linear probe on frozen pair embeddings (embed.py of bc8all s42-44 / raw BioLinkBERT seeds 42-44); S = the
§10b stacker refit per fold.

    PYTHONPATH=. .venv/bin/python docs/results/paper2/e3_probe.py PL --npz BC8ALL_S42.npz BC8ALL_S43.npz BC8ALL_S44.npz
    PYTHONPATH=. .venv/bin/python docs/results/paper2/e3_probe.py P0 --npz RAW_S42.npz RAW_S43.npz RAW_S44.npz
    PYTHONPATH=. .venv/bin/python docs/results/paper2/e3_probe.py S
      reads  data/CRED/cred_pairs.tsv, data/CRED/cv/fold{0..4}.txt; PL/P0: one embed.py npz per seed over
             data/CRED/cred_docs.json (X, pmid, e1 = disease, e2 = gene); S: docs/results/cred/ens_{biolinkbert_base,
             bc8all}/pred_cred.jsonl (cred_a1_stacker.PRED)
      writes docs/results/paper2/e3/oof_<ARM>.jsonl, one line per pool row {pmid, gene_id, disease_id, label, score,
             fold, threshold}, and oof_<ARM>.meta.json (per fold: sizes, threshold, C and inner AUPRC per seed; SHA-256)

Rows: cred_s10.load_rows of every split. Pool = PMIDs of the fold files; the other 107 PMIDs (train abstracts whose
labelled pairs are all causal) only ever train. Fold k trains on the reachable single-candidate rows outside fold k
(PL/P0: with the 107, whose rows are all causal; S: pool only, §10b) and scores every pool row of fold k strictly:
unreachable -> 0, a composite gene id -> max over its candidate pairs (as cred_s10.scores). Model: StandardScaler + L2
LogisticRegression (lbfgs); C by pooled AUPRC of the inner GroupKFold(4)-by-PMID OOF over the training rows, ties ->
smaller C; PL/P0: C in 1e-4..1, max_iter 2000; S: cred_a1_stacker's C grid, features and max_iter 1000. Every seed
picks its own C; test and inner-OOF probabilities are averaged over seeds; threshold = max F1 on the averaged inner
OOF, ties -> highest (cred_s10.best_threshold).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "docs/results/cred")]

from cred_a1_stacker import C_GRID as C_GRID_S, PRED, features, read_probs  # noqa: E402
from cred_s10 import best_threshold, load_rows  # noqa: E402

CV, OUT = ROOT / "data/CRED/cv", HERE / "e3"
C_GRID, K = (1e-4, 1e-3, 1e-2, 1e-1, 1.0), 5


def pool(cv: Path = CV) -> tuple[list[dict], dict[str, int]]:
    """(cred_s10.load_rows of every split, {pool PMID: its test fold})."""
    rows = [r for s in ("train", "val", "test") for r in load_rows(s)]
    fold = {p: k for k in range(K) for p in (cv / f"fold{k}.txt").read_text(encoding="utf-8").split()}
    outside = {r["pmid"] for r in rows} - fold.keys()
    assert len(outside) == 107 and all(r["label"] == "1" for r in rows if r["pmid"] in outside), len(outside)
    return rows, fold


def npz_features(path: Path):
    """rows with one candidate pair 'e1|e2' -> their embed.py rows of X."""
    z = np.load(path)
    at = {k: i for i, k in enumerate(zip(z["pmid"].tolist(), z["e1"].tolist(), z["e2"].tolist()))}
    X = z["X"].astype(np.float32)
    return lambda rows: X[[at[(r["pmid"], *r["pairs"].split("|"))] for r in rows]]


def fit_lr(X, y, groups, grid, max_iter: int):
    """(model refit on every row, inner OOF at the chosen C, C, inner AUPRC per C); GroupKFold(4) by PMID."""
    lr = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter))  # noqa: E731
    folds = list(GroupKFold(n_splits=4).split(X, y, groups))
    oof = {C: np.zeros(len(y)) for C in grid}
    for C in grid:
        for tr, te in folds:
            oof[C][te] = lr(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    ap = {C: float(average_precision_score(y, oof[C])) for C in grid}
    C = max(grid, key=lambda c: (ap[c], -c))
    return lr(C).fit(X, y), oof[C], C, ap


def cross_fit(rows: list[dict], fold: dict[str, int], feats: list, grid, max_iter: int, pool_only: bool):
    """(score, threshold) per row (0 outside the pool) and the per-fold log; ``feats`` = one feature map per seed."""
    y = np.array([r["label"] == "1" for r in rows])
    score, thr, log = np.zeros(len(rows)), np.zeros(len(rows)), []
    for k in range(K):
        train = [i for i, r in enumerate(rows) if r["status"] == "reachable" and ";" not in r["pairs"]
                 and fold.get(r["pmid"], -1) != k and (r["pmid"] in fold or not pool_only)]
        test = [i for i, r in enumerate(rows) if fold.get(r["pmid"]) == k]
        cand = [(i, c) for i in test for c in rows[i]["pairs"].split(";") if c]
        tr_rows, te_rows = [rows[i] for i in train], [{**rows[i], "pairs": c} for i, c in cand]
        oofs, probs, fits = [], [], []
        for f in feats:
            model, oof, C, ap = fit_lr(f(tr_rows), y[train], [r["pmid"] for r in tr_rows], grid, max_iter)
            oofs.append(oof)
            probs.append(model.predict_proba(f(te_rows))[:, 1] if cand else np.zeros(0))
            fits.append({"C": C, "inner_auprc": ap})
        t, f1 = best_threshold(y[train], np.mean(oofs, 0))
        np.maximum.at(score, np.array([i for i, _ in cand], dtype=int), np.mean(probs, 0))
        thr[test] = t
        log.append({"fold": k, "n_train": len(train), "n_train_causal": int(y[train].sum()),
                    "n_train_docs": len({r["pmid"] for r in tr_rows}), "n_test": len(test),
                    "n_test_causal": int(y[test].sum()), "threshold": t, "inner_oof_f1": f1, "fits": fits})
    return score, thr, log


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm", choices=("PL", "P0", "S"))
    ap.add_argument("--npz", nargs="+", type=Path, default=[], help="PL/P0: one embed.py output per seed")
    ap.add_argument("--cv", type=Path, default=CV, help="dir of fold{k}.txt")
    ap.add_argument("--out", type=Path, help="default docs/results/paper2/e3/oof_<ARM>.jsonl")
    a = ap.parse_args(argv)
    if (a.arm == "S") == bool(a.npz):
        ap.error("PL/P0 need --npz (one per seed); S takes none")
    rows, fold = pool(a.cv)
    if a.arm == "S":
        probs = {k: read_probs(p) for k, p in PRED.items()}
        inputs, feats, grid, it = list(PRED.values()), [lambda rs: features(rs, probs)], C_GRID_S, 1000
    else:
        inputs, feats, grid, it = a.npz, [npz_features(p) for p in a.npz], C_GRID, 2000
    score, thr, log = cross_fit(rows, fold, feats, grid, it, pool_only=a.arm == "S")
    out = a.out or OUT / f"oof_{a.arm}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if r["pmid"] in fold:
                f.write(json.dumps({"pmid": r["pmid"], "gene_id": r["gene_id"], "disease_id": r["disease_id"],
                                    "label": int(r["label"]), "score": float(score[i]), "fold": fold[r["pmid"]],
                                    "threshold": float(thr[i])}) + "\n")
    meta = {"arm": a.arm, "c_grid": list(grid), "max_iter": it, "folds": log,
            "inputs_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs}}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(out, sum(r["pmid"] in fold for r in rows), "rows;",
          [(x["fold"], [s["C"] for s in x["fits"]], round(x["threshold"], 4)) for x in log])


if __name__ == "__main__":
    main(sys.argv[1:])
