"""Preregistration §10b, arm A1: logistic stacker on frozen zero-shot outputs of Base and bc8all (CPU, no CRED test).

    PYTHONPATH=. .venv/bin/python docs/results/cred/cred_a1_stacker.py fit
      reads  data/CRED/cred_pairs.tsv, docs/results/cred/ens_{biolinkbert_base,bc8all}/pred_cred.jsonl
      writes docs/results/cred/s10_a1.json (features, C, weights, scaler, threshold, OOF metrics, input SHA-256)

Features of a reachable CRED pair (score p = probability of its candidate pair in the ensemble file):
  log(p_Pos + p_Neg), log p_Association, log p_None of Base; log(p_Pos + p_Neg) of bc8all; same-sentence flag.
Fit rows: reachable single-candidate pairs of train (without the 107 train abstracts where every labelled pair is
causal) and val, keys merged as in cred_s10.load_rows. Model: StandardScaler + L2 LogisticRegression (lbfgs), C in
{0.001, 0.01, 0.1, 1, 10} chosen by pooled out-of-fold AUPRC over GroupKFold(4) by PMID; threshold = max pooled OOF F1
(ties -> highest, cred_s10.best_threshold); refit on all fit rows. Test: unreachable pairs score 0 (strict).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from cred_s10 import best_threshold, load_rows  # noqa: E402

PRED = {"base": HERE / "ens_biolinkbert_base" / "pred_cred.jsonl", "bc8all": HERE / "ens_bc8all" / "pred_cred.jsonl"}
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
OUT = HERE / "s10_a1.json"
EPS = 1e-6


def read_probs(path: Path) -> dict[tuple, dict]:
    with open(path, encoding="utf-8") as f:
        return {(p["pmid"], p["e1"], p["e2"]): p["probs"] for p in map(json.loads, f)}


def features(rows: list[dict], probs: dict[str, dict]) -> np.ndarray:
    out = []
    for r in rows:
        k = (r["pmid"], *r["pairs"].split("|"))
        b, c = probs["base"][k], probs["bc8all"][k]
        out.append([math.log(b["Positive_Correlation"] + b["Negative_Correlation"] + EPS), math.log(b["Association"] + EPS),
                    math.log(b["None"] + EPS), math.log(c["Positive_Correlation"] + c["Negative_Correlation"] + EPS),
                    float(r["same_sent"] == "1")])
    return np.array(out)


def fit_rows() -> list[dict]:
    train = load_rows("train")
    labels: dict[str, set] = {}
    for r in train:
        labels.setdefault(r["pmid"], set()).add(r["label"])
    positive_only = {p for p, v in labels.items() if v == {"1"}}
    assert len(positive_only) == 107, len(positive_only)
    rows = [r for r in train if r["pmid"] not in positive_only] + load_rows("val")
    return [r for r in rows if r["status"] == "reachable" and ";" not in r["pairs"]]


def fit() -> dict:
    rows = fit_rows()
    probs = {k: read_probs(p) for k, p in PRED.items()}
    X, y = features(rows, probs), np.array([r["label"] == "1" for r in rows])
    groups = np.array([r["pmid"] for r in rows])
    folds = list(GroupKFold(n_splits=4).split(X, y, groups))

    def oof(C: float) -> np.ndarray:
        s = np.zeros(len(y))
        for tr, te in folds:
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(C=C, max_iter=1000).fit(sc.transform(X[tr]), y[tr])
            s[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        return s

    ap = {C: float(average_precision_score(y, oof(C))) for C in C_GRID}
    C = max(C_GRID, key=lambda c: (ap[c], -c))
    s = oof(C)
    t, f1 = best_threshold(y, s)
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, max_iter=1000).fit(sc.transform(X), y)
    return {"features": ["log p_causal base", "log p_Association base", "log p_None base", "log p_causal bc8all",
                         "same_sentence"],
            "n_rows": len(y), "n_causal": int(y.sum()), "n_docs": len(set(groups)), "oof_auprc_by_C": ap, "C": C,
            "threshold": t, "oof_f1_at_threshold": f1, "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
            "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
            "inputs_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in PRED.items()}}


def apply(model: dict, rows: list[dict]) -> np.ndarray:
    """Stacker score per CRED row; unreachable pairs score 0 (strict)."""
    probs = {k: read_probs(p) for k, p in PRED.items()}
    reach = [i for i, r in enumerate(rows) if r["status"] == "reachable"]
    s = np.zeros(len(rows))
    if reach:
        z = (features([rows[i] for i in reach], probs) - model["scaler_mean"]) / model["scaler_scale"]
        s[reach] = 1 / (1 + np.exp(-(z @ np.array(model["coef"]) + model["intercept"])))
    return s


if __name__ == "__main__":
    if sys.argv[1:] != ["fit"]:
        sys.exit(__doc__)
    res = fit()
    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print({k: res[k] for k in ("n_rows", "n_causal", "n_docs", "oof_auprc_by_C", "C", "threshold", "oof_f1_at_threshold")})
