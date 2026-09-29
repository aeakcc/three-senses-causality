"""Paper 2, E5 evaluation (sense V): which ClinGen evidence category (biocausal.data.EV_CLASSES) a (PMID, curated
gene-disease pair) unit is cited for. Systems V0, VL, VP (ev-head ensembles), TFIDF, MAJORITY on TEST_2026 and RGD_EXT.

    PYTHONPATH=. .venv/bin/python docs/results/paper2/e5_eval.py [--runs DIR] [--e5 DIR] [--out-dir DIR]
      reads  E5/train_units.tsv, clingen.json, {test_2026,rgd_ext}_{units.tsv,docs.json} (e5_sets.py; default
             data/paper2/e5) and RUNS/_p2e5_{v0,vl,vp}/pred_{test_2026,rgd_ext}.jsonl (ops/make_queue_p2_e5.py, copied
             from the PC; default runs/)
      writes OUT-DIR/e5.json, e5.md (default docs/results/paper2/e5)

Score vector over EV_CLASSES per unit: V0 / VL / VP = probs_ev of the ensemble row of the unit's single candidate pair
(extra.pair); a pred file that is absent or misses a unit makes the system 'missing'. TFIDF = TfidfVectorizer(ngram_range
(1, 2), min_df 2, sublinear_tf) on the unit doc's title + abstract + LogisticRegression (lbfgs, multinomial, max_iter
2000) fit on the train units, C in {0.1, 1, 10} by macro one-vs-rest AUROC of the pooled GroupKFold(5)-by-PMID
out-of-fold probabilities, ties -> smaller C. MAJORITY = the train class prior.
TEST_2026: one-vs-rest AUROC per class (a unit is positive for each of its classes, ties 0.5), macro = their mean
(primary); macro-F1 of the argmax on the single-class units (F1 = 2tp / (2tp + fp + fn), 0 when that is 0 / 0; mean over
the 5 classes). RGD_EXT: the AUROCs over genetic, expression, model_rescue, each score vector restricted to those 3
classes and renormalized to sum 1. Bootstrap: PMIDs with replacement, 10 000 resamples, default_rng(0) (e1_eval.boot),
the same draws for every system of a set; CI = 2.5 / 97.5 percentiles over the resamples where the statistic is defined.
Family (Holm, m = 2, alpha 0.05) on TEST_2026: H_E5a AUROC_macro(V0) - AUROC_macro(TFIDF) > 0, p = share of resampled
differences <= 0; H_E5b AUROC_macro(VL) - AUROC_macro(V0) != 0, p = min(1, 2 min(share <= 0, share >= 0)); a missing
system gives p = 1. Every other pair of systems: difference with CI, no test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / "docs/results/posthoc"), str(ROOT / "docs/results/resolution")]

from biocausal.data import EV_CLASSES  # noqa: E402
from biocausal.evaluate import read_jsonl  # noqa: E402
from causal_audit import N_BOOT, SEED, ci  # noqa: E402
from e1_eval import auroc, boot  # noqa: E402
from resolution import holm  # noqa: E402
from units import read_units  # noqa: E402

K = len(EV_CLASSES)
ARMS = {"V0": "v0", "VL": "vl", "VP": "vp"}
SETS = {"test_2026": EV_CLASSES, "rgd_ext": ("genetic", "expression", "model_rescue")}
FAMILY = {"H_E5a": ("V0", "TFIDF", "greater"), "H_E5b": ("VL", "V0", "two-sided")}
C_GRID = (0.1, 1.0, 10.0)


def texts(units: list[dict], docs_path: Path) -> np.ndarray:
    docs = {d["pmid"]: d for d in json.loads(docs_path.read_text(encoding="utf-8"))}
    return np.array([f"{docs[u['doc_id']]['title']} {docs[u['doc_id']]['abstract']}" for u in units], dtype=object)


def tfidf(x: np.ndarray, y: np.ndarray, groups: list[str]):
    """(predict: texts -> [n, K] probabilities, {C, OOF macro AUROC per C}) of the model refit on all of x at the C
    with the best pooled GroupKFold(5) out-of-fold macro one-vs-rest AUROC (ties -> smaller C)."""
    model = lambda C: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),  # noqa: E731
                                    LogisticRegression(C=C, max_iter=2000))
    auc = {}
    for C in C_GRID:
        oof = np.zeros((len(y), K))
        for tr, te in GroupKFold(n_splits=5).split(x, y, groups):
            m = model(C).fit(x[tr], y[tr])
            oof[np.ix_(te, m.classes_)] = m.predict_proba(x[te])
        auc[C] = float(np.mean([roc_auc_score(y == k, oof[:, k]) for k in range(K)]))
    C = max(C_GRID, key=lambda c: (auc[c], -c))
    m = model(C).fit(x, y)

    def predict(t: np.ndarray) -> np.ndarray:
        p = np.zeros((len(t), K))
        p[:, m.classes_] = m.predict_proba(t)
        return p
    return predict, {"C": C, "oof_macro_auroc": auc}


def arm_scores(path: Path, units: list[dict]):
    """[n, K] probs_ev of each unit's pair, or why the system is missing."""
    if not path.is_file():
        return f"no {path}"
    got = {(r["pmid"], r["e1"], r["e2"]): r.get("probs_ev") for r in read_jsonl(path)}
    p = [got.get((u["doc_id"], *u["extra"]["pair"])) for u in units]
    bad = sum(v is None or len(v) != K for v in p)
    return f"{path}: {bad} of {len(units)} units without probs_ev over {K} classes" if bad else np.array(p, float)


def macro_f1(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Macro-F1 over the K classes (0 for a class with 2tp + fp + fn = 0) per row of unit weights w [B, n]."""
    f = []
    for c in range(K):
        tp, fp, fn = (w @ m.astype(float) for m in ((pred == c) & (y == c), (pred == c) & (y != c), (pred != c) & (y == c)))
        d = 2 * tp + fp + fn
        f.append(np.divide(2 * tp, d, out=np.zeros_like(d), where=d > 0))
    return np.mean(f, 0)


def p_two(d: np.ndarray) -> float:
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


def evaluate_set(units: list[dict], scores: dict, classes) -> tuple[dict, dict[str, np.ndarray]]:
    """(results of one set, {system: resampled macro AUROC}); ``scores``: {system: [n, K] array or missing reason}."""
    idx = [EV_CLASSES.index(c) for c in classes]
    Y = np.array([[c in u["classes"] for c in classes] for u in units], dtype=bool)
    pmid = np.array([u["pmid"] for u in units])
    one = Y.sum(1) == 1
    res: dict = {"classes": list(classes), "n_units": len(units), "n_pmids": len(set(pmid)),
                 "n_positive": dict(zip(classes, Y.sum(0).tolist())), "n_single_class": int(one.sum()), "systems": {}}
    boots, points = {}, {}
    for name, S in scores.items():
        if isinstance(S, str):
            res["systems"][name] = {"missing": S}
            continue
        S = S[:, idx] / S[:, idx].sum(1, keepdims=True) if len(idx) < K else S
        fn = lambda w: np.stack([auroc(S[:, j], Y[:, j], w) for j in range(len(idx))], 1)  # noqa: E731,B023
        point, b = fn(np.ones((1, len(units))))[0], boot(pmid, fn)
        boots[name], points[name] = b.mean(1), float(point.mean())
        ok = ~np.isnan(boots[name])
        r = {"auroc_macro": points[name], "auroc_macro_ci": ci(boots[name][ok]), "n_undefined": int((~ok).sum()),
             "auroc": {c: {"auroc": float(point[j]), "ci": ci(b[~np.isnan(b[:, j]), j])} for j, c in enumerate(classes)}}
        if len(idx) == K:
            y, pred = Y[one].argmax(1), S[one].argmax(1)
            f = lambda w: macro_f1(y, pred, w)  # noqa: E731,B023
            r.update(macro_f1=float(f(np.ones((1, int(one.sum()))))[0]), macro_f1_ci=ci(boot(pmid[one], f)))
        res["systems"][name] = r
    res["differences"] = {}
    for a, b in combinations(boots, 2):
        d = boots[a] - boots[b]
        d = d[~np.isnan(d)]
        res["differences"][f"{a}-{b}"] = {"auroc_macro": points[a] - points[b], "ci": ci(d),
                                          "p_le0": float(np.mean(d <= 0)), "p_two_sided": p_two(d)}
    return res, boots


def family(boots: dict[str, np.ndarray]) -> dict:
    fam = {}
    for h, (a, b, alt) in FAMILY.items():
        d = boots[a] - boots[b] if a in boots and b in boots else None
        d = None if d is None else d[~np.isnan(d)]
        fam[h] = {"diff": f"AUROC_macro({a}) - AUROC_macro({b}) on test_2026", "alternative": alt,
                  "missing": d is None,
                  "p": 1.0 if d is None else float(np.mean(d <= 0)) if alt == "greater" else p_two(d)}
    adj, rej = holm([f["p"] for f in fam.values()])
    for f, pa, r in zip(fam.values(), adj, rej):
        f.update(p_holm=pa, reject=r)
    return fam


def markdown(res: dict) -> str:
    f = lambda x, c: f"{x:.3f} [{c[0]:.3f}; {c[1]:.3f}]"  # noqa: E731
    L = [f"# E5: evidence type (sense V); TFIDF C = {res['tfidf']['C']}", ""]
    for tag, s in res["sets"].items():
        L += [f"## {tag}: {s['n_units']} units, {s['n_pmids']} PMIDs, {s['n_single_class']} single-class", "",
              "Positives: " + ", ".join(f"{c} {n}" for c, n in s["n_positive"].items()), "",
              "| System | AUROC macro [95 % CI] | " + " | ".join(s["classes"]) + (" | macro-F1 [95 % CI] |"
                                                                             if tag == "test_2026" else " |"),
              "|---|---|" + "---|" * (len(s["classes"]) + (tag == "test_2026"))]
        for n, r in s["systems"].items():
            if "missing" in r:
                L.append(f"| {n} | missing: {r['missing']} |")
                continue
            L.append(f"| {n} | {f(r['auroc_macro'], r['auroc_macro_ci'])} | "
                     + " | ".join(f(v["auroc"], v["ci"]) for v in r["auroc"].values())
                     + (f" | {f(r['macro_f1'], r['macro_f1_ci'])} |" if "macro_f1" in r else " |"))
        L += ["", "| A - B | ΔAUROC macro [95 % CI] | p (≤ 0) | p two-sided |", "|---|---|---|---|"]
        L += [f"| {k} | {f(d['auroc_macro'], d['ci'])} | {d['p_le0']:.4f} | {d['p_two_sided']:.4f} |"
              for k, d in s["differences"].items()]
        L.append("")
    L += ["## Confirmatory family (Holm, m = 2)", "", "| H | Difference | Alternative | p | p Holm | Reject |",
          "|---|---|---|---|---|---|"]
    L += [f"| {h} | {x['diff']} | {x['alternative']} | {x['p']:.4f}{' (missing)' if x['missing'] else ''} | "
          f"{x['p_holm']:.4f} | {x['reject']} |" for h, x in res["family"].items()]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs", help="dir holding _p2e5_{v0,vl,vp}/")
    ap.add_argument("--e5", type=Path, default=ROOT / "data/paper2/e5", help="e5_sets.py output dir")
    ap.add_argument("--out-dir", type=Path, default=HERE / "e5")
    a = ap.parse_args(argv)
    train = read_units(a.e5 / "train_units.tsv")
    y = np.array([EV_CLASSES.index(u["classes"][0]) for u in train])
    predict, tf = tfidf(texts(train, a.e5 / "clingen.json"), y, [u["pmid"] for u in train])
    prior = np.bincount(y, minlength=K) / len(y)
    read = [a.e5 / "train_units.tsv", a.e5 / "clingen.json"]
    res: dict = {"tfidf": tf, "majority_prior": dict(zip(EV_CLASSES, prior.tolist())), "sets": {}}
    boots = {}
    for tag, classes in SETS.items():
        units = read_units(a.e5 / f"{tag}_units.tsv")
        preds = {s: a.runs / f"_p2e5_{arm}" / f"pred_{tag}.jsonl" for s, arm in ARMS.items()}
        scores = {s: arm_scores(p, units) for s, p in preds.items()}
        scores.update(TFIDF=predict(texts(units, a.e5 / f"{tag}_docs.json")), MAJORITY=np.tile(prior, (len(units), 1)))
        res["sets"][tag], boots[tag] = evaluate_set(units, scores, classes)
        read += [a.e5 / f"{tag}_units.tsv", a.e5 / f"{tag}_docs.json"] + [p for p in preds.values() if p.is_file()]
    res["family"] = family(boots["test_2026"])
    res["meta"] = {"ev_classes": EV_CLASSES, "n_boot": N_BOOT, "seed": SEED, "runs": str(a.runs), "e5": str(a.e5),
                   "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in read}}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "e5.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    (a.out_dir / "e5.md").write_text(markdown(res), encoding="utf-8")
    print(markdown(res))
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
