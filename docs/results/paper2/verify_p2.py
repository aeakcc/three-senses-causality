r"""Independent recomputation of the paper 2 preregistered statistics (preregistration.md §11a E3, §11b E1-E4, §11c E5),
written from the specification only (e1_eval.py, e2_eval.py, e3_eval.py, e3_probe.py, e5_eval.py were not read).

    PYTHONPATH=. .venv/bin/python docs/results/paper2/verify_p2.py e1 --pred-root RUNS
        RUNS/_p2e1_{L,P,Lp}/pred_{clingen_docs,rgd_docs,percha_docs,swap_clingen_u}.jsonl
    ... verify_p2.py e3 --arm Z=docs/results/cred/ens_bc8all/pred_cred.jsonl --arm C=same_sent \
                        --arm S=docs/results/paper2/e3/oof_S.jsonl --arm PL=oof_PL.jsonl --arm P0=oof_P0.jsonl \
                        --arm 'FL=RUNS/p2e3_f1_k*_s*/pred_cv.jsonl' --arm 'F0=RUNS/p2e3_f0_k*_s*/pred_cv.jsonl' ...
    ... verify_p2.py e5 --pred-root RUNS      RUNS/_p2e5_{v0,vl,vp}/pred_{test_2026,rgd_ext}.jsonl (TFIDF, MAJORITY fit here)
Each writes OUT_DIR/{e1,e3,e5}.json (default docs/results/paper2/verify/) and prints a table. --b sets the resamples.

Conventions (the "notes" of each JSON list the choices §11 leaves open):
* AUROC = P(s_A > s_B) + 0.5 P(tie); AP = sum_k (R_k - R_{k-1}) P_k over distinct scores (sklearn's step-wise AP).
  Both take multiset weights, so a bootstrap resample is scored by weighting each row by how often its cluster was drawn.
* Cluster bootstrap: K clusters, each resample draws K cluster indices with default_rng(0).integers; percentile 95 % CI.
  Every statistic (or paired family) starts its own default_rng(0). One-sided p = share of resamples <= null;
  two-sided p = min(1, 2 min(share <= 0, share >= 0)). Holm step-down over the family.
* A unit's candidate pairs: its gene ids present in the doc x the doc's diseases in disease_mesh, or all the doc's
  diseases when none is in it; E1/E5 score only units with exactly one candidate pair.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "docs" / "results" / "cred")]
from biocausal.data import EV_CLASSES  # noqa: E402
from cred_s10 import load_rows  # noqa: E402  (§11a: the E3 rows ARE cred_s10.load_rows, restricted to the pool)

P2, CRED = ROOT / "data" / "paper2", ROOT / "data" / "CRED"
B, EPS, ALPHA = 10_000, 1e-6, 0.05
GENE, DIS = "GeneOrGeneProduct", "DiseaseOrPhenotypicFeature"
CLV, EXP, MUTC, UP, KO, IEP = {"GEN_CLV"}, {"EXP_A", "EXP_B", "EXP_U"}, {"MUT", "CAUSE"}, {"UP"}, {"IMP", "IGI"}, {"IEP"}


# ---------------------------------------------------------------- statistics
def _auc(g, y, w):
    """AUROC from score ranks g (np.unique inverse, ascending), bool labels y and row weights w; ties count 1/2."""
    pw, nw = np.bincount(g, w * y), np.bincount(g, w * ~y)
    den = pw.sum() * nw.sum()
    return float(pw @ (np.cumsum(nw) - nw / 2) / den) if den else np.nan


def _ap(g, y, w):
    """Step-wise average precision from descending-score ranks g (np.unique inverse of -score) and row weights w."""
    tp, fp = np.cumsum(np.bincount(g, w * y)), np.cumsum(np.bincount(g, w * ~y))
    if not tp[-1]:
        return np.nan
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    return float(np.diff(tp, prepend=0) @ prec / tp[-1])


def auroc(s, y, w=None):
    y = np.asarray(y, bool)
    return _auc(np.unique(np.asarray(s, float), return_inverse=True)[1], y, np.ones(len(y)) if w is None else np.asarray(w, float))


def ap(s, y, w=None):
    y = np.asarray(y, bool)
    return _ap(np.unique(-np.asarray(s, float), return_inverse=True)[1], y, np.ones(len(y)) if w is None else np.asarray(w, float))


def boot(stats, clusters, b=B, seed=0):
    """(b, len(stats)): every stat(w) on the same b cluster resamples; w[i] = times row i's cluster was drawn."""
    rng = np.random.default_rng(seed)
    cid = np.unique(np.asarray(clusters), return_inverse=True)[1]
    k = int(cid.max()) + 1
    out = np.empty((b, len(stats)))
    for i in range(b):
        w = np.bincount(rng.integers(0, k, k), minlength=k)[cid].astype(float)
        out[i] = [f(w) for f in stats]
    return out


def ci(x):
    return [float(v) for v in np.nanpercentile(x, [2.5, 97.5])]


def p_one(x, null=0.0):
    return float(np.mean(x <= null))


def p_two(x):
    return float(min(1.0, 2 * min(np.mean(x <= 0), np.mean(x >= 0))))


def holm(p):
    p = np.asarray(p, float)
    m, o = len(p), np.argsort(p, kind="stable")
    out = np.empty(m)
    out[o] = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[o]))
    return out


def macro_f1(t, p, k):
    f = [2 * ((t == c) & (p == c)).sum() / max(1, (t == c).sum() + (p == c).sum()) for c in range(k)]
    return float(np.mean(f))


def family(defs, names, est, bs, bad=()):
    """Holm family of paired differences (hyp, arm, reference, sided); a missing or dropped arm gives p = 1."""
    out = {}
    for h, x, z, sided in defs:
        out[h] = {"test": f"{x} - {z} {'> 0' if sided == 1 else '!= 0'}"}
        if {x, z} <= set(names) and not {x, z} & set(bad):
            d = bs[:, names.index(x)] - bs[:, names.index(z)]
            out[h].update(diff=est[x] - est[z], ci95=ci(d), p=p_one(d) if sided == 1 else p_two(d))
        else:
            out[h].update(diff=None, ci95=None, p=1.0, note="arm missing or dropped -> p = 1")
    for h, q in zip(out, holm([v["p"] for v in out.values()])):
        out[h].update(p_holm=float(q), reject=bool(q <= ALPHA))
    return out


def diffs(names, est, bs):
    return {f"{a} - {z}": {"diff": est[a] - est[z], "ci95": ci(bs[:, i] - bs[:, j])}
            for i, a in enumerate(names) for j, z in enumerate(names) if i < j}


# ---------------------------------------------------------------- data
def read_tsv(path):
    return list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split(v):
    return {x for x in v.split(";") if x}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_docs(path):
    return {d["pmid"]: d for d in json.load(open(path, encoding="utf-8"))}


def candidates(doc, genes, mesh):
    """(disease, gene) candidate pairs of a unit: its gene ids in the doc x the doc's diseases in mesh (all of the doc's
    diseases when none is in mesh)."""
    ents = doc["entities"]
    dis = sorted(e for e, v in ents.items() if v["type"] == DIS)
    dis = [d for d in dis if d in mesh] or dis
    return [(d, g) for g in sorted(e for e, v in ents.items() if v["type"] == GENE and e in genes) for d in dis]


def lp(r):
    return r["probs"]["Positive_Correlation"] + r["probs"]["Negative_Correlation"]


def logit(s):
    s = np.clip(np.asarray(s, float), EPS, 1 - EPS)
    return np.log(s / (1 - s))


def dump(out_dir, name, res):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, f"{name}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"-> {Path(out_dir, name + '.json')}")


def fmt(x):
    return "   -  " if x is None or x != x else f"{x:.4f}"


# ---------------------------------------------------------------- E1 / E4 / E2 (§11b)
E1_SETS = {"clingen_docs": (P2 / "e1" / "clingen_units.tsv", P2 / "e1" / "clingen_docs.json"),
           "rgd_docs": (P2 / "e1" / "rgd_units.tsv", P2 / "e1" / "rgd_docs.json"),
           "percha_docs": (P2 / "percha_units.tsv", P2 / "percha_docs.json"),
           "swap_clingen_u": (P2 / "e1" / "clingen_units.tsv", P2 / "e1" / "swap_clingen_u.json")}
READERS = ("L", "P", "Lp")
HYP = [("H1", "clingen_docs", CLV, EXP), ("H2", "rgd_docs", KO, IEP), ("H3", "percha_docs", MUTC, UP),
       ("H4", "swap_clingen_u", CLV, EXP)]
CROSS = [("L", "clingen_docs", EXP, CLV), ("P", "clingen_docs", CLV, EXP), ("L", "percha_docs", UP, MUTC),
         ("P", "percha_docs", MUTC, UP), ("P", "rgd_docs", KO, IEP), ("L", "rgd_docs", KO, IEP)]
E1_NOTES = ["d = logit(clip(sP)) - logit(clip(sLp)), clip [1e-6, 1-1e-6]; s = p(Pos)+p(Neg)",
            "per-reader crossover AUROCs use the raw s (no clip), direction A vs B as listed",
            "L on RGD (IMP+IGI vs IEP): no predicted direction (§11b item 4), reported as an estimate",
            "a unit in both A and B is dropped from that contrast; units share no dedup (RGD has 9 pairs scored by 2 units"
            " each, none in IMP/IGI/IEP)",
            "swap units: mapping.tsv rows with file == clingen_u; the unit is the E1 ClinGen unit of (orig_doc_id, gene),"
            " its candidate pair is recomputed on the swapped doc; cluster = the original PMID"]


def e1_units(tag, swap):
    units_path, docs_path = E1_SETS[tag]
    units, docs, n = read_tsv(units_path), load_docs(docs_path), Counter()
    if tag.startswith("swap"):
        by = defaultdict(list)
        for u in units:
            for g in split(u["gene_entrez"]):
                by[u["doc_id"], g].append(u)
        units = []
        for doc_id in docs:
            hit = by.get(swap.get(doc_id), [])
            n["swap_no_unit"] += not hit
            n["swap_several_units"] += len(hit) > 1
            units += [{**u, "doc_id": doc_id} for u in hit]
    return units, docs, n


def e1_rows(units, docs, preds):
    rows, n = [], Counter()
    for u in units:
        c = candidates(docs[u["doc_id"]], split(u["gene_entrez"]), split(u["disease_mesh"]))
        if len(c) != 1:
            n["not_single_pair"] += 1
            continue
        key = (docs[u["doc_id"]]["pmid"], *c[0])
        if any(key not in p for p in preds.values()):
            n["missing_prediction"] += 1
            continue
        rows.append({"pmid": u["pmid"], "cls": split(u["classes"]), **{r: p[key] for r, p in preds.items()}})
    return rows, n


def e1_contrast(rows, a, b, score, nb):
    keep = [r for r in rows if bool(r["cls"] & a) != bool(r["cls"] & b)]
    y = np.array([bool(r["cls"] & a) for r in keep])
    g = np.unique(np.array([score(r) for r in keep], float), return_inverse=True)[1]
    bs = boot([lambda w: _auc(g, y, w)], [r["pmid"] for r in keep], nb)[:, 0]
    return {"n_a": int(y.sum()), "n_b": int((~y).sum()), "n_pmids": len({r["pmid"] for r in keep}),
            "n_both_dropped": sum(bool(r["cls"] & a) and bool(r["cls"] & b) for r in rows),
            "auroc": _auc(g, y, np.ones(len(y))), "ci95": ci(bs), "p": p_one(bs, 0.5)}


def cmd_e1(a):
    swap = {r["doc_id"]: (r["orig_doc_id"], r["gene"]) for r in read_tsv(a.swap_map) if r["file"] == "clingen_u"}
    res = {"units": {}, "hypotheses": {}, "crossover": {}, "inputs": {}, "notes": E1_NOTES}
    tables = {}
    for tag in E1_SETS:
        files = {r: Path(a.pred_root, f"_p2e1_{r}", f"pred_{tag}.jsonl") for r in READERS}
        if not all(f.exists() for f in files.values()):
            res["units"][tag] = {"missing_files": [str(f) for f in files.values() if not f.exists()]}
            continue
        units, docs, n = e1_units(tag, swap)
        preds = {r: {(x["pmid"], x["e1"], x["e2"]): lp(x) for x in read_jsonl(f)} for r, f in files.items()}
        tables[tag], m = e1_rows(units, docs, preds)
        res["units"][tag] = {"n_units": len(units), "n_scored": len(tables[tag]), **n, **m}
        res["inputs"].update({str(f): sha(f) for f in files.values()})
    for h, tag, ca, cb in HYP:
        if tag in tables:
            res["hypotheses"][h] = {"set": tag, **e1_contrast(tables[tag], ca, cb,
                                                             lambda r: logit(r["P"]) - logit(r["Lp"]), a.b)}
        else:
            res["hypotheses"][h] = {"set": tag, "auroc": None, "ci95": None, "p": 1.0, "note": "predictions missing"}
        res["hypotheses"][h]["contrast"] = f"{'+'.join(sorted(ca))} vs {'+'.join(sorted(cb))}"
    for h, q in zip(res["hypotheses"], holm([v["p"] for v in res["hypotheses"].values()])):
        res["hypotheses"][h].update(p_holm=float(q), reject=bool(q <= ALPHA))
    for r, tag, ca, cb in CROSS:
        if tag in tables:
            name = f"{r}:{tag}:{'+'.join(sorted(ca))} vs {'+'.join(sorted(cb))}"
            res["crossover"][name] = e1_contrast(tables[tag], ca, cb, lambda x, r=r: x[r], a.b)
    print(f"{'E1 (d = logit P - logit Lp)':50s} {'nA':>5s} {'nB':>5s} {'AUROC':>7s} {'CI95':>17s} {'p':>7s} {'Holm':>7s}")
    for h, v in res["hypotheses"].items():
        c = v["ci95"] or [None, None]
        print(f"{h + ' ' + v['set'] + ' ' + v['contrast']:50s} {v.get('n_a', 0):5d} {v.get('n_b', 0):5d} {fmt(v['auroc'])} "
              f"[{fmt(c[0])}, {fmt(c[1])}] {v['p']:.4f} {v['p_holm']:.4f}{' *' if v['reject'] else ''}")
    for k, v in res["crossover"].items():
        print(f"{k:50s} {v['n_a']:5d} {v['n_b']:5d} {fmt(v['auroc'])} [{fmt(v['ci95'][0])}, {fmt(v['ci95'][1])}]")
    dump(a.out_dir, "e1", res)
    return res


# ---------------------------------------------------------------- E3 (§11a)
E3_FAMILY = [("H_E3a", "PL", "Z", 1), ("H_E3b", "FL", "F0", 2)]
E3_NOTES = ["rows: cred_s10.load_rows over train/val/test restricted to the 159 PMIDs of data/CRED/cv/fold*.txt; "
            "y = label == '1'; unreachable rows score 0 (strict) whatever an arm file says",
            "an arm file is either per-row scores {pmid, gene_id, disease_id, score} or raw biocausal.score rows (per pair "
            "p_pc(1) if probs_pc is present, else p(Pos)+p(Neg)); several files average per key (seed ensemble), then a "
            "row takes the max over its candidate pairs",
            "dropped = a reachable row without a score, or a non-uniform number of files per key (e.g. 14 of 15 OOF files)",
            "C = same_sent: the CRED authors' flag, zeroed on unreachable rows (strict); same_sent_raw keeps the flag there",
            "document bootstrap: one resample = a multiset of the 159 PMIDs; all arms share the resamples (paired)"]


def e3_rows():
    fold = {p: k for k in range(5) for p in (CRED / "cv" / f"fold{k}.txt").read_text(encoding="utf-8").split()}
    return [r for s in ("train", "val", "test") for r in load_rows(s) if r["pmid"] in fold], fold


def arm_scores(spec, rows):
    if spec in ("same_sent", "same_sent_raw"):
        return (np.array([float(r["same_sent"] == "1" and (spec == "same_sent_raw" or r["status"] == "reachable"))
                          for r in rows]), {"source": spec, "dropped": False})
    files = sorted({f for part in spec.split(",") for f in glob.glob(part)})
    if not files:
        raise SystemExit(f"no files match {spec}")
    acc, kinds = defaultdict(list), set()
    for f in files:
        for r in read_jsonl(f):
            if "score" in r:
                kinds.add("score")
                acc[r["pmid"], frozenset(r["gene_id"].split(";")), r["disease_id"]].append(float(r["score"]))
            else:
                kinds.add("p_pc(1)" if "probs_pc" in r else "p(Pos)+p(Neg)")
                acc[r["pmid"], r["e1"], r["e2"]].append(float(r["probs_pc"][1]) if "probs_pc" in r else lp(r))
    if len(kinds) != 1:
        raise SystemExit(f"{spec}: mixed row formats {kinds}")
    kind = kinds.pop()
    s, n = [], Counter()
    for r in rows:
        keys = ([(r["pmid"], frozenset(r["gene_id"].split(";")), r["disease_id"])] if kind == "score"
                else [(r["pmid"], *p.split("|")) for p in r["pairs"].split(";") if p])
        got = [acc[k] for k in keys if k in acc]
        if r["status"] != "reachable":
            n["unreachable_nonzero_in_file"] += any(max(v) > 0 for v in got) if kind == "score" else 0
            s.append(0.0)
            continue
        n["reachable_missing"] += len(got) < len(keys) or not keys
        n.update(f"files_per_key={len(v)}" for v in got)
        s.append(max((float(np.mean(v)) for v in got), default=0.0))
    per_key = sorted(k for k in n if k.startswith("files_per_key"))
    meta = {"source": spec, "kind": kind, "n_files": len(files), **n,
            "dropped": bool(n["reachable_missing"] or len(per_key) > 1), "sha256": {f: sha(f) for f in files}}
    return np.array(s), meta


def cmd_e3(a):
    rows, fold = e3_rows()
    y = np.array([r["label"] == "1" for r in rows])
    arms, meta = {}, {}
    for spec in a.arm:
        name, src = spec.split("=", 1)
        arms[name], meta[name] = arm_scores(src, rows)
    names = list(arms)
    gs = [np.unique(-arms[n], return_inverse=True)[1] for n in names]
    est = {n: _ap(g, y, np.ones(len(y))) for n, g in zip(names, gs)}
    bs = boot([lambda w, g=g: _ap(g, y, w) for g in gs], [r["pmid"] for r in rows], a.b)
    res = {"n_rows": len(rows), "n_causal": int(y.sum()), "n_docs": len({r["pmid"] for r in rows}),
           "n_rows_per_fold": dict(sorted(Counter(fold[r["pmid"]] for r in rows).items())),
           "arms": {n: {"auprc": est[n], "ci95": ci(bs[:, i]), **meta[n]} for i, n in enumerate(names)},
           "family": family(E3_FAMILY, names, est, bs, [n for n in names if meta[n]["dropped"]]),
           "differences": diffs(names, est, bs), "notes": E3_NOTES}
    print(f"E3 pool: {res['n_rows']} rows, {res['n_causal']} causal, {res['n_docs']} docs")
    for n, v in res["arms"].items():
        print(f"  {n:6s} AUPRC {fmt(v['auprc'])} [{fmt(v['ci95'][0])}, {fmt(v['ci95'][1])}]  "
              f"{v.get('kind', v['source'])} files={v.get('n_files', '-')}{'  DROPPED' if v['dropped'] else ''}")
    for h, v in res["family"].items():
        c = v["ci95"] or [None, None]
        print(f"  {h} {v['test']:14s} diff {fmt(v['diff'])} [{fmt(c[0])}, {fmt(c[1])}] p {v['p']:.4f} "
              f"Holm {v['p_holm']:.4f}{' *' if v['reject'] else ''}")
    dump(a.out_dir, "e3", res)
    return res


# ---------------------------------------------------------------- E5 (§11c)
E5_SYS = {"V0": "v0", "VL": "vl", "VP": "vp"}
E5_FAMILY = [("H_E5a", "V0", "TFIDF", 1), ("H_E5b", "VL", "V0", 2)]
RGD3 = [EV_CLASSES.index(c) for c in ("genetic", "expression", "model_rescue")]
E5_NOTES = ["TEST_2026: macro one-vs-rest AUROC over the 5 classes, a multi-label unit is positive for each of its classes",
            "RGD_EXT: the 3 mapped classes, each system's probabilities renormalised over them (also for argmax F1)",
            "macro-F1: argmax on single-class units, F1 averaged over the set's classes",
            "TFIDF: TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True) on title + ' ' + abstract, "
            "LogisticRegression(lbfgs, multinomial, max_iter=1000), C by GroupKFold(5) on PMID; C picked on the "
            "pooled out-of-fold macro AUROC (the mean over folds is reported too)",
            "MAJORITY: the class shares of the TRAIN units for every unit",
            "PMID-cluster bootstrap per set; the systems share the resamples"]


def e5_units(units_path, docs_path):
    docs, out, n = load_docs(docs_path), [], Counter()
    for u in read_tsv(units_path):
        doc = docs[u["doc_id"]]
        c = candidates(doc, split(u["gene_entrez"]), split(u["disease_mesh"]))
        if len(c) != 1:
            n["not_single_pair"] += 1
            continue
        n["pair_differs_from_extra"] += list(c[0]) != json.loads(u["extra"]).get("pair")
        out.append({"pmid": u["pmid"], "key": (doc["pmid"], *c[0]), "cls": split(u["classes"]),
                    "text": f"{doc['title']} {doc['abstract']}"})
    return out, n


def tfidf(train, tests, cs=(0.1, 1, 10)):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline

    def model(c):
        return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                             LogisticRegression(C=c, max_iter=1000))
    x = np.array([u["text"] for u in train], dtype=object)
    y = np.array([EV_CLASSES.index(next(iter(u["cls"]))) for u in train])
    grid = {}
    for c in cs:
        oof, per = np.zeros((len(y), len(EV_CLASSES))), []
        for tr, va in GroupKFold(5).split(x, y, [u["pmid"] for u in train]):
            oof[va] = model(c).fit(x[tr], y[tr]).predict_proba(x[va])
            per.append(roc_auc_score(y[va], oof[va], multi_class="ovr"))
        grid[c] = {"pooled_oof_macro_auroc": float(roc_auc_score(y, oof, multi_class="ovr")),
                   "mean_fold_macro_auroc": float(np.mean(per))}
    best = max(cs, key=lambda c: grid[c]["pooled_oof_macro_auroc"])
    m = model(best).fit(x, y)
    return [m.predict_proba([u["text"] for u in t]) for t in tests], {"C": best, "grid": {str(k): v for k, v in grid.items()}}


def e5_set(units, probs, cls, nb):
    k = len(cls)
    y = np.array([[EV_CLASSES[c] in u["cls"] for c in cls] for u in units])
    single = y.sum(1) == 1
    names, res, fns = list(probs), {}, []
    for n in names:
        p = probs[n][:, cls]
        if k < len(EV_CLASSES):                  # RGD_EXT: renormalise over the mapped classes
            p = p / p.sum(1, keepdims=True)
        gs = [np.unique(p[:, j], return_inverse=True)[1] for j in range(k)]
        f = (lambda w, gs=gs: float(np.nanmean([_auc(g, y[:, j], w) for j, g in enumerate(gs)])))
        one = np.ones(len(units))
        res[n] = {"macro_auroc": f(one),
                  "per_class_auroc": {EV_CLASSES[c]: _auc(gs[j], y[:, j], one) for j, c in enumerate(cls)},
                  "macro_f1_single": macro_f1(y[single].argmax(1), p[single].argmax(1), k)}
        fns.append(f)
    bs = boot(fns, [u["pmid"] for u in units], nb)
    for i, n in enumerate(names):
        res[n]["ci95"] = ci(bs[:, i])
    est = {n: res[n]["macro_auroc"] for n in names}
    return {"n_units": len(units), "n_single": int(single.sum()), "n_pmids": len({u["pmid"] for u in units}),
            "systems": res, "differences": diffs(names, est, bs)}, names, est, bs


def cmd_e5(a):
    # §11c.6: the sealed sets are scored once, after all runs; TFIDF needs no run, so never score it on its own
    missing = [f"_p2e5_{s}/pred_{t}.jsonl" for s in E5_SYS.values() for t in ("test_2026", "rgd_ext")
               if not Path(a.pred_root, f"_p2e5_{s}", f"pred_{t}.jsonl").exists()]
    if missing:
        raise SystemExit(f"refusing to score TEST_2026 / RGD_EXT before all six V predictions exist: {missing}")
    e5 = P2 / "e5"
    train, n_tr = e5_units(e5 / "train_units.tsv", e5 / "clingen.json")
    sets = {t: e5_units(e5 / f"{t}_units.tsv", e5 / f"{t}_docs.json") for t in ("test_2026", "rgd_ext")}
    probs = {t: {} for t in sets}
    inputs = {}
    for name, arm in E5_SYS.items():
        for t, (units, _) in sets.items():
            f = Path(a.pred_root, f"_p2e5_{arm}", f"pred_{t}.jsonl")
            d ={(r["pmid"], r["e1"], r["e2"]): r["probs_ev"] for r in read_jsonl(f)}
            miss = sum(u["key"] not in d for u in units)
            if miss:
                raise SystemExit(f"{f}: {miss} units without a prediction")
            probs[t][name], inputs[str(f)] = np.array([d[u["key"]] for u in units], float), sha(f)
    tf, tf_meta = tfidf(train, [sets[t][0] for t in sets])
    prior = np.bincount([EV_CLASSES.index(next(iter(u["cls"]))) for u in train], minlength=len(EV_CLASSES)) / len(train)
    for (t, (units, _)), p in zip(sets.items(), tf):
        probs[t]["TFIDF"], probs[t]["MAJORITY"] = p, np.tile(prior, (len(units), 1))
    res = {"train": {"n_units": len(train), **n_tr, "tfidf": tf_meta, "majority_prior": prior.tolist()},
           "inputs": inputs, "notes": E5_NOTES}
    for t, cls in (("test_2026", list(range(len(EV_CLASSES)))), ("rgd_ext", RGD3)):
        res[t], names, est, bs = e5_set(sets[t][0], probs[t], cls, a.b)
        res[t].update(sets[t][1])
        if t == "test_2026":
            res["family"] = family(E5_FAMILY, names, est, bs)
    print(f"TFIDF C = {tf_meta['C']}  grid {tf_meta['grid']}")
    for t in ("test_2026", "rgd_ext"):
        print(f"{t}: {res[t]['n_units']} units ({res[t]['n_single']} single-class), {res[t]['n_pmids']} PMIDs")
        for n, v in res[t]["systems"].items():
            print(f"  {n:9s} macro AUROC {fmt(v['macro_auroc'])} [{fmt(v['ci95'][0])}, {fmt(v['ci95'][1])}]  "
                  f"macro-F1 {fmt(v['macro_f1_single'])}  " + " ".join(f"{fmt(x)}" for x in v["per_class_auroc"].values()))
    for h, v in res["family"].items():
        c = v["ci95"] or [None, None]
        print(f"  {h} {v['test']:16s} diff {fmt(v['diff'])} [{fmt(c[0])}, {fmt(c[1])}] p {v['p']:.4f} "
              f"Holm {v['p_holm']:.4f}{' *' if v['reject'] else ''}")
    dump(a.out_dir, "e5", res)
    return res


def main(argv=None):
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap_.add_subparsers(dest="cmd", required=True)
    for name in ("e1", "e3", "e5"):
        p = sub.add_parser(name)
        p.add_argument("--b", type=int, default=B, help="bootstrap resamples")
        p.add_argument("--out-dir", default=str(ROOT / "docs" / "results" / "paper2" / "verify"))
        if name in ("e1", "e5"):
            p.add_argument("--pred-root", required=True)
    sub.choices["e1"].add_argument("--swap-map", default=str(P2 / "swap" / "mapping.tsv"))
    sub.choices["e3"].add_argument("--arm", action="append", required=True, metavar="NAME=FILE|GLOB[,..]|same_sent")
    a = ap_.parse_args(argv)
    return {"e1": cmd_e1, "e3": cmd_e3, "e5": cmd_e5}[a.cmd](a)


if __name__ == "__main__":
    main()
