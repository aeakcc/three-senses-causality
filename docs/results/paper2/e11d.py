"""Paper 2, preregistration §11d (exploratory, post hoc): two readouts of one frozen encoder, bc8all s42-44.

    PYTHONPATH=. .venv/bin/python docs/results/paper2/e11d.py geometry [--jobs 12]
      reads  runs/_p2e3_emb/bc8all_s{42,43,44}.npz, docs/results/cred/ens_bc8all/pred_cred.jsonl (reader L), the CRED
             rows (e3_probe.pool)
      writes docs/results/paper2/e11d/geometry.{json,md}
    PYTHONPATH=. .venv/bin/python docs/results/paper2/e11d.py heads
      reads  the same CRED npz (to train P*), runs/_p2x_emb/{clingen,rgd,percha}_s{42,43,44}.npz, reader L's
             runs/_p2e1_L/pred_{set}_docs.jsonl, the §11b units and docs
      writes runs/_p2e11d/pred_pstar_{set}.jsonl {pmid, e1, e2, probs_pstar [s42, s43, s44], p_pstar (their mean)},
             docs/results/paper2/e11d/heads.{json,md}

P* = the PL probe of e3_probe (StandardScaler + L2 LogisticRegression, C in 1e-4..1 by the pooled AUPRC of an inner
GroupKFold(4) by PMID, ties -> smaller C), fit per seed on every PL training row: the reachable single-candidate
labelled CRED rows of all 266 abstracts (no fold held out).

geometry (item 2): per seed, common coordinates Z = StandardScaler over all CRED D-G pairs of that seed. w_P = P*'s
coefficients mapped into Z (coef / probe scale * common scale). w_L = Ridge of logit(clip(s_L)), s_L = p(Pos) + p(Neg),
on Z over all D-G pairs; alpha in ALPHAS by pooled OOF R^2 of GroupKFold(5) by PMID (ties -> smaller alpha); surrogate
fidelity = Spearman of that OOF with s_L. Statistic cos(w_L, w_P) per seed and its seed mean. CI: N_GEO_BOOT resamples
of the PMIDs (default_rng(0).multinomial), refitting scaler, ridge and probe, but at the full-data alpha and C of the
seed (per-resample selection skipped for runtime). Null: the full-data w_L against probes on permuted training labels
(N_PERM, default_rng(1); one permutation shared by the seeds), at the full-data C. Also: AUROC of the w_L projection
on the gold labels of the reachable pool rows (composite gene -> max over its candidate pairs; e1_eval.contrast, PMID
bootstrap 10 000, default_rng(0)), s_L's AUROC on the same rows, Spearman of the w_P projection with s_L over all D-G
pairs, cosine between seeds of w_P and of w_L, and (not in §11d; added because weight cosines in 2304 correlated
dimensions can be near 0 while the projections agree) the Pearson correlation of the two projections over all
D-G pairs, i.e. the cosine of w_L and w_P in the metric of the data covariance.

heads (item 1): per set, seed-averaged P* probability of every D-G pair; the §11b units scored with e1_eval
(max over units.candidate_pairs); statistics L, Pstar and delta_star = logit(clip(s_P*)) - logit(clip(s_L)), clip
[1e-6, 1 - 1e-6]; e1_eval.evaluate for H1 / H2 / H3 (AUROC, ties 0.5, PMID-cluster bootstrap 10 000, default_rng(0))
and the per-class profiles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from e1_eval import EPS, contrast, evaluate, markdown, reader_scores, score_units  # noqa: E402
from e3_probe import C_GRID, fit_lr, pool  # noqa: E402
from units import read_units  # noqa: E402

SEEDS, ALPHAS, N_GEO_BOOT, N_PERM = (42, 43, 44), (1, 10, 100, 1000, 10000), 500, 200
EMB, XEMB, LDIR = ROOT / "runs/_p2e3_emb", ROOT / "runs/_p2x_emb", ROOT / "runs/_p2e1_L"
LPRED, OUT, PRED_OUT = ROOT / "docs/results/cred/ens_bc8all/pred_cred.jsonl", HERE / "e11d", ROOT / "runs/_p2e11d"
P2 = ROOT / "data/paper2"
SETS = {  # set: (units, docs, contrast, A, B), §11b
    "clingen": (P2 / "e1/clingen_units.tsv", P2 / "e1/clingen_docs.json", "H1", {"GEN_CLV"}, {"EXP_A", "EXP_B", "EXP_U"}),
    "rgd": (P2 / "e1/rgd_units.tsv", P2 / "e1/rgd_docs.json", "H2", {"IMP", "IGI"}, {"IEP"}),
    "percha": (P2 / "percha_units.tsv", P2 / "percha_docs.json", "H3", {"MUT", "CAUSE"}, {"UP"})}


def logit(s):
    s = np.clip(s, EPS, 1 - EPS)
    return np.log(s) - np.log1p(-s)


def cos(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(path: Path) -> tuple[np.ndarray, list[tuple]]:
    z = np.load(path)
    return z["X"].astype(np.float32), list(zip(z["pmid"].tolist(), z["e1"].tolist(), z["e2"].tolist()))


def cred() -> tuple[list[np.ndarray], list[tuple], list[dict], np.ndarray, np.ndarray]:
    """(X per seed, pair keys, CRED rows, npz row of every P* training row, its label); key order is the same per seed."""
    Xs, keys = zip(*(load(EMB / f"bc8all_s{s}.npz") for s in SEEDS))
    assert all(k == keys[0] for k in keys), "seed npz files list different pairs"
    at = {k: i for i, k in enumerate(keys[0])}
    rows, _ = pool()
    tr_rows = [r for r in rows if r["status"] == "reachable" and ";" not in r["pairs"]]
    tr = np.array([at[(r["pmid"], *r["pairs"].split("|"))] for r in tr_rows])
    return list(Xs), keys[0], rows, tr, np.array([r["label"] == "1" for r in tr_rows])


def probe_w(Xt, yt, C: float, scale) -> np.ndarray:
    """P* at C on (Xt, yt), its coefficients in the coordinates whose per-feature scale is ``scale``."""
    p = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(Xt, yt)
    return p[-1].coef_[0] / p[0].scale_ * scale


def readouts(Xa, ta, Xt, yt, C: float, alpha: float) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """(w_L, w_P, common scaler): Ridge(alpha) of ta on all pairs Xa and P* at C on (Xt, yt), both in Z of Xa."""
    sc = StandardScaler().fit(Xa)
    return Ridge(alpha=alpha).fit(sc.transform(Xa), ta).coef_, probe_w(Xt, yt, C, sc.scale_), sc


def ridge_alpha(Z, t, groups) -> tuple[float, dict, np.ndarray]:
    """(alpha, pooled OOF R^2 per alpha, OOF at alpha) over GroupKFold(5) by PMID."""
    cv = list(GroupKFold(n_splits=5).split(Z, t, groups))
    oof = {a: cross_val_predict(Ridge(alpha=a), Z, t, cv=cv) for a in ALPHAS}
    r2 = {a: float(r2_score(t, oof[a])) for a in ALPHAS}
    best = max(ALPHAS, key=lambda a: (r2[a], -a))
    return best, r2, oof[best]


def boot_one(counts, Xs, ts, doc, tr, y, Cs, alphas) -> list[float]:
    rep = counts[doc]
    ia, it = np.repeat(np.arange(len(doc)), rep), np.repeat(np.arange(len(tr)), rep[tr])
    return [cos(*readouts(X[ia], t[ia], X[tr[it]], y[it], C, a)[:2]) for X, t, C, a in zip(Xs, ts, Cs, alphas)]


def null_one(perm, Xts, y, Cs, scales, wLs) -> list[float]:
    return [cos(wL, probe_w(Xt, y[perm], C, s)) for Xt, C, s, wL in zip(Xts, Cs, scales, wLs)]


def geometry(jobs: int) -> dict:
    Xs, keys, rows, tr, y = cred()
    pmid = np.array([k[0] for k in keys])
    docs, doc = np.unique(pmid, return_inverse=True)
    sl = reader_scores(LPRED)
    s_L = np.array([sl[k] for k in keys])
    t = logit(s_L)
    at = {k: i for i, k in enumerate(keys)}
    fold = pool()[1]
    ev = [r for r in rows if r["pmid"] in fold and r["status"] == "reachable"]
    ev_cand = [[at[(r["pmid"], *c.split("|"))] for c in r["pairs"].split(";")] for r in ev]
    ev_y, ev_pmid = np.array([r["label"] == "1" for r in ev]), np.array([r["pmid"] for r in ev])
    row_max = lambda v: np.array([v[c].max() for c in ev_cand])  # noqa: E731
    seeds, wL, wP, scales = {}, [], [], []
    for s, X in zip(SEEDS, Xs):
        _, _, C, ap = fit_lr(X[tr], y, pmid[tr], C_GRID, 2000)
        Z = StandardScaler().fit_transform(X)
        alpha, r2, oof = ridge_alpha(Z, t, pmid)
        w_L, w_P, sc = readouts(X, t, X[tr], y, C, alpha)
        wL.append(w_L), wP.append(w_P), scales.append(sc.scale_)
        seeds[s] = {"C": C, "inner_auprc": {str(c): v for c, v in ap.items()}, "alpha": alpha,
                    "ridge_oof_r2": {str(a): v for a, v in r2.items()},
                    "surrogate_oof_spearman": float(spearmanr(oof, s_L)[0]), "cos": cos(w_L, w_P),
                    "spearman_wP_proj_sL": float(spearmanr(Z @ w_P, s_L)[0]),
                    "pearson_wL_proj_wP_proj": float(np.corrcoef(Z @ w_L, Z @ w_P)[0, 1]),
                    "auroc_wL_proj_pool": contrast(row_max(Z @ w_L), ev_y, ev_pmid)}
        print(s, {k: v for k, v in seeds[s].items() if k not in ("inner_auprc", "ridge_oof_r2")}, flush=True)
    Cs, alphas = [seeds[s]["C"] for s in SEEDS], [seeds[s]["alpha"] for s in SEEDS]
    obs = np.array([seeds[s]["cos"] for s in SEEDS])
    counts = np.random.default_rng(0).multinomial(len(docs), np.full(len(docs), 1 / len(docs)), size=N_GEO_BOOT)
    run = Parallel(n_jobs=jobs, verbose=5)
    b = np.array(run(delayed(boot_one)(c, Xs, [t] * 3, doc, tr, y, Cs, alphas) for c in counts))
    rng = np.random.default_rng(1)
    perms = [rng.permutation(len(y)) for _ in range(N_PERM)]
    Xts = [X[tr] for X in Xs]
    nl = np.array(run(delayed(null_one)(p, Xts, y, Cs, scales, wL) for p in perms))

    def summ(o, bv, nv):
        return {"cos": float(o), "ci": [float(np.percentile(bv, 2.5)), float(np.percentile(bv, 97.5))],
                "null_2.5_97.5": [float(np.percentile(nv, 2.5)), float(np.percentile(nv, 97.5))],
                "null_mean": float(nv.mean()), "p_abs_null_ge_abs_obs": float(np.mean(np.abs(nv) >= abs(o)))}
    return {"meta": {"npz": {str(EMB / f"bc8all_s{s}.npz"): sha(EMB / f"bc8all_s{s}.npz") for s in SEEDS},
                     "reader_L": str(LPRED), "reader_L_sha256": sha(LPRED), "n_pairs": len(keys),
                     "n_pmids": len(docs), "n_train": len(tr), "n_train_causal": int(y.sum()),
                     "n_pool_reachable": len(ev), "n_boot": N_GEO_BOOT, "boot_seed": 0, "n_perm": N_PERM,
                     "perm_seed": 1, "alphas": list(ALPHAS), "c_grid": list(C_GRID),
                     "shortcut": "bootstrap and null refit at the full-data alpha and C of each seed "
                                 "(no per-resample selection, for runtime)"},
            "s_L_auroc_pool": contrast(row_max(s_L), ev_y, ev_pmid), "seeds": seeds,
            "cos": {**{str(s): summ(obs[j], b[:, j], nl[:, j]) for j, s in enumerate(SEEDS)},
                    "mean": summ(obs.mean(), b.mean(1), nl.mean(1))},
            "between_seeds": {w: {f"{SEEDS[i]}-{SEEDS[j]}": cos(v[i], v[j]) for i, j in combinations(range(3), 2)}
                              for w, v in (("w_P", wP), ("w_L", wL))}}


def geometry_md(r: dict) -> str:
    f = lambda x: f"{x:.3f}"  # noqa: E731
    fc = lambda c: f"{f(c['auroc'])} [{f(c['ci'][0])}; {f(c['ci'][1])}]"  # noqa: E731
    m = r["meta"]
    L = ["# §11d item 2: geometry of the L and P* readouts (exploratory)", "",
         f"{m['n_pairs']} CRED D-G pairs, {m['n_pmids']} PMIDs; P* trained on {m['n_train']} rows "
         f"({m['n_train_causal']} causal). CI: {m['n_boot']} PMID resamples; null: {m['n_perm']} label permutations. "
         f"Shortcut: {m['shortcut']}.", "",
         "| Seed | cos(w_L, w_P) [95 % CI] | null 2.5 / 97.5 % | P(abs null >= abs obs) |", "|---|---|---|---|"]
    for k, c in r["cos"].items():
        L.append(f"| {k} | {f(c['cos'])} [{f(c['ci'][0])}; {f(c['ci'][1])}] | {f(c['null_2.5_97.5'][0])} / "
                 f"{f(c['null_2.5_97.5'][1])} | {c['p_abs_null_ge_abs_obs']:.3f} |")
    L += ["", "| Seed | C | alpha | ridge OOF R2 | surrogate OOF Spearman | Spearman(w_P proj, s_L) | Pearson(w_L proj, "
          "w_P proj) | AUROC w_L proj (pool, reachable) |", "|---|---|---|---|---|---|---|---|"]
    for s, v in r["seeds"].items():
        L.append(f"| {s} | {v['C']:g} | {v['alpha']:g} | {f(v['ridge_oof_r2'][str(v['alpha'])])} | "
                 f"{f(v['surrogate_oof_spearman'])} | {f(v['spearman_wP_proj_sL'])} | {f(v['pearson_wL_proj_wP_proj'])} | "
                 f"{fc(v['auroc_wL_proj_pool'])} |")
    L += ["", f"s_L itself on the same pool rows: AUROC {fc(r['s_L_auroc_pool'])}.", "",
          "Cosine between seeds: " + "; ".join(f"{w} " + ", ".join(f"{k} {f(x)}" for k, x in v.items())
                                               for w, v in r["between_seeds"].items()) + "."]
    return "\n".join(L) + "\n"


def heads() -> dict:
    Xs, keys, _, tr, y = cred()
    g = np.array([k[0] for k in keys])[tr]
    fits = [fit_lr(X[tr], y, g, C_GRID, 2000) for X in Xs]
    out = {"meta": {"p_star": {str(s): {"C": f[2], "inner_auprc": {str(c): v for c, v in f[3].items()}}
                               for s, f in zip(SEEDS, fits)}, "n_train": len(tr), "n_train_causal": int(y.sum())}}
    PRED_OUT.mkdir(parents=True, exist_ok=True)
    for name, (up, dp, h, A, B) in SETS.items():
        paths = [XEMB / f"{name}_s{s}.npz" for s in SEEDS]
        loaded = [load(p) for p in paths]
        keys = loaded[0][1]
        assert all(k == keys for _, k in loaded), f"{name}: seed npz files list different pairs"
        P = np.array([f[0].predict_proba(X)[:, 1] for f, (X, _) in zip(fits, loaded)])
        pred = PRED_OUT / f"pred_pstar_{name}.jsonl"
        with open(pred, "w", encoding="utf-8") as fh:
            for k, p in zip(keys, P.T):
                fh.write(json.dumps({"pmid": k[0], "e1": k[1], "e2": k[2], "probs_pstar": p.tolist(),
                                     "p_pstar": float(p.mean())}) + "\n")
        units = read_units(up)
        docs = {d["pmid"]: d for d in json.loads(dp.read_text(encoding="utf-8"))}
        lp = LDIR / f"pred_{name}_docs.jsonl"
        anchors, st = score_units(units, docs, {"L": lp, "Pstar": dict(zip(keys, P.mean(0)))})
        st["delta_star"] = logit(st["Pstar"]) - logit(st["L"])
        out[name] = {"meta": {"units": str(up), "units_sha256": sha(up), "docs": str(dp), "docs_sha256": sha(dp),
                              "reader_L": str(lp), "reader_L_sha256": sha(lp), "pred_pstar": str(pred),
                              "npz_sha256": {str(p): sha(p) for p in paths}, "n_pairs": len(keys)},
                     **evaluate(units, anchors, st, {h: (A, B)})}
        print(name, json.dumps({k: v for k, v in out[name]["contrasts"].items()})[:1500], flush=True)
    return out


def heads_md(r: dict) -> str:
    L = ["# §11d item 1: P* head on the frozen bc8all encoder (exploratory)", "",
         f"P* per seed (trained on {r['meta']['n_train']} CRED rows, {r['meta']['n_train_causal']} causal): "
         + ", ".join(f"s{s} C={v['C']:g} inner AUPRC {v['inner_auprc'][str(v['C'])]:.3f}"
                     for s, v in r["meta"]["p_star"].items())
         + ". delta_star = logit s_P* - logit s_L.", ""]
    for name in SETS:
        L.append(markdown(r[name]).replace("# E1:", f"## {name}:", 1).replace("## Profile:", "### Profile:"))
    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("geometry", "heads"))
    ap.add_argument("--jobs", type=int, default=-1, help="geometry: joblib workers for the bootstrap and the null")
    a = ap.parse_args(argv)
    res = geometry(a.jobs) if a.cmd == "geometry" else heads()
    md = geometry_md(res) if a.cmd == "geometry" else heads_md(res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.cmd}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    (OUT / f"{a.cmd}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main(sys.argv[1:])
