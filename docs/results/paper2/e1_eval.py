"""Paper 2, E1: do a reader's scores separate evidence classes of (PMID, gene, disease) units? AUROC of a statistic for
class set A against class set B with a PMID-cluster bootstrap, per-class score profiles, anchor breakdown.

    PYTHONPATH=. .venv/bin/python docs/results/paper2/e1_eval.py --units U.tsv --docs D.json \
        --reader L=ens_bc8all/pred_x.jsonl --reader P=ens_a2/pred_x_cred.jsonl@cred --reader Lp=ens_a2/pred_x.jsonl \
        --contrast "MR_vs_EXP:MS_ORG,MS_CELL,RES:EXP_A,EXP_B" [--contrast ...] --out docs/results/paper2/e1/x.json
      reads  a units TSV (unit_id, source, doc_id, pmid, gene_entrez, disease_mesh, classes, reachable,
             disease_matched, extra), the Doc JSON the readers scored, one biocausal.score pred jsonl per reader
             (NAME=PATH, optional @NOTE, e.g. the --corpus the file was scored under)
      writes OUT (JSON) and OUT with suffix .md (markdown tables)

Unit score of a reader = max over units.candidate_pairs(doc, gene_entrez, disease_mesh) of p(Pos) + p(Neg); units
with anchor 'none' (gene or disease not an entity) or 'no_doc' (doc_id not in the docs) are left out. With readers P
and Lp, the within-model contrast d = logit(clip(s_P)) - logit(clip(s_Lp)), clip to [1e-6, 1 - 1e-6], is one more
statistic. A contrast keeps the units
whose classes meet exactly one of A and B, on every scored unit ('all') and on anchor 'pair' only ('pair').
AUROC counts ties 0.5. Bootstrap: PMIDs drawn with replacement, every unit of a drawn PMID comes along, 10 000
resamples, numpy default_rng(0) (causal_audit.weights); CI = 2.5 / 97.5 percentiles over the resamples where both
classes have weight; p_greater = share of resampled AUROC <= 0.5, p_less = share >= 0.5 (one-sided each way).
Profiles: mean statistic per class (a unit counts in each of its classes), CIs on one bootstrap of all scored units.
Holm for a family of p-values: ``holm`` (resolution.holm). Deterministic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / "docs/results/posthoc"), str(ROOT / "docs/results/resolution")]

from causal_audit import N_BOOT, SEED, ci, weights  # noqa: E402
from resolution import holm  # noqa: E402,F401   Holm-Bonferroni step-down (adjusted p, rejected), for callers
from units import candidate_pairs, read_units  # noqa: E402

CAUSAL = ("Positive_Correlation", "Negative_Correlation")
EPS, CHUNK = 1e-6, 500


def reader_scores(path: Path) -> dict[tuple, float]:
    with open(path, encoding="utf-8") as f:
        return {(p["pmid"], p["e1"], p["e2"]): p["probs"][CAUSAL[0]] + p["probs"][CAUSAL[1]]
                for p in map(json.loads, f)}


def score_units(units: list[dict], docs: dict[str, dict], readers: dict[str, Path | dict]) -> tuple[np.ndarray, dict]:
    """(anchor per unit: pair / gene / none / no_doc, {statistic: value per unit, nan when not scored}); a reader is
    a pred jsonl or an already-built {(pmid, e1, e2): score}."""
    probs = {n: p if isinstance(p, dict) else reader_scores(p) for n, p in readers.items()}
    anchors, out = [], {n: [] for n in readers}
    for u in units:
        doc = docs.get(u["doc_id"])
        pairs, anchor = candidate_pairs(doc, u["gene_entrez"], u["disease_mesh"]) if doc else ([], "no_doc")
        anchors.append(anchor)
        for n in readers:
            out[n].append(max((probs[n][(u["doc_id"], *k)] for k in pairs), default=np.nan))
    s = {n: np.array(v, dtype=float) for n, v in out.items()}
    if {"P", "Lp"} <= s.keys():
        logit = lambda x: np.log(np.clip(x, EPS, 1 - EPS)) - np.log1p(-np.clip(x, EPS, 1 - EPS))  # noqa: E731
        s["d"] = logit(s["P"]) - logit(s["Lp"])
    return np.array(anchors), s


def auroc(s: np.ndarray, a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """AUROC of score s for the rows a (bool) against the others, ties 0.5, per row of unit weights w [B, n];
    nan where a class has no weight."""
    order = np.argsort(s, kind="mergesort")
    starts = np.r_[0, np.nonzero(np.diff(s[order]))[0] + 1]                # first index of each tied-score group
    wo, ao = w[:, order], a[order]
    wa, wb = np.add.reduceat(wo * ao, starts, axis=1), np.add.reduceat(wo * ~ao, starts, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (wa * (np.cumsum(wb, axis=1) - wb / 2)).sum(1) / (wa.sum(1) * wb.sum(1))


def boot(pmid: np.ndarray, fn) -> np.ndarray:
    """fn(unit weights [b, n]) over the N_BOOT PMID-cluster resamples, stacked along axis 0."""
    docs, idx = np.unique(pmid, return_inverse=True)
    w = weights(len(docs))
    return np.concatenate([fn(w[i:i + CHUNK][:, idx]) for i in range(0, len(w), CHUNK)])


def contrast(s: np.ndarray, a: np.ndarray, pmid: np.ndarray) -> dict:
    b = boot(pmid, lambda w: auroc(s, a, w))
    ok = b[~np.isnan(b)]
    return {"auroc": float(auroc(s, a, np.ones((1, len(s))))[0]), "ci": ci(ok), "p_greater": float(np.mean(ok <= 0.5)),
            "p_less": float(np.mean(ok >= 0.5)), "n_undefined": len(b) - len(ok)}


def profile(s: np.ndarray, masks: dict[str, np.ndarray], pmid: np.ndarray) -> dict:
    m = np.array(list(masks.values()), dtype=float).T                       # [n, classes]
    with np.errstate(invalid="ignore", divide="ignore"):
        b = boot(pmid, lambda w: (w @ (m * s[:, None])) / (w @ m))
    return {c: {"n": int(m[:, j].sum()), "mean": float(s[m[:, j] > 0].mean()), "ci": ci(b[~np.isnan(b[:, j]), j])}
            for j, c in enumerate(masks)}


def evaluate(units: list[dict], anchors: np.ndarray, stats: dict[str, np.ndarray],
             contrasts: dict[str, tuple[set, set]]) -> dict:
    classes = [set(u["classes"]) for u in units]
    pmid = np.array([u["pmid"] for u in units])
    scored = np.isin(anchors, ("pair", "gene"))
    an = anchors.tolist()
    out: dict = {"n_units": len(units), "n_scored": int(scored.sum()), "anchors": dict(Counter(an)),
                 "anchor_by_class": {c: dict(Counter(a for a, cl in zip(an, classes) if c in cl))
                                     for c in sorted(set().union(*classes))},
                 "anchor_by_source": {src: dict(Counter(a for a, u in zip(an, units) if u["source"] == src))
                                      for src in sorted({u["source"] for u in units})},
                 "profiles": {}, "contrasts": {}}
    idx = np.nonzero(scored)[0]
    masks = {c: np.array([c in classes[i] for i in idx]) for c in sorted(set().union(*(classes[i] for i in idx)))}
    out["profiles"] = {n: profile(s[scored], masks, pmid[scored]) for n, s in stats.items()} if masks else {}
    for name, (A, B) in contrasts.items():
        in_a, in_b = np.array([bool(c & A) for c in classes]), np.array([bool(c & B) for c in classes])
        out["contrasts"][name] = {"A": sorted(A), "B": sorted(B),
                                  "n_dropped_both": int((scored & in_a & in_b).sum())}
        for sub, keep in (("all", scored), ("pair", anchors == "pair")):
            m = keep & (in_a ^ in_b)
            res = {"n_a": int((m & in_a).sum()), "n_b": int((m & in_b).sum()), "n_pmids": len(set(pmid[m]))}
            if res["n_a"] and res["n_b"]:
                res["stats"] = {n: contrast(s[m], in_a[m], pmid[m]) for n, s in stats.items()}
            out["contrasts"][name][sub] = res
    return out


def markdown(res: dict) -> str:
    f = lambda x: f"{x:.3f}"  # noqa: E731
    L = [f"# E1: {res['n_scored']} of {res['n_units']} units scored", "",
         f"Anchors: {json.dumps(res['anchors'])}", "",
         "| Contrast | Units | A / B | PMIDs | Statistic | AUROC [95 % CI] | p (A > B) | p (A < B) |",
         "|---|---|---|---|---|---|---|---|"]
    for name, c in res["contrasts"].items():
        for sub in ("all", "pair"):
            r = c[sub]
            for n, s in r.get("stats", {"—": None}).items():
                auc = "—" if s is None else f"{f(s['auroc'])} [{f(s['ci'][0])}; {f(s['ci'][1])}]"
                pg, pl = ("—", "—") if s is None else (f"{s['p_greater']:.4f}", f"{s['p_less']:.4f}")
                L.append(f"| {name} | {sub} | {r['n_a']} / {r['n_b']} | {r['n_pmids']} | {n} | {auc} | {pg} | {pl} |")
    for n, prof in res["profiles"].items():
        L += ["", f"## Profile: {n}", "", "| Class | Units | Mean [95 % CI] |", "|---|---|---|"]
        L += [f"| {c} | {p['n']} | {f(p['mean'])} [{f(p['ci'][0])}; {f(p['ci'][1])}] |" for c, p in prof.items()]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", required=True, type=Path)
    ap.add_argument("--docs", required=True, type=Path)
    ap.add_argument("--reader", action="append", required=True, help="NAME=PRED.jsonl[@NOTE]")
    ap.add_argument("--contrast", action="append", default=[], help="NAME:A1,A2:B1,B2 (class labels)")
    ap.add_argument("--out", required=True, type=Path, help=".json; the markdown goes next to it (.md)")
    a = ap.parse_args(argv)
    readers, notes = {}, {}
    for r in a.reader:
        name, rest = r.split("=", 1)
        path, _, notes[name] = rest.partition("@")
        readers[name] = Path(path)
    contrasts = {}
    for c in a.contrast:
        name, A, B = c.split(":")
        contrasts[name] = (set(A.split(",")), set(B.split(",")))
    units = read_units(a.units)
    unknown = set().union(*(A | B for A, B in contrasts.values())) - set().union(*(u["classes"] for u in units))
    if unknown:   # a typo would silently shrink a class set
        ap.error(f"contrast classes not in {a.units}: {sorted(unknown)}")
    docs = {d["pmid"]: d for d in json.loads(a.docs.read_text(encoding="utf-8"))}
    anchors, stats = score_units(units, docs, readers)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    res = {"meta": {"units": str(a.units), "units_sha256": sha(a.units), "docs": str(a.docs),
                    "docs_sha256": sha(a.docs),
                    "readers": {n: {"pred": str(p), "note": notes[n], "sha256": sha(p)} for n, p in readers.items()},
                    "statistics": list(stats), "n_boot": N_BOOT, "seed": SEED,
                    "d": "logit(clip(s_P)) - logit(clip(s_Lp)), clip [1e-6, 1-1e-6]" if "d" in stats else None},
           **evaluate(units, anchors, stats, contrasts)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    a.out.with_suffix(".md").write_text(markdown(res), encoding="utf-8")
    print(markdown(res))


if __name__ == "__main__":
    main(sys.argv[1:])
