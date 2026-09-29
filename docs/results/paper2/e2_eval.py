r"""Paper 2, E2: paired name-swap deltas per reader (docs/research/paper2/codemap.md §3 E2; swaps: swap_build.py).

    PYTHONPATH=. .venv/bin/python docs/results/paper2/e2_eval.py [--pred-root runs] [--out docs/results/paper2/e2/e2.json]
      reads  data/paper2/swap/mapping.tsv, data/CRED/{cred_pairs.tsv,cv/fold0-4.txt}, data/paper2/clingen_{units.tsv,
             docs.json}, data/external/research_2026-09-17/{gencc.tsv,hgnc.txt,mondo.sssom.tsv} and the biocausal.score
             predictions PRED_ROOT/_p2e1_<reader>/pred_<tag>.jsonl (layout of ops/make_queue_p2_score.py): originals
             cred_docs / clingen_docs, swaps cred_{u,k}_f0-4 / clingen_{u,k}; a missing file skips that
             reader x source x condition
      writes OUT (JSON), prints a markdown table

Paired row = one swapped gene of one target. CRED: a labelled reachable single-gene pool pair (pmid, disease, gene),
s = p(Pos) + p(Neg) of that pair. ClinGen: a reachable unit with a swap_build.PRIMARY class whose gene ids hold the
swapped gene, s = max over units.candidate_pairs(doc, [gene], disease_mesh); AUROC works on units (max over their
rows). s0 on the original doc, s1 on its swapped copy '<pmid>_s<i>' (same pairs). Per reader x source x condition
(u: unknown gene, k: known gene):
  delta = logit(s1) - logit(s0), clip [1e-6, 1 - 1e-6]: mean with PMID-cluster bootstrap CI (e1_eval.boot: 10 000
  resamples, seed 0) and Spearman rho(s0, s1); both again per KB stratum: the (Entrez gene, MeSH disease) pair has a
  GenCC Definitive / Strong / Moderate / Supportive record (HGNC -> Entrez, MONDO -> MeSH exactMatch as units.py);
  CRED: AUPRC(s1) - AUPRC(s0) with the bootstrap CI (cred_eval.weighted_ap);
  ClinGen: AUROC of every CONTRASTS pair (units in exactly one side; e1_eval.contrast) on s0 and on s1, and the
  difference with the bootstrap CI.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / "docs/results/cred")]

from cred_eval import weighted_ap  # noqa: E402
from cred_s10 import load_rows  # noqa: E402
from e1_eval import EPS, auroc, boot, ci, contrast, reader_scores  # noqa: E402
from swap_build import PRIMARY  # noqa: E402
from units import candidate_pairs, hgnc, mondo_mesh  # noqa: E402

from biocausal.data import read_pmids  # noqa: E402

P2, CV = ROOT / "data/paper2", ROOT / "data/CRED/cv"
GENCC = ROOT / "data/external/research_2026-09-17/gencc.tsv"
READERS = ("L", "P", "Lp", "L0")
KB_LEVELS = {"Definitive", "Strong", "Moderate", "Supportive"}
# §11b: the E1 primary ClinGen contrast (H1, H4); A and B split for description
CONTRASTS = {"GEN_CLV_vs_EXP": ({"GEN_CLV"}, {"EXP_A", "EXP_B", "EXP_U"}),
             "GEN_CLV_vs_EXP_A": ({"GEN_CLV"}, {"EXP_A"}), "GEN_CLV_vs_EXP_B": ({"GEN_CLV"}, {"EXP_B"})}


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x) - np.log1p(-x)


def kb_pairs() -> set[tuple[str, str]]:
    """(Entrez, MeSH) pairs with a GenCC record of a KB_LEVELS classification."""
    hg, mm = hgnc(), mondo_mesh()
    return {(hg[r["gene_curie"]][1], m) for r in csv.DictReader(open(GENCC, encoding="utf-8"), delimiter="\t")
            if r["classification_title"] in KB_LEVELS and r["gene_curie"] in hg for m in mm.get(r["disease_curie"], ())}


def cred_rows(mapping: list[dict], s0: dict, s1: dict, pairs: list[dict], kb: set) -> list[dict]:
    """Paired rows of the CRED swaps; ``pairs``: load_rows rows (reachable single-gene pool pairs)."""
    by = defaultdict(list)
    for r in pairs:
        dis, g = r["pairs"].split("|")
        by[(r["pmid"], g)].append((dis, r["label"] == "1"))
    return [{"pmid": m["orig_doc_id"], "unit": f"{m['orig_doc_id']}|{dis}|{m['gene']}", "y": y,
             "s0": s0[(m["orig_doc_id"], dis, m["gene"])], "s1": s1[(m["doc_id"], dis, m["gene"])],
             "kb": (m["gene"], dis) in kb}
            for m in mapping for dis, y in by[(m["orig_doc_id"], m["gene"])]]


def unit_rows(mapping: list[dict], s0: dict, s1: dict, units: list[dict], docs: dict, kb: set) -> list[dict]:
    """Paired rows of the unit swaps; ``units``: TSV rows (';'-joined fields) with a PRIMARY class."""
    by = defaultdict(list)
    for u in units:
        for g in u["gene_entrez"].split(";"):
            by[(u["doc_id"], g)].append(u)
    out = []
    for m in mapping:
        for u in by[(m["orig_doc_id"], m["gene"])]:
            dis = [x for x in u["disease_mesh"].split(";") if x]
            pairs, _ = candidate_pairs(docs[m["orig_doc_id"]], [m["gene"]], dis)
            if pairs and all((m["doc_id"], *q) in s1 for q in pairs):   # only the scored (E1-set) swaps
                out.append({"pmid": u["pmid"], "unit": u["unit_id"], "classes": set(u["classes"].split(";")),
                            "s0": max(s0[(m["orig_doc_id"], *p)] for p in pairs),
                            "s1": max(s1[(m["doc_id"], *p)] for p in pairs),
                            "kb": any((g, d) in kb for g in u["gene_entrez"].split(";") for d in dis)})
    return out


def delta_stats(s0: np.ndarray, s1: np.ndarray, pmid: np.ndarray) -> dict:
    if len(s0) < 3:
        return {"n": len(s0)}
    d = logit(s1) - logit(s0)
    return {"n": len(d), "n_pmids": len(set(pmid)), "mean_delta_logit": float(d.mean()),
            "ci": ci(boot(pmid, lambda w: (w @ d) / w.sum(1))), "spearman": float(spearmanr(s0, s1).statistic)}


def summarise(rows: list[dict]) -> dict:
    s0, s1 = np.array([r["s0"] for r in rows]), np.array([r["s1"] for r in rows])
    pmid, kb = np.array([r["pmid"] for r in rows]), np.array([r["kb"] for r in rows])
    out = {**delta_stats(s0, s1, pmid), "kb": delta_stats(s0[kb], s1[kb], pmid[kb]),
           "not_kb": delta_stats(s0[~kb], s1[~kb], pmid[~kb])}
    if "y" in rows[0]:
        y = np.array([r["y"] for r in rows])
        one = np.ones((1, len(y)))
        out["auprc"] = {"orig": float(weighted_ap(s0, y, one)[0]), "swap": float(weighted_ap(s1, y, one)[0]),
                        "delta_ci": ci(boot(pmid, lambda w: weighted_ap(s1, y, w) - weighted_ap(s0, y, w)))}
        out["auprc"]["delta"] = out["auprc"]["swap"] - out["auprc"]["orig"]
        return out
    units: dict[str, dict] = {}
    for r in rows:   # unit level: max over the unit's swapped genes
        u = units.setdefault(r["unit"], {"pmid": r["pmid"], "classes": r["classes"], "s0": -1.0, "s1": -1.0})
        u["s0"], u["s1"] = max(u["s0"], r["s0"]), max(u["s1"], r["s1"])
    us = list(units.values())
    s0, s1, pmid = (np.array([u[k] for u in us]) for k in ("s0", "s1", "pmid"))
    out["contrasts"] = {}
    for name, (A, B) in CONTRASTS.items():
        a, b = np.array([bool(u["classes"] & A) for u in us]), np.array([bool(u["classes"] & B) for u in us])
        m = a ^ b
        res = {"n_a": int((m & a).sum()), "n_b": int((m & b).sum())}
        if res["n_a"] and res["n_b"]:
            d = boot(pmid[m], lambda w: auroc(s1[m], a[m], w) - auroc(s0[m], a[m], w))
            res.update(orig=contrast(s0[m], a[m], pmid[m]), swap=contrast(s1[m], a[m], pmid[m]),
                       delta_ci=ci(d[~np.isnan(d)]))
            res["delta"] = res["swap"]["auroc"] - res["orig"]["auroc"]
        out["contrasts"][name] = res
    return out


def evaluate(pred_root: Path) -> dict:
    mapping = defaultdict(list)
    for m in csv.DictReader(open(P2 / "swap/mapping.tsv", encoding="utf-8"), delimiter="\t"):
        mapping[m["file"].rsplit("_f", 1)[0]].append(m)          # cred_u_f3 -> cred_u, clingen_k -> clingen_k
    pool = set().union(*(read_pmids(CV / f"fold{k}.txt") for k in range(5)))
    pairs = [r for s in ("train", "val", "test") for r in load_rows(s)
             if r["pmid"] in pool and r["status"] == "reachable" and ";" not in r["pairs"]]
    units, docs = [], {}
    if (P2 / "e1/clingen_units.tsv").exists():   # §11b: the single-pair E1 set whose swaps were scored (e1_sets.py)
        units = [u for u in csv.DictReader(open(P2 / "e1/clingen_units.tsv", encoding="utf-8"), delimiter="\t")
                 if u["reachable"] == "1" and set(u["classes"].split(";")) & set(PRIMARY)]
        docs = {d["pmid"]: d for d in json.loads((P2 / "e1/clingen_docs.json").read_text(encoding="utf-8"))}
    kb, out = kb_pairs(), {}
    for reader in READERS:
        pred = lambda tag: pred_root / f"_p2e1_{reader}" / f"pred_{tag}.jsonl"  # noqa: E731
        for src, folds in (("cred", [f"_f{k}" for k in range(5)]), ("clingen", [""])):
            if not pred(f"{src}_docs").exists():
                continue
            s0 = reader_scores(pred(f"{src}_docs"))
            for c in ("u", "k"):
                files = [pred(f"{src}_{c}{f}") if src == "cred" else pred(f"swap_{src}_{c}") for f in folds]
                if not mapping[f"{src}_{c}"] or not all(f.exists() for f in files):
                    continue
                s1 = {k: v for f in files for k, v in reader_scores(f).items()}
                rows = (cred_rows(mapping[f"{src}_{c}"], s0, s1, pairs, kb) if src == "cred"
                        else unit_rows(mapping[f"{src}_{c}"], s0, s1, units, docs, kb))
                if rows:
                    out.setdefault(reader, {}).setdefault(src, {})[c] = summarise(rows)
    for arm, run in (("FL", "p2e3_f1"), ("F0", "p2e3_f0"), ("J", "p2e3_j")):
        s0 = fold_scores([pred_root / f"{run}_k{k}_s{s}" / "pred_cv.jsonl" for k in range(5) for s in (42, 43, 44)])
        for c in ("u", "k"):
            files = [pred_root / f"_p2e3z_{arm}_k{k}" / f"pred_cred_{c}.jsonl" for k in range(5)]
            if s0 and all(f.exists() for f in files):
                s1 = fold_scores(files)
                out.setdefault(arm, {}).setdefault("cred", {})[c] = summarise(cred_rows(mapping[f"cred_{c}"], s0, s1, pairs, kb))
    return out


def fold_scores(files: list[Path]) -> dict[tuple, float]:
    """Mean over files of p_pc(1) (pc-head runs) or p(Pos) + p(Neg), per (pmid, e1, e2); {} if a file is missing."""
    acc: dict[tuple, list] = defaultdict(list)
    for f in files:
        if not f.exists():
            return {}
        for p in map(json.loads, open(f, encoding="utf-8")):
            acc[(p["pmid"], p["e1"], p["e2"])].append(p["probs_pc"][1] if p.get("probs_pc") else
                                                      p["probs"]["Positive_Correlation"] + p["probs"]["Negative_Correlation"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def table(res: dict) -> str:
    f = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
    lines = ["| Reader | Source | Swap | Rows | Δlogit [95 % CI] | ρ | KB: n, Δlogit | not KB: n, Δlogit | Δ AUPRC / Δ AUROC [CI] |",
             "|---|---|---|---|---|---|---|---|---|"]
    for reader, by in res.items():
        for src, conds in by.items():
            for c, s in conds.items():
                kb, nk = s["kb"], s["not_kb"]
                if src == "cred":
                    extra = f"{f(s['auprc']['delta'])} [{f(s['auprc']['delta_ci'][0])}; {f(s['auprc']['delta_ci'][1])}]"
                else:
                    extra = "; ".join(f"{n}: {f(v.get('delta'))}" for n, v in s["contrasts"].items())
                lines.append(f"| {reader} | {src} | {c} | {s['n']} | {f(s.get('mean_delta_logit'))} "
                             f"[{f(s.get('ci', [None])[0])}; {f(s.get('ci', [None, None])[1])}] | {f(s.get('spearman'))} "
                             f"| {kb['n']}, {f(kb.get('mean_delta_logit'))} | {nk['n']}, {f(nk.get('mean_delta_logit'))} "
                             f"| {extra} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-root", type=Path, default=ROOT / "runs", help="holds _p2e1_<reader>/ as on the PC")
    ap.add_argument("--out", type=Path, default=HERE / "e2" / "e2.json")
    a = ap.parse_args(argv)
    res = evaluate(a.pred_root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(table(res))


if __name__ == "__main__":
    main(sys.argv[1:])
