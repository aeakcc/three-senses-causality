"""Paper 2, E3 evaluation (preregistration §11a items 3-5): pooled out-of-fold AUPRC and F1 of every arm on the
159-abstract CRED pool, per-fold numbers, same/cross-sentence AUPRC, paired differences (the descriptive grid) and the
confirmatory family H_E3a / H_E3b with Holm (m = 2).

    PYTHONPATH=. .venv/bin/python docs/results/paper2/e3_eval.py [--runs DIR] [--oof-dir DIR] [--out-dir DIR]
      reads  data/CRED/cred_pairs.tsv, data/CRED/cv/fold{0..4}.txt and the arms' OOF sources:
             Z   docs/results/cred/ens_bc8all/pred_cred.jsonl (zero-shot bc8all s42-44), p(Pos) + p(Neg)
             C   the CRED authors' same-sentence flag (reachable rows)
             S, PL, P0   OOF-DIR/oof_<ARM>.jsonl (e3_probe.py; default docs/results/paper2/e3)
             FL, F0, J   RUNS/p2e3_{f1,f0,j}_k{0..4}_s{42,43,44}/pred_cv.jsonl (ops/make_queue_p2.py, copied from the
                         PC; default runs/); per candidate pair the mean over the seeds of p_pc(1) (FL, F0) or
                         p(Pos) + p(Neg) (J)
             Q   Qwen3-8B, condition T, the prompt of docs/results/cred/s10_a3.json: cred/a3/fit_scores.jsonl
                 (train + val rows) and cred/testday/a3_test_scores.jsonl (test rows)
      writes OUT-DIR/e3_results.json, e3_results.md (default docs/results/paper2/e3)

Rows: cred_s10.load_rows of every split restricted to the pool, strict: unreachable -> score 0, a composite gene id ->
max over its candidate pairs (Q: a candidate pair without an LLM score -> 0, counted). Decision "score >= threshold":
FL, F0, J 0.5; PL, P0, S the e3_probe threshold of the row's fold; Z, Q max F1 on the strict pool rows of the other four
folds (ties -> highest); C the flag. An arm with an incomplete source is 'missing' (FL/F0/J: fewer than the 15 run
files), and a hypothesis with a missing arm gets p = 1. Bootstrap: documents with replacement, 10 000 resamples,
seed 0 (causal_audit.weights), the same draws for every arm; CI = 2.5 / 97.5 percentiles. H_E3a: AUPRC(PL) - AUPRC(Z)
> 0, p = share of resampled differences <= 0; H_E3b: AUPRC(FL) - AUPRC(F0) != 0, p = min(1, 2 min(share <= 0,
share >= 0)); Holm at alpha 0.05. Every other pair of arms: difference with CI and both p's, no test.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CRED = ROOT / "docs/results/cred"
sys.path[:0] = [str(ROOT), str(HERE), str(CRED), str(ROOT / "docs/results/posthoc"),
                str(ROOT / "docs/results/resolution")]

from biocausal.evaluate import read_jsonl  # noqa: E402
from causal_audit import N_BOOT, SEED, ap_boot, ci, prf_boot  # noqa: E402
from cred_eval import prf  # noqa: E402
from cred_s10 import CAUSAL, best_threshold, scores  # noqa: E402
from e3_probe import CV, K, OUT, pool  # noqa: E402
from resolution import holm  # noqa: E402

ARMS = ("PL", "P0", "FL", "F0", "J", "S", "Z", "C", "Q")
SEEDS = (42, 43, 44)
RUN_ARMS = {"FL": ("f1", lambda p: p["probs_pc"][1]), "F0": ("f0", lambda p: p["probs_pc"][1]),
            "J": ("j", lambda p: p["probs"][CAUSAL[0]] + p["probs"][CAUSAL[1]])}
FAMILY = {"H_E3a": ("PL", "Z", "greater"), "H_E3b": ("FL", "F0", "two-sided")}


def cand_keys(r: dict) -> list[tuple]:
    return [(r["pmid"], *c.split("|")) for c in r["pairs"].split(";") if c]


def strict(rows: list[dict], prob: dict) -> np.ndarray:
    return np.array([max((prob[k] for k in cand_keys(r)), default=0.0) for r in rows])


def other_folds_threshold(y, s, fold) -> np.ndarray:
    """Per row: max-F1 threshold (ties -> highest) on the rows of the other folds."""
    t = np.zeros(len(y))
    for k in range(K):
        t[fold == k] = best_threshold(y[fold != k], s[fold != k])[0]
    return t


def oof_arm(path: Path, rows: list[dict]):
    if not path.is_file():
        return f"no {path}"
    got = {(o["pmid"], o["gene_id"], o["disease_id"]): o for o in read_jsonl(path)}
    o = [got.get((r["pmid"], r["gene_id"], r["disease_id"])) for r in rows]
    if None in o:
        return f"{path} covers {len(o) - o.count(None)} of {len(rows)} rows"
    return np.array([x["score"] for x in o]), np.array([x["threshold"] for x in o])


def run_arm(runs: Path, tag: str, fn, rows: list[dict]):
    files = [runs / f"p2e3_{tag}_k{k}_s{s}" / "pred_cv.jsonl" for k in range(K) for s in SEEDS]
    have = sum(f.is_file() for f in files)
    if have < len(files):
        return f"{have} of {len(files)} run files under {runs}"
    per: dict[tuple, list] = {}
    for f in files:
        for p in read_jsonl(f):
            per.setdefault((p["pmid"], p["e1"], p["e2"]), []).append(fn(p))
    assert all(len(v) == len(SEEDS) for v in per.values()), "a pair scored in two folds, or a seed short"
    return strict(rows, {k: float(np.mean(v)) for k, v in per.items()}), np.full(len(rows), 0.5)


def q_arm(rows: list[dict], y, fold):
    a3 = json.loads((CRED / "s10_a3.json").read_text(encoding="utf-8"))
    files = [CRED / "a3/fit_scores.jsonl", CRED / "testday/a3_test_scores.jsonl"]
    if not all(f.is_file() for f in files):
        return f"no {files}"
    prob = {(s["pmid"], *s["key"].split("|")): s["score"] for f in files for s in read_jsonl(f)
            if s["prompt"] == a3["prompt"] and s["condition"] == "T"}
    lost = {k for r in rows for k in cand_keys(r)} - prob.keys()
    s = strict(rows, {**dict.fromkeys(lost, 0.0), **prob})
    return s, other_folds_threshold(y, s, fold), {"prompt": a3["prompt"], "n_candidate_pairs_without_score": len(lost)}


def load_arms(rows: list[dict], fold: np.ndarray, runs: Path, oof_dir: Path) -> dict:
    """{arm: (score, threshold per row[, note]) or the reason it is missing}."""
    y = np.array([r["label"] == "1" for r in rows])
    z = scores(rows, CRED / "ens_bc8all/pred_cred.jsonl")
    arms = {"Z": (z, other_folds_threshold(y, z, fold)),
            "C": (np.array([r["same_sent"] == "1" and r["status"] == "reachable" for r in rows], float),
                  np.ones(len(rows))),
            "Q": q_arm(rows, y, fold),
            **{a: oof_arm(oof_dir / f"oof_{a}.jsonl", rows) for a in ("S", "PL", "P0")},
            **{a: run_arm(runs, tag, fn, rows) for a, (tag, fn) in RUN_ARMS.items()}}
    return {a: arms[a] for a in ARMS}


def p_two(d: np.ndarray) -> float:
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


def evaluate(y, pmid, fold, same, arms: dict) -> dict:
    ok = {a: v for a, v in arms.items() if not isinstance(v, str)}
    aps, ap_b = ap_boot(y, {a: v[0] for a, v in ok.items()}, pmid)
    res: dict = {"n_rows": len(y), "n_causal": int(y.sum()), "n_docs": len(set(pmid)), "arms": {}}
    f1_b = {}
    for a, v in arms.items():
        if isinstance(v, str):
            res["arms"][a] = {"missing": v}
            continue
        s, t = v[0], v[1]
        pred = s >= t
        m, f1_b[a] = prf_boot(y, pred, pmid)
        per_fold = []
        for k in range(K):
            f = fold == k
            per_fold.append({"fold": k, "threshold": float(t[f][0]),
                             "auprc": float(average_precision_score(y[f], s[f])),
                             "f1": float(prf((pred & y)[f].sum(), pred[f].sum(), y[f].sum())[2])})
        res["arms"][a] = {**aps[a], **{k: m[k] for k in ("f1", "f1_ci", "p", "r", "n_pred")}, "per_fold": per_fold,
                          **(v[2] if len(v) > 2 else {})}
    res["sentence"] = {}
    for name, m in (("same_sentence", same), ("cross_sentence", ~same)):
        sub = ap_boot(y[m], {a: v[0][m] for a, v in ok.items()}, pmid[m])[0] if ok and y[m].any() else {}
        res["sentence"][name] = {"n_rows": int(m.sum()), "n_causal": int(y[m].sum()), "arms": sub}
    res["differences"] = {}
    for a, b in combinations(ok, 2):
        d, e = ap_b[a] - ap_b[b], f1_b[a] - f1_b[b]
        res["differences"][f"{a}-{b}"] = {
            "auprc": aps[a]["auprc"] - aps[b]["auprc"], "auprc_ci": ci(d), "auprc_p_le0": float(np.mean(d <= 0)),
            "auprc_p_two_sided": p_two(d), "f1": res["arms"][a]["f1"] - res["arms"][b]["f1"], "f1_ci": ci(e)}
    fam = {}
    for h, (a, b, alt) in FAMILY.items():
        d = ap_b[a] - ap_b[b] if a in ok and b in ok else None
        fam[h] = {"diff": f"AUPRC({a}) - AUPRC({b})", "alternative": alt, "missing": d is None,
                  "p": 1.0 if d is None else float(np.mean(d <= 0)) if alt == "greater" else p_two(d)}
    adj, rej = holm([f["p"] for f in fam.values()])
    for f, pa, r in zip(fam.values(), adj, rej):
        f.update(p_holm=pa, reject=r)
    res["family"] = fam
    return res


def markdown(res: dict) -> str:
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    L = [f"# E3: {res['n_rows']} pool rows, {res['n_causal']} causal, {res['n_docs']} abstracts (strict, OOF)", "",
         "| Arm | AUPRC [95 % CI] | F1 [95 % CI] | P | R | Thresholds by fold | AUPRC by fold |",
         "|---|---|---|---|---|---|---|"]
    for a, m in res["arms"].items():
        if "missing" in m:
            L.append(f"| {a} | missing: {m['missing']} | | | | | |")
            continue
        L.append(f"| {a} | {pct(m['auprc'])} [{pct(m['auprc_ci'][0])}; {pct(m['auprc_ci'][1])}] | {pct(m['f1'])} "
                 f"[{pct(m['f1_ci'][0])}; {pct(m['f1_ci'][1])}] | {pct(m['p'])} | {pct(m['r'])} | "
                 f"{' / '.join(f'{f['threshold']:.3g}' for f in m['per_fold'])} | "
                 f"{' / '.join(pct(f['auprc']) for f in m['per_fold'])} |")
    L += ["", "## Confirmatory family (Holm, m = 2)", "", "| H | Difference | Alternative | p | p Holm | Reject |",
          "|---|---|---|---|---|---|"]
    L += [f"| {h} | {f['diff']} | {f['alternative']} | {f['p']:.4f}{' (arm missing)' if f['missing'] else ''} | "
          f"{f['p_holm']:.4f} | {f['reject']} |" for h, f in res["family"].items()]
    L += ["", "## Paired differences (descriptive)", "",
          "| A - B | ΔAUPRC [95 % CI] | p (≤ 0) | p two-sided | ΔF1 [95 % CI] |",
          "|---|---|---|---|---|"]
    L += [f"| {k} | {pct(d['auprc'])} [{pct(d['auprc_ci'][0])}; {pct(d['auprc_ci'][1])}] | {d['auprc_p_le0']:.4f} | "
          f"{d['auprc_p_two_sided']:.4f} | {pct(d['f1'])} [{pct(d['f1_ci'][0])}; {pct(d['f1_ci'][1])}] |"
          for k, d in res["differences"].items()]
    L += ["", "## AUPRC within / across sentences", "", "| Subset | Rows (causal) | " + " | ".join(
        a for a in res["arms"] if "missing" not in res["arms"][a]) + " |", "|---|---|" + "---|" * sum(
        "missing" not in m for m in res["arms"].values())]
    for name, s in res["sentence"].items():
        L.append(f"| {name} | {s['n_rows']} ({s['n_causal']}) | " + " | ".join(
            f"{pct(v['auprc'])} [{pct(v['auprc_ci'][0])}; {pct(v['auprc_ci'][1])}]" for v in s["arms"].values()) + " |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs", help="dir holding the p2e3_* run dirs")
    ap.add_argument("--oof-dir", type=Path, default=OUT, help="dir of e3_probe.py's oof_<ARM>.jsonl")
    ap.add_argument("--cv", type=Path, default=CV)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    a = ap.parse_args(argv)
    rows, fold_of = pool(a.cv)
    rows = [r for r in rows if r["pmid"] in fold_of]
    fold = np.array([fold_of[r["pmid"]] for r in rows])
    arms = load_arms(rows, fold, a.runs, a.oof_dir)
    res = evaluate(np.array([r["label"] == "1" for r in rows]), np.array([r["pmid"] for r in rows]), fold,
                   np.array([r["same_sent"] == "1" for r in rows]), arms)
    res["meta"] = {"preregistration": "docs/results/preregistration.md §11a", "runs": str(a.runs),
                   "oof_dir": str(a.oof_dir), "seeds": SEEDS, "n_boot": N_BOOT, "seed": SEED}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "e3_results.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    (a.out_dir / "e3_results.md").write_text(markdown(res), encoding="utf-8")
    print(markdown(res))


if __name__ == "__main__":
    main(sys.argv[1:])
