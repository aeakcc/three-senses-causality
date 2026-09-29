"""Independent recomputation of the CRED test day (preregistration §10 item 10, §6 in spirit).

    PYTHONPATH=. .venv/bin/python docs/results/cred/verify_cred_testday.py
      reads  docs/results/cred/cred_testday.json and every input it names
      writes docs/results/cred/verify_cred_testday.json  (comparison, leakage check, file hashes)

Written to share no code with cred_testday.py: its own row merge, its own counting, its own bootstrap (numpy only,
same documented draw: 10 000 multinomial resamples of the documents with a fresh default_rng(0)), its own Holm.
It recomputes all 11 systems of the primary table, the three Holm p-values, the leave-one-document-out range and the
drop-largest-abstract sensitivity.
Also repeats the leakage check: no CRED test PMID may appear in BioRED, BC8-400 or the CRED training file.
Exit code 1 if any recomputed number differs from the reported one by more than 1e-9 (metrics) or 2e-3 (p-values).
The two implementations compute F1 differently (2*tp/(n_pred+n_gold) here, 2pr/(p+r) there), so a handful of the
10 000 resamples land on the other side of a tie and the bootstrap p-values agree only to about 3e-4.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAUSAL = ("Positive_Correlation", "Negative_Correlation")
ARMS = ("A1", "A2", "A3_T")
TOL_M, TOL_P = 1e-9, 2e-3


def rows_test() -> list[dict]:
    """Test rows: one per (pmid, gene id set, disease id); a key with two labels is dropped."""
    groups: dict[tuple, list[dict]] = {}
    for r in csv.DictReader(open(ROOT / "data/CRED/cred_pairs.tsv", encoding="utf-8"), delimiter="\t"):
        if r["split"] == "test":
            groups.setdefault((r["pmid"], frozenset(r["gene_id"].split(";")), r["disease_id"]), []).append(r)
    return [g[0] for g in groups.values() if len({x["label"] for x in g}) == 1]


def probs(path: Path) -> dict[tuple, dict]:
    return {(d["pmid"], d["e1"], d["e2"]): d["probs"] for d in map(json.loads, open(path, encoding="utf-8"))}


def signed(rows: list[dict], p: dict) -> np.ndarray:
    return np.array([max((p[(r["pmid"], *k.split("|"))][CAUSAL[0]] + p[(r["pmid"], *k.split("|"))][CAUSAL[1]]
                          for k in r["pairs"].split(";") if k), default=0.0) for r in rows])


def predictions(rows: list[dict]) -> dict[str, np.ndarray]:
    th = json.loads((HERE / "s10_thresholds.json").read_text(encoding="utf-8"))
    a1 = json.loads((HERE / "s10_a1.json").read_text(encoding="utf-8"))
    a3 = json.loads((HERE / "s10_a3.json").read_text(encoding="utf-8"))
    base, bc8 = probs(HERE / "ens_biolinkbert_base/pred_cred.jsonl"), probs(HERE / "ens_bc8all/pred_cred.jsonl")
    out = {"R0_val": signed(rows, bc8) >= th["R0_val"]["threshold"],
           "cooccurrence": np.array([r["same_sent"] == "1" and r["status"] == "reachable" for r in rows]),
           "A2": signed(rows, probs(HERE / "testday/pred_cred_test_ens.jsonl")) >= th["A2_ens"]["threshold"]}
    first = lambda r: (r["pmid"], *r["pairs"].split(";")[0].split("|"))
    out["argmax_base"] = np.array([r["status"] == "reachable" and max(base[first(r)], key=base[first(r)].get) in CAUSAL
                                   for r in rows])
    # A1: the frozen linear model applied by hand (scaler, weights, intercept, logistic)
    w, b = np.array(a1["coef"]), a1["intercept"]
    mu, sd = np.array(a1["scaler_mean"]), np.array(a1["scaler_scale"])
    eps, s = 1e-6, np.zeros(len(rows))
    for i, r in enumerate(rows):
        if r["status"] != "reachable":
            continue
        k = first(r)
        x = np.array([np.log(base[k][CAUSAL[0]] + base[k][CAUSAL[1]] + eps), np.log(base[k]["Association"] + eps),
                      np.log(base[k]["None"] + eps), np.log(bc8[k][CAUSAL[0]] + bc8[k][CAUSAL[1]] + eps),
                      float(r["same_sent"] == "1")])
        s[i] = 1 / (1 + np.exp(-(((x - mu) / sd) @ w + b)))
    out["A1"] = s >= a1["threshold"]
    out["bc8all_bc8_threshold"] = signed(rows, bc8) >= 0.18429316672700224      # §9 item 6(a) threshold
    for seed in (42, 43, 44):
        f = HERE / f"testday/pred_cred_test_s{seed}.jsonl"
        if f.is_file():
            out[f"A2_s{seed}"] = signed(rows, probs(f)) >= th[f"A2_s{seed}"]["threshold"]
    scores = {(d["prompt"], d["condition"], d["pmid"], d["key"]): d["score"]
              for d in map(json.loads, open(HERE / "testday/a3_test_scores.jsonl", encoding="utf-8"))}
    for c in ("T", "K"):
        v = np.array([scores.get((a3["prompt"], c, r["pmid"], r["pairs"]), 0.0) if r["status"] == "reachable" else 0.0
                      for r in rows])
        out[f"A3_{c}"] = v >= a3[f"threshold_{c}"]
    return out


def f1_draws(y: np.ndarray, pred: np.ndarray, doc: np.ndarray, w: np.ndarray) -> np.ndarray:
    """F1 of the positive class per bootstrap draw of documents (weights w: [B, n_docs])."""
    per = np.zeros((doc.max() + 1, 3))
    np.add.at(per, doc, np.c_[y & pred, pred, y].astype(float))
    tp, npred, ngold = (w @ per).T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tp > 0, 2 * tp / (npred + ngold), 0.0)


def main() -> None:
    rep = json.loads((HERE / "cred_testday.json").read_text(encoding="utf-8"))
    rows = rows_test()
    y = np.array([r["label"] == "1" for r in rows])
    pm_all = [r["pmid"] for r in rows]
    docs, doc = np.unique(pm_all, return_inverse=True)
    w = np.random.default_rng(0).multinomial(len(docs), np.full(len(docs), 1 / len(docs)), size=10_000)
    pred = predictions(rows)
    bad: list[str] = []
    got: dict = {"n_pairs": len(rows), "n_causal": int(y.sum()), "n_docs": len(docs), "systems": {}, "holm": {}}
    if (len(rows), int(y.sum()), len(docs)) != (rep["n_pairs"], rep["n_causal"], rep["n_docs"]):
        bad.append("row counts differ")
    draws = {}
    for n, p in pred.items():
        tp, npred, ngold = float((y & p).sum()), float(p.sum()), float(y.sum())
        f1 = 2 * tp / (npred + ngold) if tp else 0.0
        got["systems"][n] = {"p": tp / npred if npred else 0.0, "r": tp / ngold, "f1": f1, "tp": int(tp)}
        draws[n] = f1_draws(y, p, doc, w)
        r = rep["systems"][n]
        for k in ("p", "r", "f1"):
            if abs(got["systems"][n][k] - r[k]) > TOL_M:
                bad.append(f"{n}.{k}: {got['systems'][n][k]:.6f} vs reported {r[k]:.6f}")
    p_one = {a: float((draws[a] - draws["R0_val"] <= 0).mean()) for a in ARMS}
    order = sorted(ARMS, key=lambda a: p_one[a])
    run = 0.0
    for i, a in enumerate(order):
        run = max(run, p_one[a] * (len(ARMS) - i))
        got["holm"][a] = {"p_one_sided": p_one[a], "p_holm": min(1.0, run)}
        r = rep["holm"]["arms"][a]
        for k in ("p_one_sided", "p_holm"):
            if abs(got["holm"][a][k] - r[k]) > TOL_P:
                bad.append(f"{a}.{k}: {got['holm'][a][k]:.4f} vs reported {r[k]:.4f}")
    # secondary blocks: leave-one-document-out range and the drop-largest-abstract sensitivity
    f1 = lambda yy, pp: (2 * float((yy & pp).sum()) / (pp.sum() + yy.sum())) if (yy & pp).any() else 0.0
    sec = rep.get("secondary", {})
    if "leave_one_document_out" in sec:
        for arm, r in sec["leave_one_document_out"].items():
            vals = [f1(y[[p != d for p in pm_all]], pred[arm][[p != d for p in pm_all]]) -
                    f1(y[[p != d for p in pm_all]], pred["R0_val"][[p != d for p in pm_all]]) for d in docs]
            got.setdefault("lodo", {})[arm] = {"min": min(vals), "max": max(vals)}
            for k in ("min", "max"):
                if abs(got["lodo"][arm][k] - r[k]) > TOL_M:
                    bad.append(f"lodo {arm}.{k}: {got['lodo'][arm][k]:.6f} vs reported {r[k]:.6f}")
    if "sensitivity_drop_largest_abstract" in sec:
        r = sec["sensitivity_drop_largest_abstract"]
        m = np.array([p != r["dropped_pmid"] for p in pm_all])
        got["drop_largest"] = {n: f1(y[m], pred[n][m]) for n in pred}
        for n, v in got["drop_largest"].items():
            if n in r["systems"] and abs(v - r["systems"][n]["f1"]) > TOL_M:
                bad.append(f"drop-largest {n}: {v:.6f} vs reported {r['systems'][n]['f1']:.6f}")
    test_pmids = {r["pmid"] for r in rows}
    leak = {}
    for f in ("data/BioRED/Train.BioC.JSON", "data/BioRED/Dev.BioC.JSON", "data/BioRED/Test.BioC.JSON",
              "data/BC8/bc8_test.json", "data/CRED/cred.json", "data/CRED/cred_val.json"):
        text = (ROOT / f).read_text(encoding="utf-8")
        leak[f] = sum(f'"{p}"' in text for p in test_pmids)
    got["leakage_pmids_in"] = leak
    if any(leak.values()):
        bad.append(f"CRED test PMIDs found in training or dev files: {leak}")
    got["files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((HERE / "testday").iterdir()) if p.suffix in (".jsonl", ".json")}
    got["mismatches"] = bad
    (HERE / "verify_cred_testday.json").write_text(json.dumps(got, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps({"n_pairs": got["n_pairs"], "n_causal": got["n_causal"], "systems_checked": len(got["systems"]),
                      "holm_checked": len(got["holm"]), "leakage": got["leakage_pmids_in"], "mismatches": bad}, indent=1))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
