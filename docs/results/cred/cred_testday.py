"""Preregistration §10, CRED test day: one table, the Holm family of three arms, and the estimate rows.

    PYTHONPATH=. .venv/bin/python docs/results/cred/cred_testday.py
      reads  data/CRED/cred_pairs.tsv (labels stay local), docs/results/cred/s10_{thresholds,a1,a3}.json,
             docs/results/cred/ens_{biolinkbert_base,bc8all}/pred_cred.jsonl        (zero-shot, §8.3)
             docs/results/cred/testday/pred_cred_test_{ens,s42,s43,s44}.jsonl       (A2, from the PC)
             docs/results/cred/testday/a3_test_scores.jsonl                         (A3, from the PC)
      writes docs/results/cred/cred_testday.json, cred_testday.md

Primary set: CRED test, strict, keys merged by cred_s10.load_rows (584 pairs, 52 causal); an unreachable pair is
non-causal with score 0. Confirmatory family (§10 item 6): F1(arm) > F1(R0_val) for A1, A2, A3-T, paired document
bootstrap (10 000, seed 0), one-sided p, Holm m = 3; a missing arm keeps m = 3 with p = 1. Everything else is an
estimate with a 95 % interval.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "docs" / "results" / "posthoc")]
from causal_audit import ap_boot, paired, prf_boot  # noqa: E402
from cred_a1_stacker import apply as a1_apply, read_probs  # noqa: E402
from cred_s10 import load_rows  # noqa: E402

TD = HERE / "testday"
CAUSAL = ("Positive_Correlation", "Negative_Correlation")
ARMS = ("A1", "A2", "A3_T")            # the Holm family, in this order; m stays 3 if one is missing
N_BOOT_SEED = 0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING"


def causal_score(rows: list[dict], probs: dict) -> np.ndarray:
    """p(Pos) + p(Neg) of a row's candidate pair (max over a composite), 0 for an unreachable pair."""
    out = []
    for r in rows:
        keys = [(r["pmid"], *k.split("|")) for k in r["pairs"].split(";") if k]
        out.append(max((probs[k][CAUSAL[0]] + probs[k][CAUSAL[1]] for k in keys), default=0.0))
    return np.array(out)


def systems(rows: list[dict]) -> dict[str, dict]:
    """Every scored system: score per row, decision rule and the file it came from."""
    th = json.loads((HERE / "s10_thresholds.json").read_text(encoding="utf-8"))
    a1 = json.loads((HERE / "s10_a1.json").read_text(encoding="utf-8"))
    zero = {n: read_probs(HERE / f"ens_{n}" / "pred_cred.jsonl") for n in ("biolinkbert_base", "bc8all")}
    base_s, bc8_s = causal_score(rows, zero["biolinkbert_base"]), causal_score(rows, zero["bc8all"])
    out: dict[str, dict] = {
        "R0_val": {"score": bc8_s, "t": th["R0_val"]["threshold"], "file": "ens_bc8all/pred_cred.jsonl"},
        "A1": {"score": a1_apply(a1, rows), "t": a1["threshold"], "file": "s10_a1.json"},
        "cooccurrence": {"pred": np.array([r["same_sent"] == "1" and r["status"] == "reachable" for r in rows]),
                         "file": "cred_pairs.tsv"},
        "argmax_base": {"pred": np.array([max(zero["biolinkbert_base"][(r["pmid"], *r["pairs"].split(";")[0].split("|"))],
                                              key=lambda k: zero["biolinkbert_base"][(r["pmid"], *r["pairs"].split(";")[0].split("|"))][k]) in CAUSAL
                                          if r["status"] == "reachable" else False for r in rows]),
                        "score": base_s, "file": "ens_biolinkbert_base/pred_cred.jsonl"},
        "bc8all_bc8_threshold": {"score": bc8_s, "t": 0.18429316672700224, "file": "causal_audit.json §9.6a"},
    }
    for name, f in (("A2", "pred_cred_test_ens.jsonl"), ("A2_s42", "pred_cred_test_s42.jsonl"),
                    ("A2_s43", "pred_cred_test_s43.jsonl"), ("A2_s44", "pred_cred_test_s44.jsonl")):
        p = TD / f
        if p.exists():
            key = "A2_ens" if name == "A2" else name.replace("A2_", "A2_")
            out[name] = {"score": causal_score(rows, read_probs(p)), "t": th[key]["threshold"], "file": f}
    a3_file, a3_json = TD / "a3_test_scores.jsonl", HERE / "s10_a3.json"
    if a3_file.exists() and a3_json.exists():
        a3 = json.loads(a3_json.read_text(encoding="utf-8"))
        got = {(s["prompt"], s["condition"], s["pmid"], s["key"]): s["score"]
               for s in map(json.loads, open(a3_file, encoding="utf-8"))}
        for cond in ("T", "K"):
            if any(k[1] == cond for k in got):
                out[f"A3_{cond}"] = {
                    "score": np.array([got.get((a3["prompt"], cond, r["pmid"], r["pairs"]), 0.0)
                                       if r["status"] == "reachable" else 0.0 for r in rows]),
                    "t": a3[f"threshold_{cond}"], "file": "a3_test_scores.jsonl"}
    return out


def main() -> None:
    rows = load_rows("test")
    y = np.array([r["label"] == "1" for r in rows])
    pmid = [r["pmid"] for r in rows]
    sysinfo = systems(rows)
    res: dict = {"what": "preregistration §10 CRED test day", "n_pairs": len(rows), "n_causal": int(y.sum()),
                 "n_docs": len(set(pmid)), "seed": N_BOOT_SEED, "files": {}, "systems": {}, "holm": {}, "estimates": {}}
    boots: dict[str, np.ndarray] = {}
    for name, s in sysinfo.items():
        pred = s["pred"] if "pred" in s else s["score"] >= s["t"]
        m, bf = prf_boot(y, pred, pmid)
        boots[name] = bf
        res["systems"][name] = {**m, "threshold": s.get("t"), "file": s["file"]}
        res["files"][name] = sha(TD / Path(s["file"]).name) if (TD / Path(s["file"]).name).exists() else s["file"]
    p_one = {}
    for arm in ARMS:
        p_one[arm] = 1.0 if arm not in boots else float((boots[arm] - boots["R0_val"] <= 0).mean())
    order = sorted(ARMS, key=lambda a: p_one[a])
    holm, run = {}, 0.0
    for i, arm in enumerate(order):
        run = max(run, p_one[arm] * (len(ARMS) - i))
        holm[arm] = {"p_one_sided": p_one[arm], "p_holm": min(1.0, run), "survives": min(1.0, run) < 0.05,
                     "delta_f1": None if arm not in boots else
                     res["systems"][arm]["f1"] - res["systems"]["R0_val"]["f1"],
                     **({} if arm not in boots else paired(boots[arm], boots["R0_val"]))}
    res["holm"] = {"m": len(ARMS), "alpha": 0.05, "comparator": "R0_val", "arms": holm}
    for a in sysinfo:
        for b in ("R0_val", "cooccurrence"):
            if a != b and a in boots:
                res["estimates"][f"{a} - {b}"] = {"delta_f1": res["systems"][a]["f1"] - res["systems"][b]["f1"],
                                                  **paired(boots[a], boots[b])}
    # ---- preregistered secondary analyses (§10 items 2, 6, 7; frozen predictions, no new scoring) ----------------
    sec: dict = {}
    for name, mask in (("reachable", np.array([r["status"] == "reachable" for r in rows])),
                       ("same_sentence", np.array([r["same_sent"] == "1" for r in rows])),
                       ("cross_sentence", np.array([r["same_sent"] != "1" for r in rows]))):
        sub = {"n_pairs": int(mask.sum()), "n_causal": int(y[mask].sum()), "systems": {}}
        bb: dict[str, np.ndarray] = {}
        for n, sy in sysinfo.items():
            pred = (sy["pred"] if "pred" in sy else sy["score"] >= sy["t"])[mask]
            m, bf = prf_boot(y[mask], pred, [p for p, k in zip(pmid, mask) if k])
            bb[n] = bf
            sub["systems"][n] = {k: m[k] for k in ("p", "r", "f1", "f1_ci", "tp", "n_pred", "n_gold")}
        sub["vs_comparator"] = {a: {"delta_f1": sub["systems"][a]["f1"] - sub["systems"]["R0_val"]["f1"],
                                    **paired(bb[a], bb["R0_val"])} for a in ARMS if a in bb}
        sec[name] = sub
    scored = {n: s["score"] for n, s in sysinfo.items() if "score" in s}
    scored["cooccurrence"] = sysinfo["cooccurrence"]["pred"].astype(float)
    aps, apboot = ap_boot(y, scored, pmid)
    sec["auprc"] = aps
    sec["auprc_diffs"] = {f"{a} - {b}": {"diff": aps[a]["auprc"] - aps[b]["auprc"], **paired(apboot[a], apboot[b])}
                          for a, b in (("A3_T", "A3_K"), ("A1", "cooccurrence"), ("A2", "cooccurrence"),
                                       ("R0_val", "cooccurrence")) if a in aps and b in aps}
    sec["f1_diffs_extra"] = {f"{a} - {b}": {"delta_f1": res["systems"][a]["f1"] - res["systems"][b]["f1"],
                                            **paired(boots[a], boots[b])}
                             for a, b in (("A3_T", "A3_K"), ("A1", "A2"), ("A1", "A3_T"), ("A2", "A3_T"))
                             if a in boots and b in boots}
    # realized resolution: SE of each paired difference and the one-sided Holm MDE it implies
    z_holm, z_power = 2.128, 0.8416      # one-sided alpha 0.05/3 (smallest Holm step) and 80 % power
    sec["realized_resolution"] = {}
    for arm in ARMS:
        if arm in boots:
            se = float(np.std(boots[arm] - boots["R0_val"], ddof=1))
            sec["realized_resolution"][arm] = {"se": se, "mde_one_sided_holm": (z_holm + z_power) * se}
    # leave-one-document-out range of the arm minus comparator difference
    docs = sorted(set(pmid))
    lodo = {}
    for arm in ARMS:
        if arm not in sysinfo:
            continue
        vals = []
        for d in docs:
            m = np.array([p != d for p in pmid])
            f = []
            for n in (arm, "R0_val"):
                sy = sysinfo[n]
                pred = (sy["pred"] if "pred" in sy else sy["score"] >= sy["t"])[m]
                tp = float((pred & y[m]).sum())
                f.append(0.0 if tp == 0 else 2 * tp / (pred.sum() + y[m].sum()))
            vals.append(f[0] - f[1])
        lodo[arm] = {"min": min(vals), "max": max(vals), "n_docs": len(docs)}
    sec["leave_one_document_out"] = lodo
    # sensitivity: the one abstract that supplies 11 of the 52 causal pairs
    top = max(docs, key=lambda d: sum(y[i] for i, p in enumerate(pmid) if p == d))
    m = np.array([p != top for p in pmid])
    drop = {"dropped_pmid": top, "n_causal_dropped": int(y[[p == top for p in pmid]].sum()),
            "n_pairs": int(m.sum()), "n_causal": int(y[m].sum()), "systems": {}}
    bb = {}
    for n, sy in sysinfo.items():
        pred = (sy["pred"] if "pred" in sy else sy["score"] >= sy["t"])[m]
        met, bf = prf_boot(y[m], pred, [p for p, k in zip(pmid, m) if k])
        bb[n] = bf
        drop["systems"][n] = {k: met[k] for k in ("p", "r", "f1", "f1_ci")}
    drop["vs_comparator"] = {a: {"delta_f1": drop["systems"][a]["f1"] - drop["systems"]["R0_val"]["f1"],
                                 **paired(bb[a], bb["R0_val"])} for a in ARMS if a in bb}
    sec["sensitivity_drop_largest_abstract"] = drop
    # the CRED authors' own per-row predictions on the rows we can match (their entities, their Label column)
    svm_path = ROOT / "data" / "CRED" / "source" / "test_data_analysis.csv"
    if svm_path.exists():
        import csv as _csv
        theirs = {(r["pmid"], r["id1"], r["id2"]): r for r in _csv.DictReader(open(svm_path, encoding="utf-8"))}
        match = [(r, theirs[(r["pmid"], r["gene_id"], r["disease_id"])]) for r in rows
                 if (r["pmid"], r["gene_id"], r["disease_id"]) in theirs]
        yy = np.array([t["Label"] == "1" for _, t in match])
        pm = [r["pmid"] for r, _ in match]
        out_svm = {"n_matched": len(match), "n_causal": int(yy.sum()),
                   "label_agreement_with_ours": float(np.mean([(r["label"] == "1") == (t["Label"] == "1")
                                                               for r, t in match]))}
        out_svm["their_svm"] = prf_boot(yy, np.array([t["Prediction"] == "1" for _, t in match]), pm)[0]
        idx = {(r["pmid"], r["pairs"]): i for i, r in enumerate(rows)}
        for n in ("A1", "A2", "A3_T", "R0_val"):
            if n in sysinfo:
                sy = sysinfo[n]
                pred = sy["pred"] if "pred" in sy else sy["score"] >= sy["t"]
                out_svm[n] = prf_boot(yy, np.array([pred[idx[(r["pmid"], r["pairs"])]] for r, _ in match]), pm)[0]
        sec["cred_authors_rows"] = out_svm
    res["secondary"] = sec
    (HERE / "cred_testday.json").write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    lines = [f"# CRED test day (§10): {res['n_pairs']} pairs, {res['n_causal']} causal, {res['n_docs']} abstracts", "",
             "| Система | Порог | P | R | F1 [95 % CI] |", "|---|---|---|---|---|"]
    for n, m in res["systems"].items():
        t = "—" if m["threshold"] is None else f"{m['threshold']:.4f}"
        lines.append(f"| {n} | {t} | {100 * m['p']:.1f} | {100 * m['r']:.1f} | {100 * m['f1']:.1f} "
                     f"[{100 * m['f1_ci'][0]:.1f}; {100 * m['f1_ci'][1]:.1f}] |")
    lines += ["", "| Рука − R0_val | ΔF1 | 95 % CI | p одностор. | p Холма (m=3) | значимо |", "|---|---|---|---|---|---|"]
    for arm, h in res["holm"]["arms"].items():
        d = "—" if h["delta_f1"] is None else f"{100 * h['delta_f1']:+.1f}"
        ci = "—" if "diff_ci" not in h else f"[{100 * h['diff_ci'][0]:+.1f}; {100 * h['diff_ci'][1]:+.1f}]"
        lines.append(f"| {arm} | {d} | {ci} | {h['p_one_sided']:.4f} | {h['p_holm']:.4f} | "
                     f"{'да' if h['survives'] else 'нет'} |")
    d = sec["sensitivity_drop_largest_abstract"]
    lines += ["", f"Чувствительность (пост-хок): без PMID {d['dropped_pmid']} ({d['n_causal_dropped']} из "
              f"{res['n_causal']} каузальных пар) остаётся {d['n_pairs']} пар и {d['n_causal']} каузальных.", "",
              "| Система | F1 без этой аннотации |", "|---|---|"]
    lines += [f"| {n} | {100 * m['f1']:.1f} |" for n, m in d["systems"].items()]
    lines += ["", "Leave-one-document-out разности к компаратору: " + "; ".join(
        f"{a} от {100 * v['min']:+.1f} до {100 * v['max']:+.1f}" for a, v in sec["leave_one_document_out"].items()), ""]
    (HERE / "cred_testday.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
