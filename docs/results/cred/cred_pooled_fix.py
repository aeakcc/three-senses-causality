"""Preregistration §10 item 11 (post hoc): the pooled CRED numbers of §8.3 without the partially labelled abstracts.

    PYTHONPATH=. .venv/bin/python docs/results/cred/cred_pooled_fix.py
      reads  data/CRED/cred_pairs.tsv, docs/results/cred/ens_*/pred_cred.jsonl
      writes docs/results/cred/cred_pooled_fix.json, cred_pooled_fix.md

In 107 of the 218 CRED train abstracts only causal pairs are labelled (median 0.16 of the PubTator3 gene-disease
candidate pairs), which inflates prevalence, AUPRC and the co-occurrence baseline of the pooled table. Here they are
dropped; val and test keep every abstract. Keys are merged and conflicting keys dropped as in cred_s10.load_rows.
Systems and metrics are those of docs/results/cred/cred_eval.py (§8.3): argmax causal, score p(Pos)+p(Neg),
document bootstrap 10 000 / seed 0, strict (unreachable pair = non-causal, score 0) and reachable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "docs" / "results" / "posthoc")]
from causal_audit import ap_boot, paired, prf_boot  # noqa: E402
from cred_a1_stacker import read_probs  # noqa: E402
from cred_s10 import load_rows  # noqa: E402

CAUSAL = ("Positive_Correlation", "Negative_Correlation")
ENS = {"bc8all": "ens_bc8all", "biored_only": "ens_biolinkbert_base"}


def rows_without_partial() -> list[dict]:
    train = load_rows("train")
    labels: dict[str, set] = {}
    for r in train:
        labels.setdefault(r["pmid"], set()).add(r["label"])
    partial = {p for p, v in labels.items() if v == {"1"}}
    assert len(partial) == 107, len(partial)
    return [r for r in train if r["pmid"] not in partial] + load_rows("val") + load_rows("test")


def partial_stats() -> dict:
    """The train abstracts whose labelled pairs are all causal: how many, and how much of the candidate set they label."""
    train = load_rows("train")
    labels: dict[str, set] = {}
    for r in train:
        labels.setdefault(r["pmid"], set()).add(r["label"])
    partial = sorted(p for p, v in labels.items() if v == {"1"})
    docs = {d["pmid"]: d for d in json.loads((ROOT / "data" / "CRED" / "cred_docs.json").read_text(encoding="utf-8"))}
    cov = []
    for pmid in partial:
        d = docs[pmid]
        cand = sum({d["entities"][p["e1"]]["type"], d["entities"][p["e2"]]["type"]} ==
                   {"DiseaseOrPhenotypicFeature", "GeneOrGeneProduct"} for p in d["pairs"])
        lab = sum(r["pmid"] == pmid and r["status"] == "reachable" for r in train)
        cov.append(lab / cand if cand else 1.0)
    return {"n_train_abstracts": len({r["pmid"] for r in train}), "n_partial": len(partial),
            "median_candidate_coverage": float(np.median(cov))}


def label_shares(rows: list[dict], probs: dict) -> dict:
    """Of the causal pairs of the corrected pool: share predicted non-None, Association or signed."""
    sub = [r for r in rows if r["status"] == "reachable" and r["label"] == "1" and ";" not in r["pairs"]]
    out = {}
    for n in ENS:
        lab = [max(probs[n][(r["pmid"], *r["pairs"].split("|"))], key=probs[n][(r["pmid"], *r["pairs"].split("|"))].get)
               for r in sub]
        out[n] = {"n": len(sub), "any_relation": float(np.mean([x != "None" for x in lab])),
                  "association": float(np.mean([x == "Association" for x in lab])),
                  "signed": float(np.mean([x in CAUSAL for x in lab]))}
    return out


def main() -> None:
    rows = rows_without_partial()
    probs = {n: read_probs(HERE / d / "pred_cred.jsonl") for n, d in ENS.items()}
    out: dict = {"what": "post hoc §10.11: pooled CRED without the 107 partially labelled train abstracts",
                 "n_abstracts": len({r["pmid"] for r in rows}), "n_pairs_strict": len(rows),
                 "partial_abstracts": partial_stats(), "causal_pair_labels": label_shares(rows, probs), "variants": {}}
    for variant in ("strict", "reachable"):
        sub = rows if variant == "strict" else [r for r in rows if r["status"] == "reachable"]
        y = np.array([r["label"] == "1" for r in sub])
        pmid = [r["pmid"] for r in sub]
        v: dict = {"n_pairs": len(sub), "n_causal": int(y.sum()), "n_docs": len(set(pmid)),
                   "prevalence": float(y.mean()), "systems": {}}
        scores: dict[str, np.ndarray] = {}
        for n in ENS:
            keys = [[(r["pmid"], *k.split("|")) for k in r["pairs"].split(";") if k] for r in sub]
            s = np.array([max((probs[n][k][CAUSAL[0]] + probs[n][k][CAUSAL[1]] for k in ks), default=0.0) for ks in keys])
            pred = np.array([any(max(probs[n][k], key=probs[n][k].get) in CAUSAL for k in ks) for ks in keys])
            scores[n] = s
            v["systems"][n] = prf_boot(y, pred, pmid)[0]
        co = np.array([r["same_sent"] == "1" and r["status"] == "reachable" for r in sub])
        v["systems"]["cooccurrence"] = prf_boot(y, co, pmid)[0]
        v["systems"]["pubtator3_signed"] = prf_boot(
            y, np.array([any(t in CAUSAL for t in r["pt3_type"].split(";")) for r in sub]), pmid)[0]
        scores["cooccurrence"] = co.astype(float)
        aps, boot = ap_boot(y, scores, pmid)
        v["auprc"] = aps
        v["auprc_vs_cooccurrence"] = {n: {"diff": aps[n]["auprc"] - aps["cooccurrence"]["auprc"],
                                          **paired(boot[n], boot["cooccurrence"])} for n in ENS}
        out["variants"][variant] = v
    (HERE / "cred_pooled_fix.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    L = ["# Пул CRED без частично размеченных аннотаций (пост-хок §10.11)", "",
         f"Аннотаций {out['n_abstracts']}, пар (strict) {out['n_pairs_strict']}.", ""]
    for variant, v in out["variants"].items():
        L += [f"## {variant}: {v['n_pairs']} пар, {v['n_causal']} каузальных, доля {100 * v['prevalence']:.1f} %", "",
              "| Система | P | R | F1 [95 % CI] | AUPRC [95 % CI] |", "|---|---|---|---|---|"]
        for n, m in v["systems"].items():
            ap = v["auprc"].get(n)
            a = "—" if not ap else f"{100 * ap['auprc']:.1f} [{100 * ap['auprc_ci'][0]:.1f}; {100 * ap['auprc_ci'][1]:.1f}]"
            L.append(f"| {n} | {100 * m['p']:.1f} | {100 * m['r']:.1f} | {100 * m['f1']:.1f} "
                     f"[{100 * m['f1_ci'][0]:.1f}; {100 * m['f1_ci'][1]:.1f}] | {a} |")
        L.append("")
    (HERE / "cred_pooled_fix.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
