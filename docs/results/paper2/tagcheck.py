"""Paper 2, exploratory (post hoc, not in §11): does the corpus tag change the joint model (A2 s42-44) on BioRED Dev?
Scores of ops/queue_p2/e6_000 (runs/_p2x_a2_dev/pred_dev_{biored,cred}.jsonl) -> docs/results/paper2/tagcheck.json.

    PYTHONPATH=. .venv/bin/python docs/results/paper2/tagcheck.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs" / "_p2x_a2_dev"
SIGN = ("Positive_Correlation", "Negative_Correlation")


def load(tag):
    return {(r["pmid"], r["e1"], r["e2"]): r for r in map(json.loads, open(RUN / f"pred_{tag}.jsonl", encoding="utf-8"))}


def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x) - np.log1p(-x)


if __name__ == "__main__":
    b, c = load("dev_biored"), load("dev_cred")
    keys = [k for k in b if b[k]["t1"] == "DiseaseOrPhenotypicFeature" and b[k]["t2"] in ("GeneOrGeneProduct", "SequenceVariant")]
    sb, sc = (np.array([p[k]["probs"][SIGN[0]] + p[k]["probs"][SIGN[1]] for k in keys]) for p in (b, c))
    gold = np.array([b[k]["gold"] in SIGN for k in keys])
    d = logit(sc) - logit(sb)
    out = {"n_dg_pairs": len(keys), "n_gold_signed": int(gold.sum()), "mean_abs_delta": float(abs(d).mean()),
           "max_abs_delta": float(abs(d).max()), "spearman": float(spearmanr(sb, sc).statistic),
           "argmax_agreement_all_pairs": float(np.mean([max(b[k]["probs"], key=b[k]["probs"].get) ==
                                                        max(c[k]["probs"], key=c[k]["probs"].get) for k in b])),
           "mean_s_gold_signed": {"biored_tag": float(sb[gold].mean()), "cred_tag": float(sc[gold].mean())},
           "micro_f1_dev_biored_tag": json.load(open(RUN / "metrics_dev_biored.json"))["micro"]["f1"]}
    (ROOT / "docs/results/paper2/tagcheck.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
