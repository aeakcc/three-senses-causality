"""Per-unit predictions of paper 2 as small TSVs for the public release (the raw pair-level pred files are ~100 MB
each). Uses the evaluators' own loaders, so every score here is the one their statistics were computed from.
Run from the repo root with the project venv: .venv/bin/python docs/results/paper2/export_predictions.py
Writes docs/results/paper2/predictions/*.tsv."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e1_eval import read_units, score_units  # noqa: E402
from e3_eval import ARMS as E3_ARMS, load_arms  # noqa: E402
from e3_probe import CV, OUT as E3_OUT, pool  # noqa: E402
from e5_eval import ARMS as E5_ARMS, EV_CLASSES, arm_scores  # noqa: E402

ROOT = HERE.parents[2]
RUNS, DATA, OUT = ROOT / "runs", ROOT / "data/paper2", HERE / "predictions"
OUT.mkdir(exist_ok=True)


def write(name: str, head: list[str], rows) -> None:
    with open(OUT / name, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(head)
        n = 0
        for r in rows:
            w.writerow(["" if isinstance(x, float) and np.isnan(x) else (repr(x) if isinstance(x, float) else x)
                        for x in r])
            n += 1
    print(f"{name}: {n} rows")


# E1 (curated units), E4 (construction sentences), E2 (name swaps): readers L, P, L' (= Lp) and delta (= d)
E1 = {"e1_clingen.tsv": (DATA / "e1/clingen_units.tsv", DATA / "e1/clingen_docs.json", "clingen_docs"),
      "e1_rgd.tsv": (DATA / "e1/rgd_units.tsv", DATA / "e1/rgd_docs.json", "rgd_docs"),
      "e4_percha.tsv": (DATA / "percha_units.tsv", DATA / "percha_docs.json", "percha_docs"),
      "e2_swap_clingen_u.tsv": (DATA / "e1/swap_clingen_u_units.tsv", DATA / "e1/swap_clingen_u.json", "swap_clingen_u"),
      "e2_swap_clingen_k.tsv": (DATA / "e1/swap_clingen_k_units.tsv", DATA / "e1/swap_clingen_k.json", "swap_clingen_k")}
for name, (units_path, docs_path, tag) in E1.items():
    units = read_units(units_path)
    docs = {d["pmid"]: d for d in json.loads(docs_path.read_text(encoding="utf-8"))}
    readers = {r: RUNS / f"_p2e1_{r}" / f"pred_{tag}.jsonl" for r in ("L", "P", "Lp")}
    anchors, s = score_units(units, docs, readers)
    write(name, ["unit_id", "pmid", "classes", "anchor", "s_L", "s_P", "s_Lprime", "delta"],
          ([u["unit_id"], u["pmid"], ";".join(u["classes"]), a, s["L"][i], s["P"][i], s["Lp"][i], s["d"][i]]
           for i, (u, a) in enumerate(zip(units, anchors))))

# E3: out-of-fold score of every arm on the CRED pool (FL, F0, J: mean over seeds 42-44)
rows, fold_of = pool(CV)
rows = [r for r in rows if r["pmid"] in fold_of]
fold = np.array([fold_of[r["pmid"]] for r in rows])
arms = load_arms(rows, fold, RUNS, E3_OUT)
missing = {a: v for a, v in arms.items() if isinstance(v, str)}
assert not missing, missing
write("e3_cred_pool.tsv", ["pmid", "gene_id", "disease_id", "label", "fold", "same_sentence", "status"] + list(E3_ARMS),
      ([r["pmid"], r["gene_id"], r["disease_id"], r["label"], int(fold[i]), r["same_sent"], r["status"]]
       + [float(arms[a][0][i]) for a in E3_ARMS] for i, r in enumerate(rows)))

# E5: evidence-type probabilities of the three heads (3-seed ensembles; the 2026 slice also per seed)
for tag in ("test_2026", "rgd_ext"):
    units = read_units(DATA / f"e5/{tag}_units.tsv")
    cols = {f"{s}_{c}": None for s in E5_ARMS for c in EV_CLASSES}
    probs = {s: arm_scores(RUNS / f"_p2e5_{arm}" / f"pred_{tag}.jsonl", units) for s, arm in E5_ARMS.items()}
    if tag == "test_2026":
        for s, arm in E5_ARMS.items():
            for seed in (42, 43, 44):
                probs[f"{s}_s{seed}"] = arm_scores(RUNS / "_p2e5_seed" / f"{arm}_s{seed}" / f"pred_{tag}.jsonl", units)
    bad = {k: v for k, v in probs.items() if isinstance(v, str)}
    assert not bad, bad
    head = ["unit_id", "pmid", "classes"] + [f"{s}_{c}" for s in probs for c in EV_CLASSES]
    write(f"e5_{tag}.tsv", head, ([u["unit_id"], u["pmid"], ";".join(u["classes"])]
                                  + [float(x) for s in probs for x in probs[s][i]] for i, u in enumerate(units)))
