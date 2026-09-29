"""E5 heads, seed by seed (descriptive, post hoc; asked for by the supervisor on 2026-09-28): macro AUROC and
per-class AUROC of each single seed on the 2026 slice, next to the preregistered 3-seed ensemble.
Inputs: RUNS/_p2e5_seed/{v0,vl,vp}_s{42,43,44}/pred_test_2026.jsonl (biocausal.score --ensemble with one run dir)
and RUNS/_p2e5_{v0,vl,vp}/pred_test_2026.jsonl. Run from the repo root with the project venv:
.venv/bin/python docs/results/paper2/e5_seeds.py   -> docs/results/paper2/e5/e5_seeds.md"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e5_eval import ARMS, EV_CLASSES, ROOT, arm_scores, auroc, read_units  # noqa: E402

RUNS = ROOT / "runs"
units = read_units(ROOT / "data/paper2/e5/test_2026_units.tsv")
Y = np.array([[c in u["classes"] for c in EV_CLASSES] for u in units], dtype=bool)
w = np.ones((1, len(units)))


def per_class(path: Path) -> np.ndarray:
    S = arm_scores(path, units)
    assert not isinstance(S, str), S
    return np.array([auroc(S[:, j], Y[:, j], w)[0] for j in range(len(EV_CLASSES))])


lines = ["# E5 heads by seed, 2026 slice (244 units; point estimates, no bootstrap)", "",
         "| Head | Seed | macro AUROC | " + " | ".join(EV_CLASSES) + " |", "|---|---|---|" + "---|" * len(EV_CLASSES)]
for name, arm in ARMS.items():
    macros = []
    for seed in (42, 43, 44):
        a = per_class(RUNS / "_p2e5_seed" / f"{arm}_s{seed}" / "pred_test_2026.jsonl")
        macros.append(a.mean())
        lines.append(f"| {name} | {seed} | {a.mean():.3f} | " + " | ".join(f"{x:.3f}" for x in a) + " |")
    e = per_class(RUNS / f"_p2e5_{arm}" / "pred_test_2026.jsonl")
    lines.append(f"| {name} | ensemble | {e.mean():.3f} | " + " | ".join(f"{x:.3f}" for x in e) + " |")
    lines.append(f"| {name} | seed range | {min(macros):.3f}-{max(macros):.3f} (spread {max(macros) - min(macros):.3f}) |"
                 + " |" * len(EV_CLASSES))
out = HERE / "e5" / "e5_seeds.md"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
