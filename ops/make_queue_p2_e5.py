r"""Paper 2 E5 queue (sense V, 'ev' head): writes ops/queue_p2/e5_NNN_<name>.json, stale e5_* removed (sorts after e3_*).

    python ops/make_queue_p2_e5.py

In file order: p2e5_00_smoke (VL, seed 42, 1 epoch), then per arm p2e5_{arm}_s{42,43,44} and its two ensemble scoring
jobs _p2e5_score_{arm}_{test_2026,rgd_ext} -> runs\_p2e5_{arm}\pred_{tag}.jsonl (docs/results/paper2/e5_eval.py).
Arms: v0 = raw BioLinkBERT-base, vl = --init-from re_bl_base_bc8all_s{seed}, vp = --init-from a2_cred_bc8all_s{seed};
all train the ev head only (--train empty.json, --joint data\paper2\e5\clingen.json, 5 fixed epochs, BioRED dev logged).
Code runs from D:\model-p2e5 (ops/sync_p2e5.sh; D:\model-p2 keeps serving E1 / E3), data from D:\model-second
(ops/sync_p2_data.sh). Scoring: cwd D:\model-second, code via PYTHONPATH with PYTHONSAFEPATH=1 (make_queue_p2_score.py);
--corpus clingen works because each run's own config has corpus_tag and corpora ['clingen'] (score.score_run).
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "queue_p2"
MS, CODE = r"D:\model-second", r"D:\model-p2e5"
RUNS, DATA = rf"{MS}\runs", rf"{MS}\data"
E5 = rf"{DATA}\paper2\e5"
PY = rf"{MS}\.venv\Scripts\python.exe"
INIT = {"v0": None, "vl": "re_bl_base_bc8all", "vp": "a2_cred_bc8all"}
SEEDS, TAGS = (42, 43, 44), ("test_2026", "rgd_ext")


def train(name: str, arm: str, seed: int, epochs: int = 5) -> dict:
    init = rf"--init-from {RUNS}\{INIT[arm]}_s{seed}\best " if INIT[arm] else ""
    return {"name": name, "cmd": (
        rf"cd /d {CODE} && {PY} -m biocausal.train_re --name {name} --runs-dir {RUNS} "
        rf"--backbone michiyasunaga/BioLinkBERT-base {init}--train {DATA}\empty.json --dev {DATA}\BioRED\Dev.BioC.JSON "
        rf"--heldout {DATA}\BioRED\Test.BioC.JSON --joint {E5}\clingen.json --label-handling set --corpus-tag "
        f"--aux-heads ev --aux-weight 1.0 --epochs {epochs} --stop-epoch {epochs} --patience 6 --batch-docs 4 "
        f"--grad-accum 2 --eval-batch-docs 8 --lr-encoder 3e-5 --lr-head 1e-4 --seed {seed}")}


def score(arm: str, tag: str) -> dict:
    runs = " ".join(rf"{RUNS}\p2e5_{arm}_s{s}" for s in SEEDS)
    return {"name": f"_p2e5_score_{arm}_{tag}", "cmd": (
        rf"cd /d {MS} && set PYTHONPATH={CODE}&& set PYTHONSAFEPATH=1&& {PY} -m biocausal.score --ensemble {runs} "
        rf"--data {E5}\{tag}_docs.json --tag {tag} --corpus clingen --out-dir {RUNS}\_p2e5_{arm}")}


def jobs() -> list[dict]:
    out = [train("p2e5_00_smoke", "vl", 42, epochs=1)]
    for arm in INIT:
        out += [train(f"p2e5_{arm}_s{s}", arm, s) for s in SEEDS] + [score(arm, t) for t in TAGS]
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("e5_*.json"):
        f.unlink()
    for i, j in enumerate(jobs()):
        (OUT / f"e5_{i:03d}_{j['name']}.json").write_text(json.dumps(j, indent=1) + "\n", encoding="utf-8")
    print(f"{i + 1} jobs -> {OUT}")


if __name__ == "__main__":
    main()
