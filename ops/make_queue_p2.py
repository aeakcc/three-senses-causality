r"""Paper 2 E3 queue (docs/research/paper2/codemap.md, E3): writes ops/queue_p2/e3_NNN_<name>.json, stale e3_* removed.

    python ops/make_queue_p2.py

In file order: p2e3_00_smoke (F1, fold 0, seed 42, 1 epoch), then for seeds 42-44 x folds 0-4
  F1 p2e3_f1_k{k}_s{seed}  pc head fine-tuned from re_bl_base_bc8all_s{seed} on CRED only, 12 fixed epochs
  F0 p2e3_f0_k{k}_s{seed}  the same from raw BioLinkBERT
  J  p2e3_j_k{k}_s{seed}   A2 recipe (BioRED Train + BC8, early stopping on BioRED dev) + cv/set/cred.json
each run followed by _p2e3_score_<run>: biocausal.score on cv/test_f{k}.json under [CORPUS_cred], into the run dir.
Fold k is held out of training by --heldout Test + cv/fold{k}.txt (docs/results/cred/cred_split.py cv).
Ship code and data with ops/sync_p2.sh first; the queue runner takes the files from D:\model-second\queue.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "queue_p2"
PY = r"cd /d D:\model-p2 && D:\model-second\.venv\Scripts\python.exe"
RUNS, DATA = r"D:\model-second\runs", r"D:\model-second\data"
BIORED, CV = rf"{DATA}\BioRED", rf"{DATA}\CRED\cv"
HP = "--patience 6 --batch-docs 4 --grad-accum 2 --eval-batch-docs 8 --lr-encoder 3e-5 --lr-head 1e-4"
SEEDS = (42, 43, 44)


def run(name: str, arm: str, k: int, seed: int, epochs: int = 12) -> list[dict]:
    """The training job of one arm/fold/seed and its scoring job."""
    held = rf"--dev {BIORED}\Dev.BioC.JSON --heldout {BIORED}\Test.BioC.JSON {CV}\fold{k}.txt"
    if arm == "j":
        args = (rf"--train {BIORED}\Train.BioC.JSON {DATA}\BC8\bc8_test.json {held} --joint {CV}\set\cred.json "
                f"--label-handling set --corpus-tag {HP} --epochs 30")
    else:
        init = rf"--init-from {RUNS}\re_bl_base_bc8all_s{seed}\best " if arm == "f1" else ""
        args = (rf"{init}--train {DATA}\empty.json {held} --joint {CV}\pc\cred.json --label-handling set --corpus-tag "
                f"--aux-heads pc --aux-weight 1.0 {HP} --epochs {epochs} --stop-epoch {epochs}")
    return [{"name": name, "cmd": f"{PY} -m biocausal.train_re --name {name} --runs-dir {RUNS} "
                                  f"--backbone michiyasunaga/BioLinkBERT-base {args} --seed {seed}"},
            {"name": f"_p2e3_score_{name[5:]}",
             "cmd": rf"{PY} -m biocausal.score --run {RUNS}\{name} --data {CV}\test_f{k}.json --tag cv --corpus cred "
                    rf"--out-dir {RUNS}\{name}"}]


def jobs() -> list[dict]:
    out = run("p2e3_00_smoke", "f1", 0, 42, epochs=1)
    for arm in ("f1", "f0", "j"):
        out += [j for seed in SEEDS for k in range(5) for j in run(f"p2e3_{arm}_k{k}_s{seed}", arm, k, seed)]
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("e3_*.json"):
        f.unlink()
    for i, j in enumerate(jobs()):
        (OUT / f"e3_{i:03d}_{j['name']}.json").write_text(json.dumps(j, indent=1) + "\n", encoding="utf-8")
    print(f"{i + 1} jobs -> {OUT}")


if __name__ == "__main__":
    main()
