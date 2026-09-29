r"""Paper 2, §11b item 4 (E2 on CRED): score the name-swap docs of fold k with the E3 fold-k models of arms FL, F0, J
(ensemble of seeds 42-44) -> runs\_p2e3z_{arm}_k{k}\pred_cred_{u,k}.jsonl; writes ops/queue_p2/e3z_NNN_*.json (after e3_*).

    python ops/make_queue_p2_e3z.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "queue_p2"
MS = r"D:\model-second"
ARMS = {"FL": "p2e3_f1", "F0": "p2e3_f0", "J": "p2e3_j"}


def jobs() -> list[dict]:
    out = []
    for arm, run in ARMS.items():
        for k in range(5):
            runs = " ".join(rf"{MS}\runs\{run}_k{k}_s{s}" for s in (42, 43, 44))
            for c in "uk":
                out.append({"name": f"_p2e3z_{arm}_k{k}_{c}",
                            "cmd": rf"cd /d D:\model-p2 && {MS}\.venv\Scripts\python.exe -m biocausal.score --ensemble {runs} "
                                   rf"--data {MS}\data\paper2\swap\cred_{c}_f{k}.json --tag cred_{c} --corpus cred "
                                   rf"--out-dir {MS}\runs\_p2e3z_{arm}_k{k}"})
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("e3z_*.json"):
        f.unlink()
    for i, j in enumerate(jobs()):
        (OUT / f"e3z_{i:03d}_{j['name']}.json").write_text(json.dumps(j, indent=1) + "\n", encoding="utf-8")
    print(f"{i + 1} jobs -> {OUT}")


if __name__ == "__main__":
    main()
