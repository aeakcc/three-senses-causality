r"""Paper 2 scoring queue: E3 pair embeddings, then the E1 / E2 / E4 readers; writes ops/queue_p2/e1_NNN_<name>.json
(stale e1_* removed; e1_* sorts before the E3 training files e3_*).

    python ops/make_queue_p2_score.py

In file order:
  _p2e3_emb_{bc8all,raw}_s{42,43,44}   docs/results/paper2/embed.py on data\CRED\cred_docs.json -> runs\_p2e3_emb\
                                       {bc8all,raw}_s{seed}.npz (--run re_bl_base_bc8all_s{seed} / --backbone seeded)
  _p2e1_<reader>_<tag>                 biocausal.score of a reader ensemble -> runs\_p2e1_<reader>\pred_<tag>.jsonl
Readers: L  = re_bl_base_bc8all_s42-44; P = a2_cred_bc8all_s42-44 --corpus cred; Lp = the same A2 runs, no tag;
L0 = re_biolinkbert_base_s42-44. Data (tag = file stem): data\paper2\{clingen,rgd,percha,biotriplex}_docs.json and
swap\clingen_{u,k}.json for every reader; data\CRED\cred_docs.json and swap\cred_{u,k}_f0-4.json for L and L0 only: A2
trained on the CRED train split, score.py refuses the originals and cannot see the swaps' '<pmid>_s<i>' ids.
Every job: cwd D:\model-second (the bc8all / base configs hold data paths relative to it, and check_unseen resolves
the run it is pointed at against the cwd), code from D:\model-p2 via PYTHONPATH with PYTHONSAFEPATH=1 (D:\model-second's
own older biocausal must not shadow it). Ship code with ops/sync_p2.sh and data with ops/sync_p2_data.sh first.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "queue_p2"
MS = r"D:\model-second"
RUNS, DATA = rf"{MS}\runs", rf"{MS}\data"
PY = rf"set PYTHONPATH=D:\model-p2&& set PYTHONSAFEPATH=1&& {MS}\.venv\Scripts\python.exe"
SEEDS = (42, 43, 44)
ens = lambda run: "--ensemble " + " ".join(rf"{RUNS}\{run}_s{s}" for s in SEEDS)  # noqa: E731
READERS = {"L": ens("re_bl_base_bc8all"), "P": ens("a2_cred_bc8all") + " --corpus cred", "Lp": ens("a2_cred_bc8all"),
           "L0": ens("re_biolinkbert_base")}
# §11b: E1 / E2 score the single-candidate-pair sets of docs/results/paper2/e1_sets.py; L0 reads Percha, BioTriplex
# and CRED only (GPU budget)
DOCS = [rf"paper2\e1\{s}_docs.json" for s in ("clingen", "rgd")] + [rf"paper2\e1\swap_clingen_{c}.json" for c in "uk"]
SMALL = [rf"paper2\{s}_docs.json" for s in ("percha", "biotriplex")]
CRED = [r"CRED\cred_docs.json"] + [rf"paper2\swap\cred_{c}_f{k}.json" for c in "uk" for k in range(5)]


def jobs() -> list[dict]:
    out = []
    for kind, src in (("bc8all", rf"--run {RUNS}\re_bl_base_bc8all_s"),
                      ("raw", "--backbone michiyasunaga/BioLinkBERT-base --seed ")):
        for s in SEEDS:   # embed.py does not create the output dir
            out.append({"name": f"_p2e3_emb_{kind}_s{s}",
                        "cmd": rf"cd /d {MS} && (mkdir runs\_p2e3_emb 2>nul & {PY} D:\model-p2\docs\results\paper2\embed.py "
                               rf"{src}{s} --data {DATA}\CRED\cred_docs.json --out {RUNS}\_p2e3_emb\{kind}_s{s}.npz)"})
    for reader, runs in READERS.items():
        for f in (DOCS if reader != "L0" else []) + SMALL + (CRED if reader in ("L", "L0") else []):
            tag = f.rsplit("\\", 1)[1][:-5]
            out.append({"name": f"_p2e1_{reader}_{tag}", "cmd": rf"cd /d {MS} && {PY} -m biocausal.score {runs} "
                                                              rf"--data {DATA}\{f} --tag {tag} --out-dir {RUNS}\_p2e1_{reader}"})
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("e1_*.json"):
        f.unlink()
    for i, j in enumerate(jobs()):
        (OUT / f"e1_{i:03d}_{j['name']}.json").write_text(json.dumps(j, indent=1) + "\n", encoding="utf-8")
    print(f"{i + 1} jobs -> {OUT}")


if __name__ == "__main__":
    main()
