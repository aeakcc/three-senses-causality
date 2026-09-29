"""Out-of-fold teacher probabilities for ensemble distillation (preregistration §7 row 3, improvement_plan §4.3 kd).

``--mode oof``: split the ``--train`` documents (BioRED Train) into ``--folds`` folds (seeded, by document), train one
model per fold on the other folds of ``--train`` only (BioRED-only base configuration: no BC8, whatever the student
trains on), predict the held fold with it and write every out-of-fold pair to ``<runs-dir>/<name>/oof_probs.jsonl``
(biocausal.predict rows: pmid, e1, e2, gold, pred, probs).

The student then trains with ``train_re --distill <that file> --distill-alpha 0.5 --distill-temp 2.0``: loss
CE + alpha * KL on pairs with teacher probabilities; the BC8 pairs of the bc8all student have none (BC8 is not fold
material) and get plain cross-entropy.

**No teacher probability may come from a held-out document**: the fold's own documents are the only ones scored, and
the ``--dev`` / ``--heldout`` files (BioRED Dev, BioRED Test, BC8) are asserted to be disjoint from what is written.
train_re re-checks the teacher file against the dev / held-out files recorded in ``oof_summary.json``.

Every fold is an ordinary run under ``<runs-dir>/<name>/f<k>/``, reused when it already has a metrics.json (a crashed job
can simply be restarted) — but only if the PMIDs it held out (``f<k>/fold_pmids.txt``, written before training) are
exactly the new fold's; otherwise the job stops. ``oof_summary.json`` records code_version and the flags passed through
to train_re.

    python -m biocausal.distill --mode oof --name w3_oof --runs-dir runs --train data/BioRED/Train.BioC.JSON \\
        --dev data/BioRED/Dev.BioC.JSON --heldout data/BioRED/Test.BioC.JSON data/BC8/bc8_test.json \\
        --folds 5 --seed 42 --backbone michiyasunaga/BioLinkBERT-base --epochs 30 --patience 6   # ... train_re flags
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from biocausal.data import load_docs, pmid_held, read_pmids
from biocausal.evaluate import write_jsonl
from biocausal.score import score_run
from biocausal.train_re import code_version
from biocausal.train_re import main as train_main


def folds(docs: list[dict], k: int, seed: int) -> list[list[dict]]:
    """``k`` document folds of a seeded shuffle (every document in exactly one fold)."""
    order = list(docs)
    random.Random(seed).shuffle(order)
    return [order[i::k] for i in range(k)]


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["oof"], default="oof")
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", nargs="+", required=True, help="fold material (BioRED Train)")
    ap.add_argument("--dev", required=True)
    ap.add_argument("--heldout", nargs="+", default=[], metavar="FILE", help="never a teacher document (Test, BC8)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-docs", type=int, help="smoke tests: cap docs per file (folds and train_re see the same)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a, rest = ap.parse_known_args(argv)                      # rest: --backbone, --epochs, ... -> train_re
    out_dir = Path(a.runs_dir) / a.name
    out_dir.mkdir(parents=True, exist_ok=True)

    held = set().union(*(read_pmids(p) for p in [a.dev, *a.heldout]))
    docs = [d for d in load_docs(a.train, a.limit_docs) if not pmid_held(d["pmid"], held) and d["pairs"]]
    rest += ["--limit-docs", str(a.limit_docs)] if a.limit_docs else []
    rows, meta = [], []
    for k, fold in enumerate(folds(docs, a.folds, a.seed)):
        data = out_dir / f"f{k}.json"                         # the held fold: scored, never trained on
        pmids = out_dir / f"f{k}_pmids.txt"
        run = out_dir / f"f{k}"
        fold_pmids = " ".join(d["pmid"] for d in fold)
        reuse = (run / "metrics.json").exists()
        if reuse and (not (run / "fold_pmids.txt").exists()
                      or read_pmids(run / "fold_pmids.txt") != {d["pmid"] for d in fold}):
            raise ValueError(f"{run} was trained with other held-out PMIDs than fold {k} of this split "
                             f"({run / 'fold_pmids.txt'}): remove it or use another --name")
        data.write_text(json.dumps(fold), encoding="utf-8")
        pmids.write_text(fold_pmids, encoding="utf-8")
        if not reuse:
            run.mkdir(exist_ok=True)
            (run / "fold_pmids.txt").write_text(fold_pmids, encoding="utf-8")
            if a.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()             # gpu_peak_mem_mb of this fold only
            train_main(["--name", f"f{k}", "--runs-dir", str(out_dir), "--train", *a.train, "--dev", a.dev,
                        "--heldout", *a.heldout, str(pmids), "--seed", str(a.seed), "--device", a.device, *rest])
        fold_rows = score_run(run, data, a.device, a.device.startswith("cuda"))[0]
        rows += fold_rows
        meta.append({"fold": k, "n_docs": len(fold), "n_pairs": len(fold_rows), "run": str(run), "reused": reuse})
        print(f"fold {k}: {len(fold)} documents, {len(fold_rows)} out-of-fold pairs{' (reused run)' if reuse else ''}")

    leaked = sorted({r["pmid"] for r in rows if pmid_held(r["pmid"], held)})
    if leaked:                                               # cannot happen: held documents are dropped above
        raise ValueError(f"teacher probabilities for held-out documents {leaked[:5]} (Dev / Test / BC8)")
    if {r["pmid"] for r in rows} != {d["pmid"] for d in docs}:
        raise ValueError("out-of-fold predictions do not cover exactly the fold documents")
    write_jsonl(out_dir / "oof_probs.jsonl", rows)
    summary = {"name": a.name, "mode": a.mode, "code_version": code_version(), "train": a.train, "dev": a.dev,
               "heldout": a.heldout, "folds": meta, "seed": a.seed, "n_folds": a.folds, "limit_docs": a.limit_docs,
               "device": a.device, "train_re_args": rest, "n_docs": len(docs), "n_pairs": len(rows),
               "file": str(out_dir / "oof_probs.jsonl")}
    (out_dir / "oof_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{len(rows)} out-of-fold pairs over {len(docs)} documents -> {out_dir / 'oof_probs.jsonl'}")
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
