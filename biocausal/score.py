"""Score trained RE checkpoints on a data file (test day, preregistration §4); never touches a run's metrics.json.

    python -m biocausal.score --run runs/<name> --data data/BC8/bc8_test.json [--tag bc8] [--out-dir DIR]
        -> DIR (default the run dir)/pred_<tag>.jsonl + metrics_<tag>.json
    python -m biocausal.score --ensemble runs/a runs/b runs/c --data FILE --tag TAG --out-dir DIR
        -> probabilities of the members averaged (ensemble.ensemble_rows), same two files in DIR
    python -m biocausal.score --crossfit PRED GOLD PRED GOLD --tag TAG --out-dir DIR
        -> metrics_<tag>.json with tp/fp/fn summed over the two (pred jsonl, gold data file) halves (documents are
           disjoint, so this is all_metrics on the concatenated rows) + pred_<tag>.jsonl with both halves; each PRED
           must hold exactly the candidate pairs of its GOLD

--run / --ensemble refuse a data file with documents a run trained on (its config.json: --train / --pretrain / --joint
minus held-out PMIDs, and the same for every run up its --init-from chain; check_unseen). Featurization, prediction
and metrics (unpairable gold relations counted as false negatives) are train_re's own code; aux-head runs add
``probs_<k>`` to every row (model.predict).
Tags ``dev`` and ``test`` are refused: those are train_re's files (and names clash case-insensitively on Windows).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import torch

from biocausal.data import Doc, load_docs, pmid_held, read_pmids, unpairable_gold
from biocausal.ensemble import ensemble_rows
from biocausal.evaluate import all_metrics, read_jsonl, write_jsonl
from biocausal.model import DocRE, featurize, predict
from biocausal.train_re import code_version, gold_unpairable, held_files


def check_unseen(run_dir: str | Path, docs: list[Doc]) -> None:
    """Refuse ``docs`` if ``run_dir`` or any run up its --init-from chain (config ``init_from`` = <parent>/best)
    trained on one of them: that run's --train / --pretrain / --joint minus its own held-out PMIDs (§6.3)."""
    pmids, run, base = {d["pmid"] for d in docs}, Path(run_dir), Path()
    while True:
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        held = set().union(*(read_pmids(base / p) for p in held_files(config)))
        trained = [p for f in config["train"] + (config.get("pretrain") or []) + (config.get("joint") or [])
                   for p in read_pmids(base / f) if not pmid_held(p, held) and pmid_held(p, pmids)]
        if trained:
            raise ValueError(f"{run} trained on {len(trained)} documents to score: {sorted(trained)[:5]}")
        if not config.get("init_from"):
            return
        run = Path(config["init_from"]).parent
        # a parent's relative paths are relative to its own cwd = <cwd>/<runs_dir>/<name> (bc8all: D:\model-second)
        rd = Path(json.loads((run / "config.json").read_text(encoding="utf-8")).get("runs_dir") or "")
        base = run.resolve().parents[len(rd.parts)] if rd.parts and not rd.is_absolute() else Path()


def score_run(run: str | Path, data: str | Path, device: str, bf16: bool,
              batch_docs: int | None = None, corpus: str | None = None) -> tuple[list[dict], list[str]]:
    """(prediction rows, unpairable gold types) of ``run``/best on ``data``; ``batch_docs`` defaults to the run's own
    eval_batch_docs. Refuses documents the run trained on (preregistration §6.3), read from its config.json.
    ``corpus``: score under that corpus's [CORPUS_x] tag (the run must be trained with --corpus-tag on that corpus)."""
    if not (Path(run) / "metrics.json").is_file():   # train_re saves best/ every improving epoch, metrics.json at the end
        raise ValueError(f"{run}: no metrics.json, training did not finish")
    config = json.loads((Path(run) / "config.json").read_text(encoding="utf-8"))
    if corpus is not None and not (config.get("corpus_tag") and corpus in config.get("corpora", [])):
        raise ValueError(f"{run} was not trained with --corpus-tag on corpus {corpus!r}")
    docs = load_docs(data)
    check_unseen(run, docs)
    model, tok = DocRE.load(Path(run) / "best", device)   # n_extra_none from re_config.json, corpus tags in tok
    feats = [featurize(d, tok, corpus, corpus_tag=config.get("corpus_tag", False)) for d in docs]
    batch_docs = batch_docs or config.get("eval_batch_docs", 8)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 and device.startswith("cuda") \
        else contextlib.nullcontext()
    return predict(model, feats, batch_docs, device, autocast), gold_unpairable(feats)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", help="run dir with best/")
    mode.add_argument("--ensemble", nargs="+", metavar="RUN", help="run dirs whose probabilities are averaged")
    mode.add_argument("--crossfit", nargs=4, metavar=("PRED_A", "GOLD_A", "PRED_B", "GOLD_B"))
    ap.add_argument("--data", help="BioC or Doc-schema JSON to score (--run / --ensemble)")
    ap.add_argument("--tag", help="output suffix (default: data file name up to the first dot, lower-cased)")
    ap.add_argument("--out-dir", help="default: the run dir (required for --ensemble / --crossfit)")
    ap.add_argument("--batch-docs", type=int, help="default: each run's eval_batch_docs (8 if absent)")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None, help="default: on if CUDA")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--corpus", help="score under this corpus tag, e.g. cred (runs trained with --corpus-tag on it)")
    a = ap.parse_args(argv)
    if not a.crossfit and not a.data:
        ap.error("--data is required with --run / --ensemble")
    if not (a.run or a.out_dir):
        ap.error("--out-dir is required with --ensemble / --crossfit")
    tag = a.tag or (Path(a.data).name.split(".")[0].lower() if a.data else None)
    if not tag or tag.lower() in ("dev", "test"):
        ap.error(f"--tag {tag!r}: give a tag other than dev/test (train_re's pred_dev/pred_test files)")
    out_dir = Path(a.out_dir or a.run)
    bf16 = a.device.startswith("cuda") if a.bf16 is None else a.bf16
    if a.crossfit and (out_dir / f"pred_{tag}.jsonl").resolve() in {Path(p).resolve() for p in a.crossfit}:
        ap.error("--crossfit output would overwrite one of its inputs: change --tag or --out-dir")

    info: dict = {"tag": tag, "code_version": code_version()}
    if a.crossfit:
        rows, unpairable = [], []
        for pred, gold in zip(a.crossfit[::2], a.crossfit[1::2]):
            half, gold_docs = read_jsonl(pred), load_docs(gold)
            want = {(d["pmid"], p["e1"], p["e2"]) for d in gold_docs for p in d["pairs"]}
            if len(half) != len(want) or {(r["pmid"], r["e1"], r["e2"]) for r in half} != want:
                raise ValueError(f"{pred} does not predict exactly the candidate pairs of {gold}")
            if {r["pmid"] for r in half} & {r["pmid"] for r in rows}:
                raise ValueError(f"{pred} shares documents with the other half")
            rows += half
            unpairable += [t for d in gold_docs for t in unpairable_gold(d)]
        info["crossfit"] = a.crossfit
    else:
        runs = [a.run] if a.run else a.ensemble
        scored = [score_run(r, a.data, a.device, bf16, a.batch_docs, a.corpus) for r in runs]
        rows = scored[0][0] if a.run else ensemble_rows([s[0] for s in scored])
        unpairable = scored[0][1]
        info.update({"runs": runs, "data": a.data, "device": a.device, "bf16": bf16, "corpus": a.corpus})
    metrics = {**info, **all_metrics(rows, unpairable)}
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / f"pred_{tag}.jsonl", rows)   # crossfit: both halves, for bootstrap.py
    (out_dir / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    m = metrics["micro"]
    print(f"{tag}: micro P/R/F1 {m['p']:.4f}/{m['r']:.4f}/{m['f1']:.4f}, macro {metrics['macro_f1']:.4f}, "
          f"causal F1 {metrics['causal']['causal']['f1']:.4f} -> {out_dir / f'metrics_{tag}.json'}")
    return metrics


if __name__ == "__main__":
    main(sys.argv[1:])
