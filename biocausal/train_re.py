"""Train / evaluate the document-level RE model (DESIGN.md §2, train_re.py).

One run = ``runs/<name>/`` with config.json, train.log, metrics.json, best/ (encoder+head state_dict
+ tokenizer), pred_dev.jsonl, pred_test.jsonl (per-class probabilities for offline ensembles).

* AdamW with separate encoder / head learning rates, weight decay, warmup + linear decay, grad clip
* ``--batch-docs`` x ``--grad-accum`` documents per optimizer step
* early stopping on dev micro-F1 (``--patience``); test evaluated exactly once with the best checkpoint
* ``--train`` takes several files; ``--pretrain`` runs a first stage on those files only (dev-selected),
  then a second stage on ``--train`` from the stage-1 best weights
* bf16 autocast on CUDA (``--no-bf16`` to disable); CPU works (tests)
* training/pretraining docs whose pmid is held out are dropped (PMID + 4-digit suffix ids too): BioRED Test unless
  ``--test`` is given (preregistration §1.2), plus the ``--heldout`` files, ``--test`` and ``--dev`` (not with
  ``--stop-epoch``, where Dev may be trained on); without ``--test`` nothing is evaluated on a test set
* ``--stop-epoch N``: LR schedule still spans ``--epochs``, training stops after epoch N, best/ = that checkpoint and dev
  is only logged (Train+Dev parity run)
* ``--loss atl``: adaptive-thresholding loss instead of cross-entropy (model.atl_loss)
* wave 3 (preregistration §7), all off by default: ``--pair-matrix-layers/-dim`` (model.PairMatrix), ``--ema DECAY``
  (averaged CPU copy of the weights, evaluated on dev every epoch, saved as best/), ``--distill FILE``
  (+ ``--distill-alpha/-temp``: teacher probabilities from biocausal.distill, model.kd_loss; teacher rows are refused for
  PMIDs held out by this run or by the teachers' dev / --heldout files (oof_summary.json next to FILE) and for PMIDs
  outside the first --train file; FILE's sha256 goes into config.json and metrics.json; not with --nv-weight or
  --none-weight, which kd_loss would drop) and ``--nv-weight``
* wave 2 (preregistration §5): ``--joint FILE...`` (+ ``--joint-docs-per-epoch N``), ``--label-handling``,
  ``--hard-map``, ``--corpus-tag``; external corpus = file stem, its docs flagged ``biored_train_overlap`` are never
  trained on (--joint and --pretrain); config.json records ``corpora``; the T / set tables are code (data.TRANSITIONS, covered by code_version)
* paper 2: ``--init-from DIR`` (runs/<x>/best: its tokenizer and every DocRE tensor of matching shape, new heads fresh;
  recorded in config.json, score.check_unseen follows it), ``--aux-heads pc ev`` + ``--aux-weight`` (model.DocRE.aux)
* ``code_version`` (CODE_VERSION.txt written by ops/sync.sh) goes into config.json and metrics.json
* ``--seed`` + deterministic kernels where torch has them: CPU runs are bit-reproducible, CUDA runs may
  still drift at float level (cross-entropy has no deterministic CUDA kernel)

Example (base): python -m biocausal.train_re --name re_biolinkbert_base_s42 \
    --backbone michiyasunaga/BioLinkBERT-base --train data/BioRED/Train.BioC.JSON \
    --dev data/BioRED/Dev.BioC.JSON --test data/BioRED/Test.BioC.JSON
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from biocausal.data import AUX_CLASSES, LABEL_HANDLING, LABELS, NONE_SLOTS, load_docs, pmid_held, read_pmids
from biocausal.evaluate import all_metrics, read_jsonl, write_jsonl
from biocausal.model import DocRE, add_corpus_tokens, build_model, collate, featurize, kd_loss, predict

log = logging.getLogger("biocausal.train_re")
ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--backbone", required=True, help="HF model id or local dir")
    ap.add_argument("--train", nargs="+", required=True, help="BioC or Doc-schema JSON files")
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test")
    ap.add_argument("--heldout", nargs="+", metavar="FILE",
                    help="more files whose PMIDs are dropped from every training file, never evaluated (added to "
                         "data/BioRED/Test.BioC.JSON, which is held unless --test is given or a --heldout file is "
                         "named Test.BioC.JSON; --test and, without "
                         "--stop-epoch, --dev are always held)")
    ap.add_argument("--stop-epoch", type=int, help="stop after this epoch, keep its checkpoint; LR schedule from --epochs")
    ap.add_argument("--loss", choices=["ce", "atl"], default="ce")
    ap.add_argument("--pretrain", nargs="*", default=[], help="stage-1 files (extra corpora)")
    ap.add_argument("--joint", nargs="+", default=[], metavar="FILE",
                    help="external corpora trained together with --train (corpus name = file stem)")
    ap.add_argument("--joint-docs-per-epoch", type=int,
                    help="each epoch: a fresh seeded uniform sample of this many --joint docs (default: all)")
    ap.add_argument("--label-handling", choices=LABEL_HANDLING,
                    help="labels of --joint / --pretrain corpora (biocausal.data.label_logw); --train docs stay one-hot")
    ap.add_argument("--hard-map", action="store_true", help="set/soft: no AMBIG sets, positives keep their label")
    ap.add_argument("--corpus-tag", action="store_true", help="[CORPUS_x] token before every document")
    ap.add_argument("--pair-matrix-layers", type=int, default=0, help="pair-interaction blocks (0 = module off)")
    ap.add_argument("--pair-matrix-dim", type=int, default=128, help="channels of the pair matrix (multiple of 8)")
    ap.add_argument("--ema", type=float, metavar="DECAY",
                    help="keep an EMA copy of the weights on CPU (e.g. 0.995); dev and best/ use it")
    ap.add_argument("--distill", metavar="FILE", help="teacher probabilities (biocausal.distill oof: oof_probs.jsonl)")
    ap.add_argument("--distill-alpha", type=float, default=0.5, help="weight of the KL term (0 = plain CE)")
    ap.add_argument("--distill-temp", type=float, default=2.0)
    ap.add_argument("--nv-weight", type=float, default=0.0, help="novelty head weight in the loss (0 = no head)")
    ap.add_argument("--aux-heads", nargs="+", default=[], choices=list(AUX_CLASSES), metavar="NAME",
                    help=f"aux pair heads {list(AUX_CLASSES)}: label = pair[NAME], pairs without it ignored")
    ap.add_argument("--aux-weight", type=float, default=1.0, help="weight of each aux head's loss")
    ap.add_argument("--init-from", metavar="DIR", help="runs/<x>/best: start from its tokenizer and every tensor "
                                                      "of matching shape (new heads fresh); --backbone = its backbone")
    ap.add_argument("--pretrain-epochs", type=int, default=10)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr-encoder", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--batch-docs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--eval-batch-docs", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--none-weight", type=float, default=1.0)
    ap.add_argument("--no-context-pool", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None, help="default: on if CUDA")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-docs", type=int, help="smoke tests: cap docs per split")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    if a.stop_epoch is not None and not 1 <= a.stop_epoch <= a.epochs:
        ap.error(f"--stop-epoch must be in 1..--epochs ({a.epochs})")
    if any(Path(f).stem.startswith(("cred", "clingen")) for f in a.train + a.pretrain) or \
            (any(Path(f).stem in ("cred", "clingen") for f in a.joint) and a.label_handling != "set"):
        ap.error("CRED labels only some pairs (ClinGen: aux labels only): pass it as --joint .../cred.json "
                 "(clingen.json) with --label-handling set")
    if a.label_handling and (a.loss != "ce" or a.none_weight != 1.0):
        ap.error("--label-handling needs --loss ce and --none-weight 1")
    if a.distill and (a.loss != "ce" or a.label_handling):
        ap.error("--distill needs --loss ce without --label-handling (kd_loss mixes cross-entropy with KL)")
    if a.distill and (a.nv_weight > 0 or a.none_weight != 1.0 or a.aux_heads):
        ap.error("--distill cannot be combined with --nv-weight > 0, --aux-heads or --none-weight != 1: kd_loss "
                 "replaces the model's loss with unweighted CE + alpha * KL, which would silently drop the novelty / "
                 "aux terms and the None weight")
    return a


def code_version() -> str:
    """CODE_VERSION.txt that ops/sync.sh writes into the shipped tree: the commit hash (sync refuses a dirty tree)."""
    p = ROOT / "CODE_VERSION.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else "unknown (no CODE_VERSION.txt)"


def held_files(config: dict) -> list[str]:
    """Files whose PMIDs never enter training, from a run's config (older configs have no heldout / stop_epoch)."""
    return (config.get("heldout") or []) + ([config["test"]] if config.get("test") else []) + \
        ([] if config.get("stop_epoch") else [config["dev"]])


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch and request deterministic kernels (warn-only: CE on CUDA has none)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # must be set before the first cuBLAS call
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def gold_unpairable(feats: list[dict]) -> list[str]:
    """Gold relation types with no candidate pair in these docs (fixed false negatives, see evaluate.py)."""
    return [t for f in feats for t in f["unpairable"]]


@torch.no_grad()
def ema_update(ema: dict[str, torch.Tensor], model: DocRE, decay: float) -> None:
    """In-place EMA of the whole state dict, kept on CPU (no VRAM): floats averaged, integer buffers copied."""
    for k, v in model.state_dict().items():
        v = v.detach().to("cpu", copy=False)
        if ema[k].is_floating_point():
            ema[k].mul_(decay).add_(v, alpha=1 - decay)
        else:
            ema[k].copy_(v)


@contextlib.contextmanager
def ema_weights(model: DocRE, ema: dict[str, torch.Tensor]):
    """Put the EMA weights in the model (dev evaluation, best/), then restore the raw ones (parked on CPU)."""
    raw = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    model.load_state_dict(ema)
    try:
        yield
    finally:
        model.load_state_dict(raw)


def train_stage(model: DocRE, tok: Any, train_feats: list[dict], dev_feats: list[dict], a: argparse.Namespace,
                epochs: int, best_dir: Path, autocast: Any, history: list[dict], stage: str,
                stop_epoch: int | None = None, joint_feats: list[dict] = ()) -> tuple[float, int]:
    """Train up to ``epochs`` with early stopping on dev micro-F1; best weights saved to ``best_dir``.

    ``stop_epoch``: the schedule still spans ``epochs``; stop after ``stop_epoch`` and save that checkpoint (dev logged only).
    ``joint_feats``: every epoch trains on all ``train_feats`` plus a fresh uniform sample of ``a.joint_docs_per_epoch``
    ``joint_feats`` (all by default), shuffled together; one seeded rng draws the sample and the order.
    ``a.ema``: an averaged copy of the weights (CPU) is updated after every optimizer step; dev is evaluated with both
    the raw and the EMA weights, and selection and best/ use the EMA ones. ``a.distill``: model.kd_loss instead of CE.
    """
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr_encoder}, {"params": head_params, "lr": a.lr_head}],
                            weight_decay=a.weight_decay,
                            foreach=False)   # foreach step temporaries add ~1.3 GB on BioLinkBERT-large (8 GB card)
    n_joint = min(a.joint_docs_per_epoch or len(joint_feats), len(joint_feats))
    n_batches = math.ceil((len(train_feats) + n_joint) / a.batch_docs)
    steps_per_epoch = math.ceil(n_batches / a.grad_accum)
    total = epochs * steps_per_epoch
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * total), total)
    rng = random.Random(a.seed)
    order = list(range(len(train_feats) + n_joint))
    ema = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()} if a.ema else None
    best_f1, best_epoch, bad = -1.0, 0, 0
    for epoch in range(1, (stop_epoch or epochs) + 1):
        model.train()
        pool = train_feats + rng.sample(joint_feats, n_joint) if n_joint else train_feats
        rng.shuffle(order)
        t0, losses = time.time(), []
        for bi in range(n_batches):
            chunk = [pool[i] for i in order[bi * a.batch_docs:(bi + 1) * a.batch_docs]]
            batch = collate(chunk, model.pad_id, a.device)
            teacher = batch.pop("teacher", None)
            with autocast:
                out = model(**batch)
                loss = out["loss"] if teacher is None or not out["logits"].numel() else \
                    kd_loss(out["logits"].float(), torch.cat(batch["labels"]), teacher, a.distill_alpha, a.distill_temp)
            (loss / a.grad_accum).backward()
            losses.append(loss.item())
            if (bi + 1) % a.grad_accum == 0 or bi + 1 == n_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema_update(ema, model, a.ema)
        dev = all_metrics(predict(model, dev_feats, a.eval_batch_docs, a.device, autocast), gold_unpairable(dev_feats))
        with ema_weights(model, ema) if ema else contextlib.nullcontext():   # --ema: dev, selection and best/ are EMA
            raw_f1 = dev["micro"]["f1"]
            if ema:
                dev = all_metrics(predict(model, dev_feats, a.eval_batch_docs, a.device, autocast),
                                  gold_unpairable(dev_feats))
            f1 = dev["micro"]["f1"]
            rec = {"stage": stage, "epoch": epoch, "loss": float(np.mean(losses)), "dev_f1": f1,
                   "dev_p": dev["micro"]["p"], "dev_r": dev["micro"]["r"], "dev_macro_f1": dev["macro_f1"],
                   "dev_causal_f1": dev["causal"]["causal"]["f1"], "lr": sched.get_last_lr()[0],
                   "epoch_sec": time.time() - t0}
            if ema:
                rec.update(dev_f1_raw=raw_f1, dev_f1_ema=f1)
            history.append(rec)
            log.info("%s epoch %d loss %.4f dev P/R/F1 %.4f/%.4f/%.4f macro %.4f causal %.4f%s (%.0fs)", stage, epoch,
                     rec["loss"], rec["dev_p"], rec["dev_r"], f1, rec["dev_macro_f1"], rec["dev_causal_f1"],
                     f" (raw {raw_f1:.4f})" if ema else "", rec["epoch_sec"])
            if stop_epoch:
                if epoch == stop_epoch:
                    best_f1, best_epoch = f1, epoch
                    model.save(best_dir, tok)
                    log.info("  stop epoch %d reached -> %s (dev not used for selection)", epoch, best_dir)
            elif f1 > best_f1:
                best_f1, best_epoch, bad = f1, epoch, 0
                model.save(best_dir, tok)
                log.info("  new best dev F1 %.4f -> %s", f1, best_dir)
            else:
                bad += 1
                if bad >= a.patience:
                    log.info("  early stopping (patience %d)", a.patience)
                    break
    return best_f1, best_epoch


def main(argv: list[str] | None = None) -> dict[str, Any]:
    a = parse_args(argv)
    if a.bf16 is None:
        a.bf16 = a.device.startswith("cuda")
    run_dir = Path(a.runs_dir) / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(run_dir / "train.log"), logging.StreamHandler(sys.stdout)],
                        force=True)
    # an explicit --heldout .../Test.BioC.JSON replaces the default: a code copy without data/ (D:\model-w2) runs on
    # absolute data paths, and ROOT/data would not exist there
    test = ROOT / "data" / "BioRED" / "Test.BioC.JSON"
    a.heldout = a.heldout or []
    if not a.test and all(Path(p).name.lower() != test.name.lower() for p in a.heldout):
        a.heldout = [str(test)] + a.heldout
    a.code_version = code_version()
    a.corpora = [Path(p).stem for p in a.joint + a.pretrain]
    if a.distill:
        a.distill_sha256 = hashlib.sha256(Path(a.distill).read_bytes()).hexdigest()
    json.dump(vars(a), open(run_dir / "config.json", "w"), indent=1)
    log.info("config %s", json.dumps(vars(a)))
    set_seed(a.seed)
    t_start = time.time()

    model, tok = build_model(a.backbone, context_pool=not a.no_context_pool, block=a.block, max_len=a.max_len,
                             stride=a.stride, none_weight=a.none_weight, loss=a.loss,
                             n_extra_none=len(NONE_SLOTS) if a.label_handling == "none_per_corpus" else 0,
                             pair_matrix_layers=a.pair_matrix_layers, pair_matrix_dim=a.pair_matrix_dim,
                             nv_weight=a.nv_weight, aux_heads=a.aux_heads, aux_weight=a.aux_weight)
    if a.init_from:   # DocRE weights of runs/<x>/best: tensors of matching shape load, new heads stay fresh
        tok = AutoTokenizer.from_pretrained(a.init_from)                 # markers and any [CORPUS_*] tokens
        model.encoder.resize_token_embeddings(len(tok), mean_resizing=False)
        sd, own = torch.load(Path(a.init_from) / "model.pt", map_location="cpu"), model.state_dict()
        skip = [k for k in sd if k not in own or sd[k].shape != own[k].shape]
        fresh = [k for k in own if k not in sd or k in skip]
        bad = [k for k in fresh if k.startswith("encoder.")]
        if bad:
            raise ValueError(f"--init-from {a.init_from}: encoder tensors {bad[:5]} do not fit --backbone {a.backbone}")
        # class_weight is --none-weight's buffer, not a learned tensor: this run's flag wins
        model.load_state_dict({k: v for k, v in sd.items() if k not in skip and k != "class_weight"}, strict=False)
        log.info("init from %s; not loaded %s; fresh %s", a.init_from, skip, fresh)
    if a.corpus_tag:
        add_corpus_tokens(tok, model.encoder, sorted(set(a.corpora)))
    if a.grad_ckpt:
        model.encoder.gradient_checkpointing_enable()
    model.to(a.device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if a.bf16 and a.device.startswith("cuda") \
        else contextlib.nullcontext()

    splits: dict[str, dict[str, int]] = {}

    def feats(paths: list[str], name: str, held: set[str] | None = None, external: bool = False) -> list[dict]:
        """Featurize a split; ``held`` (training only) drops pair-less docs and any held-out pmid and applies
        --label-handling; ``external`` (--joint / --pretrain): corpus = file stem, docs shared with BioRED Train dropped."""
        fs, n_shared = [], 0
        for p in paths:
            for d in load_docs(p, a.limit_docs):
                if external and d.get("biored_train_overlap"):
                    n_shared += 1
                    continue
                fs.append(featurize(d, tok, Path(p).stem if external else None,
                                    a.label_handling if held is not None else None, a.hard_map, a.corpus_tag))
        n_overlap = 0
        if held is not None:
            n_overlap = sum(pmid_held(f["pmid"], held) for f in fs)
            fs = [f for f in fs if f["pairs"] and not pmid_held(f["pmid"], held)]
        splits[name] = {"n_docs": len(fs), "n_pairs": sum(len(f["pairs"]) for f in fs),
                        "max_tokens": max((len(f["input_ids"]) for f in fs), default=0),
                        "n_docs_over_max_len": sum(len(f["input_ids"]) > a.max_len for f in fs),
                        "n_dropped_overlap": n_overlap, "n_gold_unpairable": len(gold_unpairable(fs))}
        log.info("%s: %d docs, %d pairs, max %d tokens, %d docs over max_len, %d dropped (pmid held out), "
                 "%d unpairable gold relations", name, *splits[name].values())
        if external:
            splits[name]["n_dropped_biored_train_overlap"] = n_shared
            log.info("  %d docs flagged biored_train_overlap dropped", n_shared)
        return fs

    dev_feats = feats([a.dev], "dev")
    test_feats = feats([a.test], "test") if a.test else []
    held_list = held_files(vars(a))
    held = set().union(*(read_pmids(p) for p in held_list))   # whole files: --limit-docs never shrinks the exclusion
    log.info("held-out pmids: %d from %s", len(held), held_list)
    train_feats = feats(a.train, "train", held)
    joint_feats = feats(a.joint, "joint", held, external=True) if a.joint else []
    distill: dict[str, Any] = {}
    if a.distill:
        teacher = {(r["pmid"], r["e1"], r["e2"]): [r["probs"][l] for l in LABELS] for r in read_jsonl(a.distill)}
        leaked = sorted({p for p, _, _ in teacher if pmid_held(p, held)})      # teachers may not have seen held-out docs
        if leaked:
            raise ValueError(f"--distill {a.distill}: teacher probabilities for {len(leaked)} held-out documents "
                             f"{leaked[:5]} (preregistration §7: out-of-fold on training documents only)")
        summary = Path(a.distill).parent / "oof_summary.json"                   # written by biocausal.distill
        if not summary.exists():
            raise ValueError(f"--distill {a.distill}: no {summary.name} next to it, the teachers' dev / held-out files "
                             "cannot be checked")
        oof = json.loads(summary.read_text(encoding="utf-8"))
        teacher_held = set().union(*(read_pmids(p) for p in [oof["dev"], *oof["heldout"]]))
        leaked = sorted({p for p, _, _ in teacher if pmid_held(p, teacher_held)})
        if leaked:
            raise ValueError(f"--distill {a.distill}: teacher probabilities for {len(leaked)} documents of the "
                             f"teachers' dev / held-out files {leaked[:5]} ({summary})")
        train_pmids = read_pmids(a.train[0])
        foreign = sorted({p for p, _, _ in teacher if p not in train_pmids})
        if foreign:
            raise ValueError(f"--distill {a.distill}: teacher probabilities for {len(foreign)} documents not in the "
                             f"first --train file {a.train[0]}: {foreign[:5]} (teachers = BioRED Train folds only)")
        for f in train_feats:
            f["teacher"] = [teacher.get(m[:3]) for m in f["meta"]]
        distill = {"file": a.distill, "sha256": a.distill_sha256, "alpha": a.distill_alpha, "temp": a.distill_temp,
                   "n_teacher_pairs": len(teacher),
                   "n_train_pairs_with_teacher": sum(t is not None for f in train_feats for t in f["teacher"])}
        log.info("distill: %d of %d training pairs have teacher probabilities (alpha %.2f, T %.1f)",
                 distill["n_train_pairs_with_teacher"], sum(len(f["pairs"]) for f in train_feats),
                 a.distill_alpha, a.distill_temp)
    best_dir = run_dir / "best"
    history: list[dict] = []
    stages = {}
    if a.pretrain:
        pre_feats = feats(a.pretrain, "pretrain", held, external=True)
        f1, ep = train_stage(model, tok, pre_feats, dev_feats, a, a.pretrain_epochs, best_dir, autocast, history, "pretrain")
        stages["pretrain"] = {"best_dev_f1": f1, "best_epoch": ep}
        if ep:   # stage 1 saved nothing with --pretrain-epochs 0
            model.load_state_dict(torch.load(best_dir / "model.pt", map_location=a.device))
    best_f1, best_epoch = train_stage(model, tok, train_feats, dev_feats, a, a.epochs, best_dir, autocast, history, "train",
                                      a.stop_epoch, joint_feats)
    stages["train"] = {"best_dev_f1": best_f1, "best_epoch": best_epoch}

    model.load_state_dict(torch.load(best_dir / "model.pt", map_location=a.device))
    dev_rows = predict(model, dev_feats, a.eval_batch_docs, a.device, autocast)
    write_jsonl(run_dir / "pred_dev.jsonl", dev_rows)
    metrics: dict[str, Any] = {
        "name": a.name, "backbone": a.backbone, "seed": a.seed, "code_version": a.code_version, "config": vars(a),
        "heldout": {"files": held_list, "n_pmids": len(held)}, "stages": stages,
        "best_epoch": best_epoch, "best_dev_f1": best_f1, "n_train_docs": len(train_feats),
        "n_train_pairs": sum(len(f["pairs"]) for f in train_feats), "splits": splits, "history": history,
        "dev": all_metrics(dev_rows, gold_unpairable(dev_feats)),
    }
    if distill:
        metrics["distill"] = distill
    if test_feats:
        test_rows = predict(model, test_feats, a.eval_batch_docs, a.device, autocast)
        write_jsonl(run_dir / "pred_test.jsonl", test_rows)
        metrics["test"] = all_metrics(test_rows, gold_unpairable(test_feats))
    metrics["elapsed_sec"] = time.time() - t_start
    metrics["gpu_peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20 if a.device.startswith("cuda") else 0.0
    metrics["device"] = a.device
    json.dump(metrics, open(run_dir / "metrics.json", "w"), indent=1)
    log.info("done in %.0fs: dev F1 %.4f (epoch %d)%s, peak GPU %.0f MB", metrics["elapsed_sec"], best_f1, best_epoch,
             f", test F1 {metrics['test']['micro']['f1']:.4f}" if test_feats else "", metrics["gpu_peak_mem_mb"])
    return metrics


if __name__ == "__main__":
    main(sys.argv[1:])
