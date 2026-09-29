"""Pair-level R-BERT baseline (Wu & He 2019) under the honest protocol (DESIGN.md §3, ``pair_rbert_*``).

The earlier prototype re-encoded the abstract once per candidate pair. This reproduces that method on the
same splits, label set (9 classes incl. ``None``), candidate pairs and metrics as train_re.py so the two
rows of the article are comparable:

* per pair: the document with typed markers around the mentions of the two entities of that pair only
  (every other entity unmarked), head-truncated to ``--max-len``; an entity whose start markers were all
  cut falls back to [CLS] (as the old code did) and is counted (``splits.<split>.n_cls_fallback``)
* representation = [tanh(W_c CLS); tanh(W_1 mean E1 start markers); tanh(W_2 mean E2 start markers)]
  -> dropout 0.3 -> linear -> 9 classes
* loss: cross-entropy; ``--focal-gamma 2.0 --class-weights balanced`` reproduces the old paper's focal loss
  with sklearn-style balanced weights (same formula as the old code, incl. p_t taken from the weighted CE)
* AdamW (single lr), warmup + linear decay, grad clip, ``--batch-pairs`` x ``--grad-accum`` pairs per step,
  early stopping on dev micro-F1, test evaluated exactly once with the best checkpoint
* pre-tokenised once, dynamic padding, bf16 autocast on CUDA; pairs/s logged every ``--log-every`` batches
* artifacts as train_re: config.json (``method: "pair"``), train.log, metrics.json, best/ (weights + tokenizer),
  pred_dev.jsonl, pred_test.jsonl -> server/runs.py, ensemble.py and the report pick them up unchanged

Example: python -m biocausal.train_pair --name pair_rbert_biobert_s42 --backbone dmis-lab/biobert-v1.1 \
    --train data/BioRED/Train.BioC.JSON --dev data/BioRED/Dev.BioC.JSON --test data/BioRED/Test.BioC.JSON
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import (AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase,
                          get_linear_schedule_with_warmup)

from biocausal.data import LABEL2ID, LABELS, MARKERS, TYPE_ORDER, Doc, load_docs, mark_text, unpairable_gold
from biocausal.evaluate import all_metrics, write_jsonl
from biocausal.model import add_markers
from biocausal.train_re import set_seed

log = logging.getLogger("biocausal.train_pair")
Feature = dict[str, Any]


def pair_text(doc: Doc, e1: str, e2: str) -> tuple[str, dict[int, int]]:
    """Document text with typed markers around the mentions of ``e1`` and ``e2`` only (others unmarked)."""
    return mark_text({**doc, "entities": {e: doc["entities"][e] for e in (e1, e2)}})


def featurize(doc: Doc, tok: PreTrainedTokenizerBase, max_len: int = 512) -> list[Feature]:
    """One feature per candidate pair: truncated ids + start-marker token positions of each entity.

    An entity whose start markers all fell beyond ``max_len`` gets position [0] (= [CLS], the old code's
    fallback); ``n_fallback`` counts such entities (0-2) and ``at_max_len`` flags a pair cut to ``max_len``.
    """
    if not doc["pairs"]:
        return []
    ents = doc["entities"]
    texts, mposs = zip(*(pair_text(doc, p["e1"], p["e2"]) for p in doc["pairs"]))
    enc = tok(list(texts), truncation=True, max_length=max_len, return_offsets_mapping=True, verbose=False)
    open_ids = set(tok.convert_tokens_to_ids([MARKERS[t][0] for t in TYPE_ORDER]))
    out = []
    for p, mpos, ids, offs in zip(doc["pairs"], mposs, enc["input_ids"], enc["offset_mapping"]):
        start2tok = {off[0]: i for i, (tid, off) in enumerate(zip(ids, offs)) if tid in open_ids}
        f: Feature = {"pmid": doc["pmid"], "e1": p["e1"], "e2": p["e2"], "t1": ents[p["e1"]]["type"],
                      "t2": ents[p["e2"]]["type"], "gold": p["label"], "label": LABEL2ID[p["label"]],
                      "input_ids": torch.tensor(ids), "n_fallback": 0, "at_max_len": len(ids) >= max_len}
        for k in ("e1", "e2"):
            pos = [start2tok[mpos[mi]] for mi in ents[p[k]]["mentions"] if mpos[mi] in start2tok]
            f["n_fallback"] += not pos
            f[k + "_pos"] = pos or [0]
        out.append(f)
    return out


def collate(feats: list[Feature], pad_id: int, device: torch.device | str = "cpu") -> dict[str, Tensor]:
    B, L = len(feats), max(len(f["input_ids"]) for f in feats)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    mask = torch.zeros((B, L), dtype=torch.long)
    for b, f in enumerate(feats):
        n = len(f["input_ids"])
        ids[b, :n], mask[b, :n] = f["input_ids"], 1
    batch = {"input_ids": ids, "attention_mask": mask, "labels": torch.tensor([f["label"] for f in feats])}
    for k in ("e1", "e2"):
        M = max(len(f[k + "_pos"]) for f in feats)
        pos = torch.zeros((B, M), dtype=torch.long)
        pm = torch.zeros((B, M), dtype=torch.bool)
        for b, f in enumerate(feats):
            n = len(f[k + "_pos"])
            pos[b, :n], pm[b, :n] = torch.tensor(f[k + "_pos"]), True
        batch[k + "_pos"], batch[k + "_mask"] = pos, pm
    return {k: v.to(device) for k, v in batch.items()}


def focal_loss(logits: Tensor, y: Tensor, weight: Tensor | None = None, gamma: float = 0.0) -> Tensor:
    """mean(weight[y] * (1 - p_t)^gamma * CE); gamma = 0 -> per-sample-weighted cross-entropy (old code's formula)."""
    ce = F.cross_entropy(logits, y, weight=weight, reduction="none")
    if gamma:
        ce = (1 - torch.exp(-ce)) ** gamma * ce
    return ce.mean()


def balanced_weights(labels: list[int], n_classes: int = len(LABELS)) -> Tensor:
    """sklearn ``class_weight='balanced'``: n / (n_present_classes * count); absent classes get count 1."""
    c = torch.bincount(torch.tensor(labels), minlength=n_classes)
    return len(labels) / ((c > 0).sum() * c.clamp(min=1).float())


class PairRE(nn.Module):
    """R-BERT: [CLS] + mean E1 start markers + mean E2 start markers, each through Linear+tanh, concat -> classes."""

    def __init__(self, encoder: PreTrainedModel, num_labels: int = len(LABELS), dropout: float = 0.3,
                 focal_gamma: float = 0.0, class_weight: Tensor | None = None, pad_id: int = 0) -> None:
        super().__init__()
        h = encoder.config.hidden_size
        self.encoder = encoder
        self.fc_cls, self.fc_e1, self.fc_e2 = nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, h)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(3 * h, num_labels)
        self.focal_gamma, self.pad_id = focal_gamma, pad_id
        self.register_buffer("class_weight", torch.ones(num_labels) if class_weight is None else class_weight.float())

    def forward(self, input_ids: Tensor, attention_mask: Tensor, e1_pos: Tensor, e1_mask: Tensor, e2_pos: Tensor,
                e2_mask: Tensor, labels: Tensor | None = None) -> dict[str, Tensor]:
        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state   # [B,L,h]
        rows = torch.arange(H.size(0), device=H.device)[:, None]

        def mean_at(pos: Tensor, m: Tensor) -> Tensor:                                            # [B,M] -> [B,h]
            return (H[rows, pos] * m[..., None]).sum(1) / m.sum(1, keepdim=True)

        z = torch.cat([torch.tanh(self.fc_cls(H[:, 0])), torch.tanh(self.fc_e1(mean_at(e1_pos, e1_mask))),
                       torch.tanh(self.fc_e2(mean_at(e2_pos, e2_mask)))], -1)
        out = {"logits": self.classifier(self.drop(z))}
        if labels is not None:
            out["loss"] = focal_loss(out["logits"].float(), labels, self.class_weight, self.focal_gamma)
        return out

    def save(self, out_dir: str | Path, tok: PreTrainedTokenizerBase) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(out_dir)
        self.encoder.config.save_pretrained(out_dir)
        json.dump({"method": "pair", "dropout": self.drop.p, "focal_gamma": self.focal_gamma, "pad_id": self.pad_id,
                   "class_weight": self.class_weight.tolist()}, open(out_dir / "pair_config.json", "w"), indent=1)
        torch.save(self.state_dict(), out_dir / "model.pt")


@torch.no_grad()
def predict(model: PairRE, feats: list[Feature], batch_pairs: int = 64, device: torch.device | str = "cpu",
            autocast: Any = None) -> list[dict]:
    """Rows {pmid,e1,e2,t1,t2,gold,pred,probs{label:p}} in input order (pred_*.jsonl / evaluate.py format)."""
    model.eval()
    rows: list[dict] = []
    for i in range(0, len(feats), batch_pairs):
        chunk = feats[i:i + batch_pairs]
        batch = collate(chunk, model.pad_id, device)
        batch.pop("labels")
        with (autocast or contextlib.nullcontext()):
            logits = model(**batch)["logits"]
        for f, p in zip(chunk, torch.softmax(logits.float(), -1).cpu().tolist()):
            rows.append({**{k: f[k] for k in ("pmid", "e1", "e2", "t1", "t2", "gold")},
                         "pred": LABELS[max(range(len(p)), key=p.__getitem__)], "probs": dict(zip(LABELS, p))})
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--backbone", required=True, help="HF model id or local dir")
    ap.add_argument("--train", nargs="+", required=True, help="BioC or Doc-schema JSON files")
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--batch-pairs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--eval-batch-pairs", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--focal-gamma", type=float, default=0.0, help="0 = cross-entropy; the old paper used 2.0")
    ap.add_argument("--class-weights", choices=["none", "balanced"], default="none")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None, help="default: on if CUDA")
    ap.add_argument("--log-every", type=int, default=200, help="batches between throughput/ETA log lines (0 = off)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-docs", type=int, help="smoke tests: cap docs per split")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args(argv)


def train(model: PairRE, tok: PreTrainedTokenizerBase, train_feats: list[Feature], dev_feats: list[Feature],
          dev_unpairable: list[str], a: argparse.Namespace, best_dir: Path, autocast: Any,
          history: list[dict]) -> tuple[float, int]:
    """Train up to ``a.epochs`` with early stopping on dev micro-F1; best weights saved to ``best_dir``."""
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, foreach=False)
    n_batches = math.ceil(len(train_feats) / a.batch_pairs)
    total = a.epochs * math.ceil(n_batches / a.grad_accum)
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * total), total)
    rng = random.Random(a.seed)
    order = list(range(len(train_feats)))
    best_f1, best_epoch, bad = -1.0, 0, 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        rng.shuffle(order)
        t0, losses = time.time(), []
        for bi in range(n_batches):
            chunk = [train_feats[i] for i in order[bi * a.batch_pairs:(bi + 1) * a.batch_pairs]]
            with autocast:
                loss = model(**collate(chunk, model.pad_id, a.device))["loss"]
            (loss / a.grad_accum).backward()
            losses.append(loss.item())
            if (bi + 1) % a.grad_accum == 0 or bi + 1 == n_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if a.log_every and (bi + 1) % a.log_every == 0:
                rate = (bi + 1) * a.batch_pairs / (time.time() - t0)
                log.info("epoch %d batch %d/%d loss %.4f %.1f pairs/s, epoch ETA %.0f s", epoch, bi + 1, n_batches,
                         float(np.mean(losses[-a.log_every:])), rate, (n_batches - bi - 1) * a.batch_pairs / rate)
        t_train = time.time() - t0
        t1 = time.time()
        dev = all_metrics(predict(model, dev_feats, a.eval_batch_pairs, a.device, autocast), dev_unpairable)
        f1 = dev["micro"]["f1"]
        rec = {"stage": "train", "epoch": epoch, "loss": float(np.mean(losses)), "dev_f1": f1,
               "dev_p": dev["micro"]["p"], "dev_r": dev["micro"]["r"], "dev_macro_f1": dev["macro_f1"],
               "dev_causal_f1": dev["causal"]["causal"]["f1"], "lr": sched.get_last_lr()[0],
               "epoch_sec": time.time() - t0, "train_pairs_per_sec": len(train_feats) / max(t_train, 1e-9),
               "eval_pairs_per_sec": len(dev_feats) / max(time.time() - t1, 1e-9)}
        history.append(rec)
        # "(Ns)" must close right after the seconds: server/runs.py parses this line for the live dashboard
        log.info("epoch %d loss %.4f dev P/R/F1 %.4f/%.4f/%.4f macro %.4f causal %.4f (%.0fs) train %.1f / eval %.1f pairs/s",
                 epoch, rec["loss"], rec["dev_p"], rec["dev_r"], f1, rec["dev_macro_f1"], rec["dev_causal_f1"],
                 rec["epoch_sec"], rec["train_pairs_per_sec"], rec["eval_pairs_per_sec"])
        if f1 > best_f1:
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
    cfg = {**vars(a), "method": "pair"}
    json.dump(cfg, open(run_dir / "config.json", "w"), indent=1)
    log.info("config %s", json.dumps(cfg))
    set_seed(a.seed)
    t_start = time.time()

    tok = AutoTokenizer.from_pretrained(a.backbone)
    enc = AutoModel.from_pretrained(a.backbone)
    add_markers(tok, enc)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if a.bf16 and a.device.startswith("cuda") \
        else contextlib.nullcontext()

    splits: dict[str, dict[str, int]] = {}

    def feats(paths: list[str], name: str, held: set[str] | None = None) -> tuple[list[Feature], list[str]]:
        """Featurize a split; ``held`` (training only) drops pair-less docs and any pmid found in dev/test."""
        docs = load_docs(paths, a.limit_docs)
        n_overlap = 0
        if held is not None:
            n_overlap = sum(d["pmid"] in held for d in docs)
            docs = [d for d in docs if d["pairs"] and d["pmid"] not in held]
        t0 = time.time()
        fs = [f for d in docs for f in featurize(d, tok, a.max_len)]
        unpairable = [t for d in docs for t in unpairable_gold(d)]
        splits[name] = {"n_docs": len(docs), "n_pairs": len(fs), "n_pairs_at_max_len": sum(f["at_max_len"] for f in fs),
                        "n_cls_fallback": sum(f["n_fallback"] for f in fs), "n_dropped_overlap": n_overlap,
                        "n_gold_unpairable": len(unpairable)}
        log.info("%s: %d docs, %d pairs, %d pairs cut to max_len, %d entity->CLS fallbacks, %d dropped (pmid in "
                 "dev/test), %d unpairable gold relations (tokenized in %.1fs)", name, *splits[name].values(),
                 time.time() - t0)
        return fs, unpairable

    dev_feats, dev_unp = feats([a.dev], "dev")
    test_feats, test_unp = feats([a.test], "test") if a.test else ([], [])
    held = {f["pmid"] for f in dev_feats + test_feats}
    train_feats, _ = feats(a.train, "train", held)
    cw = balanced_weights([f["label"] for f in train_feats]) if a.class_weights == "balanced" else None
    if cw is not None:
        log.info("balanced class weights %s", dict(zip(LABELS, [round(x, 3) for x in cw.tolist()])))
    model = PairRE(enc, dropout=a.dropout, focal_gamma=a.focal_gamma, class_weight=cw, pad_id=tok.pad_token_id or 0)
    if a.grad_ckpt:
        model.encoder.gradient_checkpointing_enable()
    model.to(a.device)
    best_dir = run_dir / "best"
    history: list[dict] = []
    best_f1, best_epoch = train(model, tok, train_feats, dev_feats, dev_unp, a, best_dir, autocast, history)

    model.load_state_dict(torch.load(best_dir / "model.pt", map_location=a.device))
    dev_rows = predict(model, dev_feats, a.eval_batch_pairs, a.device, autocast)
    write_jsonl(run_dir / "pred_dev.jsonl", dev_rows)
    metrics: dict[str, Any] = {
        "name": a.name, "backbone": a.backbone, "seed": a.seed, "method": "pair", "config": cfg,
        "best_epoch": best_epoch, "best_dev_f1": best_f1, "n_train_docs": splits["train"]["n_docs"],
        "n_train_pairs": len(train_feats), "class_weight": cw.tolist() if cw is not None else None,
        "splits": splits, "history": history, "dev": all_metrics(dev_rows, dev_unp),
    }
    if test_feats:
        test_rows = predict(model, test_feats, a.eval_batch_pairs, a.device, autocast)
        write_jsonl(run_dir / "pred_test.jsonl", test_rows)
        metrics["test"] = all_metrics(test_rows, test_unp)
    metrics["elapsed_sec"] = time.time() - t_start
    metrics["train_pairs_per_sec"] = float(np.mean([h["train_pairs_per_sec"] for h in history]))
    metrics["eval_pairs_per_sec"] = float(np.mean([h["eval_pairs_per_sec"] for h in history]))
    metrics["gpu_peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20 if a.device.startswith("cuda") else 0.0
    metrics["device"] = a.device
    json.dump(metrics, open(run_dir / "metrics.json", "w"), indent=1)
    log.info("done in %.0fs: dev F1 %.4f (epoch %d)%s, %.1f train pairs/s, peak GPU %.0f MB", metrics["elapsed_sec"],
             best_f1, best_epoch, f", test F1 {metrics['test']['micro']['f1']:.4f}" if test_feats else "",
             metrics["train_pairs_per_sec"], metrics["gpu_peak_mem_mb"])
    return metrics


if __name__ == "__main__":
    main(sys.argv[1:])
