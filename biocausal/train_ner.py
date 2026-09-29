"""BIO token-classification NER over the 6 BioRED entity types (DESIGN.md §2, train_ner.py).

* ``AutoModelForTokenClassification`` head, 13 labels (O + B/I x 6 types), first-sub-token labelling
  (other sub-tokens and special tokens get -100 and are ignored by the loss)
* one full tokenization per document; inputs longer than ``max_len`` are cut into overlapping windows of
  ``max_len - 2`` content tokens with ``stride`` (same formula as ``model.DocRE.encode``); windows are
  independent training examples, at prediction time per-token logits are averaged over the windows
  covering the token (``tokenizers`` overflow/stride is not used: it drops windows on this version)
* nested / overlapping gold mentions cannot be expressed in BIO: the longest-first non-overlapping subset
  is used for training, **all** gold mentions count at evaluation (dropped nested ones = fixed false
  negatives, reported as ``n_nested_dropped``), like ``evaluate.unpairable`` for RE
* entity-level span P/R/F1 (exact (start, end, type) match) overall and per type + macro-F1
* AdamW, warmup + linear decay, grad clip, early stopping on dev F1, test evaluated exactly once with
  the best checkpoint, which is saved to ``runs/<name>/model`` (HF ``save_pretrained``: model + tokenizer,
  label map in config.json) — the directory ``predictor.Extractor(ner_dir=...)`` loads
* bf16 autocast on CUDA (``--no-bf16`` to disable); CPU works (tests); ``--limit-docs`` / ``--epochs``
  for smoke runs

Example: python -m biocausal.train_ner --name ner_biolinkbert_base --backbone michiyasunaga/BioLinkBERT-base \
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
from transformers import (AutoModelForTokenClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase,
                          get_linear_schedule_with_warmup)

from biocausal.data import TYPE_ORDER, Doc, load_docs
from biocausal.evaluate import prf
from biocausal.train_re import set_seed

log = logging.getLogger("biocausal.train_ner")

NER_TYPES = [*TYPE_ORDER, "OrganismTaxon", "CellLine"]
NER_LABELS = ["O"] + [f"{p}-{t}" for t in NER_TYPES for p in ("B", "I")]
NER_LABEL2ID = {l: i for i, l in enumerate(NER_LABELS)}
IGNORE = -100
Span = tuple[int, int, str]   # (char start, char end, type)
Window = dict[str, list[int]]  # {"input_ids", "labels"}


# ---- text -> tokens / windows ----------------------------------------------------------------
def windows(n: int, max_len: int, stride: int) -> list[tuple[int, int]]:
    """Content-token ranges [s, e) covering n tokens with max_len-2 per window (CLS/SEP added later)."""
    w = max_len - 2
    if n <= w:
        return [(0, n)]
    return [(s, s + w) for s in list(range(0, n - w, stride)) + [n - w]]


def tokenize(text: str, tok: PreTrainedTokenizerBase) -> dict[str, list]:
    """Full encoding without special tokens: ids, char offsets, and a first-sub-token flag per token."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False, verbose=False)
    wids = enc.word_ids()
    return {"ids": enc["input_ids"], "offsets": enc["offset_mapping"],
            "first": [i == 0 or wids[i] != wids[i - 1] for i in range(len(wids))]}


def flat_mentions(mentions: list[dict]) -> list[dict]:
    """Longest-first non-overlapping subset (BIO cannot express nesting)."""
    out: list[dict] = []
    end = -1
    for m in sorted(mentions, key=lambda m: (m["start"], -m["end"])):
        if m["start"] >= end:
            out.append(m)
            end = m["end"]
    return out


def featurize(doc: Doc, tok: PreTrainedTokenizerBase, max_len: int = 512, stride: int = 128) -> list[Window]:
    """Training windows of a document: first sub-token of every word gets the BIO label of its first char."""
    text, t = doc["text"], tokenize(doc["text"], tok)
    tag = [0] * (len(text) + 1)
    for m in flat_mentions(doc["mentions"]):
        tag[m["start"]] = NER_LABEL2ID[f"B-{m['type']}"]
        tag[m["start"] + 1:m["end"]] = [NER_LABEL2ID[f"I-{m['type']}"]] * (m["end"] - m["start"] - 1)
    lab = [tag[off[0]] if first else IGNORE for first, off in zip(t["first"], t["offsets"])]
    return [{"input_ids": [tok.cls_token_id, *t["ids"][s:e], tok.sep_token_id], "labels": [IGNORE, *lab[s:e], IGNORE]}
            for s, e in windows(len(t["ids"]), max_len, stride)]


def collate(wins: list[Window], pad_id: int, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    L = max(len(w["input_ids"]) for w in wins)
    ids = torch.full((len(wins), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(wins), L), dtype=torch.long)
    labels = torch.full((len(wins), L), IGNORE, dtype=torch.long)
    for i, w in enumerate(wins):
        n = len(w["input_ids"])
        ids[i, :n] = torch.tensor(w["input_ids"])
        mask[i, :n] = 1
        if w["labels"]:   # prediction windows carry none
            labels[i, :n] = torch.tensor(w["labels"])
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device), "labels": labels.to(device)}


# ---- prediction / decoding / metric ----------------------------------------------------------
def decode(tags: list[str], first: list[bool], offsets: list[tuple[int, int]]) -> list[Span]:
    """BIO tags over tokens -> char spans; a word spans its sub-tokens; a stray I- opens a new span."""
    spans: list[list] = []
    cur: list | None = None
    for i, (tag, is_first) in enumerate(zip(tags, first)):
        if not is_first:
            if cur is not None:
                cur[1] = offsets[i][1]
            continue
        if tag == "O":
            cur = None
            continue
        bio, typ = tag.split("-", 1)
        if bio == "I" and cur is not None and cur[2] == typ:
            cur[1] = offsets[i][1]
        else:
            cur = [offsets[i][0], offsets[i][1], typ]
            spans.append(cur)
    return [tuple(s) for s in spans]   # type: ignore[misc]


@torch.no_grad()
def predict_spans(model: PreTrainedModel, tok: PreTrainedTokenizerBase, texts: list[str], max_len: int = 512,
                  stride: int = 128, device: torch.device | str | None = None, autocast: Any = None) -> list[list[Span]]:
    """Entity spans per text; per-token logits averaged over the windows that cover the token."""
    model.eval()
    device = next(model.parameters()).device if device is None else device
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    pad = tok.pad_token_id or 0
    out: list[list[Span]] = []
    for text in texts:
        t = tokenize(text, tok)
        n = len(t["ids"])
        if not n:
            out.append([])
            continue
        wins = windows(n, max_len, stride)
        batch = collate([{"input_ids": [tok.cls_token_id, *t["ids"][s:e], tok.sep_token_id], "labels": []}
                         for s, e in wins], pad, device)
        batch.pop("labels")
        with (autocast or contextlib.nullcontext()):
            logits = model(**batch).logits.float()
        acc = logits.new_zeros((n, len(labels)))
        cnt = logits.new_zeros(n)
        for k, (s, e) in enumerate(wins):
            acc[s:e] += logits[k, 1:1 + e - s]
            cnt[s:e] += 1
        tags = [labels[i] for i in (acc / cnt[:, None]).argmax(-1).tolist()]
        out.append(decode(tags, t["first"], t["offsets"]))
    return out


def span_metrics(gold: list[set[Span]], pred: list[set[Span]]) -> dict[str, Any]:
    """Exact-match entity P/R/F1 overall, per type and macro-F1 over NER_TYPES."""
    def counts(sel: str | None) -> tuple[int, int, int]:
        tp = fp = fn = 0
        for g, p in zip(gold, pred):
            g = {s for s in g if sel is None or s[2] == sel}
            p = {s for s in p if sel is None or s[2] == sel}
            tp, fp, fn = tp + len(g & p), fp + len(p - g), fn + len(g - p)
        return tp, fp, fn

    per = {}
    for t in NER_TYPES:
        tp, fp, fn = counts(t)
        per[t] = {**prf(tp, fp, fn), "support": tp + fn}
    return {**prf(*counts(None)), "per_type": per, "macro_f1": sum(v["f1"] for v in per.values()) / len(per)}


def gold_spans(doc: Doc) -> set[Span]:
    return {(m["start"], m["end"], m["type"]) for m in doc["mentions"]}


def evaluate(model: PreTrainedModel, tok: PreTrainedTokenizerBase, docs: list[Doc], a: argparse.Namespace,
             autocast: Any) -> dict[str, Any]:
    pred = predict_spans(model, tok, [d["text"] for d in docs], a.max_len, a.stride, a.device, autocast)
    return span_metrics([gold_spans(d) for d in docs], [set(p) for p in pred])


# ---- training ----------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--backbone", required=True, help="HF model id or local dir")
    ap.add_argument("--train", nargs="+", required=True, help="BioC or Doc-schema JSON files")
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8, help="windows per optimizer step")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None, help="default: on if CUDA")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-docs", type=int, help="smoke tests: cap docs per split")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    a = parse_args(argv)
    if a.bf16 is None:
        a.bf16 = a.device.startswith("cuda")
    run_dir = Path(a.runs_dir) / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(run_dir / "train.log"), logging.StreamHandler(sys.stdout)],
                        force=True)
    json.dump(vars(a), open(run_dir / "config.json", "w"), indent=1)
    log.info("config %s", json.dumps(vars(a)))
    set_seed(a.seed)
    t_start = time.time()

    tok = AutoTokenizer.from_pretrained(a.backbone)
    model = AutoModelForTokenClassification.from_pretrained(
        a.backbone, num_labels=len(NER_LABELS), id2label=dict(enumerate(NER_LABELS)), label2id=NER_LABEL2ID)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
    model.to(a.device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if a.bf16 and a.device.startswith("cuda") \
        else contextlib.nullcontext()

    splits: dict[str, dict[str, int]] = {}

    def load(paths: list[str], name: str, held: set[str] | None = None) -> list[Doc]:
        docs = load_docs(paths, a.limit_docs)
        n_overlap = 0
        if held is not None:   # harmonised extra corpora may overlap BioRED dev/test
            n_overlap = sum(d["pmid"] in held for d in docs)
            docs = [d for d in docs if d["pmid"] not in held]
        n_men = sum(len(d["mentions"]) for d in docs)
        toks = [len(tokenize(d["text"], tok)["ids"]) for d in docs]
        splits[name] = {"n_docs": len(docs), "n_mentions": n_men,
                        "n_nested_dropped": n_men - sum(len(flat_mentions(d["mentions"])) for d in docs),
                        "n_windows": sum(len(windows(n, a.max_len, a.stride)) for n in toks),
                        "max_tokens": max(toks, default=0), "n_dropped_overlap": n_overlap}
        log.info("%s: %d docs, %d mentions (%d nested dropped for training), %d windows, max %d tokens, "
                 "%d dropped (pmid in dev/test)", name, *splits[name].values())
        return docs

    dev_docs = load([a.dev], "dev")
    test_docs = load([a.test], "test") if a.test else []
    train_docs = load(a.train, "train", {d["pmid"] for d in dev_docs + test_docs})
    train_wins = [w for d in train_docs for w in featurize(d, tok, a.max_len, a.stride)]

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    n_batches = math.ceil(len(train_wins) / a.batch)
    total = a.epochs * n_batches
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * total), total)
    rng = random.Random(a.seed)
    order = list(range(len(train_wins)))
    model_dir = run_dir / "model"
    history: list[dict] = []
    best_f1, best_epoch, bad = -1.0, 0, 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        rng.shuffle(order)
        t0, losses = time.time(), []
        for bi in range(n_batches):
            batch = collate([train_wins[i] for i in order[bi * a.batch:(bi + 1) * a.batch]], tok.pad_token_id or 0, a.device)
            with autocast:
                loss = model(**batch).loss
            loss.backward()
            losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        dev = evaluate(model, tok, dev_docs, a, autocast)
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "dev_f1": dev["f1"], "dev_p": dev["p"], "dev_r": dev["r"],
               "dev_macro_f1": dev["macro_f1"], "lr": sched.get_last_lr()[0], "epoch_sec": time.time() - t0}
        history.append(rec)
        log.info("epoch %d loss %.4f dev P/R/F1 %.4f/%.4f/%.4f macro %.4f (%.0fs)", epoch, rec["loss"], dev["p"],
                 dev["r"], dev["f1"], dev["macro_f1"], rec["epoch_sec"])
        if dev["f1"] > best_f1:
            best_f1, best_epoch, bad = dev["f1"], epoch, 0
            model.save_pretrained(model_dir)
            tok.save_pretrained(model_dir)
            log.info("  new best dev F1 %.4f -> %s", dev["f1"], model_dir)
        else:
            bad += 1
            if bad >= a.patience:
                log.info("  early stopping (patience %d)", a.patience)
                break

    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(a.device)
    metrics: dict[str, Any] = {
        "name": a.name, "backbone": a.backbone, "seed": a.seed, "task": "ner", "config": vars(a), "labels": NER_LABELS,
        "best_epoch": best_epoch, "best_dev_f1": best_f1, "n_train_docs": len(train_docs), "n_train_windows": len(train_wins),
        "splits": splits, "history": history, "dev": evaluate(model, tok, dev_docs, a, autocast),
    }
    if test_docs:
        metrics["test"] = evaluate(model, tok, test_docs, a, autocast)
    metrics["elapsed_sec"] = time.time() - t_start
    metrics["gpu_peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20 if a.device.startswith("cuda") else 0.0
    metrics["device"] = a.device
    json.dump(metrics, open(run_dir / "metrics.json", "w"), indent=1)
    log.info("done in %.0fs: dev F1 %.4f (epoch %d)%s, peak GPU %.0f MB", metrics["elapsed_sec"], best_f1, best_epoch,
             f", test F1 {metrics['test']['f1']:.4f}" if test_docs else "", metrics["gpu_peak_mem_mb"])
    return metrics


if __name__ == "__main__":
    main(sys.argv[1:])
