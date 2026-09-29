"""Paper 2, E3 probe features: a fixed disease-gene pair representation, the relation head's input
(docs/research/paper2/codemap.md §1.2), captured by a forward hook on ``head_ext``; the model is not changed.

    PYTHONPATH=. .venv/bin/python docs/results/paper2/embed.py --run RUN --data FILE.json --out X.npz [--corpus cred]
    PYTHONPATH=. .venv/bin/python docs/results/paper2/embed.py --backbone BACKBONE --data FILE.json --out Y.npz
      writes OUT: X [n D-G pairs, 3*hidden] float16 = [e_disease; e_gene; c] (2304 for base; + the pair-matrix cell g
             with pair_matrix_layers), pmid / e1 (disease) / e2 (gene) [n], rows in document and pair order

--run: RUN/best (DocRE.load) under the run's corpus tag, with score.py's leak guard and featurization. --backbone: a
fresh build_model, the raw-LM control; the head input is encoder states and attention pooling only, so the untrained
head does not touch it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from biocausal.data import load_docs
from biocausal.model import DocRE, build_model, featurize, predict
from biocausal.score import check_unseen

D, G = "DiseaseOrPhenotypicFeature", "GeneOrGeneProduct"


def embed(model: DocRE, feats: list[dict], device: str = "cpu") -> tuple[np.ndarray, list[dict]]:
    """(X float16 [n D-G pairs, 3*hidden], their predict rows); canonical pair order puts the disease first."""
    cap: list[torch.Tensor] = []   # per forward with pairs: head_ext(hs) for all pairs of the chunk, then head_ext(ho)
    # ponytail: every candidate pair is held on CPU in fp16 (~6 KB per pair for base), filter per chunk if that bites
    hook = model.head_ext.register_forward_hook(lambda m, i, o: cap.append(i[0].to("cpu", torch.float16)))
    try:
        rows = predict(model, feats, device=device)
    finally:
        hook.remove()
    h = model.hidden
    hs, ho = torch.cat(cap[0::2]), torch.cat(cap[1::2])
    keep = torch.tensor([(r["t1"], r["t2"]) == (D, G) for r in rows], dtype=torch.bool)
    x = torch.cat([hs[:, :h], ho[:, :h], hs[:, h:]], -1)[keep]
    return x.numpy(), [r for r, k in zip(rows, keep.tolist()) if k]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="run dir with best/ and config.json")
    src.add_argument("--backbone", help="HF id or dir: fresh model, raw-LM control")
    ap.add_argument("--data", required=True, help="Doc-schema or BioC JSON")
    ap.add_argument("--out", required=True, help=".npz")
    ap.add_argument("--corpus", help="--run only: featurize under this corpus tag, e.g. cred")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42, help="--backbone only: seeds the fresh marker embeddings")
    a = ap.parse_args(argv)
    docs = load_docs(a.data)
    if a.run:
        config = json.loads((Path(a.run) / "config.json").read_text(encoding="utf-8"))
        if a.corpus is not None and not (config.get("corpus_tag") and a.corpus in config.get("corpora", [])):
            ap.error(f"{a.run} was not trained with --corpus-tag on corpus {a.corpus!r}")
        check_unseen(a.run, docs)   # score.py's guard: no document the run (or its --init-from parents) trained on
        model, tok = DocRE.load(Path(a.run) / "best", a.device)
        feats = [featurize(d, tok, a.corpus, corpus_tag=config.get("corpus_tag", False)) for d in docs]
    else:
        if a.corpus:
            ap.error("--corpus needs --run")
        torch.manual_seed(a.seed)
        model, tok = build_model(a.backbone)
        feats = [featurize(d, tok) for d in docs]
    X, rows = embed(model.to(a.device), feats, a.device)
    np.savez(a.out, X=X, **{k: np.array([r[k] for r in rows]) for k in ("pmid", "e1", "e2")})
    print(f"{a.out}: {X.shape[0]} disease-gene pairs x {X.shape[1]} from {len(docs)} documents")


if __name__ == "__main__":
    main(sys.argv[1:])
