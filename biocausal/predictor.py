"""End-to-end extractor behind ``POST /api/extract`` (DESIGN.md §2 predictor.py, §5 ExtractResult).

``Extractor(re_dir, ner_dir=None, device=None)`` loads the RE checkpoint written by ``train_re``
(``runs/<name>/best``) and, optionally, the NER model written by ``train_ner`` (``runs/<name>/model``).
``extract(title, abstract, entities=None)`` returns the ``ExtractResult`` dict of §5:

    {doc_id, title, text, entities: [{id, type, name, mentions: [{start, end, text}]}],
     relations: [{id, e1, e2, type, score, probs: {label: p}, causal, evidence}], meta: {model, ner, elapsed_ms}}

* ``text = title + " " + abstract`` (or just the abstract when the title is empty) — the same rule as the
  server, so mention offsets line up; entities given by the caller are used as-is, otherwise the NER model
  tags the text and mentions are grouped into entities by (type, casefolded mention text)
* candidate pairs, typed markers, windowing and scoring are ``data.py`` / ``model.py`` unchanged: the entity
  list becomes a Doc, then ``build_entities`` -> ``make_pairs`` -> ``featurize`` -> ``predict``
* a relation is emitted for every candidate pair whose argmax label is not ``None``; ``causal`` is the
  dissertation view of §1 ({Gene, Variant} x Disease with a correlation label); ``evidence`` is the first
  sentence mentioning both entities (else the sentence of e1's first mention)
* ``doc_id`` is left empty: the server assigns one

CLI (also the manual self-check on the serving box):
    python -m biocausal.predictor --re-dir runs/<name>/best [--ner-dir runs/ner_x/model] --title T --text ...
Automated checks: tests/test_predictor_*.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from biocausal.data import build_entities, make_pairs
from biocausal.evaluate import CAUSAL_TYPES, is_causal_pair
from biocausal.model import DocRE, featurize, predict
from biocausal.train_ner import NER_TYPES, Span, predict_spans

MAX_PAIRS = 5000   # ponytail: bounds the bilinear head's memory for absurd entity lists; real abstracts have ~70 pairs
_PREFIX = {"GeneOrGeneProduct": "G", "DiseaseOrPhenotypicFeature": "D", "ChemicalEntity": "C", "SequenceVariant": "V",
           "OrganismTaxon": "O", "CellLine": "L"}
_SENT = re.compile(r"[^.!?]+(?:[.!?]+|$)")   # same splitter as the server's FakeExtractor (biocausal must not import server)


def full_text(title: str, abstract: str) -> str:
    return f"{title} {abstract}" if title else abstract


def group_mentions(text: str, spans: list[Span]) -> list[dict]:
    """NER spans -> entities grouped by (type, casefolded text), ids G1, D1, ... in order of first appearance."""
    groups: dict[tuple[str, str], dict] = {}
    for start, end, typ in sorted(spans):
        key = (typ, text[start:end].casefold())
        if key not in groups:
            groups[key] = {"id": f"{_PREFIX[typ]}{sum(k[0] == typ for k in groups) + 1}", "type": typ,
                           "name": text[start:end], "mentions": []}
        groups[key]["mentions"].append({"start": start, "end": end, "text": text[start:end]})
    return list(groups.values())


def check_entity(e: dict, text: str) -> dict:
    """Validate one caller-supplied entity (trust boundary) and fill ``name`` / mention ``text``."""
    if not str(e.get("id", "")):
        raise ValueError("entity without id")
    if e.get("type") not in NER_TYPES:
        raise ValueError(f"entity {e.get('id')!r}: unknown type {e.get('type')!r}; expected one of {NER_TYPES}")
    mentions = []
    for m in e.get("mentions") or []:
        start, end = int(m["start"]), int(m["end"])
        if not 0 <= start < end <= len(text):
            raise ValueError(f"entity {e['id']!r}: mention [{start}, {end}) outside the text (length {len(text)})")
        mentions.append({"start": start, "end": end, "text": m.get("text") or text[start:end]})
    return {"id": str(e["id"]), "type": e["type"], "name": e.get("name") or (mentions[0]["text"] if mentions else ""),
            "mentions": mentions}


def evidence(text: str, sents: list[tuple[int, int]], m1: list[dict], m2: list[dict]) -> str:
    def where(ms: list[dict]) -> set[int]:
        return {i for i, (s, t) in enumerate(sents) for m in ms if s <= m["start"] < t}

    s1, s2 = where(m1), where(m2)
    both = s1 & s2
    idx = min(both) if both else min(s1) if s1 else None
    return text[sents[idx][0]:sents[idx][1]].strip() if idx is not None else ""


class Extractor:
    def __init__(self, re_dir: str | Path, ner_dir: str | Path | None = None,
                 device: torch.device | str | None = None) -> None:
        self.device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.tok = DocRE.load(re_dir, self.device)
        re_dir = Path(re_dir)
        self.name = re_dir.parent.name if re_dir.name == "best" else re_dir.name
        self.ner = self.ner_tok = self.ner_name = None
        if ner_dir:
            ner_dir = Path(ner_dir)
            self.ner = AutoModelForTokenClassification.from_pretrained(ner_dir).to(self.device).eval()
            self.ner_tok = AutoTokenizer.from_pretrained(ner_dir)
            self.ner_name = ner_dir.parent.name if ner_dir.name == "model" else ner_dir.name

    def ner_entities(self, text: str) -> list[dict]:
        if self.ner is None:
            raise RuntimeError("no NER model loaded: pass entities explicitly or give ner_dir")
        return group_mentions(text, predict_spans(self.ner, self.ner_tok, [text], device=self.device)[0])

    def extract(self, title: str, abstract: str, entities: list[dict] | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        text = full_text(title, abstract)
        ents = [check_entity(e, text) for e in entities] if entities is not None else self.ner_entities(text)
        mentions = [{"start": m["start"], "end": m["end"], "text": m["text"], "type": e["type"], "ids": [e["id"]]}
                    for e in ents for m in e["mentions"]]
        mentions.sort(key=lambda m: (m["start"], -m["end"]))   # data.py mention order
        doc_ents = build_entities(mentions)
        doc = {"pmid": "", "title": title, "abstract": abstract, "text": text, "mentions": mentions,
               "entities": doc_ents, "pairs": make_pairs(doc_ents, [])[:MAX_PAIRS], "relations": []}
        rows = predict(self.model, [featurize(doc, self.tok)], device=self.device) if doc["pairs"] else []
        sents = [(m.start(), m.end()) for m in _SENT.finditer(text)]
        by_id = {e["id"]: e for e in ents}
        rels: list[dict] = []
        for r in rows:
            if r["pred"] == "None":
                continue
            rels.append({"id": f"R{len(rels) + 1}", "e1": r["e1"], "e2": r["e2"], "type": r["pred"],
                         "score": r["probs"][r["pred"]], "probs": r["probs"],
                         "causal": is_causal_pair(r["t1"], r["t2"]) and r["pred"] in CAUSAL_TYPES,
                         "evidence": evidence(text, sents, by_id[r["e1"]]["mentions"], by_id[r["e2"]]["mentions"])})
        return {"doc_id": "", "title": title, "text": text, "entities": ents, "relations": rels,
                "meta": {"model": self.name, "ner": "given" if entities is not None else self.ner_name,
                         "n_pairs": len(doc["pairs"]), "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="run the extractor on one text and print the ExtractResult JSON")
    ap.add_argument("--re-dir", required=True)
    ap.add_argument("--ner-dir")
    ap.add_argument("--title", default="")
    ap.add_argument("--text", required=True)
    ap.add_argument("--device")
    a = ap.parse_args(argv)
    print(json.dumps(Extractor(a.re_dir, a.ner_dir, a.device).extract(a.title, a.text), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
