"""Name-swap counterfactuals (paper 2, E2; docs/research/paper2/codemap.md §1.5): Doc -> Doc before featurize, so the
model, featurize and mark_text see consistent offsets with no change of their own."""
from __future__ import annotations

import re

from biocausal.data import Doc, check_doc


def swap_names(doc: Doc, repl: dict[str, str], unannotated: bool = True) -> Doc:
    """``doc`` with every mention of each entity id in ``repl`` replaced by its string; ``unannotated``: also every
    whole-word, case-sensitive occurrence of those mentions' surface strings that no replaced mention covers
    (PubTator3 misses mentions; longest surface first), including one inside another mention.

    Offsets, ``text``/``title``/``abstract`` and ``mention.text`` are recomputed: a mention (or the title) containing a
    replaced span grows or shrinks with it; one that partially overlaps a replaced span or lies inside it raises
    ValueError (the caller skips and counts the document). Entity ids, ``pairs`` and ``relations`` are unchanged, so
    prediction rows pair up with the original ones by (pmid, e1, e2).
    """
    text, ms = doc["text"], doc["mentions"]
    spans: dict[tuple[int, int], str] = {}
    for e, new in repl.items():
        for i in doc["entities"][e]["mentions"]:
            if spans.setdefault((ms[i]["start"], ms[i]["end"]), new) != new:
                raise ValueError(f"{doc['pmid']}: mention {ms[i]} of two replaced entities")
    if unannotated:
        surfaces = {(ms[i]["text"], new) for e, new in repl.items() for i in doc["entities"][e]["mentions"]}
        for s, new in sorted(surfaces, key=lambda x: (-len(x[0]), x)):
            for m in re.finditer(rf"(?<!\w){re.escape(s)}(?!\w)", text):
                if all(m.end() <= a or m.start() >= b for a, b in spans):
                    spans[m.span()] = new
    order = sorted(spans)
    if any(s1 < e0 for (_, e0), (s1, _) in zip(order, order[1:])):
        raise ValueError(f"{doc['pmid']}: overlapping replaced mentions")
    for a, b in [(0, len(doc["title"]))] + [(m["start"], m["end"]) for m in ms]:
        if not all(b <= s or a >= e or a <= s <= e <= b for s, e in order):
            raise ValueError(f"{doc['pmid']}: span {a}-{b} partially overlaps a replaced span")

    def at(p: int) -> int:   # new offset of a boundary that is not strictly inside a replaced span
        return p + sum(len(spans[k]) - (k[1] - k[0]) for k in order if k[1] <= p)

    parts, cur = [], 0
    for s, e in order:
        parts += [text[cur:s], spans[(s, e)]]
        cur = e
    new_text = "".join(parts) + text[cur:]
    t = at(len(doc["title"]))
    mentions = []
    for m in ms:
        a, b = at(m["start"]), at(m["end"])
        mentions.append({**m, "start": a, "end": b, "text": new_text[a:b]})
    out = {**doc, "title": new_text[:t], "abstract": new_text[t + 1:], "text": new_text, "mentions": mentions}
    check_doc(out, strict=False)   # relations are untouched; strict=False also admits BC8 docs
    return out
