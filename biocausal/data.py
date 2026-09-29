"""BioRED BioC JSON -> Doc schema, typed entity markers, candidate pairs (DESIGN.md §1).

Doc = {pmid, title, abstract, text,
       mentions: [{start, end, text, type, ids:[str]}],   # char offsets into text, all 6 types
       entities: {id: {type, mentions:[int]}},             # core types only, id != "-"
       pairs:    [{e1, e2, label}],                        # canonical order, label in LABELS
       relations:[{e1, e2, type, novel}]}                  # gold as-is

Entity type = per-document majority over its mentions. DESIGN §1 says corpus-level majority for the
one id with two types ("9685": OrganismTaxon x3 in two docs, GeneOrGeneProduct x1 in pmid 18772193);
per-doc majority keeps each doc as annotated. Measured effect of the deviation: 9685 is a gene entity
in 18772193 only -> +13 candidate pairs, all labelled None, 0 gold relations touched.

Self-check: ``python -m biocausal.data --selfcheck [--data-dir data/BioRED]``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

TYPE_ORDER = ["ChemicalEntity", "DiseaseOrPhenotypicFeature", "GeneOrGeneProduct", "SequenceVariant"]
CORE_TYPES = frozenset(TYPE_ORDER)
_SHORT = {"ChemicalEntity": "CHEMICAL", "DiseaseOrPhenotypicFeature": "DISEASE",
          "GeneOrGeneProduct": "GENE", "SequenceVariant": "VARIANT"}
MARKERS = {t: (f"[{s}]", f"[/{s}]") for t, s in _SHORT.items()}   # type -> (open, close)
MARKER_TOKENS = [m for t in TYPE_ORDER for m in MARKERS[t]]
MARKER_INIT_WORD = {t: s.lower() for t, s in _SHORT.items()}      # embedding init word

LABELS = ["None", "Association", "Positive_Correlation", "Negative_Correlation", "Bind",
          "Cotreatment", "Comparison", "Drug_Interaction", "Conversion"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
NOVELTY = ["No", "Novel"]        # relation.novel (BioRED and BC8 annotate it on every gold relation); index = class id
# paper 2 aux pair heads (model.DocRE.aux): pair[k] = class index, absent = ignored; pc = perturbation-causal (CRED),
# ev = coarse ClinGen evidence category (E5, docs/results/paper2/e5_sets.py), index into EV_CLASSES
EV_CLASSES = ("genetic", "expression", "model_rescue", "functional_alteration", "biochemical_interaction")
AUX_CLASSES = {"pc": 2, "ev": len(EV_CLASSES)}

_C, _D, _G, _V = TYPE_ORDER
# canonical (TYPE_ORDER) order; Gene-Variant (1 gold relation) deliberately excluded
ALLOWED_PAIRS = frozenset({(_C, _C), (_C, _D), (_C, _G), (_C, _V), (_D, _G), (_D, _V), (_G, _G), (_V, _V)})

Doc = dict[str, Any]


def canonical_pair(e1: str, t1: str, e2: str, t2: str) -> tuple[str, str]:
    """Order two entity ids by (TYPE_ORDER.index(type), id)."""
    return (e1, e2) if (TYPE_ORDER.index(t1), e1) <= (TYPE_ORDER.index(t2), e2) else (e2, e1)


def build_entities(mentions: list[dict]) -> dict[str, dict]:
    types: dict[str, Counter] = defaultdict(Counter)
    mids: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(mentions):
        for eid in m["ids"]:
            if eid == "-":
                continue
            types[eid][m["type"]] += 1
            mids[eid].append(i)
    ents = {}
    for eid, c in types.items():
        t = c.most_common(1)[0][0]
        if t in CORE_TYPES:
            ents[eid] = {"type": t, "mentions": mids[eid]}
    return ents


def make_pairs(entities: dict[str, dict], relations: list[dict]) -> list[dict]:
    """All unordered core-entity pairs of an allowed type combo, canonical order, gold label or None."""
    gold: dict[tuple[str, str], str] = {}
    for r in relations:
        if r["e1"] in entities and r["e2"] in entities and r["e1"] != r["e2"]:
            key = canonical_pair(r["e1"], entities[r["e1"]]["type"], r["e2"], entities[r["e2"]]["type"])
            gold[key] = r["type"]
    order = sorted(entities, key=lambda e: (TYPE_ORDER.index(entities[e]["type"]), e))
    return [{"e1": a, "e2": b, "label": gold.get((a, b), "None")}
            for a, b in combinations(order, 2)
            if (entities[a]["type"], entities[b]["type"]) in ALLOWED_PAIRS]


def unpairable_gold(doc: Doc) -> list[str]:
    """Types of gold relations with no candidate pair (self-relations, Gene-Variant): fixed false negatives."""
    have = {frozenset((p["e1"], p["e2"])) for p in doc["pairs"] if p["label"] != "None"}
    return [r["type"] for r in doc.get("relations", []) if frozenset((r["e1"], r["e2"])) not in have]


def doc_from_bioc(bd: dict) -> Doc:
    text = ""
    for p in bd["passages"]:
        text = text.ljust(p["offset"]) + p["text"]
    title = bd["passages"][0]["text"]
    mentions = []
    for p in bd["passages"]:
        for a in p["annotations"]:
            loc = a["locations"][0]
            mentions.append({"start": loc["offset"], "end": loc["offset"] + loc["length"], "text": a["text"],
                             "type": a["infons"]["type"], "ids": [x.strip() for x in a["infons"]["identifier"].split(",") if x.strip()]})
    mentions.sort(key=lambda m: (m["start"], -m["end"]))
    entities = build_entities(mentions)
    relations = [{"e1": r["infons"]["entity1"], "e2": r["infons"]["entity2"],
                  "type": r["infons"]["type"], "novel": r["infons"].get("novel")} for r in bd.get("relations", [])]
    return {"pmid": str(bd["id"]), "title": title, "abstract": text[len(title) + 1:], "text": text,
            "mentions": mentions, "entities": entities, "pairs": make_pairs(entities, relations),
            "relations": relations}


def load_docs(paths: list[str | Path] | str | Path, limit: int | None = None) -> list[Doc]:
    """Load BioC JSON (BioRED) or Doc-schema JSON (data/extra) files; optional cap per file."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    docs: list[Doc] = []
    for path in paths:
        obj = json.load(open(path, encoding="utf-8"))
        items = obj if isinstance(obj, list) else obj.get("documents", obj.get("docs", []))
        if items and "passages" in items[0]:
            items = [doc_from_bioc(d) for d in items]
        docs.extend(items[:limit] if limit else items)
    return docs


def read_pmids(path: str | Path) -> set[str]:
    """PMIDs of a Doc/BioC JSON file (``.json``), or of a text file of whitespace-separated numeric PMIDs."""
    if Path(path).suffix.lower() == ".json":
        return {d["pmid"] for d in load_docs(path)}
    tokens = open(path, encoding="utf-8").read().split()
    bad = [t for t in tokens if not t.isdigit()]
    if bad:   # a wrong file (README, jsonl, xml) would otherwise exclude nothing, silently
        raise ValueError(f"{path}: not a PMID list (non-numeric tokens {bad[:3]})")
    return set(tokens)


def pmid_held(doc_id: str, held: set[str]) -> bool:
    """``doc_id`` is a held PMID, or a held PMID + 4 digits when longer than a PMID (8 digits): BioREx DisGeNET ids
    are PMID + 4 digits (``251455021001``), which an exact match would silently miss (improvement_plan K1)."""
    return doc_id in held or (len(doc_id) > 8 and doc_id[:-4] in held)


# ---- wave 2: labels of external corpora (preregistration §5, improvement_plan §4.2) -------------------------------
# Fixed before the first wave-2 run, never tuned on dev. Corpus = data/extra_full file stem. Type pairs in TYPE_ORDER.
_A, _P, _N, _BIND, _DI = "Association", "Positive_Correlation", "Negative_Correlation", "Bind", "Drug_Interaction"
# labels a corpus annotates for a type pair tau: its None can be any other label observed for tau in BioRED Train, L(tau)
EXPRESS = {
    "bc5cdr": {(_C, _D): {_P}},
    "drugprot": {(_C, _G): {_A, _P, _N}},
    "ddi2013": {(_C, _C): {_DI}},
    "gda": {(_D, _G): {_A, _P, _N}},
    "biorex_disgenet": {(_D, _G): {_A, _P, _N}},
    "biorex_emu": {(_D, _G): {_A, _P, _N}},
    "biorex_pharmgkb": {(_C, _G): {_A, _P, _N, _BIND}},
    "biorex_hprd50": {(_G, _G): {_A, _BIND}},
    "biorex_aimed": {(_G, _G): {_A, _BIND}},
}
# ambiguous positive mappings: BioRED label, or a source label in pair["src_labels"] -> admissible labels (--hard-map: off)
AMBIG = {
    "drugprot": {"DIRECT-REGULATOR": {_A, _BIND}},
    "ddi2013": {_DI: {_DI, _A}},
    "gda": {_A: {_A, _P, _N}},                  # distant positive: a relation of unknown type
    "biorex_disgenet": {_A: {_A, _P, _N}},
    "biorex_emu": {_A: {_A, _P, _N}},
    "biorex_pharmgkb": {_A: {_A, _P, _N, _BIND}},
    "biorex_hprd50": {_A: {_A, _BIND}},
    "biorex_aimed": {_A: {_A, _BIND}},
}
ESTIMATED = ("bc5cdr", "gda")                  # set = S table extended to 95 % coverage on docs shared with BioRED Train
SOFT = ("bc5cdr", "gda", "biorex_disgenet")    # soft-T (DisGeNET: GDA's T); every other corpus keeps its set
DROP_NONE = ("gda",)                           # distant None is unreliable: dropped in set/soft (K3)
# None pairs of external corpora in the baselines: c2 = labelled None, c3 = pair dropped, c1 = per-corpus None class
NONE_HANDLING = {
    "naive": dict.fromkeys(EXPRESS, "c2"),
    "drop": dict.fromkeys(EXPRESS, "c3"),
    "none_per_corpus": dict.fromkeys(EXPRESS, "c1"),
    "mixed": {**dict.fromkeys(EXPRESS, "c2"), "bc5cdr": "c3", "gda": "c3"},   # mst_seq_drop: the old --drop-none recipe
}
LABEL_HANDLING = (*NONE_HANDLING, "set", "soft")
NONE_SLOTS = ("bc5cdr", "drugprot", "ddi2013", "gda", "biorex_pharmgkb", "biorex_hprd50", "biorex_aimed")  # c1: class 9+i
# preregistration §10, arm A2: CRED (docs/results/cred/cred_split.py) labels only some gene-disease pairs, causal as
# Positive_Correlation and non-causal as Association; under "set" those become these sets and every None pair is dropped
CRED_SETS = {_P: {_P, _N}, _A: {"None", _A}}


def type_key(t1: str, t2: str) -> str:
    return f"{t1}|{t2}"


# `python -m data_tools.transitions` on the 138 BC5CDR / 45 GDA documents shared with BioRED Train (counts and derivation
# in data/extra_full/README.md; tests/test_w2_labels.py checks this copy against the estimate). L = L(tau) of BioRED Train.
# T[y_ext][k] = P(y_ext | k) = (n(y_ext, k) + 1) / (n(k) + |Y(tau)|), written as those counts; hard = S-table set plus the
# most frequent missing labels up to 95 % coverage (= L(tau), i.e. dropped in `set`, for BC5CDR None and GDA positives).
TRANSITIONS = {
    "L": {
        type_key(_C, _C): ["None", _A, _P, _N, "Cotreatment", "Comparison", _DI, "Conversion"],
        type_key(_C, _D): ["None", _A, _P, _N],
        type_key(_C, _G): ["None", _A, _P, _N, _BIND],
        type_key(_C, _V): ["None", _A, _P, _N],
        type_key(_D, _G): ["None", _A, _P, _N],
        type_key(_D, _V): ["None", _A, _P, _N],
        type_key(_G, _G): ["None", _A, _P, _N, _BIND],
        type_key(_V, _V): ["None", _A],
    },
    "corpora": {
        "bc5cdr": {type_key(_C, _D): {
            "T": {
                "None": {"None": 857 / 887, _A: 41 / 48, _P: 125 / 322, _N: 189 / 196},
                _P: {"None": 30 / 887, _A: 7 / 48, _P: 197 / 322, _N: 7 / 196},
            },
            "hard": {"None": ["None", _A, _P, _N], _P: ["None", _A, _P]},
        }},
        "gda": {type_key(_D, _G): {
            "T": {
                _A: {"None": 14 / 105, _A: 40 / 76, _P: 2 / 6, _N: 3 / 4},
                "None": {"None": 91 / 105, _A: 36 / 76, _P: 4 / 6, _N: 1 / 4},
            },
            "hard": {_A: ["None", _A, _P, _N], "None": ["None", _A]},
        }},
    },
}
# DisGeNET: same source as GDA, no shared documents -> GDA's T (preregistered fallback)
TRANSITIONS["corpora"]["biorex_disgenet"] = {
    type_key(_D, _G): {"T": {_A: TRANSITIONS["corpora"]["gda"][type_key(_D, _G)]["T"][_A]}}}


def label_set(corpus: str, tau: tuple[str, str], label: str, L: set[str], src_labels: list[str] = (),
              hard_map: bool = False) -> set[str]:
    """S table for a pair of a corpus whose set is not estimated: None -> L(tau) minus EXPRESS, positive -> AMBIG."""
    if label == "None":
        return L - EXPRESS[corpus][tau]
    hits = [] if hard_map else [AMBIG[corpus][k] for k in (label, *src_labels) if k in AMBIG.get(corpus, {})]
    return set().union(*hits) if hits else {label}


def label_logw(doc: Doc, corpus: str | None, mode: str, hard_map: bool = False,
               transitions: dict | None = None) -> list[list[float] | None]:
    """Per candidate pair: log-weights W over LABELS (loss = logsumexp(z) - logsumexp(z + W)), or None = pair dropped.

    ``corpus`` None (BioRED, BC8): one-hot gold in every mode. naive/drop/mixed: one-hot, None pairs dropped where
    NONE_HANDLING says c3. set: 0/-inf over S (ESTIMATED corpora: TRANSITIONS coverage sets); a set covering
    L(tau) carries no information and is dropped (K3). soft: log T for SOFT corpora, set otherwise. c1 has no weights.
    cred / clingen (set only): CRED_SETS for labelled CRED pairs; a pair with an aux label only (cred "pc", clingen
    "ev") gets W = 0, i.e. exactly zero main-head loss and gradient (its aux head learns); every other pair is dropped.
    """
    if mode == "none_per_corpus" or mode not in LABEL_HANDLING:
        raise ValueError(f"label_logw: mode {mode!r} (c1 uses extra None classes, see model.featurize)")
    if corpus in ("cred", "clingen"):
        if mode != "set":
            raise ValueError(f"label_logw: corpus {corpus!r} needs mode 'set', got {mode!r}")
        aux = "pc" if corpus == "cred" else "ev"
        return [[0.0 if l in CRED_SETS[p["label"]] else -math.inf for l in LABELS]
                if corpus == "cred" and p["label"] != "None" else [0.0] * len(LABELS) if aux in p else None
                for p in doc["pairs"]]
    if corpus is not None and corpus not in EXPRESS:
        raise ValueError(f"unknown corpus {corpus!r}: the file stem must be one of {list(EXPRESS)}")
    ninf = -math.inf
    out: list[list[float] | None] = []
    for p in doc["pairs"]:
        y = p["label"]
        if corpus is None or mode in NONE_HANDLING:
            drop = corpus is not None and y == "None" and NONE_HANDLING[mode][corpus] == "c3"
            out.append(None if drop else [0.0 if l == y else ninf for l in LABELS])
            continue
        tr = transitions or TRANSITIONS
        tau = (doc["entities"][p["e1"]]["type"], doc["entities"][p["e2"]]["type"])
        key = type_key(*tau)
        if y == "None" and corpus in DROP_NONE:
            out.append(None)
        elif mode == "soft" and corpus in SOFT:
            T = tr["corpora"][corpus][key]["T"][y]
            out.append([math.log(T[l]) if T.get(l) else ninf for l in LABELS])
        else:
            L = set(tr["L"][key])
            S = set(tr["corpora"][corpus][key]["hard"][y]) if corpus in ESTIMATED else \
                label_set(corpus, tau, y, L, p.get("src_labels", ()), hard_map)
            out.append(None if S >= L else [0.0 if l in S else ninf for l in LABELS])
    return out


def mark_text(doc: Doc) -> tuple[str, dict[int, int]]:
    """Insert typed markers around every mention of every core entity.

    Returns (marked_text, {mention_index: char offset of its start marker in marked_text}).
    Marker type = the entity's type (falls back to the mention's own type only for consistency).
    """
    ents = doc["entities"]
    inserts: list[tuple[int, int, tuple[int, int], int, str]] = []   # (pos, is_open, tiebreak, mention_idx, marker)
    for i, m in enumerate(doc["mentions"]):
        etypes = [ents[e]["type"] for e in m["ids"] if e in ents]
        if not etypes:
            continue
        t = min(etypes, key=TYPE_ORDER.index)
        op, cl = MARKERS[t]
        inserts.append((m["start"], 1, (-m["end"], i), i, op))    # outer span opens first, then mention order
        inserts.append((m["end"], 0, (-m["start"], -i), i, cl))   # inner span closes first; same span: last opened first
    inserts.sort(key=lambda x: x[:3])
    out: list[str] = []
    n = 0
    cur = 0
    pos: dict[int, int] = {}
    text = doc["text"]
    for at, is_open, _, i, marker in inserts:
        out.append(text[cur:at])
        n += at - cur
        cur = at
        out.append(" ")
        n += 1
        if is_open:
            pos[i] = n
        out.append(marker)
        n += len(marker)
        out.append(" ")
        n += 1
    out.append(text[cur:])
    return "".join(out), pos


def check_doc(d: Doc, strict: bool = True) -> list[tuple]:
    """Assert the Doc invariants (offsets = text slices, relations resolve to core entities and sit in ``pairs``
    unless unpairable, markers); returns the unpairable gold relations.

    ``strict=False`` (BC8): a relation on an unlinked (``-``), unmentioned or non-core id is unpairable, not an error.
    """
    for m in d["mentions"]:
        assert d["text"][m["start"]:m["end"]] == m["text"], (d["pmid"], m)
    ents = d["entities"]
    all_ids = {i for m in d["mentions"] for i in m["ids"]}
    index = {(p["e1"], p["e2"]): p["label"] for p in d["pairs"]}
    assert len(index) == len(d["pairs"]), ("duplicate pair", d["pmid"])
    excluded = []
    for r in d["relations"]:
        if not strict and not (r["e1"] in ents and r["e2"] in ents):
            excluded.append((d["pmid"], r["e1"], None, r["e2"], None, r["type"]))
            continue
        assert r["e1"] in all_ids and r["e2"] in all_ids, ("unresolved relation id", d["pmid"], r)
        assert r["e1"] in ents and r["e2"] in ents, ("relation on non-core entity", d["pmid"], r)
        t1, t2 = ents[r["e1"]]["type"], ents[r["e2"]]["type"]
        key = canonical_pair(r["e1"], t1, r["e2"], t2)
        if r["e1"] == r["e2"] or (ents[key[0]]["type"], ents[key[1]]["type"]) not in ALLOWED_PAIRS:
            excluded.append((d["pmid"], r["e1"], t1, r["e2"], t2, r["type"]))
            continue
        assert index.get(key) == r["type"], ("gold relation missing from pairs", d["pmid"], r)
    assert len(unpairable_gold(d)) == len(excluded), ("unpairable_gold mismatch", d["pmid"])
    marked, pos = mark_text(d)
    assert all(marked[p:].startswith(MARKERS[ents[[e for e in d["mentions"][i]["ids"] if e in ents][0]]["type"]][0])
               for i, p in pos.items()), d["pmid"]
    return excluded


def selfcheck(data_dir: str | Path = "data/BioRED") -> None:
    data_dir = Path(data_dir)
    for split in ("Train", "Dev", "Test"):
        docs = load_docs(data_dir / f"{split}.BioC.JSON")
        hist: Counter = Counter()
        excluded: list[tuple] = []
        n_pairs = n_mentions = n_composite = n_unlinked = 0
        for d in docs:
            for m in d["mentions"]:
                n_mentions += 1
                n_composite += len(m["ids"]) > 1
                n_unlinked += "-" in m["ids"]
            excluded += check_doc(d)
            n_pairs += len(d["pairs"])
            hist.update(p["label"] for p in d["pairs"])
        n_rel = sum(len(d["relations"]) for d in docs)
        print(f"{split}: {len(docs)} docs, {n_mentions} mentions ({n_composite} composite ids, {n_unlinked} unlinked), "
              f"{sum(len(d['entities']) for d in docs)} core entities, {n_rel} gold relations, "
              f"{n_pairs} candidate pairs ({n_pairs / len(docs):.1f}/doc, "
              f"{100 * hist['None'] / n_pairs:.1f}% None), {len(excluded)} relations not pairable")
        print("  labels:", dict(sorted(hist.items(), key=lambda kv: -kv[1])))
        for ex in excluded:
            print("  excluded relation:", ex)
    print("selfcheck OK")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--data-dir", default="data/BioRED")
    ap.add_argument("--dump", help="write the Doc-schema JSON of the given --data-dir files to this directory")
    a = ap.parse_args(argv)
    if a.dump:
        out = Path(a.dump)
        out.mkdir(parents=True, exist_ok=True)
        for split in ("Train", "Dev", "Test"):
            docs = load_docs(Path(a.data_dir) / f"{split}.BioC.JSON")
            json.dump(docs, open(out / f"{split}.json", "w", encoding="utf-8"), ensure_ascii=False)
            print(f"wrote {len(docs)} docs to {out / f'{split}.json'}")
    if a.selfcheck or not a.dump:
        selfcheck(a.data_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
