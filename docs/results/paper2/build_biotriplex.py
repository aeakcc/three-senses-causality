"""Paper 2, secondary external test: BioTriplex abstract triples -> (PMID, gene Entrez, disease MeSH) units.

    PYTHONPATH=. .venv/bin/python docs/results/paper2/build_biotriplex.py
      reads  data/external/biotriplex/ids.json, repo/data/BioTriplex Corpus/<id>/<id>_<Abstract*>.xml
             (external_tests.md §(a); abstract files are named Abstract, ABSTRACT or "Abstract Rev")
      writes data/paper2/biotriplex_docs.json, biotriplex_units.tsv, biotriplex_manifest.json   (units.finish)

BioTriplex has no identifiers: a triple's gene / disease text is linked to the PT3 Doc of its PMID by case-insensitive,
whitespace-normalised equality with a mention text of that type (gene -> Entrez, disease -> MeSH); a gene text with no
such mention falls back to the doc's gene entities among its HGNC symbol / alias / previous-symbol Entrez ids (PT3
misses mentions: "survivin" -> 332, "MMP9" -> 4318). Unlinked -> ''.
Unit = (PMID, linked gene ids, linked disease ids); classes = the regrouped fine relation types of its triples.
"""
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

from pt3_docs import build_docs
from units import EXT, HGNC, D, G, finish

SRC = EXT / "biotriplex"
# fine relation type -> class; every type seen in the corpus is listed, anything else (relation undefined,
# no relation, modulator increase/decrease disease, a link without relation) -> OTHER
CLASS = {**dict.fromkeys(["causative activation", "causative inhibition", "pathological role"], "CAUSAL_MECH"),
         "causative mutation": "CAUSAL_MUT",
         **dict.fromkeys(["associated mutation", "genetic susceptibility"], "ASSOC_GEN"),
         **dict.fromkeys(["increased expression", "decreased expression", "dysregulation"], "LEVEL"),
         **dict.fromkeys(["negative prognostic marker", "positive prognostic marker", "prognostic indicator",
                          "diagnostic tool", "biomarker", "epigenetic marker"], "MARKER"),
         **dict.fromkeys(["therapeutic target", "therapy resistance"], "THERAPY")}


def relation_class(rel: str | None) -> str:
    return CLASS.get(rel, "OTHER")


def norm(s: str) -> str:
    return " ".join(s.split()).lower()


def triples() -> list[tuple[str, str, str, str | None, str, str]]:
    """(pmid, gene text, disease text, fine relation type, pmcid, split) of every abstract ENTITY_LINKING."""
    out = []
    for d, v in json.load(open(SRC / "ids.json", encoding="utf-8")).items():
        for f in sorted((SRC / "repo/data/BioTriplex Corpus" / d).glob(f"{d}_*.xml")):
            if not f.stem.split("_", 1)[1].lower().startswith("abstract"):
                continue
            tags = ET.parse(f).getroot().find("TAGS")
            rel = {e.get("id"): e.get("relation") for e in tags if e.tag == "RELATION"}
            out += [(str(v["pmid"]), e.get("geneText"), e.get("diseaseText"), rel.get(e.get("relationID")),
                     v["pmcid"], v["split"]) for e in tags if e.tag == "ENTITY_LINKING"]
    return out


def hgnc_symbols() -> dict[str, set[str]]:
    """Lowercase HGNC symbol / alias / previous symbol -> Entrez ids."""
    out = defaultdict(set)
    for r in csv.DictReader(open(HGNC, encoding="utf-8"), delimiter="\t"):
        for s in [r["symbol"], *r["alias_symbol"].split("|"), *r["prev_symbol"].split("|")]:
            if s and r["entrez_id"]:
                out[s.lower()].add(r["entrez_id"])
    return out


def link(doc: dict | None, text: str, typ: str, symbols: dict[str, set[str]] | None = None) -> list[str]:
    """Entity ids of type ``typ`` among the doc's mentions whose text equals ``text`` (case/whitespace-insensitive);
    if none, the doc's ``typ`` entities among ``symbols[text]``."""
    if doc is None:
        return []
    ents = doc["entities"]
    ok = lambda i: ents.get(i, {}).get("type") == typ  # noqa: E731
    ids = {i for m in doc["mentions"] if norm(m["text"]) == norm(text) for i in m["ids"] if ok(i)}
    return sorted(ids or {i for i in (symbols or {}).get(norm(text), ()) if ok(i)})


def main() -> None:
    tr = triples()
    by_pmid = {d["pmid"]: d for d in build_docs({t[0] for t in tr})[0]}   # linking needs the docs; finish rereads
    symbols = hgnc_symbols()
    units = defaultdict(lambda: {"types": set(), "genes": set(), "diseases": set(), "n": 0})
    n_link = Counter()
    for pmid, g, d, rel, pmcid, split in tr:
        doc = by_pmid.get(pmid)
        gids, dids = link(doc, g, G, symbols), link(doc, d, D)
        n_link.update(triples=1, gene_linked_text=bool(link(doc, g, G)), gene_linked=bool(gids),
                      both_linked=bool(gids and dids))
        u = units[(pmid, tuple(gids), tuple(dids))]
        u["types"].add(rel or "None")
        u["genes"].add(g)
        u["diseases"].add(d)
        u["n"] += 1
        u["pmcid"], u["split"] = pmcid, split
    rows = [{"unit_id": f"biotriplex:{pmid}:{';'.join(g) or '-'}:{';'.join(d) or '-'}", "source": "biotriplex",
             "pmid": pmid, "gene_entrez": list(g), "disease_mesh": list(d),
             "classes": sorted({relation_class(t) for t in u["types"]}),
             "extra": {"pmcid": u["pmcid"], "split": u["split"], "types": sorted(u["types"]),
                       "gene_texts": sorted(u["genes"]), "disease_texts": sorted(u["diseases"]), "n_triples": u["n"]}}
            for (pmid, g, d), u in units.items()]
    finish("biotriplex", rows, triples_links=dict(n_link),
           fine_types=dict(Counter(t[3] or "None" for t in tr).most_common()))


if __name__ == "__main__":
    main()
