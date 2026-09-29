"""Paper 2 units: the unit TSV shared by the external evidence sets (ClinGen, RGD, Percha, BioTriplex), the identifier
maps, and the candidate-pair rule every evaluator uses.

Unit TSV (header, one row per unit): unit_id, source, doc_id (the Doc 'pmid' in the docs JSON), pmid, gene_entrez,
disease_mesh, classes (';'-joined), reachable (a gene_entrez id is a gene entity of the doc), disease_matched (a
disease_mesh id is a disease entity of the doc), extra (JSON). A unit's score under a model = max over
candidate_pairs(doc, gene_entrez, disease_mesh) of the pair score.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pt3_docs  # also puts ROOT, docs/results/cred and docs/results/posthoc on sys.path
from biocausal.data import read_pmids
from cred_build import split_ids

ROOT = pt3_docs.ROOT
EXT = ROOT / "data/external"
HGNC, SSSOM = EXT / "research_2026-09-17/hgnc.txt", EXT / "research_2026-09-17/mondo.sssom.tsv"
OBOS = (EXT / "doid/doid.obo", EXT / "rgd/RDO.obo")
OUT = ROOT / "data/paper2"
D, G = "DiseaseOrPhenotypicFeature", "GeneOrGeneProduct"
COLS = ("unit_id", "source", "doc_id", "pmid", "gene_entrez", "disease_mesh", "classes", "reachable",
        "disease_matched", "extra")
LISTS = ("gene_entrez", "disease_mesh", "classes")
csv.field_size_limit(sys.maxsize)


# ---- candidate pairs ---------------------------------------------------------------------------------------------
def candidate_pairs(doc: dict, gene_entrez, disease_mesh) -> tuple[list[tuple[str, str]], str]:
    """(pairs, anchor): the doc's disease-gene pairs (e1 disease, e2 gene: canonical order) whose gene is one of
    ``gene_entrez``; anchor 'pair' = only those whose disease is one of ``disease_mesh``, when there are some;
    'gene' = the gene with every disease entity; 'none' = no pair (gene not an entity, or no disease entity)."""
    ents, genes, dis = doc["entities"], set(gene_entrez), set(disease_mesh)
    dg = [(p["e1"], p["e2"]) for p in doc["pairs"]
          if p["e2"] in genes and ents[p["e2"]]["type"] == G and ents[p["e1"]]["type"] == D]
    hit = [x for x in dg if x[0] in dis]
    return (hit, "pair") if hit else (dg, "gene" if dg else "none")


def reach(doc: dict, gene_entrez, disease_mesh) -> tuple[int, int]:
    """(reachable, disease_matched): some gene id is a gene entity / some MeSH id is a disease entity of the doc."""
    ents = doc["entities"]
    return tuple(int(any(ents.get(i, {}).get("type") == t for i in ids)) for ids, t in ((gene_entrez, G), (disease_mesh, D)))


# ---- identifier maps ---------------------------------------------------------------------------------------------
def hgnc(path: Path = HGNC) -> dict[str, tuple[str, str]]:
    """HGNC id ('HGNC:5') -> (approved symbol, Entrez id), for the rows with an Entrez id."""
    return {r["hgnc_id"]: (r["symbol"], r["entrez_id"])
            for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t") if r["entrez_id"]}


def _exact(path: Path) -> dict[str, set[str]]:
    rows = csv.DictReader((l for l in open(path, encoding="utf-8") if not l.startswith("#")), delimiter="\t")
    out = defaultdict(set)
    for r in rows:
        if r["predicate_id"] == "skos:exactMatch":
            out[r["subject_id"]].add(r["object_id"])
    return out


def _mesh(ids) -> set[str]:
    return {i.split(":", 1)[1] for i in ids if i.upper().startswith("MESH:")}


def mondo_mesh(path: Path = SSSOM) -> dict[str, set[str]]:
    """MONDO id -> MeSH ids (prefix stripped) of its skos:exactMatch mappings in the MONDO SSSOM file."""
    return {m: _mesh(o) for m, o in _exact(path).items()}


def doid_mesh(sssom: Path = SSSOM, obos=OBOS) -> dict[str, set[str]]:
    """DOID id -> MeSH ids, as docs/research/paper2/external_reach.py maps RGD terms: DOID =exactMatch= MONDO
    =exactMatch= MeSH, plus the MESH xrefs of the DOID stanzas of DO and of RGD's RDO (RGD-only DOID:900xxxx terms)."""
    out = defaultdict(set)
    for objs in _exact(sssom).values():
        for o in objs:
            if o.startswith("DOID:"):
                out[o] |= _mesh(objs)
    for block in "\n\n".join(Path(f).read_text(encoding="utf-8") for f in obos).split("\n\n"):
        if m := re.search(r"^id: (DOID:\d+)$", block, re.M):
            out[m[1]] |= set(re.findall(r"^xref: MESH:(\S+)", block, re.M))
    return dict(out)


def gene_names(pmid: str) -> dict[str, str]:
    """Gene id -> lower-case PubTator3 symbol (the 'name' infon) in the cached PT3 document of ``pmid``."""
    bd = json.loads((pt3_docs.PT3_DIR / f"{pmid}.json").read_text(encoding="utf-8"))
    return {i: str(a["infons"].get("name") or "").lower() for p in bd["passages"] for a in p["annotations"]
            if a["infons"].get("type") == "Gene" for i in split_ids(a["infons"].get("identifier"))}


# ---- unit TSV ----------------------------------------------------------------------------------------------------
def write_units(path: Path, units: list[dict]) -> None:
    """Unit dicts -> TSV (COLS); list fields ';'-joined (a str is written as is), extra as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(COLS) + "\n")
        for u in units:
            v = {**u, **{k: u[k] if isinstance(u[k], str) else ";".join(u[k]) for k in LISTS},
                 "extra": json.dumps(u.get("extra", {}), ensure_ascii=False, sort_keys=True)}
            row = [str(v[c]) for c in COLS]
            assert not any(c in x for x in row for c in "\t\n"), row
            f.write("\t".join(row) + "\n")


def read_units(path: Path) -> list[dict]:
    """TSV -> unit dicts: list fields as lists, reachable / disease_matched as int, extra as dict."""
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        rows = [dict(zip(head, l.rstrip("\n").split("\t"))) for l in f]
    for u in rows:
        u.update({k: u[k].split(";") if u[k] else [] for k in LISTS}, reachable=int(u["reachable"]),
                 disease_matched=int(u["disease_matched"]), extra=json.loads(u["extra"]))
    return rows


# ---- shared builder tail -----------------------------------------------------------------------------------------
def finish(source: str, units: list[dict], out_dir: Path = OUT, **counts) -> dict:
    """PT3 docs of the units' PMIDs (pt3_docs.build_docs: BioRED/BC8/CRED PMIDs excluded, missing/no-abstract skipped),
    units without a doc dropped and counted, ortholog gene ids added, reach flags set; writes <source>_docs.json,
    <source>_units.tsv and <source>_manifest.json (sha256 of both, counts, per-class coverage); returns the manifest.
    ``units``: dicts with unit_id, source, pmid, gene_entrez (human Entrez ids), disease_mesh, classes, extra."""
    docs, dman = pt3_docs.build_docs({u["pmid"] for u in units})
    by = {d["pmid"]: d for d in docs}
    held = set().union(*(read_pmids(ROOT / f) for fs in pt3_docs.SOURCES.values() for f in fs))
    symbol = {e: s.lower() for s, e in hgnc(HGNC).values()}
    kept, dropped = [], Counter()
    for u in units:
        d = by.get(u["pmid"])
        if d is None:
            dropped["excluded" if u["pmid"] in held else "no_doc"] += 1
            continue
        human = list(u["gene_entrez"])
        # PT3 often normalizes a model-organism study's gene to the mouse/rat/fish id: same symbol, other Entrez id
        # ponytail: ortholog = same symbol ignoring case; NCBI gene_orthologs if symbol drift ever matters
        want = {symbol.get(g) for g in human} - {None}
        orth = sorted(i for i, n in gene_names(u["pmid"]).items() if n in want and i not in human)
        if orth:
            u = {**u, "gene_entrez": human + orth, "extra": {**u["extra"], "ortholog_entrez": orth}}
        r, m = reach(d, u["gene_entrez"], u["disease_mesh"])
        kept.append({**u, "doc_id": d["pmid"], "reachable": r, "disease_matched": m,
                     "reachable_human": reach(d, human, ())[0]})
    cov = defaultdict(Counter)
    for u in kept:
        for c in [*u["classes"], "_all"]:
            cov[c].update(units=1, reachable=u["reachable"], reachable_human=u["reachable_human"],
                          disease_matched=u["disease_matched"], both=u["reachable"] & u["disease_matched"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, fu = out_dir / f"{source}_docs.json", out_dir / f"{source}_units.tsv"
    with open(fd, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    write_units(fu, sorted(kept, key=lambda u: (int(u["pmid"]), u["unit_id"])))
    man = {"source": source, **counts, "docs": dman, "n_units_in": len(units), "n_units_dropped": dict(dropped),
           "n_units": len(kept), "n_units_ortholog": sum("ortholog_entrez" in u["extra"] for u in kept),
           "sha256": {f.name: hashlib.file_digest(open(f, "rb"), "sha256").hexdigest() for f in (fd, fu)},
           "coverage": {c: dict(v) for c, v in sorted(cov.items())}}
    (out_dir / f"{source}_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k != "coverage"}))
    print(f"{'class':10} {'units':>6} {'reach':>6} {'r_hum':>6} {'dis':>6} {'both':>6}")
    for c, v in man["coverage"].items():
        print(f"{c:10} {v['units']:6} {v['reachable']:6} {v['reachable_human']:6} {v['disease_matched']:6} {v['both']:6}")
    return man
