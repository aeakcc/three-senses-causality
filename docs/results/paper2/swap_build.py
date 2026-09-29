r"""Paper 2, E2: name-swap counterfactuals (docs/research/paper2/codemap.md §1.5, §3 E2) via biocausal.swap.swap_names.

    PYTHONPATH=. .venv/bin/python docs/results/paper2/swap_build.py
      reads  data/CRED/cv/{fold,test_f}{0-4}.{txt,json}, data/CRED/cred_pairs.tsv (cred_s10.load_rows),
             data/paper2/clingen_{units.tsv,docs.json} (skipped with a note when absent),
             data/external/research_2026-09-17/{hgnc.txt,gencc.tsv,en_product6.xml},
             data/external/clingen_gdv_2026-09-23/curations.tsv, data/external/rgd/rgd_human_manual.tsv.gz
      writes data/paper2/swap/cred_{u,k}_f{0-4}.json   fold k's docs (cv/test_f{k}.json) swapped: fold models score them
             data/paper2/swap/clingen_{u,k}.json       ClinGen docs with an E1 primary-contrast unit swapped
             data/paper2/swap/mapping.tsv              file, doc_id, orig_doc_id, gene, surface, replacement
             data/paper2/swap/manifest.json            counts per file, matching rule, sha256 of every file

Targets: CRED - the gene of every labelled reachable single-gene pair of the 159-document pool (cv folds); ClinGen -
the genes (that are gene entities of the doc) of reachable units with a PRIMARY class. Each target gene gets its own
copy of the doc, id '<doc id>_s<i>' (i = index in the doc's sorted target gene ids), with every mention of that gene
entity (and unannotated whole-word occurrences of its surfaces) renamed; entity ids and pairs are unchanged, so a
prediction row pairs with the original by (orig_doc_id, e1, e2).
Conditions: u - an 'unknown' gene: HGNC approved protein-coding symbol with no GenCC, ClinGen, RGD (human manual) or
Orphanet (product 6) record and no entity in any BioRED / BC8 / CRED doc; k - a 'known' gene: ClinGen Definitive.
Both: >= 3 characters, neither an entity id nor a whole word (case-insensitive) of the doc, and as many BioLinkBERT
wordpieces as the gene's most frequent surface (character length +-1 if the tokenizer is not cached); drawn by
random.Random(int(orig_doc_id) + i) from the sorted candidates. No candidate / swap_names ValueError: skipped, counted.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "docs/results/cred"), str(Path(__file__).resolve().parent)]

from biocausal.data import load_docs, read_pmids  # noqa: E402
from biocausal.swap import swap_names  # noqa: E402
from cred_s10 import load_rows  # noqa: E402
from pt3_docs import SOURCES  # noqa: E402

EXT = ROOT / "data/external"
CV, P2 = ROOT / "data/CRED/cv", ROOT / "data/paper2"
OUT = P2 / "swap"
TOKENIZER = "michiyasunaga/BioLinkBERT-base"
PRIMARY = ("GEN_CLV", "EXP_A", "EXP_B")        # E1 primary contrast: genetic case-level variants vs experimental
GENE = "GeneOrGeneProduct"
csv.field_size_limit(sys.maxsize)              # unit TSV extra (JSON) can be long


def sizer() -> tuple:
    """(size function of a string, tolerance, rule): BioLinkBERT wordpiece count, or character length +-1 when the
    tokenizer is not in the local HF cache."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
        return (lambda s: len(tok.tokenize(s))), 0, f"{TOKENIZER} wordpieces, equal"
    except OSError:
        return len, 1, f"character length +-1 ({TOKENIZER} tokenizer not cached)"


def pools(size) -> dict[str, dict[int, list[tuple[str, str]]]]:
    """{'u': unknown, 'k': known} -> size -> sorted [(symbol, entrez)] of HGNC approved genes with >= 3 characters."""
    rd = lambda p: csv.DictReader(open(p, encoding="utf-8"), delimiter="\t")  # noqa: E731
    hgnc = [r for r in rd(EXT / "research_2026-09-17/hgnc.txt") if r["entrez_id"] and len(r["symbol"]) >= 3]
    gencc = list(rd(EXT / "research_2026-09-17/gencc.tsv"))
    cur = list(rd(EXT / "clingen_gdv_2026-09-23/curations.tsv"))
    orpha = (EXT / "research_2026-09-17/en_product6.xml").read_text(encoding="utf-8")
    ids = ({r["gene_curie"] for r in gencc} | {r["hgnc_id"] for r in cur}
           | {f"HGNC:{x}" for x in re.findall(r"<Source>HGNC</Source>\s*<Reference>(\d+)</Reference>", orpha)})
    syms = ({r["gene_symbol"] for r in gencc} | {r["gene_symbol"] for r in cur} | set(re.findall(r"<Symbol>([^<]+)</Symbol>", orpha))
            | {r["symbol"] for r in csv.DictReader(gzip.open(EXT / "rgd/rgd_human_manual.tsv.gz", "rt"), delimiter="\t")})
    seen = {e for f in (f for fs in SOURCES.values() for f in fs) for d in load_docs(ROOT / f)
            for e, v in d["entities"].items() if v["type"] == GENE}
    definitive = {r["hgnc_id"] for r in cur if r["classification"] == "Definitive"}
    keep = {"u": [r for r in hgnc if r["locus_group"] == "protein-coding gene" and r["hgnc_id"] not in ids
                  and r["entrez_id"] not in seen and not {r["symbol"], *r["prev_symbol"].split("|")} & syms],
            "k": [r for r in hgnc if r["hgnc_id"] in definitive]}
    out: dict[str, dict[int, list]] = {}
    for c, rows in keep.items():
        out[c] = defaultdict(list)
        for r in sorted(rows, key=lambda r: r["symbol"]):
            out[c][size(r["symbol"])].append((r["symbol"], r["entrez_id"]))
    return out


def pick(pool: dict[int, list[tuple[str, str]]], n: int, tol: int, doc: dict, seed: int) -> str | None:
    """A symbol of size n +- tol that is neither an entity id nor a whole word of ``doc``, drawn with ``seed``."""
    low = doc["text"].lower()
    words = set(re.findall(r"\w+", low)) | set(re.findall(r"[\w-]+", low))
    ok = [s for k in range(n - tol, n + tol + 1) for s, e in pool.get(k, ())
          if e not in doc["entities"] and s.lower() not in words]
    return random.Random(seed).choice(ok) if ok else None


def swap_docs(docs: list[dict], targets: dict[str, set[str]], pool: dict, size, tol: int
              ) -> tuple[list[dict], list[list[str]], Counter]:
    """(swapped docs, mapping rows [doc_id, orig_doc_id, gene, surface, replacement], counts) of one condition."""
    out, mapping, n = [], [], Counter()
    for d in docs:
        genes = sorted(g for g in targets.get(d["pmid"], ()) if d["entities"].get(g, {}).get("type") == GENE)
        for i, g in enumerate(genes):
            n["targets"] += 1
            surface = Counter(d["mentions"][j]["text"] for j in d["entities"][g]["mentions"]).most_common(1)[0][0]
            new = pick(pool, size(surface), tol, d, int(d["pmid"]) + i)
            if new is None:
                n["no_candidate"] += 1
                continue
            try:
                s = swap_names(d, {g: new})
            except ValueError:
                n["value_error"] += 1
                continue
            out.append({**s, "pmid": f"{d['pmid']}_s{i}"})
            mapping.append([f"{d['pmid']}_s{i}", d["pmid"], g, surface, new])
    n["docs"] = len(out)
    return out, mapping, n


def cred_targets() -> dict[str, set[str]]:
    pool = set().union(*(read_pmids(CV / f"fold{k}.txt") for k in range(5)))
    t: dict[str, set[str]] = defaultdict(set)
    for r in (r for s in ("train", "val", "test") for r in load_rows(s)):
        if r["pmid"] in pool and r["status"] == "reachable" and ";" not in r["pairs"]:
            t[r["pmid"]].add(r["pairs"].split("|")[1])
    return t


def unit_targets(units_tsv: Path) -> dict[str, set[str]]:
    t: dict[str, set[str]] = defaultdict(set)
    for u in csv.DictReader(open(units_tsv, encoding="utf-8"), delimiter="\t"):
        if u["reachable"] == "1" and set(u["classes"].split(";")) & set(PRIMARY):
            t[u["doc_id"]].update(u["gene_entrez"].split(";"))   # with PT3 ortholog ids (units.finish)
    return t


def main() -> None:
    size, tol, rule = sizer()
    cand = pools(size)
    ct = cred_targets()
    jobs = [(f"cred_{{c}}_f{k}", load_docs(CV / f"test_f{k}.json"), ct) for k in range(5)]
    if (P2 / "clingen_units.tsv").exists() and (P2 / "clingen_docs.json").exists():
        jobs.append(("clingen_{c}", load_docs(P2 / "clingen_docs.json"), unit_targets(P2 / "clingen_units.tsv")))
    else:
        print("data/paper2/clingen_{units.tsv,docs.json} not there yet: ClinGen swaps skipped")
    OUT.mkdir(parents=True, exist_ok=True)
    counts, mapping = {}, [["file", "doc_id", "orig_doc_id", "gene", "surface", "replacement"]]
    for name, docs, targets in jobs:
        for c in ("u", "k"):
            f = name.format(c=c)
            out, rows, counts[f] = swap_docs(docs, targets, cand[c], size, tol)
            json.dump(out, open(OUT / f"{f}.json", "w", encoding="utf-8"), ensure_ascii=False)
            mapping += [[f, *r] for r in rows]
            print(f, dict(counts[f]))
    with open(OUT / "mapping.tsv", "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, delimiter="\t", lineterminator="\n").writerows(mapping)
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.name != "manifest.json"}
    (OUT / "manifest.json").write_text(json.dumps(
        {"what": "paper 2 E2 name swaps (swap_build.py)", "match": rule, "primary": PRIMARY,
         "pool_sizes": {c: sum(map(len, p.values())) for c, p in cand.items()}, "counts": counts, "sha256": sha},
        indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
