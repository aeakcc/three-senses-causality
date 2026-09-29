"""Paper 2, E4: Percha & Altman 2018 gene-disease sentences -> one-sentence Docs per construction class.

    PYTHONPATH=. .venv/bin/python docs/results/paper2/build_percha.py      (~8 min, two passes over part-ii)
      reads  data/external/percha2018/part-{i,ii}-*.txt.gz  (parsed by docs/research/paper2/percha_sample.py)
      writes data/paper2/percha_docs.json, percha_units.tsv (unit format: units.py), percha_manifest.json,
             docs/research/paper2/percha_constructions_sample.md (15 random units per class, for the manual check)

Construction class = lexical trigger family among the lowercase words (heads and dependents) of the dependency path,
start_entity/end_entity excluded, whole-word match (a word equals a trigger; hyphenated words are single words).
EXPR is dropped when UP or DOWN is present, MUT + CAUSE -> MUT; then exactly one family -> that class, several ->
MULTI, none -> NONE (neither sampled). Records: human gene (single Entrez id), MESH:D/C disease, PMID not in
BioRED/BC8/CRED, <= 80 tokens, both entity strings whole tokens of the sentence. Per class up to N_PER_CLASS
sentences: paths in random order, one sentence per path, then a second and third per path while short
(random.Random(0)); a sentence already taken by an earlier class is skipped (its doc id would collide).
Doc: text = title = detokenized sentence, abstract = ''; doc 'pmid' = PMID + 4-digit sentence number.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict

from units import OUT, ROOT, D, G, write_units  # units -> pt3_docs puts ROOT on sys.path

sys.path.insert(0, str(ROOT / "docs/research/paper2"))
from biocausal.data import build_entities, check_doc, make_pairs, read_pmids  # noqa: E402
from percha_sample import PART_I, PART_II, THEMES, argmax_theme, load_themes, records  # noqa: E402
from pt3_docs import SOURCES  # noqa: E402

MD = ROOT / "docs/research/paper2/percha_constructions_sample.md"
MANUAL = "## Manual check"
N_PER_CLASS, PER_PATH, MAX_TOKENS, SEED = 600, 3, 80, 0
FAMILIES = {k: frozenset(v.split()) for k, v in {
    "MUT": "mutation mutations mutant mutants mutated variant variants deletion deletions truncating nonsense missense "
           "frameshift germline biallelic heterozygous homozygous loss-of-function",
    "CAUSE": "cause causes caused causing causative underlie underlies underlying responsible",
    "UP": "overexpressed overexpression upregulated up-regulated upregulation elevated increased amplified "
          "amplification hyperexpression",
    "DOWN": "downregulated down-regulated downregulation decreased reduced silenced silencing hypermethylated "
            "hypermethylation underexpressed",
    "EXPR": "expressed expression levels level mrna immunoreactivity staining",
    "ASSOC": "associated association associations linked correlated correlation correlates",
    "MARKER": "marker markers biomarker biomarkers prognostic predictor",
    "RISK": "polymorphism polymorphisms allele alleles snp snps susceptibility risk genotype genotypes",
    "THERAPY": "target targets targeting inhibitor inhibitors inhibition therapy treatment therapeutic",
}.items()}
PTB = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}", "``": '"', "''": '"'}


def construction(path: str) -> str:
    """Construction class of a Percha dependency path ('head|rel|dep' edges, any case)."""
    words = {w for e in path.lower().split() for w in e.split("|")[::2]} - {"start_entity", "end_entity"}
    fams = {c for c, trig in FAMILIES.items() if words & trig}
    if fams & {"UP", "DOWN"}:
        fams.discard("EXPR")
    if {"MUT", "CAUSE"} <= fams:
        fams.discard("CAUSE")
    return fams.pop() if len(fams) == 1 else "MULTI" if fams else "NONE"


def sentence_doc(doc_id: str, sentence: str, gene: str, gene_id: str, disease: str, mesh: str) -> dict | None:
    """One-sentence Doc: PTB brackets restored, '_' -> ' ' in every token (PubTator glues every multi-word entity;
    a lone '_' token, as in "Alzheimer 's _ disease", is dropped), tokens re-joined by single spaces; mentions = every
    token equal to the formatted gene / disease string. None when either string is not a whole token (or both are
    the same string)."""
    text, mentions = "", []
    for tok in sentence.split():
        word = " ".join(PTB.get(tok, tok).replace("_", " ").split())
        if not word:
            continue
        start = len(text) + bool(text)
        text += (" " if text else "") + word
        for fmt, typ, eid in ((gene, G, gene_id), (disease, D, mesh)):
            if tok == fmt:
                mentions.append({"start": start, "end": len(text), "text": word, "type": typ, "ids": [eid]})
    if gene == disease or {m["type"] for m in mentions} != {G, D}:
        return None
    ents = build_entities(mentions)
    doc = {"pmid": doc_id, "title": text, "abstract": "", "text": text, "mentions": mentions, "entities": ents,
           "pairs": make_pairs(ents, []), "relations": []}
    check_doc(doc)
    return doc


def eligible(r: dict, held: set[str]) -> bool:
    toks = r["sentence"].split()
    return (r["pmid"] not in held and r["e1_type"] == "Gene" and r["e1_id"].isdigit()
            and re.fullmatch(r"MESH:[CD]\d+", r["e2_id"]) is not None and len(toks) <= MAX_TOKENS
            and r["e1_fmt"] != r["e2_fmt"] and r["e1_fmt"] in toks and r["e2_fmt"] in toks)


def select(res: dict[str, dict[str, list]], rng: random.Random) -> dict[int, str]:
    """{record index: class}. res[class][path] = up to PER_PATH (random key, record index, sentence id), key-sorted,
    so position 0 is a uniformly random eligible sentence of the path."""
    chosen, used = {}, set()
    for c in FAMILIES:
        paths = sorted(res.get(c, {}))
        order = rng.sample(paths, len(paths))
        n = 0
        for _, i, sid in (res[c][p][k] for k in range(PER_PATH) for p in order if k < len(res[c][p])):
            if n == N_PER_CLASS:
                break
            if sid not in used:
                used.add(sid)
                chosen[i] = c
                n += 1
    return chosen


def build(part_i=PART_I, part_ii=PART_II, exclude=("biored", "bc8", "cred")):
    held = set().union(*(read_pmids(ROOT / f) for n in exclude for f in SOURCES[n]))
    rng = random.Random(SEED)
    res: dict[str, dict[str, list]] = defaultdict(dict)
    n_rec = Counter()
    for i, r in enumerate(records(part_ii)):   # pass 1: per (class, path) keep the PER_PATH smallest random keys
        if not eligible(r, held):
            continue
        key = r["path"].lower()
        c = construction(key)
        if c in ("NONE", "MULTI"):
            continue
        n_rec[c] += 1
        slot = res[c].setdefault(key, [])
        bisect.insort(slot, (rng.random(), i, int(r["pmid"]) * 10000 + int(r["sent_no"])))
        del slot[PER_PATH:]
    chosen = select(res, rng)
    themes = load_themes(part_i)
    docs, units = [], []
    last = max(chosen, default=-1)
    for i, r in enumerate(records(part_ii)):   # pass 2: build the chosen records
        if i > last:
            break
        if i not in chosen:
            continue
        assert int(r["sent_no"]) < 10000, r
        doc_id = r["pmid"] + f"{int(r['sent_no']):04d}"
        mesh = r["e2_id"].removeprefix("MESH:")
        doc = sentence_doc(doc_id, r["sentence"], r["e1_fmt"], r["e1_id"], r["e2_fmt"], mesh)
        assert doc is not None, r   # eligible() guarantees both whole tokens
        sup, flags = themes[r["path"].lower()]
        t = argmax_theme(sup)
        extra = {"theme": THEMES[t] if t is not None else "tie", "path": r["path"], "sentence": doc["text"],
                 "flagship": THEMES[flags.index(1)] if 1 in flags else "", "gene": r["e1_raw"], "disease": r["e2_raw"]}
        docs.append(doc)
        units.append({"unit_id": f"percha:{doc_id}", "source": "percha", "doc_id": doc_id, "pmid": r["pmid"],
                      "gene_entrez": [r["e1_id"]], "disease_mesh": [mesh], "classes": [chosen[i]], "reachable": 1,
                      "disease_matched": 1, "extra": extra})
    order = sorted(range(len(docs)), key=lambda j: int(docs[j]["pmid"]))
    docs, units = [docs[j] for j in order], [units[j] for j in order]
    sampled = Counter(u["classes"][0] for u in units)
    dpaths = {c: len({u["extra"]["path"].lower() for u in units if u["classes"] == [c]}) for c in FAMILIES}
    table = {c: {"paths": len(res.get(c, {})), "records": n_rec[c], "sampled": sampled[c],
                 "distinct_paths": dpaths[c]} for c in FAMILIES}
    manifest = {"source": "Percha & Altman 2018, Zenodo 10.5281/zenodo.3459420 (CC BY 4.0)", "exclude": list(exclude),
                "seed": SEED, "n_per_class": N_PER_CLASS, "per_path": PER_PATH, "max_tokens": MAX_TOKENS,
                "per_class": table, "n_docs": len(docs),
                "doc_ids_sha256": hashlib.sha256("\n".join(d["pmid"] for d in docs).encode()).hexdigest()}
    return docs, units, manifest


def sample_md(units: list[dict], n=15) -> str:
    rng = random.Random(SEED)
    out = ["# Percha construction classes: random sample for the manual check", "",
           f"Generated by `docs/results/paper2/build_percha.py` from `data/paper2/percha_units.tsv`: {n} random units "
           "per class (random.Random(0)). Gene and disease are the two entities of the unit's pair.", ""]
    for c in FAMILIES:
        us = [u for u in units if u["classes"] == [c]]
        out += [f"## {c} ({min(n, len(us))} of {len(us)})", ""]
        for k, u in enumerate(rng.sample(us, min(n, len(us))), 1):
            x = u["extra"]
            out += [f"{k}. `{u['doc_id']}` gene **{x['gene']}** ({u['gene_entrez'][0]}), disease **{x['disease']}** "
                    f"({u['disease_mesh'][0]}); path `{x['path']}`", f"   > {x['sentence']}", ""]
    return "\n".join(out)


def main() -> None:
    docs, units, manifest = build()
    OUT.mkdir(parents=True, exist_ok=True)
    fd, fu = OUT / "percha_docs.json", OUT / "percha_units.tsv"
    with open(fd, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    write_units(fu, units)
    manifest["sha256"] = {f.name: hashlib.file_digest(open(f, "rb"), "sha256").hexdigest() for f in (fd, fu)}
    (OUT / "percha_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    old = MD.read_text(encoding="utf-8") if MD.exists() else ""   # keep the hand-written check below MANUAL
    MD.write_text(sample_md(units) + (old[old.find(MANUAL) - 1:] if MANUAL in old else ""), encoding="utf-8")
    print(f"{'class':8} {'paths':>8} {'records':>10} {'sampled':>8} {'dist.paths':>10}")
    for c, v in manifest["per_class"].items():
        print(f"{c:8} {v['paths']:>8} {v['records']:>10} {v['sampled']:>8} {v['distinct_paths']:>10}")
    print("docs", manifest["n_docs"])
    for c in FAMILIES:
        for u in [u for u in units if u["classes"] == [c]][:3]:
            print(f"  {c:8} {u['extra']['sentence']}")


if __name__ == "__main__":
    main()
