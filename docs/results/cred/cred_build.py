"""CRED (Bansal et al., bioRxiv 2024.09.17.613424 v2) -> project Doc schema with silver PubTator3 entities
(preregistration §8.3). No training on CRED; pair labels in the docs are dummy "None".

    PYTHONPATH=. .venv/bin/python docs/results/cred/cred_build.py
      reads  data/CRED/source/{new_train_data,val_data,test_data,only_augmented_data.tsv}  (github BIRDSgroup/CRED)
      caches data/CRED/raw/<pmid>.json   (PubTator3 export/biocjson document, verbatim)
      writes data/CRED/cred_docs.json    (Doc schema, biocausal.data)
             data/CRED/cred_pairs.tsv    (one row per unique non-augmented CRED pair -> project entity ids)
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from biocausal.data import build_entities, canonical_pair, check_doc, make_pairs, pmid_held, read_pmids

ROOT = Path(__file__).resolve().parents[3]
CRED = ROOT / "data/CRED"
API = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson?pmids="
SPLITS = {"train": "new_train_data", "val": "val_data", "test": "test_data"}
# PubTator3 type -> BioRED type; anything else fails loudly
TYPES = {"Gene": "GeneOrGeneProduct", "Disease": "DiseaseOrPhenotypicFeature", "Chemical": "ChemicalEntity",
         "Species": "OrganismTaxon", "CellLine": "CellLine", "Variant": "SequenceVariant",
         "DNAMutation": "SequenceVariant", "ProteinMutation": "SequenceVariant", "SNP": "SequenceVariant",
         "DNAAllele": "SequenceVariant", "ProteinAllele": "SequenceVariant", "AcidChange": "SequenceVariant",
         "OtherMutation": "SequenceVariant", "Chromosome": None,   # BioRED has no chromosome type: mention dropped
         "RefSeq": None, "GenomicRegion": None}   # transcript accessions (NM_000208.2), regions: no BioRED type either
LEAK_FILES = ["data/BioRED/Train.BioC.JSON", "data/BioRED/Dev.BioC.JSON", "data/BioRED/Test.BioC.JSON",
              "data/BC8/bc8_test.json"]
csv.field_size_limit(sys.maxsize)


def split_ids(identifier: str | None) -> list[str]:
    """PubTator3 / CRED id -> BioRED-style ids: composite ids split on ';' / ',', 'MESH:' dropped, '-' or empty -> []."""
    out = []
    for x in str(identifier or "").replace(",", ";").split(";"):
        x = x.strip().removeprefix("MESH:")
        if x and x != "-" and x not in out:
            out.append(x)
    return out


def variant_id(infons: dict) -> list[str]:
    """PubTator3 variant identifier ('tmVar:c|SUB|C|1|T;HGVS:...;RS#:123;...') -> BioRED style: rs id, else tmVar."""
    parts = dict(p.split(":", 1) for p in str(infons.get("identifier") or "").split(";") if ":" in p)
    if parts.get("RS#"):
        return ["rs" + parts["RS#"]]
    return [parts["tmVar"]] if parts.get("tmVar") else []


def fetch(pmids: list[str]) -> None:
    """Cache every missing PMID as data/CRED/raw/<pmid>.json (batches of 100, 1 s apart)."""
    raw = CRED / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    todo = [p for p in pmids if not (raw / f"{p}.json").exists()]
    for i in range(0, len(todo), 100):
        data = json.loads(urllib.request.urlopen(API + ",".join(todo[i:i + 100]), timeout=120).read())
        for d in data["PubTator3"]:
            (raw / f"{d['id']}.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        time.sleep(1)


def to_doc(bd: dict) -> tuple[dict, dict[frozenset, str]]:
    """PubTator3 BioC document -> (Doc with dummy None pair labels, {frozenset(ids): PubTator3 relation type})."""
    text = ""
    mentions = []
    for p in bd["passages"]:
        assert p["infons"].get("type") in ("title", "abstract"), (bd["id"], p["infons"].get("type"))
        text = text.ljust(p["offset"]) + p["text"]
        for a in p["annotations"]:
            t = TYPES[a["infons"]["type"]]
            if t is None:
                continue
            ids = variant_id(a["infons"]) if t == "SequenceVariant" else split_ids(a["infons"].get("identifier"))
            for loc in a["locations"]:
                s = text[loc["offset"]:loc["offset"] + loc["length"]]
                # offsets validated against the text; PubTator3 may keep a '\' where its passage has a space ("and\or",
                # 2 of ~45k cached abstracts), the mention then takes the passage's text
                assert s in (a["text"], a["text"].replace("\\", " ")), (bd["id"], a["text"], s)
                mentions.append({"start": loc["offset"], "end": loc["offset"] + loc["length"], "text": s,
                                 "type": t, "ids": ids or ["-"]})
    mentions.sort(key=lambda m: (m["start"], -m["end"]))
    title = bd["passages"][0]["text"]
    entities = build_entities(mentions)
    doc = {"pmid": str(bd["id"]), "title": title, "abstract": text[len(title) + 1:], "text": text,
           "mentions": mentions, "entities": entities, "pairs": make_pairs(entities, []), "relations": []}
    check_doc(doc)
    rels: dict[frozenset, str] = {}
    for r in bd.get("relations", []):
        i1, i2 = split_ids(r["infons"]["role1"]["identifier"]), split_ids(r["infons"]["role2"]["identifier"])
        for a in i1:
            for b in i2:
                rels[frozenset((a, b))] = r["infons"]["type"]
    return doc, rels


def read_cred() -> tuple[dict[tuple, dict], list[tuple]]:
    """Unique non-augmented (pmid, gene id, disease id) -> {label, split, same_sent, n_rows}; plus label conflicts.

    Non-augmented = rows not identical in (index, id1, id2, sentence) to a row of only_augmented_data.tsv (the train
    file is original + augmented rows). Exact duplicate keys collapse to one pair; keys with contradicting labels are
    returned separately and left out of the evaluation.
    """
    aug = {(r["index"], r["id1"], r["id2"], r["sentence"])
           for r in csv.DictReader(open(CRED / "source/only_augmented_data.tsv", encoding="utf-8"), delimiter="\t")}
    rows = defaultdict(list)
    for split, fname in SPLITS.items():
        for r in csv.DictReader(open(CRED / "source" / fname, encoding="utf-8"), delimiter="\t"):
            if (r["index"], r["id1"], r["id2"], r["sentence"]) not in aug:
                rows[(r["index"], r["id1"], r["id2"])].append((split, r))
    pairs, conflicts = {}, []
    for key, rs in rows.items():
        labels = {r["label"] for _, r in rs}
        splits = {s for s, _ in rs}
        same = {r["is_in_same_sent"] for _, r in rs}
        assert len(same) == 1 and labels <= {"0", "1"}, (key, same, labels)
        if len(labels) > 1:
            conflicts.append(key)
            continue
        # a key in several split files (PMID 36348357 sits in val and test) counts as test, the authors' test file
        pairs[key] = {"label": int(labels.pop()), "split": next(s for s in ("test", "val", "train") if s in splits),
                      "same_sent": same.pop() == "True", "n_rows": len(rs), "in_splits": "+".join(sorted(splits))}
    return pairs, conflicts


def main() -> None:
    pairs, conflicts = read_cred()
    pmids = sorted({k[0] for k in pairs} | {k[0] for k in conflicts}, key=int)
    held = set().union(*(read_pmids(ROOT / f) for f in LEAK_FILES))
    leak = [p for p in pmids if pmid_held(p, held)]
    assert not leak, f"CRED PMIDs in BioRED/BC8: {leak}"
    fetch(pmids)
    docs, rels = {}, {}
    for p in pmids:
        docs[p], rels[p] = to_doc(json.loads((CRED / "raw" / f"{p}.json").read_text(encoding="utf-8")))

    out = []
    reach = Counter()
    for (pmid, gid, did), v in sorted(pairs.items(), key=lambda kv: (int(kv[0][0]), kv[0][1], kv[0][2])):
        ents = docs[pmid]["entities"]
        genes, dis = split_ids(gid), split_ids(did)
        assert len(dis) == 1, did
        ok_g = [g for g in genes if ents.get(g, {}).get("type") == "GeneOrGeneProduct"]
        ok_d = ents.get(dis[0], {}).get("type") == "DiseaseOrPhenotypicFeature"
        status = "reachable" if ok_d and len(ok_g) == len(genes) else \
            "no_gene" if ok_d else "no_disease" if len(ok_g) == len(genes) else "no_both"
        # composite CRED gene ids ('5595;5594'): one candidate pair per component gene, all must be reachable
        cand = [canonical_pair(dis[0], "DiseaseOrPhenotypicFeature", g, "GeneOrGeneProduct") for g in genes] \
            if status == "reachable" else []
        pt3 = sorted({rels[pmid][frozenset((g, dis[0]))] for g in genes if frozenset((g, dis[0])) in rels[pmid]})
        reach[(status, v["label"])] += 1
        out.append({"pmid": pmid, "gene_id": gid, "disease_id": did, "label": v["label"], "split": v["split"],
                    "same_sent": int(v["same_sent"]), "n_rows": v["n_rows"], "in_splits": v["in_splits"],
                    "status": status,
                    "pairs": ";".join(f"{a}|{b}" for a, b in cand), "pt3_type": ";".join(pt3)})
    json.dump(list(docs.values()), open(CRED / "cred_docs.json", "w", encoding="utf-8"), ensure_ascii=False)
    with open(CRED / "cred_pairs.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(out)

    ds = list(docs.values())
    print(f"PMIDs {len(pmids)}, docs {len(ds)}, mentions {sum(len(d['mentions']) for d in ds)}, "
          f"entities {sum(len(d['entities']) for d in ds)}, candidate pairs {sum(len(d['pairs']) for d in ds)}, "
          f"leak vs BioRED+BC8: {len(leak)}")
    print("mention types", dict(Counter(m["type"] for d in ds for m in d["mentions"])))
    print(f"CRED pairs {len(out)} (label conflicts dropped: {len(conflicts)} {conflicts}), "
          f"labels {dict(Counter(r['label'] for r in out))}, splits {dict(Counter(r['in_splits'] for r in out))}")
    print("status x label", dict(sorted(reach.items())))
    print("pt3 types on CRED pairs", dict(Counter(r["pt3_type"] for r in out)))


if __name__ == "__main__":
    main()
