"""Paper 2, E5 (sense V): the evidence-type sets of the 'ev' head, from the single-candidate-pair units of e1_sets.py.

    PYTHONPATH=. .venv/bin/python docs/results/paper2/e5_sets.py
      reads  data/paper2/e1/{clingen,rgd}_{units.tsv,docs.json},
             data/external/clingen_gdv_2026-09-23/{evidence,curations}.tsv, sealed_strict_ids.txt
      writes data/paper2/e5/clingen.json (train Docs, the stem makes train_re's corpus 'clingen'), train_units.tsv,
             test_2026_{docs.json,units.tsv}, rgd_ext_{docs.json,units.tsv}, manifest.json

Classes = biocausal.data.EV_CLASSES. ClinGen unit classes: GEN_CLV, GEN_SEG, GEN_CC -> genetic; EXP_A/B/U -> expression;
MS_ORG, MS_CELL, RES -> model_rescue; FA_PAT, FA_NONPAT -> functional_alteration; BIO, PI -> biochemical_interaction;
NONSCOR and CONTRA ignored (a unit with only those is excluded). PMID split over ALL rows of evidence.tsv (approval_date
of curations.tsv): TRAIN if every curation citing the PMID was approved before 2026-01-01, TEST_2026 if every one is in
sealed_strict_ids.txt, else dropped. A train unit with more than one class is dropped; a test unit keeps all its
classes. Each train unit's single candidate pair gets {"label": "None", "ev": EV_CLASSES index}; a pair two units of one
doc give different classes loses 'ev' and both units leave train_units.tsv. RGD_EXT = e1 RGD units with IEP ->
expression, IMP / IGI -> model_rescue, IAGP -> genetic (other codes ignored, multi-label), no TRAIN PMID. Unit TSVs:
classes = coarse classes, extra.fine_classes = the e1 classes, extra.pair = [e1, e2] (the single candidate pair).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from biocausal.data import EV_CLASSES  # noqa: E402
from units import candidate_pairs, read_units, write_units  # noqa: E402

E1, OUT = ROOT / "data/paper2/e1", ROOT / "data/paper2/e5"
SRC = ROOT / "data/external/clingen_gdv_2026-09-23"
CUTOFF = "2026-01-01"
IGNORED = {"NONSCOR", "CONTRA"}
CLINGEN = {**dict.fromkeys(("GEN_CLV", "GEN_SEG", "GEN_CC"), "genetic"),
           **dict.fromkeys(("EXP_A", "EXP_B", "EXP_U"), "expression"),
           **dict.fromkeys(("MS_ORG", "MS_CELL", "RES"), "model_rescue"),
           **dict.fromkeys(("FA_PAT", "FA_NONPAT"), "functional_alteration"),
           **dict.fromkeys(("BIO", "PI"), "biochemical_interaction")}
RGD = {"IEP": "expression", "IMP": "model_rescue", "IGI": "model_rescue", "IAGP": "genetic"}


def pmid_split(evidence: list[dict], approval: dict[str, str], strict: set[str]) -> dict[str, str]:
    """PMID -> 'train' / 'test_2026' / 'dropped' over every evidence row with a PMID."""
    cites = defaultdict(set)
    for r in evidence:
        if r["pmid"]:
            cites[r["pmid"]].add(r["curation_id"])
    return {p: "train" if all(approval[c] < CUTOFF for c in cs) else "test_2026" if cs <= strict else "dropped"
            for p, cs in cites.items()}


def coarse(u: dict, table: dict[str, str], strict: bool) -> dict:
    """The unit with coarse classes (EV_CLASSES order); ``strict``: a class neither in ``table`` nor IGNORED fails."""
    got = {table[c] for c in u["classes"] if c in table or (strict and c not in IGNORED)}
    return {**u, "classes": sorted(got, key=EV_CLASSES.index), "extra": {**u["extra"], "fine_classes": u["classes"]}}


def with_pair(units: list[dict], docs: dict[str, dict]) -> list[dict]:
    out = []
    for u in units:
        pairs, _ = candidate_pairs(docs[u["doc_id"]], u["gene_entrez"], u["disease_mesh"])
        assert len(pairs) == 1, u["unit_id"]      # e1 sets hold single-candidate-pair units only
        out.append({**u, "extra": {**u["extra"], "pair": list(pairs[0])}})
    return out


def label_docs(units: list[dict], docs: dict[str, dict]) -> tuple[list[dict], list[dict], int]:
    """(train Docs with 'ev' on the units' pairs, the units whose pair kept 'ev', pairs dropped for a class conflict)."""
    want = defaultdict(set)
    for u in units:
        want[(u["doc_id"], *u["extra"]["pair"])].add(u["classes"][0])
    ok = {k: EV_CLASSES.index(next(iter(v))) for k, v in want.items() if len(v) == 1}
    out = []
    for i in sorted({k[0] for k in ok}, key=int):
        d = docs[i]
        pairs = [{"e1": p["e1"], "e2": p["e2"], "label": "None",
                  **({"ev": ok[(i, p["e1"], p["e2"])]} if (i, p["e1"], p["e2"]) in ok else {})} for p in d["pairs"]]
        out.append({**d, "pairs": pairs})
    return out, [u for u in units if (u["doc_id"], *u["extra"]["pair"]) in ok], len(want) - len(ok)


def counts(units: list[dict]) -> dict:
    return {"n_units": len(units), "n_docs": len({u["doc_id"] for u in units}),
            "n_pmids": len({u["pmid"] for u in units}), "n_multi_label": sum(len(u["classes"]) > 1 for u in units),
            "units_per_class": {c: sum(c in u["classes"] for u in units) for c in EV_CLASSES},
            "single_class_units_per_class": {c: sum(u["classes"] == [c] for u in units) for c in EV_CLASSES}}


def main() -> None:
    rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"))  # noqa: E731
    csv.field_size_limit(sys.maxsize)
    approval = {r["curation_id"]: r["approval_date"] for r in rd(SRC / "curations.tsv")}
    split = pmid_split(rd(SRC / "evidence.tsv"), approval, set((SRC / "sealed_strict_ids.txt").read_text().split()))
    load = lambda s: {d["pmid"]: d for d in json.loads((E1 / f"{s}_docs.json").read_text(encoding="utf-8"))}  # noqa
    cg_docs, rgd_docs = load("clingen"), load("rgd")
    cg = with_pair([coarse(u, CLINGEN, True) for u in read_units(E1 / "clingen_units.tsv")], cg_docs)
    rgd = with_pair([coarse(u, RGD, False) for u in read_units(E1 / "rgd_units.tsv")], rgd_docs)
    status = lambda u: "no_class" if not u["classes"] else "multi_class" if len(u["classes"]) > 1 else "single_class"  # noqa
    drop = Counter(f"clingen_{split[u['pmid']]}_{status(u)}" for u in cg)
    train = [u for u in cg if split[u["pmid"]] == "train" and len(u["classes"]) == 1]
    test = [u for u in cg if split[u["pmid"]] == "test_2026" and u["classes"]]
    train_docs, train, drop["train_pairs_class_conflict"] = label_docs(train, cg_docs)
    drop["train_units_class_conflict"] = drop["clingen_train_single_class"] - len(train)
    rgd_ext = [u for u in rgd if u["classes"] and split.get(u["pmid"]) != "train"]
    drop.update(rgd_no_mapped_class=sum(not u["classes"] for u in rgd),
                rgd_train_pmid=sum(bool(u["classes"]) and split.get(u["pmid"]) == "train" for u in rgd))
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"clingen.json": train_docs, "train_units.tsv": train,
             "test_2026_docs.json": [cg_docs[i] for i in sorted({u["doc_id"] for u in test}, key=int)],
             "test_2026_units.tsv": test,
             "rgd_ext_docs.json": [rgd_docs[i] for i in sorted({u["doc_id"] for u in rgd_ext}, key=int)],
             "rgd_ext_units.tsv": rgd_ext}
    for name, v in files.items():
        if name.endswith(".tsv"):
            write_units(OUT / name, v)
        else:
            (OUT / name).write_text(json.dumps(v, ensure_ascii=False) + "\n", encoding="utf-8")
    man = {"rule": __doc__.split("\n\n")[2].strip(), "ev_classes": EV_CLASSES, "cutoff": CUTOFF,
           "pmids": dict(Counter(split.values())), "dropped": dict(sorted(drop.items())),
           "splits": {"train": {**counts(train), "n_ev_pairs": sum("ev" in p for d in train_docs for p in d["pairs"])},
                      "test_2026": counts(test), "rgd_ext": counts(rgd_ext)},
           "sha256": {n: hashlib.sha256((OUT / n).read_bytes()).hexdigest() for n in files}}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k != "rule"}, indent=1))


if __name__ == "__main__":
    main()
