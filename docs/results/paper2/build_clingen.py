"""Paper 2 (E1, E5): ClinGen Gene-Disease Validity evidence rows -> units + PT3 docs (docs/research/paper2/clingen.md).

    PYTHONPATH=. .venv/bin/python docs/results/paper2/build_clingen.py
      reads  data/external/clingen_gdv_2026-09-23/{evidence,curations}.tsv
             data/external/research_2026-09-17/{hgnc.txt,mondo.sssom.tsv}
      caches data/PT3/raw/<pmid>.json   (every unit PMID, ~24k; resumable)
      writes data/paper2/clingen_docs.json, clingen_units.tsv, clingen_manifest.json   (unit format: units.py)

Unit = (PMID, curated gene's Entrez id) over the evidence rows with a PMID; disease_mesh = union of MONDO -> MeSH
(exactMatch) over the unit's rows; classes = row_class of its rows.
"""
from __future__ import annotations

import csv
from collections import defaultdict

from units import EXT, finish, hgnc, mondo_mesh

SRC = EXT / "clingen_gdv_2026-09-23"
# category, or (category, subcategory) where the subcategory splits it; an unlisted Model Systems subtype fails loudly
CLASS = {"Case-level variant": "GEN_CLV", "Segregation": "GEN_SEG", "Case-control": "GEN_CC",
         "Expression": "EXP_U", ("Expression", "A"): "EXP_A", ("Expression", "B"): "EXP_B",
         "Functional Alteration": "FA_U", ("Functional Alteration", "Patient cells"): "FA_PAT",
         ("Functional Alteration", "Non-patient cells"): "FA_NONPAT",
         ("Model Systems", "Non-human model organism"): "MS_ORG", ("Model Systems", "Animal model"): "MS_ORG",
         ("Model Systems", "Cell culture model"): "MS_CELL",
         "Rescue": "RES", "Biochemical Function": "BIO", "Protein Interaction": "PI", "Non-scorable": "NONSCOR"}


def row_class(r: dict) -> str:
    """Unit class of one evidence row; a row whose score_status is Contradicts is CONTRA whatever its category."""
    if r["score_status"] == "Contradicts":
        return "CONTRA"
    return CLASS.get((r["category"], r["subcategory"])) or CLASS[r["category"]]


def main() -> None:
    hg, mm = hgnc(), mondo_mesh()
    gcep = {r["curation_id"]: r["gcep"]
            for r in csv.DictReader(open(SRC / "curations.tsv", encoding="utf-8"), delimiter="\t")}
    by, no_entrez = defaultdict(list), 0
    for r in csv.DictReader(open(SRC / "evidence.tsv", encoding="utf-8"), delimiter="\t"):
        if r["pmid"] and r["hgnc_id"] in hg:
            by[(r["pmid"], hg[r["hgnc_id"]][1])].append(r)
        elif r["pmid"]:
            no_entrez += 1
    units = [{"unit_id": f"clingen:{p}:{g}", "source": "clingen", "pmid": p, "gene_entrez": [g],
              "disease_mesh": sorted(set().union(*(mm.get(r["mondo_id"], set()) for r in rows))),
              "classes": sorted({row_class(r) for r in rows}),
              "extra": {"curation_ids": sorted({r["curation_id"] for r in rows}),
                        "classifications": sorted({r["classification"] for r in rows}),
                        "approval_date_min": min(r["approval_date"] for r in rows),
                        "gcep": sorted({gcep[r["curation_id"]] for r in rows}), "n_rows": len(rows)}}
             for (p, g), rows in by.items()]
    finish("clingen", units, n_rows_pmid_no_entrez=no_entrez)


if __name__ == "__main__":
    main()
