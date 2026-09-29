"""Paper 2 (E1, E5 external): RGD human manual gene-disease annotations -> units + PT3 docs
(docs/research/paper2/external_tests.md §4).

    PYTHONPATH=. .venv/bin/python docs/results/paper2/build_rgd.py
      reads  data/external/rgd/{rgd_human_manual.tsv.gz,RDO.obo}, data/external/doid/doid.obo,
             data/external/research_2026-09-17/{hgnc.txt,mondo.sssom.tsv}
      caches data/PT3/raw/<pmid>.json   (every unit PMID; resumable)
      writes data/paper2/rgd_docs.json, rgd_units.tsv, rgd_manifest.json   (unit format: units.py)

Unit = (PMID, gene Entrez id from the HGNC symbol, MeSH set of the DOID/RDO term); classes = evidence codes of its rows,
NOT_<code> for qualifier no_association. Rows with other qualifiers (treatment, severity, disease_progression, ...) are
dropped, as in external_reach.py.
"""
from __future__ import annotations

import csv
import gzip
from collections import Counter, defaultdict

from units import EXT, doid_mesh, finish, hgnc

SRC = EXT / "rgd/rgd_human_manual.tsv.gz"
QUALIFIERS = ("", "susceptibility", "no_association")


def main() -> None:
    entrez, dm = {s: e for s, e in hgnc().values()}, doid_mesh()
    by, skipped = defaultdict(list), Counter()
    for r in csv.DictReader(gzip.open(SRC, "rt", encoding="utf-8"), delimiter="\t"):
        if r["qualifier"] not in QUALIFIERS:
            skipped["qualifier"] += 1
        elif r["symbol"] not in entrez:
            skipped["symbol_not_hgnc"] += 1
        else:
            by[(r["pmid"], entrez[r["symbol"]], tuple(sorted(dm.get(r["term"], ()))))].append(r)
    units = [{"unit_id": f"rgd:{p}:{g}:{'+'.join(m) or '-'}", "source": "rgd", "pmid": p, "gene_entrez": [g],
              "disease_mesh": list(m),
              "classes": sorted({("NOT_" if r["qualifier"] == "no_association" else "") + r["evidence"] for r in rows}),
              "extra": {"doid": sorted({r["term"] for r in rows}), "qualifiers": sorted({r["qualifier"] for r in rows}),
                        "date_min": min(r["date"] for r in rows), "n_rows": len(rows)}}
             for (p, g, m), rows in by.items()]
    finish("rgd", units, rows_skipped=dict(skipped))


if __name__ == "__main__":
    main()
