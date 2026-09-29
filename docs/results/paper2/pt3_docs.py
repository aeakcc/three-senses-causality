"""Paper 2: PMIDs -> Doc-schema JSON with silver PubTator3 entities, leak-filtered (paper2/codemap.md §1.1).

    PYTHONPATH=. .venv/bin/python docs/results/paper2/pt3_docs.py --pmids-file PMIDS.txt --out data/X/x_docs.json
      caches data/PT3/raw/<pmid>.json   (causal_audit.fetch_pt3: batches of 100, 1 s apart, {"missing": true} kept)
      writes OUT (JSON list of Doc, pair labels dummy "None") + OUT's .manifest.json

PMIDs of BioRED Train/Dev/Test, BC8 and CRED are dropped before fetching: score.py refuses a whole file with one PMID
a run trained on. The manifest's pmids_sha256 is over the written documents' PMIDs, sorted as integers, "\n"-joined.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "docs/results/cred"), str(ROOT / "docs/results/posthoc")]

from biocausal.data import read_pmids  # noqa: E402
from causal_audit import PT3_DIR, fetch_pt3  # noqa: E402
from cred_build import LEAK_FILES, to_doc  # noqa: E402

# cred_docs.json holds every CRED PMID (cred_pairs.tsv's 266 plus label-conflict keys, which are the same PMIDs)
SOURCES = {"biored": LEAK_FILES[:3], "bc8": LEAK_FILES[3:], "cred": ["data/CRED/cred_docs.json"]}


def build_docs(pmids: list[str] | set[str],
               exclude: tuple[str, ...] = ("biored", "bc8", "cred")) -> tuple[list[dict], dict]:
    """(Docs in PMID order, manifest) for ``pmids`` minus the PMIDs of the ``exclude`` corpora (SOURCES keys); PMIDs
    PubTator3 does not return, or returns without abstract text, are skipped and counted."""
    held = set().union(*(read_pmids(ROOT / f) for n in exclude for f in SOURCES[n]))
    pmids = sorted(set(pmids), key=int)
    keep = [p for p in pmids if p not in held]
    fetch_pt3(set(keep))
    docs, missing, no_abstract = [], 0, 0
    for p in keep:
        bd = json.loads((PT3_DIR / f"{p}.json").read_text(encoding="utf-8"))
        if bd.get("missing"):
            missing += 1
        elif not any(x["infons"].get("type") == "abstract" and x["text"].strip() for x in bd["passages"]):
            no_abstract += 1
        else:
            docs.append(to_doc(bd)[0])
    manifest = {"exclude": list(exclude), "n_pmids": len(pmids), "n_excluded": len(pmids) - len(keep),
                "n_missing": missing, "n_no_abstract": no_abstract, "n_docs": len(docs),
                "pmids_sha256": hashlib.sha256("\n".join(d["pmid"] for d in docs).encode()).hexdigest()}
    return docs, manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pmids-file", required=True, help="whitespace-separated PMIDs (or a Doc/BioC JSON)")
    ap.add_argument("--out", required=True, help="Doc JSON list; the manifest goes next to it")
    a = ap.parse_args(argv)
    docs, manifest = build_docs(read_pmids(a.pmids_file))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(docs, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main(sys.argv[1:])
