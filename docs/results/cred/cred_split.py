"""Preregistration §10 (arm A2): CRED documents split for training and scoring under the [CORPUS_cred] tag.

    PYTHONPATH=. .venv/bin/python docs/results/cred/cred_split.py
      reads  data/CRED/cred_docs.json, data/CRED/cred_pairs.tsv (cred_build.py)
      writes data/CRED/cred.json       train PMIDs; labelled reachable single-gene pairs: causal -> Positive_Correlation,
                                       non-causal -> Association; every other candidate pair None (dropped by
                                       data.label_logw for corpus "cred", so unlabelled pairs are never negatives)
             data/CRED/cred_val.json, data/CRED/cred_test.json   val / test PMIDs, all pairs None (scoring input only)

    PYTHONPATH=. .venv/bin/python docs/results/cred/cred_split.py cv
      paper 2 E3: 5-fold cross-fit over every split. Pool = all PMIDs except the 107 train abstracts whose labelled
      pairs are all causal (cred_pooled_fix rule); those are in every fold's training data, never in a test fold.
      StratifiedGroupKFold(5, shuffle, seed 0) by PMID, stratum = has a causal pair. Rows as cred_s10.load_rows
      (duplicate keys merged, conflicting keys dropped); labels on reachable single-gene pairs as build.
      writes data/CRED/cv/set/cred.json   all 266 docs, labels as cred.json (train_re --joint --label-handling set)
             data/CRED/cv/pc/cred.json    same docs, labelled pairs "None" + "pc": 1 causal / 0 not (aux head pc)
             data/CRED/cv/fold{k}.txt     test PMIDs of fold k (train_re --heldout)
             data/CRED/cv/test_f{k}.json  their docs, all pairs None (score.py input)
             data/CRED/cv/manifest.json   sha256 of every file above, fold sizes (docs, load_rows pairs, causal)

The file stem "cred" is the corpus name train_re.py passes to label_logw. Composite gene ids ('5595;5594') and
unreachable pairs get no training label. A PMID's split is the split of its pairs (no PMID has two).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CRED = ROOT / "data" / "CRED"
LABEL = {"1": "Positive_Correlation", "0": "Association"}


def build(docs: list[dict], rows: list[dict]) -> dict[str, list[dict]]:
    split = {}
    for r in rows:
        assert split.setdefault(r["pmid"], r["split"]) == r["split"], f"PMID {r['pmid']} in two splits"
    label = {(r["pmid"], *r["pairs"].split("|")): LABEL[r["label"]]
             for r in rows if r["split"] == "train" and r["status"] == "reachable" and ";" not in r["pairs"]}
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for d in docs:
        s = split[d["pmid"]]
        pairs = [{**p, "label": label.get((d["pmid"], p["e1"], p["e2"]), "None") if s == "train" else "None"}
                 for p in d["pairs"]]
        out[s].append({**d, "pairs": pairs})
    assert sum(label[k] != "None" for k in label) == sum(p["label"] != "None" for d in out["train"] for p in d["pairs"])
    return out


def cv(docs: list[dict], tsv: Path = CRED / "cred_pairs.tsv"
       ) -> tuple[list[dict], list[dict], list[list[str]], list[dict]]:
    """(set docs, pc docs, test PMIDs per fold, load_rows of every split) of the E3 cross-fit."""
    from cred_s10 import load_rows
    from sklearn.model_selection import StratifiedGroupKFold
    rows = [r for s in ("train", "val", "test") for r in load_rows(s, tsv)]
    labels: dict[str, set] = {}
    for r in rows:
        labels.setdefault(r["pmid"], set()).add(r["label"])
    pool = sorted(labels.keys() - {r["pmid"] for r in rows if r["split"] == "train" and labels[r["pmid"]] == {"1"}})
    y = ["1" in labels[p] for p in pool]
    folds = [sorted(pool[i] for i in te)
             for _, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(pool, y, pool)]
    lab = {(r["pmid"], *r["pairs"].split("|")): r["label"]
           for r in rows if r["status"] == "reachable" and ";" not in r["pairs"]}
    key = lambda d, p: (d["pmid"], p["e1"], p["e2"])  # noqa: E731
    set_docs = [{**d, "pairs": [{**p, "label": LABEL[lab[key(d, p)]] if key(d, p) in lab else "None"}
                                for p in d["pairs"]]} for d in docs]
    pc_docs = [{**d, "pairs": [{**p, "label": "None", **({"pc": int(lab[key(d, p)])} if key(d, p) in lab else {})}
                               for p in d["pairs"]]} for d in docs]
    assert len(lab) == sum(p["label"] != "None" for d in set_docs for p in d["pairs"])   # every label lands once
    return set_docs, pc_docs, folds, rows


def cv_main(out: Path = CRED / "cv") -> dict:
    docs = json.loads((CRED / "cred_docs.json").read_text(encoding="utf-8"))
    set_docs, pc_docs, folds, rows = cv(docs)
    files: dict[str, object] = {"set/cred.json": set_docs, "pc/cred.json": pc_docs}
    sizes = []
    for k, f in enumerate(folds):
        files[f"fold{k}.txt"] = "\n".join(f) + "\n"
        files[f"test_f{k}.json"] = [d for d in docs if d["pmid"] in f]
        r = [x for x in rows if x["pmid"] in f]
        sizes.append({"docs": len(f), "pairs": len(r), "causal": sum(x["label"] == "1" for x in r),
                      "docs_with_causal": len({x["pmid"] for x in r if x["label"] == "1"})})
    manifest = {"what": "paper 2 E3 folds (cred_split.py cv); pairs/causal = cred_s10.load_rows rows, strict",
                "folds": sizes, "sha256": {}}
    for name, obj in files.items():
        b = (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)).encode("utf-8")
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(b)
        manifest["sha256"][name] = hashlib.sha256(b).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print("fold  docs  pairs  causal  docs_with_causal")
    for k, s in enumerate(sizes):
        print(f"{k:>4}  {s['docs']:>4}  {s['pairs']:>5}  {s['causal']:>6}  {s['docs_with_causal']:>16}")
    print("labels", dict(Counter(p["label"] for d in set_docs for p in d["pairs"])))
    return manifest


def main() -> None:
    docs = json.loads((CRED / "cred_docs.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(CRED / "cred_pairs.tsv", encoding="utf-8"), delimiter="\t"))
    out = build(docs, rows)
    for s, name in (("train", "cred.json"), ("val", "cred_val.json"), ("test", "cred_test.json")):
        (CRED / name).write_text(json.dumps(out[s], ensure_ascii=False), encoding="utf-8")
        print(name, len(out[s]), "docs", dict(Counter(p["label"] for d in out[s] for p in d["pairs"])))


if __name__ == "__main__":
    cv_main() if sys.argv[1:2] == ["cv"] else main()
