"""Preregistration §9 items 3-6: causal-label audit, CRED baselines, BC8-400 genes vs variants, curator rules.

POST HOC: added after all three test days and after the internal review of the draft; never used to select a system.
CPU only: no training, no inference; stored prediction files, gold annotations and PubTator3 silver annotations only.

    PYTHONPATH=. .venv/bin/python docs/results/posthoc/causal_audit.py
      reads  data/BioRED/{Train,Test}.BioC.JSON, data/BC8/bc8_test.json, data/CRED/{cred_docs.json,cred_pairs.tsv,raw/}
             data/PT3/raw/<pmid>.json  PubTator3 biocjson of BioRED Train/Test + BC8-400 (downloaded once if missing)
             docs/results/cred/ens_{bc8all,biolinkbert_base}/pred_cred.jsonl
             docs/results/testday_w1/runs/_testday_w1/{ens_re_biolinkbert_base,crossfit_ens}/pred_bc8.jsonl
      writes docs/results/posthoc/causal_audit.json, causal_audit.md

Rules (also in the JSON ``meta``):
* Variant -> gene. Gold BioRED/BC8 variant annotations hold only an rsID or a tmVar string, no gene (checked on
  data/BioRED, data/BC8 and the BC8 organisers' BioC/PubTator files). Genes come from PubTator3 (tmVar3): a PubTator3
  variant annotation gives its CorrespondingGene ids to a gold variant entity when its ids (rs<RS#>, rsids, the tmVar
  string) contain the entity id or its text equals the text of one of the entity's mentions. Offsets are not used
  (PubMed text changed after annotation for some documents). On CRED the entities are PubTator3's, so it links exactly.
* Variant context of a gene-disease pair: some variant linked to the gene has a signed label (Positive_/Negative_
  Correlation) with that disease; gold labels in A/D, predicted labels in E(b).
* Bootstrap: documents with replacement, 10 000 resamples, a fresh numpy default_rng(0) per document set, so systems
  scored on the same documents get the same draws (paired) and cred_eval.py's CIs are reproduced; 2.5/97.5 percentiles.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.request
from bisect import bisect_right
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "docs/results/cred"), str(ROOT / "docs/results/errors")]

from biocausal.data import load_docs  # noqa: E402
from biocausal.evaluate import causal_metrics, is_causal_pair, read_jsonl  # noqa: E402
from cred_build import API, TYPES, split_ids  # noqa: E402
from cred_eval import prf, weighted_ap  # noqa: E402
from errors import sentence_starts, sentences  # noqa: E402

N_BOOT, SEED = 10_000, 0
D, G, V = "DiseaseOrPhenotypicFeature", "GeneOrGeneProduct", "SequenceVariant"
CAUSAL = ("Positive_Correlation", "Negative_Correlation")
FOUR = ("Association", "Positive_Correlation", "Negative_Correlation", "None")
PT3_DIR, CRED_DIR = ROOT / "data/PT3/raw", ROOT / "data/CRED"
PT3_FETCHED = "2026-09-17"
GOLD = {"biored_train": "data/BioRED/Train.BioC.JSON", "biored_test": "data/BioRED/Test.BioC.JSON",
        "bc8": "data/BC8/bc8_test.json"}
TD = ROOT / "docs/results/testday_w1/runs/_testday_w1"
BC8_PRED = {"biored_only": TD / "ens_re_biolinkbert_base", "crossfit": TD / "crossfit_ens"}
CROSSFIT_HALVES = [TD / "ens_re_bl_base_bc8B/pred_bc8A.jsonl", TD / "ens_re_bl_base_bc8A/pred_bc8B.jsonl"]
CRED_PRED = {"bc8all": "ens_bc8all", "biored_only": "ens_biolinkbert_base"}   # cred_eval.py system names
KS, TOP = (10, 50, 100), 50

# Fixed in code before any of the top-50 false positives was looked at (preregistration §9 item 4); never edited after.
LEVEL_RE = re.compile(r"express|level|serum|plasma|mrna|up-?regulat|down-?regulat|increas|decreas|elevat", re.I)
ANIMAL_RE = re.compile(r"\b(?:mice|mouse|murine|rats?|rodents?|zebrafish|drosophila|xenopus|pigs?|porcine|rabbits?|"
                       r"dogs?|canine|monkeys?|macaques?|primates?|animal models?)\b", re.I)


# ---- linking and rules --------------------------------------------------------------------------------------------
def fetch_pt3(pmids: set[str]) -> None:
    """Cache PubTator3 biocjson verbatim; a PMID the API does not return is cached as {"missing": true}."""
    PT3_DIR.mkdir(parents=True, exist_ok=True)
    todo = sorted((p for p in pmids if not (PT3_DIR / f"{p}.json").exists()), key=int)
    for i in range(0, len(todo), 100):
        batch = todo[i:i + 100]
        got = json.loads(urllib.request.urlopen(API + ",".join(batch), timeout=120).read())["PubTator3"]
        for d in got:
            (PT3_DIR / f"{d['id']}.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        for p in set(batch) - {str(d["id"]) for d in got}:
            (PT3_DIR / f"{p}.json").write_text(json.dumps({"id": p, "passages": [], "missing": True}), encoding="utf-8")
        time.sleep(1)


def pt3_variants(pt3doc: dict) -> list[tuple[set[str], set[str], str]]:
    """(ids, genes, text) per PubTator3 variant annotation; ids = rs<RS#>, rsids, tmVar string; genes = CorrespondingGene
    (equal to infons.gene_ids on all 178 CRED variant annotations)."""
    out = []
    for p in pt3doc["passages"]:
        for a in p["annotations"]:
            inf = a["infons"]
            if TYPES.get(inf.get("type")) != V:
                continue
            parts = dict(x.split(":", 1) for x in str(inf.get("identifier") or "").split(";") if ":" in x)
            ids = {*inf.get("rsids", []), *(["rs" + parts["RS#"]] if parts.get("RS#") else []),
                   *([parts["tmVar"]] if parts.get("tmVar") else [])}
            out.append((ids, set(split_ids(parts.get("CorrespondingGene"))), a["text"]))
    return out


def link_variants(doc: dict, pt3doc: dict) -> dict[str, set[str]]:
    """{variant entity id: NCBI gene ids} by id or exact mention text (see module docstring)."""
    # ponytail: text match could borrow the gene of another variant written identically in the same abstract
    anns = pt3_variants(pt3doc)
    out = {}
    for eid, e in doc["entities"].items():
        if e["type"] == V:
            texts = {doc["mentions"][i]["text"] for i in e["mentions"]}
            genes = set().union(*(g for ids, g, t in anns if eid in ids or t in texts))
            if genes:
                out[eid] = genes
    return out


def via_variant(rows: list[dict], links: dict[str, dict], key: str) -> set[tuple[str, str, str]]:
    """{(pmid, disease, gene)}: a disease-variant row with a signed ``key`` label whose variant is linked to the gene
    (canonical pair order puts the disease first in both disease-variant and disease-gene pairs)."""
    return {(r["pmid"], r["e1"], g) for r in rows if (r["t1"], r["t2"]) == (D, V) and r[key] in CAUSAL
            for g in links.get(r["pmid"], {}).get(r["e2"], ())}


def shared_sentences(doc: dict, ids_a: list[str], ids_b: list[str]) -> list[str]:
    """Sentences (errors.py splitter) holding a mention of an id in ids_a and a mention of an id in ids_b."""
    st = sentence_starts(doc["text"], len(doc["title"]), [(m["start"], m["end"]) for m in doc["mentions"]])
    where = [{bisect_right(st, doc["mentions"][i]["start"]) - 1
              for e in ids for i in doc["entities"].get(e, {}).get("mentions", [])} for ids in (ids_a, ids_b)]
    return [doc["text"][s:e].strip() for k, (s, e) in enumerate(sentences(doc["text"], st)) if k in where[0] & where[1]]


def level_change(sents: list[str]) -> bool:
    return any(LEVEL_RE.search(s) for s in sents)


def animal_model(text: str) -> bool:
    return bool(ANIMAL_RE.search(text))


# ---- statistics ---------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def weights(n_docs: int) -> np.ndarray:
    return np.random.default_rng(SEED).multinomial(n_docs, np.full(n_docs, 1 / n_docs), size=N_BOOT)


def ci(v) -> list[float]:
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def prf_boot(y, pred, pmid) -> tuple[dict, np.ndarray]:
    """P/R/F1 of the positive class with document-bootstrap CIs, plus the F1 of every resample (paired differences)."""
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    docs, idx = np.unique(np.asarray(pmid), return_inverse=True)
    counts = np.zeros((len(docs), 3))
    np.add.at(counts, idx, np.c_[y & pred, pred, y].astype(float))
    tot = counts.sum(0)
    p, r, f1 = (float(x) for x in prf(*tot))
    bp, br, bf = prf(*(weights(len(docs)) @ counts).T)
    return {"n": len(y), "n_docs": len(docs), "tp": int(tot[0]), "n_pred": int(tot[1]), "n_gold": int(tot[2]),
            "p": p, "p_ci": ci(bp), "r": r, "r_ci": ci(br), "f1": f1, "f1_ci": ci(bf)}, bf


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    return {"diff_ci": ci(a - b), "p_one_sided_a_not_better": float((a - b <= 0).mean())}


def ap_boot(y, scores: dict[str, np.ndarray], pmid) -> tuple[dict, dict]:
    """Step AUPRC (tied scores form one threshold, as sklearn) of each score, CIs on the same draws."""
    docs, idx = np.unique(np.asarray(pmid), return_inverse=True)
    w = weights(len(docs))
    boot = {k: np.concatenate([weighted_ap(s, y, w[i:i + 500][:, idx]) for i in range(0, N_BOOT, 500)])
            for k, s in scores.items()}
    return {k: {"auprc": float(weighted_ap(s, y, np.ones((1, len(y))))[0]), "auprc_ci": ci(boot[k])}
            for k, s in scores.items()}, boot


def best_threshold(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Threshold t (predict s >= t) maximising F1; ties in F1 -> the highest t."""
    order = np.argsort(-s, kind="mergesort")
    ss, tp = s[order], np.cumsum(y[order])
    last = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]
    f1 = 2 * tp[last] / (last + 1 + y.sum())
    i = int(np.argmax(f1))
    return float(ss[last[i]]), float(f1[i])


def four(label: str) -> str:
    return label if label in FOUR else "None"


def dist(rows: list[dict]) -> dict:
    c = Counter(four(r["gold"]) for r in rows)
    return {"n": len(rows), "n_docs": len({r["pmid"] for r in rows}), "counts": {l: c[l] for l in FOUR},
            "share": {l: c[l] / len(rows) if rows else 0.0 for l in FOUR}}


def share_shift(rows_a: list[dict], rows_b: list[dict], labels: tuple[str, ...]) -> dict:
    """Share of ``labels`` among rows: a, b, b - a; the two corpora resampled independently (one rng, drawn a then b)."""
    rng, out = np.random.default_rng(SEED), []
    for rows in (rows_a, rows_b):
        docs, idx = np.unique([r["pmid"] for r in rows], return_inverse=True)
        c = np.zeros((len(docs), 2))
        np.add.at(c, idx, np.c_[[four(r["gold"]) in labels for r in rows], np.ones(len(rows))])
        b = rng.multinomial(len(docs), np.full(len(docs), 1 / len(docs)), size=N_BOOT) @ c
        out.append((float(c[:, 0].sum() / c[:, 1].sum()), b[:, 0] / b[:, 1]))
    (a, ba), (b, bb) = out
    return {"labels": list(labels), "a": a, "a_ci": ci(ba), "b": b, "b_ci": ci(bb), "diff": b - a, "diff_ci": ci(bb - ba)}


# ---- data ---------------------------------------------------------------------------------------------------------
def gold_rows(doc: dict) -> list[dict]:
    e = doc["entities"]
    return [{"pmid": doc["pmid"], "e1": p["e1"], "e2": p["e2"], "t1": e[p["e1"]]["type"], "t2": e[p["e2"]]["type"],
             "gold": p["label"]} for p in doc["pairs"]]


def load_pt3(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cred_view(pairs: list[dict], rows: list[dict]) -> dict[str, np.ndarray]:
    """Per CRED pair from its candidate pairs (several for a composite gene id; none when unreachable), as cred_eval.py:
    argmax of the stored probabilities, score = max p(Pos)+p(Neg)."""
    by_key = {(r["pmid"], r["e1"], r["e2"]): r for r in rows}
    out = {"causal": [], "any": [], "assoc": [], "score": []}
    for c in pairs:
        comps = [by_key[(c["pmid"], *k.split("|"))]["probs"] for k in c["pairs"].split(";")] if c["pairs"] else []
        am = [max(p, key=p.get) for p in comps]
        causal = any(a in CAUSAL for a in am)
        out["causal"].append(causal)
        out["any"].append(any(a != "None" for a in am))
        out["assoc"].append(not causal and "Association" in am)
        out["score"].append(max((p[CAUSAL[0]] + p[CAUSAL[1]] for p in comps), default=0.0))
    return {k: np.array(v) for k, v in out.items()}


# ---- analyses -----------------------------------------------------------------------------------------------------
def part_a(gold: dict, links: dict, pt3: dict) -> tuple[dict, dict]:
    out, ctx_all, inside_rows, outside_rows = {}, {}, {}, {}
    for name, docs in gold.items():
        rows = [r for d in docs for r in gold_rows(d)]
        ents = {d["pmid"]: d["entities"] for d in docs}
        ctx = via_variant(rows, links, "gold")
        gd = [r for r in rows if (r["t1"], r["t2"]) == (D, G)]
        inside_rows[name] = [r for r in gd if (r["pmid"], r["e1"], r["e2"]) in ctx]
        outside_rows[name] = [r for r in gd if (r["pmid"], r["e1"], r["e2"]) not in ctx]
        signed_dv = [r for r in rows if (r["t1"], r["t2"]) == (D, V) and r["gold"] in CAUSAL]
        out[name] = {
            "n_docs": len(docs), "n_docs_missing_in_pubtator3": sum(bool(pt3[d["pmid"]].get("missing")) for d in docs),
            "variant_entities": sum(e["type"] == V for d in docs for e in d["entities"].values()),
            "variant_entities_linked": sum(len(links[d["pmid"]]) for d in docs),
            "variant_entities_linked_by_text_only": sum(
                not any(v in ids for ids, _, _ in pt3_variants(pt3[d["pmid"]])) for d in docs for v in links[d["pmid"]]),
            "signed_disease_variant_pairs": len(signed_dv),
            "signed_disease_variant_pairs_linked": sum(r["e2"] in links[r["pmid"]] for r in signed_dv),
            "context_triples": len(ctx),
            "context_triples_gene_not_gene_entity": sum(ents[p].get(g, {}).get("type") != G for p, _, g in ctx),
            "gene_disease_pairs": len(gd),
            "in_variant_context": dist(inside_rows[name]), "outside_variant_context": dist(outside_rows[name])}
        ctx_all[name] = ctx
    out["shift_bc8_minus_biored_train"] = {
        grp: {lab: share_shift(rows["biored_train"], rows["bc8"], labs)
              for lab, labs in (("association", ("Association",)), ("signed", CAUSAL), ("none", ("None",)))}
        for grp, rows in (("in_variant_context", inside_rows), ("outside_variant_context", outside_rows))}
    return out, ctx_all


def main() -> None:
    gold = {name: load_docs(ROOT / path) for name, path in GOLD.items()}
    fetch_pt3({d["pmid"] for docs in gold.values() for d in docs})
    pt3 = {d["pmid"]: load_pt3(PT3_DIR / f"{d['pmid']}.json") for docs in gold.values() for d in docs}
    links = {d["pmid"]: link_variants(d, pt3[d["pmid"]]) for docs in gold.values() for d in docs}
    res: dict = {}

    # A ----------------------------------------------------------------------------------------------------------
    res["A_label_audit"], ctx_gold = part_a(gold, links, pt3)

    # CRED data ----------------------------------------------------------------------------------------------------
    pairs = list(csv.DictReader(open(CRED_DIR / "cred_pairs.tsv", encoding="utf-8"), delimiter="\t"))
    cdocs = {d["pmid"]: d for d in json.load(open(CRED_DIR / "cred_docs.json", encoding="utf-8"))}
    craw = {p: load_pt3(CRED_DIR / "raw" / f"{p}.json") for p in cdocs}
    clinks = {p: link_variants(cdocs[p], craw[p]) for p in cdocs}
    doc_var_genes = {p: set().union(*(g for _, g, _ in pt3_variants(craw[p]))) for p in cdocs}
    crows = {n: read_jsonl(ROOT / "docs/results/cred" / d / "pred_cred.jsonl") for n, d in CRED_PRED.items()}
    cv = {n: cred_view(pairs, crows[n]) for n in CRED_PRED}
    y = np.array([c["label"] == "1" for c in pairs])
    reach = np.array([c["status"] == "reachable" for c in pairs])
    same = np.array([c["same_sent"] == "1" for c in pairs])
    pmid = np.array([c["pmid"] for c in pairs])
    genes = [split_ids(c["gene_id"]) for c in pairs]
    disease = [split_ids(c["disease_id"])[0] for c in pairs]
    masks = {"reachable": reach, "strict": np.ones(len(pairs), bool)}

    # self-check: the CRED numbers of cred_eval.py (point and CI) are reproduced
    cres = json.loads((ROOT / "docs/results/cred/cred_results.json").read_text(encoding="utf-8"))["results"]["all"]["all"]
    for n, sysname in CRED_PRED.items():
        for var, m in masks.items():
            r, _ = prf_boot(y[m], cv[n]["causal"][m], pmid[m])
            ref = cres[var][sysname]
            assert abs(r["f1"] - ref["f1"]) < 1e-12 and np.allclose(r["f1_ci"], ref["f1_ci"], atol=1e-12), (n, var)
            assert abs(weighted_ap(cv[n]["score"][m], y[m], np.ones((1, m.sum())))[0] - ref["auprc"]) < 1e-12, (n, var)

    # B ----------------------------------------------------------------------------------------------------------
    in_ctx = np.array([any(g in doc_var_genes[c["pmid"]] for g in gs) for c, gs in zip(pairs, genes)])
    res["B_cred_variant_context"] = {
        "rule": "causal reachable CRED pair; 'with_variant' = some PubTator3 variant annotation of the abstract has a "
                "CorrespondingGene equal to (a component of) the pair's gene id",
        **{n: {grp: {"n": int(m.sum()), "n_docs": len(set(pmid[m])),
                     "association_n": int(cv[n]["assoc"][m].sum()), "association": float(cv[n]["assoc"][m].mean()),
                     "none": float((~cv[n]["any"][m]).mean()),
                     "causal_recall_n": int(cv[n]["causal"][m].sum()), "causal_recall": float(cv[n]["causal"][m].mean())}
               for grp, m in (("with_variant", y & reach & in_ctx), ("without_variant", y & reach & ~in_ctx))}
           for n in CRED_PRED}}

    # C ----------------------------------------------------------------------------------------------------------
    c_out: dict = {"cooccurrence": {}, "any_relation": {}, "auprc_model_minus_cooccurrence": {}, "precision_at_k": {},
                   "top50_signed_false_positives": {}}
    for var, m in masks.items():
        co = same & reach          # strict: an unreachable pair scores 0 for every system, as in cred_eval.py
        c_out["cooccurrence"][var], _ = prf_boot(y[m], co[m], pmid[m])
        aps, boot = ap_boot(y[m], {"cooccurrence": co[m].astype(float),
                                   **{n: cv[n]["score"][m] for n in CRED_PRED}}, pmid[m])   # unreachable: 0
        c_out["cooccurrence"][var]["auprc_step"] = aps["cooccurrence"]
        for n in CRED_PRED:
            c_out["any_relation"].setdefault(n, {})[var], _ = prf_boot(y[m], cv[n]["any"][m], pmid[m])
            c_out["auprc_model_minus_cooccurrence"].setdefault(n, {})[var] = {
                "model": aps[n], "cooccurrence": aps["cooccurrence"],
                "diff": aps[n]["auprc"] - aps["cooccurrence"]["auprc"], **paired(boot[n], boot["cooccurrence"])}
    animal_docs = {p for p, d in cdocs.items() if animal_model(d["text"])}
    for n in CRED_PRED:
        s = cv[n]["score"]
        order_key = lambda i, s=s: (-s[i], int(pairs[i]["pmid"]), pairs[i]["gene_id"], pairs[i]["disease_id"])
        ranked = sorted(np.nonzero(reach)[0], key=order_key)
        c_out["precision_at_k"][n] = {f"p@{k}": float(y[ranked[:k]].mean()) for k in KS}
        top = [i for i in ranked if cv[n]["causal"][i]][:TOP]
        items = []
        for i in top:
            c = pairs[i]
            sents = shared_sentences(cdocs[c["pmid"]], genes[i], [disease[i]])
            items.append({"pmid": c["pmid"], "gene_id": c["gene_id"], "disease_id": c["disease_id"],
                          "cred_causal": bool(y[i]), "score": float(s[i]),
                          "rule": "no_shared_sentence" if not sents else
                          "level_change" if level_change(sents) else "other",
                          "animal_model_abstract": c["pmid"] in animal_docs, "sentences": sents})
        fp, tp = [x for x in items if not x["cred_causal"]], [x for x in items if x["cred_causal"]]
        summary = lambda xs: {"n": len(xs), **Counter(x["rule"] for x in xs),
                              "animal_model_abstract": sum(x["animal_model_abstract"] for x in xs)}
        c_out["top50_signed_false_positives"][n] = {"n_signed_predictions": int((cv[n]["causal"] & reach).sum()),
                                                    "top": len(top), "false_positives": summary(fp),
                                                    "true_positives": summary(tp), "items_false_positive": fp}
    c_out["animal_model_abstracts_all_cred"] = {"n": len(animal_docs), "of_docs": len(cdocs)}
    res["C_cred_baselines"] = c_out

    # D ----------------------------------------------------------------------------------------------------------
    bc8 = {n: read_jsonl(d / "pred_bc8.jsonl") for n, d in BC8_PRED.items()}
    halves = {(r["pmid"], r["e1"], r["e2"]): r["pred"] for f in CROSSFIT_HALVES for r in read_jsonl(f)}
    assert halves == {(r["pmid"], r["e1"], r["e2"]): r["pred"] for r in bc8["crossfit"]}, "crossfit != its two halves"
    for n, d in BC8_PRED.items():
        stored = json.loads((d / "metrics_bc8.json").read_text(encoding="utf-8"))["causal"]["causal"]
        assert causal_metrics(bc8[n])["causal"]["f1"] == stored["f1"], n
    bdocs = {d["pmid"]: d for d in gold["bc8"]}
    err = json.loads((ROOT / "docs/results/errors/errors.json").read_text(encoding="utf-8"))["by_kind"]
    rs_text = lambda r: any(re.fullmatch(r"rs\s?\d+", bdocs[r["pmid"]]["mentions"][i]["text"].strip(), re.I)
                            for i in bdocs[r["pmid"]]["entities"][r["e2"]]["mentions"])
    d_out: dict = {}
    d_boot: dict = {}
    for n, rows in bc8.items():
        kinds = {"gene": [r for r in rows if (r["t1"], r["t2"]) == (D, G)],
                 "variant": [r for r in rows if (r["t1"], r["t2"]) == (D, V)]}
        o = d_out[n] = {}
        for k, rs in kinds.items():
            o[k], d_boot[n, k] = prf_boot([r["gold"] in CAUSAL for r in rs], [r["pred"] in CAUSAL for r in rs],
                                          [r["pmid"] for r in rs])
            if n == "biored_only":
                assert all(abs(o[k][x] - err[k]["causal"][x]) < 1e-12 for x in ("p", "r", "f1")), k
        for split, is_rs in (("by_id", lambda r: bool(re.fullmatch(r"rs\d+", r["e2"]))), ("by_mention_text", rs_text)):
            for grp, want in (("rsID", True), ("HGVS/tmVar", False)):
                rs = [r for r in kinds["variant"] if is_rs(r) == want]
                o.setdefault(f"variant_{split}", {})[grp], _ = prf_boot(
                    [r["gold"] in CAUSAL for r in rs], [r["pred"] in CAUSAL for r in rs], [r["pmid"] for r in rs])
        gold_causal = [r for r in kinds["gene"] if r["gold"] in CAUSAL]
        inside = [(r["pmid"], r["e1"], r["e2"]) in ctx_gold["bc8"] for r in gold_causal]
        o["gold_causal_gene_pairs"] = {"n": len(gold_causal), "in_variant_context": sum(inside),
                                       "share_in_variant_context": sum(inside) / len(gold_causal)}
        for grp, want in (("in_variant_context", True), ("outside_variant_context", False)):
            rs = [r for r, i in zip(gold_causal, inside) if i == want]
            full, _ = prf_boot([True] * len(rs), [r["pred"] in CAUSAL for r in rs], [r["pmid"] for r in rs])
            o[f"recall_{grp}"] = {x: full[x] for x in ("n", "n_docs", "tp", "r", "r_ci")}
            o[f"recall_{grp}"]["pred_association"] = sum(r["pred"] == "Association" for r in rs)
    res["D_bc8_gene_vs_variant"] = d_out
    # paired: both systems score the same BC8-400 pairs, so the same documents get the same draws
    assert all(d_out["crossfit"][k]["n_docs"] == d_out["biored_only"][k]["n_docs"] for k in ("gene", "variant"))
    res["D_bc8_crossfit_minus_biored_only"] = {
        k: {"f1_diff": d_out["crossfit"][k]["f1"] - d_out["biored_only"][k]["f1"],
            **paired(d_boot["crossfit", k], d_boot["biored_only", k])} for k in ("gene", "variant")}

    # E(a) -------------------------------------------------------------------------------------------------------
    sub = [r for r in bc8["biored_only"] if is_causal_pair(r["t1"], r["t2"])]
    ys = np.array([r["gold"] in CAUSAL for r in sub])
    ss = np.array([r["probs"][CAUSAL[0]] + r["probs"][CAUSAL[1]] for r in sub])
    t, f1_t = best_threshold(ys, ss)
    bc8_t, _ = prf_boot(ys, ss >= t, [r["pmid"] for r in sub])
    assert abs(bc8_t["f1"] - f1_t) < 1e-12
    ea = {"threshold": t, "selected_on": "BC8-400 causal subset (gene+variant x disease), BioRED-only ensemble, "
                                         "predict p(Pos)+p(Neg) >= t, max causal F1 (in-sample on BC8)",
          "bc8_at_threshold": bc8_t,
          "bc8_argmax": prf_boot(ys, [r["pred"] in CAUSAL for r in sub], [r["pmid"] for r in sub])[0], "cred": {}}
    for n in CRED_PRED:
        for var, m in masks.items():
            at_t, bt = prf_boot(y[m], (cv[n]["score"] >= t)[m], pmid[m])
            am, ba = prf_boot(y[m], cv[n]["causal"][m], pmid[m])
            ea["cred"].setdefault(n, {})[var] = {"threshold": at_t, "argmax": am, "f1_threshold_minus_argmax": paired(bt, ba)}
    res["E_a_bc8_threshold_on_cred"] = ea

    # E(b) -------------------------------------------------------------------------------------------------------
    eb: dict = {"bc8": {}, "cred": {}}
    for n, rows in bc8.items():
        ctx = via_variant(rows, links, "pred")
        gd = [r for r in rows if (r["t1"], r["t2"]) == (D, G)]
        yy, pm = [r["gold"] in CAUSAL for r in gd], [r["pmid"] for r in gd]
        direct = np.array([r["pred"] in CAUSAL for r in gd])
        prop = direct | np.array([(r["pmid"], r["e1"], r["e2"]) in ctx for r in gd])
        a, ba = prf_boot(yy, direct, pm)
        b, bb = prf_boot(yy, prop, pm)
        eb["bc8"][n] = {"direct": a, "propagated": b, "f1_propagated_minus_direct": paired(bb, ba),
                        "flipped": int((prop & ~direct).sum()), "flipped_gold_causal": int((prop & ~direct & yy).sum())}
    for n in CRED_PRED:
        ctx = via_variant(crows[n], clinks, "pred")
        prop = cv[n]["causal"] | np.array([st and any((c["pmid"], dz, g) in ctx for g in gs)
                                           for c, st, dz, gs in zip(pairs, reach, disease, genes)])
        for var, m in masks.items():
            a, ba = prf_boot(y[m], cv[n]["causal"][m], pmid[m])
            b, bb = prf_boot(y[m], prop[m], pmid[m])
            flip = (prop & ~cv[n]["causal"])[m]
            eb["cred"].setdefault(n, {})[var] = {"direct": a, "propagated": b,
                                                 "f1_propagated_minus_direct": paired(bb, ba),
                                                 "flipped": int(flip.sum()), "flipped_cred_causal": int((flip & y[m]).sum())}
    res["E_b_variant_to_gene_propagation"] = eb

    meta = {
        "status": "post hoc (preregistration §9 items 3-6), not used for system selection",
        "n_resamples": N_BOOT, "seed": SEED,
        "variant_gene_link": "gold BioRED/BC8 variant annotations have no gene field; PubTator3 (fetched "
                             f"{PT3_FETCHED}, data/PT3/raw) variant annotation -> CorrespondingGene, matched to a gold "
                             "variant entity when its ids (rs<RS#>, rsids, tmVar) contain the entity id or its text equals "
                             "one of the entity's mention texts; CRED: same rule on its own PubTator3 entities",
        "variant_context": "gene-disease pair such that a variant linked to the gene has Positive_/Negative_Correlation "
                           "with the disease (gold in A and D, predicted in E(b))",
        "cooccurrence_baseline": "causal iff CRED is_in_same_sent = True; as a score it is 0/1, step AUPRC with the tied "
                                 "group as one threshold = R1*P1 + (1-R1)*prevalence; strict: unreachable pairs score 0",
        "level_change_rule": f"regex {LEVEL_RE.pattern!r} (case-insensitive) in a sentence holding both entities "
                             "(errors.py sentence splitter); fixed before the false positives were read",
        "animal_model_rule": f"regex {ANIMAL_RE.pattern!r} (case-insensitive) anywhere in title+abstract",
        "top50": "reachable CRED pairs with argmax in {Pos, Neg}, sorted by p(Pos)+p(Neg) desc, then pmid, gene, disease",
        "threshold_rule": "max causal F1 over distinct scores, predict score >= t, ties -> highest t",
        "propagation": "candidate gene-disease pair is causal if predicted signed, or a variant linked to its gene is "
                       "predicted signed with the disease; only candidate (reachable) pairs can flip",
        "bc8_notation": "by_id: entity id rs<digits> vs tmVar string; by_mention_text: some mention written rs<digits>",
        "files": {"bc8": {n: str(d.relative_to(ROOT)) for n, d in BC8_PRED.items()},
                  "cred": {n: f"docs/results/cred/{d}/pred_cred.jsonl" for n, d in CRED_PRED.items()}},
    }
    out = {"meta": meta, **res}
    (HERE / "causal_audit.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=lambda o: o.item())
                                            + "\n", encoding="utf-8")
    (HERE / "causal_audit.md").write_text(render_md(out), encoding="utf-8")


# ---- report -------------------------------------------------------------------------------------------------------
def pct(v: float) -> str:
    return f"{100 * v:.1f}"


def iv(bounds: list[float]) -> str:
    return f"[{pct(bounds[0])}; {pct(bounds[1])}]"


def wci(d: dict, k: str) -> str:
    return f"{pct(d[k])} {iv(d[k + '_ci'])}"


ENS_RU = {"biored_only": "BioRED-only", "bc8all": "bc8all", "crossfit": "cross-fit"}


def render_md(o: dict) -> str:
    a, b, c, d, ea, eb = (o[k] for k in ("A_label_audit", "B_cred_variant_context", "C_cred_baselines",
                                         "D_bc8_gene_vs_variant", "E_a_bc8_threshold_on_cred",
                                         "E_b_variant_to_gene_propagation"))
    L = ["# Пост-хок аудит каузальных меток (предрегистрация §9, пп. 3–6)", "",
         "**Post hoc.** Добавлено после всех дней теста и внутреннего рецензирования; для выбора системы не "
         "используется. Без обучения и инференса: только сохранённые прогнозы, золото и серебряная разметка PubTator3. "
         "Интервалы: document bootstrap, 95 %, 10 000 ресэмплов, seed 0. Числа ×100. Код `causal_audit.py`, все "
         "числа и правила в `causal_audit.json`.", "",
         "**Связь вариант → ген.** В золоте BioRED и BC8 у варианта только rsID или строка tmVar, поля гена нет. Ген "
         f"берётся из PubTator3 (tmVar3, `CorrespondingGene`, выгрузка {PT3_FETCHED}): аннотация PubTator3 "
         "сопоставляется варианту золота по идентификатору (rs, tmVar) или по точному тексту упоминания. На CRED "
         "сущности сами из PubTator3, связь точная. **Контекст варианта** пары ген–болезнь: вариант этого гена имеет "
         "с той же болезнью Positive или Negative_Correlation.", "",
         "## A. Метки пар ген–болезнь в контексте варианта (золото, без модели)", "",
         "| Корпус | Вариантов связано | Знаковых пар вариант–болезнь со связью | В контексте: пар | Assoc | Pos | Neg | "
         "None | Вне контекста: пар | Assoc | Pos | Neg | None |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    names = {"biored_train": "BioRED Train", "biored_test": "BioRED Test", "bc8": "BC8-400"}
    for k, nm in names.items():
        x, i, u = a[k], a[k]["in_variant_context"], a[k]["outside_variant_context"]
        L.append(f"| {nm} | {x['variant_entities_linked']}/{x['variant_entities']} | "
                 f"{x['signed_disease_variant_pairs_linked']}/{x['signed_disease_variant_pairs']} | {i['n']} | "
                 + " | ".join(pct(i["share"][l]) for l in FOUR) + f" | {u['n']} | "
                 + " | ".join(pct(u["share"][l]) for l in FOUR) + " |")
    L += ["", "Сдвиг BC8-400 минус BioRED Train (доля, п. п., независимый bootstrap двух корпусов):", ""]
    for grp, ru in (("in_variant_context", "в контексте"), ("outside_variant_context", "вне контекста")):
        s = a["shift_bc8_minus_biored_train"][grp]
        L.append(f"- {ru}: Association {pct(s['association']['a'])} → {pct(s['association']['b'])}, "
                 f"разница {wci(s['association'], 'diff')}; знаковые {pct(s['signed']['a'])} → {pct(s['signed']['b'])}, "
                 f"разница {wci(s['signed'], 'diff')}; None {pct(s['none']['a'])} → {pct(s['none']['b'])}, "
                 f"разница {wci(s['none'], 'diff')}.")
    L += ["", "Документов без ответа PubTator3: " + ", ".join(f"{names[k]} {a[k]['n_docs_missing_in_pubtator3']}"
                                                              for k in names)
          + ". Связанных вариантов только по тексту упоминания: " + ", ".join(
              f"{names[k]} {a[k]['variant_entities_linked_by_text_only']}" for k in names)
          + ". Тройки контекста, где связанный ген не сущность-ген золота: " + ", ".join(
              f"{names[k]} {a[k]['context_triples_gene_not_gene_entity']} из {a[k]['context_triples']}" for k in names)
          + ".", "",
          "## B. CRED: каузальные пары и упоминание варианта гена", "",
          "Достижимые каузальные пары CRED; «с вариантом» — в аннотации есть вариант PubTator3 с `CorrespondingGene` "
          "гена пары.", "",
          "| Ансамбль | Группа | Пар (док.) | Association | None | Полнота каузального |", "|---|---|---|---|---|---|"]
    for n in CRED_PRED:
        for grp, ru in (("with_variant", "с вариантом"), ("without_variant", "без варианта")):
            x = b[n][grp]
            L.append(f"| {ENS_RU[n]} | {ru} | {x['n']} ({x['n_docs']}) | {pct(x['association'])} | {pct(x['none'])} | "
                     f"{pct(x['causal_recall'])} |")
    L += ["", "## C. CRED: бейзлайны и вершина списка", "",
          "| Система | Вариант | P | R | F1 [95 % CI] |", "|---|---|---|---|---|"]
    for var in ("reachable", "strict"):
        x = c["cooccurrence"][var]
        L.append(f"| совпадение в предложении (бинарный) | {var} | {pct(x['p'])} | {pct(x['r'])} | {wci(x, 'f1')} |")
        for n in CRED_PRED:
            x = c["any_relation"][n][var]
            L.append(f"| {ENS_RU[n]}, «любая связь» (argmax ≠ None) | {var} | {pct(x['p'])} | {pct(x['r'])} | "
                     f"{wci(x, 'f1')} |")
    L += ["", "AUPRC совпадения в предложении — ступенчатая AP 0/1-оценки: одна точка P/R, связанная группа — один "
          "порог, AP = R·P + (1 − R)·доля каузальных. Это не площадь под кривой непрерывной оценки.", "",
          "| Ансамбль | Вариант | AUPRC модели | AUPRC совпадения (ступенчатая) | Разница [95 % CI] | доля ресэмплов ≤ 0 |",
          "|---|---|---|---|---|---|"]
    for n in CRED_PRED:
        for var in ("reachable", "strict"):
            x = c["auprc_model_minus_cooccurrence"][n][var]
            L.append(f"| {ENS_RU[n]} | {var} | {wci(x['model'], 'auprc')} | {wci(x['cooccurrence'], 'auprc')} | "
                     f"{pct(x['diff'])} {iv(x['diff_ci'])} | "
                     f"{x['p_one_sided_a_not_better']:.4f} |")
    L += ["", "| Ансамбль | P@10 | P@50 | P@100 |", "|---|---|---|---|"]
    L += [f"| {ENS_RU[n]} | " + " | ".join(pct(c["precision_at_k"][n][f"p@{k}"]) for k in KS) + " |" for n in CRED_PRED]
    L += ["", f"Верхние {TOP} знаковых прогнозов (argmax Pos/Neg, по убыванию p(Pos)+p(Neg)). Правило «изменение уровня» "
          "(ключевые слова в предложении с обеими сущностями) и правило «животная модель» (по тексту аннотации) "
          "зафиксированы в коде до просмотра.", "",
          "| Ансамбль | Знаковых всего | Группа | Пар | изменение уровня | другое | нет общего предложения | "
          "животная модель |", "|---|---|---|---|---|---|---|---|"]
    for n in CRED_PRED:
        x = c["top50_signed_false_positives"][n]
        for grp, ru in (("false_positives", "некаузальные по CRED"), ("true_positives", "каузальные по CRED")):
            g = x[grp]
            L.append(f"| {ENS_RU[n]} | {x['n_signed_predictions']} | {ru} | {g['n']} | {g.get('level_change', 0)} | "
                     f"{g.get('other', 0)} | {g.get('no_shared_sentence', 0)} | {g['animal_model_abstract']} |")
    am = c["animal_model_abstracts_all_cred"]
    L += ["", f"Для сравнения: правило «животная модель» срабатывает на {am['n']} из {am['of_docs']} аннотаций CRED "
          f"({pct(am['n'] / am['of_docs'])} %).", "",
          "## D. BC8-400: гены и варианты", "",
          "| Ансамбль | Пары | Пар (кауз.) | P [95 % CI] | R [95 % CI] | F1 [95 % CI] |", "|---|---|---|---|---|---|"]
    rows_d = [("gene", "ген–болезнь"), ("variant", "вариант–болезнь"),
              (("variant_by_id", "rsID"), "вариант, id rsID"), (("variant_by_id", "HGVS/tmVar"), "вариант, id tmVar"),
              (("variant_by_mention_text", "rsID"), "вариант, в тексте rs…"),
              (("variant_by_mention_text", "HGVS/tmVar"), "вариант, в тексте не rs")]
    for n in BC8_PRED:
        for key, ru in rows_d:
            x = d[n][key] if isinstance(key, str) else d[n][key[0]][key[1]]
            L.append(f"| {ENS_RU[n]} | {ru} | {x['n']} ({x['n_gold']}) | {wci(x, 'p')} | {wci(x, 'r')} | {wci(x, 'f1')} |")
    L += ["", "| Ансамбль | Золотых каузальных пар ген–болезнь | из них в контексте варианта | Полнота в контексте "
          "[95 % CI] (Assoc) | Полнота вне контекста [95 % CI] (Assoc) |", "|---|---|---|---|---|"]
    for n in BC8_PRED:
        g, i, u = d[n]["gold_causal_gene_pairs"], d[n]["recall_in_variant_context"], d[n]["recall_outside_variant_context"]
        L.append(f"| {ENS_RU[n]} | {g['n']} | {g['in_variant_context']} ({pct(g['share_in_variant_context'])} %) | "
                 f"{wci(i, 'r')} ({i['pred_association']}) | {wci(u, 'r')} ({u['pred_association']}) |")
    L += ["", "Парная разница F1 каузального класса, cross-fit минус BioRED-only (те же пары и ресэмплы): " + "; ".join(
        f"{ru} {pct(x['f1_diff'])} {iv(x['diff_ci'])}, доля ресэмплов ≤ 0 {x['p_one_sided_a_not_better']:.4f}"
        for ru, x in (("гены", o["D_bc8_crossfit_minus_biored_only"]["gene"]),
                      ("варианты", o["D_bc8_crossfit_minus_biored_only"]["variant"]))) + "."]
    L += ["", "## E. Правила куратора на готовых прогнозах", "",
          f"**(а) Порог с BC8 на CRED.** Порог p(Pos)+p(Neg) ≥ {ea['threshold']:.4f} выбран по максимуму F1 "
          "каузального класса на BC8-400 у BioRED-only (в выборке, оптимистично): F1 "
          f"{wci(ea['bc8_at_threshold'], 'f1')} против argmax {wci(ea['bc8_argmax'], 'f1')}. На CRED без изменений:", "",
          "| Ансамбль | Вариант | P | R | F1 при пороге [95 % CI] | F1 argmax | Разница F1 [95 % CI] |",
          "|---|---|---|---|---|---|---|"]
    for n in CRED_PRED:
        for var in ("reachable", "strict"):
            x = ea["cred"][n][var]
            L.append(f"| {ENS_RU[n]} | {var} | {pct(x['threshold']['p'])} | {pct(x['threshold']['r'])} | "
                     f"{wci(x['threshold'], 'f1')} | {pct(x['argmax']['f1'])} | "
                     f"{pct(x['threshold']['f1'] - x['argmax']['f1'])} {iv(x['f1_threshold_minus_argmax']['diff_ci'])} |")
    L += ["", "**(б) Перенос вариант → ген.** Пара ген–болезнь каузальна, если так предсказана сама или если вариант "
          "этого гена (связь PubTator3) предсказан знаковым с той же болезнью. BC8-400: только пары ген–болезнь; "
          "CRED: пары CRED.", "",
          "| Корпус | Ансамбль | Вариант | F1 прямой [95 % CI] | P / R перенос | F1 перенос [95 % CI] | Разница [95 % CI] | "
          "Перевёрнуто (из них верно) |", "|---|---|---|---|---|---|---|---|"]
    for corpus, ens in (("bc8", BC8_PRED), ("cred", CRED_PRED)):
        for n in ens:
            for var, x in ([("—", eb["bc8"][n])] if corpus == "bc8" else list(eb["cred"][n].items())):
                good = x.get("flipped_gold_causal", x.get("flipped_cred_causal"))
                L.append(f"| {'BC8-400' if corpus == 'bc8' else 'CRED'} | {ENS_RU[n]} | {var} | {wci(x['direct'], 'f1')} | "
                         f"{pct(x['propagated']['p'])} / {pct(x['propagated']['r'])} | {wci(x['propagated'], 'f1')} | "
                         f"{pct(x['propagated']['f1'] - x['direct']['f1'])} {iv(x['f1_propagated_minus_direct']['diff_ci'])} | "
                         f"{x['flipped']} ({good}) |")
    L += ["", "## Оговорки", "",
          "- Связь вариант → ген серебряная (PubTator3); варианты без сопоставленной аннотации в контекст не попадают.",
          "- Правила «изменение уровня» и «животная модель» — грубые словари; счётчики не заменяют чтения текста.",
          "- Порог (а) выбран на том же BC8-400, где и отчитан; на CRED он не подбирался.",
          "- `reachable` — только пары с сущностями PubTator3; `strict` — недостижимые пары считаются некаузальными "
          "(оценка 0) для всех систем, как в `docs/results/cred/cred.md`.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
