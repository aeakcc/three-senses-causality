"""Paper 2: the five preregistered secondary analyses (preregistration §11a p. 6, §11b p. 4) that the main text leaves
out, from the stored reader predictions and E3 run metrics; nothing is scored or trained here.

    PYTHONPATH=. .venv/bin/python docs/results/paper2/secondary/secondary.py
      writes docs/results/paper2/secondary/secondary.{json,md}

1. curation: per ClinGen curation (extra.curation_ids of the single-candidate units data/paper2/e1/clingen_units.tsv),
   the mean score s of readers L and P over its units; AUROC of Definitive against Refuted or Disputed curations with a
   curation bootstrap (each curation its own cluster). Preregistered expectation: no difference. Post hoc (not
   preregistered): the same on 'pair'-anchor units only and on GEN_CLV units only.
2. human: H1, H2, H4 (and single readers) with extra.ortholog_entrez removed from gene_entrez. Candidate sets are
   recomputed; a unit is kept only with exactly one candidate left. Only documents of the E1 sets were scored, so the
   set is the subset of the E1 sets that stays single-candidate; units that become single-candidate only under human
   ids (nearly all in unscored documents) are counted, not scored.
3. biotriplex: profiles of L, P, L' and d on all BioTriplex units (scored whole, max over candidates) and the contrast
   CAUSAL_MECH or CAUSAL_MUT against LEVEL.
4. percha_L0: profile of L0 on the construction set and its AUROC on the H3 contrast (MUT or CAUSE against UP).
5. dev_f1: BioRED Dev micro-F1 of the checkpoint every FL and J run kept (FL: epoch 12, fixed; J: best epoch by dev),
   mean and sd (ddof 1) over the 15 runs, against bc8all 64.38 +- 1.56 (seeds 42-44, docs/results/RESULTS.md).
AUROC, profiles and bootstraps are docs/results/paper2/e1_eval.py (10 000 resamples, default_rng(0)). Deterministic.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import e1_eval as E  # noqa: E402  (puts ROOT and the helpers on sys.path)
from units import candidate_pairs, read_units  # noqa: E402

ROOT, D = E.ROOT, E.ROOT / "data/paper2"
R = lambda n, tag: ROOT / f"runs/_p2e1_{n}/pred_{tag}.jsonl"  # noqa: E731
EXP = {"EXP_A", "EXP_B", "EXP_U"}
BC8ALL = {"mean": 64.38, "sd": 1.56, "seeds": [63.55, 66.18, 63.42]}
csv.field_size_limit(sys.maxsize)


def load_docs(path: Path) -> dict[str, dict]:
    return {d["pmid"]: d for d in json.loads(path.read_text(encoding="utf-8"))}


def run(units, docs, tag, readers, contrasts) -> dict:
    anchors, stats = E.score_units(units, docs, {n: R(n, tag) for n in readers})
    return E.evaluate(units, anchors, stats, contrasts)


def human_only(u: dict) -> dict:
    orth = set(u["extra"].get("ortholog_entrez", []))
    return {**u, "gene_entrez": [g for g in u["gene_entrez"] if g not in orth]}


def curation() -> dict:
    units = read_units(D / "e1/clingen_units.tsv")
    anchors, s = E.score_units(units, load_docs(D / "e1/clingen_docs.json"),
                               {n: R(n, "clingen_docs") for n in ("L", "P")})
    cls = {r["curation_id"]: r["classification"] for r in csv.DictReader(
        open(ROOT / "data/external/clingen_gdv_2026-09-23/curations.tsv", encoding="utf-8"), delimiter="\t")}

    def compare(keep) -> dict:
        by = defaultdict(list)
        for i, u in enumerate(units):
            for c in u["extra"]["curation_ids"]:
                if keep(i, u) and cls[c] in ("Definitive", "Refuted", "Disputed"):
                    by[c].append(i)
        cids = sorted(by)
        a = np.array([cls[c] == "Definitive" for c in cids])
        out = {"n_curations": {k: sum(cls[c] == k for c in cids) for k in ("Definitive", "Refuted", "Disputed")},
               "n_units": {g: len({i for c, x in zip(cids, a) if x == (g == "Definitive") for i in by[c]})
                           for g in ("Definitive", "Refuted_or_Disputed")},
               "share_units_NONSCOR": {g: float(np.mean([("NONSCOR" in units[i]["classes"]) for c, x in zip(cids, a)
                                                         if x == (g == "Definitive") for i in by[c]]))
                                       for g in ("Definitive", "Refuted_or_Disputed")}, "readers": {}}
        for n in ("L", "P"):
            x = np.array([s[n][by[c]].mean() for c in cids])
            assert not np.isnan(x).any()
            r = E.contrast(x, a, np.array(cids))
            r["p_two_sided"] = min(1.0, 2 * min(r["p_greater"], r["p_less"]))
            out["readers"][n] = {**r, "mean_Definitive": float(x[a].mean()),
                                 "mean_Refuted_or_Disputed": float(x[~a].mean())}
        return out

    # post hoc (not preregistered): Refuted / Disputed curations cite more NONSCOR papers, and a unit's single pair may
    # link the gene to another disease, so the comparison is repeated on 'pair'-anchor units and on GEN_CLV units
    return {**compare(lambda i, u: True), "post_hoc": {
        "pair_anchor": compare(lambda i, u: anchors[i] == "pair"),
        "GEN_CLV_units": compare(lambda i, u: "GEN_CLV" in u["classes"])}}


def human() -> dict:
    sets = {"H1": ("clingen", "e1/clingen_units.tsv", "e1/clingen_docs.json", "clingen_docs", {"GEN_CLV"}, EXP),
            "H2": ("rgd", "e1/rgd_units.tsv", "e1/rgd_docs.json", "rgd_docs", {"IMP", "IGI"}, {"IEP"}),
            "H4": ("swap_clingen_u", "e1/swap_clingen_u_units.tsv", "e1/swap_clingen_u.json", "swap_clingen_u",
                   {"GEN_CLV"}, EXP)}
    out = {}
    for h, (name, uf, df, tag, A, B) in sets.items():
        orig, docs = read_units(D / uf), load_docs(D / df)
        units = [human_only(u) for u in orig]
        n_cand = [len(candidate_pairs(docs[u["doc_id"]], u["gene_entrez"], u["disease_mesh"])[0]) for u in units]
        assert set(n_cand) <= {0, 1}, "a unit gained candidates"   # removing ids only removes pairs here
        kept = [u for u, k in zip(units, n_cand) if k == 1]
        res = run(kept, docs, tag, ("L", "P", "Lp"), {h: (A, B)})["contrasts"][h]
        prim = json.loads((HERE.parent / f"e1/{name}.json").read_text(encoding="utf-8"))["contrasts"][h]["all"]
        out[h] = {"set": name, "n_units_e1": len(orig), "n_units_ortholog": sum("ortholog_entrez" in u["extra"] for u in orig),
                  "n_units_kept": len(kept), "human_only": res,
                  "primary": {"n_a": prim["n_a"], "n_b": prim["n_b"],
                              "stats": {n: {k: v[k] for k in ("auroc", "ci", "p_greater")} for n, v in prim["stats"].items()}}}
    for src in ("clingen", "rgd"):   # units single-candidate only under human ids: never scored (their docs were not)
        e1 = {u["unit_id"] for u in read_units(D / f"e1/{src}_units.tsv")}
        scored, docs = set(load_docs(D / f"e1/{src}_docs.json")), load_docs(D / f"{src}_docs.json")
        n = lambda u: len(candidate_pairs(docs[u["doc_id"]], u["gene_entrez"], u["disease_mesh"])[0])  # noqa: E731
        new = [u for u in read_units(D / f"{src}_units.tsv") if u["unit_id"] not in e1 and u["doc_id"] in docs
               and n(u) != 1 and n(human_only(u)) == 1]
        out[f"not_scored_{src}"] = {"n_new_single_candidate": len(new), "in_scored_docs": sum(u["doc_id"] in scored for u in new)}
    return out


def dev_f1() -> dict:
    out = {"bc8all": BC8ALL}
    for arm, pre in (("FL", "p2e3_f1"), ("J", "p2e3_j")):
        runs = sorted((ROOT / "runs").glob(f"{pre}_k*_s*/metrics.json"))
        assert len(runs) == 15, (arm, len(runs))
        m = [json.loads(p.read_text(encoding="utf-8")) for p in runs]
        if arm == "FL":   # fixed 12 epochs: the kept checkpoint is the last one
            assert all(r["best_epoch"] == 12 == r["history"][-1]["epoch"] for r in m)
        f = np.array([100 * r["best_dev_f1"] for r in m])
        out[arm] = {"runs": {r["name"]: round(float(x), 2) for r, x in zip(m, f)}, "mean": float(f.mean()),
                    "sd": float(f.std(ddof=1)), "min": float(f.min()), "max": float(f.max()),
                    "best_epoch": [r["best_epoch"] for r in m],
                    "epoch1_mean": float(np.mean([100 * r["history"][0]["dev_f1"] for r in m]))}
    return out


def fmt(s: dict) -> str:
    return f"{s['auroc']:.3f} [{s['ci'][0]:.3f}; {s['ci'][1]:.3f}]"


def markdown(res: dict) -> str:
    c, h, bt, l0, dv = (res[k] for k in ("curation", "human", "biotriplex", "percha_L0", "dev_f1"))
    L = ["# Paper 2: preregistered secondary analyses", "",
         "Generated by `secondary.py` (see its docstring for the rules); AUROC [95 % CI], PMID-cluster bootstrap unless "
         "stated, 10 000 resamples, seed 0. None of these is a confirmatory test.", "",
         "## 1. Curation level: Definitive vs Refuted or Disputed (exploratory)", "",
         f"Curations: {json.dumps(c['n_curations'])}; units: {json.dumps(c['n_units'])}. Curation bootstrap.", "",
         "| Reader | mean s, Definitive | mean s, Refuted/Disputed | AUROC (Definitive > R/D) | p two-sided |",
         "|---|---|---|---|---|"]
    L += [f"| {n}{sub} | {r['mean_Definitive']:.3f} | {r['mean_Refuted_or_Disputed']:.3f} | {fmt(r)} | {r['p_two_sided']:.4f} |"
          for sub, b in (("", c), *((f", post hoc {k} ({json.dumps(v['n_curations'])})", v)
                                    for k, v in c["post_hoc"].items()))
          for n, r in b["readers"].items()]
    L += ["", "## 2. Human Entrez identifiers only", "",
          "| Hyp. | Set | Units kept / E1 (ortholog units) | A / B | Statistic | AUROC human-only | AUROC primary |",
          "|---|---|---|---|---|---|---|"]
    for k in ("H1", "H2", "H4"):
        r, a = h[k], h[k]["human_only"]["all"]
        for n, s in a["stats"].items():
            L.append(f"| {k} | {r['set']} | {r['n_units_kept']} / {r['n_units_e1']} ({r['n_units_ortholog']}) | "
                     f"{a['n_a']} / {a['n_b']} (primary {r['primary']['n_a']} / {r['primary']['n_b']}) | {n} | "
                     f"{fmt(s)}, p(>) {s['p_greater']:.4f} | {fmt(r['primary']['stats'][n])} |")
    L += ["", "Units outside the E1 sets that become single-candidate only under human ids (not added: their documents, except the number in parentheses, were never scored): "
          + ", ".join(f"{k[11:]} {v['n_new_single_candidate']} ({v['in_scored_docs']} in scored docs)"
                      for k, v in h.items() if k.startswith("not_scored")), "",
          "## 3. BioTriplex (descriptive)", "", E.markdown(bt).split("\n", 1)[1],
          "## 4. Reader L0 on the construction set", "", E.markdown(l0).split("\n", 1)[1],
          "## 5. BioRED Dev micro-F1 of FL and J (15 runs each)", "",
          "| System | mean | sd | min | max | note |", "|---|---|---|---|---|---|",
          f"| bc8all (L) | {BC8ALL['mean']:.2f} | {BC8ALL['sd']:.2f} | {min(BC8ALL['seeds']):.2f} | {max(BC8ALL['seeds']):.2f} | seeds 42-44 |"]
    for arm, note in (("FL", f"epoch 12 (fixed); epoch 1 mean {dv['FL']['epoch1_mean']:.2f}"),
                      ("J", f"best epoch by dev, {min(dv['J']['best_epoch'])}-{max(dv['J']['best_epoch'])}")):
        r = dv[arm]
        L.append(f"| {arm} | {r['mean']:.2f} | {r['sd']:.2f} | {r['min']:.2f} | {r['max']:.2f} | {note} |")
    return "\n".join(L) + "\n"


def main() -> None:
    pe = {"H3": ({"MUT", "CAUSE"}, {"UP"})}
    res = {"meta": {"script": "docs/results/paper2/secondary/secondary.py", "n_boot": E.N_BOOT, "seed": E.SEED},
           "curation": curation(), "human": human(),
           "biotriplex": run(read_units(D / "biotriplex_units.tsv"), load_docs(D / "biotriplex_docs.json"),
                             "biotriplex_docs", ("L", "P", "Lp"),
                             {"CAUSAL_vs_LEVEL": ({"CAUSAL_MECH", "CAUSAL_MUT"}, {"LEVEL"})}),
           "percha_L0": run(read_units(D / "percha_units.tsv"), load_docs(D / "percha_docs.json"), "percha_docs",
                            ("L0",), pe),
           "dev_f1": dev_f1()}
    (HERE / "secondary.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    (HERE / "secondary.md").write_text(markdown(res), encoding="utf-8")
    print(markdown(res))


if __name__ == "__main__":
    main()
