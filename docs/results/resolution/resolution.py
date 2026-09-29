"""Preregistration §8 item 1 (exploratory): how much BioRED-100 and BC8-400 can resolve.

No new predictions: the test-day files of test days 1-3. Reuses ``biocausal/bootstrap.py`` (per-document counts,
``resample``, the point self-check) with the same document draw ``bootstrap.main`` makes for a gold file (10 000
resamples, seed 0), so every resample below is paired with the test-day bootstrap json files, and the script asserts
that it reproduces their CIs and paired comparisons.

1. 95 % CI width of micro-F1 of every single run and ensemble scored on each set;
2. SD of micro-F1 across seeds per configuration;
3. MDE = (z_0.975 + z_0.8) * SE of the paired resampled difference (two-sided alpha 0.05, power 0.8) for
   (a) every pair of seeds of one configuration, (b) the ensemble pairs of the test-day comparisons;
4. Holm-Bonferroni at alpha 0.05 jointly over every paired test-set comparison of the three bootstrap json files;
5. forest.svg / forest.png, resolution.json, resolution.md.

    .venv/bin/python docs/results/resolution/resolution.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from itertools import combinations
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from biocausal.bootstrap import check_point, doc_counts, metrics, resample  # noqa: E402
from biocausal.data import load_docs, unpairable_gold  # noqa: E402
from biocausal.evaluate import read_jsonl  # noqa: E402

RES = ROOT / "docs/results"
OUT = RES / "resolution"
N, SEED, ALPHA, POWER = 10000, 0, 0.05, 0.8
Z = NormalDist().inv_cdf(1 - ALPHA / 2) + NormalDist().inv_cdf(POWER)          # 2.8016
SETS = {"biored_test": ("BioRED-100", ROOT / "data/BioRED/Test.BioC.JSON"),
        "bc8": ("BC8-400", ROOT / "data/BC8/bc8_test.json")}
BOOT = {"1": RES / "testday_w1/bootstrap_and_selection.json", "2": RES / "testday_w2/bootstrap_and_criteria.json",
        "3": RES / "testday_w3/bootstrap_and_criteria.json"}
ENS_NAME = {"pred_test_5seed.jsonl": "ens_re_biolinkbert_base_5seed",               # the names the json files use
            "pred_test_3seed.jsonl": "ens_re_biolinkbert_base_3seed_context"}
ALIAS = {("bc8", "ens_re_biolinkbert_base_5seed"): "ens_re_biolinkbert_base"}        # the 5-seed BioRED-only on BC8
SHORT = {"re_biolinkbert_base_5seed": "BioRED-only", "crossfit_ens": "cross-fit BC8"}
SEED_TOKEN = re.compile(r"_s\d+$")
TOL = 1e-6                                                                           # the json files round to 6 digits


def read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def holm(p: list[float], alpha: float = ALPHA) -> tuple[list[float], list[bool]]:
    """Holm-Bonferroni step-down: adjusted p (monotone, capped at 1) and which hypotheses are rejected at alpha."""
    adj, run = [0.0] * len(p), 0.0
    for rank, i in enumerate(sorted(range(len(p)), key=p.__getitem__)):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj, [a <= alpha for a in adj]


def inventory(tag: str) -> dict[str, list[Path]]:
    """Every single run and ensemble scored on this set on test days 1-3 (the §6.2 repro files are duplicates)."""
    p = f"pred_{tag}.jsonl"
    single = sorted(RES.glob(f"testday_w?/runs/*_s4?/{p}")) + sorted(RES.glob(f"testday_w1/runs/_testday_w1/crossfit_s4?/{p}"))
    ens = sorted(RES.glob(f"testday_w?/runs/_testday_w?/ens_*/{p}")) + sorted(RES.glob(f"testday_w1/runs/_testday_w1/crossfit_ens/{p}"))
    if tag == "biored_test":   # BioRED-only s42-46: old-protocol pred_test.jsonl, test seen before the preregistration (§0)
        single += sorted(RES.glob("testday_w1/runs/re_biolinkbert_base_s4?/pred_test.jsonl"))
        ens += sorted((RES / "testday_w1/ens_biored_only").glob("pred_test_*seed.jsonl"))
    return {"single": single, "ensemble": ens}


def score(tag: str) -> dict[str, dict]:
    """name -> kind, test day, config, file, point micro-F1 and its 10 000 resampled values (fractions)."""
    docs = load_docs(SETS[tag][1])
    pmids, unp = [d["pmid"] for d in docs], {d["pmid"]: unpairable_gold(d) for d in docs}
    want = {(d["pmid"], p["e1"], p["e2"]) for d in docs for p in d["pairs"]}
    w = np.random.default_rng(SEED).multinomial(len(pmids), np.full(len(pmids), 1 / len(pmids)), size=N)  # = bootstrap.main
    out: dict[str, dict] = {}
    for kind, files in inventory(tag).items():
        for f in files:
            name = ENS_NAME.get(f.name, f.parent.name)
            assert name not in out, f"duplicate system name {name}"
            rows = read_jsonl(f)
            assert len(rows) == len(want) and {(r["pmid"], r["e1"], r["e2"]) for r in rows} == want, f
            c = doc_counts(rows, unp, pmids)
            check_point(rows, unp, c)
            out[name] = {"kind": kind, "test_day": f.relative_to(RES).parts[0][-1], "file": str(f.relative_to(ROOT)),
                         "config": SEED_TOKEN.sub("", name) if kind == "single" else None, "n_docs": len(pmids),
                         "point": float(metrics(c.sum(0))["micro_f1"]), "boot": resample(c, w)["micro_f1"]}
    return out


def pp(x: float) -> float:
    return round(100 * x, 3)


def spread(xs: list[float]) -> dict:
    return {"n": len(xs), "median": round(statistics.median(xs), 3), "min": round(min(xs), 3), "max": round(max(xs), 3)}


def short(name: str) -> str:
    name = name.removeprefix("ens_").removeprefix("re_bl_base_")
    return SHORT.get(name, name)


def main() -> dict:
    S = {tag: score(tag) for tag in SETS}
    out: dict = {"what": "preregistration §8 item 1 (exploratory): resolution of BioRED-100 and BC8-400",
                 "n_resamples": N, "seed": SEED, "alpha": ALPHA, "power": POWER, "z_sum": round(Z, 4),
                 "units": "micro-F1, percentage points",
                 "ci": "2.5 / 97.5 percentiles of the document bootstrap (biocausal/bootstrap.py, same draw as the "
                       "test-day json files)",
                 "mde": "MDE = (z_0.975 + z_0.8) * SE, SE = SD (ddof 1) of the 10 000 paired resampled differences",
                 "p_two_sided": "2 * min(share of resamples with a - b <= 0, share with a - b >= 0), capped at 1"}

    # ---- the test-day json rows must be reproduced exactly (same draw, same counting) ----------------------------
    n_rows = 0
    for td, path in BOOT.items():
        for key, row in read(path)["rows"].items():
            tag, name = key.split(" ", 1)
            if tag not in SETS:                     # the 200-document BC8 halves of the cross-fit
                continue
            s, m = S[tag][name], row["micro_f1"]
            lo, hi = np.percentile(s["boot"], [2.5, 97.5])
            assert max(abs(s["point"] - m["point"]), abs(lo - m["lo"]), abs(hi - m["hi"])) <= TOL, key
            n_rows += 1
    out["reproduced_json_rows"] = n_rows

    # ---- 1-3 per set: CI widths, seed SD, MDE of two seeds of one configuration ----------------------------------
    for tag, (label, gold) in SETS.items():
        systems, ci = S[tag], {}
        for name, s in systems.items():
            lo, hi = np.percentile(s["boot"], [2.5, 97.5])
            ci[name] = {k: s[k] for k in ("kind", "test_day", "config", "file")} | {
                "micro_f1": pp(s["point"]), "lo": pp(lo), "hi": pp(hi), "width": pp(hi - lo)}
        configs: dict[str, list[str]] = {}
        for name, s in systems.items():
            if s["kind"] == "single":
                configs.setdefault(s["config"], []).append(name)
        sd = {}
        for cfg, names in configs.items():
            v = [100 * systems[n]["point"] for n in names]
            sd[cfg] = {"test_day": systems[names[0]]["test_day"], "runs": names, "micro_f1_per_seed": [round(x, 3) for x in v],
                       "mean": round(statistics.mean(v), 3), "sd": round(statistics.stdev(v), 3) if len(v) > 1 else None}
        seed_pairs = []
        for cfg, names in configs.items():
            for a, b in combinations(names, 2):
                se = 100 * float(np.std(systems[a]["boot"] - systems[b]["boot"], ddof=1))
                seed_pairs.append({"config": cfg, "a": a, "b": b, "se": round(se, 3), "mde": round(Z * se, 3)})
        out[tag] = {
            "label": label, "gold": str(gold.relative_to(ROOT)), "n_docs": next(iter(systems.values()))["n_docs"],
            "ci_width": {k: spread([c["width"] for c in ci.values() if c["kind"] == k]) for k in ("single", "ensemble")},
            "systems": ci,
            "seed_sd": sd, "seed_sd_summary": spread([c["sd"] for c in sd.values() if c["sd"] is not None]),
            "mde_two_seeds_same_config": {"pairs": "every pair of seeds of every configuration with >= 2 seeds",
                                          "se": spread([p["se"] for p in seed_pairs]),
                                          "mde": spread([p["mde"] for p in seed_pairs]), "rows": seed_pairs}}

    # ---- 4. every paired test-set comparison of test days 1-3, recomputed, then Holm jointly ---------------------
    comps = []
    for td, path in BOOT.items():
        for key, v in read(path)["paired"].items():
            tag, pair = key.split(" ", 1)
            a, b = pair.split(" vs ")
            m = v["a_minus_b"]["micro_f1"]
            sa, sb = S[tag][a], S[tag][ALIAS.get((tag, b), b)]
            d = sa["boot"] - sb["boot"]
            lo, hi = np.percentile(d, [2.5, 97.5])
            le, ge = float(np.mean(d <= 0)), float(np.mean(d >= 0))
            assert max(abs(sa["point"] - sb["point"] - m["point"]), abs(lo - m["lo"]), abs(hi - m["hi"]),
                       abs(le - m["p_one_sided"])) <= TOL, key
            se = 100 * float(np.std(d, ddof=1))
            comps.append({"test_day": td, "comparison": key, "set": tag, "a": a, "b": b, "files": [sa["file"], sb["file"]],
                          "delta": pp(m["point"]), "lo": pp(m["lo"]), "hi": pp(m["hi"]), "p_one_sided": m["p_one_sided"],
                          "p_two_sided": min(1.0, 2 * min(le, ge)), "se": round(se, 3), "mde": round(Z * se, 3),
                          "abs_delta_ge_mde": abs(pp(m["point"])) >= Z * se})
    adj2, keep2 = holm([c["p_two_sided"] for c in comps])
    adj1, keep1 = holm([c["p_one_sided"] for c in comps])
    for c, a2, k2, a1, k1 in zip(comps, adj2, keep2, adj1, keep1):
        c |= {"holm_p": round(a2, 6), "holm_survives": k2, "holm_one_sided_p": round(a1, 6), "holm_one_sided_survives": k1}
    for tag in SETS:
        mine = [c for c in comps if c["set"] == tag]
        out[tag]["mde_two_ensembles"] = {"pairs": "the ensemble pairs of every test-day comparison on this set",
                                         "se": spread([c["se"] for c in mine]), "mde": spread([c["mde"] for c in mine]),
                                         "n_abs_delta_ge_mde": sum(c["abs_delta_ge_mde"] for c in mine)}

    # the six paired dev bootstraps (criteria of §2, §5, §7) are not test-set comparisons; with them m = 38
    w1, w2, w3 = (read(p) for p in BOOT.values())
    dev = {"td1 dev ens_atl vs ens_re_biolinkbert_base_3seed (§2)": w1["dev_criteria"]["atl"]["dev_paired_vs_biored_only_3seed"],
           "td2 dev ens_mst_joint_soft vs ens_mst_joint_drop (§5 criterion 2)": w2["criteria"]["2"]["paired_dev"],
           **{f"td3 dev ens_{c} vs ens_bc8all (§7)": v for c, v in w3["paired_dev_vs_bc8all"].items()}}
    dev_p = {k: min(1.0, 2 * min(p, 1 - p)) for k, v in dev.items() for p in [v["a_minus_b"]["micro_f1"]["p_one_sided"]]}
    _, keep38 = holm([c["p_two_sided"] for c in comps] + list(dev_p.values()))
    out["holm"] = {
        "m": len(comps), "per_test_day": {td: sum(c["test_day"] == td for c in comps) for td in BOOT},
        "per_set": {tag: sum(c["set"] == tag for c in comps) for tag in SETS},
        "rule": "Holm-Bonferroni step-down at alpha 0.05 over all m comparisons jointly, on p_two_sided; "
                "holm_one_sided_* repeats it on the reported one-sided p ('a better than b')",
        "p_resolution": 1 / N,
        "same_survivors_with_zero_p_as_2_over_n_plus_1": holm([max(c["p_two_sided"], 2 / (N + 1)) for c in comps])[1] == keep2,
        "n_nominal_two_sided_p_below_alpha": sum(c["p_two_sided"] < ALPHA for c in comps),
        "survivors": [c["comparison"] for c in comps if c["holm_survives"]],
        "survivors_one_sided": [c["comparison"] for c in comps if c["holm_one_sided_survives"]],
        "dev_comparisons_not_included": {"two_sided_p_from_json": {k: round(v, 4) for k, v in dev_p.items()},
                                         "survivors_if_included_m38": [c["comparison"] for c, k in zip(comps, keep38) if k],
                                         "dev_survivors_if_included": [k for k, s in zip(dev_p, keep38[len(comps):]) if s]}}
    out["comparisons"] = comps

    forest(comps)
    (OUT / "resolution.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "resolution.md").write_text(report(out), encoding="utf-8")
    print(f"-> {OUT}: m = {len(comps)}, survivors {out['holm']['survivors']}")
    return out


def forest(comps: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory

    ink, ink2, muted, grid, base, accent = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#2a78d6"
    plt.rcParams.update({"font.size": 8, "svg.hashsalt": "resolution", "axes.edgecolor": base,
                         "xtick.color": ink2, "ytick.color": ink2, "xtick.labelcolor": ink2})
    groups = [(tag, [c for c in comps if c["set"] == tag]) for tag in SETS]
    rows = [len(g) for _, g in groups]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8.6, 0.2 * sum(rows) + 2.3),
                             gridspec_kw={"height_ratios": [r + 1.6 for r in rows], "hspace": 0.2})
    fig.subplots_adjust(left=0.33, right=0.70, top=0.885, bottom=0.07)
    axes[0].xaxis.set_major_locator(plt.MultipleLocator(2))
    for ax, (tag, g) in zip(axes, groups):
        y = np.arange(len(g))
        ax.axvline(0, color=ink2, lw=0.8, zorder=1)
        for yi, c in zip(y, g):
            col = accent if c["holm_survives"] else muted
            ax.hlines(yi, c["lo"], c["hi"], color=col, lw=1.6, zorder=2, capstyle="round")
            ax.plot(c["delta"], yi, "o", ms=5, color=col, mec="white", mew=1, zorder=3)
        ax.set_ylim(len(g) - 0.4, -1.2)
        ax.set_yticks(y, [f"TD{c['test_day']}  {short(c['a'])} − {short(c['b'])}" for c in g])
        for lab, c in zip(ax.get_yticklabels(), g):
            lab.set_color(ink if c["holm_survives"] else ink2)
            lab.set_fontweight("bold" if c["holm_survives"] else "normal")
        ax.tick_params(axis="y", length=0, pad=6)
        ax.tick_params(axis="x", labelbottom=True, color=base)
        ax.grid(axis="x", color=grid, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        tr = blended_transform_factory(fig.transFigure, ax.transData)
        ax.text(0.02, -1.0, f"{SETS[tag][0]}  ·  {len(g)} comparisons", transform=tr, ha="left", va="center",
                fontsize=9, fontweight="bold", color=ink)
        ax.text(0.715, -1.0, "Δ [95% CI]", transform=tr, ha="left", va="center", color=ink2, fontweight="bold")
        ax.text(0.985, -1.0, "p Holm", transform=tr, ha="right", va="center", color=ink2, fontweight="bold")
        for yi, c in zip(y, g):
            ax.text(0.715, yi, f"{sgn(c['delta'])} [{sgn(c['lo'])}; {sgn(c['hi'])}]", transform=tr, ha="left", va="center",
                    color=ink if c["holm_survives"] else ink2)
            ax.text(0.985, yi, fmt_p(c["holm_p"], 3), transform=tr, ha="right", va="center",
                    color=ink if c["holm_survives"] else ink2)
    axes[-1].set_xlabel("Difference in micro-F1, A − B (percentage points)", color=ink2)
    fig.text(0.02, 0.975, "Paired test-set comparisons of test days 1–3", fontsize=11, fontweight="bold", color=ink, va="top")
    fig.text(0.02, 0.948, f"TD = test day. Ensemble A − ensemble B, document bootstrap ({N:,} resamples, seed {SEED});\n"
             f"Holm–Bonferroni over all {len(comps)} comparisons, α = {ALPHA}, two-sided", color=ink2, va="top")
    fig.legend(handles=[Line2D([], [], color=accent, lw=1.6, marker="o", ms=5, mec="white", label="survives Holm"),
                        Line2D([], [], color=muted, lw=1.6, marker="o", ms=5, mec="white", label="does not survive")],
               loc="upper right", bbox_to_anchor=(0.985, 0.985), ncol=2, frameon=False, labelcolor=ink2)
    fig.savefig(OUT / "forest.svg", facecolor="white", metadata={"Date": None})
    fig.savefig(OUT / "forest.png", dpi=300, facecolor="white")
    plt.close(fig)


def sgn(x: float, nd: int = 1) -> str:
    return f"{x:+.{nd}f}".replace("-", "−")


def fmt_p(p: float, nd: int = 4) -> str:
    return f"< {10 ** -nd:.{nd}f}" if p < 10 ** -nd else ("1" if p >= 1 else f"{p:.{nd}f}")


def ru(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def report(o: dict) -> str:
    """resolution.md: Russian, facts only, every number from ``o``."""
    b, c8, h, comps = o["biored_test"], o["bc8"], o["holm"], o["comparisons"]
    med = lambda s: f"{s['median']:.1f} [{s['min']:.1f}; {s['max']:.1f}]"          # noqa: E731
    med_ru = lambda s: f"{ru(s['median'])} [{ru(s['min'])}; {ru(s['max'])}]"        # noqa: E731
    name = lambda c: f"`{short(c['a'])}` − `{short(c['b'])}`"                          # noqa: E731
    lines = [
        "# Разрешающая способность тестов BioRED-100 и BC8-400 (§8, п. 1)", "",
        "Анализ exploratory (§8), выбор системы по §4 не меняет. Новых прогнозов нет: файлы прогнозов дней теста 1–3. "
        f"Document bootstrap: 10 000 ресэмплов, seed {SEED}, `biocausal/bootstrap.py` (та же выборка документов, что в "
        "`bootstrap_and_*.json`). Все числа — micro-F1 в пунктах. Скрипт `resolution.py`, все числа `resolution.json`, "
        "рисунок `forest.svg`, `forest.png`.", "",
        f"Проверка: скрипт воспроизвёл все {o['reproduced_json_rows']} строк CI на BioRED-100 и BC8-400 из трёх файлов "
        f"bootstrap и все {h['m']} парных сравнения (разница, CI, односторонний p) с точностью 1e-6.", "",
        "## 1. Ширина 95 % CI", "",
        "| Набор | Документов | Одиночные прогоны: n, медиана [мин; макс] | Ансамбли: n, медиана [мин; макс] |",
        "|---|---|---|---|"]
    for s in (b, c8):
        lines.append(f"| {s['label']} | {s['n_docs']} | {s['ci_width']['single']['n']}: {med(s['ci_width']['single'])} | "
                     f"{s['ci_width']['ensemble']['n']}: {med(s['ci_width']['ensemble'])} |")
    lines += ["", "Одиночные прогоны — все, что оценены в дни теста 1–3 (на BC8-400 с cross-fit s42–44), плюс BioRED-only "
              "s42–46 на BioRED-100 по `pred_test.jsonl` старого протокола (тест виден до предрегистрации). Ансамбли — все "
              "`ens_*` дней теста, cross-fit и ансамбли BioRED-only 5 и 3 сидов. Половины BC8 (200 документов) не входят. "
              "Ширина каждой системы — `resolution.json`, блок `<набор>.systems`.", "",
              "## 2. SD micro-F1 по сидам", "",
              "| Конфигурация | День теста | BioRED-100: сидов, среднее ± SD | BC8-400: сидов, среднее ± SD |",
              "|---|---|---|---|"]
    for cfg in dict.fromkeys([*b["seed_sd"], *c8["seed_sd"]]):
        cells = [f"{len(s['runs'])}: {s['mean']:.1f} ± {s['sd']:.1f}" if s else "—"
                 for s in (b["seed_sd"].get(cfg), c8["seed_sd"].get(cfg))]
        td = (b["seed_sd"].get(cfg) or c8["seed_sd"].get(cfg))["test_day"]
        lines.append(f"| `{short(cfg)}` | {td} | {cells[0]} | {cells[1]} |")
    lines += ["", f"Медиана SD [мин; макс]: BioRED-100 {med(b['seed_sd_summary'])} ({b['seed_sd_summary']['n']} конфигураций), "
              f"BC8-400 {med(c8['seed_sd_summary'])} ({c8['seed_sd_summary']['n']}). У `mst_seq_drop` 2 сида.", "",
              f"## 3. Минимальный обнаружимый эффект (α = 0,05 двусторонний, мощность 0,8)", "",
              f"MDE = {ru(o['z_sum'], 2)} × SE, SE — стандартное отклонение 10 000 парных ресэмплированных разниц.",
              "", "| Набор | (a) два сида одной конфигурации: пар, SE, MDE медиана [мин; макс] | "
              "(b) два ансамбля: пар, SE, MDE медиана [мин; макс] | Сравнений с \\|Δ\\| ≥ MDE |", "|---|---|---|---|"]
    for s in (b, c8):
        a_, e_ = s["mde_two_seeds_same_config"], s["mde_two_ensembles"]
        lines.append(f"| {s['label']} | {a_['mde']['n']}: SE {a_['se']['median']:.2f}, MDE {med(a_['mde'])} | "
                     f"{e_['mde']['n']}: SE {e_['se']['median']:.2f}, MDE {med(e_['mde'])} | "
                     f"{e_['n_abs_delta_ge_mde']} из {e_['mde']['n']} |")
    lines += ["", "Пары: (a) все пары сидов внутри каждой конфигурации с ≥ 2 сидами (список — `mde_two_seeds_same_config.rows`); "
              "(b) пары ансамблей всех сравнений дней теста на этом наборе (раздел 4, MDE каждого — `comparisons`).", "",
              "## 4. Поправка Холма–Бонферрони", "",
              f"Сравнений в дни теста 1–3: **{h['m']}** (день 1 — {h['per_test_day']['1']}, день 2 — {h['per_test_day']['2']}, "
              f"день 3 — {h['per_test_day']['3']}; BioRED-100 — {h['per_set']['biored_test']}, BC8-400 — {h['per_set']['bc8']}). "
              f"Число 32 подтверждено. Поправка Холма при α = 0,05 по всем {h['m']} сразу, p двусторонний = 2 × min(доля "
              "ресэмплов с разницей ≤ 0, доля ≥ 0). Разрешение p — 1e-4; при замене нулевого p на 2/10 001 выжившие те же"
              f"{'' if h['same_survivors_with_zero_p_as_2_over_n_plus_1'] else ' — НЕТ, см. json'}. "
              f"До поправки p < 0,05 у {h['n_nominal_two_sided_p_below_alpha']} из {h['m']}.", "",
              "| № | День | Набор | A − B | Разница [CI 95 %] | p одност. | p двуст. | p Холма | MDE | Выживает |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(comps, 1):
        yes = "**да**" if c["holm_survives"] else "нет"
        lines.append(f"| {i} | {c['test_day']} | {SETS[c['set']][0]} | {name(c)} | {sgn(c['delta'])} [{sgn(c['lo'])}; "
                     f"{sgn(c['hi'])}] | {fmt_p(c['p_one_sided'])} | {fmt_p(c['p_two_sided'])} | {fmt_p(c['holm_p'])} | "
                     f"{c['mde']:.1f} | {yes} |")
    surv = [c for c in comps if c["holm_survives"]]
    surv1 = [c for c in comps if c["holm_one_sided_survives"]]
    dev = h["dev_comparisons_not_included"]
    lines += ["", f"**Выживают {len(surv)} из {h['m']}:** " + "; ".join(
                  f"{name(c)} на {SETS[c['set']][0]} ({sgn(c['delta']).replace('.', ',')}, p Холма {fmt_p(c['holm_p']).replace('.', ',')})"
                  for c in surv) + ".", "",
              f"По одностороннему p («A лучше B», как в отчётах дней теста) выживают {len(surv1)}: "
              + "; ".join(f"{name(c)} на {SETS[c['set']][0]}" for c in surv1) + ".", "",
              f"Шесть парных bootstrap на dev (критерии §2, §5, §7) — не сравнения на тестах, в {h['m']} не входят. "
              f"С ними (m = 38) выживших на тестах {len(dev['survivors_if_included_m38'])}, те же; из dev-сравнений "
              f"выживает {len(dev['dev_survivors_if_included'])}.", "",
              "## 5. Что может разрешить каждый тест", "",
              f"На BioRED-100 95 % CI одной системы шириной {ru(b['ci_width']['ensemble']['median'])} пункта у ансамблей и "
              f"{ru(b['ci_width']['single']['median'])} у одиночных прогонов (медианы). MDE при α = 0,05 и мощности 0,8 — "
              f"{med_ru(b['mde_two_ensembles']['mde'])} пункта для двух ансамблей и {med_ru(b['mde_two_seeds_same_config']['mde'])} "
              f"для двух сидов одной конфигурации; SD по сидам — {med_ru(b['seed_sd_summary'])}. С мощностью 0,8 BioRED-100 "
              f"обнаруживает разницу двух ансамблей от {ru(b['mde_two_ensembles']['mde']['min'])} пункта у самой близкой пары "
              f"и от {ru(b['mde_two_ensembles']['mde']['median'])} у типичной; |Δ| ≥ MDE у "
              f"{b['mde_two_ensembles']['n_abs_delta_ge_mde']} из {h['per_set']['biored_test']} сравнения. На BC8-400 CI "
              f"{ru(c8['ci_width']['ensemble']['median'])} пункта у ансамблей и {ru(c8['ci_width']['single']['median'])} у "
              f"одиночных, MDE двух ансамблей {med_ru(c8['mde_two_ensembles']['mde'])}, двух сидов "
              f"{med_ru(c8['mde_two_seeds_same_config']['mde'])}, SD по сидам {med_ru(c8['seed_sd_summary'])}: с мощностью 0,8 "
              f"обнаруживается разница двух ансамблей от {ru(c8['mde_two_ensembles']['mde']['min'])} пункта у самой близкой "
              f"пары и от {ru(c8['mde_two_ensembles']['mde']['median'])} у типичной, |Δ| ≥ MDE у "
              f"{c8['mde_two_ensembles']['n_abs_delta_ge_mde']} из {h['per_set']['bc8']}. После поправки Холма по всем "
              f"{h['m']} сравнениям значимы {len(surv)}.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
