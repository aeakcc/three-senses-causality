# Three Senses of Gene-Disease Causality in Biomedical Text

Code, evaluation units, per-unit predictions, results and the preregistration of the paper

> Y. Armanuly and Zh. Sadirmekova, "Three Senses of Gene-Disease Causality in Biomedical Text,"
> submitted to ICTACS-2026 (IEEE), 2026.

The paper separates three senses of "causal" for a gene-disease pair in text: the **level** change that BioRED
signs (L), the **perturbation** causality of CRED (P) and the ClinGen **evidence category** (V). Five preregistered
experiments measure how extractors trained under each convention score Medline construction sentences (E4),
curator-labeled ClinGen and RGD units (E1), gene-name swaps (E2), the CRED pool under probing, fine-tuning and joint
training (E3), and evidence types of 2026 ClinGen curations (E5).

## What is here

| Path | Content |
|---|---|
| `biocausal/` | the document-level extractor (BioLinkBERT-base, typed entity markers, ATLOP-style pair head) and its training / scoring code |
| `docs/results/paper2/*.py` | unit builders, evaluators of E1-E5, the exploratory separate-readout analysis (`e11d.py`), the independent verifier (`verify_p2.py`), the post hoc descriptive scripts (`excluded_units.py`, `e5_seeds.py`) and `export_predictions.py` |
| `docs/results/paper2/{e1,e2,e3,e5,e11d,secondary,verify}/` | every statistic of the paper as `.md` and `.json` |
| `docs/results/paper2/predictions/` | per-unit scores behind those statistics (see below) |
| `data/paper2/` | evaluation units: identifiers, curated classes and candidate status; no abstract text |
| `docs/preregistration_section11_ru.md` | the preregistration of this paper (section 11 of the project protocol, in Russian, as committed) |
| `ops/` | the exact training and scoring commands (queue files) |
| `docs/results/{cred,posthoc,resolution}/` | helper modules the evaluators import |

## Where each result comes from

| Paper | File |
|---|---|
| Table I, H1-H4 | `docs/results/paper2/e1/{clingen,rgd,percha,swap_clingen_u}.md` |
| Table I, H_E3a/b; Table II | `docs/results/paper2/e3/e3_results.md` |
| Table I, H_E5a/b; E5 per class | `docs/results/paper2/e5/e5.md`; seeds: `e5/e5_seeds.md` |
| Fig. 2 (constructions) | `docs/results/paper2/e1/percha.md` |
| Fig. 3, separate readouts (exploratory) | `docs/results/paper2/e11d/` |
| Name swaps (E2) | `docs/results/paper2/e2/`, `e1/swap_clingen_{u,k}.md` |
| Excluded ClinGen units | `docs/results/paper2/excluded_units.md` |

## Per-unit predictions

`docs/results/paper2/predictions/` holds one TSV per set, written by `export_predictions.py` from the evaluators'
own loaders (full float precision; recomputing the statistics from these files gives the numbers of the paper).

- `e1_clingen.tsv`, `e1_rgd.tsv`, `e4_percha.tsv`, `e2_swap_clingen_{u,k}.tsv`: `s_L`, `s_P`, `s_Lprime` (reader
  scores p(Positive_Correlation) + p(Negative_Correlation)) and `delta` = logit s_P - logit s_L' for the unit's
  candidate pair; `anchor` says whether the pair matches the curated disease (`pair`) or only the gene (`gene`).
- `e3_cred_pool.tsv`: out-of-fold score of every E3 arm (Z, C, S, Q, P0, PL, F0, FL, J) for the 3,362 pool rows,
  with the CRED label and the fold.
- `e5_test_2026.tsv`, `e5_rgd_ext.tsv`: evidence-type probabilities of the heads V0, VL, VP (3-seed ensembles; the
  2026 slice also per seed).

## Reproducing

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The units in `data/paper2/` are enough to re-run every evaluator once the predictions or models exist. Rebuilding
from scratch needs the public sources below (the builders in `docs/results/paper2/build_*.py` download or read
them) and PubTator3 annotations of each PMID (`pt3_docs.py`). Training and scoring commands are in `ops/queue_p2/`
(they were run on one 8 GB GPU); model checkpoints are not included because of their size and are available from
the authors. The joint model A2 and the reader L come from a companion study (under review).

## Data sources and licences

The code is under the MIT licence (`LICENSE`). The derived data keep the licences of their sources:

- ClinGen gene-disease validity curations, CC0 1.0 (not for direct diagnostic use without review by a genetics
  professional, as ClinGen asks);
- Rat Genome Database annotations, CC BY 4.0;
- Percha and Altman's Medline dependency paths (Zenodo 3459420), CC BY 4.0: `percha_units.tsv` includes the
  sentences of that release;
- Orphadata "Rare diseases with their associated genes" (release of 23 June 2026), CC BY 4.0;
- CRED labels (BIRDS Group, IIT Madras), GNU LGPL: `e3_cred_pool.tsv` includes them;
- BioRED and BioCreative VIII were used for training only and are not redistributed here.

## Notes

- The preregistration was committed to the project's private repository before each experiment ran (section 11a
  on 2026-09-23 19:07, 11b on 21:47, 11c on 22:43, clarifications on 23:09, exploratory 11d on 2026-09-24 10:29);
  this public repository is a later export, so those timestamps are not verifiable from its history.
- An AI assistant (Claude, Anthropic) wrote the code and ran the analyses under the authors' direction; the authors
  reviewed code and results.
