# §11d item 2: geometry of the L and P* readouts (exploratory)

5345 CRED D-G pairs, 261 PMIDs; P* trained on 2680 rows (345 causal). CI: 500 PMID resamples; null: 200 label permutations. Shortcut: bootstrap and null refit at the full-data alpha and C of each seed (no per-resample selection, for runtime).

| Seed | cos(w_L, w_P) [95 % CI] | null 2.5 / 97.5 % | P(abs null >= abs obs) |
|---|---|---|---|
| 42 | 0.026 [-0.022; 0.070] | -0.071 / 0.074 | 0.485 |
| 43 | 0.030 [-0.018; 0.079] | -0.085 / 0.075 | 0.500 |
| 44 | 0.064 [0.012; 0.113] | -0.085 / 0.079 | 0.165 |
| mean | 0.040 [-0.005; 0.080] | -0.070 / 0.072 | 0.305 |

| Seed | C | alpha | ridge OOF R2 | surrogate OOF Spearman | Spearman(w_P proj, s_L) | Pearson(w_L proj, w_P proj) | AUROC w_L proj (pool, reachable) |
|---|---|---|---|---|---|---|---|
| 42 | 0.001 | 1000 | 0.849 | 0.920 | 0.715 | 0.739 | 0.851 [0.804; 0.893] |
| 43 | 0.001 | 1000 | 0.873 | 0.931 | 0.742 | 0.752 | 0.849 [0.801; 0.892] |
| 44 | 0.001 | 1000 | 0.857 | 0.932 | 0.716 | 0.723 | 0.849 [0.800; 0.892] |

s_L itself on the same pool rows: AUROC 0.828 [0.777; 0.876].

Cosine between seeds: w_P 42-43 0.700, 42-44 0.688, 43-44 0.647; w_L 42-43 0.477, 42-44 0.440, 43-44 0.422.
