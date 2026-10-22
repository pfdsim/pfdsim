# Nannoolal groups for absolute aqueous solubility

Predict log10(S / (mol/L)) from molecular counts. This compares learned regressions on structural encodings, not Nannoolal boiling-point predictions or UNIFAC activity coefficients.

## Method

- Reuse the previous 10,421 deduplicated water labels, atom/UNIFAC feature matrices and five-fold molecular splits repeated five times. The principal domain is the same neutral, single-component, carbon-containing population of 9,044 structures.
- Extract Nannoolal counts using pfdsim nannoolal_method.fragment. First-order silicon environments are retained with each native recorded environment attribute.
- Compare first-order group counts, and first-order counts plus native second-order structural corrections. No group-interaction pair features, estimated pure-component properties or property contribution values are used.
- Native property-specific do_not_estimate flags (e.g. COOH/NH2 coexistence) are recorded but not automatically excluded: we fit aqueous-solubility coefficients independently, rather than invoking those pure-property equations. Failed structural assignments are unavailable. Native local-extension structural group 219 is retained and disclosed in the fragmentation audit; no fitted property values enter the regression.
- Use the identical regression_job implementation, training-only scaling, OLS, three-fold inner CV, ridge grid and optional nonlinear size bases as the preceding atom/UNIFAC assessment. Hyperparameters are selected on training validation errors only.
- Each pool refits all included regressions on exactly the same chemical identities and outer molecular folds. Pairwise UNIFAC intersections and the all-encoding common intersection control coverage differences.
- A prediction is invalid if nonfinite or outside [-30, 3] log10 mol/L. This deliberately generous numerical screen is not a rigorous physical bound or a fitted threshold. Do not clip predictions or delete their original outputs. Invalid counts accompany every score.
- Main tables score the same molecules where EVERY model in the pool is valid in EVERY repeat. Paired comparisons require both compared models valid in every repeat and explicitly report exclusions. Conditional accuracy with exclusions is not full-population accuracy; compare invalid rates as well.
- Paired bootstrap uses 20,000 draws of whole molecules and split repetitions, seed 20261022. Familywise intervals cover all 50 displayed comparisons. Bootstrap comparisons run in five or fewer processes; no model is selected by test performance.
- Sparse descriptors absent in training receive zero slope and are counted by the shared regression implementation. Absolute solubility includes solid-state and ionization effects; random molecule CV is not scaffold-disjoint or external validation.

## Coverage

```json
{
  "eligible_neutral_organic": 9044,
  "Nannoolal_available": 7278,
  "first_order_columns": 115,
  "with_corrections_columns": 131,
  "native_property_refusal_flags": 234,
  "local_extension_219_molecules": 12,
  "pool_counts": {
    "Nannoolal_own": 7278,
    "with_UNIFAC": 4118,
    "with_UNIFDMD": 4198,
    "with_UNIFNIST": 4588,
    "with_UNIFLBY": 2010,
    "common_all": 1888
  }
}
```

## Nannoolal_own

| Model | Common valid molecules | MAE log10 | RMSE log10 | R² | Invalid predictions / total |
|---|---:|---:|---:|---:|---:|
| size_selected | 7,272 | 1.4465 | 1.8698 | 0.3564 | 5 / 36390 |
| element_counts_OLS | 7,272 | 1.0746 | 1.3887 | 0.6450 | 5 / 36390 |
| element_counts_adaptive | 7,272 | 1.0079 | 1.3042 | 0.6869 | 16 / 36390 |
| Nannoolal_first_OLS | 7,272 | 0.8889 | 1.2074 | 0.7316 | 12 / 36390 |
| Nannoolal_first_adaptive | 7,272 | 0.8308 | 1.1207 | 0.7688 | 17 / 36390 |
| Nannoolal_corrected_OLS | 7,272 | 0.8832 | 1.2007 | 0.7346 | 12 / 36390 |
| Nannoolal_corrected_adaptive | 7,272 | 0.8268 | 1.1173 | 0.7702 | 18 / 36390 |

## with_UNIFAC

| Model | Common valid molecules | MAE log10 | RMSE log10 | R² | Invalid predictions / total |
|---|---:|---:|---:|---:|---:|
| size_selected | 4,112 | 1.4573 | 1.8813 | 0.4079 | 0 / 20590 |
| element_counts_OLS | 4,112 | 1.0038 | 1.3242 | 0.7066 | 0 / 20590 |
| element_counts_adaptive | 4,112 | 0.9425 | 1.2292 | 0.7472 | 12 / 20590 |
| Nannoolal_first_OLS | 4,112 | 0.8365 | 1.1529 | 0.7776 | 13 / 20590 |
| Nannoolal_first_adaptive | 4,112 | 0.7635 | 1.0304 | 0.8224 | 12 / 20590 |
| Nannoolal_corrected_OLS | 4,112 | 0.8274 | 1.1426 | 0.7816 | 12 / 20590 |
| Nannoolal_corrected_adaptive | 4,112 | 0.7574 | 1.0252 | 0.8242 | 12 / 20590 |
| UNIFAC_subgroup_OLS | 4,112 | 0.8543 | 1.1662 | 0.7725 | 2 / 20590 |
| UNIFAC_subgroup_adaptive | 4,112 | 0.7810 | 1.0536 | 0.8143 | 10 / 20590 |

## with_UNIFDMD

| Model | Common valid molecules | MAE log10 | RMSE log10 | R² | Invalid predictions / total |
|---|---:|---:|---:|---:|---:|
| size_selected | 4,194 | 1.4275 | 1.8409 | 0.4296 | 0 / 20990 |
| element_counts_OLS | 4,194 | 1.0102 | 1.3286 | 0.7029 | 0 / 20990 |
| element_counts_adaptive | 4,194 | 0.9342 | 1.2210 | 0.7490 | 0 / 20990 |
| Nannoolal_first_OLS | 4,194 | 0.8377 | 1.1497 | 0.7775 | 3 / 20990 |
| Nannoolal_first_adaptive | 4,194 | 0.7468 | 1.0078 | 0.8290 | 4 / 20990 |
| Nannoolal_corrected_OLS | 4,194 | 0.8293 | 1.1405 | 0.7811 | 4 / 20990 |
| Nannoolal_corrected_adaptive | 4,194 | 0.7430 | 1.0064 | 0.8295 | 3 / 20990 |
| UNIFDMD_subgroup_OLS | 4,194 | 0.8548 | 1.1633 | 0.7722 | 3 / 20990 |
| UNIFDMD_subgroup_adaptive | 4,194 | 0.7688 | 1.0346 | 0.8198 | 8 / 20990 |

## with_UNIFNIST

| Model | Common valid molecules | MAE log10 | RMSE log10 | R² | Invalid predictions / total |
|---|---:|---:|---:|---:|---:|
| size_selected | 4,575 | 1.4917 | 1.9209 | 0.3864 | 0 / 22940 |
| element_counts_OLS | 4,575 | 1.0542 | 1.3832 | 0.6819 | 0 / 22940 |
| element_counts_adaptive | 4,575 | 0.9837 | 1.2846 | 0.7256 | 12 / 22940 |
| Nannoolal_first_OLS | 4,575 | 0.8667 | 1.1889 | 0.7650 | 22 / 22940 |
| Nannoolal_first_adaptive | 4,575 | 0.8045 | 1.0848 | 0.8043 | 19 / 22940 |
| Nannoolal_corrected_OLS | 4,575 | 0.8580 | 1.1795 | 0.7687 | 20 / 22940 |
| Nannoolal_corrected_adaptive | 4,575 | 0.7969 | 1.0798 | 0.8061 | 16 / 22940 |
| UNIFNIST_subgroup_OLS | 4,575 | 0.8596 | 1.1851 | 0.7665 | 24 / 22940 |
| UNIFNIST_subgroup_adaptive | 4,575 | 0.7931 | 1.0832 | 0.8049 | 19 / 22940 |

## with_UNIFLBY

| Model | Common valid molecules | MAE log10 | RMSE log10 | R² | Invalid predictions / total |
|---|---:|---:|---:|---:|---:|
| size_selected | 2,006 | 1.3051 | 1.6910 | 0.4691 | 0 / 10050 |
| element_counts_OLS | 2,006 | 0.9879 | 1.3066 | 0.6830 | 1 / 10050 |
| element_counts_adaptive | 2,006 | 0.8825 | 1.1496 | 0.7546 | 0 / 10050 |
| Nannoolal_first_OLS | 2,006 | 0.9026 | 1.2239 | 0.7219 | 11 / 10050 |
| Nannoolal_first_adaptive | 2,006 | 0.7763 | 1.0385 | 0.7997 | 11 / 10050 |
| Nannoolal_corrected_OLS | 2,006 | 0.9067 | 1.2267 | 0.7206 | 12 / 10050 |
| Nannoolal_corrected_adaptive | 2,006 | 0.7775 | 1.0390 | 0.7996 | 10 / 10050 |
| UNIFLBY_subgroup_OLS | 2,006 | 0.9291 | 1.2504 | 0.7097 | 5 / 10050 |
| UNIFLBY_subgroup_adaptive | 2,006 | 0.8046 | 1.0748 | 0.7855 | 4 / 10050 |

## common_all

| Model | Common valid molecules | MAE log10 | RMSE log10 | R² | Invalid predictions / total |
|---|---:|---:|---:|---:|---:|
| size_selected | 1,881 | 1.3389 | 1.7217 | 0.4619 | 0 / 9440 |
| element_counts_OLS | 1,881 | 0.9859 | 1.3143 | 0.6864 | 4 / 9440 |
| element_counts_adaptive | 1,881 | 0.8691 | 1.1399 | 0.7641 | 0 / 9440 |
| Nannoolal_first_OLS | 1,881 | 0.9066 | 1.2276 | 0.7264 | 12 / 9440 |
| Nannoolal_first_adaptive | 1,881 | 0.7618 | 1.0218 | 0.8105 | 10 / 9440 |
| Nannoolal_corrected_OLS | 1,881 | 0.9109 | 1.2331 | 0.7240 | 13 / 9440 |
| Nannoolal_corrected_adaptive | 1,881 | 0.7663 | 1.0284 | 0.8080 | 10 / 9440 |
| UNIFAC_subgroup_OLS | 1,881 | 0.9302 | 1.2574 | 0.7130 | 7 / 9440 |
| UNIFAC_subgroup_adaptive | 1,881 | 0.7969 | 1.0726 | 0.7912 | 6 / 9440 |
| UNIFDMD_subgroup_OLS | 1,881 | 0.9245 | 1.2489 | 0.7168 | 8 / 9440 |
| UNIFDMD_subgroup_adaptive | 1,881 | 0.7727 | 1.0435 | 0.8023 | 8 / 9440 |
| UNIFNIST_subgroup_OLS | 1,881 | 0.8902 | 1.2263 | 0.7270 | 9 / 9440 |
| UNIFNIST_subgroup_adaptive | 1,881 | 0.7594 | 1.0429 | 0.8025 | 10 / 9440 |
| UNIFLBY_subgroup_OLS | 1,881 | 0.9368 | 1.2615 | 0.7111 | 5 / 9440 |
| UNIFLBY_subgroup_adaptive | 1,881 | 0.8068 | 1.0818 | 0.7875 | 4 / 9440 |

## Paired comparisons

| Population | First model | Second model | Molecules | MAE difference | Familywise 95% CI | Invalid molecules removed |
|---|---|---|---:|---:|---|---:|
| Nannoolal_own | Nannoolal_corrected_OLS | Nannoolal_first_OLS | 7,275 | -0.0056 | -0.0111 to -0.0004 | 3 |
| Nannoolal_own | Nannoolal_corrected_adaptive | Nannoolal_first_adaptive | 7,274 | -0.0041 | -0.0087 to +0.0004 | 4 |
| Nannoolal_own | Nannoolal_first_adaptive | element_counts_adaptive | 7,273 | -0.1767 | -0.1996 to -0.1543 | 5 |
| common_all | Nannoolal_corrected_OLS | Nannoolal_first_OLS | 1,885 | +0.0043 | -0.0042 to +0.0129 | 3 |
| common_all | Nannoolal_corrected_OLS | UNIFAC_subgroup_OLS | 1,885 | -0.0196 | -0.0483 to +0.0083 | 3 |
| common_all | Nannoolal_corrected_OLS | UNIFDMD_subgroup_OLS | 1,885 | -0.0150 | -0.0425 to +0.0105 | 3 |
| common_all | Nannoolal_corrected_OLS | UNIFLBY_subgroup_OLS | 1,885 | -0.0263 | -0.0547 to +0.0019 | 3 |
| common_all | Nannoolal_corrected_OLS | UNIFNIST_subgroup_OLS | 1,882 | +0.0207 | -0.0095 to +0.0512 | 6 |
| common_all | Nannoolal_corrected_adaptive | Nannoolal_first_adaptive | 1,886 | +0.0043 | -0.0038 to +0.0130 | 2 |
| common_all | Nannoolal_corrected_adaptive | UNIFAC_subgroup_adaptive | 1,885 | -0.0301 | -0.0561 to -0.0032 | 3 |
| common_all | Nannoolal_corrected_adaptive | UNIFDMD_subgroup_adaptive | 1,886 | -0.0072 | -0.0304 to +0.0156 | 2 |
| common_all | Nannoolal_corrected_adaptive | UNIFLBY_subgroup_adaptive | 1,886 | -0.0400 | -0.0666 to -0.0120 | 2 |
| common_all | Nannoolal_corrected_adaptive | UNIFNIST_subgroup_adaptive | 1,884 | +0.0070 | -0.0209 to +0.0330 | 4 |
| common_all | Nannoolal_first_OLS | UNIFAC_subgroup_OLS | 1,885 | -0.0238 | -0.0513 to +0.0032 | 3 |
| common_all | Nannoolal_first_OLS | UNIFDMD_subgroup_OLS | 1,885 | -0.0193 | -0.0448 to +0.0056 | 3 |
| common_all | Nannoolal_first_OLS | UNIFLBY_subgroup_OLS | 1,885 | -0.0306 | -0.0580 to -0.0040 | 3 |
| common_all | Nannoolal_first_OLS | UNIFNIST_subgroup_OLS | 1,882 | +0.0164 | -0.0124 to +0.0455 | 6 |
| common_all | Nannoolal_first_adaptive | UNIFAC_subgroup_adaptive | 1,885 | -0.0344 | -0.0610 to -0.0086 | 3 |
| common_all | Nannoolal_first_adaptive | UNIFDMD_subgroup_adaptive | 1,886 | -0.0115 | -0.0349 to +0.0107 | 2 |
| common_all | Nannoolal_first_adaptive | UNIFLBY_subgroup_adaptive | 1,886 | -0.0443 | -0.0718 to -0.0161 | 2 |
| common_all | Nannoolal_first_adaptive | UNIFNIST_subgroup_adaptive | 1,884 | +0.0026 | -0.0257 to +0.0292 | 4 |
| common_all | Nannoolal_first_adaptive | element_counts_adaptive | 1,886 | -0.1060 | -0.1402 to -0.0692 | 2 |
| with_UNIFAC | Nannoolal_corrected_OLS | Nannoolal_first_OLS | 4,113 | -0.0091 | -0.0178 to -0.0003 | 5 |
| with_UNIFAC | Nannoolal_corrected_OLS | UNIFAC_subgroup_OLS | 4,113 | -0.0266 | -0.0442 to -0.0089 | 5 |
| with_UNIFAC | Nannoolal_corrected_adaptive | Nannoolal_first_adaptive | 4,115 | -0.0062 | -0.0142 to +0.0012 | 3 |
| with_UNIFAC | Nannoolal_corrected_adaptive | UNIFAC_subgroup_adaptive | 4,114 | -0.0233 | -0.0412 to -0.0054 | 4 |
| with_UNIFAC | Nannoolal_first_OLS | UNIFAC_subgroup_OLS | 4,113 | -0.0175 | -0.0350 to +0.0000 | 5 |
| with_UNIFAC | Nannoolal_first_adaptive | UNIFAC_subgroup_adaptive | 4,114 | -0.0171 | -0.0343 to -0.0004 | 4 |
| with_UNIFAC | Nannoolal_first_adaptive | element_counts_adaptive | 4,115 | -0.1778 | -0.2099 to -0.1461 | 3 |
| with_UNIFDMD | Nannoolal_corrected_OLS | Nannoolal_first_OLS | 4,196 | -0.0084 | -0.0163 to -0.0005 | 2 |
| with_UNIFDMD | Nannoolal_corrected_OLS | UNIFDMD_subgroup_OLS | 4,195 | -0.0252 | -0.0432 to -0.0079 | 3 |
| with_UNIFDMD | Nannoolal_corrected_adaptive | Nannoolal_first_adaptive | 4,196 | -0.0038 | -0.0099 to +0.0023 | 2 |
| with_UNIFDMD | Nannoolal_corrected_adaptive | UNIFDMD_subgroup_adaptive | 4,195 | -0.0255 | -0.0417 to -0.0096 | 3 |
| with_UNIFDMD | Nannoolal_first_OLS | UNIFDMD_subgroup_OLS | 4,195 | -0.0168 | -0.0349 to +0.0012 | 3 |
| with_UNIFDMD | Nannoolal_first_adaptive | UNIFDMD_subgroup_adaptive | 4,195 | -0.0218 | -0.0370 to -0.0061 | 3 |
| with_UNIFDMD | Nannoolal_first_adaptive | element_counts_adaptive | 4,196 | -0.1864 | -0.2172 to -0.1564 | 2 |
| with_UNIFLBY | Nannoolal_corrected_OLS | Nannoolal_first_OLS | 2,007 | +0.0042 | -0.0045 to +0.0135 | 3 |
| with_UNIFLBY | Nannoolal_corrected_OLS | UNIFLBY_subgroup_OLS | 2,007 | -0.0225 | -0.0496 to +0.0032 | 3 |
| with_UNIFLBY | Nannoolal_corrected_adaptive | Nannoolal_first_adaptive | 2,007 | +0.0011 | -0.0094 to +0.0113 | 3 |
| with_UNIFLBY | Nannoolal_corrected_adaptive | UNIFLBY_subgroup_adaptive | 2,007 | -0.0267 | -0.0527 to -0.0017 | 3 |
| with_UNIFLBY | Nannoolal_first_OLS | UNIFLBY_subgroup_OLS | 2,007 | -0.0267 | -0.0529 to -0.0007 | 3 |
| with_UNIFLBY | Nannoolal_first_adaptive | UNIFLBY_subgroup_adaptive | 2,007 | -0.0278 | -0.0525 to -0.0036 | 3 |
| with_UNIFLBY | Nannoolal_first_adaptive | element_counts_adaptive | 2,007 | -0.1053 | -0.1405 to -0.0711 | 3 |
| with_UNIFNIST | Nannoolal_corrected_OLS | Nannoolal_first_OLS | 4,582 | -0.0092 | -0.0172 to -0.0008 | 6 |
| with_UNIFNIST | Nannoolal_corrected_OLS | UNIFNIST_subgroup_OLS | 4,577 | -0.0015 | -0.0215 to +0.0182 | 11 |
| with_UNIFNIST | Nannoolal_corrected_adaptive | Nannoolal_first_adaptive | 4,582 | -0.0078 | -0.0150 to -0.0009 | 6 |
| with_UNIFNIST | Nannoolal_corrected_adaptive | UNIFNIST_subgroup_adaptive | 4,578 | +0.0038 | -0.0148 to +0.0231 | 10 |
| with_UNIFNIST | Nannoolal_first_OLS | UNIFNIST_subgroup_OLS | 4,577 | +0.0075 | -0.0127 to +0.0271 | 11 |
| with_UNIFNIST | Nannoolal_first_adaptive | UNIFNIST_subgroup_adaptive | 4,578 | +0.0114 | -0.0074 to +0.0303 | 10 |
| with_UNIFNIST | Nannoolal_first_adaptive | element_counts_adaptive | 4,582 | -0.1774 | -0.2087 to -0.1480 | 6 |

## Files

- fragmentation.csv: structural assignment coverage, native property flags, local-extension notes and failure reasons.
- features.npz and feature_metadata.json: first-order and corrected Nannoolal count matrices.
- *_folds.csv and *_coefficients.csv: nested choices and descriptive full-data refits.
- model .npz archives: all raw held-out predictions, including invalid values; prior study outputs are unchanged.
- summary.json, paired_comparisons.csv and manifest.json: full scores, uncertainty and portable provenance.
