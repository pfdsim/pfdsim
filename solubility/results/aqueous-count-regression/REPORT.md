# Absolute room-temperature aqueous solubility from atom and UNIFAC counts

Predict log10(S / (mol/L)), with one target per chemical identity. This is absolute solubility; solid-state and ionization effects are included in the experimental target.

## Data and features

- AqSolDB supplies the curated label wherever available. Add BigSolDB-only water compounds measured at 293.15–303.15 K. For BigSolDB, median duplicates within source/temperature, use the temperature(s) nearest 298.15 K per publication, then median across publications.
- Deduplicate by full standard InChI, preserving stereochemistry and salt/protonation distinctions. Recover invalid AqSolDB SMILES from its supplied InChI where possible.
- Include ions and multicomponent salts in the atom-count population; preserve the entire supplied chemical identity. UNIFAC’s native fragmenter determines its own domain.
- Count every element, including explicit and implicit hydrogens. Scalar size baselines use total atom count or heavy-atom count.
- Test size bases N, log(1+N), N^(2/3), [N,N²], and [N,log(1+N),N^(2/3)], using heavy or total atom count. These are candidate size relationships, not a universal physical law.
- Count UNIFAC subgroups and main groups with pfdsim’s authoritative fragmenter and group tables. No solvent interaction parameters, gamma calculations, melting points, logP values, or other precomputed descriptors are model features.
- Compare ordinary least squares with adaptive ridge regressions augmented by optional log, surface-like, or quadratic heavy-atom size terms.
- Equivalent variant feature encodings are identified after extraction and evaluated once. The aliases and coverage audit are saved.

## Validation

- 5-fold molecule CV, repeated 5 times, seed 20261022. Each chemical identity has a single target and never appears in both training and test within a split.
- Select size form and ridge strength using three-fold inner CV on OUTER TRAINING data only, minimizing validation squared log error. Feature centering/scaling and removal of constant columns are fitted within each training split.
- Plain OLS results are reported separately, without choosing them after seeing test performance. SVD handles dependent feature columns.
- Feature columns absent/constant in training receive zero fitted slope; test extrapolations involving such features are retained and counted, not silently discarded.
- Report pooled out-of-fold MAE, RMSE and R², averaging losses across repetitions. No test predictions are ensembled before scoring.
- Group-coverage pools refit atom-count and size baselines on the SAME molecules, using identical outer folds. The common pool compares all distinct UNIFAC encodings on exactly the same chemical population.
- Full-data coefficient refits are descriptive only. Their values do not establish individual atomic/group causal effects or statistical significance, especially with dependent features.
- Labels may have unknown pH/solid form; random molecule splits do not establish unseen-scaffold or independently collected external-data accuracy.

## Dataset audit

```json
{
  "AqSolDB_input_rows": 9982,
  "AqSolDB_InChI_recoveries": 2,
  "BigSolDB_room_temperature_water_rows": 2045,
  "compounds": 10421,
  "AqSolDB_unique": 9982,
  "BigSolDB_unique": 694,
  "overlapping_compounds": 255,
  "BigSolDB_only": 439,
  "overlap_median_absolute_logS_difference": 0.1307895316864654,
  "overlap_absolute_logS_difference_95pct": 2.7408019566673953
}
```

## UNIFAC encoding coverage and aliases

```json
{
  "UNIFAC": {
    "aliases": [
      "UNIFAC",
      "UNIFAC2"
    ],
    "molecules": 4190,
    "subgroups": 104,
    "main_groups": 51
  },
  "UNIFDMD": {
    "aliases": [
      "UNIFDMD",
      "UNIFM2"
    ],
    "molecules": 4281,
    "subgroups": 111,
    "main_groups": 55
  },
  "UNIFNIST": {
    "aliases": [
      "UNIFNIST"
    ],
    "molecules": 4690,
    "subgroups": 186,
    "main_groups": 87
  },
  "UNIFLBY": {
    "aliases": [
      "UNIFLBY"
    ],
    "molecules": 2067,
    "subgroups": 45,
    "main_groups": 21
  }
}
```

## all_water

| Model | Molecules | MAE log10 | RMSE log10 | R² | Within factor 2 | Within factor 10 |
|---|---:|---:|---:|---:|---:|---:|
| training_mean | 10,421 | 1.8752 | 2.3524 | -0.0001 | 10.7% | 33.1% |
| heavy_atoms_OLS | 10,421 | 1.7215 | 2.2090 | 0.1182 | 12.0% | 37.5% |
| total_atoms_OLS | 10,421 | 1.7320 | 2.2226 | 0.1073 | 12.0% | 37.4% |
| size_selected | 10,421 | 1.6347 | 2.1717 | 0.1476 | 12.8% | 40.2% |
| element_counts_OLS | 10,421 | 1.3025 | 1.7844 | 0.4246 | 15.3% | 48.5% |
| element_counts_adaptive | 10,421 | 1.2336 | 1.8790 | 0.3619 | 16.8% | 51.6% |

## UNIFAC

| Model | Molecules | MAE log10 | RMSE log10 | R² | Within factor 2 | Within factor 10 |
|---|---:|---:|---:|---:|---:|---:|
| training_mean | 4,190 | 1.9605 | 2.4558 | -0.0003 | 10.2% | 32.4% |
| heavy_atoms_OLS | 4,190 | 1.6014 | 2.1194 | 0.2550 | 13.9% | 41.3% |
| total_atoms_OLS | 4,190 | 1.6890 | 2.2303 | 0.1750 | 12.6% | 40.3% |
| size_selected | 4,190 | 1.4886 | 1.9373 | 0.3776 | 14.6% | 43.9% |
| element_counts_OLS | 4,190 | 1.1515 | 1.6485 | 0.5493 | 16.8% | 54.0% |
| element_counts_adaptive | 4,190 | 0.9908 | 1.3515 | 0.6971 | 21.1% | 60.4% |
| UNIFAC_main_OLS | 4,190 | 1.0706 | 1.5828 | 0.5845 | 19.6% | 57.6% |
| UNIFAC_subgroup_OLS | 4,190 | 1.0014 | 1.4790 | 0.6372 | 21.8% | 61.6% |
| UNIFAC_subgroup_adaptive | 4,190 | 0.8439 | 1.2441 | 0.7433 | 26.7% | 69.2% |

## UNIFDMD

| Model | Molecules | MAE log10 | RMSE log10 | R² | Within factor 2 | Within factor 10 |
|---|---:|---:|---:|---:|---:|---:|
| training_mean | 4,281 | 1.9531 | 2.4488 | -0.0003 | 10.2% | 32.7% |
| heavy_atoms_OLS | 4,281 | 1.5712 | 2.0821 | 0.2769 | 14.0% | 41.6% |
| total_atoms_OLS | 4,281 | 1.6697 | 2.2093 | 0.1858 | 12.7% | 40.8% |
| size_selected | 4,281 | 1.4546 | 1.8934 | 0.4020 | 15.2% | 44.8% |
| element_counts_OLS | 4,281 | 1.1584 | 1.6479 | 0.5470 | 16.7% | 53.6% |
| element_counts_adaptive | 4,281 | 0.9732 | 1.2901 | 0.7224 | 21.3% | 61.3% |
| UNIFDMD_main_OLS | 4,281 | 1.0524 | 1.5591 | 0.5945 | 20.2% | 58.9% |
| UNIFDMD_subgroup_OLS | 4,281 | 1.0071 | 1.4796 | 0.6348 | 21.6% | 61.5% |
| UNIFDMD_subgroup_adaptive | 4,281 | 0.8162 | 1.1544 | 0.7777 | 27.4% | 70.1% |

## UNIFNIST

| Model | Molecules | MAE log10 | RMSE log10 | R² | Within factor 2 | Within factor 10 |
|---|---:|---:|---:|---:|---:|---:|
| training_mean | 4,690 | 1.9647 | 2.4633 | -0.0003 | 10.8% | 32.3% |
| heavy_atoms_OLS | 4,690 | 1.6396 | 2.1453 | 0.2413 | 13.8% | 40.0% |
| total_atoms_OLS | 4,690 | 1.7294 | 2.2568 | 0.1603 | 12.2% | 38.7% |
| size_selected | 4,690 | 1.5190 | 1.9662 | 0.3627 | 14.3% | 43.5% |
| element_counts_OLS | 4,690 | 1.1836 | 1.6522 | 0.5500 | 16.4% | 52.3% |
| element_counts_adaptive | 4,690 | 1.0272 | 1.3668 | 0.6920 | 20.3% | 58.9% |
| UNIFNIST_main_OLS | 4,690 | 1.0547 | 1.5601 | 0.5987 | 20.5% | 58.3% |
| UNIFNIST_subgroup_OLS | 4,690 | 1.0139 | 1.5019 | 0.6282 | 21.3% | 61.2% |
| UNIFNIST_subgroup_adaptive | 4,690 | 0.8427 | 1.2246 | 0.7528 | 27.1% | 69.3% |

## UNIFLBY

| Model | Molecules | MAE log10 | RMSE log10 | R² | Within factor 2 | Within factor 10 |
|---|---:|---:|---:|---:|---:|---:|
| training_mean | 2,067 | 1.9064 | 2.3512 | -0.0007 | 10.0% | 31.2% |
| heavy_atoms_OLS | 2,067 | 1.5246 | 1.9752 | 0.2938 | 12.8% | 40.2% |
| total_atoms_OLS | 2,067 | 1.5296 | 2.0000 | 0.2759 | 13.1% | 39.9% |
| size_selected | 2,067 | 1.3409 | 1.7494 | 0.4460 | 15.3% | 48.2% |
| element_counts_OLS | 2,067 | 1.2271 | 1.7609 | 0.4387 | 15.5% | 50.6% |
| element_counts_adaptive | 2,067 | 0.9559 | 1.2733 | 0.7065 | 22.0% | 62.2% |
| UNIFLBY_main_OLS | 2,067 | 1.2285 | 1.7640 | 0.4367 | 15.3% | 50.9% |
| UNIFLBY_subgroup_OLS | 2,067 | 1.1456 | 1.6522 | 0.5059 | 18.4% | 54.8% |
| UNIFLBY_subgroup_adaptive | 2,067 | 0.8854 | 1.2615 | 0.7119 | 25.7% | 67.4% |

## common_UNIFAC

| Model | Molecules | MAE log10 | RMSE log10 | R² | Within factor 2 | Within factor 10 |
|---|---:|---:|---:|---:|---:|---:|
| training_mean | 1,945 | 1.9248 | 2.3805 | -0.0007 | 10.4% | 31.1% |
| heavy_atoms_OLS | 1,945 | 1.5652 | 2.0170 | 0.2816 | 12.2% | 38.9% |
| total_atoms_OLS | 1,945 | 1.5688 | 2.0401 | 0.2651 | 12.6% | 38.6% |
| size_selected | 1,945 | 1.3759 | 1.7902 | 0.4340 | 15.0% | 46.8% |
| element_counts_OLS | 1,945 | 1.2517 | 1.7966 | 0.4300 | 15.2% | 49.7% |
| element_counts_adaptive | 1,945 | 0.9610 | 1.3972 | 0.6553 | 21.9% | 63.0% |
| UNIFAC_main_OLS | 1,945 | 1.2517 | 1.7856 | 0.4370 | 14.7% | 49.3% |
| UNIFAC_subgroup_OLS | 1,945 | 1.1526 | 1.6564 | 0.5155 | 17.6% | 55.0% |
| UNIFAC_subgroup_adaptive | 1,945 | 0.8774 | 1.2024 | 0.7447 | 25.3% | 68.1% |
| UNIFDMD_main_OLS | 1,945 | 1.2381 | 1.7810 | 0.4399 | 15.2% | 50.4% |
| UNIFDMD_subgroup_OLS | 1,945 | 1.1507 | 1.6571 | 0.5151 | 17.3% | 54.8% |
| UNIFDMD_subgroup_adaptive | 1,945 | 0.8501 | 1.1603 | 0.7623 | 26.4% | 68.6% |
| UNIFNIST_main_OLS | 1,945 | 1.2329 | 1.7843 | 0.4378 | 15.9% | 50.4% |
| UNIFNIST_subgroup_OLS | 1,945 | 1.1511 | 1.6694 | 0.5079 | 18.1% | 55.2% |
| UNIFNIST_subgroup_adaptive | 1,945 | 0.8435 | 1.1788 | 0.7546 | 27.6% | 69.4% |
| UNIFLBY_main_OLS | 1,945 | 1.2493 | 1.7939 | 0.4317 | 15.2% | 50.0% |
| UNIFLBY_subgroup_OLS | 1,945 | 1.1651 | 1.6818 | 0.5006 | 18.2% | 53.8% |
| UNIFLBY_subgroup_adaptive | 1,945 | 0.8900 | 1.2709 | 0.7148 | 25.7% | 67.2% |

## Size choices and paired comparisons

Training-fold selection frequencies are in summary.json. Paired comparisons resample entire molecules and split repetitions; their intervals use a family size covering both OLS/adaptive group-versus-element comparisons on own and common populations.

```json
[
  {
    "pool": "UNIFAC",
    "group_model": "UNIFAC_subgroup_OLS",
    "atomic_model": "element_counts_OLS",
    "molecules": 4190,
    "group_minus_atomic_MAE": -0.15005853630006785,
    "CI95": [
      -0.1710671571345985,
      -0.12875298505025523
    ],
    "familywise_CI95": [
      -0.18232733423942216,
      -0.11768736496328545
    ]
  },
  {
    "pool": "UNIFAC",
    "group_model": "UNIFAC_subgroup_adaptive",
    "atomic_model": "element_counts_adaptive",
    "molecules": 4190,
    "group_minus_atomic_MAE": -0.14691983960706162,
    "CI95": [
      -0.16733588651488135,
      -0.1256335524161482
    ],
    "familywise_CI95": [
      -0.17799006240883497,
      -0.11407008548999341
    ]
  },
  {
    "pool": "common_UNIFAC",
    "group_model": "UNIFAC_subgroup_OLS",
    "atomic_model": "element_counts_OLS",
    "molecules": 1945,
    "group_minus_atomic_MAE": -0.09903278743742461,
    "CI95": [
      -0.1296162001254543,
      -0.06906971503945378
    ],
    "familywise_CI95": [
      -0.14701887109292044,
      -0.05673217936028129
    ]
  },
  {
    "pool": "common_UNIFAC",
    "group_model": "UNIFAC_subgroup_adaptive",
    "atomic_model": "element_counts_adaptive",
    "molecules": 1945,
    "group_minus_atomic_MAE": -0.0836060834687458,
    "CI95": [
      -0.11605964656429463,
      -0.05659444685417289
    ],
    "familywise_CI95": [
      -0.1372448437853809,
      -0.04258783525715473
    ]
  },
  {
    "pool": "UNIFDMD",
    "group_model": "UNIFDMD_subgroup_OLS",
    "atomic_model": "element_counts_OLS",
    "molecules": 4281,
    "group_minus_atomic_MAE": -0.15129535124740245,
    "CI95": [
      -0.17269783162077593,
      -0.12999327496125712
    ],
    "familywise_CI95": [
      -0.18243241998927442,
      -0.12139275094039212
    ]
  },
  {
    "pool": "UNIFDMD",
    "group_model": "UNIFDMD_subgroup_adaptive",
    "atomic_model": "element_counts_adaptive",
    "molecules": 4281,
    "group_minus_atomic_MAE": -0.15702651880673923,
    "CI95": [
      -0.17697125935061395,
      -0.13619282542040473
    ],
    "familywise_CI95": [
      -0.18583921973619846,
      -0.12428332710503047
    ]
  },
  {
    "pool": "common_UNIFAC",
    "group_model": "UNIFDMD_subgroup_OLS",
    "atomic_model": "element_counts_OLS",
    "molecules": 1945,
    "group_minus_atomic_MAE": -0.10099212187992748,
    "CI95": [
      -0.1331085473665247,
      -0.07005818575989493
    ],
    "familywise_CI95": [
      -0.14980497732163536,
      -0.05475637701936344
    ]
  },
  {
    "pool": "common_UNIFAC",
    "group_model": "UNIFDMD_subgroup_adaptive",
    "atomic_model": "element_counts_adaptive",
    "molecules": 1945,
    "group_minus_atomic_MAE": -0.110873365024982,
    "CI95": [
      -0.14259439074580874,
      -0.08289336496592777
    ],
    "familywise_CI95": [
      -0.16252341956273098,
      -0.06913226625039687
    ]
  },
  {
    "pool": "UNIFNIST",
    "group_model": "UNIFNIST_subgroup_OLS",
    "atomic_model": "element_counts_OLS",
    "molecules": 4690,
    "group_minus_atomic_MAE": -0.16970598692076513,
    "CI95": [
      -0.19131322359511987,
      -0.1474251744012248
    ],
    "familywise_CI95": [
      -0.2013862596475548,
      -0.13587205955997234
    ]
  },
  {
    "pool": "UNIFNIST",
    "group_model": "UNIFNIST_subgroup_adaptive",
    "atomic_model": "element_counts_adaptive",
    "molecules": 4690,
    "group_minus_atomic_MAE": -0.1844415979413244,
    "CI95": [
      -0.20630896285070927,
      -0.16152412414000852
    ],
    "familywise_CI95": [
      -0.21779812191366757,
      -0.1478597786336045
    ]
  },
  {
    "pool": "common_UNIFAC",
    "group_model": "UNIFNIST_subgroup_OLS",
    "atomic_model": "element_counts_OLS",
    "molecules": 1945,
    "group_minus_atomic_MAE": -0.10054230742102979,
    "CI95": [
      -0.13503877611899184,
      -0.06692109526971017
    ],
    "familywise_CI95": [
      -0.1525712029498175,
      -0.05091934078551507
    ]
  },
  {
    "pool": "common_UNIFAC",
    "group_model": "UNIFNIST_subgroup_adaptive",
    "atomic_model": "element_counts_adaptive",
    "molecules": 1945,
    "group_minus_atomic_MAE": -0.11750612919953107,
    "CI95": [
      -0.15364693967861254,
      -0.0861788285904792
    ],
    "familywise_CI95": [
      -0.1749721075623686,
      -0.06816919753376388
    ]
  },
  {
    "pool": "UNIFLBY",
    "group_model": "UNIFLBY_subgroup_OLS",
    "atomic_model": "element_counts_OLS",
    "molecules": 2067,
    "group_minus_atomic_MAE": -0.08152967310043603,
    "CI95": [
      -0.10815921662827595,
      -0.055756914553624325
    ],
    "familywise_CI95": [
      -0.12163798709832119,
      -0.043038677002406045
    ]
  },
  {
    "pool": "UNIFLBY",
    "group_model": "UNIFLBY_subgroup_adaptive",
    "atomic_model": "element_counts_adaptive",
    "molecules": 2067,
    "group_minus_atomic_MAE": -0.07043384798822731,
    "CI95": [
      -0.09411870231925183,
      -0.04457758179400328
    ],
    "familywise_CI95": [
      -0.10632011630523981,
      -0.027264489545947592
    ]
  },
  {
    "pool": "common_UNIFAC",
    "group_model": "UNIFLBY_subgroup_OLS",
    "atomic_model": "element_counts_OLS",
    "molecules": 1945,
    "group_minus_atomic_MAE": -0.08656252990706732,
    "CI95": [
      -0.11476725125569233,
      -0.05890781913144565
    ],
    "familywise_CI95": [
      -0.12927437988389037,
      -0.046461688109718575
    ]
  },
  {
    "pool": "common_UNIFAC",
    "group_model": "UNIFLBY_subgroup_adaptive",
    "atomic_model": "element_counts_adaptive",
    "molecules": 1945,
    "group_minus_atomic_MAE": -0.07105309437716843,
    "CI95": [
      -0.09748344633911647,
      -0.046977855795370366
    ],
    "familywise_CI95": [
      -0.11688002407196103,
      -0.035592931842543564
    ]
  }
]
```

## Files

- molecules.csv: deduplicated structures, labels and source-overlap targets.
- fragmentation.csv: every variant’s success/failure per compound.
- features.npz: atom and group count matrices; feature_metadata.json labels their columns.
- *_folds.csv: chosen size transforms/ridge penalties and training errors for every outer fit.
- *_coefficients.csv: descriptive full-data refit coefficients, with zero slopes for nonidentifiable columns.
- *.npz per model: every molecule’s held-out prediction across repeats.
- summary.json, manifest.json: results and portable fingerprints.
- Inputs and pfdsim production implementations are unchanged.
