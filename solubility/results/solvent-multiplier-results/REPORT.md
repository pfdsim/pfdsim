# Solvent mean transfer multipliers versus UNIFAC

One positive factor M_s per solvent: x_pred = x_reference M_target/M_reference. The physical main baseline caps x_pred at 1. The literal uncapped multiplier is also scored as a sensitivity check.

## Fitting and held-out evaluation

- Fit solvent effects beta_s = log10(M_s) by least squares to training log-solubility differences. This estimates geometric mean multiplicative effects, rather than allowing a few enormous raw ratios to dominate.
- Training observations use the prior benchmark’s same-publication, exact-temperature room-temperature groups and deduplication. Each molecule has equal total weight; each group within a molecule has equal weight; each distinct solvent pair within a group has equal weight.
- The objective is the mean squared error of observed pairwise log ratios minus beta_target-beta_reference. This cancels group-specific solid/temperature/source intercepts without fitting or using a target molecule’s solid effect.
- 5-fold cross-validation, repeated 10 times; seed 20261014. All temperatures, sources and solvents of a chemical identity remain in the same fold. Training includes eligible molecules that UNIFAC cannot represent; it is not selected using UNIFAC success.
- Fit one shared set of solvent factors per training fold for all variants. No solute features, solvent-pair-specific factors, regularization or tuned hyperparameters.
- Unseen solvents and disconnected solvent components are unavailable, rather than guessed. Main comparisons require baseline availability in every repeat and use identical UNIFAC/baseline cases.
- Score the mean held-out loss across repeats, not the loss of an ensemble-averaged prediction. Every molecule receives equal evaluation weight, matching the previous benchmark.
- Bootstrap 10,000 times, resampling whole molecules and whole split repetitions. Differences are UNIFAC MAE minus baseline MAE; negative favors UNIFAC. “Convincingly” requires the upper endpoint of the simultaneous familywise 95% interval for six variants to be below zero.
- Statistical intervals condition on these fitted cross-validation predictions; they do not establish performance outside this dataset or remove the previous solid-form/pH and model-training-overlap limitations.

## Main: each variant on its available compounds

Each row compares exactly the same cases for UNIFAC and the multiplier. Across rows, compound coverage differs.

| Variant | Transfers | Molecules | UNIFAC MAE | Multiplier MAE | Difference | Simultaneous 95% CI |
|---|---:|---:|---:|---:|---:|---|
| UNIFAC | 40,910 | 256 | 0.5438 | 0.5329 | +0.0109 | -0.0635 to +0.0943 |
| UNIFAC2 | 75,208 | 346 | 0.5984 | 0.6272 | -0.0288 | -0.0976 to +0.0422 |
| UNIFDMD | 33,249 | 235 | 0.5317 | 0.5253 | +0.0064 | -0.0736 to +0.1007 |
| UNIFM2 | 74,824 | 351 | 0.6252 | 0.6255 | -0.0003 | -0.0712 to +0.0787 |
| UNIFNIST | 24,947 | 181 | 0.5756 | 0.5531 | +0.0225 | -0.0704 to +0.1330 |
| UNIFLBY | 13,930 | 118 | 0.4314 | 0.5068 | -0.0755 | -0.1641 to +0.0206 |

## Same cases for all six variants

| Variant | Transfers | Molecules | UNIFAC MAE | Multiplier MAE | Difference | Simultaneous 95% CI |
|---|---:|---:|---:|---:|---:|---|
| UNIFAC | 7,425 | 86 | 0.3712 | 0.5279 | -0.1567 | -0.2555 to -0.0646 |
| UNIFAC2 | 7,425 | 86 | 0.3736 | 0.5279 | -0.1543 | -0.2547 to -0.0539 |
| UNIFDMD | 7,425 | 86 | 0.3967 | 0.5279 | -0.1312 | -0.2418 to -0.0192 |
| UNIFM2 | 7,425 | 86 | 0.3786 | 0.5279 | -0.1493 | -0.2530 to -0.0429 |
| UNIFNIST | 7,425 | 86 | 0.4692 | 0.5279 | -0.0588 | -0.1692 to +0.0544 |
| UNIFLBY | 7,425 | 86 | 0.3657 | 0.5279 | -0.1622 | -0.2634 to -0.0500 |

## Larger common five-variant cohort

| Variant | Transfers | Molecules | UNIFAC MAE | Multiplier MAE | Difference | Simultaneous 95% CI |
|---|---:|---:|---:|---:|---:|---|
| UNIFAC | 17,135 | 145 | 0.3636 | 0.5411 | -0.1775 | -0.2643 to -0.0955 |
| UNIFAC2 | 17,135 | 145 | 0.3664 | 0.5411 | -0.1747 | -0.2741 to -0.0868 |
| UNIFDMD | 17,135 | 145 | 0.4335 | 0.5411 | -0.1076 | -0.2187 to +0.0220 |
| UNIFM2 | 17,135 | 145 | 0.3797 | 0.5411 | -0.1614 | -0.2601 to -0.0702 |
| UNIFNIST | 17,135 | 145 | 0.5080 | 0.5411 | -0.0331 | -0.1407 to +0.1026 |

## Directional results on the common five-variant cohort

| Variant | Direction | Transfers | Molecules | UNIFAC MAE | Multiplier MAE | Difference | Simultaneous 95% CI |
|---|---|---:|---:|---:|---:|---:|---|
| UNIFAC | water_to_nonwater | 221 | 17 | 0.6499 | 1.3935 | -0.7435 | -1.2755 to -0.2339 |
| UNIFAC | nonwater_to_water | 423 | 32 | 1.1276 | 1.3899 | -0.2622 | -0.9581 to +0.7428 |
| UNIFAC | nonwater_to_nonwater | 16,491 | 143 | 0.3387 | 0.4741 | -0.1354 | -0.2071 to -0.0655 |
| UNIFAC2 | water_to_nonwater | 221 | 17 | 0.6593 | 1.3935 | -0.7341 | -1.3717 to -0.1355 |
| UNIFAC2 | nonwater_to_water | 423 | 32 | 1.1627 | 1.3899 | -0.2271 | -1.0689 to +0.8788 |
| UNIFAC2 | nonwater_to_nonwater | 16,491 | 143 | 0.3438 | 0.4741 | -0.1303 | -0.1996 to -0.0626 |
| UNIFDMD | water_to_nonwater | 221 | 17 | 0.7714 | 1.3935 | -0.6220 | -1.2009 to -0.0839 |
| UNIFDMD | nonwater_to_water | 423 | 32 | 1.2478 | 1.3899 | -0.1420 | -0.8211 to +0.6651 |
| UNIFDMD | nonwater_to_nonwater | 16,491 | 143 | 0.4036 | 0.4741 | -0.0705 | -0.1663 to +0.0449 |
| UNIFM2 | water_to_nonwater | 221 | 17 | 0.6171 | 1.3935 | -0.7764 | -1.3653 to -0.2085 |
| UNIFM2 | nonwater_to_water | 423 | 32 | 1.2238 | 1.3899 | -0.1661 | -0.9659 to +0.8808 |
| UNIFM2 | nonwater_to_nonwater | 16,491 | 143 | 0.3622 | 0.4741 | -0.1119 | -0.1899 to -0.0333 |
| UNIFNIST | water_to_nonwater | 221 | 17 | 1.0099 | 1.3935 | -0.3836 | -0.9554 to +0.1755 |
| UNIFNIST | nonwater_to_water | 423 | 32 | 1.5708 | 1.3899 | +0.1809 | -0.5316 to +1.0448 |
| UNIFNIST | nonwater_to_nonwater | 16,491 | 143 | 0.4567 | 0.4741 | -0.0174 | -0.1216 to +0.1123 |

## Sensitivity: pooled_publications

The same molecule-held-out factors trained on the main same-publication data are used here. AqSolDB remains approximate because temperature, pH and solid form are unverified.

| Variant | Transfers | Molecules | UNIFAC MAE | Multiplier MAE | Difference | Simultaneous 95% CI |
|---|---:|---:|---:|---:|---:|---|
| UNIFAC | 8,444 | 86 | 0.3851 | 0.5492 | -0.1641 | -0.2586 to -0.0760 |
| UNIFAC2 | 8,444 | 86 | 0.3825 | 0.5492 | -0.1667 | -0.2609 to -0.0712 |
| UNIFDMD | 8,444 | 86 | 0.4105 | 0.5492 | -0.1388 | -0.2463 to -0.0291 |
| UNIFM2 | 8,444 | 86 | 0.3866 | 0.5492 | -0.1627 | -0.2620 to -0.0582 |
| UNIFNIST | 8,444 | 86 | 0.4787 | 0.5492 | -0.0706 | -0.1749 to +0.0374 |
| UNIFLBY | 8,444 | 86 | 0.3808 | 0.5492 | -0.1684 | -0.2691 to -0.0551 |

## Sensitivity: AqSolDB_approximate_water_anchor

The same molecule-held-out factors trained on the main same-publication data are used here. AqSolDB remains approximate because temperature, pH and solid form are unverified.

| Variant | Transfers | Molecules | UNIFAC MAE | Multiplier MAE | Difference | Simultaneous 95% CI |
|---|---:|---:|---:|---:|---:|---|
| UNIFAC | 107 | 16 | 0.8076 | 1.8281 | -1.0205 | -1.7892 to -0.3451 |
| UNIFAC2 | 107 | 16 | 0.7184 | 1.8281 | -1.1097 | -1.9018 to -0.3839 |
| UNIFDMD | 107 | 16 | 1.2764 | 1.8281 | -0.5517 | -1.3457 to +0.1913 |
| UNIFM2 | 107 | 16 | 0.9994 | 1.8281 | -0.8288 | -1.5899 to -0.1399 |
| UNIFNIST | 107 | 16 | 1.0803 | 1.8281 | -0.7479 | -1.4442 to -0.1393 |
| UNIFLBY | 107 | 16 | 0.5697 | 1.8281 | -1.2584 | -2.2477 to -0.4456 |

## Audit

```json
{
  "data_audit": {
    "BigSolDB_room_temperature_rows": 31448,
    "charged_or_multicomponent_solute_rows_excluded": 2803,
    "unresolved_or_charged_solvent_rows_excluded": 100,
    "water_pure_solvent_molarity_at_298_15_K": 55.3446851783597,
    "AqSolDB_overlap_nondilute_water_anchor_solutes_excluded": 70,
    "eligible_multisolvent_solutes": 1088,
    "same_publication_groups": 3768,
    "same_publication_directed_predictions": 217778,
    "pooled_publications_groups": 3263,
    "pooled_publications_directed_predictions": 238764,
    "AqSolDB_approximate_water_anchor_groups": 267,
    "AqSolDB_approximate_water_anchor_directed_predictions": 2235
  },
  "solvents": 195,
  "molecules": 1088,
  "training_groups": 3768,
  "training_molecules": 1080,
  "cases": 458777,
  "baseline_cases_supported_all_repeats": 448424,
  "maximum_normal_equation_residual": 1.716671249596402e-11,
  "fold_details": [
    {
      "repeat": 0,
      "fold": 0,
      "held_out_molecules": 218,
      "supported_cases": 98937,
      "unsupported_cases": 1064,
      "predictions_above_one_before_capping": 1335
    },
    {
      "repeat": 0,
      "fold": 1,
      "held_out_molecules": 218,
      "supported_cases": 92908,
      "unsupported_cases": 1002,
      "predictions_above_one_before_capping": 1449
    },
    {
      "repeat": 0,
      "fold": 2,
      "held_out_molecules": 218,
      "supported_cases": 83319,
      "unsupported_cases": 1705,
      "predictions_above_one_before_capping": 1021
    },
    {
      "repeat": 0,
      "fold": 3,
      "held_out_molecules": 217,
      "supported_cases": 90527,
      "unsupported_cases": 2457,
      "predictions_above_one_before_capping": 650
    },
    {
      "repeat": 0,
      "fold": 4,
      "held_out_molecules": 217,
      "supported_cases": 85941,
      "unsupported_cases": 917,
      "predictions_above_one_before_capping": 556
    },
    {
      "repeat": 1,
      "fold": 0,
      "held_out_molecules": 218,
      "supported_cases": 77761,
      "unsupported_cases": 2336,
      "predictions_above_one_before_capping": 719
    },
    {
      "repeat": 1,
      "fold": 1,
      "held_out_molecules": 218,
      "supported_cases": 83371,
      "unsupported_cases": 1667,
      "predictions_above_one_before_capping": 1001
    },
    {
      "repeat": 1,
      "fold": 2,
      "held_out_molecules": 218,
      "supported_cases": 97709,
      "unsupported_cases": 360,
      "predictions_above_one_before_capping": 896
    },
    {
      "repeat": 1,
      "fold": 3,
      "held_out_molecules": 217,
      "supported_cases": 90510,
      "unsupported_cases": 1619,
      "predictions_above_one_before_capping": 1240
    },
    {
      "repeat": 1,
      "fold": 4,
      "held_out_molecules": 217,
      "supported_cases": 102789,
      "unsupported_cases": 655,
      "predictions_above_one_before_capping": 1086
    },
    {
      "repeat": 2,
      "fold": 0,
      "held_out_molecules": 218,
      "supported_cases": 93220,
      "unsupported_cases": 1042,
      "predictions_above_one_before_capping": 701
    },
    {
      "repeat": 2,
      "fold": 1,
      "held_out_molecules": 218,
      "supported_cases": 100316,
      "unsupported_cases": 1948,
      "predictions_above_one_before_capping": 1076
    },
    {
      "repeat": 2,
      "fold": 2,
      "held_out_molecules": 218,
      "supported_cases": 90967,
      "unsupported_cases": 1096,
      "predictions_above_one_before_capping": 1447
    },
    {
      "repeat": 2,
      "fold": 3,
      "held_out_molecules": 217,
      "supported_cases": 85579,
      "unsupported_cases": 633,
      "predictions_above_one_before_capping": 949
    },
    {
      "repeat": 2,
      "fold": 4,
      "held_out_molecules": 217,
      "supported_cases": 81768,
      "unsupported_cases": 2208,
      "predictions_above_one_before_capping": 903
    },
    {
      "repeat": 3,
      "fold": 0,
      "held_out_molecules": 218,
      "supported_cases": 92038,
      "unsupported_cases": 2898,
      "predictions_above_one_before_capping": 991
    },
    {
      "repeat": 3,
      "fold": 1,
      "held_out_molecules": 218,
      "supported_cases": 82776,
      "unsupported_cases": 420,
      "predictions_above_one_before_capping": 1204
    },
    {
      "repeat": 3,
      "fold": 2,
      "held_out_molecules": 218,
      "supported_cases": 100309,
      "unsupported_cases": 2600,
      "predictions_above_one_before_capping": 1103
    },
    {
      "repeat": 3,
      "fold": 3,
      "held_out_molecules": 217,
      "supported_cases": 93520,
      "unsupported_cases": 228,
      "predictions_above_one_before_capping": 940
    },
    {
      "repeat": 3,
      "fold": 4,
      "held_out_molecules": 217,
      "supported_cases": 83009,
      "unsupported_cases": 979,
      "predictions_above_one_before_capping": 717
    },
    {
      "repeat": 4,
      "fold": 0,
      "held_out_molecules": 218,
      "supported_cases": 89563,
      "unsupported_cases": 1415,
      "predictions_above_one_before_capping": 1079
    },
    {
      "repeat": 4,
      "fold": 1,
      "held_out_molecules": 218,
      "supported_cases": 96097,
      "unsupported_cases": 1913,
      "predictions_above_one_before_capping": 1237
    },
    {
      "repeat": 4,
      "fold": 2,
      "held_out_molecules": 218,
      "supported_cases": 85082,
      "unsupported_cases": 526,
      "predictions_above_one_before_capping": 828
    },
    {
      "repeat": 4,
      "fold": 3,
      "held_out_molecules": 217,
      "supported_cases": 73037,
      "unsupported_cases": 1189,
      "predictions_above_one_before_capping": 620
    },
    {
      "repeat": 4,
      "fold": 4,
      "held_out_molecules": 217,
      "supported_cases": 107621,
      "unsupported_cases": 2334,
      "predictions_above_one_before_capping": 1168
    },
    {
      "repeat": 5,
      "fold": 0,
      "held_out_molecules": 218,
      "supported_cases": 91209,
      "unsupported_cases": 861,
      "predictions_above_one_before_capping": 941
    },
    {
      "repeat": 5,
      "fold": 1,
      "held_out_molecules": 218,
      "supported_cases": 85583,
      "unsupported_cases": 715,
      "predictions_above_one_before_capping": 952
    },
    {
      "repeat": 5,
      "fold": 2,
      "held_out_molecules": 218,
      "supported_cases": 97044,
      "unsupported_cases": 608,
      "predictions_above_one_before_capping": 1118
    },
    {
      "repeat": 5,
      "fold": 3,
      "held_out_molecules": 217,
      "supported_cases": 87426,
      "unsupported_cases": 630,
      "predictions_above_one_before_capping": 829
    },
    {
      "repeat": 5,
      "fold": 4,
      "held_out_molecules": 217,
      "supported_cases": 91323,
      "unsupported_cases": 3378,
      "predictions_above_one_before_capping": 1047
    },
    {
      "repeat": 6,
      "fold": 0,
      "held_out_molecules": 218,
      "supported_cases": 94310,
      "unsupported_cases": 712,
      "predictions_above_one_before_capping": 1088
    },
    {
      "repeat": 6,
      "fold": 1,
      "held_out_molecules": 218,
      "supported_cases": 81067,
      "unsupported_cases": 2127,
      "predictions_above_one_before_capping": 1127
    },
    {
      "repeat": 6,
      "fold": 2,
      "held_out_molecules": 218,
      "supported_cases": 96573,
      "unsupported_cases": 1112,
      "predictions_above_one_before_capping": 1462
    },
    {
      "repeat": 6,
      "fold": 3,
      "held_out_molecules": 217,
      "supported_cases": 98363,
      "unsupported_cases": 1220,
      "predictions_above_one_before_capping": 378
    },
    {
      "repeat": 6,
      "fold": 4,
      "held_out_molecules": 217,
      "supported_cases": 82023,
      "unsupported_cases": 1270,
      "predictions_above_one_before_capping": 972
    },
    {
      "repeat": 7,
      "fold": 0,
      "held_out_molecules": 218,
      "supported_cases": 92489,
      "unsupported_cases": 2030,
      "predictions_above_one_before_capping": 1020
    },
    {
      "repeat": 7,
      "fold": 1,
      "held_out_molecules": 218,
      "supported_cases": 92623,
      "unsupported_cases": 557,
      "predictions_above_one_before_capping": 1178
    },
    {
      "repeat": 7,
      "fold": 2,
      "held_out_molecules": 218,
      "supported_cases": 103252,
      "unsupported_cases": 2048,
      "predictions_above_one_before_capping": 946
    },
    {
      "repeat": 7,
      "fold": 3,
      "held_out_molecules": 217,
      "supported_cases": 82882,
      "unsupported_cases": 594,
      "predictions_above_one_before_capping": 910
    },
    {
      "repeat": 7,
      "fold": 4,
      "held_out_molecules": 217,
      "supported_cases": 80874,
      "unsupported_cases": 1428,
      "predictions_above_one_before_capping": 923
    },
    {
      "repeat": 8,
      "fold": 0,
      "held_out_molecules": 218,
      "supported_cases": 92159,
      "unsupported_cases": 2396,
      "predictions_above_one_before_capping": 741
    },
    {
      "repeat": 8,
      "fold": 1,
      "held_out_molecules": 218,
      "supported_cases": 85565,
      "unsupported_cases": 1298,
      "predictions_above_one_before_capping": 875
    },
    {
      "repeat": 8,
      "fold": 2,
      "held_out_molecules": 218,
      "supported_cases": 93078,
      "unsupported_cases": 1919,
      "predictions_above_one_before_capping": 854
    },
    {
      "repeat": 8,
      "fold": 3,
      "held_out_molecules": 217,
      "supported_cases": 102615,
      "unsupported_cases": 1342,
      "predictions_above_one_before_capping": 1749
    },
    {
      "repeat": 8,
      "fold": 4,
      "held_out_molecules": 217,
      "supported_cases": 77684,
      "unsupported_cases": 721,
      "predictions_above_one_before_capping": 933
    },
    {
      "repeat": 9,
      "fold": 0,
      "held_out_molecules": 218,
      "supported_cases": 101531,
      "unsupported_cases": 2823,
      "predictions_above_one_before_capping": 1155
    },
    {
      "repeat": 9,
      "fold": 1,
      "held_out_molecules": 218,
      "supported_cases": 93193,
      "unsupported_cases": 314,
      "predictions_above_one_before_capping": 694
    },
    {
      "repeat": 9,
      "fold": 2,
      "held_out_molecules": 218,
      "supported_cases": 84529,
      "unsupported_cases": 912,
      "predictions_above_one_before_capping": 762
    },
    {
      "repeat": 9,
      "fold": 3,
      "held_out_molecules": 217,
      "supported_cases": 91919,
      "unsupported_cases": 1904,
      "predictions_above_one_before_capping": 1162
    },
    {
      "repeat": 9,
      "fold": 4,
      "held_out_molecules": 217,
      "supported_cases": 80710,
      "unsupported_cases": 942,
      "predictions_above_one_before_capping": 1193
    }
  ]
}
```

## Reproducibility

- folds.csv contains each molecule’s held-out fold for every repeat.
- fold_multipliers.csv contains every fitted fold’s solvent factors and connected component.
- full_data_multipliers.csv is a descriptive fit on all molecules, relative to water where connected; it was not used to score held-out predictions.
- cases.csv and baseline_errors.npz preserve case identities and every held-out prediction error.
- summary.json contains capped/uncapped errors, paired ordinary and simultaneous confidence intervals, and split-to-split ranges.
- manifest.json contains input/code hashes and the exact command arguments.
- Existing UNIFAC predictions were reused. No UNIFAC model was rerun and no production code or source data was modified.
