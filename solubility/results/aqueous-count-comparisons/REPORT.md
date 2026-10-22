# Paired comparisons of aqueous count regressions

Use saved held-out errors on identical molecules/folds. Negative differences favor the first model. Bootstrap whole molecules and split repetitions 20,000 times, seed 20261022; familywise intervals account for all 45 comparisons below. This assesses random-molecule generalization within this dataset, not unseen-scaffold or independently collected external data.

| Population | First model | Second model | Molecules | MAE difference | Familywise 95% CI |
|---|---|---|---:|---:|---|
| neutral_single_organic | size_selected | heavy_atoms_OLS | 9,044 | -0.0544 | -0.0708 to -0.0378 |
| neutral_single_organic | element_counts_OLS | size_selected | 9,044 | -0.3223 | -0.3614 to -0.2842 |
| neutral_single_organic | element_counts_adaptive | element_counts_OLS | 9,044 | -0.0690 | -0.0851 to -0.0538 |
| neutral_UNIFAC | size_selected | heavy_atoms_OLS | 4,122 | -0.0805 | -0.1118 to -0.0507 |
| neutral_UNIFAC | element_counts_OLS | size_selected | 4,122 | -0.4551 | -0.5180 to -0.3918 |
| neutral_UNIFAC | element_counts_adaptive | element_counts_OLS | 4,122 | -0.0587 | -0.0805 to -0.0357 |
| neutral_UNIFAC | UNIFAC_subgroup_OLS | element_counts_OLS | 4,122 | -0.1493 | -0.1784 to -0.1202 |
| neutral_UNIFAC | UNIFAC_subgroup_adaptive | element_counts_adaptive | 4,122 | -0.1582 | -0.1870 to -0.1292 |
| neutral_UNIFAC | UNIFAC_subgroup_adaptive | UNIFAC_subgroup_OLS | 4,122 | -0.0676 | -0.0891 to -0.0463 |
| neutral_UNIFDMD | size_selected | heavy_atoms_OLS | 4,204 | -0.0731 | -0.1017 to -0.0459 |
| neutral_UNIFDMD | element_counts_OLS | size_selected | 4,204 | -0.4188 | -0.4814 to -0.3566 |
| neutral_UNIFDMD | element_counts_adaptive | element_counts_OLS | 4,204 | -0.0762 | -0.0998 to -0.0536 |
| neutral_UNIFDMD | UNIFDMD_subgroup_OLS | element_counts_OLS | 4,204 | -0.1549 | -0.1832 to -0.1252 |
| neutral_UNIFDMD | UNIFDMD_subgroup_adaptive | element_counts_adaptive | 4,204 | -0.1647 | -0.1943 to -0.1344 |
| neutral_UNIFDMD | UNIFDMD_subgroup_adaptive | UNIFDMD_subgroup_OLS | 4,204 | -0.0859 | -0.1089 to -0.0650 |
| neutral_UNIFNIST | size_selected | heavy_atoms_OLS | 4,602 | -0.0884 | -0.1190 to -0.0600 |
| neutral_UNIFNIST | element_counts_OLS | size_selected | 4,602 | -0.4429 | -0.5028 to -0.3833 |
| neutral_UNIFNIST | element_counts_adaptive | element_counts_OLS | 4,602 | -0.0663 | -0.0874 to -0.0458 |
| neutral_UNIFNIST | UNIFNIST_subgroup_OLS | element_counts_OLS | 4,602 | -0.1897 | -0.2214 to -0.1577 |
| neutral_UNIFNIST | UNIFNIST_subgroup_adaptive | element_counts_adaptive | 4,602 | -0.1870 | -0.2182 to -0.1562 |
| neutral_UNIFNIST | UNIFNIST_subgroup_adaptive | UNIFNIST_subgroup_OLS | 4,602 | -0.0637 | -0.0835 to -0.0455 |
| neutral_UNIFLBY | size_selected | heavy_atoms_OLS | 2,012 | -0.1090 | -0.1573 to -0.0638 |
| neutral_UNIFLBY | element_counts_OLS | size_selected | 2,012 | -0.3272 | -0.4119 to -0.2459 |
| neutral_UNIFLBY | element_counts_adaptive | element_counts_OLS | 2,012 | -0.1050 | -0.1468 to -0.0665 |
| neutral_UNIFLBY | UNIFLBY_subgroup_OLS | element_counts_OLS | 2,012 | -0.0573 | -0.0862 to -0.0289 |
| neutral_UNIFLBY | UNIFLBY_subgroup_adaptive | element_counts_adaptive | 2,012 | -0.0777 | -0.1124 to -0.0431 |
| neutral_UNIFLBY | UNIFLBY_subgroup_adaptive | UNIFLBY_subgroup_OLS | 2,012 | -0.1254 | -0.1607 to -0.0916 |
| neutral_common | size_selected | heavy_atoms_OLS | 1,890 | -0.1067 | -0.1581 to -0.0629 |
| neutral_common | element_counts_OLS | size_selected | 1,890 | -0.3613 | -0.4471 to -0.2764 |
| neutral_common | element_counts_adaptive | element_counts_OLS | 1,890 | -0.1158 | -0.1624 to -0.0731 |
| neutral_common | UNIFAC_subgroup_OLS | element_counts_OLS | 1,890 | -0.0540 | -0.0861 to -0.0219 |
| neutral_common | UNIFAC_subgroup_adaptive | element_counts_adaptive | 1,890 | -0.0703 | -0.1046 to -0.0353 |
| neutral_common | UNIFAC_subgroup_adaptive | UNIFAC_subgroup_OLS | 1,890 | -0.1321 | -0.1694 to -0.0966 |
| neutral_common | UNIFDMD_subgroup_OLS | element_counts_OLS | 1,890 | -0.0590 | -0.0949 to -0.0250 |
| neutral_common | UNIFDMD_subgroup_adaptive | element_counts_adaptive | 1,890 | -0.0924 | -0.1285 to -0.0562 |
| neutral_common | UNIFDMD_subgroup_adaptive | UNIFDMD_subgroup_OLS | 1,890 | -0.1493 | -0.1940 to -0.1066 |
| neutral_common | UNIFNIST_subgroup_OLS | element_counts_OLS | 1,890 | -0.0908 | -0.1332 to -0.0491 |
| neutral_common | UNIFNIST_subgroup_adaptive | element_counts_adaptive | 1,890 | -0.1050 | -0.1460 to -0.0629 |
| neutral_common | UNIFNIST_subgroup_adaptive | UNIFNIST_subgroup_OLS | 1,890 | -0.1300 | -0.1683 to -0.0976 |
| neutral_common | UNIFLBY_subgroup_OLS | element_counts_OLS | 1,890 | -0.0474 | -0.0783 to -0.0181 |
| neutral_common | UNIFLBY_subgroup_adaptive | element_counts_adaptive | 1,890 | -0.0621 | -0.0961 to -0.0280 |
| neutral_common | UNIFLBY_subgroup_adaptive | UNIFLBY_subgroup_OLS | 1,890 | -0.1306 | -0.1678 to -0.0965 |
| neutral_common | UNIFNIST_subgroup_adaptive | UNIFDMD_subgroup_adaptive | 1,890 | -0.0126 | -0.0380 to +0.0129 |
| neutral_common | UNIFDMD_subgroup_adaptive | UNIFAC_subgroup_adaptive | 1,890 | -0.0221 | -0.0452 to +0.0001 |
| neutral_common | UNIFNIST_subgroup_adaptive | UNIFAC_subgroup_adaptive | 1,890 | -0.0348 | -0.0587 to -0.0065 |

Regularization and size augmentation are selected jointly for the adaptive count models. Their advantage over OLS does not isolate the effect of the nonlinear size term alone. Model choices used inner training CV; the paired statistics do not select or refit models.
