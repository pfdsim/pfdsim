# Aqueous count regressions: chemical-domain check

The full-data assessment is retained unchanged in ../aqueous-count-regression/. This companion refits the same models on neutral, single-component organic identities, defined structurally before fitting, without filtering on solubility or prediction error.

This domain contains 9,044 records; 1,377 records are outside it. The same molecular folds, nested size/ridge selection and training-only scaling are used. All group and atom baselines within a pool are refitted on identical molecules. Neutral representations can still ionize in water; unknown pH and solid form remain limitations.

## neutral_single_organic

| Model | Molecules | MAE log10 | RMSE log10 | R² | Within factor 2 | Within factor 10 |
|---|---:|---:|---:|---:|---:|---:|
| training_mean | 9,044 | 1.8180 | 2.2987 | -0.0001 | 10.7% | 34.5% |
| heavy_atoms_OLS | 9,044 | 1.4700 | 1.9348 | 0.2915 | 14.4% | 43.8% |
| total_atoms_OLS | 9,044 | 1.5624 | 2.0610 | 0.1960 | 13.0% | 42.6% |
| size_selected | 9,044 | 1.4156 | 1.8412 | 0.3584 | 14.6% | 45.0% |
| element_counts_OLS | 9,044 | 1.0933 | 1.4387 | 0.6082 | 18.2% | 56.3% |
| element_counts_adaptive | 9,044 | 1.0243 | 1.3437 | 0.6582 | 20.1% | 59.1% |

## neutral_UNIFAC

| Model | Molecules | MAE log10 | RMSE log10 | R² | Within factor 2 | Within factor 10 |
|---|---:|---:|---:|---:|---:|---:|
| training_mean | 4,122 | 1.9471 | 2.4455 | -0.0002 | 10.2% | 32.7% |
| heavy_atoms_OLS | 4,122 | 1.5404 | 2.0227 | 0.3157 | 13.6% | 43.0% |
| total_atoms_OLS | 4,122 | 1.6439 | 2.1763 | 0.2079 | 12.4% | 41.8% |
| size_selected | 4,122 | 1.4598 | 1.8842 | 0.4063 | 15.0% | 44.1% |
| element_counts_OLS | 4,122 | 1.0047 | 1.3268 | 0.7056 | 20.3% | 59.9% |
| element_counts_adaptive | 4,122 | 0.9460 | 1.2414 | 0.7423 | 22.0% | 62.3% |
| UNIFAC_main_OLS | 4,122 | 0.8906 | 1.2081 | 0.7559 | 23.8% | 66.0% |
| UNIFAC_subgroup_OLS | 4,122 | 0.8554 | 1.1679 | 0.7719 | 25.4% | 67.7% |
| UNIFAC_subgroup_adaptive | 4,122 | 0.7878 | 1.0720 | 0.8078 | 27.9% | 71.2% |

## neutral_UNIFDMD

| Model | Molecules | MAE log10 | RMSE log10 | R² | Within factor 2 | Within factor 10 |
|---|---:|---:|---:|---:|---:|---:|
| training_mean | 4,204 | 1.9391 | 2.4383 | -0.0002 | 10.1% | 33.2% |
| heavy_atoms_OLS | 4,204 | 1.5025 | 1.9592 | 0.3542 | 13.4% | 43.5% |
| total_atoms_OLS | 4,204 | 1.6196 | 2.1466 | 0.2248 | 12.8% | 42.0% |
| size_selected | 4,204 | 1.4294 | 1.8432 | 0.4284 | 15.2% | 44.8% |
| element_counts_OLS | 4,204 | 1.0105 | 1.3308 | 0.7021 | 20.0% | 59.4% |
| element_counts_adaptive | 4,204 | 0.9344 | 1.2214 | 0.7490 | 22.3% | 62.5% |
| UNIFDMD_main_OLS | 4,204 | 0.8788 | 1.1896 | 0.7619 | 24.0% | 66.5% |
| UNIFDMD_subgroup_OLS | 4,204 | 0.8556 | 1.1644 | 0.7719 | 25.3% | 67.9% |
| UNIFDMD_subgroup_adaptive | 4,204 | 0.7697 | 1.0406 | 0.8178 | 28.5% | 72.6% |

## neutral_UNIFNIST

| Model | Molecules | MAE log10 | RMSE log10 | R² | Within factor 2 | Within factor 10 |
|---|---:|---:|---:|---:|---:|---:|
| training_mean | 4,602 | 1.9517 | 2.4543 | -0.0003 | 10.9% | 32.8% |
| heavy_atoms_OLS | 4,602 | 1.5844 | 2.0698 | 0.2886 | 14.0% | 41.4% |
| total_atoms_OLS | 4,602 | 1.6897 | 2.2140 | 0.1861 | 12.2% | 40.2% |
| size_selected | 4,602 | 1.4959 | 1.9254 | 0.3844 | 14.4% | 43.7% |
| element_counts_OLS | 4,602 | 1.0530 | 1.3817 | 0.6830 | 19.2% | 57.9% |
| element_counts_adaptive | 4,602 | 0.9866 | 1.2945 | 0.7218 | 21.5% | 60.8% |
| UNIFNIST_main_OLS | 4,602 | 0.8806 | 1.2043 | 0.7592 | 24.6% | 66.8% |
| UNIFNIST_subgroup_OLS | 4,602 | 0.8633 | 1.1922 | 0.7640 | 25.4% | 68.0% |
| UNIFNIST_subgroup_adaptive | 4,602 | 0.7996 | 1.1021 | 0.7983 | 28.2% | 71.0% |

## neutral_UNIFLBY

| Model | Molecules | MAE log10 | RMSE log10 | R² | Within factor 2 | Within factor 10 |
|---|---:|---:|---:|---:|---:|---:|
| training_mean | 2,012 | 1.8766 | 2.3229 | -0.0007 | 10.3% | 31.8% |
| heavy_atoms_OLS | 2,012 | 1.4241 | 1.8017 | 0.3980 | 14.2% | 42.7% |
| total_atoms_OLS | 2,012 | 1.4184 | 1.8361 | 0.3747 | 14.4% | 43.8% |
| size_selected | 2,012 | 1.3150 | 1.7036 | 0.4617 | 16.3% | 48.6% |
| element_counts_OLS | 2,012 | 0.9878 | 1.3067 | 0.6833 | 21.0% | 59.4% |
| element_counts_adaptive | 2,012 | 0.8829 | 1.1496 | 0.7549 | 23.6% | 65.1% |
| UNIFLBY_main_OLS | 2,012 | 0.9512 | 1.2788 | 0.6967 | 22.2% | 61.6% |
| UNIFLBY_subgroup_OLS | 2,012 | 0.9305 | 1.2520 | 0.7093 | 22.1% | 64.0% |
| UNIFLBY_subgroup_adaptive | 2,012 | 0.8052 | 1.0749 | 0.7857 | 26.7% | 69.7% |

## neutral_common

| Model | Molecules | MAE log10 | RMSE log10 | R² | Within factor 2 | Within factor 10 |
|---|---:|---:|---:|---:|---:|---:|
| training_mean | 1,890 | 1.8924 | 2.3506 | -0.0007 | 10.7% | 31.9% |
| heavy_atoms_OLS | 1,890 | 1.4541 | 1.8344 | 0.3906 | 13.6% | 41.8% |
| total_atoms_OLS | 1,890 | 1.4613 | 1.8773 | 0.3617 | 13.5% | 42.0% |
| size_selected | 1,890 | 1.3474 | 1.7333 | 0.4559 | 16.0% | 46.9% |
| element_counts_OLS | 1,890 | 0.9861 | 1.3145 | 0.6871 | 21.5% | 59.8% |
| element_counts_adaptive | 1,890 | 0.8702 | 1.1401 | 0.7646 | 24.6% | 66.1% |
| UNIFAC_main_OLS | 1,890 | 0.9578 | 1.2869 | 0.7001 | 21.8% | 60.8% |
| UNIFAC_subgroup_OLS | 1,890 | 0.9321 | 1.2594 | 0.7127 | 22.5% | 63.8% |
| UNIFAC_subgroup_adaptive | 1,890 | 0.8000 | 1.0774 | 0.7898 | 26.7% | 70.7% |
| UNIFDMD_main_OLS | 1,890 | 0.9548 | 1.2870 | 0.7000 | 22.6% | 61.8% |
| UNIFDMD_subgroup_OLS | 1,890 | 0.9271 | 1.2533 | 0.7155 | 22.6% | 64.6% |
| UNIFDMD_subgroup_adaptive | 1,890 | 0.7778 | 1.0512 | 0.7999 | 27.5% | 71.7% |
| UNIFNIST_main_OLS | 1,890 | 0.9239 | 1.2713 | 0.7073 | 23.7% | 64.4% |
| UNIFNIST_subgroup_OLS | 1,890 | 0.8952 | 1.2369 | 0.7229 | 24.3% | 67.3% |
| UNIFNIST_subgroup_adaptive | 1,890 | 0.7652 | 1.0589 | 0.7970 | 28.8% | 73.4% |
| UNIFLBY_main_OLS | 1,890 | 0.9592 | 1.2879 | 0.6996 | 21.9% | 61.2% |
| UNIFLBY_subgroup_OLS | 1,890 | 0.9387 | 1.2635 | 0.7109 | 21.8% | 63.4% |
| UNIFLBY_subgroup_adaptive | 1,890 | 0.8081 | 1.0841 | 0.7872 | 26.7% | 70.1% |

## Why retain both populations?

The broad dataset contains salts, inorganic species and disconnected mixtures. Its strongest adaptive-atom extrapolation failure is a 24-fragment, 388-heavy-atom record: several held-out predictions exceed log10(S/mol/L)=79, which is physically impossible. Such results are retained as failures of this empirical regression rather than hidden by clipping. The structural domain check distinguishes ordinary molecular prediction from that broader input population.

Full-data coefficients are descriptive refits, not held-out predictions. Original data, prior predictions, features and production code were unchanged.
