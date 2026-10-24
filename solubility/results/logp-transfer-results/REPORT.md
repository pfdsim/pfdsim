# Solubility transfer: solvent offsets plus computed logP

Fit two coefficient types per solvent: offset a_s and cLogP slope b_s. For a held-out solute i, log10(x_target / x_reference) = (a_target − a_reference) + (b_target − b_reference) cLogP_i. A single global solute logP term would cancel in this transfer.

## Protocol

- Reuse the multiplier benchmark’s exact molecule identities, cases, five folds and ten repeats. Match temperature and publication at 20–30 °C for the main training groups; all measurements of a molecule stay together in each split.
- Fit least squares with the same equal-molecule, equal-group, equal-pair weighting as the multiplier benchmark. No ridge penalty, tuning, extra descriptors or fitted solid terms.
- cLogP is RDKit Wildman–Crippen MolLogP computed from the full standard InChI used for the original molecular identity. No measured logP, target solubility, ionization correction, largest-fragment substitution or learned molecular descriptor.
- Center and scale cLogP using distinct training molecules only. Fix both water coefficients to zero in its connected component; anchor other components separately. Absolute solvent coefficients are gauge-dependent; transfers depend only on differences.
- Use a rank-revealing, diagonally equilibrated eigensolve. Reject disconnected, unseen-solvent, nonfinite-descriptor, or nullspace-dependent prediction contrasts. A connected graph alone does not identify solvent slopes. No arbitrary minimum-norm extrapolations are accepted where the transfer itself is not identifiable.
- Refit the multiplier-only baseline on identical training groups, and reconcile it with the old held-out baseline when all cLogP descriptors are available. Successful UNIFAC predictions are reused without rerunning thermodynamics.
- Main predicted log mole fractions are capped at zero (x ≤ 1), matching the old multiplier baseline. Literal uncapped errors and cap counts are retained; there is no arbitrary lower-solubility cutoff or clipping of large negative predictions.
- Every score and test compares identical cases, requiring availability in every repeat. Score individual held-out losses, then weight each molecule equally. R² and Pearson r use those same molecule weights, not a fit through the plot.

## Main and directional multiplier comparisons

| Population | Model | Transfers | Molecules | MAE | RMSE | R² | Pearson r | Within ×10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| same_publication | multiplier | 212,156 | 1,073 | 0.6573 | 0.9707 | 0.5154 | 0.7581 | 79.1% |
| same_publication | multiplier_logP | 212,156 | 1,073 | 0.5784 | 2.5359 | -2.3074 | 0.4182 | 82.9% |
| water_to_nonwater | multiplier | 7,678 | 467 | 1.2950 | 1.6654 | -0.8516 | 0.3541 | 49.5% |
| water_to_nonwater | multiplier_logP | 7,678 | 467 | 0.9903 | 1.3031 | -0.1336 | 0.5008 | 61.3% |
| nonwater_to_water | multiplier | 7,678 | 467 | 1.3341 | 1.7194 | 0.0412 | 0.3840 | 48.0% |
| nonwater_to_water | multiplier_logP | 7,678 | 467 | 1.0301 | 1.4905 | 0.2795 | 0.6446 | 60.4% |
| nonwater_to_nonwater | multiplier | 196,800 | 1,023 | 0.5306 | 0.7504 | 0.6176 | 0.8095 | 84.8% |
| nonwater_to_nonwater | multiplier_logP | 196,800 | 1,023 | 0.4998 | 2.5510 | -3.4192 | 0.3720 | 86.9% |
| pooled_publications | multiplier | 230,488 | 1,081 | 0.6880 | 1.0206 | 0.4755 | 0.7415 | 77.8% |
| pooled_publications | multiplier_logP | 230,488 | 1,081 | 0.6033 | 2.8424 | -3.0681 | 0.3807 | 82.0% |
| AqSolDB_approximate_water_anchor | multiplier | 2,188 | 267 | 1.7337 | 2.2037 | -2.1457 | 0.1136 | 36.9% |
| AqSolDB_approximate_water_anchor | multiplier_logP | 2,188 | 267 | 1.2298 | 1.7791 | -1.0502 | 0.2463 | 47.1% |

## UNIFAC on its own matched coverage

| Variant | Transfers | Molecules | UNIFAC MAE | Multiplier MAE | Multiplier + logP MAE |
|---|---:|---:|---:|---:|---:|
| UNIFAC | 40,558 | 256 | 0.5437 | 0.5329 | 0.4593 |
| UNIFAC2 | 74,672 | 346 | 0.5979 | 0.6271 | 0.5357 |
| UNIFDMD | 32,994 | 235 | 0.5319 | 0.5252 | 0.4339 |
| UNIFM2 | 74,303 | 351 | 0.6254 | 0.6254 | 0.5323 |
| UNIFNIST | 24,712 | 181 | 0.5754 | 0.5530 | 0.4555 |
| UNIFLBY | 13,820 | 118 | 0.4313 | 0.5067 | 0.4035 |

## Identical-case cohort: common_five

| Variant | Transfers | Molecules | UNIFAC MAE | Multiplier MAE | Multiplier + logP MAE |
|---|---:|---:|---:|---:|---:|
| UNIFAC | 16,928 | 145 | 0.3653 | 0.5411 | 0.4467 |
| UNIFAC2 | 16,928 | 145 | 0.3662 | 0.5411 | 0.4467 |
| UNIFDMD | 16,928 | 145 | 0.4343 | 0.5411 | 0.4467 |
| UNIFM2 | 16,928 | 145 | 0.3798 | 0.5411 | 0.4467 |
| UNIFNIST | 16,928 | 145 | 0.5083 | 0.5411 | 0.4467 |

## Identical-case cohort: common_six

| Variant | Transfers | Molecules | UNIFAC MAE | Multiplier MAE | Multiplier + logP MAE |
|---|---:|---:|---:|---:|---:|
| UNIFAC | 7,371 | 86 | 0.3732 | 0.5292 | 0.4240 |
| UNIFAC2 | 7,371 | 86 | 0.3750 | 0.5292 | 0.4240 |
| UNIFDMD | 7,371 | 86 | 0.3947 | 0.5292 | 0.4240 |
| UNIFM2 | 7,371 | 86 | 0.3798 | 0.5292 | 0.4240 |
| UNIFNIST | 7,371 | 86 | 0.4620 | 0.5292 | 0.4240 |
| UNIFLBY | 7,371 | 86 | 0.3659 | 0.5292 | 0.4240 |

## Paired molecule/repetition uncertainty

Differences use the order stated in each comparison name. The simultaneous intervals cover the seven primary comparisons: the new versus old baseline overall, and each of six UNIFAC variants versus the new baseline on its own matched coverage. Shared-cohort and directional scores are descriptive.

| Comparison | MAE difference | Ordinary 95% CI | Familywise 95% CI |
|---|---:|---|---|
| logP_minus_multiplier | -0.0789 | [-0.10181701643231829, -0.05141800989265251] | [-0.10989383552509246, -0.035400820974508695] |
| UNIFAC_minus_logP | +0.0844 | [0.027144296040548545, 0.14663929562236064] | [0.0072128335018128015, 0.1724021002915056] |
| UNIFAC2_minus_logP | +0.0622 | [0.017525409146117884, 0.11115978896818354] | [0.0020390657485197805, 0.13010903276054342] |
| UNIFDMD_minus_logP | +0.0979 | [0.03817297705189036, 0.1609207640587037] | [0.019796116063201967, 0.18995490856415193] |
| UNIFM2_minus_logP | +0.0931 | [0.04248869775953735, 0.14898625494912496] | [0.024847568944830818, 0.17268607408652895] |
| UNIFNIST_minus_logP | +0.1199 | [0.05220028517178111, 0.1942380170867156] | [0.0303509274262348, 0.22330885743192386] |
| UNIFLBY_minus_logP | +0.0278 | [-0.040668313627712685, 0.10107975257462493] | [-0.0632316061810782, 0.13250434013538018] |

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
  "solutes": 1088,
  "solvents": 195,
  "descriptor_failures": 0,
  "computed_logP_min": -6.295800000000012,
  "computed_logP_max": 17.853899999999975,
  "main_candidate_transfers": 217778,
  "baseline_supported_main_transfers": 213610,
  "new_model_supported_main_transfers": 212156,
  "baseline_reconciled_with_original": true,
  "maximum_relative_normal_residual": 1.3776352378177218e-13,
  "main_capped_metrics": {
    "predictions": 212156,
    "solutes": 1073,
    "MAE_log10": 0.5783802441139124,
    "RMSE_log10": 2.5358932221519663,
    "bias_log10": -0.01653345699352782,
    "geometric_mean_fold_error": 3.7877407310884044,
    "within_factor_2": 0.4302967042885859,
    "within_factor_10": 0.8291901551628471,
    "R2": -2.307425925181161,
    "Pearson_r": 0.418152545311787
  },
  "main_uncapped_metrics": {
    "predictions": 212156,
    "solutes": 1073,
    "MAE_log10": 0.5949137011074402,
    "RMSE_log10": 3.4930927493774018,
    "bias_log10": -3.313990898555692e-19,
    "geometric_mean_fold_error": 3.934718806756904,
    "within_factor_2": 0.42826983659660534,
    "within_factor_10": 0.8261232477598859,
    "R2": -5.275501460344006,
    "Pearson_r": 0.3185964070230629
  },
  "total_capped_predictions": 50671,
  "fold_graph_identifiability_refusals": 38038
}
```

## Interpretation and reproducibility

This predicts target solubility given a measured reference solubility; it is not an absolute aqueous-solubility regression. The scalar logP feature can express solute-dependent solvent preferences but cannot separately identify hydrogen bonding, specific functional groups, crystal forms, or aqueous ionization. Random molecular splits do not establish scaffold-held-out or external-dataset performance. Publication/temperature matching cannot guarantee identical solid form. Pooled-publication and AqSolDB-anchor scopes are separately labeled sensitivities; AqSolDB temperatures, pH, density conversion and species remain approximate.

The paired bootstrap conditions on the fitted predictions and resamples molecules and split repetitions (20,000 draws, seed 20261014). It does not include a complete refitting bootstrap.

- errors_capped.npz / errors_uncapped.npz: every case/repeat prediction error.
- baseline_errors.npz: the exactly training-matched multiplier-only errors.
- fold_parameters.csv: per-fold solvent offsets/slopes, anchors and training scaling.
- fold_audit.csv: numerical rank, support, extrapolation refusals and cap counts.
- molecules.csv: exact identities, resolved structures and computed logP.
- summary.json: full headline metrics, shared-case results and paired uncertainty.
- manifest.json: input, implementation and output fingerprints; RDKit version.
- No production code, datasets or previous predictions were changed.
