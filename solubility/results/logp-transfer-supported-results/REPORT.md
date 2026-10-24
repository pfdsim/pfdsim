# Solubility transfer: solvent offsets plus computed logP

Fit two coefficient types per solvent: offset a_s and cLogP slope b_s. For a held-out solute i, log10(x_target / x_reference) = (a_target − a_reference) + (b_target − b_reference) cLogP_i. A single global solute logP term would cancel in this transfer.

## Protocol

- Reuse the multiplier benchmark’s exact molecule identities, cases, five folds and ten repeats. Match temperature and publication at 20–30 °C for the main training groups; all measurements of a molecule stay together in each split.
- Fit least squares with the same equal-molecule, equal-group, equal-pair weighting as the multiplier benchmark. No ridge penalty, tuning, extra descriptors or fitted solid terms.
- A solvent slope is fixed to ZERO unless at least 20 distinct training molecules were measured in that solvent. Repeated temperatures, publications and pairs cannot inflate support. Counts and eligibility are recalculated within each outer training fold. The offset remains fitted for every connected observed solvent. The threshold is a declared support rule, not selected from test errors.
- cLogP is RDKit Wildman–Crippen MolLogP computed from the full standard InChI used for the original molecular identity. No measured logP, target solubility, ionization correction, largest-fragment substitution or learned molecular descriptor.
- Center and scale cLogP using distinct training molecules only. Fix both water coefficients to zero in its connected component; anchor other components separately. Zero slopes for sparse solvents mean no logP dependence relative to that anchor; this is an explicit simplifying assumption, not a claim about their physical behavior.
- Use a rank-revealing, diagonally equilibrated eigensolve. Reject disconnected, unseen-solvent, nonfinite-descriptor, or nullspace-dependent prediction contrasts. A connected graph alone does not identify solvent slopes. No arbitrary minimum-norm extrapolations are accepted where the transfer itself is not identifiable.
- Refit the multiplier-only baseline on identical training groups, and reconcile it with the old held-out baseline when all cLogP descriptors are available. Successful UNIFAC predictions are reused without rerunning thermodynamics.
- Main predicted log mole fractions are capped at zero (x ≤ 1), matching the old multiplier baseline. Literal uncapped errors and cap counts are retained; there is no arbitrary lower-solubility cutoff or clipping of large negative predictions.
- Every score and test compares identical cases, requiring availability in every repeat. Score individual held-out losses, then weight each molecule equally. R² and Pearson r use those same molecule weights, not a fit through the plot.

## Main and directional multiplier comparisons

| Population | Model | Transfers | Molecules | MAE | RMSE | R² | Pearson r | Within ×10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| same_publication | multiplier | 213,610 | 1,074 | 0.6578 | 0.9710 | 0.5146 | 0.7578 | 79.1% |
| same_publication | multiplier_logP | 213,610 | 1,074 | 0.6002 | 0.8739 | 0.6068 | 0.8048 | 81.1% |
| water_to_nonwater | multiplier | 7,697 | 467 | 1.2955 | 1.6656 | -0.8504 | 0.3548 | 49.4% |
| water_to_nonwater | multiplier_logP | 7,697 | 467 | 1.0205 | 1.3372 | -0.1927 | 0.4804 | 59.9% |
| nonwater_to_water | multiplier | 7,697 | 467 | 1.3346 | 1.7196 | 0.0410 | 0.3839 | 47.9% |
| nonwater_to_water | multiplier_logP | 7,697 | 467 | 1.0411 | 1.3692 | 0.3920 | 0.6495 | 58.9% |
| nonwater_to_nonwater | multiplier | 198,216 | 1,024 | 0.5312 | 0.7512 | 0.6168 | 0.8092 | 84.8% |
| nonwater_to_nonwater | multiplier_logP | 198,216 | 1,024 | 0.5188 | 0.7460 | 0.6221 | 0.8150 | 85.4% |
| pooled_publications | multiplier | 232,612 | 1,082 | 0.6884 | 1.0209 | 0.4748 | 0.7413 | 77.7% |
| pooled_publications | multiplier_logP | 232,612 | 1,082 | 0.6273 | 0.9216 | 0.5720 | 0.7899 | 80.0% |
| AqSolDB_approximate_water_anchor | multiplier | 2,202 | 267 | 1.7337 | 2.2038 | -2.1450 | 0.1137 | 36.9% |
| AqSolDB_approximate_water_anchor | multiplier_logP | 2,202 | 267 | 1.3448 | 1.6892 | -0.8477 | 0.2540 | 44.1% |

## UNIFAC on its own matched coverage

| Variant | Transfers | Molecules | UNIFAC MAE | Multiplier MAE | Multiplier + logP MAE |
|---|---:|---:|---:|---:|---:|
| UNIFAC | 40,910 | 256 | 0.5438 | 0.5329 | 0.4900 |
| UNIFAC2 | 75,208 | 346 | 0.5984 | 0.6272 | 0.5643 |
| UNIFDMD | 33,249 | 235 | 0.5317 | 0.5253 | 0.4788 |
| UNIFM2 | 74,824 | 351 | 0.6252 | 0.6255 | 0.5670 |
| UNIFNIST | 24,947 | 181 | 0.5756 | 0.5531 | 0.5061 |
| UNIFLBY | 13,930 | 118 | 0.4314 | 0.5068 | 0.4508 |

## Identical-case cohort: common_five

| Variant | Transfers | Molecules | UNIFAC MAE | Multiplier MAE | Multiplier + logP MAE |
|---|---:|---:|---:|---:|---:|
| UNIFAC | 17,135 | 145 | 0.3636 | 0.5411 | 0.4974 |
| UNIFAC2 | 17,135 | 145 | 0.3664 | 0.5411 | 0.4974 |
| UNIFDMD | 17,135 | 145 | 0.4335 | 0.5411 | 0.4974 |
| UNIFM2 | 17,135 | 145 | 0.3797 | 0.5411 | 0.4974 |
| UNIFNIST | 17,135 | 145 | 0.5080 | 0.5411 | 0.4974 |

## Identical-case cohort: common_six

| Variant | Transfers | Molecules | UNIFAC MAE | Multiplier MAE | Multiplier + logP MAE |
|---|---:|---:|---:|---:|---:|
| UNIFAC | 7,425 | 86 | 0.3712 | 0.5279 | 0.4798 |
| UNIFAC2 | 7,425 | 86 | 0.3736 | 0.5279 | 0.4798 |
| UNIFDMD | 7,425 | 86 | 0.3967 | 0.5279 | 0.4798 |
| UNIFM2 | 7,425 | 86 | 0.3786 | 0.5279 | 0.4798 |
| UNIFNIST | 7,425 | 86 | 0.4692 | 0.5279 | 0.4798 |
| UNIFLBY | 7,425 | 86 | 0.3657 | 0.5279 | 0.4798 |

## Paired molecule/repetition uncertainty

Differences use the order stated in each comparison name. The simultaneous intervals cover the seven primary comparisons: the new versus old baseline overall, and each of six UNIFAC variants versus the new baseline on its own matched coverage. Shared-cohort and directional scores are descriptive.

| Comparison | MAE difference | Ordinary 95% CI | Familywise 95% CI |
|---|---:|---|---|
| logP_minus_multiplier | -0.0575 | [-0.0730782321348871, -0.04265452943407224] | [-0.07938757682616725, -0.037543583337288106] |
| UNIFAC_minus_logP | +0.0538 | [-0.0037078748191790227, 0.11569393558509165] | [-0.022631245895056253, 0.14220747425016925] |
| UNIFAC2_minus_logP | +0.0341 | [-0.01136118754773136, 0.08334775923513889] | [-0.027748157878114924, 0.10149191850849405] |
| UNIFDMD_minus_logP | +0.0529 | [-0.007811654779610787, 0.11715924146264221] | [-0.02801687096349089, 0.1447535495954738] |
| UNIFM2_minus_logP | +0.0582 | [0.007879992293825328, 0.11290247018923337] | [-0.011862509996789367, 0.13717460483912317] |
| UNIFNIST_minus_logP | +0.0695 | [-0.001343372925612202, 0.1463871648900779] | [-0.025789189434935942, 0.17838097803741984] |
| UNIFLBY_minus_logP | -0.0195 | [-0.08832426311485683, 0.05424854093986904] | [-0.1134646192191925, 0.08495448544365335] |

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
  "new_model_supported_main_transfers": 213610,
  "baseline_reconciled_with_original": true,
  "maximum_relative_normal_residual": 4.038229577635199e-14,
  "main_capped_metrics": {
    "predictions": 213610,
    "solutes": 1074,
    "MAE_log10": 0.6002466909125653,
    "RMSE_log10": 0.8739436166864357,
    "bias_log10": -0.007298430216747649,
    "geometric_mean_fold_error": 3.983333703404406,
    "within_factor_2": 0.4167869413156366,
    "within_factor_10": 0.8110707425968368,
    "R2": 0.6068295104692912,
    "Pearson_r": 0.8048290937851652
  },
  "main_uncapped_metrics": {
    "predictions": 213610,
    "solutes": 1074,
    "MAE_log10": 0.607545121129313,
    "RMSE_log10": 0.8879670417794173,
    "bias_log10": 4.730424983661433e-19,
    "geometric_mean_fold_error": 4.0508402925261775,
    "within_factor_2": 0.4147540616200964,
    "within_factor_10": 0.8073581226361416,
    "R2": 0.5941105368708209,
    "Pearson_r": 0.8008693778370141
  },
  "total_capped_predictions": 67964,
  "fold_graph_identifiability_refusals": 0
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
