# UNIFAC solubility transfer at room temperature

Completed evaluation of the six pfdsim liquid-activity variants using BigSolDB 2.0 and AqSolDB.

## Main protocol

- Include BigSolDB measurements at 293.15–303.15 K inclusive. Match exact numeric temperature and source DOI.
- Group solutes and solvents by full standard InChI. Keep stereochemistry, salts and protonation distinctions; normalize standard-InChI tautomers.
- Within each solute/solvent/temperature/DOI group, use the median log10 mole fraction. Count each directed solvent transfer once.
- Keep single-component molecules with zero total formal charge. Neutral nitro resonance structures remain eligible. Unsupported structures are rejected by pfdsim, not approximated with different groups.
- Use the first canonical dataset SMILES for each chemical identity. Dataset graphs are the supplied identities; this is not a comprehensive structure-curation study.
- Use pfdsim get_unifac_groups, the shipped variant-specific parameter files, and CompiledUNIFACBackend. No parameters are fitted.
- Require every ordered main-group interaction used by a binary model. Do not silently accept the runtime zero-interaction fallback.
- Infer a_ref = x_ref gamma_ref(T,x_ref), then solve x_pred gamma_target(T,x_pred) = a_ref. Target measurements are used only to score the prediction.
- Search the full binary composition interval (mole-fraction floor 1e-30). Use the first stable solvent-rich root. Check binary tangent-plane distances on a grid and refine detected minima; tolerance -1e-7 RT. This is a numerical stability screen, not an analytic proof.
- Reject unstable reference liquids. Preserve failure counts; successful predictions do not replace unavailable or invalid cases.
- Score e = log10(x_pred/x_measured). Average |e| and e² within each solute first, then across solutes. RMSE is the square root of mean solute mean-square error. Within-factor fractions are also solute-balanced.
- Confidence intervals resample whole solutes (1,000 draws for scores; 5,000 paired draws for differences; seed 20261014). Correlated temperatures, directions and publications are not treated as independent samples.

The same solid form, pressure and chemical species must apply in both solvents for the solid contribution to cancel. Matching a publication and temperature improves comparability but does not verify polymorph/solvate identity or solution pH.

## Input audit

```json
{
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
  "AqSolDB_approximate_water_anchor_directed_predictions": 2235,
  "evaluated_solutes": 1088
}
```

## Same cases, all six variants

7,647 successful directed predictions common to all six; 86 distinct solutes. These are the same cases for every row.

| Variant | MAE log10 | 95% CI | RMSE log10 | Geometric mean fold error | Within 2x | Within 10x |
|---|---:|---|---:|---:|---:|---:|
| UNIFAC | 0.3789 | 0.3140–0.4437 | 0.6555 | 2.393 | 58.3% | 92.8% |
| UNIFAC2 | 0.3754 | 0.3081–0.4481 | 0.6433 | 2.373 | 60.5% | 91.4% |
| UNIFDMD | 0.3965 | 0.3109–0.4876 | 0.7079 | 2.492 | 60.9% | 90.6% |
| UNIFM2 | 0.3823 | 0.3054–0.4648 | 0.6881 | 2.411 | 64.0% | 91.3% |
| UNIFNIST | 0.4694 | 0.3937–0.5580 | 0.7841 | 2.947 | 54.8% | 86.9% |
| UNIFLBY | 0.3700 | 0.2896–0.4612 | 0.7006 | 2.344 | 65.6% | 91.5% |
| Ideal solution | 0.4738 | 0.4030–0.5447 | 0.7147 | 2.977 | 53.6% | 85.0% |

## Larger common set excluding Lyngby

17,713 directed predictions across 145 solutes. Lyngby is not compared on this cohort because its coverage differs.

| Variant | MAE log10 | 95% CI | RMSE log10 | Geometric mean fold error | Within 2x | Within 10x |
|---|---:|---|---:|---:|---:|---:|
| UNIFAC | 0.3673 | 0.3219–0.4144 | 0.6199 | 2.330 | 60.2% | 92.8% |
| UNIFAC2 | 0.3670 | 0.3218–0.4159 | 0.6147 | 2.328 | 61.0% | 92.5% |
| UNIFDMD | 0.4329 | 0.3630–0.5144 | 0.8108 | 2.710 | 58.5% | 89.7% |
| UNIFM2 | 0.3813 | 0.3269–0.4383 | 0.6644 | 2.406 | 62.9% | 91.3% |
| UNIFNIST | 0.5078 | 0.4333–0.5943 | 0.9145 | 3.220 | 54.3% | 84.9% |
| Ideal solution | 0.5365 | 0.4642–0.6234 | 0.8728 | 3.439 | 52.8% | 82.6% |

## Full eligible-data coverage

Own-coverage MAEs below use different compounds and must not be used as an unbiased accuracy ranking.

| Variant | Successful directed predictions | Solutes | Own-coverage MAE | Unavailable model | Unstable reference |
|---|---:|---:|---:|---:|---:|
| UNIFAC | 42,080 | 256 | 0.5446 | 174,676 | 1,022 |
| UNIFAC2 | 76,701 | 348 | 0.5969 | 139,714 | 1,363 |
| UNIFDMD | 34,286 | 237 | 0.5323 | 181,972 | 1,520 |
| UNIFM2 | 76,272 | 354 | 0.6214 | 139,054 | 2,452 |
| UNIFNIST | 25,622 | 182 | 0.5729 | 190,064 | 2,092 |
| UNIFLBY | 14,262 | 118 | 0.4358 | 203,064 | 452 |

## Water versus other solvents

Same five-variant common cohort. The successful reference-stability screen can make the two water directions have different cohorts.

| Variant | Non-water → non-water MAE | Water → non-water MAE | Non-water → water MAE |
|---|---:|---:|---:|
| UNIFAC | 0.3427 | 0.6499 | 1.1276 |
| UNIFAC2 | 0.3446 | 0.6593 | 1.1627 |
| UNIFDMD | 0.4032 | 0.7714 | 1.2478 |
| UNIFM2 | 0.3641 | 0.6171 | 1.2238 |
| UNIFNIST | 0.4567 | 1.0099 | 1.5708 |
- water_to_nonwater: 221 directed predictions, 17 solutes.
- nonwater_to_water: 423 directed predictions, 32 solutes.
- nonwater_to_nonwater: 17,069 directed predictions, 143 solutes.

## Pooled-publication sensitivity

Match solute and exact temperature, but pool publications before taking the median log(x). Solid-form mismatches are more plausible here.

| Variant | MAE log10 | 95% CI | RMSE log10 | Geometric mean fold error | Within 2x | Within 10x |
|---|---:|---|---:|---:|---:|---:|
| UNIFAC | 0.3897 | 0.3272–0.4527 | 0.6554 | 2.453 | 56.0% | 92.6% |
| UNIFAC2 | 0.3827 | 0.3190–0.4539 | 0.6494 | 2.414 | 59.3% | 91.2% |
| UNIFDMD | 0.4100 | 0.3254–0.5030 | 0.7234 | 2.570 | 59.8% | 90.3% |
| UNIFM2 | 0.3894 | 0.3113–0.4725 | 0.6914 | 2.451 | 63.0% | 91.2% |
| UNIFNIST | 0.4780 | 0.4030–0.5623 | 0.7889 | 3.006 | 53.7% | 86.7% |
| UNIFLBY | 0.3842 | 0.3069–0.4728 | 0.7045 | 2.422 | 62.7% | 91.3% |
| Ideal solution | 0.4860 | 0.4174–0.5562 | 0.7273 | 3.062 | 52.4% | 84.2% |

## AqSolDB water anchors: exploratory only

AqSolDB has no per-compound temperature. Use nominal 298.15 K, BigSolDB targets measured exactly at 298.15 K, and aqueous concentration <=0.1 mol/L. Convert C_water to approximate x_water = C/(55.3446851784+C), using the supplied Water density at 298.15 K. This assumes pure-solvent water inventory per solution volume and neglects solute partial volume; it is deliberately restricted to dilute water anchors. Unknown temperature, pH/speciation and solid form prevent interpreting this subset as an isolated test of liquid nonideality.

| Variant | MAE log10 | 95% CI | RMSE log10 | Geometric mean fold error | Within 2x | Within 10x |
|---|---:|---|---:|---:|---:|---:|
| UNIFAC | 0.8106 | 0.5615–1.1062 | 1.0362 | 6.465 | 27.3% | 64.7% |
| UNIFAC2 | 0.7210 | 0.4514–1.0573 | 0.9821 | 5.260 | 30.9% | 76.6% |
| UNIFDMD | 1.2805 | 0.9864–1.6534 | 1.4853 | 19.078 | 5.6% | 37.7% |
| UNIFM2 | 1.0037 | 0.7060–1.3483 | 1.2380 | 10.086 | 18.1% | 54.7% |
| UNIFNIST | 1.0821 | 0.7900–1.4147 | 1.2840 | 12.082 | 11.2% | 49.7% |
| UNIFLBY | 0.5736 | 0.4108–0.7445 | 0.7095 | 3.746 | 36.4% | 83.2% |
| Ideal solution | 3.2311 | 2.5927–3.9855 | 3.5621 | 1702.501 | 0.0% | 1.6% |

## Paired uncertainty

Differences are first model MAE minus second model MAE; negative favors the first model.

```json
{
  "common5_UNIFAC2_minus_UNIFAC": {
    "predictions": 17713,
    "solutes": 145,
    "MAE_difference_log10": -0.000345383695493272,
    "95pct_paired_solute_bootstrap_CI": [
      -0.029711952994985573,
      0.02830421687625688
    ]
  },
  "common5_UNIFAC2_minus_Dortmund": {
    "predictions": 17713,
    "solutes": 145,
    "MAE_difference_log10": -0.06596618285929372,
    "95pct_paired_solute_bootstrap_CI": [
      -0.12671080135242865,
      -0.009798558757744847
    ]
  },
  "common5_UNIFAC2_minus_modified2": {
    "predictions": 17713,
    "solutes": 145,
    "MAE_difference_log10": -0.014314291373210173,
    "95pct_paired_solute_bootstrap_CI": [
      -0.05291592761051846,
      0.025167389264991673
    ]
  },
  "common5_UNIFAC2_minus_NIST": {
    "predictions": 17713,
    "solutes": 145,
    "MAE_difference_log10": -0.14087598297773804,
    "95pct_paired_solute_bootstrap_CI": [
      -0.21097604303555917,
      -0.07910608690068445
    ]
  },
  "common5_UNIFAC2_minus_ideal": {
    "predictions": 17713,
    "solutes": 145,
    "MAE_difference_log10": -0.16951242493439286,
    "95pct_paired_solute_bootstrap_CI": [
      -0.25849604837138224,
      -0.09936559929849338
    ]
  },
  "common6_Lyngby_minus_UNIFAC2": {
    "predictions": 7647,
    "solutes": 86,
    "MAE_difference_log10": -0.005367551534053578,
    "95pct_paired_solute_bootstrap_CI": [
      -0.05744357371915681,
      0.04999670934826665
    ]
  }
}
```

## Numerical and reproducibility checks

- Compiled and readable pfdsim activity coefficients agree within rtol=atol=1e-10 for six variants at four nontrivial compositions.
- Identical-component binaries reproduce ideal transfer at x=1e-12, 1e-4, 0.1 and 0.9 for every variant.
- Largest successful prediction equation residual: 1e-10 in ln activity.
- Model/root search never uses the measured target to initialize, select or calibrate predictions.
- Original input CSVs and pfdsim production code are unchanged. Input, code and parameter SHA256 fingerprints are in manifest.json.
- Main model evaluation used five workers; total runtime was 157.6 seconds.

## Interpretation limits

Coverage is a major limit: most drug-like structures are outside the strict fragmenters or lack required interactions. A low error on the common small-molecule cohort does not establish performance on unsupported structures. Model-dependent stability rejection additionally selects the successful comparison set; status counts are therefore part of the assessment. Acids and bases can be neutral in the SMILES but ionize in water; pH and apparent-versus-intrinsic solubility are not recorded here. Original and modified UNIFAC parameter collections may have used some of the underlying literature; this is an external-dataset assessment, not a proven training-disjoint test. A small local melting-point audit found no known liquid solutes in the retained multi-solvent groups, but verified solid identities and forms are not available for all compounds.

## Files

- predictions.csv: every method/case, including failures, predictions, errors and reference stability.
- deduplicated_observations.csv: source-keyed and pooled median observations, duplicate counts and observed spread.
- model_coverage.csv: binary fragmentation and missing-interaction audit.
- metric_records.npy: compact numerical records used by this renderer; no model rerun is needed.
- summary.json, paired_comparisons.json: full numerical results and uncertainty.
- comparison.png and comparison.svg: shareable accuracy/coverage figure.
