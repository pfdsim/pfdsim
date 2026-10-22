# Study-only expanded Nannoolal features

The production nannoolal_method.py, group tables, and previous studies are unchanged. These are new aqueous-solubility descriptors and regressions, not extensions to the published pure-property equations.

## Structural conflict resolution

- Preserve every complete native assignment. Apply experimental repairs only when native fragmentation fails, so a supported native molecule cannot lose descriptive coverage merely by acquiring a rare composite.
- Reuse the native fragmenter function code in a separate globals dictionary; prioritize specific sulfonamide group 105 ahead of the broad sulfone group 82. The original module registry is never modified.
- Find connected overlapping native amide/urea/carbamate assignments involving at least two carbonyl centers. Replace the entire overlap component and any intersecting assigned units by one disjoint amide-system descriptor. Its detailed descriptor includes composition, carbonyl count, N hydrogens, aromaticity, ring membership and boundary substituents; the backoff drops boundary detail.
- Preserve all remaining native assigned units, including silicon environments. Every atom is assigned exactly once; no partial amide group is double-counted.
- Enumerate every remaining unassigned atom environment, including central element/charge/H/aromatic/ring/radical state and bonded neighbor elements/aromaticity/charge/bond order. Detailed environments add neighbor hydrogen counts/degrees; the chemically consistent core environment omits that detail.
- Native one-heavy-atom refusal remains unavailable. Structural groups too rare to regress remain unavailable and are explicitly audited.

## Support and validation

- New groups require at least 20 distinct fully representable TRAINING molecules. Counts of atoms, occurrences, temperatures or CV repeats cannot inflate this support.
- Determine feature eligibility and detailed-to-core backoff inside every outer AND inner training split. Iteratively remove rare selected detail groups before testing backoff, recompute actual selected-group support, and remove molecules with another unsupported unit until the vocabulary stabilizes.
- Existing native descriptors are retained; the new-group support threshold does not retroactively delete published native groups. Native groups absent from training are unavailable at test time.
- Reuse the previous 9,044 neutral connected organic identities, labels, five molecular folds and five repeats. Choose ridge strength and optional nonlinear size terms on three-fold inner validation only. Reuse the shared SVD/scaling implementation.
- Compare repaired-only assignments (no residual atom additions) and fully expanded assignments, each with OLS and nested size/ridge selection.
- Predictions outside the declared [-30,3] log10 mol/L interval or nonfinite predictions are invalid, never clipped. Unavailable structures and invalid numerical predictions are reported separately. Metrics require validity in every repeat; paired tests use identical molecules.
- Element baselines and the original Nannoolal predictor reuse their previous held-out predictions. Expanded models train on all represented training molecules in the larger domain; the native model had a smaller training domain. This measures the combined practical effect of the representation and expanded training data.
- Bootstrap molecules and split repetitions 20,000 times; familywise intervals cover the 9 reported comparisons. These tests condition on the fitted CV predictions and do not establish independently collected external or unseen-scaffold accuracy.

## Full-data vocabulary audit (descriptive, not CV eligibility)

```json
{
  "total_molecules": 9044,
  "native_available": 7278,
  "full_data_expanded_coverage": 8228,
  "recovered_native_failures": 950,
  "lost_native_coverage": 0,
  "candidate_groups": 775,
  "supported_new_groups": 23,
  "supported_group_types": {
    "residual_detail": 16,
    "amide_system_detail": 4,
    "amide_system_core": 1,
    "residual_core": 2
  },
  "minimum_selected_group_support": 20
}
```

## Held-out results

| Model | Supported in every repeat | Valid in every repeat | Invalid supported predictions | MAE log10 | RMSE log10 | R² |
|---|---:|---:|---:|---:|---:|---:|
| repaired_OLS | 7749 | 7746 | 12 | 0.8845 | 1.2077 | 0.7217 |
| repaired_adaptive | 7749 | 7745 | 16 | 0.8264 | 1.1172 | 0.7618 |
| expanded_OLS | 8133 | 8130 | 13 | 0.9034 | 1.2333 | 0.7166 |
| expanded_adaptive | 8133 | 8128 | 21 | 0.8352 | 1.1235 | 0.7648 |
| element_adaptive | 9044 | 9041 | 14 | 1.0230 | 1.3407 | 0.6598 |
| element_OLS | 9044 | 9044 | 0 | 1.0933 | 1.4387 | 0.6082 |
| native_adaptive | 7278 | 7274 | 17 | 0.8317 | 1.1235 | 0.7677 |

Own-coverage rows have different populations. Use the paired comparisons below for accuracy conclusions.

## Paired comparisons

| First | Second | Molecules | First MAE | Second MAE | Difference | Familywise 95% CI |
|---|---|---:|---:|---:|---:|---|
| expanded_OLS | native_adaptive | 7266 | 0.8995 | 0.8318 | +0.0677 | +0.0555 to +0.0800 |
| expanded_adaptive | element_adaptive | 8127 | 0.8350 | 1.0203 | -0.1853 | -0.2042 to -0.1668 |
| expanded_OLS | element_OLS | 8130 | 0.9034 | 1.0909 | -0.1876 | -0.2069 to -0.1681 |
| expanded_adaptive | element_OLS | 8128 | 0.8352 | 1.0889 | -0.2537 | -0.2746 to -0.2332 |
| expanded_OLS | element_adaptive | 8129 | 0.9033 | 1.0204 | -0.1171 | -0.1398 to -0.0942 |
| expanded_adaptive | native_adaptive | 7265 | 0.8314 | 0.8311 | +0.0003 | -0.0033 to +0.0039 |
| repaired_adaptive | native_adaptive | 7266 | 0.8317 | 0.8318 | -0.0001 | -0.0030 to +0.0033 |
| expanded_adaptive | repaired_adaptive | 7743 | 0.8254 | 0.8255 | -0.0001 | -0.0026 to +0.0023 |
| expanded_adaptive | expanded_OLS | 8128 | 0.8352 | 0.9019 | -0.0667 | -0.0787 to -0.0551 |

## Remaining limitations

Twenty molecules is a declared minimum support criterion, not a guarantee of independently identifiable coefficients or broad chemical diversity. Overlapping feature counts can remain correlated; ridge/validation controls prediction stability. Unknown pH, solution species and solid forms remain part of the target uncertainty. Descriptor units are experimental study features and must not be passed to the production Nannoolal property tables.

## Artifacts

- candidate_features.json: disjoint native units and every candidate detailed/core extension per molecule.
- group_catalogue.csv: every candidate, total and selected-molecule support and acceptance status.
- coverage.csv: full-data descriptive coverage and unsatisfied candidate groups.
- *_groups.csv: every inner-independent outer-training vocabulary and its selected molecule support.
- folds.json: model selections, inner support audits and train/test identity indices.
- predictions.npz: all held-out predictions, including unavailable and invalid cases.
- summary.json, REPORT.md, manifest.json: results and provenance.
