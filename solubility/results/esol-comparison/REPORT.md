# Frozen ESOL versus expanded Nannoolal

Target: absolute aqueous log10(S / (mol/L)). The requested population is exactly the 8,128 molecules valid for expanded Nannoolal in all five saved CV repeats. ESOL coefficients are frozen; no local fitting, bias correction, or model selection is performed.

## Matched metrics

| Population | Model | Molecules | MAE | RMSE | Predictive R² | Pearson r | Bias | Within ×10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| same_expanded_cohort | ESOL, fixed RDKit coefficients | 8,128 | 1.0038 | 1.4849 | 0.5892 | 0.8092 | -0.3178 | 63.9% |
| same_expanded_cohort | Original ESOL coefficients + RDKit descriptors | 8,128 | 0.9724 | 1.3094 | 0.6805 | 0.8263 | +0.0899 | 62.6% |
| same_expanded_cohort | Expanded Nannoolal, held-out | 8,128 | 0.8352 | 1.1235 | 0.7648 | 0.8745 | -0.0029 | 69.2% |
| excluding_public_training_structures | ESOL, fixed RDKit coefficients | 6,992 | 1.0464 | 1.5526 | 0.5640 | 0.8033 | -0.3696 | 62.3% |
| excluding_public_training_structures | Original ESOL coefficients + RDKit descriptors | 6,992 | 0.9975 | 1.3452 | 0.6727 | 0.8216 | +0.0581 | 61.6% |
| excluding_public_training_structures | Expanded Nannoolal, held-out | 6,992 | 0.8527 | 1.1475 | 0.7618 | 0.8730 | -0.0443 | 68.4% |

## Paired MAE differences: Nannoolal minus ESOL

| Population | ESOL variant | Difference | Paired 95% interval | Familywise 95% interval |
|---|---|---:|---|---|
| same_expanded_cohort | ESOL_RDKit | -0.1685 | [-0.1869444477517673, -0.15078116990876186] | [-0.19247369255850597, -0.14542346425658884] |
| same_expanded_cohort | ESOL_original_RDKit | -0.1371 | [-0.14963617364251017, -0.12478409076047385] | [-0.1532453855562229, -0.12156669641567207] |
| excluding_public_training_structures | ESOL_RDKit | -0.1937 | [-0.2145839088455193, -0.17330277835756452] | [-0.22049812910339434, -0.16748592930004078] |
| excluding_public_training_structures | ESOL_original_RDKit | -0.1448 | [-0.15890418111622043, -0.13111251507026953] | [-0.1627257858321778, -0.12769460859663279] |

## Descriptor and coefficient definitions

- Main model: Pat Walters’s fixed RDKit ESOL coefficients, applied using the pinned upstream class. logS = 0.26121066137801696 − 0.0066138847738667125 MW − 0.7416739523408995 MolLogP + 0.003451545565957996 rotors − 0.42624840441316975 aromatic proportion.
- Sensitivity only: original Delaney coefficients with the same RDKit descriptors: logS = 0.16 − 0.63 MolLogP − 0.0062 MW + 0.066 rotors − 0.74 aromatic proportion. This is not an exact recreation of the original Daylight-descriptor model; the upstream implementation explicitly discourages treating it as such.
- RDKit average molecular weight (MolWt, not exact mass), Wildman–Crippen MolLogP, Lipinski.NumRotatableBonds, and the upstream aromatic SMARTS match count divided by Mol.GetNumAtoms(). Normal SMILES parsing uses implicit hydrogen; any retained explicit/isotopic atoms are handled exactly as in the reference. No AddHs, neutralization, tautomer standardization, or largest-fragment substitution is applied.
- Nannoolal is the saved support-aware expanded model, with five-fold molecular CV repeated five times and training-only vocabulary/size/ridge selection. ESOL is one deterministic prediction per molecule; its prediction is repeated only to align losses with the five Nannoolal CV repeats, not to create independent observations.
- All score rows use identical molecules across both ESOL equations and Nannoolal. Nonfinite predictions or predictions outside [-30, 3] are invalid, never clipped; raw predictions and failure reasons are retained. Pearson correlation and predictive R² are distinct; scores use individual held-out predictions, not a CV ensemble.

## Calibration overlap and limitations

```json
{
  "requested_molecules": 8128,
  "descriptor_failures": 0,
  "invalid_ESOL_predictions": {
    "ESOL_RDKit": 0,
    "ESOL_original_RDKit": 0
  },
  "matched_valid_molecules": 8128,
  "public_calibration_table": {
    "rows": 1144,
    "invalid_identities": 0,
    "unique_exact": 1117,
    "unique_without_stereo": 1115
  },
  "exact_public_training_overlap": 1038,
  "public_training_overlap_without_stereo": 1136,
  "remaining_after_overlap_exclusion": 6992
}
```

The public delaney.csv accompanies the upstream coefficient-refit routine. Its intersections are reported by full standard InChI and, conservatively, after removing stereochemistry. The exclusion sensitivity removes the latter union; it does not prove external validation or exclude overlap with Delaney’s full original 2,874-molecule training set, which includes proprietary structures. Nannoolal CV training populations are unchanged; the exclusion only selects test cases. Unknown pH, solution species, solid form, and scaffold generalization remain limitations.
The paired bootstrap resamples molecules and Nannoolal repeat indices (20,000 draws, seed 20261023), conditioning on saved predictions. Familywise intervals cover the two frozen equations on both populations (four comparisons). They do not include uncertainty from refitting either model.

## Plots and provenance

![Identical-molecule prediction comparison](correlation.png)

All five Nannoolal predictions are displayed individually; ESOL’s deterministic predictions are replicated for identical hexagon density weighting. Axes, bins, and color scale are shared; no valid points are cropped.

- predictions.csv: every requested molecule, descriptors, both ESOL predictions, all five Nannoolal predictions, validity and public-calibration overlap.
- summary.json and headline_metrics.csv: full-precision metrics, counts and uncertainty.
- manifest.json: portable input/output fingerprints and RDKit version.
- Reference implementation, calibration table and MIT notice: ../../provenance/esol-reference/. No production code or existing predictions changed.

[Original paper](https://doi.org/10.1021/ci034243x); [pinned RDKit implementation](https://github.com/PatWalters/solubility/blob/9e0e55fd7040575e8427079c44759d51dc7c2216/esol.py).
