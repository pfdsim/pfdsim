# Important solubility MAE results

For the current **Nannoolal-focused headline report**, including R², RMSE,
correlation plots, coverage and invalid predictions, see
[METRICS_SUMMARY.md](METRICS_SUMMARY.md). This document retains the broader
MAE detail and the solvent-transfer findings.

Summary of the completed investigations, 2026-10-23. Values below are held-out
mean absolute errors in base-10 logarithms. An MAE of 1 means a geometric mean
absolute error factor of 10; 0.835 means about 6.84. Sample sizes and matched
populations matter: lower errors on different populations do not establish
that one model is better.

## Main findings

- For absolute aqueous solubility, chemical group counts outperform size alone
  and element counts. Nannoolal offers competitive accuracy with much broader
  structural coverage than the UNIFAC encodings.
- Study-only expanded Nannoolal increases descriptive coverage from 7,278 to
  8,228 of 9,044 neutral organic molecules, preserving every complete native
  assignment. Its adaptive regression has MAE **0.835 on 8,128 valid molecules**.
  Matched comparisons show essentially unchanged accuracy on the old domain
  and a clear advantage over element counts on the expanded domain.
- For transfer between solvents, original UNIFAC and UNIFAC 2.0 are effectively
  tied on shared cases. UNIFAC 2.0 covers more compounds. Neither solvent-bias
  correction nor finite-concentration solving produces a convincing broad
  accuracy advantage over the stronger empirical comparisons described below.

## Absolute aqueous log(S): size and atoms

Target: log10(S / (mol/L)). One label per chemical identity; AqSolDB's curated
value takes priority on overlaps, with BigSolDB-only water measurements at
20–30 °C added. The molecular domain contains 9,044 neutral, connected,
carbon-containing structures. Five-fold molecule CV is repeated five times;
size-form and ridge choices use three-fold inner training validation.

| Model | Molecules | MAE |
|---|---:|---:|
| Training-set mean | 9,044 | 1.818 |
| Linear heavy-atom count | 9,044 | 1.470 |
| Linear total atom count, including H | 9,044 | 1.562 |
| Training-selected nonlinear size relationship | 9,044 | 1.416 |
| Element counts, OLS | 9,044 | 1.093 |
| Element counts plus selected size/ridge | 9,044 | **1.024** |

Size alone is weak. Element identity adds much more information. Nonlinear
size terms help, but the empirical curves are not universal physical size laws.
Regularization and size augmentation are selected jointly, so adaptive-model
improvements do not isolate the size term's contribution.

The unfiltered 10,421-compound population also includes salts, inorganic
species and disconnected mixtures: element OLS MAE is 1.303 and adaptive
element MAE is 1.234. An extreme 24-fragment, 388-heavy-atom record causes
severe extrapolation failures, including physically impossible predictions.
Those original results remain preserved separately from the molecular-domain
check. See the [count report](results/aqueous-count-regression/REPORT.md) and
[domain check](results/aqueous-count-domain-check/REPORT.md).

## Absolute aqueous log(S): matched group comparisons

The following regressions use identical 1,881 molecules with valid predictions
from every compared model in every repeat. Adaptive means training-selected
size augmentation and ridge strength. Group coefficients are learned from
aqueous data; no activity coefficients or pure-property estimates are inputs.

| Features | OLS MAE | Adaptive MAE |
|---|---:|---:|
| Element counts | 0.986 | 0.869 |
| Original / UNIFAC 2.0 subgroups | 0.930 | 0.797 |
| Dortmund / modified UNIFAC 2.0 subgroups | 0.924 | 0.773 |
| NIST subgroups | **0.890** | **0.759** |
| Lyngby subgroups | 0.937 | 0.807 |
| Nannoolal first-order counts | 0.907 | **0.762** |
| Nannoolal plus structural corrections | 0.911 | 0.766 |

Nannoolal and NIST are effectively tied here. On their larger pairwise
intersections, Nannoolal first-order adaptive MAEs are 0.764 versus original
UNIFAC's 0.781 (4,114 molecules), 0.747 versus Dortmund's 0.769 (4,195),
0.805 versus NIST's 0.793 (4,578), and 0.777 versus Lyngby's 0.805 (2,007).
The gains over original, Dortmund and Lyngby survive correction across 50
comparisons; the original-UNIFAC result is borderline. The NIST difference is
not resolved. Native Nannoolal represents 7,278 molecules, versus 4,122,
4,204, 4,602 and 2,012 for the four UNIFAC encodings respectively.
See the [Nannoolal report](results/nannoolal-aqueous-regression/REPORT.md).

## Expanded Nannoolal: coverage gain without sacrificing old-domain accuracy

These are study-only repairs and training-supported new environments; the
production Nannoolal implementation and parameter tables are unchanged.
Each new group requires 20 distinct fully representable training molecules.
Vocabulary discovery/backoff happens inside every inner and outer training
fold. Complete native feature assignments are preserved exactly.

| Model | Valid molecules in every repeat | MAE |
|---|---:|---:|
| Native first-order adaptive Nannoolal | 7,274 | 0.832 |
| Study-repaired adaptive Nannoolal | 7,745 | 0.826 |
| Study-expanded OLS | 8,130 | 0.903 |
| **Study-expanded adaptive Nannoolal** | **8,128** | **0.835** |

These own-coverage rows are not a controlled accuracy ranking. The important
matched comparisons are:

- **Expanded versus native:** 0.83135 versus 0.83108 on 7,265 molecules;
  difference +0.00027, familywise 95% interval [-0.00331, +0.00388]. Accuracy is
  essentially unchanged while coverage grows.
- **Expanded versus adaptive elements:** 0.83504 versus 1.02034 on 8,127
  molecules; difference -0.18530, interval [-0.20418, -0.16679], an 18.2% MAE
  reduction. Intervals account for nine comparisons.

Full-data descriptive coverage is 8,228/9,044 (91.0%); 8,133 are represented in
every held-out repeat. The adaptive model has 21 invalid supported predictions,
affecting five molecules. Invalid means nonfinite or outside [-30, 3] log10
mol/L, a declared numerical screen rather than a universal physical bound.
Predictions are not clipped. Conditional scores exclude invalid molecules on
matched cases and always retain failure counts and raw outputs. See the
[expanded study report](results/expanded-nannoolal-regression/REPORT.md).

## Solvent-transfer results: a different task and target

These predict log mole-fraction solubility in a target solvent GIVEN a measured
reference-solvent solubility. They are not directly comparable with absolute
aqueous log mol/L regressions. Main measurements match molecule, exact
temperature and publication within 20–30 °C. Transfers and temperatures are
averaged within each molecule before scoring molecules equally.

| Finite-concentration activity variant | MAE on identical 7,647 transfers / 86 molecules |
|---|---:|
| Original UNIFAC | 0.379 |
| UNIFAC 2.0 | 0.375 |
| Dortmund | 0.397 |
| Modified UNIFAC 2.0 | 0.382 |
| NIST | 0.469 |
| Lyngby | 0.370 |
| Ideal-solution baseline | 0.474 |

Original and UNIFAC 2.0 are effectively tied; Lyngby's small apparent advantage
is also unresolved. Water transfers are harder than non-water transfers.
See the [transfer report](results/unifac-transfer-results/REPORT.md).

The stronger empirical baseline learns a constant log multiplier per solvent
with five-fold molecule CV repeated ten times. UNIFAC 2.0 scores **0.598 versus
0.627** for the baseline on 75,208 matched transfers / 346 molecules, but the
corrected interval includes zero. No variant convincingly beats this baseline
across its own full usable coverage. On the narrower five-variant shared
population, UNIFAC 2.0 does win: **0.366 versus 0.541**, on 17,135 transfers /
145 molecules. See the [multiplier report](results/solvent-multiplier-results/REPORT.md).

Infinite dilution generally costs little accuracy on dilute subsets. For
UNIFAC 2.0, with BOTH observed mole fractions <0.05, finite/infinite MAEs are
**0.651 / 0.663** (43,758 transfers / 287 molecules); at <0.01 they are
**0.831 / 0.823** (21,129 / 203). Different cutoffs select different chemical
populations. NIST has a modest penalty at 0.05. See the
[dilution report](results/infinite-dilution-assessment/REPORT.md).

Held-out solvent-bias corrections remove much signed bias but do not establish
a convincing broad MAE win over a solvent-only baseline fitted to identical
training cases. For finite UNIFAC 2.0, uncorrected/relative-corrected/target-only
corrected MAEs are **0.600 / 0.606 / 0.596**, versus **0.630** for that baseline;
both corrected-versus-baseline intervals include zero. See the
[calibration report](results/bias-corrected-transfer-results/REPORT.md).

## Solvent multipliers plus computed logP

The additional transfer model fits an offset and logP slope per solvent:
`log10(x_target/x_reference) = a_target − a_reference + (b_target − b_reference) cLogP`.
A shared global logP term would cancel. RDKit Wildman–Crippen logP is computed
from the standard InChI identity; no measured partition coefficient is used.

Following PFDSim's requested sparse-data fallback, each solvent slope is
**exactly zero below 20 distinct training molecules**, while its multiplier is
still fitted. Eligibility is recomputed within each of the original five
folds and ten repeats. Repeated measurements cannot inflate support. The
threshold was declared, not optimized on held-out errors. There are 41–44
freely fitted solvent slopes per fold; the water/connected-component reference
slope is fixed to zero.

On **213,610 identical transfers across 1,074 molecules**, matching exact
temperature and publication:

| Model | MAE | RMSE | Predictive R² | Pearson r | Within ×10 |
|---|---:|---:|---:|---:|---:|
| Solvent multiplier only | 0.658 | 0.971 | 0.515 | 0.758 | 79.1% |
| **Multiplier + supported logP slope** | **0.600** | **0.874** | **0.607** | **0.805** | **81.1%** |

The MAE reduction is **8.7%**; new minus old MAE is **−0.0575**, simultaneous
95% interval **[−0.0794, −0.0375]**. RMSE is descriptively 10.0% lower. All
headline metrics weight molecules equally. The intervals cover seven primary
comparisons and condition on the saved fitted CV predictions.

The larger improvements are water-related: water → non-water MAE falls from
**1.295 to 1.020**, and non-water → water from **1.335 to 1.041** (7,697
transfers / 467 molecules in each direction). Non-water → non-water changes
only from **0.531 to 0.519** (198,216 transfers / 1,024 molecules). These are
different directional populations; their scores are descriptive.

| Variant | UNIFAC MAE | Multiplier-only MAE | Multiplier + logP MAE |
|---|---:|---:|---:|
| Original UNIFAC | 0.544 | 0.533 | 0.490 |
| UNIFAC 2.0 | 0.598 | 0.627 | 0.564 |
| Dortmund | 0.532 | 0.525 | 0.479 |
| Modified UNIFAC 2.0 | 0.625 | 0.625 | 0.567 |
| NIST | 0.576 | 0.553 | 0.506 |
| Lyngby | 0.431 | 0.507 | 0.451 |

Each row compares identical cases, but different rows cover different molecules.
Despite these point differences, **none of the UNIFAC-versus-supported-logP
MAE intervals excludes zero after the seven-comparison correction**. In
particular, the better UNIFAC point estimates on narrower shared populations
do not establish superiority across its broader usable coverage.

The multiplier baseline was refitted on the identical training groups and
verified against the saved original predictions. Both models retain the
original physical `x ≤ 1` cap. The new model's uncapped main MAE is **0.608**;
no arbitrary lower-solubility invalid screen is applied. There are no supported
predictions below log10(x) = −30, and no descriptor failures.

The unrestricted diagnostic fit initially gave MAE 0.578 but RMSE 2.536 on
its narrower 212,156-case population. One solvent had only two training
molecules separated by 0.0042 logP units, allowing a huge extrapolation slope.
Those numerical results are preserved separately; they motivated the
requested zero-slope fallback. The support rule removes that exploding tail,
rather than excluding its predictions from the final score.

See the [supported-logP report](results/logp-transfer-supported-results/REPORT.md)
and [full metrics/uncertainty](results/logp-transfer-supported-results/summary.json).
The [unrestricted diagnostic report](results/logp-transfer-results/REPORT.md)
retains its exact source snapshot and original numerical outputs.

## Interpretation and complete inventory

Absolute-solubility labels include solid-state and aqueous ionization effects;
pH and polymorph/solvate identity are not consistently available. Same-source
transfer pairing improves comparability but cannot guarantee the same solid
form/species. Random molecule CV is not scaffold-disjoint or independently
collected external validation. Historical UNIFAC parameter-training overlap
with underlying publications is unverified. Conditional valid-case accuracy
must be considered together with structural coverage and prediction failures.

The [complete MAE inventory](results/mae-inventory-20261023/all_saved_MAEs.csv)
contains 3,169 scalar MAE statistics from completed summary files, excluding
preliminary runs and difference/interval estimates. Assessment names and JSON
paths identify each cohort and metric. Rows are not all mutually comparable.
No experiments were rerun to produce this summary.
