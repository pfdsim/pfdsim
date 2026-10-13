# GRAPPA against Perry and CoolProp

Benchmark performed on 2026-10-12 with the unchanged published GRAPPA checkpoint
from upstream commit `68f99d012ff1e489418369448a607d27185ca527`.

Maintained entry point: [benchmark_grappa.py](benchmark_grappa.py).

## Protocol

- Perry Table 2-8: ten uniformly spaced temperatures per continuous correlation.
- CoolProp 7.2.0: ten uniformly spaced temperatures per supported pure-fluid
  HEOS saturation curve; pseudo-pure fluids/mixtures excluded.
- Perry Table 2-10: all eligible tabulated points, with no interpolation.
- Continuous sampling includes endpoints after intersecting the source range
  with 250--600 K, reference pressures 1--1e7 Pa, known melting/triple-point
  lower bounds, and the critical upper bound. Pressure clipping happens before
  temperature sampling. The upper bound is at most `Tc * (1 - 1e-7)`; there is
  no additional reduced-temperature cutoff.
- Perry 2-8 reference values use the existing `PerryPropertyLibrary` evaluator.
  Structures are resolved offline by exact CAS through `chemicals`, checked
  against source molecular formulas when available, and validated with the
  published GRAPPA graph converter. Known ambiguous CAS assignments are excluded.
- One Antoine-parameter prediction per distinct canonical isomeric SMILES.
  Pressures follow `ln(P/kPa) = A - B/(T/K + C)` directly, without rerunning the
  network at each temperature. No coefficients are fitted to, or anchored on,
  reference pressures or boiling temperatures.
- CPU inference with one PyTorch thread and `torch.inference_mode()`. No random
  selection is used.

## Results

Compound error is the mean absolute percentage error over that compound's
points, using reference pressure in the denominator. Each compound has equal
weight in the compound-level statistics.

| Source | Compounds | Points | Median compound error | Mean compound error | Compound-error p95 |
|---|---:|---:|---:|---:|---:|
| Perry 2-8 | 305 | 3,050 | 6.09% | 9.60% | 20.33% |
| Perry 2-10 | 838 | 7,538 | 9.46% | 18.29% | 54.62% |
| CoolProp HEOS | 84 | 840 | 3.64% | 6.62% | 15.27% |

| Source | Median point error | Mean point error | Point-error p95 | Points within 20% | MAE in ln(P) | RMSE in ln(P) |
|---|---:|---:|---:|---:|---:|---:|
| Perry 2-8 | 4.89% | 9.60% | 29.90% | 90.72% | 0.0906 | 0.1746 |
| Perry 2-10 | 8.78% | 18.73% | 59.18% | 76.28% | 0.1836 | 0.3839 |
| CoolProp HEOS | 3.40% | 6.62% | 16.49% | 96.07% | 0.0608 | 0.1308 |

There were **zero numerical prediction failures across 11,428 evaluated points**.
The three reference populations cover **982 distinct CAS numbers and 982 distinct
canonical SMILES**; their compound counts must not be added as independent cases.
Pairwise overlaps are 196 compounds for the two Perry tables, 43 for Perry 2-8
and CoolProp, and 36 for Perry 2-10 and CoolProp.

## Coverage and exclusions

- Perry 2-8: 305 of 345 compounds evaluated. Exclusions comprised 25 without
  carbon, six with unsupported silicon, five rejected for formal charges, one
  missing from the offline identity database, methane (four attached hydrogens
  are outside the published encoder's categories), and two without temperature
  overlap. A known melting bound was unavailable for 122 included compounds.
- Perry 2-10: 838 of 921 CAS-resolved compounds evaluated. Excluded compounds
  comprised 56 model-input/identity rejections, three source-recorded identity
  collisions, and 24 with no eligible temperature/phase points. Among compounds
  passing identity/chemistry checks, 875 points were outside 250--600 K and 141
  more were below known melting temperatures. The source's 167 unresolved rows
  were not assigned guessed identities. A known melting bound was unavailable
  for 290 included compounds.
- CoolProp: 84 of 124 listed fluids evaluated. Six pseudo-pure/mixture entries,
  28 chemistry rejections, five identity-resolution rejections, and one fluid
  without temperature overlap were excluded. All evaluated fluids had
  provider-declared triple-point bounds.

The exact excluded records and reasons are saved in `excluded.csv`.

## Pressure dependence and outliers

At reference pressures below 1 kPa, mean point errors were 15.95% for Perry
2-8 (371 points), 33.69% for Perry 2-10 (1,284 points), and 30.61% for CoolProp
(48 points). For pressures from 1 to below 100 kPa, the corresponding means
were 10.29%, 16.36%, and 8.70%. The pressure populations differ, so these are
within-source diagnostics rather than controlled comparisons between sources.

The worst compound in each source, by mean absolute percentage error, was:

| Source | Compound | Mean error |
|---|---|---:|
| Perry 2-8 | Oxalic acid | 385.75% |
| Perry 2-10 | 2,3,4-Pentanetriol | 965.95% |
| CoolProp | Methyl linolenate | 151.64% |

Reference disagreement can also inflate apparent model error. Octanal's mean
error against Perry 2-10 was 457.99%, but its low-pressure table points disagree
with the other Perry reference. At **346.55 K**, the saved Table 2-10 value is
**0.133322 kPa**, the Table 2-8 correlation gives **2.950757 kPa**, and GRAPPA
gives **2.792701 kPa**. At **441.65 K**, the values are **101.325**, **92.333212**,
and **87.193029 kPa**, respectively. This establishes disagreement between the
stored sources; it does not establish which source is wrong. The points remain
in the reported statistics, and neither dataset was edited.

## Timing and verification

Reference collection took **5.24 s**. Model loading plus prediction of all 982
sets of Antoine parameters took **1.24 s**. The complete benchmark took
**7.83 s excluding imports**, on one CPU thread. These are single-run timings,
not repeated performance measurements.

Ruff and Python compilation passed. A separate verification checked all 11,428
points for temperature/pressure/known-phase bounds and recomputed their
percentage errors, checked every continuous grid for ten equally spaced
temperatures, independently recomputed the headline compound medians, and
compared 18 representative points with `GRAPPAdirect` at relative tolerance
`2e-5`. All checks passed. Dependency compatibility checks also passed.

## Artifacts and reproduction

The temporary environment is `/tmp/grappa-rlpws2CN/venv`. It uses Python 3.11.16,
PyTorch 2.2.2+cpu, PyTorch Geometric 2.3.1, RDKit 2023.9.6, NumPy 1.26.4,
chemicals 1.5.2, and CoolProp 7.2.0.

Run from the repository root with a new output directory:

```bash
set -o pipefail
/tmp/grappa-rlpws2CN/venv/bin/python -u \
  scripts/psat_experiments/benchmark_grappa.py \
  --grappa-root /tmp/grappa-rlpws2CN/GRAPPA \
  --output-dir /tmp/grappa-repeat-results \
  2>&1 | tee /tmp/grappa-repeat.log
```

Original artifacts are in `/tmp/grappa-reference-benchmark-Q2LXFd6o/`:

- `results/summary.json`: complete metrics, pressure/Tr strata, coverage,
  dependency versions, source/model/script hashes, and upstream model commit.
- `results/points.csv`: every evaluated reference/model pressure and its error.
- `results/compounds.csv`: per-compound errors and sampled ranges.
- `results/antoine_parameters.csv`: the 982 predicted parameter sets.
- `results/excluded.csv`: excluded source records and points.
- `results/report.md`: generated numerical report and worst-compound lists.
- `requirements.lock.txt`, `run.log`, and `verification.log`.

## Interpretation

GRAPPA agrees well with the typical included reference compound, particularly
the CoolProp population, but has a substantial tail of large errors. Some
outliers require source review before being attributed entirely to the model.
These are comparisons with handbook points, fitted correlations, and a
reference EOS, rather than an established independent experimental test set.
The released GRAPPA checkpoint was retrained on all its curated DDB data;
training-compound overlap with our references is unknown. Consequently, these
results assess practical agreement but do not settle the overfitting question.
