# HANNA versus UNIFAC: raw binary/ternary VLE and agreeing fits

Investigation completed 2026-10-13. The earlier GRAPPA boiling-point
investigation was committed separately as `e64e63b`.

## Model and scope

This comparison uses the current
[multicomponent HANNA](https://github.com/marco-hoffmann/HANNA), commit
`6fe873ca1a92c306eb9b5be3e9adb2ebbbb95365`, rather than the original
2024 binary-only prototype. The authors' updated
[paper](https://arxiv.org/html/2509.06484v2) describes training on 824,481
binary-mixture observations across VLE, LLE, infinite-dilution activities and
excess enthalpies. The released model was trained on all available data,
so overlap with our papers and fitted-reference data is unknown.

The published ten-model ensemble predicts gE/RT and derives ln(gamma) by
automatic differentiation. Inputs are molecular SMILES, temperature and
composition. There are **651,900 parameters across the ten thermodynamic
networks**, plus **3,427,440 in the shared pretrained molecular encoder**.
The latter's embeddings are calculated once per distinct SMILES, using the
official tokenizer, canonicalization, scaling and embedding function.
No model parameters or liquid interaction coefficients are refitted here.

Experimental and fitted-reference results are deliberately separate. Each
binary pair has equal weight within a cohort; each ternary system has equal
weight within the ternary cohort. Both/all three component ln(gamma) errors
are weighted equally where the observations identify finite values. RMS is
the square root of the mean per-system MSE. Different cohorts overlap in
chemical systems and their counts must not be added as independent samples.

## Raw binary grand total

The exact saved UNIFAC liquid predictions and measured states are reused.
All six models share **43 pairs / 1,805 states** for bubble-equation tests.
**26 pairs / 577 measured-vapor states / 1,154 component values** identify
ln(gamma). Their saved inferred activities, including HOC association
corrections for acids, are reused exactly; reference change is **zero**.

| Model | Mean pair ln(gamma) MAE | Median pair MAE | Pair-weighted ln(gamma) RMS | Mean pair bubble-equation absolute residual / % |
|---|---:|---:|---:|---:|
| UNIFAC | **0.26525** | **0.04897** | 0.97713 | 8.3758 |
| UNIFAC2 | 0.26671 | 0.05533 | 0.97097 | 7.9951 |
| Dortmund | 0.27577 | 0.06462 | 0.93937 | 10.3713 |
| Modified UNIFAC 2.0 | 0.27340 | 0.07847 | 0.93252 | 9.0170 |
| NIST | 0.28303 | 0.08362 | 0.91909 | 8.1516 |
| HANNA | 0.27241 | 0.07773 | **0.91725** | **6.5371** |

HANNA has the lowest RMS, but only slightly below NIST, and does not have
the lowest mean or median ln(gamma) error. Alcohol/glycerol measurements
continue to dominate the RMS tail, as in the earlier UNIFAC investigation.
The pressure-equation residual is smaller with HANNA; its interpretation
must remain distinct from actual pressure/temperature prediction error.

### Definition of the bubble-equation residual

At measured T/P/x, compute

```text
r = sum_i[x_i gamma_i phi_i_sat Psat_i Poynting_i / (phi_i_vapor P)] - 1
```

Where vapor composition is measured, phi_vapor is evaluated at that measured
composition. Otherwise a bounded, damped fixed-P composition iteration is
used; no pressure or temperature root is solved. For measured-y binary
states, the same residual is exactly
`sum_i[y_i exp(ln_gamma_predicted_i - ln_gamma_inferred_i)] - 1`.

Pure Psat, saturation fugacity, Poynting and vapor correction paths are shared
across all methods. Nonacids use Tsonopoulos; acids retain their previously
computed association-aware HOC references. The residual is a useful test of
the VLE equation, **not a solved pressure AARD or temperature RMS**. The
previous report's forward-pressure numbers therefore must not be placed in
this residual column. No slow nested equilibrium solvers were introduced.

## Closely agreeing runtime NRTL/UNIQUAC references

Selection is the same saved diagnostic as the earlier investigation:
both fitted models have overlapping declared temperature ranges and their
mean absolute ln(gamma) discrepancy is <=0.1 on the shared grid. There are
83 eligible pairs; **82** are fully supported by all five UNIFAC variants.
This gives **3,570 unique T/x states**, scored against both references:
7,140 reference-state cases and 14,280 component errors per method.

The reference-to-reference mean absolute discrepancy on this common cohort
is **0.01971**. Each pair receives one total weight, with its NRTL and
UNIQUAC errors averaged within that pair. These are fitted-model agreement
scores, not additional independent experimental measurements.

| Model | Mean pair MAE | Median pair MAE | Pair-weighted RMS |
|---|---:|---:|---:|
| UNIFAC | 0.12006 | 0.07483 | 0.27001 |
| UNIFAC2 | 0.11277 | 0.07166 | 0.25238 |
| Dortmund | 0.09308 | 0.05675 | 0.25066 |
| Modified UNIFAC 2.0 | 0.10033 | 0.05898 | 0.24815 |
| NIST | 0.08629 | 0.06179 | 0.19956 |
| HANNA | **0.08602** | **0.05300** | **0.19811** |

HANNA is effectively tied with NIST on mean and RMS error. Its lower median
does not establish superiority over NIST across the population.

## Measured ternary VLE/VLLE

[`collect_hanna_ternary_vle.py`](collect_hanna_ternary_vle.py) extracted
**152 observed liquid states from five ternary systems**, 116 with measured
vapor composition. The retained systems are:

- Diisopropyl ether / isopropanol / 2-methoxyethanol: 42 states.
- Ethanol / water / ethylene glycol: 33 states.
- Ethanol / water / toluene: 41 liquid states, including both observed liquid
  phases in VLLE, without pretending they are independently solved phases.
- Methanol / water / glycerol: 18 states.
- Ethanol / water / glycerol: 18 states.

Ternary mass and mole fractions are reconstructed from linear composition
constraints; complementary fractions close each phase to one. Mixed bases
are supported. One genuine source metadata defect was corrected against the
original paper: ThermoML labels Lladosa 2007 Table 11's x2/y2 columns as mass
fractions, whereas the paper reports mole fractions. The first printed row
maps to x=(0.684, 0.059, 0.257), y=(0.750, 0.023, 0.227) in our CAS order.
The PDF hash and correction are recorded. No values were changed to improve
a model's errors. Two binary-edge VLLE liquid states and one salt-containing
dataset with 125 observations are excluded explicitly. The salt dataset
requires electrolyte/solvent-relative treatment outside the shared molecular
Raoult-law reference calculation.

Classic UNIFAC lacks a required group interaction for the DIPE ternary,
so the all-six common cohort is **four systems / 109 states**. Only two of
these systems report measured y: **74 states / 222 component errors**.
The glycerol systems contribute bubble-equation residuals only. One
methanol/water/glycerol point is outside the chosen pure-Psat curve domain.

| Model | Mean system ln(gamma) MAE, 2 systems | ln(gamma) RMS | Mean bubble-equation absolute residual / %, 4 systems |
|---|---:|---:|---:|
| UNIFAC | 0.22004 | 0.29790 | 12.6096 |
| UNIFAC2 | 0.17610 | 0.27630 | 6.4295 |
| Dortmund | 0.15605 | **0.24246** | 5.1484 |
| Modified UNIFAC 2.0 | 0.17734 | 0.28301 | 5.8070 |
| NIST | 0.17083 | 0.25889 | 7.0243 |
| HANNA | **0.13801** | 0.24545 | **2.8496** |

The larger comparisons excluding classic UNIFAC cover **five systems /
151 states**, with **three systems / 116 measured-y states** for ln(gamma).
On this same larger population, HANNA's MAE is **0.11876**, versus
**0.15488 / 0.13740 / 0.16006 / 0.13330** for UNIFAC2 / Dortmund /
modified 2.0 / NIST. Its RMS is **0.21425**, slightly worse than Dortmund's
**0.21193**. Its bubble-equation residual is **2.7560%**, versus
**6.5645% / 5.3312% / 6.4969% / 5.9874%**. These are promising local results,
but three measured-y ternary systems do not establish a broad ranking.
No ternary miscibility-gap or phase-count predictions were tested.

## Ethanol/water

All six methods use the same 37 measured states:

| Model | ln(gamma) MAE | Bubble-equation absolute residual / % |
|---|---:|---:|
| UNIFAC | 0.03670 | 1.6748 |
| UNIFAC2 | 0.06385 | 3.3510 |
| Dortmund | 0.03277 | **1.2493** |
| Modified UNIFAC 2.0 | 0.07633 | 3.7902 |
| NIST | **0.03154** | 1.3060 |
| HANNA | 0.04435 | 2.0164 |

HANNA improves on the two 2.0 models here, but does not beat the legacy
models or NIST.

## Uncertainty and checks

[`analyze_hanna_results.py`](analyze_hanna_results.py) independently verifies
all saved point errors, matched sample counts, and system-weighted MAE/RMS/
bubble-equation statistics, then performs 20,000 paired bootstrap replicates
with seed 20261013. It resamples whole systems, never individual points.

All five raw-binary ln(gamma) MAE difference intervals include zero. The
HANNA-minus-NIST difference is **-0.01062**, 95% interval
**[-0.03968, +0.01320]**. On agreeing fitted references, HANNA-minus-NIST is
**-0.00027**, interval **[-0.01676, +0.01619]**. HANNA improves on classic
UNIFAC in that fitted cohort: difference **-0.03404**, unadjusted interval
**[-0.05632, -0.01236]**, conservative familywise interval across 15
comparisons **[-0.06796, -0.00290]**. This is the only comparison that excludes
zero after that adjustment. Shared components, reference uncertainty and
unknown training overlap limit interpretation of these exploratory intervals.

The official public predictor agrees with the cached/batched path. Binary
and ternary component permutation, pure limits and identical-component
checks pass. Autograd Gibbs-Duhem residuals are **1.19e-7** for binary and
**4.84e-8** for ternary probes. All states satisfy sum(x_i ln(gamma_i))=gE/RT
within float32 tolerance. Twelve fresh predictions reproduce saved results;
all ten checkpoint dictionaries exactly match the expected architecture keys.
Ruff, compilation and dependency compatibility checks pass.

The first setup exceeded the temporary filesystem's copy quota; the final
isolated environment links packages from the existing cache instead of
duplicating them. It uses Python 3.11.16, PyTorch 2.10.0+cpu, NumPy 2.2.6,
Transformers 5.1.0, scikit-learn 1.7.2 and RDKit 2025.9.5. Source/weight
hashes, the exact dependency lock and repository commit are retained.

The first complete pass took **105.56 s excluding imports**, but lacked
acid vapor-backend dependencies. The final pass restored all four acid
pairs by reusing their saved corrected reference activities, took **70.62 s**,
and reused **5,462 completed HANNA state predictions**. Neither timing is
a standalone inference benchmark. A subsequent script edit caches pure
standard-state factors once per source/T/P, avoiding six redundant reference
calculations per point. Cached and uncached factors were checked to give
identical results; the executed script snapshot and checksum manifest are
retained. No completed model predictions were rerun to recover output.

## Artifacts and reproduction

Maintained files: [benchmark_hanna.py](benchmark_hanna.py),
[collector](collect_hanna_ternary_vle.py), [uncertainty analysis](analyze_hanna_results.py),
and [normalized ternary measurements](../../data/source/activity_fitting/hanna_ternary_vle_measurements.json).
No production code, enabled interactions, or existing reference tables changed.

Temporary artifacts are under `/tmp/hanna-investigation-KEruC4eG/`:
`final-results/` contains points, system scores, exclusions and summary;
`uncertainty/` contains paired intervals. The directory also preserves source
code/weights, environment, logs, executed-script snapshots and provenance.
`corrected-results/` is an intermediate checkpoint used to avoid repeating
completed predictions; `results/` belongs to an early failed input-layout
check and is not a numerical result set.

Run from the repository root, using a new output directory:

```bash
set -o pipefail
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  NUMBA_CACHE_DIR=/tmp/hanna-cache HF_HUB_OFFLINE=1 \
  HF_HUB_DISABLE_PROGRESS_BARS=1 MPLCONFIGDIR=/tmp/hanna-matplotlib \
  /tmp/hanna-investigation-KEruC4eG/venv-linked/bin/python -u \
  scripts/activity_fitting/benchmark_hanna.py \
  --hanna-root /tmp/hanna-investigation-KEruC4eG/HANNA \
  --binary-baseline /tmp/unifac-raw-vle-5c4T8rc5/direct-final \
  --fit-baseline /tmp/unifac-runtime-benchmark-YfgViJRE/validated-results \
  --output-dir /tmp/hanna-repeat \
  2>&1 | tee /tmp/hanna-repeat.log
```

`--resume-predictions` can reuse a completed HANNA results directory after
checking reference hashes and model repository commit. It does not bypass
reference evaluation or aggregation. The predecessor binary and agreeing-fit
benchmark scripts generate the required baseline files; the saved baseline
predictions and targets are intentionally fixed for this investigation.
