# UNIFAC families: runtime-fit agreement and direct phenolic measurements

Investigation performed on 2026-10-12. The grand total across the runtime
interaction collection is the main result; the Cesari 2017 measurements are
a local experimental check, not a substitute for the broad population.

No production code, enabled interaction parameters, or existing reference
data were changed. New scripts and an extracted experimental source are saved.

## Main result: grand total across runtime NRTL and UNIQUAC

The authoritative runtime loaders select **445 NRTL pairs and 414 UNIQUAC
pairs**, covering **533 distinct binary pairs**. Disabled and superseded records
are excluded by those loaders. All 859 selected reference cases were evaluated.
UNIFAC chemistry/parameter coverage differs; **462 distinct pairs / 768 fitted
reference cases** are fully parameterized and evaluable by all five methods.

For each fitted reference, score both components on the same temperature and
composition grid. Average the absolute ln(gamma) errors across its states;
then, when a pair has both NRTL and UNIQUAC references, average those two scores.
Finally, give each distinct pair equal weight. MSE is combined the same way,
with RMS equal to the square root of the mean per-pair MSE. This prevents a
pair with two fitted references from receiving twice the weight of another.

**All errors below are dimensionless natural-log errors, not percentages.**

| Method | Mean pair MAE | Median pair MAE | Pair-weighted RMS |
|---|---:|---:|---:|
| UNIFAC | 0.238006 | 0.093987 | 1.120838 |
| UNIFAC 2.0 (`UNIFAC2`) | 0.232963 | 0.100293 | 1.124366 |
| Dortmund (`UNIFDMD`) | 0.237397 | 0.089232 | 1.113656 |
| Modified UNIFAC 2.0 (`UNIFM2`) | 0.242416 | 0.098650 | 1.122734 |
| NIST (`UNIFNIST`) | 0.238994 | 0.094280 | 1.117307 |

UNIFAC 2.0 reduces the grand mean MAE by **2.12% relative to UNIFAC**, but has
a higher median and slightly higher RMS. Modified UNIFAC 2.0's mean MAE is
**2.11% higher than Dortmund's**; its median and RMS are also higher. The totals
do not show a decisive universal improvement or a single winner.

### Flagged reference and sensitivity of the total

The active ethanol/o-cresol NRTL reference reaches the runtime gamma floor.
It is retained in the main total and explicitly flagged. Excluding only that
reference leaves **461 pairs / 767 fitted reference cases**:

| Method | Mean pair MAE | Median pair MAE | Pair-weighted RMS |
|---|---:|---:|---:|
| UNIFAC | 0.208023 | 0.093574 | 0.680999 |
| UNIFAC2 | 0.203073 | 0.099719 | 0.665232 |
| UNIFDMD | 0.207549 | 0.089131 | 0.660143 |
| UNIFM2 | 0.212591 | 0.098081 | 0.672771 |
| UNIFNIST | 0.209176 | 0.094217 | 0.666701 |

The same mean-MAE ordering survives. The RMS falls markedly for every method,
demonstrating the influence of this reference rather than an improvement in
any UNIFAC calculation.

Only 123 NRTL and 112 UNIQUAC references declare temperature ranges. On the
common, bounded, unclipped subset (**122 distinct pairs / 227 fitted cases**):

| Method | Mean pair MAE | Median pair MAE | Pair-weighted RMS |
|---|---:|---:|---:|
| UNIFAC | 0.190117 | 0.108408 | 0.624131 |
| UNIFAC2 | 0.178149 | 0.105705 | 0.568794 |
| UNIFDMD | 0.177009 | 0.083259 | 0.600557 |
| UNIFM2 | 0.183557 | 0.099894 | 0.596403 |
| UNIFNIST | 0.172876 | 0.092384 | 0.587195 |

NIST has the lowest mean MAE in this bounded subset, while UNIFAC2 has the
lowest RMS and Dortmund the lowest median. Those subset results do not override
the much larger grand-total population.

### Larger family-wise comparisons and coverage

Each family comparison uses all fully parameterized fitted references available
to both methods, then combines reference scores equally within distinct pairs.

| Comparison | Distinct pairs | Fitted cases | Old mean MAE | New mean MAE | New-model pair wins | Ties |
|---|---:|---:|---:|---:|---:|---:|
| UNIFAC vs UNIFAC2 | 476 | 788 | 0.241438 | 0.233374 | 240 | 14 |
| UNIFDMD vs UNIFM2 | 481 | 797 | 0.239296 | 0.245794 | 209 | 16 |

Ties use a `1e-12` relative/absolute tolerance and are excluded from wins.

| Reference | Method | Evaluated | Fully parameterized | Missing-interaction fallback | Unsupported/failed |
|---|---|---:|---:|---:|---:|
| NRTL (445) | UNIFAC | 427 | 424 | 3 | 18 |
| NRTL (445) | UNIFAC2 | 427 | 427 | 0 | 18 |
| NRTL (445) | UNIFDMD | 430 | 426 | 4 | 15 |
| NRTL (445) | UNIFM2 | 430 | 430 | 0 | 15 |
| NRTL (445) | UNIFNIST | 435 | 430 | 5 | 10 |
| UNIQUAC (414) | UNIFAC | 366 | 364 | 2 | 48 |
| UNIQUAC (414) | UNIFAC2 | 366 | 366 | 0 | 48 |
| UNIQUAC (414) | UNIFDMD | 379 | 371 | 8 | 35 |
| UNIQUAC (414) | UNIFM2 | 379 | 379 | 0 | 35 |
| UNIQUAC (414) | UNIFNIST | 382 | 374 | 8 | 32 |

The completed 2.0 matrices eliminate missing interaction parameters for
representable structures. They do not create new structural groups for every
unsupported molecule. This already-curated collection has relatively few
missing interactions, so the observed applicability gain is modest here.

## Broad benchmark protocol and reference uncertainty

Maintained entry point: [benchmark_unifac_runtime_fits.py](benchmark_unifac_runtime_fits.py).

- Uses the runtime-selected NRTL/UNIQUAC records and actual runtime activity
  evaluators, including UNIQUAC's selected r/q and q-prime structural basis.
- Declared temperature ranges: Tmin, midpoint, Tmax, deduplicated for a
  single-temperature fit. Unspecified ranges: 298.15, 323.15, 348.15 K, clearly
  separated in the saved results. Unspecified probes do not establish validity.
- Fifteen symmetric compositions `x1 = 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2,
  0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999`, plus compositions
  maximizing the reference's absolute gE/(RT) on 49 interior points at each
  selected temperature. Selection is reference-only, not based on model errors.
- Both component ln(gamma) values are scored equally. Dilute-solute and bulk
  diagnostics are saved separately. No percent error of ln(gamma) is computed.
- Incomplete interaction matrices retain flagged runtime fallback results,
  but are excluded from the primary fully parameterized comparisons.
- No Psat, vapor fugacity, bubble solver, or LLE solver enters these ln(gamma)
  scores. Homogeneous extensions inside miscibility gaps and very dilute
  compositions are deliberate sensitivity probes; fitted composition ranges
  are generally not supplied by the interaction records.

All selected references contain **41,716 sampled states / 83,432 ln(gamma)
values**. The fitted references are not experimental truth. Among 326 pairs
with both references, NRTL/UNIQUAC agreement has a median pair MAE of
**0.04682**, a mean of **0.19132**, and RMS **0.60558**. Among the 107 pairs with
overlapping declared ranges these become **0.02511**, **0.10453**, and **0.40599**.
Large discrepancies are concentrated in some difficult phenolic/water pairs
and dilute regions.

An additional matched diagnostic selects bounded pairs where the NRTL/UNIQUAC
MAE is <=0.1, and evaluates all methods on the same intersected temperature
range and base composition grid. It covers **82 pairs / 3,570 states**. Reference
spread is then MAE **0.01971**, RMS **0.05047**, while UNIFAC errors remain
roughly MAE **0.086--0.121**, RMS **0.198--0.270**, depending on model/reference.
This supports using fitted references in well-agreeing cases, but does not
justify assuming every runtime fit has negligible uncertainty. That diagnostic
is not the headline ranking. Runtime r/q fallback warnings occur in 58 UNIQUAC
references and are recorded.

### Phenolic reference issue

At **348.5 K**, ethanol/o-cresol NRTL has `tau12 = -465.91338` in the runtime's
ethanol-first orientation. At x_ethanol=0.5, ln(gamma_ethanol)=**0.12643** and
ln(gamma_o-cresol)=**-27.63102**, the latter being the gamma=`1e-12` floor.
Finite differentiation gives a Gibbs-Duhem residual
`x*dln(gamma1)/dx + (1-x)*dln(gamma2)/dx = -0.27605`; it should be zero.
The returned clipped activity coefficients therefore violate Gibbs-Duhem in
this region. No production fix or refit was performed.

The stored coefficients match the numbers printed in the local Cesari paper's
Table 7. The builder applies the declared `Delta_g = a + b*T` conversion.
Faithful transcription alone does not resolve the physical behavior or
establish the interpretation/orientation of those printed parameters. Direct
comparison with the measurements, below, shows the active fits need review.

## Direct check: Cesari 2017 experimental VLE/LLE

Source: Cesari et al., *Fluid Phase Equilibria* 453 (2017) 58--66,
[DOI 10.1016/j.fluid.2017.09.008](https://doi.org/10.1016/j.fluid.2017.09.008).
The local PDF is under `data/reference/multiphase-equilibria/`.

[NIST ThermoML](https://trc.nist.gov/ThermoML/10.1016/j.fluid.2017.09.008.html)
provides the machine-readable measurements. The normalized source is
[cesari_2017_phase_equilibria.json](../../data/source/activity_fitting/cesari_2017_phase_equilibria.json).
Its extraction script [extract_cesari_phase_equilibria.py](extract_cesari_phase_equilibria.py)
checks counts, selected values against PDF Tables 3/6, and extracts the original
Antoine coefficients from the publisher's supporting-information DOCX Table S2.

The direct benchmark is [benchmark_cesari_phase_equilibria.py](benchmark_cesari_phase_equilibria.py).
It uses the same source Psat correlations for all models, with the paper's
ideal-vapor and unit-Poynting assumptions. The SI's ethanol coefficient unit
label is inconsistent: its ln(kPa) coefficients require T in kelvin, giving
Tb=351.4865 K. Using Celsius with those coefficients is physically inconsistent.
This interpretation is explicitly retained in source metadata.

### VLE grand total: 172 measured bubble points

| Method | RMS T / K | Pressure RMS / bar | Pressure AARD / % |
|---|---:|---:|---:|
| UNIFAC | 5.3103 | 0.08577 | 13.371 |
| UNIFAC2 | 4.9007 | 0.12029 | 15.951 |
| UNIFDMD | 4.6803 | 0.08168 | 10.158 |
| UNIFM2 | 2.9063 | 0.05937 | 8.457 |
| UNIFNIST | 3.9233 | 0.06923 | 8.353 |
| NRTL runtime | 10.2947 | 0.24352 | 38.708 |

All six methods evaluated all 172 points and obtained pressure-consistent
bubble-temperature roots. Modified 2.0 has the lowest RMS(T); NIST has the
lowest pressure AARD. UNIFAC2 improves RMS(T) but worsens pressure RMS/AARD.

| Method | Phenol/ethanol: 54 points, RMS T/K | Guaiacol/ethanol: 63 points, RMS T/K | o-Cresol/ethanol: 55 points, RMS T/K |
|---|---:|---:|---:|
| UNIFAC | 2.8398 | 7.5666 | 3.8327 |
| UNIFAC2 | 6.6112 | 1.5237 | 5.4346 |
| UNIFDMD | 1.7790 | 7.3681 | 1.7921 |
| UNIFM2 | 2.6011 | 3.8942 | 1.5494 |
| UNIFNIST | 1.8825 | 6.0785 | 1.5283 |
| NRTL runtime | 12.5038 | 7.0008 | 11.0358 |

There are no enabled UNIQUAC entries for these three ethanol pairs, so no
UNIQUAC VLE reference was invented. The paper reports P, T, and liquid
composition, but **does not report measured vapor composition**, so RMS(y) is
unavailable. The paper's printed pressure-fit RMS figures are approximately
0.021--0.025 bar, not the same metric as RMS(T).

### LLE grand total: 32 temperatures, including 26 complete tie-lines

The other six records are phenol aqueous-solubility points with no reported
organic endpoint. Syringol's three high-temperature records are LLE; its four
low-temperature SLE records and all other SLE data are excluded.

| Method | Predicted two liquids / 32 | Aqueous phenolic x RMS | Organic phenolic x RMS | Measured-tie RMS delta ln(activity) |
|---|---:|---:|---:|---:|
| UNIFAC | 17 | 0.009915 | 0.137893 | 1.15276 |
| UNIFAC2 | 32 | 0.003901 | 0.123617 | 0.31656 |
| UNIFDMD | 32 | 0.004736 | 0.144397 | 0.34993 |
| UNIFM2 | 32 | 0.005466 | 0.137806 | 0.41556 |
| UNIFNIST | 32 | 0.001581 | 0.141185 | 0.32242 |
| NRTL runtime | 32 | 0.051925 | 0.227425 | 0.39958 |

Endpoint composition RMS includes successful splits only: UNIFAC's 17
successes cannot be directly equated with the other methods' 32. Experimental
tie residuals use all 26 complete ties and no phase solve; equality of both
component activities requires zero residual. UNIFAC2 improves both split
coverage and tie agreement relative to classic UNIFAC. Modified 2.0 does not
uniformly improve Dortmund LLE, and NIST best matches the aqueous endpoints.
Organic endpoints remain materially imperfect for all UNIFAC variants.

Runtime UNIQUAC has no active guaiacol/water or syringol/water entry. It predicts
no split for the 17 cresol/water records that it can evaluate. It predicts a
split for six phenol/water temperatures but with aqueous-x RMS **0.102604**.
The partial 17-tie activity residual is **3.37646**. These are not comparable
full-coverage UNIQUAC totals.

## Verification, artifacts, and reproduction

The broad benchmark took **38.11 seconds**, and the direct check **23.11 seconds**
on one Numba thread with an isolated temporary compiled cache. These are
single-run investigation timings, not performance benchmarks.

Verification checked identical broad grids across methods, all broad point
and pair error arithmetic, common-cohort aggregation, all three unique-pair
grand totals, all 1,204 direct VLE and 224 direct LLE records, root and phase
residuals, and four independently reproduced bubble pressures. Six
NRTL/UNIQUAC runtime/native checks and **104 UNIFAC runtime/native checks**
passed; the largest UNIFAC ln(gamma) difference was **2.462e-15**. Ruff and
Python compilation passed. Input and implementation hashes are retained.
The saved direct-benchmark script differs from the executed version only by
removal of an unused import. The executed source was reconstructed and checked
against its recorded checksum; its snapshot and a provenance manifest are
saved with the direct results. No numerical code changed after that run.

Saved evaluations:

- `/tmp/unifac-runtime-benchmark-YfgViJRE/validated-results/`: full point/pair
  evaluations, reference inventory, reference-agreement diagnostics, exclusions,
  native checks, summary, and generated report.
- `/tmp/unifac-runtime-benchmark-YfgViJRE/grand-results/`: final grand-total
  summary/report, referencing those unchanged evaluations. Earlier output
  directories are intermediate runs.
- `/tmp/cesari-direct-benchmark-3RIetGLQ/results/`: direct VLE/LLE point records,
  per-method checkpoints, failures, summary, and generated report.
- Verification logs and source downloads are in the surrounding temporary
  directories. The source JSON retains download URLs/hashes and uncertainties.

Run from the repository root, using new output directories:

```bash
set -o pipefail
NUMBA_CACHE_DIR=/tmp/unifac-repeat-cache OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  python3 -u scripts/activity_fitting/benchmark_unifac_runtime_fits.py \
  --output-dir /tmp/unifac-repeat-results \
  2>&1 | tee /tmp/unifac-repeat.log

NUMBA_CACHE_DIR=/tmp/unifac-repeat-cache OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  python3 -u scripts/activity_fitting/benchmark_cesari_phase_equilibria.py \
  --output-dir /tmp/cesari-repeat-results \
  2>&1 | tee /tmp/cesari-repeat.log
```

`benchmark_unifac_runtime_fits.py --summarize-dir <saved evaluations>` can
recompute the grand-total analysis without repeating model predictions.

The grand-total conclusion remains **mixed, with no decisive universal 2.0
advantage against this runtime-fit collection**. The experimental phenolic
check shows useful specific improvements, particularly modified 2.0 VLE and
classic 2.0 LLE, but cannot establish a broad ranking from a few pairs. Fitted
reference uncertainty, undefined temperature/composition coverage, and ML
training overlap remain limits on interpretation.
