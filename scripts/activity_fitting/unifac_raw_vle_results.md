# Pair-weighted UNIFAC comparison with raw experimental VLE

Investigation on 2026-10-12. This comparison uses measured binary P/T/x and,
where independently measured, y. It does not use fitted NRTL/UNIQUAC surfaces
as reference truth, and no liquid interaction parameters were fitted.

## Grand totals

The collected source contains **1,976 mixed-composition points across 49 binary
pairs and 24 sources**. Five UNIFAC methods share **1,805 qualified points across
43 pairs**. The source includes 694 points with measured vapor composition;
the fully matched, usable ln(gamma) population is **26 pairs / 577 states /
1,154 component values**. Different metrics therefore have different coverage.

For each metric, first average point errors within a binary pair, then average
equally across pairs. RMS is the square root of the equally weighted average
of pair MSE values. Multiple papers or temperature series do not give a binary
pair more total weight. The two component ln(gamma) errors are equally weighted.

### Direct ln(gamma) errors at measured T/P/x/y

| Method | Mean pair MAE | Median pair MAE | Pair-weighted RMS |
|---|---:|---:|---:|
| UNIFAC | 0.265247 | 0.048965 | 0.977127 |
| UNIFAC 2.0 | 0.266709 | 0.055333 | 0.970969 |
| Dortmund | 0.275769 | 0.064623 | 0.939370 |
| Modified UNIFAC 2.0 | 0.273399 | 0.078466 | 0.932524 |
| NIST | 0.283031 | 0.083617 | 0.919093 |

These are dimensionless natural-log errors, not percentages. UNIFAC 2.0 has
a **0.55% higher** mean MAE than classic UNIFAC but a slightly lower RMS.
Modified 2.0 has a **0.86% lower** mean MAE than Dortmund, a lower RMS, and a
higher median. NIST has the lowest RMS but the highest mean/median MAE.
The raw comparison does not establish a consistent overall winner.

Inferred experimental values follow the same modified Raoult expression used
by PFDSim's gamma-phi models:

```text
gamma_i = y_i * phi_i^V * P / (x_i * phi_i^sat * Psat_i * Poynting_i)
```

Here P/T/x/y are measured; phi, Psat and Poynting are shared reference-model
quantities. Thus ln(gamma) is inferred from measurements, not measured directly,
and inherits pure-reference and vapor-model uncertainty. Rounded y=0 or 1 does
not identify a finite solute gamma and is not converted to a logarithm.

### Completed forward checks: all 43 common pairs

The forward calculations had already completed before slow optional temperature
solves were removed. Their correctly checked predictions are retained as
supplementary raw-data comparisons; reanalysis does not repeat those solves.

| Method | Pair-mean pressure AARD | Pair-weighted RMS y (26 pairs) |
|---|---:|---:|
| UNIFAC | 8.5498% | 0.048536 |
| UNIFAC 2.0 | 8.2150% | 0.048148 |
| Dortmund | 10.6433% | 0.052778 |
| Modified UNIFAC 2.0 | 9.2955% | 0.052173 |
| NIST | 8.5194% | 0.056348 |

On pressure AARD, UNIFAC 2.0 improves classic UNIFAC by **3.92%**, and modified
2.0 improves Dortmund by **12.66%**, relative to their predecessors' errors.
These gains do not translate into uniform improvement in every metric or pair.
Vapor compositions in this supplemental calculation are evaluated at predicted
bubble pressure for the measured temperature and liquid composition.

The final script instead evaluates the fugacity equations directly at measured
states. Its bubble-equation closure residual is **not a solved pressure error**.
For measured-y states it is exactly
`sum_i[y_i * exp(ln(gamma_pred,i)-ln(gamma_inferred,i))] - 1`.
For pressure-only data the script uses a bounded fixed-P composition iteration,
without a pressure or temperature root. Pressure-only points cannot identify
the two individual component activity coefficients.

## Coverage and source collection

| Method | Represented pairs | Qualified mixed points |
|---|---:|---:|
| UNIFAC | 43 | 1,805 |
| UNIFAC2 | 47 | 1,922 |
| UNIFDMD | 44 | 1,826 |
| UNIFM2 | 47 | 1,922 |
| UNIFNIST | 44 | 1,826 |

The 2.0 matrices provide a clear applicability gain in this population even
where the accuracy gains are modest. The all-method comparison excludes:

- Two sodium-chloride pairs, outside the molecular UNIFAC structural domain.
- Three fluorinated acid/ester pairs for which classic/Dortmund/NIST interaction
  matrices are incomplete, though both 2.0 models can evaluate them.
- DIPE/2-methoxyethanol, for which classic UNIFAC interactions are incomplete.
- 39 mixed points outside the selected glycerol pure-reference domain.

Together these exclusions remove 171 of 1,976 points. Excluded records and
reasons are saved; missing group parameters are not silently set to zero.

The collector scans the existing local ThermoML files and the NIST archives
associated with the local PDFs. Eighteen downloaded archives were available;
four existing local archives add more data. Source-local literal measurement
tables supply Fendu's two alcohol binaries, Hartanto ethanol/hexanol, Arce
ethanol/octanol, and Mustain alcohol/glycerol pairs. Existing fitting scripts
are parsed with `ast.literal_eval`; their regression code is not executed.

The 24 included sources are:

| DOI / source identifier | Mixed points |
|---|---:|
| 10.1016/j.jct.2012.09.032 | 247 |
| 10.1016/j.fluid.2017.09.008 | 172 |
| 10.3390/pr13072034 | 158 |
| 10.1016/j.jct.2013.09.046 | 152 |
| 10.1016/j.fluid.2014.09.014 | 142 |
| 10.1016/j.jct.2009.11.020 | 130 |
| 10.1016/j.jct.2012.11.005 | 113 |
| 10.1016/j.fluid.2015.08.012 | 96 |
| 10.1016/j.fluid.2015.09.052 | 79 |
| 10.1016/j.fluid.2006.06.017 | 76 |
| 10.1016/j.fluid.2007.09.014 | 76 |
| 10.1021/je200655s | 67 |
| 10.1021/acs.jced.0c00204 | 60 |
| 10.1021/je2008704 | 55 |
| 10.1021/acs.jced.1c00937 | 54 |
| 10.1016/j.fluid.2012.01.007 | 48 |
| 10.1016/j.fluid.2016.05.031 | 48 |
| 10.1016/j.jct.2010.05.014 | 46 |
| 10.1021/acs.jced.0c00089 | 36 |
| 10.1021/je501019y | 36 |
| 10.1016/j.fluid.2005.03.034 | 26 |
| 10.1016/j.jct.2008.06.005 | 23 |
| 10.1021/je00020a063 | 23 |
| 10.13140/RG.2.1.4542.3766 | 13 |

Source hashes, dataset IDs, row IDs, methods and point coordinates are retained.
Some other local papers have no NIST archive and were not transcribed in this
run; the download/inventory logs identify them. This is a broad available-data
comparison, not a claim that every local PDF was fully extracted.

Mass fractions, including Veneral's glycerol systems, are converted using exact
resolved molecular weights. Three duplicate measured states are deduplicated.
Pure endpoints are separate diagnostics and do not inflate liquid-interaction
scores. Barker-derived Moreau vapor compositions are excluded as independent
observations. Six Reddy vapor values could not be attached uniquely to measured
bubble states; their pressure measurements remain, while ambiguous y is not
guessed. All 85 relevant composition-variable definitions were checked to refer
to the liquid phase.

## Shared physics and uncertainty

Every liquid model uses the same pure/vapor inputs for a given state:

- Source-specific Antoine/DIPPR curves where extracted from the papers or
  existing source-basis scripts. Fendu's pure Antoine curves are fitted only
  to its pure-fluid measurements, using its existing helper.
- Otherwise, in-range Perry 2-8, Poling Wagner, Landolt Antoine, or VDI PPDS
  curves. Their source labels are saved per point. No liquid mixture data is
  used to adjust these pure references.
- Tsonopoulos vapor corrections for nonacids; association-aware Hayden-O'Connell
  for carboxylic-acid systems, including saturation fugacity and Poynting.
- Common vapor-input critical properties prefer Perry/local/reference banks;
  Nannoolal fills missing critical inputs only for this shared vapor correction.
  Runtime warnings and numerical inputs are saved.

Of 182 collected pure endpoints, 179 were evaluable with these curves. Their
median absolute pressure discrepancy was **0.389%**, and p95 was **5.83%**.
This is a reference diagnostic, not a liquid-model score. The largest relative
discrepancies occur at glycerol pressures of only a few pascals; absolute
rounding/uncertainty matters strongly there.

The ln(gamma) RMS tail is dominated by alcohol/glycerol data. For example,
Batutah 2020's Table 3 reports ethanol/glycerol at 352.35 K and 101.3 kPa,
`x_ethanol=0.9410`, `y_ethanol=0.9924`. With the shared pure/vapor reference,
the inferred glycerol ln(gamma) is **7.68546**, versus UNIFAC **0.16872**.
For 1-propanol/glycerol at 330.05 K and 16 kPa, the corresponding values are
**7.46632** and **0.57546**. These values were checked against the local PDF,
which describes refractometric measurement of liquid and vapor samples, with
`u(y)=0.002`. They are not transcription guesses or calculated vapor curves.

These inferred gammas are sensitive to the very small heavy-component Psat and
trace vapor fraction. They warrant source/reference review; the comparison
alone does not establish which input is wrong. The records are retained in
the grand total rather than selectively removed to improve scores. The large
gap between mean and median pair error is therefore material context.

## Algorithm correction, verification, and artifacts

The initial algorithm unnecessarily nested a bubble-pressure solve inside each
bubble-temperature solve. Its first run took **411 s**, and a corrected partial
rerun took **190 s**, reusing 34 pair checkpoints. Both completed before attempted
worker interrupts reached them. Their evaluated outputs remain available.
The pressure/temperature roots have now been removed from the maintained
benchmark; its default evaluates measured-state fugacity relations directly.
The 1.66-second final aggregation reused saved numerical predictions rather
than repeating the forward solves. Three pairs/five methods were also freshly
evaluated directly in **4.98 s**, including initialization; all 15 evaluations
matched the saved measured-state gamma values at `1e-10` tolerance.

The forward checks revealed that the runtime's capped vapor-composition
iteration can leave small fugacity residuals for associated vapors. Those
forward states were refined to satisfy the same physical equations rather than
discarded as liquid-model failures. The remaining stored maximum residual is
`1.22e-6` among reused states; corrected states were admitted at `1e-8`.
Production code and runtime parameters were not changed. The final direct-state
metric does not depend on this forward-solver convergence behavior.

Verification independently checked identical raw-state sets, ln(gamma) error
arithmetic, pair-equal MAE/median/RMS, bubble-equation residual identities,
source/script checksums, source composition phases, and the 15 fresh direct
evaluations. Ruff and Python compilation passed.

Saved files:

- [Raw measurement source](../../data/source/activity_fitting/raw_vle_measurements.json).
- [Collector](collect_raw_vle_benchmark.py).
- [Maintained direct-state benchmark](benchmark_unifac_raw_vle.py).
- `/tmp/unifac-raw-vle-5c4T8rc5/direct-final/`: final pair-weighted summary,
  point/pair records, exclusions, and model information.
- `/tmp/unifac-raw-vle-5c4T8rc5/final-results/`: completed forward checks and
  original numerical predictions; earlier directories are intermediate runs.
- The surrounding temporary directory preserves source downloads, archive
  inventory/status, pure-endpoint checks, verification log, and snapshots of
  the previously executed forward scripts matching their recorded hashes.

Run the direct-state benchmark with a new output directory:

```bash
set -o pipefail
NUMBA_CACHE_DIR=/tmp/raw-vle-cache OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  python3 -u scripts/activity_fitting/benchmark_unifac_raw_vle.py \
  --output-dir /tmp/raw-vle-direct-results --workers 5 \
  2>&1 | tee /tmp/raw-vle-direct.log
```

`--summarize-dir <saved prediction directory>` reaggregates measured-state
gamma predictions without running any numerical model calculations.

The pressure results show useful modest 2.0 gains, and complete interaction
matrices improve coverage. The grand direct ln(gamma) comparison remains mixed.
The data are real measurements, but pure/vapor reference uncertainty, trace
vapor measurement sensitivity, source quality, and training overlap remain
limitations on interpreting small differences as universal model superiority.

## Paired uncertainty and ethanol/water

[analyze_unifac_raw_vle_uncertainty.py](analyze_unifac_raw_vle_uncertainty.py)
analyzes the saved pair scores without repeating equilibrium calculations.
It uses 20,000 paired bootstrap replicates, seed 20261012, resampling binary
pairs rather than individual observations. Both old/new methods use the same
resampled pairs. Negative new-minus-old error differences favor 2.0.

| Comparison | Metric | Difference | Unadjusted 95% interval | Bonferroni familywise 95% interval across 8 comparisons |
|---|---|---:|---:|---:|
| UNIFAC to UNIFAC2 | Pressure AARD, percentage points | -0.33486 | [-1.98717, 1.19583] | [-2.69519, 1.77148] |
| UNIFAC to UNIFAC2 | Temperature RMS, K | -0.31639 | [-0.92290, 0.12231] | [-1.26282, 0.29647] |
| UNIFDMD to UNIFM2 | Pressure AARD, percentage points | -1.34778 | [-4.08572, 0.65128] | [-5.45654, 1.15131] |
| UNIFDMD to UNIFM2 | Temperature RMS, K | -0.76069 | [-1.51858, -0.10177] | [-1.82786, 0.09766] |

Modified 2.0's temperature improvement excludes zero in the unadjusted
interval. Its ln(gamma) RMS difference also barely excludes zero, at
[-0.01468, -0.00007]. Neither survives the conservative eight-comparison
adjustment. None of the eight adjusted intervals excludes zero.

These are exploratory intervals conditional on the available pairs. Pairs
share components and are not a random sample of chemical space, so bootstrap
independence across pairs is only approximate. Measurement and pure/vapor
reference uncertainties and ML training overlap are not propagated. Absence
of a resolved difference does not establish equivalence.

For ethanol/water, all methods use the same 37 measured states:

| Method | Pressure AARD | Temperature RMS | Vapor-y RMS | ln(gamma) MAE |
|---|---:|---:|---:|---:|
| UNIFAC | 1.7071% | 0.7523 K | 0.01811 | 0.03670 |
| UNIFAC2 | 3.4304% | 0.9799 K | 0.02191 | 0.06385 |
| UNIFDMD | 1.2725% | 0.5359 K | 0.01662 | 0.03277 |
| UNIFM2 | 3.8805% | 1.1135 K | 0.02562 | 0.07633 |
| UNIFNIST | 1.3302% | 0.5407 K | 0.01606 | 0.03154 |

Both 2.0 variants regress relative to their predecessors on this pair in the
tested conditions. The uncertainty summary and complete eight-comparison
tables are saved in `/tmp/unifac-raw-vle-5c4T8rc5/uncertainty-adjusted/`.
