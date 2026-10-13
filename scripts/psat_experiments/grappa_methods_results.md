# GRAPPA, Nannoolal, and Ambrose-Walton comparison

Run on 2026-10-12 using [benchmark_grappa_methods.py](benchmark_grappa_methods.py).
The comparison reuses every temperature, reference pressure, and GRAPPA
prediction from the [original GRAPPA benchmark](grappa_results.md), without
resampling or rerunning the neural network.

## Configurations

1. **GRAPPA:** saved structure-only predictions from its published checkpoint.
2. **Nannoolal:** `nannoolal_method.estimate_psat(smiles)`, including its internal
   normal-boiling-point estimate.
3. **Nannoolal, real Tb:** the same implementation with the reference source's
   own Tb supplied. Perry 2-8 Tb is solved from the full declared correlation
   range, Perry 2-10 uses its 760-mmHg point, and CoolProp uses physical HEOS
   liquid-vapor saturation at 101325 Pa. Source Tb may lie outside the sampled
   250--600 K interval, but is not extrapolated beyond the reference's domain.
4. **AW, real Tb, estimated criticals:** Nannoolal Tc/Pc estimated using that
   source Tb; omega solved so the AW equation reproduces 101325 Pa at Tb.
5. **AW, real criticals:** supplied Tc/Pc/omega, with no boiling-point anchoring.
   CoolProp comparisons use HEOS values. Perry comparisons use Perry 2-106,
   falling back to exact-CAS HEOS data when those constants are unavailable.
   For Perry 2-10 this gives 196 Perry sets plus six HEOS sets.

Nannoolal uses PFDSim's default local refits and extensions. These results
therefore describe the existing local implementation, not a strictly
unmodified reproduction of the original papers. The shared PFDSim AW functions
are also used directly. No production code or reference data was changed.

## Matched comparison across all five methods

Each included source/compound retains its entire original sampled curve.
Compounds unavailable for any method, or with any sampled point outside an AW
model's liquid-vapor domain, are excluded from this matched table. Remaining
methods use exactly the same points. Original source counts were 305, 838, and
84 compounds; matched counts are smaller because information and model
coverage differ.

Errors are absolute percentage errors relative to reference pressure. Average
the points within each compound first, then take the median/mean across
compounds. Each cell below is **median (mean)** compound error, in percent.

| Method | Perry 2-8: 185 compounds, 1,850 points | Perry 2-10: 192 compounds, 1,508 points | CoolProp: 39 compounds, 390 points |
|---|---:|---:|---:|
| GRAPPA | 6.56 (9.16) | 5.51 (12.65) | 3.81 (8.96) |
| Nannoolal | 16.41 (25.33) | 22.43 (34.32) | 15.19 (28.34) |
| Nannoolal, real Tb | 6.11 (9.03) | 3.88 (12.40) | 4.98 (6.99) |
| AW, real Tb, estimated criticals | 6.58 (13.51) | 5.26 (15.55) | 3.10 (7.49) |
| AW, real criticals | 3.23 (9.40) | 5.65 (15.27) | 0.38 (2.31) |

Mean errors expose the outlier tail; ranking by the median alone does not
guarantee the same ranking by the mean.

## Larger pairwise comparisons with GRAPPA

These compare each competitor with GRAPPA on every complete original curve
that both can evaluate. Each row is a separate matched population.

| Source | Competitor | Compounds | Points | GRAPPA median | Competitor median | GRAPPA mean | Competitor mean | GRAPPA compound wins |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Perry 2-8 | Nannoolal | 299 | 2,990 | 6.03% | 17.04% | 8.29% | 26.73% | 248/299 |
| Perry 2-8 | Nannoolal, real Tb | 298 | 2,980 | 6.02% | 4.60% | 8.26% | 7.43% | 127/298 |
| Perry 2-8 | AW, real Tb, estimated criticals | 185 | 1,850 | 6.56% | 6.58% | 9.16% | 13.51% | 80/185 |
| Perry 2-8 | AW, real criticals | 305 | 3,050 | 6.09% | 2.30% | 9.60% | 6.89% | 64/305 |
| Perry 2-10 | Nannoolal | 812 | 7,314 | 9.34% | 21.63% | 18.13% | 47.76% | 608/812 |
| Perry 2-10 | Nannoolal, real Tb | 807 | 7,285 | 9.33% | 7.45% | 17.74% | 16.42% | 319/807 |
| Perry 2-10 | AW, real Tb, estimated criticals | 758 | 6,844 | 9.01% | 9.14% | 17.39% | 19.06% | 376/758 |
| Perry 2-10 | AW, real criticals | 202 | 1,568 | 5.59% | 5.90% | 12.74% | 15.31% | 104/202 |
| CoolProp | Nannoolal | 82 | 820 | 3.64% | 18.74% | 6.69% | 25.77% | 71/82 |
| CoolProp | Nannoolal, real Tb | 80 | 800 | 3.63% | 3.62% | 6.41% | 4.98% | 36/80 |
| CoolProp | AW, real Tb, estimated criticals | 39 | 390 | 3.81% | 3.10% | 8.96% | 7.49% | 16/39 |
| CoolProp | AW, real criticals | 84 | 840 | 3.64% | 0.31% | 6.62% | 1.28% | 3/84 |

## Coverage

Numbers below are complete curves / original reference curves. Predictions
with missing inputs or model failures are recorded; they are not silently
replaced by another method.

| Method | Perry 2-8 | Perry 2-10 | CoolProp |
|---|---:|---:|---:|
| GRAPPA | 305/305 | 838/838 | 84/84 |
| Nannoolal | 299/305 | 812/838 | 82/84 |
| Nannoolal, real Tb | 298/305 | 807/838 | 80/84 |
| AW, real Tb, estimated criticals | 185/305 | 758/838 | 39/84 |
| AW, real criticals | 305/305 | 202/838 | 84/84 |

Estimated AW Tc falls below one or more sampled temperatures for **103 Perry
2-8 curves and 41 CoolProp curves**. Such points have no physical liquid-vapor
saturation pressure for that estimated model; they are marked unavailable,
without evaluating an extension above Tc or clamping the answer to Pc. This
accounts for much of the reduced AW-estimated-critical complete-curve coverage.
Perry 2-10's principal limitation for real-critical AW is missing reference
Tc/Pc/omega for 636 compounds.

## Applicability and numerical details

- Source 1-atm points remain in the original sample when present, including
  Perry 2-10's 760-mmHg rows. The supplied-Tb methods therefore receive the
  requested extra information, and may be evaluated on that anchor itself.
- Nannoolal is evaluated on the full original grids, which include points
  beyond its stated approximate upper applicability limit of Tr=0.75--0.8.
  On the matched **Tr <= 0.8** points, median compound errors are:

| Method | Perry 2-8: 185 compounds / 1,296 points | Perry 2-10: 187 compounds / 1,491 points | CoolProp: 38 compounds / 219 points |
|---|---:|---:|---:|
| GRAPPA | 6.09% | 5.53% | 4.68% |
| Nannoolal | 18.78% | 22.29% | 18.57% |
| Nannoolal, real Tb | 5.35% | 4.18% | 3.01% |
| AW, real Tb, estimated criticals | 6.66% | 5.48% | 2.21% |
| AW, real criticals | 4.41% | 5.98% | 0.45% |

  The temperature subset uses the available baseline reference Tc and retains
  original points; it is not resampled.
- The existing Nannoolal mono-alcohol correction is nonzero at supplied Tb.
  Source-Tb pressure residuals were nonzero for 25 Perry 2-8, 37 Perry 2-10,
  and two CoolProp real-Tb cases; maxima were 9.79%, 9.75%, and 9.80%,
  respectively, with methanol the largest in each population. This documented
  method behavior was retained. Supplying real Tb is not equivalent to exactly
  reanchoring the corrected alcohol curve at 1 atm.
- AW's quadratic omega inversion suffers mild cancellation near Tr=0.7.
  The benchmark accepts a relative anchor-pressure residual up to `1e-8`;
  the largest observed successful residual was `1.139e-10`. This avoids
  misclassifying negligible floating-point residuals as unavailable models.
- Known reference disagreements, including octanal's Perry 2-10 low-pressure
  values, remain in every method's results. Training-data overlap for GRAPPA
  and calibration overlap for local Nannoolal extensions remain possible.

## Artifacts and reproduction

Final artifacts are in `/tmp/grappa-method-comparison-sKYQTcYk/final-results/`:
`summary.json`, `points.csv`, `compounds.csv`, `model_parameters.csv`, and the
generated `report.md`. The surrounding directory contains `final-run.log` and
`final-verification.log`. The earlier `results/` directory is an intermediate run,
not the final reported comparison.

The comparison took **17.21 seconds excluding imports and output writing**.
It generated **57,140 method/point records** from the original **11,428 points**.
GRAPPA was not recomputed. Methods and anchors are cached per structure/source
configuration to avoid repeating estimation for each temperature.

Independent verification checked all 57,140 records for exact preservation of
the baseline point identities, temperatures, pressures, and GRAPPA predictions;
recomputed every valid percentage error, every common-complete median/mean,
and every pairwise median/count; checked AW domain admission and blank failed
predictions; and reproduced 36 classical-method predictions, including an
independent expansion of the AW equation. All checks passed, as did Ruff and
Python compilation. Baseline and implementation hashes match the executed run.

An initial separate verifier stalled inside CoolProp initialization after a
different import order. It was stopped; the replacement verifier avoids the
unnecessary reference-provider/PyTorch imports and checks the saved artifacts
in **2.36 seconds excluding imports**. The actual benchmark completed normally
and its results were not changed by that verification issue.

```bash
set -o pipefail
/tmp/grappa-rlpws2CN/venv/bin/python -u \
  scripts/psat_experiments/benchmark_grappa_methods.py \
  --baseline-dir /tmp/grappa-reference-benchmark-Q2LXFd6o/results \
  --output-dir /tmp/grappa-method-repeat-results \
  2>&1 | tee /tmp/grappa-method-repeat.log
```

Run from the repository root and use a new output directory. Summary metadata
includes baseline, implementation, and source hashes. Reference files must
match the original baseline; the script rejects changes rather than mixing
different source versions.
