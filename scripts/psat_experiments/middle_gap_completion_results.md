# Middle-Gap Vapor-Pressure Completion

## Scope

This experiment simulates missing interior intervals between two retained hard
vapor-pressure segments.

Each generated gap has exact hard-source pressure and `dln(P)/dT` at both
boundaries. Candidate relations are evaluated only inside the removed interval.

The benchmark uses:

- `345` Perry 9th Table 2-8 correlations with analytic derivatives.
- `114` CoolProp HEOS fluids represented by dense log-pressure PCHIP caches.
- Low-, middle-, and high-pressure gap locations.
- Gap widths equal to `10%`, `25%`, or `50%` of the usable log-pressure range.
- A usable reference range from approximately `0.35 Tc` through `0.92 Tc`.
- `101` samples uniformly spaced in reference `ln(P)` inside each gap.

The complete run contains:

```text
4,131 gap cases
37,179 method evaluations
0 numerical failures
0 nonmonotonic generated curves
```

The detailed CSV is written to:

```text
/tmp/middle_gap_completion_benchmark.csv
```

## Candidate Methods

### Linear log pressure

`ln(P)` is linear in temperature between the two pressure anchors. This is a
value-only baseline and does not reproduce the hard-source slopes.

### Direct cubic Hermite bridges

Cubic Hermite interpolation of `ln(P)` is performed in:

- Temperature, `T`.
- Inverse temperature, `1/T`.
- Logarithmic temperature, `ln(T)`.

For the logarithmic-temperature form:

```text
x = ln(T)
dy/dx = T dln(P)/dT
```

The cubic therefore reproduces pressure and temperature derivative exactly at
both hard boundaries.

### Physical-omega Ambrose-Walton

Ordinary physical-omega AW is evaluated without boundary correction as a weak
baseline.

The corrected variant applies a cubic Hermite residual in temperature so that
AW reproduces both boundary values and slopes.

### Effective-omega interpolation

Effective omega is inverted from pressure at both boundaries.

The experiment tests:

- Omega linear in temperature, matching boundary values only.
- Cubic-Hermite omega using slope-inferred `domega/dT` at both boundaries,
  producing an exactly C1 AW relation.
- A smooth blend between two one-sided, slope-matched dynamic-omega AW
  continuations.

## Aggregate Results

Per-gap curve MARD:

| Method | Median | p95 | Worst curve | Worst point |
|---|---:|---:|---:|---:|
| Linear `ln(P)` in `T` | `10.136%` | `50.092%` | `74.629%` | `93.095%` |
| Cubic Hermite in `T` | `0.075%` | `6.761%` | `30.506%` | `69.019%` |
| Cubic Hermite in `1/T` | `0.003%` | `1.244%` | `9.362%` | `18.262%` |
| **Cubic Hermite in `ln(T)`** | **`0.003%`** | **`0.162%`** | **`2.540%`** | **`5.131%`** |
| Physical-omega AW | `2.712%` | `50.119%` | `479.198%` | `1344.763%` |
| Physical AW plus C1 correction | `0.005%` | `0.510%` | `13.034%` | `28.308%` |
| Linear endpoint-effective omega | `0.099%` | `2.241%` | `26.735%` | `42.041%` |
| Cubic endpoint-effective omega C1 | `0.001%` | `0.404%` | `5.017%` | `9.990%` |
| Two-sided dynamic-omega blend | `0.077%` | `2.065%` | `16.880%` | `36.194%` |

The cubic effective-omega route has the smallest median, but the direct
log-temperature Hermite bridge has substantially better p95, worst-curve, and
worst-point robustness.

## Results by Reference Family

| Reference | Log-T Hermite median | p95 | Worst curve |
|---|---:|---:|---:|
| Perry 2-8 | `0.003%` | `0.258%` | `2.540%` |
| CoolProp HEOS | `0.002%` | `0.090%` | `1.401%` |

The same method wins on robustness for both analytic Perry boundaries and
interpolated CoolProp boundaries.

## Results by Synthetic Gap Width

### Perry 2-8

| Width fraction | Median | p95 | Worst curve |
|---:|---:|---:|---:|
| `0.10` | `<0.001%` | `<0.001%` | `0.002%` |
| `0.25` | `0.003%` | `0.023%` | `0.078%` |
| `0.50` | `0.045%` | `0.797%` | `2.540%` |

### CoolProp HEOS

| Width fraction | Median | p95 | Worst curve |
|---:|---:|---:|---:|
| `0.10` | `<0.001%` | `<0.001%` | `0.002%` |
| `0.25` | `0.002%` | `0.014%` | `0.091%` |
| `0.50` | `0.031%` | `0.210%` | `1.401%` |

The `50%` gaps are deliberately severe and commonly span several pressure
decades. Real middle gaps between accepted overlapping data sources should
usually be much narrower.

## Results by Pressure Span

Log-temperature Hermite median/p95 MARD:

| Pressure span | Perry 2-8 | CoolProp HEOS |
|---:|---:|---:|
| `<0.5` decade | `<0.001% / 0.001%` | `<0.001% / 0.001%` |
| `0.5-1` decade | `<0.001% / 0.002%` | `<0.001% / 0.014%` |
| `1-2` decades | `0.004% / 0.038%` | `0.004% / 0.036%` |
| `2-4` decades | `0.046% / 0.829%` | `0.044% / 0.194%` |
| `>4` decades | `0.108% / 1.076%` | `0.187% / 0.842%` |

## Recommendation

Use a direct cubic Hermite bridge in logarithmic temperature:

```text
x = ln(T)
y = ln(P/bar)
m = dy/dx = T dln(P)/dT
```

This route:

- Exactly matches both hard pressure values.
- Exactly matches both hard derivatives.
- Requires no `Tc`, `Pc`, `Tb`, or acentric factor.
- Introduces no effective-omega state or property-quality dependency.
- Remained monotonic in every tested case.
- Has materially stronger tails than the AW-based alternatives.

The existing direct-temperature `make_c1_bridge_segment()` should not be used
unchanged for broad middle gaps. It is excellent for narrow handoff smoothing,
but direct-temperature Hermite error grows quickly over multi-decade gaps.

### Validation gates

A production middle bridge should still require:

- Finite pressure and derivative over the complete gap.
- Strictly positive `dln(P)/dT`.
- Exact endpoint value and derivative reproduction.
- No internal pressure reversal.
- A configurable maximum pressure span or reduced confidence for very broad
  gaps.

### Fallback

If the log-temperature bridge fails monotonic or finite probing:

1. Try cubic endpoint-effective-omega AW when admissible `Tc/Pc` are available.
2. Otherwise report the gap as unfillable rather than silently using linear
   log pressure.

The effective-omega route is a reasonable fallback, but its additional
critical-property dependencies and weaker tail robustness do not justify using
it as the primary middle-gap method.

## Limitations

- Perry and CoolProp are internally generated reference families rather than
  independent experimental validation.
- Endpoint derivatives are exact Perry derivatives or dense-PCHIP CoolProp
  derivatives. Sparse, noisy one-sided table derivatives were not included.
- Gaps are normalized over a broad common subcritical range. Actual provider
  gaps may cluster around narrower temperature and pressure intervals.
- The experiment evaluates the assembled relation, not the subsequent global
  canonical coefficient fit.

## Reproduction

From the repository root:

```bash
PYTHONPATH=.:scripts/psat_experiments \
python scripts/psat_experiments/benchmark_middle_gap_completion.py
```

The script uses up to eight forked worker processes. Override this with:

```bash
PSAT_BENCHMARK_WORKERS=4 \
PYTHONPATH=.:scripts/psat_experiments \
python scripts/psat_experiments/benchmark_middle_gap_completion.py
```
