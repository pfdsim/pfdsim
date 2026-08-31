# Direct Hard-Boundary Completion Below 0.25 Bar

## Scope

This experiment compares lower-pressure completion methods when a selected
hard-pinned vapor-pressure curve itself reaches one of:

- `0.25 bar`
- `0.10 bar`
- `0.05 bar`

The hard endpoint supplies exact pressure and `dln(P)/dT`. Predictions are
evaluated at:

```text
0.04, 0.02, 0.01, 0.005, 0.002, and 0.001 bar
```

Points are retained only when both the reference Psat curve and the selected
Hvap source cover the complete integration interval.

No production vapor-pressure code was changed.

## Reference Sets

### Perry

- Psat reference: Perry 9th Table 2-8.
- Hvap source: Perry 9th Table 2-69.
- `307` curves produced at least one evaluated point.
- `5,226` pressure points were evaluated.
- Acids are reported separately because ordinary Hvap/Clapeyron is known to
  omit vapor association.

### CoolProp

- Psat reference: CoolProp HEOS saturation pressure.
- Hvap source: saturated-vapor minus saturated-liquid molar enthalpy.
- Exact diagnostic `deltaZ`: saturated-vapor minus saturated-liquid
  compressibility factor.
- `82` ordinary-fluid curves produced valid round-tripped points.
- `1,287` pressure points were evaluated.
- CoolProp pressure inversions must round-trip through Psat within `1e-6`
  relative error. This excludes helium's nonstandard low-temperature branch.

## Methods

### Dynamic Effective Omega

The hard endpoint pressure determines `omega_h`; its derivative determines
`domega/dT`. Omega is extrapolated linearly in temperature.

This is the same C1 dynamic-omega construction selected for analytic lower-AW
completion.

### Raw Clapeyron

```text
dln(P)/dT = Hvap(T)/(R T^2)
```

This uses the source Hvap with `deltaZ = 1`. It is value-anchored but generally
not C1 at the hard boundary.

### Constant Boundary-Scaled Clapeyron

```text
scale_h = R Th^2 [dln(P)/dT]_hard / Hvap(Th)
dln(P)/dT = scale_h Hvap(T)/(R T^2)
```

This is exactly C1 at the hard boundary but carries the complete boundary
correction through the deep-vacuum tail.

### Pressure-Relaxed Effective Delta Z

The hard endpoint determines:

```text
deltaZ_h = Hvap(Th)/(R Th^2 [dln(P)/dT]_hard)
```

The inferred correction relaxes toward ideal-vapor behavior:

```text
deltaZ(P) = 1 + (deltaZ_h - 1) (P/Ph)^n
```

Two exponents were tested:

- `n = 1/2`
- `n = 1`

The resulting pressure-dependent ODE is integrated directly. Both variants are
C1 at the hard boundary and approach `deltaZ = 1` as pressure tends to zero.

### Exact CoolProp Delta Z

CoolProp Hvap and saturated-state `deltaZ(T)` are integrated together. This is
a diagnostic for the thermodynamic consistency of the benchmark, not a generic
fallback assumption.

## CoolProp Results

Across all endpoint pressures:

| Method | Median absolute error | p95 | Maximum |
|---|---:|---:|---:|
| Dynamic omega | `0.160%` | `4.029%` | `21.361%` |
| Raw Clapeyron | `1.047%` | `3.614%` | `6.412%` |
| Constant scaled Clapeyron | `1.409%` | `7.432%` | `19.099%` |
| Relaxed `deltaZ`, `n=1/2` | `0.290%` | `1.216%` | `2.868%` |
| Relaxed `deltaZ`, `n=1` | `0.192%` | `0.760%` | `3.381%` |
| Exact CoolProp `deltaZ` | `<0.001%` | `0.002%` | `0.007%` |

By hard endpoint:

| Endpoint | Dynamic omega median / p95 | Raw Clapeyron | Relaxed `deltaZ`, `n=1` |
|---:|---:|---:|---:|
| `0.25 bar` | `0.362% / 6.602%` | `2.204% / 4.372%` | `0.472% / 0.995%` |
| `0.10 bar` | `0.139% / 3.228%` | `1.000% / 2.002%` | `0.174% / 0.425%` |
| `0.05 bar` | `0.061% / 1.719%` | `0.503% / 1.109%` | `0.073% / 0.218%` |

For a `0.25 bar` hard endpoint:

| Target | Dynamic omega median / p95 | Raw Clapeyron | Relaxed `deltaZ`, `n=1` |
|---:|---:|---:|---:|
| `0.04 bar` | `0.128% / 1.329%` | `1.747% / 3.436%` | `0.298% / 0.532%` |
| `0.01 bar` | `0.366% / 4.646%` | `2.168% / 4.191%` | `0.484% / 0.917%` |
| `0.005 bar` | `0.497% / 7.340%` | `2.258% / 4.405%` | `0.560% / 1.068%` |
| `0.001 bar` | `0.786% / 15.563%` | `2.456% / 5.753%` | `0.633% / 1.265%` |

CoolProp shows:

- Dynamic omega has the best near-boundary median.
- Raw `deltaZ=1` Clapeyron becomes more tail-robust in deep vacuum but retains
  approximately `1-2.5%` typical bias.
- Constant C1 scaling is generally worse because low-pressure nonideality
  should decay instead of remaining fixed.
- Linear pressure relaxation of inferred `deltaZ` preserves most near-boundary
  accuracy and removes the dynamic-omega deep-tail failures.
- Exact `deltaZ` reproduces the reference essentially exactly, confirming that
  source Hvap itself is not the limiting factor.

## Perry Non-Acid Results

Across all endpoint pressures:

| Method | Median absolute error | p95 |
|---|---:|---:|
| Dynamic omega | `0.816%` | `7.517%` |
| Raw Clapeyron | `0.830%` | `7.735%` |
| Constant scaled Clapeyron | `1.340%` | `7.460%` |
| Relaxed `deltaZ`, `n=1/2` | `0.518%` | `5.000%` |
| Relaxed `deltaZ`, `n=1` | `0.231%` | `5.651%` |

By hard endpoint:

| Endpoint | Dynamic omega median / p95 | Raw Clapeyron | Constant scaled | Relaxed `n=1/2` | Relaxed `n=1` |
|---:|---:|---:|---:|---:|---:|
| `0.25 bar` | `1.757% / 11.277%` | `1.678% / 9.750%` | `3.643% / 9.712%` | `0.960% / 6.757%` | `0.343% / 7.840%` |
| `0.10 bar` | `0.742% / 6.120%` | `0.687% / 7.426%` | `1.173% / 4.324%` | `0.462% / 4.808%` | `0.194% / 5.383%` |
| `0.05 bar` | `0.323% / 3.852%` | `0.311% / 5.772%` | `0.396% / 2.399%` | `0.221% / 3.204%` | `0.124% / 4.118%` |

Perry broadly agrees that boundary-conditioned Clapeyron can improve on
dynamic omega. The best correction shape is less decisive than for CoolProp:

- Linear pressure relaxation gives the best median.
- Square-root pressure relaxation often gives the better p95.
- Constant scaling becomes competitive when the hard curve reaches
  `0.10-0.05 bar`.
- Source inconsistencies leave substantially larger worst cases than the
  internally consistent CoolProp test.

## Acid Results

Across `309` acid points:

| Method | Median absolute error | p95 | Maximum |
|---|---:|---:|---:|
| Dynamic omega | `1.587%` | `8.552%` | `15.521%` |
| Raw Clapeyron | `10.787%` | `272.048%` | `606.785%` |
| Constant scaled Clapeyron | `1.892%` | `12.796%` | `57.330%` |
| Relaxed `deltaZ`, `n=1/2` | `3.288%` | `132.003%` | `301.270%` |
| Relaxed `deltaZ`, `n=1` | `5.838%` | `182.485%` | `414.548%` |

Generic effective-`deltaZ` relaxation does not repair carboxylic-acid
Clapeyron. Their pressure dependence is not represented by a simple
low-density relaxation from one boundary slope.

The previously tested monomer/dimer equilibrium model remains the appropriate
acid-specific path.

## Effective-Omega Behavior

Reference-implied effective omega was inverted continuously from each Psat
curve between `0.25` and `0.001 bar`.

There is no universal direction as pressure falls:

| Reference set | Increasing | Decreasing | Nonmonotonic |
|---|---:|---:|---:|
| CoolProp | `44` | `24` | `10` |
| Perry non-acids | `117` | `119` | `54` |
| Perry acids | `10` | `8` | `1` |

From a `0.25 bar` anchor to `0.001 bar`:

| Reference set | Median `|delta omega|` | p95 | Median error from local linear omega | p95 |
|---|---:|---:|---:|---:|
| CoolProp | `0.00335` | `0.05406` | `0.00067` | `0.02469` |
| Perry non-acids | `0.00802` | `0.06876` | `0.00547` | `0.01899` |

From a `0.05 bar` anchor to `0.001 bar`:

| Reference set | Median `|delta omega|` | p95 | Median error from local linear omega | p95 |
|---|---:|---:|---:|---:|
| CoolProp | `0.00217` | `0.03529` | `0.00035` | `0.00833` |
| Perry non-acids | `0.00561` | `0.03977` | `0.00244` | `0.00687` |

The median magnitude of `domega/dT` does not decay over this pressure range;
it remains roughly constant and often grows slightly. The asymptotic
constant-omega regime expected from low-temperature `1/T` behavior is not
reached before `0.001 bar`.

Neither the sign of `delta omega` nor the sign of the curvature has a useful
global correlation with physical omega or reduced temperature. Higher omega
and higher reduced temperature correlate moderately with larger curvature
magnitude, but not its direction.

Consequently:

- Linear effective omega is a strong local continuation.
- Its error becomes increasingly tail-dominated over long pressure spans.
- There is no defensible universal rule that omega must increase, decrease, or
  relax toward physical omega.
- A generic imposed flattening is not supported over `0.25-0.001 bar`.

## Conclusions

Clapeyron does offer an advantage over indefinite effective-omega
extrapolation, but not as raw `deltaZ=1` integration alone.

For ordinary fluids with a derivative-bearing hard endpoint:

1. Infer effective `deltaZ_h` from hard slope and reliable Hvap.
2. Relax that correction toward `deltaZ=1` as pressure decreases.
3. Prefer linear pressure relaxation as the general physical baseline.
4. Retain dynamic omega as a strong near-boundary alternative.

For acids:

- Do not use generic raw or relaxed-`deltaZ` Clapeyron.
- Continue with the calibrated dimer-equilibrium model.

For actual volume-backed `deltaZ(T)`:

- Direct Clapeyron is effectively exact and should take precedence.

## Reproduction

From the repository root:

```bash
PYTHONPATH=.:scripts/psat_experiments \
python scripts/psat_experiments/benchmark_direct_lower_completion.py
```

Detailed rows are written to:

```text
/tmp/direct_lower_completion_benchmark.csv
```
