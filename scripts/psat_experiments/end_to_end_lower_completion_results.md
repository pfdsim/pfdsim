# End-to-End Lower Psat Completion

## Scope

This benchmark evaluates the actual production-path lower completion sequence:

```text
simulated retained hard Psat segment
    -> production anchored lower Ambrose-Walton relation
    -> generated 0.25 bar pressure and derivative
    -> deep-vacuum completion through 0.001 bar
```

Hard-segment lower endpoints are placed at:

```text
1.01325, 0.75, 0.50, and 0.35 bar
```

Deep-vacuum predictions are evaluated at:

```text
0.04, 0.02, 0.01, 0.005, 0.002, and 0.001 bar
```

The production lower-AW implementation is imported from
`property_resolution.vapor_pressure_adapter`. It is not reimplemented inside
the benchmark.

## Reference Sets

### Perry

- Psat reference: Perry 9th Table 2-8.
- Hvap source: Perry 9th Table 2-69.
- Hard derivative: analytic Perry 2-8 derivative.
- Production lower-AW route: endpoint-slope dynamic omega, quality factor
  `0.95`.
- `307` curves, `1,228` hard-boundary cases, and `6,968` target points.

### CoolProp

- Psat reference: CoolProp HEOS saturation pressure.
- Hvap source: saturated-vapor minus saturated-liquid molar enthalpy.
- Diagnostic exact delta Z: saturated-vapor minus saturated-liquid
  compressibility factor.
- Hard derivative: the production segment's one-sided numerical derivative.
- Production lower-AW route: separate omega anchor plus C1 correction, quality
  factor `0.94`.
- Psat/Tsat round trips must agree within `1e-6`; this excludes helium's
  nonstandard branch.
- `82` fluids, `328` hard-boundary cases, and `1,716` target points.

## Runtime Optimization

CoolProp Psat, Hvap, and delta Z are evaluated on a fixed 121-point temperature
grid over the benchmarked pressure interval and represented by log-pressure or
property PCHIP interpolators.

Representative interpolation errors were approximately:

```text
ln(P): below 1e-8 absolute
Hvap: below 2e-9 relative
delta Z: below 5e-9 relative
```

This reduces the complete benchmark from several minutes of repeated adaptive
CoolProp calls to approximately one minute while leaving the reported errors
unchanged at meaningful precision.

## Methods

### Continued Lower AW

The bound production lower-AW evaluator is extrapolated below its declared
`0.25 bar` pressure limit. This measures the cost of using dynamic/effective
omega indefinitely.

### Raw Clapeyron

```text
dln(P)/dT = Hvap(T)/(R T^2)
```

The generated lower-AW crossing supplies exactly `P = 0.25 bar`.

### Boundary-Inferred Relaxed Delta Z

The generated lower-AW derivative determines:

```text
deltaZ_h =
    Hvap(Th)
    / [R Th^2 (dln(P)/dT)_AW]
```

The correction relaxes toward ideal-vapor behavior:

```text
deltaZ(P)
    = 1 + (deltaZ_h - 1) (P/0.25 bar)^n
```

Both:

```text
n = 1/2
n = 1
```

are integrated directly as pressure-dependent ODEs.

### One-Point Watson Variants

Only one source-backed `Hvap(Tb)` value is retained:

```text
Hvap(T)
    = Hvap(Tb)
      [(1 - T/Tc)/(1 - Tb/Tc)]^0.38
```

Raw, square-root relaxed-delta-Z, and linear relaxed-delta-Z forms are tested
from `0.25 bar`.

### Five-Point Watson Variants

A two-parameter Watson relation:

```text
Hvap(T) = A (1 - T/Tc)^n
```

is fitted in logarithmic space to exactly five source Hvap values:

```text
20, 40, 60, 80, and 100 degC
```

All five temperatures must lie inside the source Hvap range. The exponent is
tested both unbounded and clipped to:

```text
0.20 <= n <= 0.60
```

Raw and boundary-relaxed delta-Z forms are evaluated from `0.25 bar`.

### Tb-Local Five-Point Watson

An adaptive local sampling policy is also tested:

```text
preferred:
    Tb, Tb-20, Tb-40, Tb-60, Tb-80 K

if any preferred point is outside the source range:
    Tb, Tb-10, Tb-20, Tb-30, Tb-40 K

otherwise:
    unavailable
```

The same bounded two-parameter Watson fit and raw/square-root/linear
Clapeyron variants are used.

### Exact Delta Z

CoolProp Hvap and saturated-state delta Z are integrated together. Because the
pressure anchor and derivative came through lower AW first, this measures the
remaining lower-AW switch error rather than reproducing CoolProp identically.

### Peng-Robinson Delta Z

Standard pure-fluid Peng-Robinson liquid and vapor roots are evaluated at the
current integrated `(T, P)` state:

```text
deltaZ_PR = Z_vapor - Z_liquid
```

No scaling, additive correction, or pressure relaxation is applied to PR.

Two omega inputs are compared:

- The admissible physical acentric factor.
- The lower-AW effective omega evaluated at the `0.25 bar` switch and then held
  constant inside PR.

PR produced three physical roots for every evaluated saturation state from
`0.25` through `0.001 bar`.

### Calibrated Monoacid Dimer Clapeyron

For the 18 recognized monocarboxylic acids:

```text
2 M <-> D
deltaH_dim = -60.5 kJ/mol dimer
```

The generated lower-AW slope determines the required boundary dimer extent and
therefore an effective compound-specific dimerization entropy. The equilibrium
extent is then solved dynamically while integrating pressure.

## CoolProp Results

Aggregate point errors:

| Method | Median | p95 | Maximum |
|---|---:|---:|---:|
| Continued lower AW | `0.899%` | `13.634%` | `55.264%` |
| Raw Clapeyron | `2.245%` | `4.859%` | `6.400%` |
| Relaxed delta Z, `n=1/2` | `0.527%` | `1.840%` | `8.058%` |
| Relaxed delta Z, `n=1` | `0.624%` | `1.999%` | `4.920%` |
| One-point Watson raw | `3.202%` | `19.884%` | `63.117%` |
| One-point Watson relaxed, `n=1/2` | `1.610%` | `13.575%` | `42.078%` |
| One-point Watson relaxed, `n=1` | `2.008%` | `16.198%` | `51.449%` |
| Exact delta Z | `0.038%` | `0.515%` | `2.105%` |

Per-case curve-MARD:

| Method | Median | p95 | Maximum |
|---|---:|---:|---:|
| Continued lower AW | `0.949%` | `13.290%` | `29.012%` |
| Raw Clapeyron | `2.139%` | `4.802%` | `5.924%` |
| Relaxed delta Z, `n=1/2` | `0.501%` | `1.716%` | `7.172%` |
| Relaxed delta Z, `n=1` | `0.607%` | `1.848%` | `4.668%` |
| Exact delta Z | `0.038%` | `0.476%` | `2.104%` |

### By Hard Endpoint

Point median/p95:

| Hard endpoint | Continued AW | Raw Clapeyron | Relaxed `n=1/2` | Relaxed `n=1` | Exact delta Z |
|---:|---:|---:|---:|---:|---:|
| `1.01325 bar` | `1.493% / 16.370%` | `2.320% / 5.007%` | `0.435% / 2.451%` | `0.773% / 3.213%` | `0.143% / 1.029%` |
| `0.75 bar` | `0.944% / 14.443%` | `2.254% / 5.027%` | `0.456% / 2.151%` | `0.662% / 2.127%` | `0.069% / 0.873%` |
| `0.50 bar` | `0.756% / 11.581%` | `2.214% / 4.576%` | `0.538% / 1.381%` | `0.574% / 1.668%` | `0.026% / 0.338%` |
| `0.35 bar` | `0.579% / 9.698%` | `2.206% / 4.385%` | `0.613% / 1.469%` | `0.508% / 1.296%` | `0.005% / 0.077%` |

The square-root relaxation gives the best CoolProp median and p95 for most
handoffs. Linear relaxation has the smaller aggregate maximum and becomes best
when the hard endpoint is already near the `0.25 bar` switch.

## Perry Ordinary-Fluid Results

Aggregate point errors:

| Method | Median | p95 | Maximum |
|---|---:|---:|---:|
| Continued lower AW | `3.494%` | `20.704%` | `812.145%` |
| Raw Clapeyron | `1.786%` | `9.968%` | `152.764%` |
| Relaxed delta Z, `n=1/2` | `1.347%` | `8.468%` | `194.339%` |
| Relaxed delta Z, `n=1` | `0.755%` | `8.569%` | `175.041%` |
| One-point Watson raw | `4.787%` | `27.425%` | `164.806%` |
| One-point Watson relaxed, `n=1/2` | `3.622%` | `22.560%` | `199.445%` |
| One-point Watson relaxed, `n=1` | `4.008%` | `24.551%` | `183.507%` |

Per-case curve-MARD:

| Method | Median | p95 | Maximum |
|---|---:|---:|---:|
| Continued lower AW | `4.344%` | `17.400%` | `339.517%` |
| Raw Clapeyron | `1.782%` | `9.780%` | `76.152%` |
| Relaxed delta Z, `n=1/2` | `1.353%` | `8.344%` | `101.067%` |
| Relaxed delta Z, `n=1` | `0.801%` | `9.655%` | `90.830%` |

### By Hard Endpoint

Point median/p95:

| Hard endpoint | Continued AW | Raw Clapeyron | Relaxed `n=1/2` | Relaxed `n=1` | One-point Watson raw |
|---:|---:|---:|---:|---:|---:|
| `1.01325 bar` | `5.032% / 27.263%` | `1.973% / 10.180%` | `1.924% / 10.641%` | `1.379% / 10.590%` | `4.987% / 28.180%` |
| `0.75 bar` | `4.071% / 22.490%` | `1.854% / 9.953%` | `1.520% / 8.813%` | `1.042% / 8.441%` | `4.807% / 27.401%` |
| `0.50 bar` | `3.004% / 17.593%` | `1.725% / 9.730%` | `1.189% / 7.307%` | `0.650% / 8.112%` | `4.666% / 26.969%` |
| `0.35 bar` | `2.279% / 14.036%` | `1.686% / 9.757%` | `1.014% / 6.987%` | `0.439% / 7.778%` | `4.631% / 27.018%` |

Linear pressure relaxation consistently gives the best Perry median.
Square-root relaxation is sometimes slightly better in the p95 tail.

## Lower-AW Switch Accuracy

### CoolProp

| Diagnostic | Median | p95 | Maximum |
|---|---:|---:|---:|
| Absolute switch-temperature error | `0.0080 K` | `0.1248 K` | `0.4928 K` |
| Absolute switch-pressure error | `0.0380%` | `0.4752%` | `2.1492%` |
| Absolute switch-slope error | `0.0919%` | `0.8863%` | `3.1057%` |

### Perry

| Diagnostic | Median | p95 | Maximum |
|---|---:|---:|---:|
| Absolute switch-temperature error | `0.0272 K` | `0.3049 K` | `4.6139 K` |
| Absolute switch-pressure error | `0.1113%` | `1.1330%` | `11.3036%` |
| Absolute switch-slope error | `0.2941%` | `1.8659%` | `16.3527%` |

Closer hard endpoints sharply improve all switch diagnostics.

## Target-Pressure Dependence

### CoolProp

Point median/p95:

| Target | Continued AW | Raw Clapeyron | Relaxed `n=1/2` | Relaxed `n=1` |
|---:|---:|---:|---:|---:|
| `0.04 bar` | `0.389% / 4.197%` | `1.795% / 3.801%` | `0.260% / 1.229%` | `0.420% / 1.381%` |
| `0.02 bar` | `0.604% / 6.783%` | `2.061% / 4.376%` | `0.389% / 1.343%` | `0.542% / 1.694%` |
| `0.01 bar` | `0.917% / 10.057%` | `2.216% / 4.896%` | `0.521% / 1.697%` | `0.631% / 1.988%` |
| `0.005 bar` | `1.208% / 14.451%` | `2.315% / 5.120%` | `0.637% / 1.951%` | `0.694% / 2.087%` |
| `0.002 bar` | `1.670% / 23.485%` | `2.417% / 5.471%` | `0.758% / 2.156%` | `0.752% / 2.236%` |
| `0.001 bar` | `1.932% / 29.922%` | `2.525% / 5.523%` | `0.834% / 2.380%` | `0.771% / 2.231%` |

### Perry Ordinary Fluids

| Target | Continued AW | Raw Clapeyron | Relaxed `n=1/2` | Relaxed `n=1` |
|---:|---:|---:|---:|---:|
| `0.04 bar` | `1.389% / 6.906%` | `1.548% / 4.594%` | `0.747% / 4.398%` | `0.514% / 4.223%` |
| `0.02 bar` | `2.219% / 10.509%` | `1.711% / 6.079%` | `1.031% / 5.672%` | `0.601% / 5.378%` |
| `0.01 bar` | `3.340% / 14.434%` | `1.806% / 8.576%` | `1.283% / 7.892%` | `0.727% / 7.497%` |
| `0.005 bar` | `4.705% / 18.789%` | `1.881% / 11.825%` | `1.533% / 9.142%` | `0.853% / 10.699%` |
| `0.002 bar` | `6.830% / 25.818%` | `1.909% / 15.481%` | `1.814% / 13.081%` | `1.036% / 14.935%` |
| `0.001 bar` | `8.781% / 31.547%` | `1.985% / 20.124%` | `2.004% / 17.331%` | `1.178% / 19.568%` |

Dynamic omega is competitive close to the switch, but its tail error grows
rapidly with every vacuum decade. Relaxed-delta-Z Clapeyron remains
comparatively flat. Linear relaxation gives the best median; square-root
relaxation increasingly improves the Perry p95 tail in deep vacuum.

## Sensitivity to the Lower-AW Handoff

Linear relaxed-delta-Z results with a true hard endpoint at `0.25 bar` from the
earlier direct benchmark are compared with the generated lower-AW switch:

| Reference | Direct hard at `0.25 bar` | AW from `0.35 bar` | AW from `1.01325 bar` |
|---|---:|---:|---:|
| CoolProp | `0.472% / 0.995%` | `0.508% / 1.296%` | `0.769% / 3.222%` |
| Perry non-acids | `0.343% / 7.840%` | `0.439% / 7.778%` | `1.379% / 10.590%` |

The result is therefore not completely independent of lower AW:

- A nearby `0.35 bar` hard endpoint produces nearly direct-hard-boundary
  performance.
- A distant `1 atm` endpoint measurably degrades the generated switch.
- Relaxed-delta-Z Clapeyron still retains the same ranking and a substantial
  advantage over indefinite AW continuation.

## Monocarboxylic Acids

Aggregate point errors over 18 monoacids:

| Method | Median | p95 | Maximum |
|---|---:|---:|---:|
| Continued lower AW | `6.012%` | `19.323%` | `31.330%` |
| Raw Clapeyron | `18.342%` | `389.549%` | `606.479%` |
| Relaxed delta Z, `n=1/2` | `5.774%` | `197.663%` | `309.598%` |
| Relaxed delta Z, `n=1` | `11.077%` | `271.819%` | `413.019%` |
| Calibrated dimer Clapeyron | `5.711%` | `14.635%` | `25.306%` |

Per-case curve-MARD:

| Method | Median | p95 | Maximum |
|---|---:|---:|---:|
| Continued lower AW | `6.541%` | `15.315%` | `18.700%` |
| Calibrated dimer Clapeyron | `5.710%` | `11.264%` | `13.610%` |

### By Hard Endpoint

Point median/p95:

| Hard endpoint | Continued AW | Calibrated dimer Clapeyron |
|---:|---:|---:|
| `1.01325 bar` | `8.318% / 23.387%` | `7.122% / 16.615%` |
| `0.75 bar` | `7.277% / 20.420%` | `6.342% / 15.126%` |
| `0.50 bar` | `5.318% / 16.943%` | `5.239% / 13.736%` |
| `0.35 bar` | `4.081% / 14.344%` | `4.440% / 12.593%` |

Boundary-calibrated dimerization consistently improves tail robustness. Generic
raw or relaxed-delta-Z Clapeyron remains unsafe for monoacids.

Malonic and succinic acid were evaluated as other acids but not passed through
the monoacid dimer model. Continued lower AW remained preferable to generic
Clapeyron for those cases.

## One-Point Hvap Result

Switching a lone Watson-scaled `Hvap(Tb)` source at `0.25 bar` is not supported.

It is worse than full-Hvap Clapeyron and usually worse than continued lower AW:

```text
CoolProp p95:
    continued AW = 13.634%
    Watson raw = 19.883%
    Watson relaxed n=1/2 = 13.571%

Perry ordinary p95:
    continued AW = 20.704%
    Watson raw = 27.425%
    Watson relaxed n=1/2 = 22.560%
```

This agrees with the earlier crossover evidence: one-point Watson/Clapeyron
should switch much lower, approximately `20-40 mmHg`, rather than at
`0.25 bar`.

## Five-Point Watson Result

Five-point fits were available for:

```text
CoolProp: 65 fluids, 1,388 target points
Perry ordinary: 252 curves, 5,828 target points
```

Bounded fit quality at the five source temperatures:

| Reference | Median maximum sample error | p95 | Maximum |
|---|---:|---:|---:|
| CoolProp | `0.139%` | `1.494%` | `3.446%` |
| Perry ordinary | `0.090%` | `0.490%` | `2.228%` |

Aggregate point median/p95 over eligible cases:

| Reference and method | Median | p95 |
|---|---:|---:|
| CoolProp continued AW | `0.982%` | `11.542%` |
| CoolProp five-point raw | `3.084%` | `12.790%` |
| CoolProp five-point relaxed `n=1/2` | `1.293%` | `8.454%` |
| CoolProp five-point relaxed `n=1` | `1.131%` | `10.088%` |
| CoolProp full-Hvap relaxed `n=1/2` | `0.541%` | `1.788%` |
| Perry continued AW | `3.585%` | `19.689%` |
| Perry five-point raw | `2.401%` | `16.309%` |
| Perry five-point relaxed `n=1/2` | `1.937%` | `13.555%` |
| Perry five-point relaxed `n=1` | `1.548%` | `14.299%` |
| Perry full-Hvap relaxed `n=1` | `0.812%` | `8.779%` |

### Deep-Pressure Behavior

At `0.001 bar`, median/p95:

| Reference and method | Result |
|---|---:|
| CoolProp continued AW | `2.082% / 22.402%` |
| CoolProp five-point relaxed `n=1/2` | `2.479% / 13.981%` |
| CoolProp five-point relaxed `n=1` | `2.367% / 16.021%` |
| Perry continued AW | `8.877% / 30.091%` |
| Perry five-point relaxed `n=1/2` | `3.408% / 23.849%` |
| Perry five-point relaxed `n=1` | `2.823% / 25.672%` |

The five-point fit is materially useful:

- It is substantially better than one-point Watson.
- It improves Perry median and p95 relative to continued AW.
- For CoolProp it mainly improves tail robustness; AW retains a slightly
  better median.
- It remains clearly weaker than a full temperature-dependent Hvap source.

The exponent bounds matter. Unbounded fitted exponents fell outside
`[0.20, 0.60]` for 20/65 CoolProp and 44/252 Perry curves. For CoolProp,
unbounded raw-Watson p95 rose from `12.790%` to `35.127%`; the bounded fit
should therefore be used.

## Tb-Local Five-Point Watson Result

The adaptive `Tb`-local fit is materially better than fixed `20-100 degC`
sampling.

Coverage:

| Reference | Fixed-temperature fit | Tb-local fit |
|---|---:|---:|
| CoolProp | `65/82` fluids | `80/82` fluids |
| Perry ordinary | `252/287` curves | `285/287` curves |

Most cases used the preferred `20 K` spacing:

```text
CoolProp cases: 284 at 20 K, 36 at 10 K
Perry cases: 1104 at 20 K, 36 at 10 K
```

Bounded fit quality at the five sampled points:

| Reference | Median maximum sample error | p95 | Maximum |
|---|---:|---:|---:|
| CoolProp | `0.129%` | `0.413%` | `3.682%` |
| Perry ordinary | `0.086%` | `0.380%` | `1.424%` |

Aggregate point median/p95:

| Reference and method | Median | p95 |
|---|---:|---:|
| CoolProp continued AW | `0.898%` | `13.865%` |
| CoolProp fixed five-point relaxed `n=1/2` | `1.293%` | `8.454%` |
| CoolProp Tb-local raw | `2.432%` | `9.378%` |
| CoolProp Tb-local relaxed `n=1/2` | `0.491%` | `5.540%` |
| CoolProp Tb-local relaxed `n=1` | `0.882%` | `7.225%` |
| CoolProp full-Hvap relaxed `n=1/2` | `0.531%` | `1.814%` |
| Perry continued AW | `3.525%` | `20.781%` |
| Perry fixed five-point relaxed `n=1` | `1.548%` | `14.299%` |
| Perry Tb-local raw | `2.066%` | `11.748%` |
| Perry Tb-local relaxed `n=1/2` | `1.479%` | `10.456%` |
| Perry Tb-local relaxed `n=1` | `1.224%` | `10.531%` |
| Perry full-Hvap relaxed `n=1` | `0.762%` | `8.703%` |

### Target-Pressure Behavior

Point median/p95:

| Reference | Target | Continued AW | Fixed five-point best | Tb-local `n=1/2` | Tb-local `n=1` |
|---|---:|---:|---:|---:|---:|
| CoolProp | `0.04 bar` | `0.39% / 4.20%` | `0.58% / 2.67%` | `0.30% / 1.60%` | `0.38% / 2.02%` |
| CoolProp | `0.01 bar` | `0.92% / 10.28%` | `1.48% / 6.51%` | `0.50% / 3.66%` | `0.78% / 4.71%` |
| CoolProp | `0.001 bar` | `1.93% / 29.92%` | `2.48% / 13.98%` | `1.10% / 10.18%` | `2.35% / 11.35%` |
| Perry ordinary | `0.04 bar` | `1.40% / 6.94%` | `0.78% / 5.26%` | `0.79% / 4.55%` | `0.58% / 4.51%` |
| Perry ordinary | `0.01 bar` | `3.37% / 14.58%` | `1.43% / 11.74%` | `1.36% / 8.43%` | `1.05% / 8.46%` |
| Perry ordinary | `0.001 bar` | `8.78% / 31.55%` | `2.82% / 25.67%` | `2.84% / 21.14%` | `2.66% / 21.41%` |

The preferred local policy is therefore:

- Use `20 K` spacing whenever all five values exist.
- Fall back to `10 K` spacing only when necessary.
- Use square-root delta-Z relaxation for CoolProp-like tail robustness.
- For Perry-like data, linear relaxation gives the better median while
  square-root relaxation gives the slightly better deep p95.
- Continue to rank a full temperature-dependent Hvap relation above either
  five-point fit.

## Peng-Robinson Delta-Z Result

### Broad/Direct Hvap

Aggregate point median/p95/maximum:

| Reference and method | Median | p95 | Maximum |
|---|---:|---:|---:|
| CoolProp relaxed delta Z, `n=1/2` | `0.527%` | `1.840%` | `8.058%` |
| CoolProp relaxed delta Z, `n=1` | `0.624%` | `1.999%` | `4.920%` |
| CoolProp physical-omega PR delta Z | `0.678%` | `1.746%` | `5.570%` |
| CoolProp exact delta Z | `0.038%` | `0.515%` | `2.105%` |
| Perry ordinary relaxed delta Z, `n=1/2` | `1.347%` | `8.468%` | `194.339%` |
| Perry ordinary relaxed delta Z, `n=1` | `0.755%` | `8.569%` | `175.041%` |
| Perry ordinary physical-omega PR delta Z | `0.579%` | `9.376%` | `148.391%` |

PR gives:

- The best Perry ordinary-fluid median.
- CoolProp p95 performance slightly better than inferred relaxation.
- A slightly worse CoolProp median than inferred relaxation.
- Better Perry maximum robustness than either inferred relaxation exponent.

### Target-Pressure Behavior

Point median/p95:

| Reference | Target | Relaxed best median | PR delta Z |
|---|---:|---:|---:|
| CoolProp | `0.04 bar` | `0.26% / 1.23%` | `0.53% / 1.36%` |
| CoolProp | `0.01 bar` | `0.52% / 1.70%` | `0.69% / 1.71%` |
| CoolProp | `0.001 bar` | `0.77% / 2.23%` | `0.83% / 1.96%` |
| Perry ordinary | `0.04 bar` | `0.51% / 4.22%` | `0.43% / 4.50%` |
| Perry ordinary | `0.01 bar` | `0.73% / 7.50%` | `0.55% / 8.70%` |
| Perry ordinary | `0.001 bar` | `1.18% / 19.57%` | `0.84% / 20.42%` |

PR generally improves median accuracy while inferred square-root relaxation
often retains the better deep Perry p95.

### Switch-Slope Compatibility

Because PR is not boundary-corrected, its Clapeyron slope does not exactly
match the incoming lower-AW slope:

| Reference | Median mismatch | p95 | Maximum |
|---|---:|---:|---:|
| CoolProp ordinary | `0.364%` | `1.005%` | `3.566%` |
| Perry ordinary | `0.466%` | `2.942%` | `23.801%` |
| Perry monoacids | `9.196%` | `40.495%` | `40.878%` |

The ordinary-fluid mismatch is generally small enough for normal junction
assessment. The acid mismatch confirms that cubic-EOS delta Z cannot represent
association.

### Effective-Omega Matching

Replacing physical omega with the lower-AW switch effective omega had
negligible effect:

| Reference | Physical omega median/p95 | Effective omega median/p95 |
|---|---:|---:|
| CoolProp | `0.678% / 1.746%` | `0.678% / 1.747%` |
| Perry ordinary | `0.579% / 9.376%` | `0.577% / 9.363%` |

At the switch, the median absolute delta-Z change from substituting effective
omega was:

```text
CoolProp: 0.000014
Perry ordinary: 0.000032
```

Physical omega should therefore be retained. Effective-omega matching adds no
material value.

### PR with Sparse Hvap

Aggregate point median/p95:

| Reference and Hvap source | PR result |
|---|---:|
| CoolProp one-point Watson | `2.391% / 16.796%` |
| CoolProp fixed five-point Watson | `1.599% / 12.658%` |
| CoolProp Tb-local five-point Watson | `0.886% / 6.575%` |
| Perry ordinary one-point Watson | `4.420% / 26.513%` |
| Perry ordinary fixed five-point Watson | `1.518% / 15.793%` |
| Perry ordinary Tb-local five-point Watson | `0.959% / 10.671%` |

For Tb-local five-point Hvap:

- PR improves Perry median versus inferred relaxation.
- Inferred square-root relaxation retains a slightly better p95.
- CoolProp inferred square-root relaxation remains better in both median and
  p95.
- One-point Watson remains poor regardless of how delta Z is obtained.

### Acids

PR is unsafe for associating acids:

```text
Monoacid PR median/p95:
    16.324% / 383.797%

Calibrated dimer median/p95:
    5.711% / 14.635%
```

Using the lower-AW effective omega does not improve acid PR.

## Outliers and Limitations

The largest Perry ordinary-fluid outliers include:

- 1,3,5-trinitrobenzene.
- 2,4,6-trinitrotoluene.
- Di-isopropylamine.
- cis/trans-2-methylcyclohexanol.

These expose combinations of:

- Lower-AW switch error inherited from a distant hard endpoint.
- Perry 2-8/Perry 2-69 source inconsistency.
- Deep-tail curvature not represented by one boundary correction.

The inferred ordinary-fluid boundary delta Z remained superficially plausible:

```text
Perry ordinary range: 0.9367 to 1.2222
CoolProp range: 0.8514 to 1.0031
```

Therefore, a simple broad delta-Z plausibility gate cannot identify all
outliers.

Other limitations:

- Perry and CoolProp tests are internally consistent/reference-family tests,
  not independent experimental validation.
- Aggregate tables are point-weighted; curve-MARD tables prevent compounds
  with more target points from dominating the interpretation.
- The monoacid set is identified by an explicit CAS list for the experiment.
  Production chemistry classification must use structural functionality.
- Exact CoolProp delta Z is a diagnostic and requires actual saturated-state
  volume information in production.

## Recommended Production Direction

### Ordinary fluids with broad/direct Hvap

Use boundary-conditioned Clapeyron from the generated `0.25 bar` lower-AW
endpoint:

```text
deltaZ_h =
    Hvap(Th)
    / [R Th^2 (dln(P)/dT)_incoming]

deltaZ(P) =
    1 + (deltaZ_h - 1) (P/Ph)
```

Linear pressure relaxation is the preferred initial default because:

- It gives the best Perry median at every hard endpoint.
- Its Perry p95 is close to the square-root variant.
- It has a lower CoolProp maximum than the square-root variant.
- The earlier direct-hard-boundary experiment also favored it.

The square-root form remains a reasonable future tail-robust alternative.

### Actual volume-backed delta Z

Use the actual frozen saturated-state delta-Z relation directly. It clearly
outperforms inferred relaxation.

### Peng-Robinson delta Z

For ordinary non-associating fluids with admissible `Tc`, `Pc`, and physical
omega:

- PR delta Z is a strong physically based alternative to inferred relaxation.
- Use physical omega directly.
- Require the ordinary junction slope assessment to pass.
- Do not rescale or otherwise correct PR.
- Rank actual saturated-volume delta Z above PR.
- Do not use PR delta Z for carboxylic or other strongly associating fluids.

### Monocarboxylic acids

Use boundary-calibrated dimer Clapeyron. Do not apply generic relaxed delta Z.

### Only one reliable Hvap point

Do not switch at `0.25 bar`. Continue the lower-AW relation until a lower
pressure switch around `20-40 mmHg`, then use Watson/Clapeyron if still needed.

### Five reliable Hvap points from 20-100 degC

Use the bounded five-point Watson fit as an intermediate-quality fallback.
At the `0.25 bar` switch:

- Prefer square-root delta-Z relaxation when tail robustness matters.
- Prefer linear relaxation when median accuracy is more important.
- Rank either below a broad/direct Hvap relation and above a lone
  Watson-scaled `Hvap(Tb)` source.

### Five reliable Hvap points centered at Tb

Prefer the adaptive `Tb`-local fit over fixed absolute temperatures:

```text
20 K spacing if possible
10 K spacing otherwise
```

It provides broader coverage, lower fit residuals, and materially better
deep-vacuum tails. Retain the same `[0.20, 0.60]` exponent bounds.

### Missing usable Hvap

Continue lower AW. Nannoolal remains a fallback when AW prerequisites or a
usable incoming derivative are absent.

## Reproduction

From the repository root:

```bash
PYTHONPATH=.:scripts/psat_experiments \
python scripts/psat_experiments/benchmark_end_to_end_lower_completion.py
```

Detailed rows are written to:

```text
/tmp/end_to_end_lower_completion_benchmark.csv
```
