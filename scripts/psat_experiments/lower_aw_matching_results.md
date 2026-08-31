# Lower Ambrose–Walton Hard-Endpoint Matching

## Status

This document records the lower-completion variant study implemented in
`benchmark_lower_aw_matching.py`.

The study asks:

> Given a selected hard-pinned vapor-pressure segment with a lower endpoint
> above `0.25 bar`, how should an Ambrose–Walton-like relation be attached and
> continued downward to the `0.25 bar` handoff?

The results in this document combine:

- A broad 44-method run over 2,421 reference cases.
- A focused follow-up covering four additional `Tb`-localized C¹ variants and
  the unresolved `1–5 bar` hard-endpoint range.
- A targeted switch-slope diagnostic for the candidate winners.

No production vapor-pressure code was changed by this experiment.

## Executive Summary

The best route depends on whether the hard segment's lower endpoint lies below
or above a trustworthy normal boiling point.

### Hard endpoint at or below `Tb`

The strongest Perry-internal method is to:

1. Invert Ambrose–Walton at the hard endpoint to obtain an effective endpoint
   omega.
2. Infer the local derivative of effective omega from the hard segment's
   `dln(P)/dT`.
3. Continue that effective omega linearly in temperature down to the
   `0.25 bar` crossing.

This construction is exactly C¹ at the hard endpoint and does not require a
reported physical acentric factor.

With noisier one-sided sparse-table derivatives, an additive affine C¹
correction around physical-omega or endpoint-to-`0.7Tc` AW is marginally more
robust. The direct dynamic-omega method remains close and stays below
approximately `1%` p95 MARD through a `0.5 bar` hard endpoint.

### Hard endpoint above `Tb`

Do not extrapolate the hard endpoint's slope all the way to `0.25 bar`.

Instead:

1. Invert effective omega at the hard endpoint.
2. Invert effective omega at trusted `Tb`.
3. Make omega linear between those two points and continue that line below
   `Tb`.
4. Apply any remaining C¹ slope correction only between the hard endpoint and
   `Tb`.
5. Make the correction vanish in value and slope at `Tb`.

This route produced p95 MARD from `0.68%` at a `1.25 bar` hard endpoint to
`1.22%` at a `5 bar` hard endpoint, with no nonmonotonic curves.

### Important negative result

Forcing the lower C¹ correction to disappear at `0.25 bar` is generally the
wrong boundary condition. It bends the relation back toward a weaker baseline
and discards useful hard-segment curvature.

The correction should:

- Remain active through the `0.25 bar` crossing when the hard endpoint is at or
  below `Tb`.
- Relax by `Tb` when the hard endpoint is above `Tb`.

## Benchmark Design

### Reference sets

#### Perry 2-8 continuous correlations

- `334` curves had usable `0.25 bar`, normal-boiling, and hard-boundary
  crossings.
- Hard-boundary pressures:
  - `0.35 bar`
  - `0.50 bar`
  - `0.75 bar`
  - `1.01325 bar`
  - `1.25 bar`
  - `1.50 bar`
  - `2 bar`
  - `3 bar`
  - `5 bar`
- Boundary derivatives were analytic Perry derivatives.
- These tests measure internal curve-reconstruction performance and isolate the
  mathematical continuation behavior.

#### Perry 2-10 sparse tables

- `158` table curves supported `0.35 bar` and `0.50 bar` tests.
- `119` supported the `1.01325 bar` test.
- The complete table PCHIP was used as the lower-range reference.
- Two boundary-slope cases were tested:
  - Full-PCHIP oracle derivative.
  - Realistic upper-only PCHIP derivative using only the simulated retained
    hard segment.

The upper-only derivative case is important because it tests sensitivity to a
noisier one-sided slope at the lower edge of a sparse hard segment.

#### Acids

- `21` Perry 2-8 acid curves had a usable `0.25 bar` crossing.
- `105` acid cases were evaluated over the original five boundary pressures.
- The acid subset did not show a reason to exclude AW lower completion.

### Evaluation interval

Each candidate was evaluated from the reference `0.25 bar` temperature to the
hard-boundary temperature.

Samples were spaced uniformly in `ln(P)`, not uniformly in temperature. This
weights the pressure interval more naturally and resembles the intended
pressure-recursive canonical fitting grid.

### Metrics

The benchmark records:

- Per-curve MARD.
- Per-curve maximum absolute relative pressure error.
- Error in the predicted `0.25 bar` crossing temperature.
- Error in `dln(P)/dT` at the generated `0.25 bar` crossing.
- Hard-endpoint value and slope residuals.
- Effective omega at the endpoint and switch.
- Nonmonotonicity.
- Missing crossings, numerical failures, and divergence.

## Methods Tested

### AW baselines

Five practical baseline omega constructions were tested:

1. **Physical omega**
   - Ordinary AW using the reported acentric factor.

2. **Constant endpoint omega**
   - Invert AW at the hard endpoint and hold that effective omega constant.

3. **Endpoint-to-`0.7Tc` omega**
   - Invert omega at the hard endpoint.
   - Connect it linearly to physical omega at `Tr = 0.7`.
   - Extrapolate the line downward.

4. **Endpoint-to-`Tb` omega**
   - Invert omega at the hard endpoint and at trusted `Tb`.
   - Connect those values linearly and extrapolate downward.

5. **Existing `Tb`-variable omega**
   - Use the previously benchmarked line between effective `omega(Tb)` and
     physical omega at `Tr = 0.7`.

### Additive boundary corrections

Each baseline was tested with:

- No correction.
- Constant value shift.
- Affine C¹ correction in `T`.
- Affine C¹ correction in `1/T`.
- C¹ correction relaxed at `0.25 bar` using:
  - Cubic temperature-coordinate Hermite form.
  - Quintic temperature-coordinate Hermite form.
  - Cubic inverse-temperature coordinate.
  - Cubic logarithmic-temperature coordinate.

### Direct dynamic-omega continuations

The hard endpoint value determines `omega_h` through AW inversion.

The endpoint slope determines the required effective-omega derivative:

```text
y(T, omega) = ln(Psat)

omega'_h =
    [y'_hard - (partial y / partial T)_omega]
    / (partial y / partial omega)
```

The following continuations were tested:

- Omega linear in `T`.
- Omega linear in `1/T`.
- Omega linear in `ln(T)`.
- Cubic-Hermite omega joining the endpoint to physical omega at `Tr = 0.7`.

### `Tb`-localized C¹ corrections

The focused follow-up added four corrections for hard endpoints above `Tb`.

They:

- Use the endpoint-to-`Tb` effective-omega baseline.
- Match the hard endpoint's value and slope.
- Vanish in value and slope at `Tb`.
- Remain identically zero below `Tb`.

Temperature-cubic, temperature-quintic, inverse-temperature-cubic, and
log-temperature-cubic forms were compared.

The coordinate choice had negligible influence. The temperature-quintic form
was slightly preferable and supplies a smooth second-derivative relaxation.

## Results

### Hard Endpoint at or Below `Tb`

#### Perry 2-8 continuous reference

The direct temperature-linear dynamic-omega route was best overall.

| Hard endpoint | Median MARD | p95 MARD | Worst curve MARD | Nonmonotonic |
|---:|---:|---:|---:|---:|
| `0.35 bar` | `0.006%` | `0.027%` | `0.153%` | `0` |
| `0.50 bar` | `0.029%` | `0.126%` | `0.708%` | `0` |
| `0.75 bar` | `0.076%` | `0.344%` | `1.959%` | `0` |
| `1.01325 bar` | `0.137%` | `0.601%` | `3.394%` | `0` |

The endpoint-to-`0.7Tc` baseline with an affine inverse-temperature C¹
correction was a close second:

| Hard endpoint | Median MARD | p95 MARD |
|---:|---:|---:|
| `0.35 bar` | `0.008%` | `0.036%` |
| `0.50 bar` | `0.034%` | `0.154%` |
| `0.75 bar` | `0.089%` | `0.377%` |
| `1.01325 bar` | `0.152%` | `0.655%` |

#### Effective-omega movement

For the direct temperature-linear dynamic-omega route:

| Hard endpoint | Median `|Δomega|` to switch | p95 `|Δomega|` |
|---:|---:|---:|
| `0.35 bar` | `0.001` | `0.007` |
| `0.50 bar` | `0.002` | `0.016` |
| `0.75 bar` | `0.003` | `0.027` |
| `1.01325 bar` | `0.004` | `0.036` |

The strong performance therefore does not depend on extreme effective-omega
excursions.

#### Sparse Perry 2-10 reference

For realistic upper-only PCHIP boundary slopes:

| Hard endpoint | Best robust construction | Median MARD | p95 MARD |
|---:|---|---:|---:|
| `0.35 bar` | Endpoint-to-`0.7Tc` + affine C¹ in `T` | `0.085%` | `0.366%` |
| `0.50 bar` | Physical AW + affine C¹ in `1/T` | `0.237%` | `0.846%` |

The direct temperature-linear dynamic-omega route remained close:

| Hard endpoint | Median MARD | p95 MARD |
|---:|---:|---:|
| `0.35 bar` | `0.086%` | `0.435%` |
| `0.50 bar` | `0.211%` | `0.978%` |

This indicates that direct omega-slope continuation is strongest when the hard
derivative is trustworthy, while additive C¹ correction is slightly more
robust to a noisy one-sided derivative.

#### Acid subset

For the direct temperature-linear dynamic-omega route:

| Hard endpoint | Median MARD | p95 MARD |
|---:|---:|---:|
| `0.35 bar` | `0.010%` | `0.022%` |
| `0.50 bar` | `0.044%` | `0.101%` |
| `0.75 bar` | `0.108%` | `0.284%` |
| `1.01325 bar` | `0.170%` | `0.502%` |

No acid-specific instability appeared in the AW portion of lower completion.
The known acid problem remains the subsequent uncorrected Hvap/Clapeyron
region, not this AW continuation.

### Hard Endpoint Above `Tb`

The best construction was the endpoint-to-`Tb` effective-omega baseline with a
C¹ correction localized to the hard-endpoint-to-`Tb` interval.

| Hard endpoint | Median MARD | p95 MARD | Worst curve MARD | Nonmonotonic |
|---:|---:|---:|---:|---:|
| `1.25 bar` | `0.152%` | `0.683%` | `3.696%` | `0` |
| `1.50 bar` | `0.164%` | `0.738%` | `3.926%` | `0` |
| `2 bar` | `0.178%` | `0.848%` | `5.410%` | `0` |
| `3 bar` | `0.200%` | `0.957%` | `4.655%` | `0` |
| `5 bar` | `0.247%` | `1.220%` | `5.043%` | `0` |

The value-only endpoint-to-`Tb` baseline was already strong. Localized C¹
matching produced a small but consistent improvement without allowing the hard
endpoint slope to distort the entire lower interval.

By contrast, direct temperature-linear omega-slope extrapolation degraded as
the hard endpoint moved upward:

| Hard endpoint | Dynamic-omega p95 MARD |
|---:|---:|
| `1.25 bar` | `0.882%` |
| `1.50 bar` | `1.190%` |
| `2 bar` | `1.801%` |
| `3 bar` | `3.052%` |
| `5 bar` | `5.779%` |

### Switch-Temperature and Switch-Slope Behavior

#### Candidate switch-temperature errors

For the winning Perry-continuous routes:

- Median absolute `0.25 bar` temperature error remained below approximately
  `0.3 K` through a `5 bar` hard endpoint.
- p95 remained below approximately `1.7 K` through `5 bar`.

#### Candidate switch-slope errors

The direct dynamic-omega route, for endpoints at or below `Tb`:

| Hard endpoint | Median switch-slope error | p95 |
|---:|---:|---:|
| `0.35 bar` | `0.114%` | `0.480%` |
| `0.50 bar` | `0.258%` | `1.108%` |
| `0.75 bar` | `0.435%` | `1.936%` |
| `1.01325 bar` | `0.586%` | `2.631%` |

The `Tb`-localized route, for endpoints above `Tb`:

| Hard endpoint | Median switch-slope error | p95 |
|---:|---:|---:|
| `1.25 bar` | `0.648%` | `2.933%` |
| `1.50 bar` | `0.717%` | `3.266%` |
| `2 bar` | `0.816%` | `3.762%` |
| `3 bar` | `0.997%` | `4.489%` |
| `5 bar` | `1.171%` | `5.841%` |

For upper-only Perry 2-10 slopes, candidate switch-slope p95 errors were
approximately `4.5–7.1%`.

These are small enough to provide a useful incoming derivative for a calibrated
Clapeyron relation, including the proposed monoacid dimer correction.

### Value-Only Boundary Results

When the hard endpoint has no trustworthy derivative:

- Below `Tb`, the endpoint-to-`Tb` effective-omega line was the strongest
  Perry-continuous value-only method:
  - `0.35 bar`: p95 MARD `0.176%`.
  - `0.50 bar`: p95 MARD `0.338%`.
  - `0.75 bar`: p95 MARD `0.504%`.
- At a boundary essentially equal to `Tb`, endpoint-to-`0.7Tc` omega was
  preferable:
  - `1.01325 bar`: p95 MARD `1.160%`.
- Above `Tb`, endpoint-to-`Tb` omega remained the preferred value-only route:
  - `2 bar`: p95 MARD `0.861%`.

Sparse-table value-only tails were weaker:

- Approximately `0.95%` p95 at a `0.35 bar` endpoint.
- Approximately `2.67%` p95 at a `0.50 bar` endpoint.
- Approximately `4.51%` p95 at a `1 atm` endpoint.

Value-only lower completion is usable, but it should receive lower quality than
a derivative-bearing hard-boundary completion.

### Rejected or Inferior Constructions

#### Relaxing the correction at `0.25 bar`

This was generally inferior to retaining the hard-boundary curvature through
the switch.

The relaxed correction must bend back to the baseline before the handoff. That
artificial curvature often costs more accuracy than it gains in formal
baseline recovery.

#### Cubic-Hermite omega to `Tr = 0.7`

The cubic omega relation was defined between the hard endpoint and `Tr = 0.7`
but had to be extrapolated below the hard endpoint.

It produced:

- Missing crossings.
- Nonmonotonic curves.
- Severe effective-omega excursions.
- Numerical divergence.

It should not be used.

#### Constant endpoint omega

AW inversion at the hard endpoint fixes value but not curvature. Holding that
omega constant was consistently weaker than using either:

- The hard endpoint derivative.
- A second value anchor at `Tb`.
- Physical omega at `Tr = 0.7`.

#### Long-range local-slope extrapolation

The local hard-boundary slope is highly predictive over a short lower interval,
but becomes progressively less suitable when carried from several bar down to
`0.25 bar`.

For endpoints above `Tb`, slope information should be localized to the
hard-endpoint-to-`Tb` interval.

#### Plain physical AW

Physical omega alone does not incorporate the accepted hard segment's local
value or curvature and remained substantially weaker than the matched methods.

## Recommended Production Policy

### Eligibility

The lower AW completion relation should require:

- A selected hard-pinned lower boundary.
- Admissible `Tc` and `Pc`.
- Hard-boundary pressure greater than the lower switch pressure.
- A terminal lower gap that reaches the configured `0.25 bar` switch.

A physical acentric factor is useful but not mandatory.

Trusted `Tb` is strongly preferred, especially when the hard endpoint is above
normal boiling.

### Route A: Hard endpoint at or below trusted `Tb`

Use the direct endpoint-effective dynamic-omega route when the hard derivative
is trustworthy.

Let:

```text
Th = hard endpoint temperature
yh = ln(Ph)
sh = hard dln(P)/dT
```

Solve:

```text
AW(Th, omega_h) = yh
```

Then:

```text
omega'_h =
    [sh - (partial AW / partial T)(Th, omega_h)]
    / (partial AW / partial omega)(Th, omega_h)
```

Continue:

```text
omega(T) = omega_h + omega'_h (T - Th)
ln(P) = AW(T, omega(T))
```

Trim the resulting monotonic relation at its `0.25 bar` crossing.

Do not force `omega(T)` or an additive correction back to ordinary AW at the
switch.

#### Derivative-uncertainty alternative

When the hard derivative is numerically estimated or known to be noisy:

1. Use endpoint-to-`0.7Tc` effective omega when physical omega is admissible.
2. Otherwise use ordinary physical-omega AW.
3. Apply an affine C¹ additive correction in `1/T` or `T`.
4. Retain that correction through the `0.25 bar` crossing.

Inverse-temperature affine correction is a reasonable general default:

```text
q(T) = delta_y + M (1/T - 1/Th)
M = -delta_s Th^2
```

where `delta_y` and `delta_s` are the hard-boundary value and slope residuals
against the selected AW baseline.

### Route B: Hard endpoint above trusted `Tb`

Solve effective omega at both value anchors:

```text
AW(Th, omega_h) = ln(Ph)
AW(Tb, omega_b) = ln(1.01325 bar)
```

Define the baseline omega line:

```text
omega_base(T) =
    omega_b
    + (omega_h - omega_b) (T - Tb) / (Th - Tb)
```

Use this line through and below `Tb`.

Apply a localized quintic correction `q(T)` only on `[Tb, Th]` such that:

```text
q(Tb) = 0
q'(Tb) = 0
q''(Tb) = 0

q(Th) = hard value residual
q'(Th) = hard slope residual
q''(Th) = 0
```

Below `Tb`:

```text
q(T) = 0
```

This preserves:

- Exact hard-boundary value and slope.
- Exact normal-boiling pressure.
- A smooth return to the value-anchored effective-omega baseline by `Tb`.

### Route C: Missing hard derivative

Use value anchors only:

- If `Th < Tb`, use the endpoint-to-`Tb` effective-omega line.
- If `Th` is effectively `Tb`, use endpoint-to-`0.7Tc` omega when physical
  omega is available.
- If `Th > Tb`, use the endpoint-to-`Tb` effective-omega line.

Assign lower quality than the derivative-bearing routes.

### Route D: Missing trusted `Tb`

If physical omega or hard `Psat(0.7Tc)` is available:

- Prefer endpoint-to-`0.7Tc` AW with additive C¹ correction.

If neither `Tb` nor physical omega is available:

- The endpoint-inverted, slope-derived dynamic-omega route remains usable.
- Penalize quality increasingly with the pressure span from the hard endpoint
  to `0.25 bar`.
- Reject numerical divergence, missing crossings, or nonmonotonicity.

For a hard endpoint several bar above the switch, this route should rank below a
qualified Nannoolal alternative because long-range local-slope extrapolation
was measurably weaker.

### Handoff at `0.25 bar`

The lower AW relation should declare:

```text
P_min_bar = 0.25
```

The completion coordinator can trim it at the exact pressure crossing and pass
that endpoint to the lower-pressure relation.

At the switch:

- Pressure is exact by construction.
- Preserve the incoming AW derivative.
- A calibrated Clapeyron relation may consume that derivative.
- Do not distort the AW relation solely to match an uncalibrated Hvap slope.

For monoacids:

- The incoming derivative can calibrate the dimer-aware Clapeyron relation.
- If that calibration is unavailable, trusted-`Tb` Nannoolal remains the safer
  deep-vacuum fallback.

### Validation Gates

Every generated relation should be probed before acceptance:

- Positive and finite pressure.
- Strictly increasing `ln(P)` with temperature.
- Exactly one descending `0.25 bar` crossing.
- Finite endpoint omega and omega derivative.
- No overflow or extreme pressure excursions.
- Exact hard-boundary value match.
- Exact hard-boundary slope match when C¹ is required.
- Exact `Tb` pressure for the endpoint-to-`Tb` route.
- No discontinuity at the `Tb` correction-release point.

Effective omega is a curve-shape parameter in these routes, not a newly
resolved physical acentric factor. It should remain in generated-segment
metadata and must not overwrite the component's physical omega property.

### Quality and Provenance Guidance

The lower relation should retain:

- Hard-boundary segment source, method, quality, and endpoint.
- `Tc` and `Pc` sources and qualities.
- `Tb` source and quality when used.
- Physical-omega or hard-`Psat(0.7Tc)` source when used.
- Endpoint-inverted effective omega.
- Effective-omega derivative or localized correction coefficients.
- Chosen route and correction coordinate.
- Predicted switch temperature, pressure, and derivative.

A provisional quality basis is:

```text
minimum quality of all consumed anchors
× Ambrose–Walton method factor
× optional derivative-confidence factor
```

The benchmark supports a lower quality for:

- Value-only boundaries.
- One-sided sparse-table derivatives.
- Missing `Tb` over a long pressure span.

It does not support making these routes ineligible merely because a calculated
effective-omega quality falls below the ordinary physical-property admission
threshold. Effective omega here is internal completion state, not a physical
property resolution result.

### Scope Boundary

This policy applies only when at least one hard-pinned segment exists.

It does not decide:

- Full fallback curves with no hard-pinned segments.
- Middle-gap completion between two hard segments.
- The deep-vacuum Hvap/Nannoolal routing below `0.25 bar`.
- Runtime canonical-curve integration.

Those remain separate layers.

## Reproduction

From the repository root:

```bash
PYTHONPATH=.:scripts/psat_experiments \
python scripts/psat_experiments/benchmark_lower_aw_matching.py
```

The detailed CSV is written to:

```text
/tmp/lower_aw_matching_benchmark.csv
```

The benchmark script currently includes all broad and focused variants
described here.
