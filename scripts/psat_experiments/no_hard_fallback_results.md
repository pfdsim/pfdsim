# No-Hard-Segment Vapor-Pressure Fallback

## Scope

These experiments evaluate vapor-pressure construction when no hard-pinned
Psat segment is available, but qualified critical constants, physical omega,
and usually a trusted normal boiling point are available.

Two benchmark scripts were added:

```text
benchmark_no_hard_aw_relaxation.py
benchmark_no_hard_clapeyron_switch.py
```

The first compares low-pressure modifications of the previously selected
Tb-variable-omega Ambrose-Walton curve.

The second compares broad/direct-Hvap Clapeyron switch pressures, Peng-Robinson
versus ideal delta Z, and Tb-anchored Nannoolal.

Detailed rows are written to:

```text
/tmp/no_hard_aw_relaxation_benchmark.csv
/tmp/no_hard_clapeyron_switch_benchmark.csv
```

## No-Hard AW Construction

For eligible compounds:

```text
omega(Tb) = AW inversion at Psat(Tb) = 1.01325 bar

omega(T) =
    omega(Tb)
    + [omega_physical - omega(Tb)]
      (T - Tb)/(0.7 Tc - Tb)
```

The same linear effective-omega trend is continued below `Tb`.

The benchmark compares:

- Ordinary physical-omega AW.
- Unmodified Tb-variable omega.
- Omega clamped at its `Tb` value.
- Omega frozen at `0.25`, `0.10`, or `0.05 bar`.
- Linear-omega deviation relaxed toward physical omega.
- Linear-omega deviation relaxed toward a halfway asymptote.

Relaxation weights use the unmodified variable-omega pressure:

```text
r = min(1, Pvariable/Pswitch)

omega_relaxed =
    omega_physical
    + [a + (1-a) r^n]
      [omega_linear - omega_physical]
```

where `a=0` relaxes fully to physical omega and `a=0.5` retains half of the
variable-omega deviation.

## AW Relaxation Results

Per-curve MARD median/p95:

### Perry 2-8

| Method | Median | p95 |
|---|---:|---:|
| Physical AW | `5.953%` | `43.075%` |
| Tb-variable omega | `3.936%` | `16.359%` |
| Freeze omega at `0.10 bar` | **`3.309%`** | `13.767%` |
| Half-relax from `0.25 bar`, `n=0.5` | `3.451%` | `14.440%` |
| Clamp at `omega(Tb)` | `4.480%` | `25.267%` |

### CoolProp HEOS

| Method | Median | p95 |
|---|---:|---:|
| Physical AW | `2.500%` | `11.250%` |
| **Tb-variable omega** | **`0.869%`** | **`5.892%`** |
| Freeze omega at `0.10 bar` | `1.234%` | `7.602%` |
| Half-relax from `0.25 bar`, `n=0.5` | `1.115%` | `7.127%` |
| Clamp at `omega(Tb)` | `1.786%` | `7.498%` |

No relaxation or clamping form wins on both reference families.

The Perry family favors freezing near `0.10 bar`, while the independent
CoolProp family favors leaving variable omega unchanged.

## Opposite Error Direction

The physical-AW and variable-omega errors do often have opposite signs.

For Perry:

| Pressure | Opposite-sign cases |
|---:|---:|
| `0.25 bar` | `66.3%` |
| `0.10 bar` | `64.9%` |
| `0.05 bar` | `62.7%` |
| `0.0133 bar` | `56.7%` |
| `0.001 bar` | `47.9%` |

For CoolProp, the fraction is approximately `46-51%`.

However, physical AW generally has the larger absolute error. A 50/50
log-pressure average therefore worsens the variable-omega result:

```text
Perry at 0.10 bar:
    variable omega median error = 1.46%
    50/50 log-pressure average = 2.00%

CoolProp at 0.10 bar:
    variable omega median error = 0.48%
    50/50 log-pressure average = 0.94%
```

The opposite signs are real but are not sufficiently balanced for a universal
cancellation rule.

## Broad-Hvap Clapeyron Switch

The switch benchmark uses full temperature-dependent Perry 2-69 or CoolProp
Hvap and compares switch pressures:

```text
0.25, 0.10, 0.05, 0.0266645, and 0.0133322 bar
```

Clapeyron is integrated with either:

```text
Peng-Robinson delta Z
ideal delta Z = 1
```

Monoacids are excluded from the ordinary-fluid aggregate.

To avoid availability bias, the following table uses only compounds supporting
every tested switch.

### Perry ordinary fluids

PR-Clapeyron per-curve MARD median/p95:

| Switch | Median | p95 |
|---:|---:|---:|
| **`0.25 bar`** | **`0.917%`** | **`6.135%`** |
| `0.10 bar` | `1.588%` | `8.257%` |
| `0.05 bar` | `2.123%` | `8.567%` |
| `0.0266645 bar` | `2.511%` | `10.672%` |
| `0.0133322 bar` | `2.843%` | `12.581%` |

### CoolProp ordinary fluids

| Switch | Median | p95 |
|---:|---:|---:|
| **`0.25 bar`** | **`0.548%`** | **`2.665%`** |
| `0.10 bar` | `0.566%` | `3.948%` |
| `0.05 bar` | `0.624%` | `5.186%` |
| `0.0266645 bar` | `0.736%` | `6.314%` |
| `0.0133322 bar` | `0.851%` | `7.505%` |

For a broad/direct qualified Hvap relation, `0.25 bar` is the clear preferred
switch in the no-hard case.

## Peng-Robinson Versus Ideal Delta Z

At a `0.25 bar` switch on the same common cases:

| Reference | PR median/p95 | Ideal median/p95 |
|---|---:|---:|
| Perry ordinary | `0.917% / 6.135%` | `1.513% / 6.801%` |
| CoolProp ordinary | `0.548% / 2.665%` | `1.534% / 3.557%` |

PR should remain the preferred ordinary-fluid delta-Z model, with ideal
delta Z retained only as its numerical/property fallback.

## Nannoolal Comparison

Results on compounds where every AW switch and Tb-anchored Nannoolal were
available:

### Perry ordinary

| Route | Curve median | p95 |
|---|---:|---:|
| Tb-variable omega | `4.244%` | `16.391%` |
| Freeze omega at `0.10 bar` | `3.306%` | `13.821%` |
| Tb-anchored Nannoolal | `5.708%` | `26.429%` |
| Variable omega to `0.25 bar`, then PR-Clapeyron | **`0.924%`** | **`6.269%`** |

### CoolProp ordinary

| Route | Curve median | p95 |
|---|---:|---:|
| Tb-variable omega | `0.987%` | `10.229%` |
| Freeze omega at `0.10 bar` | `1.234%` | `10.527%` |
| Tb-anchored Nannoolal | `7.264%` | `21.372%` |
| Variable omega to `0.25 bar`, then PR-Clapeyron | **`0.584%`** | **`2.885%`** |

Nannoolal should not compete with AW when qualified `Tc/Pc/omega/Tb` are
available. It remains useful when AW prerequisites are absent.

## Recommended No-Hard Routing

### Qualified `Tc`, `Pc`, physical omega, and trusted `Tb`

Use:

```text
physical omega above Tr=0.7
linear effective omega from Tb to Tr=0.7
continue unmodified linear omega below Tb
```

Do not add low-pressure clamping or relaxation. The modest Perry improvement
does not survive the independent CoolProp comparison.

Local segment quality factors:

| Pressure region | Factor |
|---|---:|
| `0.25 bar` through `Tc` | `0.91` |
| `0.05-0.25 bar` | `0.88` |
| Below `0.05 bar` | `0.85` |

The extended `0.91` region applies specifically when linear variable omega is
available.

### Only one of trusted `Tb` and omega

Use ordinary constant-effective-omega AW:

- Omega only: use the admitted physical/effective omega.
- `Tb` only: invert AW at `(Tb, 1.01325 bar)` and hold that effective omega
  constant.

Do not invent a second anchor solely to create a variable-omega relation.

For ordinary constant-effective-omega AW with a `Tb` anchor:

| Region | Factor |
|---|---:|
| `Tb-Tc` | `0.90` |
| `0.25 bar-Tb` | `0.88` |
| `0.05-0.25 bar` | `0.85` |
| Below `0.05 bar` | `0.80` |

All constant-omega routes that use `Tb` invert their effective omega from the
`(Tb, 1.01325 bar)` anchor and use the same `0.90` upper factor.

For omega-only AW, use pressure bands:

| Region | Factor |
|---|---:|
| Above `1.01325 bar` | `0.91` |
| `0.25-1.01325 bar` | `0.88` |
| `0.05-0.25 bar` | `0.85` |
| Below `0.05 bar` | `0.80` |

### Neither trusted `Tb` nor omega

When admitted `Tc/Pc` remain available, permit the best locally resolved or
estimated `Tb`, invert its effective omega, and use ordinary constant-omega AW.

Record the estimated anchor source, quality, and a warning.

### Broad/direct qualified Hvap

Switch at:

```text
0.25 bar
```

Then use:

```text
PR-delta-Z Clapeyron
ideal-delta-Z fallback
```

Once the route switches at `0.25 bar`, the tested variable-omega relaxation
forms become irrelevant below the handoff because all share the same exact
pressure anchor.

### Medium-quality multi-point Hvap

Retain the existing conservative:

```text
0.10 bar
```

policy. This benchmark used broad/direct Hvap, but `0.10 bar` was consistently
the second-best direct-Hvap switch and remains appropriate when extrapolation
quality is lower.

### No qualified Hvap

Continue unmodified Tb-variable omega to the domain floor.

The benchmark does not justify a universal freeze or relaxation rule.

### Missing trusted `Tb`

Use ordinary physical-omega AW.

### Missing qualified AW prerequisites

If either critical quality is below `0.90`, use the best available
Tb-anchored Nannoolal relation through `0.8Tc`, then attach the existing
anchored C1 AW upper completion to the available critical point.

The critical point may itself be estimated or lower quality in this final
fallback because Nannoolal selection already establishes that qualified
criticals do not exist.

Local factors:

| Region | Factor |
|---|---:|
| Above `0.8Tc`, anchored C1 AW | Standard upper `0.91 × 0.95` |
| `0.05 bar-0.8Tc`, Nannoolal | `0.85` |
| Below `0.05 bar`, Nannoolal | `0.75` |

The Nannoolal-region factors do not consume critical quality. The upper AW
region retains normal upper-completion quality inputs.

### Soft critical and omega quality

Admit `Tc`, `Pc`, and otherwise-unrecoverable omega at quality `0.90-0.93`,
but retain explicit warnings. Values below `0.90` are not eligible for AW
shape construction.

PR delta Z may use any finite positive critical constants regardless of source
or quality. Criticals below `0.90` receive a mild `0.97` factor rather than
forcing an early ideal-delta-Z fallback.

## Final Curve Quality

Generated segments retain the local quality factors above.

The final canonical curve exposes one aggregate quality calculated as a
log-pressure-weighted average of selected segment qualities. The process-
relevant `0.1-5 bar` region receives `3×` weight relative to pressure decades
outside it.

This avoids assigning the entire canonical curve the quality of a rarely used
deep-vacuum tail while preserving every local quality in segment provenance.

Hard-boundary lower AW completion also receives a mild span penalty:

```text
max(0.85, 1 - 0.03 × pressure-span-decades)
```

This depends on the actual hard-endpoint-to-switch extrapolation distance
rather than fixed pressure cutoffs.

### Monoacids

The ordinary PR-Clapeyron aggregates exclude monoacids. A no-hard acid route
should use the variable-omega curve as its incoming boundary and then the
existing dimer-aware deep completion, but that complete route still needs a
separate benchmark.

## Reproduction

```bash
PYTHONPATH=.:scripts/psat_experiments \
python scripts/psat_experiments/benchmark_no_hard_aw_relaxation.py

PYTHONPATH=.:scripts/psat_experiments \
python scripts/psat_experiments/benchmark_no_hard_clapeyron_switch.py
```

The switch benchmark uses up to eight worker processes. Override with:

```bash
PSAT_BENCHMARK_WORKERS=4
```
