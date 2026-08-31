# Canonical Vapor-Pressure Resolution

Status: canonicalizer, component adapter, PFD adapters, domain selection, and
CoolProp source segments are implemented. The remaining source adapters and
component-initialization/runtime migration are still pending.

## Runtime Form

Canonical curves use

```text
ln(Psat/bar) = A + B/T + C*ln(T) + D*T + E*T^2 + F*T^5
dln(Psat)/dT = -B/T^2 + C/T + D + 2*E*T + 5*F*T^4
```

If the A-F fit exceeds its quality/provenance-specific retry tolerance, the
fitter first retries an A-G form by adding `G*T^3`. If that remains outside the
retry tolerance, it tries an A-H inverse tail term:

```text
H*((T/Tc)^p - 1), where p is -3, -5, or -7
```

The best monotone retry is selected by maximum error, then p95, MARD, and
condition number. Retry and hard-rejection limits depend on source quality and
whether one source covers the full domain; the selected profile is retained in
fit metadata.

The curve ends at `Tc`. Any critical-tangent continuation above `Tc` is a
thermo initialization aid, not vapor pressure.

## Domain Selection

`PsatCanonicalizationAdapter.resolve_domain()` consumes already resolved
component anchors and selects the lower temperature in this order:

1. Resolved triple point.
2. Melting point as the stable-liquid proxy.
3. `Tsat(P_floor)` supplied by the inverse resolution chain.

The default `P_floor` is `0.001 bar` (`100 Pa`) and may be overridden. A triple
or melting point is accepted only when its pressure is at least `P_floor`.
For `Tt`, a resolved `Pt` is used when available; otherwise `Psat(Tt)` is
evaluated through the supplied pressure-resolution chain. `Psat(Tm)` is always
evaluated through that chain. If either pressure lies below the floor, the
domain starts at `Tsat(P_floor)`.

PFD, CoolProp, and later source-specific behavior belongs only to the ordered
input/property resolution methods. Domain selection itself does not inspect or
invoke any named provider.

## Source Priority

Hard source segments are collected in this order:

1. PFD override, quality `1.0`.
2. CoolProp pure-fluid saturation, quality `0.995`.
3. Validated exceptional local tables, principally acids.
4. Local `chemicals.json` Antoine correlations.
5. Perry 2-8 correlations.
6. Perry 2-10 tables.
7. Other Antoine correlations only after Tb or overlap validation.

Higher-priority segments occupy overlaps. Lower-priority segments may only
fill uncovered intervals. Boundary-conditioned AW, Clapeyron, and Nannoolal
relations are supplied externally; the canonicalizer does not branch on their
names or equations.

## PFD Contracts

- PFD scalar and correlation overrides have quality `1.0`.
- A full-domain `canonical_psat` or `canonical_psat_af` canonical relation is used
  directly and bypasses completion and regression.
- A narrower canonical relation and legacy `poly_x`, `exp_poly_x`, or
  `reduced_vapor_pressure` relation becomes a pinned segment over its declared
  validity range.
- Direct canonical overrides are trusted. Coefficients are checked for finite
  values and metadata conflicts, but the canonicalizer deliberately does not
  impose monotonicity or Tb/Pc residual diagnostics on a direct override.

## CoolProp Contracts

- Runtime fluid discovery uses the bundled static alias map. Unknown fluids do
  not trigger dynamic CoolProp indexing.
- The runtime dependency is pinned to CoolProp `7.2.0`, matching the version
  recorded in the bundled alias map.
- Only pure fluids are accepted. Water deliberately uses `IF97`; other fluids
  use `HEOS`.
- Matching requires an exact resolved CAS, including its check digit. Names,
  symbols, formulas, and arbitrary CoolProp aliases are not accepted as fluid
  identity by themselves.
- Normalized aliases that belong to multiple CoolProp fluids are omitted from
  the runtime map and retained only in the map's `ambiguous_aliases` metadata.
- CoolProp globally replaces non-PFD Tb, triple-point, and critical properties
  for supported fluids. `Zc` is recomputed from the final selected
  `Tc/Pc/Vc` triplet rather than selected independently, so partial PFD
  primitive overrides cannot retain a stale CoolProp identity. A one-sided PFD
  `Tt` or `Pt` override does not inherit the other member from CoolProp.
- A saturation segment spans CoolProp `Ttriple` through `Tcrit`. Its pressure
  validity endpoints come from the same `P(T,Q=0)` evaluator used by the
  segment. CoolProp's separately reported `ptriple` remains provenance only;
  it is not used as a strict evaluator bound.

## CoolProp Benchmarks

`scripts/psat_experiments/benchmark_coolprop_psat.py` compares local tables and
Perry 2-8 against CoolProp, compares IF97 and HEOS water, and fits every
supported CoolProp pure-fluid curve to the canonical form.

Current stored default-uniform-grid canonical results for 114 fluids using the selected
domain floor:

```text
selected forms: A-F 109, A-G 5, A-H 0
per-curve MARD: median 0.0448%, p95 0.0785%, max 0.0982%
per-curve p95 error: median 0.1215%, p95 0.2275%, max 0.3084%
per-curve maximum error: median 0.1553%, p95 0.2965%, max 0.7468%
maximum error >1%: 0 curves
```

The largest remaining maximum error is R13 (`0.7468%`). Fits retain diagnostics,
and contextual hard rejection protects future provider inputs that remain too
inaccurate after the retry ladder.

IF97 water versus HEOS water:

```text
pressure MARD 0.0063%, p95 0.0172%
slope MARD 0.0105%, p95 0.0262%
```

## Remaining Integration

1. Add acid-table, Antoine, Perry 2-8, Perry 2-10, and validated external
   Antoine segment adapters.
2. Add boundary-conditioned AW, frozen-Hvap Clapeyron, and Nannoolal factories.
3. Supply the pressure-recursive fitting grid and exact transition pressures.
4. Canonicalize once during component initialization and store coefficients,
   domain, quality, provenance, and diagnostics.
5. Move supercritical critical-tangent use into thermo initialization.
6. Retire pointwise runtime resolver switching after integration tests pass.
