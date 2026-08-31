# Psat physical domains, numerical continuations, and final validation

## Status

This note records the design direction reached during the canonical Psat
migration investigation. It deliberately distinguishes current behavior from
planned work.

Currently implemented:

- Canonical lower-domain selection now prefers a valid `Tt`, then a valid
  `Tm`, and uses the configured minimum pressure only when neither physical
  boundary is available.
- Completion relations follow the selected physical temperature domain rather
  than stopping at the configured pressure fallback.
- Deep A-H retries are enabled from the actual assembled lower-bound pressure.
- The canonical curve exposes its fitted lower temperature, critical
  temperature, lower logarithmic slope, and critical pressure slope.
- `thermo.Psat()` evaluates a cached numeric payload without invoking the
  checked pointwise resolver in its hot path.
- Below the canonical lower endpoint, `thermo.Psat()` uses a tangent
  continuation of `ln(P)` in inverse temperature.
- Above the critical point, `thermo.Psat()` continues pressure linearly in
  temperature.
- Checked pointwise property resolution remains limited to the canonical
  curve domain.

Not yet implemented:

- Unit-operation setup and final-state bounds validation.

## Motivation

The former pressure-floor-first domain policy made checked Psat
unavailable in ordinary liquid states. Glycerol illustrates the problem:

```text
Tm                         = 290.93 K
pipe temperature           = 303-318 K
former canonical T_min     = 393.30 K
configured pressure floor  = 0.001 bar
```

Glycerol is a valid liquid at the pipe temperatures, but checked Psat is
unavailable because its vapor pressure is below the numerical fitting floor.
That absence propagates into other properties: the Lucas compressed-liquid
viscosity correction silently skips its pressure correction when its internal
checked Psat lookup fails.

Separately, evaluating the compact canonical polynomial without bounds can be
catastrophic. The glycerol polynomial produced an artificial 10 bar saturation
root near 211 K, even though the legacy resolver returned an effectively zero
pressure there. A smooth, explicitly defined continuation fixes the numerical
problem, but it does not make the original pressure-floor domain physically
appropriate.

## Canonical domain policy

The canonical stable-fluid domain should be selected in this order:

1. Use a valid triple-point temperature `Tt` when available.
2. Otherwise use a valid melting-point temperature `Tm`.
3. Only when neither physical boundary is available, infer the lower
   temperature from the configured fallback pressure, currently 0.001 bar by
   default.

The resulting domain is:

```text
Tt ... Tc
```

or:

```text
Tm ... Tc
```

The configured pressure should no longer trim a curve whose physical lower
boundary is already known. In that case it is neither a volatility cutoff nor
a statement that lower vapor pressures do not exist.

### Lower-bound anchors

Selecting a domain boundary and imposing a hard pressure anchor are separate
decisions.

- When `Tb < Tt`, the reported one-atmosphere phase-change point represents a
  sublimation relationship rather than a normal liquid boiling point. Ignore
  `Tb` and its quality, and use the reliable `Tt/Pt` pair as the hard lower
  anchor. Anchor quality comes only from `Tt` and `Pt`.
- For an ordinary `Tt` or `Tm` boundary, do not invent a hard boundary pressure.
  Use the assembled source/completion curve unless an independently reliable
  pressure is available.
- When physical lower-bound temperatures are unavailable and the pressure
  fallback is used, that fallback pressure is the hard lower anchor.

The selected basis should be retained in metadata as one of:

```text
triple_point
melting_point
pressure_fallback
```

The implementation retains the existing metadata spelling `pressure_floor`
for that fallback basis.

### Deep-range fitting

Using `Tt` or `Tm` may expand the fit across many pressure decades. The fitter
must choose its deep-range forms from the actual assembled lower-bound pressure,
not merely from the configured fallback pressure. If the physical boundary is
well below 0.001 bar, A-H inverse-power retries may be required even though the
PFD retained its default pressure setting.

Fit diagnostics and quality penalties must continue to cover the full selected
domain. A physical endpoint must not silently force acceptance of an
ill-conditioned or inaccurate compact fit.

### A/B evidence for physical lower domains

On 2026-08-04, a runtime-only five-process probe compared the current
pressure-floor-first behavior against strict `Tt`, then `Tm`, then pressure
fallback selection. The proposed arm made A-H retries available for every
physical-boundary case. The input was the union of bundled curated, exact-CAS
CoolProp, and Perry components:

```text
423 unique components
66 curated components
124 CoolProp components
345 Perry components
```

The source populations overlap. Both policies canonicalized the same 402
components and failed on the same 21; physical-domain prioritization introduced
no new failures.

| Population | Policy | A-F | A-G | A-H | Successful |
|---|---|---:|---:|---:|---:|
| All unique | Current | 367 | 35 | 0 | 402 |
| All unique | Physical boundary | 348 | 7 | 47 | 402 |
| Curated | Current | 49 | 7 | 0 | 56 |
| Curated | Physical boundary | 40 | 1 | 15 | 56 |
| CoolProp | Current | 82 | 25 | 0 | 107 |
| CoolProp | Physical boundary | 65 | 1 | 41 | 107 |
| Perry | Current | 310 | 27 | 0 | 337 |
| Perry | Physical boundary | 295 | 6 | 36 | 337 |

Among the 402 paired successes, form transitions were:

```text
A-F -> A-F  348
A-F -> A-G    2
A-F -> A-H   17
A-G -> A-G    5
A-G -> A-H   30
```

There were 275 candidates with a usable physical boundary: 114 selected `Tt`
and 161 selected `Tm`. Of the 264 successful physical-boundary fits, 211 used
A-F, 6 used A-G, and 47 used A-H. Every case that needed H succeeded; there
were zero canonical fit rejections after H retries were available.

Eleven physical-boundary input bundles still failed in the broad probe, but not
because A-H was insufficient:

- Seven CoolProp cases had tiny endpoint coverage gaps because the probe paired
  raw, unhydrated curated/Perry critical temperatures with exact CoolProp
  saturation-segment endpoints. Fully hydrated production components select
  the matching CoolProp critical bundle, and all seven canonicalize
  successfully. This is a probe-input artifact, although generic reconciliation
  of mixed rounded/exact endpoints could still be made more robust.
- Two curated salts lacked `Tc/Pc`.
- One Perry acetylene record retained an incompatible boiling/domain anchor.
- One curated dichloroacetic-acid record had substantive source coverage gaps.

All eleven failed in both arms of the same probe. The A/B result therefore
supports physical-domain prioritization while showing that its main cost is
more frequent A-H selection, not a higher canonicalization failure rate. The
raw failure count should not be interpreted as the production failure count for
fully hydrated components.

## Thermo coefficient payload and continuations

The current coefficient payload contains:

```text
[A, B, C, D, E, F, G, H,
 Tc, inverse_power, supercritical_slope,
 T_min, lower_continuation_slope]
```

Here:

- `supercritical_slope` is `dP/dT` at `Tc`, in bar/K.
- `lower_continuation_slope` is `dln(P)/dT` at `T_min`, in 1/K.

The intended unchecked thermo evaluator is piecewise.

### Below `T_min`

Use a tangent Clausius-Clapeyron-style continuation:

```text
ln(P(T)) = ln(P(T_min))
           - T_min^2 (dln(P)/dT at T_min) (1/T - 1/T_min)
```

This preserves pressure and its first temperature derivative at the boundary,
approaches zero monotonically at low temperature, and avoids false roots from
global polynomial extrapolation.

### From `T_min` through `Tc`

Evaluate the selected canonical A-F, A-G, or A-H equation directly.

### Above `Tc`

Continue pressure, rather than logarithmic pressure, linearly:

```text
P(T) = P(Tc) + supercritical_slope (T - Tc)
```

This provides a finite effective K-value input for numerical solvers. It is not
a physical saturation pressure above the critical point.

## Checked resolution versus thermo execution

The two APIs have different responsibilities.

Checked property resolution:

- Provides physical domain enforcement.
- Returns source, method, quality, notes, and canonical fit diagnostics.
- Rejects point evaluation outside the selected physical domain.

Thermo coefficient execution:

- Avoids Python resolver dispatch in residual and Jacobian loops.
- Uses the explicit lower and upper continuations during numerical exploration.
- Does not throw merely because an intermediate iterate leaves the canonical
  domain.

The continuations therefore make numerical evaluation defined; they do not
declare every trial state physically acceptable.

## Planned unit-operation bounds validation

Unit operations eventually need to distinguish unrestricted trial evaluation
from acceptance of a final physical state. Bounds checks should not be placed
inside every scalar coefficient evaluation.

### Setup

At setup, a unit obtains component domain metadata sufficient to interpret
later evaluations:

- `T_min` and its domain basis.
- `Tc`.
- `Tt` and `Tm` when available.
- Whether lower or supercritical continuation is permitted for numerical
  trials.
- Property quality and warnings relevant to the unit's phase model.

### Hot-loop telemetry

Compiled residual and Jacobian loops should evaluate without exceptions, but
may accumulate inexpensive flags or extrema such as:

```text
used_lower_continuation
used_supercritical_continuation
encountered_nonfinite_value
minimum_trial_temperature
maximum_trial_temperature
```

These are diagnostics, not automatic failures. Intermediate Newton, line-search,
and initialization trials are expected to leave the physical region.

### Final-state validation

Before accepting a converged unit result, validate its physical states through
a shared thermo API rather than duplicating rules in every unit. A future API
might resemble:

```python
thermo.validate_equilibrium_state(
    T,
    P,
    composition,
    liquid_composition=x,
    vapor_composition=y,
    context="rigorous_distillation_stage",
)
```

Validation should distinguish:

- In-domain condensable evaluation: accepted; optionally compare compiled and
  checked Psat within tolerance.
- Lower continuation in an accepted fluid state: retain a structured warning
  or apply unit-specific policy.
- A claimed stable liquid below `Tt/Tm`: warn or reject because a solid phase
  may be required.
- A component above `Tc`: require supercritical/noncondensable treatment rather
  than interpreting the continuation as ordinary saturation.
- Negative or non-finite pressure, K-value, composition, or temperature:
  reject.

Trace composition should affect severity. A continuation used only for a
numerically negligible component may warrant a suppressed diagnostic, while a
bulk component outside the unit's supported phase domain should be an error.

### Unit-specific interpretation

- Flash units should validate the accepted phase split and reject a liquid
  assignment that depends materially on an invalid saturation state.
- Rigorous columns should validate final stage temperatures and phase
  assignments. Supercritical components should follow the established
  noncondensable policy.
- Pipes should validate the accepted local phase path, while still allowing
  unrestricted continuation during PH reconstruction.
- Heat exchangers, heaters, compressors, and expanders should validate only
  accepted outlet states, not every enthalpy-inversion probe.

## Volatility cutoffs and active components

The canonical lower domain and a unit-operation nonvolatile cutoff are separate
concepts. Volatility depends on both temperature and system pressure; a fixed
component label is generally insufficient.

Rigorous columns already floor K-values and retain scaled component balances,
which safely routes effectively nonvolatile components to numerical trace
amounts in the distillate. If explicit active-component selection is later
needed, select a stable active set during initialization and keep it fixed for
the nonlinear solve. Dynamically changing equation dimensions when a component
crosses a threshold would damage Jacobian continuity.

## Resolved integration issue: viscosity pressure correction

The Lucas compressed-liquid viscosity correction requests checked pointwise
Psat internally. Under the former pressure-floor domain, glycerol Psat was
unavailable at ordinary pipe temperatures and Lucas silently skipped the
correction.

The physical `Tm` domain now makes checked glycerol Psat available at 318.15 K.
Lucas is applied again, producing approximately 0.20911104 Pa s at 10 bar,
compared with the uncorrected 0.20869432 Pa s. The glycerol pipe regression
baselines now represent the restored pressure-corrected path.

## Implementation sequence

Completed:

1. Replaced pressure-floor-first trimming with `Tt`, then `Tm`, then pressure
   fallback domain selection.
2. Preserved the special `Tb < Tt` hard `Tt/Pt` anchor rule.
3. Selected deep canonical forms from the actual physical lower-bound pressure.
4. Revalidated focused canonical fit and property-resolution behavior across
   the expanded domains.
5. Rechecked the Lucas viscosity correction and restored pressure-corrected
   pipe baselines.

Remaining:

1. Add shared thermo final-state validation and unit-level acceptance checks.
2. Add regression tests for final states below `Tt/Tm`, above `Tc`, and for
   continuation use confined to intermediate solver trials.
