# Psat Construction Policy and Justification

Final synthesis of the implemented policy and its supporting investigations,
as inspected on **2026-09-30**.

This document explains the decisions behind canonical vapor-pressure
construction: source authority, admission, domains, source handoffs, completion,
regression, confidence, and numerical continuations. It adds a current synthesis
without replacing the earlier design notes or experiment reports. Those reports
remain the detailed record of alternatives, measurements, and limitations.

The implementation is the authority for current behavior. A recommendation in
an experiment report is not automatically an implemented feature. The evidence
below consists of existing reports and inspection of their maintained scripts;
no benchmarks were rerun to prepare this document.

## 1. How to interpret the justification

There are three different bases for a decision:

- **Mathematical or physical requirement:** for example, matching a qualified
  critical pressure or keeping a generated liquid-vapor curve monotonic.
- **Comparative experimental evidence:** a maintained benchmark compares
  alternatives on defined reference populations and supports a preference.
- **Engineering judgment:** an authority ranking, confidence factor, tolerance,
  numerical resolution, or fallback rule has a practical purpose but has not
  been demonstrated to be an empirical optimum.

These bases can coexist. A method may have strong comparative support while its
exact quality multiplier remains a judgment. Where this document explains the
practical purpose of a constant without a recorded calibration study, that is
an engineering interpretation, not a claim about its historical derivation.

Quality is a trust ranking on `[0, 1]`, **not a probability or an experimental
error bound**. A quality of `0.95` does not mean 95% certainty or 5% error.
See [the general quality semantics](property_resolution_design.md#quality-semantics).

Likewise, two errors must be distinguished:

1. A completion benchmark's error against a Perry or CoolProp reference curve.
2. The canonical regression's error against the assembled source/completion
   target.

A small error of the second kind establishes faithful compression of that
target, not independent accuracy of the target itself. Most construction
experiments use reference correlations or EOS results, not independent raw
experimental measurements. Figures from different studies also have different
case filters and weightings and must not be compared as though they came from
one common population.

## 2. Construction architecture

The construction sequence is:

```text
resolve component properties and qualified anchors
  -> select a physical lower domain, or a pressure fallback
  -> collect direct source segments
  -> admit eligible sources and coordinate hard-source handoffs
  -> complete uncovered intervals
  -> fit one constrained canonical relation
  -> retain its domain, local source confidence, and fit diagnostics
  -> evaluate cached coefficients during thermodynamic calculations
```

The component adapter owns provider knowledge. The segment assembler,
handoff coordinator, completion coordinator, and fitter operate on generic
bounded relations. This lets source admission and interval construction remain
separate from the final equation used by runtime calculations.

This separation also prevents a completion relation from being treated as
direct empirical data merely because it fills the same interval.

Implementation references:

- [Component and provider adapter](../property_resolution/vapor_pressure_adapter.py).
- [Generic assembly, handoff, completion, and fitting](../property_resolution/vapor_pressure_canonical.py).
- [Resolver construction and persistent canonical curves](../property_resolution/vapor_pressure.py).
- [Thermodynamic coefficient evaluation](../thermodynamics_models/base.py).

## 3. Source authority and admission

### 3.1 Priority is distinct from uncertainty

The current priority hierarchy is:

| Input or relation class | Priority |
|---|---:|
| PFD override | 1000 |
| CoolProp pure-fluid saturation | 900 |
| Local direct correlation, including curated exceptional relations | 800 |
| Curated local Antoine | 700 |
| Perry Table 2-8 | 600 |
| High-quality external/Smith Antoine | 550 |
| Perry Table 2-10 | 500 |
| Other and cached-NIST Antoine | 400 |
| Two-hard-boundary middle completion | 320 |
| Anchored AW and Clapeyron completion | 300 |
| No-hard AW fallback | 200 |
| Last fallback, including no-hard Nannoolal regions | 100 |

Priority determines which source occupies overlapping intervals; quality
describes confidence in its values. They are not interchangeable. In the
assembler, equal-priority segments retain collection order. Handoff rejection
compares priority, then quality, then insertion order. The numeric priority
spacing has no physical meaning; the ordering is the substantive policy.

PFD authority is an explicit input contract. A full-domain canonical PFD
relation bypasses completion and regression. A narrower canonical relation or
other supported PFD correlation becomes a pinned interval. Direct canonical
overrides are trusted rather than subjected to the generated-curve monotonicity
and anchor-residual tests. Their metadata and finite coefficients are still
checked. This is a deliberate authority choice, not a claim that every supplied
curve is physically validated.

CoolProp requires an exact resolved CAS and a supported pure fluid. Water uses
IF97; other fluids use HEOS. The selected CoolProp bundle supplies non-PFD phase
points and critical properties so a curve is not assembled around unrelated
scalar anchors. Explicit PFD overrides take precedence. Partial PFD triple-point
overrides do not acquire the other member of the pair from CoolProp.

Exact-CAS matching and the bundled static fluid map protect component identity;
ambiguous normalized aliases are not used as independent fluid matches. CoolProp
segment bounds use saturation pressures evaluated by the segment's own function
rather than enforcing the separately reported triple pressure as an evaluator
bound. This avoids treating small provider-internal endpoint differences as
missing physical coverage.

The historical IF97/HEOS water comparison reports pressure MARD/p95 of
`0.0063%/0.0172%` and slope MARD/p95 of `0.0105%/0.0262%`. This documents close
agreement of the backend choice on the tested range; it does not establish a
universal accuracy advantage of IF97. See the
[CoolProp benchmark summary](vapor_pressure_canonicalization.md#coolprop-benchmarks).

Common direct-source confidence defaults are `1.0` for PFD, `0.995` for
CoolProp, `0.98` for Perry 2-8 and curated local Antoine, and `0.96` for a local
portable correlation without a more specific score. Supplied source metadata
can change the local-correlation/curated-Antoine score. These values rank trust;
they are not uncertainty estimates obtained from the backend comparison.

The authority ordering is supported by provider characteristics and the
comparison scripts below. It is not a statistically optimized universal source
ranking, and agreement with a higher-ranked provider is not proof that every
lower-ranked record is wrong.

### 3.2 Shared normal-boiling-point validation

Direct sources below priority `600` are eligible for the shared hard-`Tb` gate.
Sources at or above `600` are exempt from this gate; exemption does not exempt
generated assemblies from their separate finite/monotonicity and handoff checks.

A qualified normal boiling point is source-backed, finite, positive, normally
quality `>= 0.90`, and not estimated or soft. Sublimation/triple-point rules must
also permit its use as a liquid boiling anchor. The existing direct
unprovenanced programmatic-value convention is retained.

For an eligible segment:

- Evaluate at `Tb` only when it lies within the declared source range, with a
  `1 K` endpoint allowance.
- Compare pressure with `1.01325 bar`; reject an in-range relative discrepancy
  greater than `2%` or an invalid pressure.
- Missing, estimated, or out-of-range `Tb` lowers confidence rather than
  rejecting empirical data by extrapolation.
- Agreement with a strictly higher-priority surviving source over a real
  overlap may promote confidence. Same-priority agreement does not promote it.

The current confidence ladders are:

| Source profile | Standalone | Higher-priority overlap | Hard-`Tb` validated |
|---|---:|---:|---:|
| Standard direct table/ordinary Antoine | 0.90 | 0.93 | 0.95 |
| Cached-NIST Antoine | 0.87 | 0.90 | 0.95 |
| High-quality Antoine | 0.95 | 0.96 | 0.97 |

The policy addresses a demonstrated class of source inconsistency. The motivating
thiodiglycol Perry 2-10 row gives approximately `0.129 bar` at an independently
resolved `555.15 K` normal boiling point, an approximately `87.25%` discrepancy.
The shared gate prevents such a row from influencing canonical construction or
Psat-derived omega, without a compound-specific blacklist.

The [Tb validation plan](TB_VALIDATION_PLAN.md) explains the authority boundary,
confidence ladders, independence rule, and in-range-only rejection. Its
"design only" status and cache-version targets are historical; the current
shared admission functions and adapter implement the policy. The exact `2%`,
`1 K`, and confidence-tier values are inherited engineering settings. The plan
explicitly preserves the `2%` tolerance pending a separate benchmark and decision.

Supporting source/admission studies include
[local tables versus Perry](../scripts/psat_experiments/benchmark_local_tables_perry.py),
[CoolProp comparisons](../scripts/psat_experiments/benchmark_coolprop_psat.py),
[other Antoine sources](../scripts/psat_experiments/benchmark_other_antoine_sources.py),
and [the Antoine Tb gate](../scripts/psat_experiments/benchmark_antoine_tb_gate.py).
Their existence supports investigation and reproducibility; it does not by
itself establish a calibrated optimum for every acceptance setting.

## 4. Domain and phase-change anchors

The lower-domain preference is valid subcritical `Tt`, then valid subcritical
`Tm`, then a temperature inferred using the pressure fallback. The default
fallback is `0.001 bar` (`100 Pa`). A physical boundary is not discarded merely
because its vapor pressure is below that fallback. Completion follows the
physical temperature domain when one is available.

This corrects a concrete applicability failure: glycerol at `303-318 K` is above
its `290.93 K` melting point, but the former pressure-floor policy started its
canonical curve near `393.30 K`. Checked Psat was then unavailable in ordinary
liquid states, including internal viscosity calculations.

The [physical-domain investigation](investigations/psat-domain-and-final-validation-design.md)
reports an A/B comparison over `423` unique components. Both policies succeeded
on the same `402` and failed on the same `21`. Among `264` successful
physical-boundary cases, `211` used A-F, `6` A-G, and `47` A-H. The report explains
the common failures and the seven CoolProp endpoint mismatches caused by
unhydrated probe inputs. This supports physical-domain prioritization and the
need for deeper fit forms; it is not independent validation of every deep tail.

Selecting a domain temperature and selecting a hard pressure anchor are
different decisions:

- A qualified `Tb` inside the domain anchors pressure at `1.01325 bar`.
- `Tc/Pc` anchor the upper endpoint.
- If `Tb < Tt`, it is not used as a liquid normal-boiling anchor. A qualified
  `Tt/Pt` pair supplies the replacement lower pressure anchor when available.
  A qualified triple point above one atmosphere can also supply that anchor
  when normal boiling is absent.
- Ordinary `Tm` and `Tt` domain selection does not invent a pressure constraint.
- The generic fitter can accept an explicit lower-pressure constraint. The
  runtime builder supplies the applicable replacement triple-point pressure;
  pressure-fallback curves are selected/trimmed using pressure crossings, with
  a further fitted-domain adjustment if the fitted lower pressure is below the
  floor. The floor is not universally imposed as a regression constraint.

The main older [canonicalization document](vapor_pressure_canonicalization.md)
describes pressure-floor clipping of physical endpoints. That description is
historical and should not be used to infer the current domain policy.

## 5. Hard-source handoffs and middle gaps

### 5.1 Overlapping sources

The handoff coordinator first checks hard segments for finite, increasing
behavior. It prefers an already compatible priority boundary; otherwise it
searches the overlap for a direct handoff or a narrow monotonic C1 bridge.
Incompatible lower-preference sources can be rejected and the assembly rebuilt.

Current defaults distinguish direct and smoothed compatibility:

| Setting | Value |
|---|---:|
| Direct relative pressure mismatch | 2% |
| Direct relative slope mismatch | 10% |
| Smoothing relative pressure mismatch | 2.5% |
| Smoothing relative slope mismatch | 15% |
| Direct relocation compensation-integral budget | 0.12 K |
| Overlap search samples | 201 |
| Bridge probe samples | 201 |
| Candidate bridge fractions of overlap width | 0.10, 0.20, 0.35, 0.50, 0.75, 1.00 |

The purpose is to join compatible evidence without hiding substantial source
conflict inside an arbitrarily broad smoothing region. The compensation budget
limits how much target-curve change is introduced by moving a direct handoff.
The narrowest successful bridge limits that change locally.

These are engineering acceptance/search settings. The Antoine comparison script
uses the pressure/slope settings, but no retained report establishes that all
these exact values, especially `0.12 K`, are optimally calibrated.

Some completion relations explicitly permit a slope mismatch so an independent
physical relation is not rescaled solely to force C1 agreement. A mismatch beyond
the ordinary limit then produces a warning and a one-time `0.90` quality factor
on the affected exception-bearing segment. This is a separate explicit policy,
not a relaxation of pressure continuity or positive-slope requirements.

### 5.2 Interior gaps between hard segments

For an interior gap with pressure and derivative at both hard boundaries,
production uses cubic Hermite interpolation in `x = ln(T)`:

```text
y = ln(P/bar)
dy/dx = T dln(P)/dT
```

This exactly matches both boundary values and temperature derivatives without
requiring critical constants or omega.

The [middle-gap study](../scripts/psat_experiments/middle_gap_completion_results.md)
compares nine methods across `4,131` gaps from `345` Perry correlations and `114`
CoolProp fluids. It records `37,179` evaluations, no numerical failures, and no
nonmonotonic generated curves. Per-gap MARD comparisons are:

| Method | Median | p95 | Worst curve |
|---|---:|---:|---:|
| Hermite in T | 0.075% | 6.761% | 30.506% |
| Hermite in 1/T | 0.003% | 1.244% | 9.362% |
| Hermite in ln(T) | 0.003% | 0.162% | 2.540% |
| Cubic endpoint-effective-omega AW | 0.001% | 0.404% | 5.017% |

Log-T Hermite wins on tail robustness, although effective-omega AW has the
smaller median. It also avoids extra property dependencies. These are the
reasons for the choice, rather than simply selecting the lowest median.

The production factor `0.97` is a confidence judgment. The report's proposed
broad-gap confidence reduction/span limit and dedicated effective-omega middle
fallback are not implemented as separate policies in the current middle-gap
adapter. Generic completion can consider other eligible relations, but the
report's complete recommendation must not be equated with current behavior.

The study excludes noisy one-sided sparse-table derivatives, uses synthetic gap
locations, and evaluates the assembled bridge rather than the later global fit.

## 6. Completion above a hard segment

The terminal upper gap uses anchored Ambrose-Walton. A correction matches the
hard left-boundary pressure and slope and vanishes in value and slope at `Tc`.
This preserves the hard source while retaining the critical endpoint and the
AW baseline near it.

Omega preference is a sufficiently qualified nonestimated component omega,
then its physical definition from hard `Psat(0.7Tc)`, then inversion of AW at the
hard endpoint. Endpoint inversion is internal shape state, not a replacement
component acentric factor. When inversion occurs below `0.7Tc`, its quality is
reduced according to the distance from that reference condition.

Anchored upper/lower completion admits criticals at quality `>= 0.80`, with a
`0.97` factor below `0.90`. Preferred physical omega for these hard-boundary
routes is quality `>= 0.93`. The upper method factor is `0.91`.

The mathematical boundary correction and the need to avoid source switching
have direct justification. Comparisons are maintained in
[Perry AW](../scripts/psat_experiments/benchmark_perry_aw.py),
[Antoine AW](../scripts/psat_experiments/benchmark_antoine_aw.py),
[validated Antoine AW](../scripts/psat_experiments/benchmark_antoine_tb_gate.py),
and [Perry 2-10 AW](../scripts/psat_experiments/benchmark_table210_aw.py).
The retained synthesis of upper-route alternatives and exact confidence
calibration is less complete than for lower and middle completion.

## 7. Completion below a hard segment

### 7.1 Effective omega from reliable boundary derivatives

At a hard lower endpoint `(Th, Ph)`, invert AW to find `omega_h`. With a reliable
analytic derivative, infer its local temperature derivative:

```text
omega'_h = [s_h - partial_T AW(Th, omega_h)] / partial_omega AW(Th, omega_h)
omega(T) = omega_h + omega'_h (T - Th)
```

This matches the endpoint pressure and slope. With a nonanalytic derivative,
production instead prefers a separate value anchor: `Tb`, qualified omega at
`0.7Tc`, hard pressure at `0.7Tc`, or an admitted softer omega. A C1 correction
then matches the boundary slope without deriving the whole extrapolation trend
from that derivative alone.

Above trusted `Tb`, the baseline is the endpoint-to-`Tb` effective-omega line.
A quintic correction is confined to `[Tb, Th]` and vanishes in value and its
first two derivatives at `Tb`. At/below `Tb`, the separate-anchor route uses an
affine inverse-temperature correction that remains active through the switch.

The [lower-AW matching report](../scripts/psat_experiments/lower_aw_matching_results.md)
includes a `44`-method study over `2,421` cases and focused follow-ups. It tests
analytic derivatives and realistic upper-only PCHIP derivatives. It explains
why local-slope extrapolation is weaker when the hard endpoint is well above
`Tb`, and why forcing corrections to vanish at `0.25 bar` discards useful
curvature. The localized-`Tb` construction gives p95 MARD from `0.68%` at a
`1.25 bar` endpoint to `1.22%` at `5 bar`, with no nonmonotonic curves.

Current method factors are `0.95` for endpoint-slope dynamic omega and `0.94`
for the separate-anchor construction. Extrapolation also receives:

```text
span_factor = max(0.85, 1 - 0.03 * pressure_span_decades)
```

The relevant span runs from the hard endpoint to the switch, or to the physical
lower domain when no switch is available. The reason for a span penalty is
supported by deteriorating long extrapolations; its exact coefficient/floor and
the route factors are engineering judgments.

The report also studies value-only boundaries. The current production lower-AW
relation requires a boundary derivative; those value-only recommendations are
experimental guidance, not a separately implemented route.

### 7.2 Why Nannoolal is not the primary hard-boundary continuation

The [direct Nannoolal comparison](../scripts/psat_experiments/direct_lower_nannoolal_results.md)
anchors the structural slope relation at a reliable pressure endpoint and tests
multiplicative endpoint-slope calibration. Calibrating the slope improves
Nannoolal but does not generally overtake dynamic omega from a strong hard
boundary. The report finds no broadly predictive ordinary-fluid derivative
correction beyond the endpoint ratio itself.

This supports preserving hard-source slope information rather than replacing
it with a structural estimate. It does not establish that the particular
no-hard Nannoolal/upper-AW fallback is fully validated; that has different inputs.

## 8. Deep completion using Hvap

### 8.1 Freeze the source before integration

The adapter samples the requested lower-domain-to-`Tb` interval at `41`
temperatures. A usable plan requires positive finite Hvap, a consistent source
and method, an accepted relation class, and minimum quality `0.80`. It freezes
the sampled relation with PCHIP before pressure integration.

Currently accepted classes are recognized broad/direct relations and NIST
multi-point Watson fits. A lone scalar/one-point Watson source is not a separate
accepted deep-plan class. The experiments on one-point and synthetic five-point
Watson are evidence about possible routes, not proof that those routes are all
implemented by this adapter.

Freezing prevents a changing property-resolution branch from introducing
uncontrolled derivative changes inside an ODE. The `41`-point resolution is a
numerical setting, not a demonstrated optimum for every Hvap curvature.

Switch pressure currently follows the minimum frozen-plan quality:

| Hvap quality | Switch |
|---|---:|
| >= 0.95 | 0.25 bar |
| >= 0.90 and < 0.95 | 0.10 bar |
| >= 0.80 and < 0.90 | 0.05 bar |

The switch function presently uses quality, not a separate class-specific
formula. A usable broad-Hvap `0.25 bar` switch has direct comparative support.
The lower-quality switches are conservative engineering extensions. The
no-hard report explicitly says its `0.10 bar` medium-quality recommendation is
inferred from a broad/direct-Hvap study rather than tested on that source class.

### 8.2 Ordinary fluids: PR-corrected differential Clapeyron

The production ordinary-fluid relation integrates:

```text
dln(P)/dT = Hvap(T) / [R T^2 deltaZ(T, P)]
```

It uses Peng-Robinson vapor/liquid root separation when an admitted omega is
available, and `deltaZ = 1` when properties or valid roots are unavailable.
Physical omega is retained rather than calibrated to the incoming AW curve.
The relation is not rescaled simply to force exact incoming-slope agreement.

The [end-to-end lower study](../scripts/psat_experiments/end_to_end_lower_completion_results.md)
imports the production lower-AW implementation. Its reference sets include
`307` Perry curves (`1,228` hard-boundary cases) and `82` CoolProp fluids (`328`
cases), with targets through `0.001 bar`. It compares ideal, volume-backed,
PR, and boundary-inferred relaxed volume corrections, including sparse-Hvap
variants and inherited switch errors.

The [no-hard study](../scripts/psat_experiments/no_hard_fallback_results.md)
compares switches and PR against ideal delta Z on common cases:

| Reference | PR curve-MARD median/p95 | Ideal median/p95 |
|---|---:|---:|
| Perry ordinary | 0.917% / 6.135% | 1.513% / 6.801% |
| CoolProp ordinary | 0.548% / 2.665% | 1.534% / 3.557% |

At `0.25 bar`, PR also outperforms later switches in these populations. Using
effective instead of physical omega has negligible effect in the end-to-end
study: CoolProp point median/p95 is `0.678%/1.746%` versus `0.678%/1.747%`.

An important reconciliation of the reports is:

- The earlier direct/end-to-end recommendations favor boundary-inferred linear
  delta-Z relaxation as an initial baseline and describe PR as a strong physical
  alternative.
- The no-hard report explicitly prefers PR over ideal correction.
- Current production uses PR/ideal for ordinary fluids; the adapter does not
  implement the boundary-inferred relaxation as a competing deep route.

These studies do not demonstrate that PR dominates every relaxed alternative
on every objective. The end-to-end report shows tradeoffs in median, p95, and
maximum error. The implemented choice has comparative support and avoids an
additional boundary-calibration state, but is not a universal empirical winner.

Actual saturated-volume delta Z is an especially strong diagnostic in the
reports. It is not currently an independent selectable deep-plan route in this
adapter. Nor does taking PR roots at the supplied pressure establish exact
fugacity equilibrium; PR delta Z is a volume approximation in this construction.

### 8.3 Monocarboxylic acids

Acids use a separate boundary-calibrated dimer relation, with fixed dimerization
enthalpy `-60.5 kJ/mol dimer`. The incoming slope and Hvap determine the apparent
association extent and entropy. When they require no positive association, the
implementation suppresses it using the very negative entropy sentinel
`-1000 J/mol/K`. That sentinel is a numerical suppression setting, not a measured
dimer entropy. The integrated slope is:

```text
dln(P)/dT = Hvap(T) / [R T^2 (1 - alpha(T, P))]
```

Ordinary PR-Clapeyron is unsuitable for this benchmark population: the
end-to-end monoacid point median/p95 is `16.324%/383.797%`, versus
`5.711%/14.635%` for the calibrated dimer construction.

The [hard-segment parameter-fit study](../scripts/psat_experiments/hard_dimer_parameter_fit_results.md)
tests freeing both enthalpy and entropy over `18` acids and `72` hard-endpoint
cases for each fitting span. A `0.15`-decade fit improves point median error
from `5.711%` to `1.679%`, but worsens p95 from `14.635%` to `30.221%`.
Sensitivity to a smooth `0.1%` derivative drift and endpoint-dependent fitted
enthalpies demonstrate why formal rank and high R-squared do not establish
practical identifiability. This supports retaining the fixed-enthalpy approach,
not adopting free fits solely for better median performance.

The [Perry entropy-inversion study](../scripts/psat_experiments/perry_dimer_entropy_results.md)
is related evidence, but its generic `-144 J/mol/K` entropy decision concerns
the shared dimer fallback. Canonical deep completion calibrates entropy at its
boundary rather than universally assigning that generic value. The report
also states an important thermodynamic interpretation: Perry Hvap is treated
as the nominal phase-change enthalpy represented by the published curve.
Interpreting it as strictly monomer-only enthalpy would require a different
association-enthalpy balance. That convention limits the physical interpretation
of inferred dimer parameters.

The ordinary-fluid PR aggregates exclude monoacids. The complete no-hard acid
route is explicitly identified in the no-hard report as needing a separate
benchmark; the hard-boundary acid results must not be presented as that test.

## 9. Construction when no hard source is available

With qualified critical constants, production selects among:

| Available inputs | AW shape |
|---|---|
| Trusted Tb and admitted omega, with Tb < 0.7Tc | Effective omega linear from the Tb pressure anchor to physical omega at 0.7Tc; continue the trend below Tb |
| Trusted Tb without a usable second anchor | Constant effective omega inverted at Tb |
| Admitted omega without trusted Tb | Constant physical omega |
| Neither trusted Tb nor admitted omega | Attempt an estimated Tb and use a warned, lower-confidence constant-effective-omega route |

The variable-omega transition is eased near its upper endpoint to join the
constant physical-omega region smoothly. The connection at `0.7Tc` follows
omega's reference condition; the easing width of `0.05` in the normalized
transition coordinate is a numerical design
choice rather than an independently calibrated physical scale.

When critical qualities are below `0.90`, the adapter first attempts
Tb-anchored Nannoolal through `0.8Tc`, then attaches C1 upper AW to the available
critical point. If that route is unavailable, production can still use a
warned quality-weighted no-hard AW fallback. Thus the documentation preference
for qualified criticals is not an unconditional prohibition of all final AW
fallbacks below `0.90`.

Nannoolal requires a usable structure and boiling anchor. Its region factors
are `0.85` above `0.05 bar` and `0.75` below it; upper AW receives its usual
factor with an additional `0.95`. The `0.8Tc` transition and these factors are
conservative engineering settings, with less direct validation for the complete
low-quality-input hybrid than for hard-boundary AW.

The no-hard AW local factors are:

| Route | Pressure region | Factor |
|---|---|---:|
| Tb-variable omega | >= 0.25 bar | 0.91 |
| Tb-variable omega | 0.05-0.25 bar | 0.88 |
| Tb-variable omega | < 0.05 bar | 0.85 |
| Tb-anchored constant omega | >= 1 atm | 0.90 |
| Tb-anchored constant omega | 0.25 bar-1 atm | 0.88 |
| Tb-anchored constant omega | 0.05-0.25 bar | 0.85 |
| Tb-anchored constant omega | < 0.05 bar | 0.80 |
| Omega-only | >= 1 atm | 0.91 |
| Omega-only | 0.25 bar-1 atm | 0.88 |
| Omega-only | 0.05-0.25 bar | 0.85 |
| Omega-only | < 0.05 bar | 0.80 |

The [no-hard experiments](../scripts/psat_experiments/no_hard_fallback_results.md)
test freezing, clamping, and relaxing the extrapolated omega. Improvements on
Perry do not consistently survive CoolProp comparison, so production does not
apply a universal low-pressure freeze or relaxation rule. With qualified
frozen Hvap, AW yields to deep completion at the chosen switch. Otherwise it
continues to the selected domain.

The experiments strongly support this method ranking with qualified inputs;
they provide weaker evidence for missing-input and estimated-anchor branches.
The tabulated factors express that confidence ordering and are not calibrated
probabilities or percentage-error predictions.

## 10. Canonical equation and constrained fitting

### 10.1 Representation and retries

The base representation is:

```text
ln(P/bar) = A + B/T + C ln(T) + D T + E T^2 + F T^5
```

The retry ladder adds `G T^3` for A-G, then for A-H adds:

```text
H [(T/Tc)^p - 1], p in {-3, -5, -7}
```

The inverse term vanishes at `Tc` and supplies low-temperature flexibility.
Production enables inverse retries when the actual assembled lower pressure
is below `0.001 bar`, or a smaller pressure fallback was requested. It does not
enable them solely because a component has a physical domain boundary.

A-F is retained when acceptable; otherwise A-G is tried. If needed and enabled,
A-H candidates are compared. Acceptable simpler forms stop the retry ladder.
When a best-candidate comparison is needed, monotonic candidates are preferred
and ranked by maximum pressure error, p95 error, MARD, then reduced condition
number. This controls tails before median/mean performance and avoids adding
terms when the simpler form already suffices.

[The constrained hybrid-fit experiment](../scripts/psat_experiments/benchmark_hybrid_curve_fit_constrained.py)
compares the five base terms with added T5 and T6 terms. The
[CoolProp canonical benchmark](../scripts/psat_experiments/benchmark_coolprop_psat.py)
uses the canonicalizer, and the physical-domain investigation demonstrates the
need for A-H on expanded domains. These support the representation strategy.
They do not supply a complete retained comparison proving that T3 and exactly
the three inverse powers are optimal among all reasonable bases.

### 10.2 Objective, anchors, and numerical resolution

The runtime fits `401` uniformly spaced temperature samples of the completed
assembly, with equal regression sample weights. The quality aggregation's
pressure weights are **not** these regression weights. Older experiments using
pressure-recursive/log-pressure grids are not descriptions of the current
runtime grid.

Regression minimizes squared log-pressure residuals, using scaled matrix
columns and an exact-constraint particular solution plus null-space least
squares. This handles the very different magnitudes of the basis terms while
preserving selected pressure anchors. Least-squares `rcond` is `1e-13`.

Diagnostics include sample pressure errors, anchor residuals, and reduced
condition number. Monotonicity is probed at `1001` temperatures, and each
selected source slice gets pressure-error diagnostics at `201` temperatures.
The overall MARD/p95/maximum diagnostics are calculated on the regression
samples; the `1001`-point grid is not a separate global pressure-error dataset.
Sampling-based monotonicity checks are not a mathematical proof between points.

The `401/1001/201` resolutions and `rcond` are numerical engineering settings.
No retained sensitivity study establishes them as universal minimum resolutions.

### 10.3 Retry thresholds, rejection, and quality penalty

For aggregate source quality `q`, current retry thresholds in percent are:

```text
MARD threshold = 20 (1 - q)
maximum-error threshold = 100 (1 - q)
```

For example, `q = 0.995` gives `0.10%` MARD and `0.50%` maximum error;
`q = 0.95` gives `1.0%` and `5.0%` respectively. One-source and assembled
profiles currently use the same threshold formulas; the profile names do not
introduce different numeric limits.

After retries, exceeding these quality-scaled thresholds causes a confidence
penalty, not an automatic hard rejection:

```text
e_m = max(0, MARD_percent - threshold_m - epsilon) / 100
e_x = max(0, max_error_percent - threshold_x - epsilon) / 100
penalty = 7.5 e_m + 1.5 e_x
final_quality = max(0, q - penalty)
epsilon = 1e-10 percentage points
```

Separate hard rejection applies to nonmonotonic fits, invalid/rank-deficient
construction, insufficient samples, anchor log residuals above `1e-9`, and
configured hard error limits. The default global hard MARD/p95/max limits are
unset. Pinned PFD slices have explicit hard fit limits of `0.25%` MARD and
`1.0%` maximum error, because a regression must reproduce an authoritative
override sufficiently closely. Direct full-domain canonical PFD overrides
bypass this regression path.

These distinctions correct older wording that could imply every contextual
error-threshold exceedance is a hard rejection. The exact `20/100`, `7.5/1.5`,
and PFD fit limits are policy judgments; retained reports do not establish
their calibration. A high condition number is recorded and can break a
candidate-ranking tie, but is not itself a default hard rejection threshold.

## 11. Local and aggregate confidence

For most generated relations, confidence combines consumed input qualities
and method factors. Hard-source intervals retain their own confidence. Lower
AW additionally accounts for extrapolation span; deep relations incorporate
Hvap and relevant boundary/property confidence. Low-quality criticals used
by PR receive a `0.97` factor, and deep method factors are `0.95` for ordinary
Clapeyron and `0.90` for the acid relation.

The canonical aggregate is a log-pressure-span-weighted mean of selected slice
qualities, with `3x` weight for the part of each span inside `0.1-5 bar`.
If there is no positive weighted span, the implementation falls back to the
minimum slice quality. The fit penalty is then subtracted from aggregate
quality; checked pointwise lookup subtracts that penalty from the local source
quality rather than replacing local confidence with the aggregate.

The [no-hard quality rationale](../scripts/psat_experiments/no_hard_fallback_results.md#final-curve-quality)
explains the purpose: a rarely used deep-vacuum tail should not set confidence
for the whole curve, while process-relevant pressure ranges deserve more
weight. The exact `0.1-5 bar`, `3x`, method factors, and span penalty remain
engineering judgments. The evidence supports their qualitative ordering more
strongly than their exact magnitudes.

## 12. Runtime continuations

Checked property resolution is restricted to the canonical domain. The
thermodynamic hot path evaluates cached coefficients and uses explicit
continuations outside it:

```text
Below Tmin:
ln(P(T)) = ln(P(Tmin)) - Tmin^2 s_min (1/T - 1/Tmin)

Above Tc:
P(T) = P(Tc) + (dP/dT at Tc) (T - Tc)
```

The lower inverse-temperature tangent preserves value and first derivative
and approaches zero monotonically for positive boundary slope. It avoids
artificial low-temperature roots from unrestricted polynomial extrapolation.
The upper continuation is linear in pressure so numerical trial states do
not inherit exponential growth from extending ln(P).

These are numerical continuation choices. Above-critical continuation is not
physical vapor pressure; below-domain continuation does not establish stable
liquid validity. The [domain/continuation investigation](investigations/psat-domain-and-final-validation-design.md)
records the motivating glycerol false root near `211 K` and the separation
between smooth trial-state calculations and physical-state validation. Its
planned unit-operation setup/final-state validation is a separate concern,
not something this document declares implemented.

## 13. Evidence coverage and remaining qualifications

The current evidence is extensive for the principal method choices, but not
complete for every policy setting or branch.

| Area | Evidence status |
|---|---|
| Lower-AW route selection and rejected corrections | Strong comparative study, including realistic sparse derivative sensitivity |
| Log-T middle bridge | Strong synthetic-gap comparison; noisy sparse boundary derivatives not tested |
| Broad-Hvap ordinary deep completion and switches | Strong Perry/CoolProp comparison with explicit objective tradeoffs |
| Hard-boundary acid dimer versus ordinary PR/free parameter fitting | Quantitative evidence and identifiability analysis, mainly Perry-based |
| Physical domain preference and need for inverse retries | Physical motivation and broad A/B construction evidence |
| Source authority, Tb exemption, exact confidence tiers | Explicit design rationale; not a universal calibrated source ranking |
| Medium/low-quality Hvap switch thresholds | Conservative extensions; source-class-specific validation incomplete |
| Low-quality-critical Nannoolal/AW hybrid and estimated-anchor branches | Documented fallback rationale, less direct route-specific evidence |
| Complete no-hard acid construction | Report explicitly requests a separate benchmark |
| Exact fit/handoff tolerances, quality multipliers, sampling resolutions | Engineering settings without comprehensive retained calibration studies |
| Entire current assembled-and-fitted curve across all physical deep domains | Evidence distributed across studies; no single comprehensive validation report |

Several experiments stop at `0.001 bar`; physical domains may extend far below
it. A successful fit over that expanded domain is evidence of numerical
representability, not equivalent reference accuracy in those additional decades.

The middle-gap report explicitly excludes the later global fit. The lower
end-to-end study exercises production lower AW but implements candidate deep
models within the benchmark. No-hard studies often require qualified `Tb` and
omega; their aggregates cannot establish accuracy for missing-input branches.

Detailed benchmark CSVs are generally written under `/tmp`. Maintained scripts
and retained report summaries provide reproducibility, but the reports do not
consistently archive immutable full outputs with code/dependency/data revisions.
Scripts importing production internals can change their results when production
changes. Historical reported numbers should therefore remain labeled as
historical results rather than promises about a fresh run.

Older documents contain obsolete status statements, source-adapter TODOs,
pressure-floor behavior, and recommendations not selected for production.
They remain useful evidence of the investigation. This synthesis records the
current reconciliation without rewriting that history.

## 14. Maintained evidence map and reproduction

| Question | Maintained script | Retained report |
|---|---|---|
| Lower AW hard-boundary matching | [benchmark_lower_aw_matching.py](../scripts/psat_experiments/benchmark_lower_aw_matching.py) | [Lower AW matching](../scripts/psat_experiments/lower_aw_matching_results.md) |
| Direct hard-boundary deep alternatives | [benchmark_direct_lower_completion.py](../scripts/psat_experiments/benchmark_direct_lower_completion.py) | [Direct lower completion](../scripts/psat_experiments/direct_lower_completion_results.md) |
| Endpoint Nannoolal versus dynamic omega | [benchmark_direct_lower_nannoolal.py](../scripts/psat_experiments/benchmark_direct_lower_nannoolal.py) | [Direct Nannoolal comparison](../scripts/psat_experiments/direct_lower_nannoolal_results.md) |
| Production lower AW followed by deep candidates | [benchmark_end_to_end_lower_completion.py](../scripts/psat_experiments/benchmark_end_to_end_lower_completion.py) | [End-to-end lower completion](../scripts/psat_experiments/end_to_end_lower_completion_results.md) |
| Interior gap coordinate and method | [benchmark_middle_gap_completion.py](../scripts/psat_experiments/benchmark_middle_gap_completion.py) | [Middle-gap completion](../scripts/psat_experiments/middle_gap_completion_results.md) |
| No-hard omega modifications | [benchmark_no_hard_aw_relaxation.py](../scripts/psat_experiments/benchmark_no_hard_aw_relaxation.py) | [No-hard fallback](../scripts/psat_experiments/no_hard_fallback_results.md) |
| No-hard switches, PR/ideal, Nannoolal | [benchmark_no_hard_clapeyron_switch.py](../scripts/psat_experiments/benchmark_no_hard_clapeyron_switch.py) | [No-hard fallback](../scripts/psat_experiments/no_hard_fallback_results.md) |
| Acid free-parameter identifiability | [benchmark_hard_dimer_parameter_fit.py](../scripts/psat_experiments/benchmark_hard_dimer_parameter_fit.py) | [Hard-segment dimer fitting](../scripts/psat_experiments/hard_dimer_parameter_fit_results.md) |
| Generic acid entropy and size trend | [benchmark_perry_dimer_entropy.py](../scripts/psat_experiments/benchmark_perry_dimer_entropy.py) | [Perry entropy inversion](../scripts/psat_experiments/perry_dimer_entropy_results.md) |
| CoolProp source/backend and canonical comparisons | [benchmark_coolprop_psat.py](../scripts/psat_experiments/benchmark_coolprop_psat.py) | [Historical canonical/CoolProp summary](vapor_pressure_canonicalization.md#coolprop-benchmarks) |
| Exact-anchor canonical basis experiments | [benchmark_hybrid_curve_fit_constrained.py](../scripts/psat_experiments/benchmark_hybrid_curve_fit_constrained.py) | Script output; no dedicated retained report in this directory |
| Source/admission comparisons | [benchmark_other_antoine_sources.py](../scripts/psat_experiments/benchmark_other_antoine_sources.py), [benchmark_antoine_tb_gate.py](../scripts/psat_experiments/benchmark_antoine_tb_gate.py) | [Tb policy rationale](TB_VALIDATION_PLAN.md); consult script outputs for measured comparisons |

Run an individual study from the repository root with the project's normal
Python environment. The following example persists stdout and stderr and limits
scripts that honor the worker setting to four processes:

```bash
set -o pipefail
PYTHONPATH=.:scripts/psat_experiments PSAT_BENCHMARK_WORKERS=4 \
  python scripts/psat_experiments/benchmark_middle_gap_completion.py \
  2>&1 | tee /tmp/psat-policy-middle-gap.log > /dev/null
```

Use a distinct log name for each study. Consult the linked script/report for
its CSV paths, dependencies, filters, and population definitions. These commands
are reproduction instructions, not benchmarks executed for this synthesis.

## 15. Maintenance standard for future policy decisions

A change to the construction policy should retain:

1. The concrete problem and affected input populations.
2. The implemented route and alternatives considered.
3. Whether the justification is mathematical, comparative, or judgment-based.
4. Measured median, tail, worst-case, failure, and continuity behavior where
   relevant, with point-versus-curve weighting made explicit.
5. The maintained reproduction script and the exact production stages it uses.
6. Input/dependency/code revisions and durable results when publishing a new
   benchmark conclusion.
7. Excluded populations, extrapolation limits, and unresolved physical/model
   assumptions.
8. Which earlier recommendations the decision adopts or supersedes.

The purpose is a traceable argument for each consequential choice, not an
assertion that every engineering constant must have a statistically optimal
value. The existing reports already meet much of this standard for the core
methods; the main remaining work is exact-setting rationale, fallback coverage,
and consistent linkage between experiments and current production behavior.
