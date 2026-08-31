# pfdsim failure-source investigation

## Scope

This investigation excludes `tests/test_examples.py` and starts from the
six-worker canonical run recorded in `/tmp/pfdsim-test-results.xml`.

- Non-example failed outcomes: 95 across 65 test methods
- Previously classified small numerical drifts: 20 outcomes
- Other failures investigated here: 75 outcomes

## Legacy resolver A/B result

The preserved pointwise resolver in
`property_resolution/vapor_pressure.py.bak` was installed temporarily through
a pytest-only plugin. The 65 previously failing non-example methods were rerun
with six workers. The plugin was then removed and a representative canonical
failure was rerun to verify that the current route was restored.

- All non-example failures fixed by legacy: 89 of 95 outcomes
- Small drifts fixed by legacy: 16 of 20 outcomes
- Other failures fixed by legacy: 73 of 75 outcomes
- Effective failures initially remaining under legacy: 6 outcomes
  - Four small drifts
  - Two substantive failures
- Two stale test baselines were subsequently corrected, leaving three small
  drifts and the top-decanter phase-label failure.

## Root-cause breakdown for the 75 non-drift failures

### 1. Canonical domain contract versus solver search bounds: 65 outcomes

The canonical runtime rejects calls outside `curve.T_min <= T <= Tc`. Its
default lower pressure floor is `0.001 bar`. Rigorous-column temperature
bounds are independently constructed as `0.45 * min(Tb)` at the lower end and
can exceed component critical temperatures at the upper end. Bubble-point and
feed-condition searches evaluate every component at both endpoints.

This produces calls such as:

- Water or methanol at 151.9 K
- Methanol at 9.2 K
- Nitrogen or oxygen near 298 K, well above their critical temperatures
- Ethylene at 313 K, above its critical temperature

The legacy route supplies an initialization-oriented effective pressure above
`Tc` and uses Ambrose-Walton, Lee-Kesler, Clausius-Clapeyron, or Antoine
extrapolation below source ranges. Sixty failures pass immediately when that
contract is restored.

Five additional failures originally looked unrelated because the canonical
range exceptions were caught internally and surfaced later as incorrect
results. A canonical-in-range/legacy-out-of-range A/B route fixed all five:

- Bubble-point scanning above methanol's critical pressure
- Superheated ethylene expansion
- Gas/gas heat-exchanger service classification for nitrogen and oxygen
- Heater PH inversion for nitrogen and oxygen
- Rigorous-distillation initializer routing

This is a caller/resolver contract mismatch. A qualified physical saturation
curve is intentionally undefined above `Tc`, while the solvers currently
expect a total numerical function over broad search brackets.

### 2. In-range canonical-versus-legacy differences: 8 outcomes

The canonical adapter prioritizes CoolProp ahead of local correlations and
then evaluates a fitted canonical polynomial. The legacy route prioritizes
direct tables, provided correlations, Perry correlations, and Antoine before
corresponding-states fallbacks.

Only one of these eight has a large Psat change:

- The PFD alias/Psat-override test gets canonical ethanol/CoolProp Psat instead
  of its explicit PFD `poly_x` correlation. At 300 K the values are
  `0.0803869` versus `0.0546050 bar`, a `47.2%` difference. This is a PFD
  override-precedence/integration bug, not numerical sensitivity.

One has moderate, unsurprising propagation:

- The benzene/toluene pressure-unit flash changes Psat by `0.14-0.28%` at the
  flash temperature and shifts toluene recovery by `0.85%`. The pressure-unit
  conversion itself remains correct; the hard-coded recovery baseline belongs
  to the legacy property route.

The other six are numerically or discretely delicate:

- Water/chloroform binary-invariant VLLE changes Psat by only `0.15-0.18%`.
  The fixed legacy invariant temperature is then classified as ordinary VLE.
  This affects both the compiled-backend and Flash3 status tests.
- The ternary water/ethanol/cyclohexane reference changes Psat by
  `0.16-0.88%`; its hard-coded bubble temperature moves by `0.0114 K`.
- NRTL VLLE PH and PS use self-generated canonical reference enthalpy/entropy,
  yet select a different root after `0.16-0.32%` Psat changes. These are
  genuine root-selection sensitivity issues.
- The McCabe-Thiele result moves from 16 to 17 stages with roughly
  `0.1-0.2%` Psat changes near its reference temperature, a discrete
  stage-boundary effect.

All eight pass with the exact legacy pointwise route, but only the PFD override
case represents a large vapor-pressure change.

### 3. Remaining substantive failures: 2 outcomes

#### Top-decanter phase labeling

`test_rigorous_distillation_top_decanter_selects_reflux_phase_and_purge`
gets past the canonical range error with legacy Psat, then converges with the
two liquid phase labels reversed. `top_decanter_solution` always assigns the
decoded stage liquid to reflux phase 2 and the separately decoded liquid to
distillate phase 1. The requested benzene-rich reflux component is used for
initialization and checked only after convergence; it is not used to
canonicalize or swap the symmetric LLE phase labels.

#### Incorrect C6H7N test expectation

`test_curated_reduced_psat_records_preserve_internal_criticals` expects the
aniline scalar criticals and reduced-Psat internal parameters to agree at
`Tc=705.0 K` and `Pc=56.3 bar`. That expectation is incorrect. The more
accurate physical scalar criticals are `Tc=698.8 K`, `Pc=53.1 bar`, while the
self-contained reduced-Psat fit legitimately uses `Tc_K=705` and
`Pc_Pa=5,630,000` as its internal reducing parameters.

The adapter deliberately reads those internal parameters from the correlation
instead of substituting the component scalars, and a separate adapter test
explicitly verifies that the two pairs may differ. The correct expected tuple
for C6H7N is therefore `(698.8, 53.1, 705.0, 56.3)`. This is a test-baseline
error independent of the runtime vapor-pressure resolver; `chemicals.json`
should not be changed to satisfy it.

This test expectation has since been corrected.

## Remaining small drifts under legacy

- Two glycerol pipe-result baselines
- Glycerol effective critical volume
