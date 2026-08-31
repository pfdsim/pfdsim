# Unchecked thermodynamics Psat experiment

## Change

- Backed up `thermodynamics_models/base.py` as
  `thermodynamics_models/base.py.pre_unchecked_psat_2026-07-30.bak`.
- Added `IdealThermodynamics.get_Psat_coefficients(comp)`.
- Canonical `[A..H, Tc, inverse_power]` payloads are cached once per component.
- `IdealThermodynamics.Psat(comp, T)` evaluates the canonical equation directly
  without checking `T_min <= T <= Tc`.
- Exponential overflow returns positive infinity, matching the behavior expected
  from an unchecked NumPy/compiled exponential.
- Simulator property refreshes clear both scalar Psat and coefficient caches.

## Failure comparison

The directly comparable set is every non-example method that failed in the
original six-worker run.

| Result | Original | Unchecked coefficients |
|---|---:|---:|
| Failed outcomes | 95 | 32 |
| Distinct failing methods | 65 | 27 |
| Canonical range-check failures | 57 | 0 |
| Outcomes containing `Cannot calculate vapor pressure` | 61 | 1 |

The remaining vapor-pressure construction error is CO2: canonical coefficient
construction rejects its boiling-point/critical-point fit before an unchecked
evaluation can occur.

Two of the 63 eliminated original outcomes came from the separately corrected
C3H8O3/C6H7N test baselines. Attributing only code effects, the unchecked
coefficient path eliminated 61 original failure outcomes.

## Regression check

The complete non-example suite finished with:

- 876 passed tests
- 874 passed subtests
- 40 failed outcomes across 35 methods
- Runtime: 303.94 seconds

Compared with the original recorded non-example result, 38 previously failing
methods now pass, 27 still fail, and 8 previously passing methods newly fail.
All eight new failures are quality/provenance reporting tests: direct coefficient
evaluation bypasses `_record_lazy_property_source`, so temperature-dependent
Psat source contexts are no longer recorded.

## Full-suite limitation

The full suite and a second run excluding only the all-examples sweep both
reached their final few tests but did not terminate in a reasonable window.
Example simulations that previously failed immediately at a Psat range boundary
now enter long solver iterations. Those runs were interrupted and are not
reported as completed suites.
