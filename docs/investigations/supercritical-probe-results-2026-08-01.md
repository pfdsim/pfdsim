# CO2 and linear-supercritical Psat probe

## Runtime-only patch

- CO2 bypassed canonical coefficient construction and used Ambrose-Walton
  below `Tc`.
- At and above `Tc`, every component used
  `P = Pc + (dP/dT at Tc) * (T - Tc)`.
- The canonical coefficient derivative supplied `dP/dT` when available;
  Ambrose-Walton supplied the fallback slope.
- Every non-example test call had a strict 60-second signal timeout.

## Focused examples

All five examples that previously failed or timed out passed with the patch:

- `3methylpyridine_ether_extraction_recycle.pfd`
- `ammonia_synthesis.pfd`
- `ethylene_oxide.pfd`
- `ethylene_oxide_simple.pfd`
- `methanol_synthesis.pfd`

## Complete non-example suite

- Passed tests: 876
- Passed subtests: 874
- Failed outcomes: 40
- Distinct failing methods: 35
- Per-test timeouts: 1
- Runtime: 328.50 seconds

The unpatched unchecked-coefficient run also had 40 failed outcomes across 35
methods. The patch therefore produced no net reduction in the complete
non-example suite.

Fixed methods:

- `test_rigorous_absorber_and_stripper_report_initializer_paths`
- `test_shortcut_distillation_uses_partial_condenser_for_noncondensables`

Newly failing/exposed methods:

- `test_rigorous_distillation_top_decanter_selects_reflux_phase_and_purge`
  reaches the known reversed liquid-phase labeling failure.
- `test_rigorous_distillation_fails_fast_for_noncondensables` exceeds the
  strict 60-second timeout instead of rejecting the case quickly.

The runtime-only monkeypatch was removed after the run.
