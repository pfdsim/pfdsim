# Native rigorous-column vapor Murphree efficiencies

## Scope and implementation

RigorousDistillation supports scalar, ordered-profile, and stage/range-map
efficiency specifications in VLE, adaptive VL(L)E, and VLLE modes. All
components share the efficiency on a tray. Condensers and reboilers retain
their previous equilibrium/temperature boundaries. The model assumes a common
tray temperature and saturated-liquid equilibrium reference; this remains an
empirical efficiency model, not a mass-transfer-rate or two-temperature model.
In VLLE the two liquids equilibrate immediately, while outgoing vapor can
differ from their common equilibrium vapor.

The authoritative VLE and VLLE models share `VaporStageEfficiencies` for
vapor variables, feed mixing, residuals, derivatives, and reporting. A vapor
composition is an independent unknown only on a tray where E differs from
one. Default/explicit E=1 therefore retains the original smaller system.
Mixed/two-phase feeds use actual feed-vapor compositions; vapor is mixed with
rising gas on a molar-flow basis before applying the efficiency equation.
Actual vapor composition and calorics, including EOS enthalpy departures,
enter balances and vapor side draws. Liquid stability and vapor-reference
equilibrium audits remain separate from the Murphree residual audit.

VLE packing and VLLE profiles retain actual vapor through recycle warm starts,
coarse initialization, and active-topology rebuilds. Approximate CMO/graphical
initializers provide equilibrium endpoint estimates; a rigorous coarse column
uses a remapped efficiency profile. The compact PFD parser validates fractions,
profile lengths, inclusive ranges, overlaps, units, and end-stage restrictions.
Unsupported approximate columns reject efficiency input.

The first repeated VLLE topology disables projected phase contraction only for
the remainder of that coupled solve. The solver keeps the current iterate,
actual vapor variables, normal phase-stability checks, full work/history, and
the original topology budget. A second cycle still fails; configuration is
not mutated. `vlle_projection_cycle_fallback=false` disables this recovery for
diagnostics or to exercise alternate-profile recovery.

## Integration and numerical validation

`tests/test_distillation_efficiencies.py` contains unittest cases for input
contracts, efficiency/phase balances, variable profiles, partial/mixed/subcooled
condensers, vapor/two-phase/multiple feeds, mass specifications, side draws,
NRTL-RK and PR, VLLE liquid equilibrium, independent Jacobian comparisons,
recycle states, explicit initializers, and bounded cycle recovery accounting.

The complete compact-PFD integration test runs the same 20-stage methanol/water
NRTL column at unit and 50% efficiency. Distillate purity falls from
99.7664 mol% to 97.6169 mol%. It pins a reduction greater than 2 percentage
points and verifies convergence and balances through the public Simulator.

Existing tests specifically exercising alternate-initializer recovery disable
the new lower-level cycle recovery, preserving their original test purpose.
Default equilibrium, routing, subcooling, parser, and package regressions are
also checked; no full or performance suite is required for this change.

## Native public-solve timing, October 2, 2026

The maintained benchmark is
`scripts/performance/benchmark_native_stage_efficiencies.py`. It calls public
RigorousDistillation.solve without monkeypatching. Timings include initializer
construction and audits, exclude imports/property hydration, use one thread,
sequential workers, alternating variant order, one excluded warm-up per variant,
and three measured repetitions. There is no randomness or continuation seed.
Workers have a 90-second timeout and persist incremental results and logs.

| Case | Default, ms | Explicit E=1, ms | E=0.7, ms | E=0.5, ms |
|---|---:|---:|---:|---:|
| Methanol/water NRTL VLE | 69.7 | 66.3 | 78.2 | 66.2 |
| Benzene/toluene PR, partial condenser, vapor feed | 68.5 | 68.6 | 74.0 | 73.0 |
| Butanol/water NRTL VLLE, selective overhead | 367.4 | 378.6 | 319.4 | 250.0 |
| Butanol/water NRTL VLLE, co-routed overhead | 648.8 | 614.1 | 1503.2 | 564.6 |
| Butanol/water NRTL-RK VLLE, partial condenser, vapor feed | 236.6 | 224.7 | 258.1 | 247.6 |

All 60 measured solves and 20 warm-ups passed. These small local medians are
not normalized-suite baselines or evidence that efficiency universally improves
solver speed. The co-routed E=0.7 case incurred one phase-cycle recovery per
solve; other cases did not. The selective column retains external purity and
recovery at fixed cut, but changes reflux/duties/topology as expected.

Across those native runs:

- Maximum MESH residual: `1.3608e-8`, against `1e-7`.
- Maximum Murphree equation residual: `1.5009e-9`.
- Maximum external component imbalance: `5.623e-9 kmol/h`.
- Maximum external energy imbalance: `2.046e-4 kJ/h`.

Outputs are saved under
`/tmp/pfdsim-native-stage-efficiency-benchmark-20261002/`.

```bash
set -o pipefail
python scripts/performance/benchmark_native_stage_efficiencies.py \
    --output /tmp/native-efficiency --repeats 3 --timeout 90 \
    2>&1 | tee /tmp/native-efficiency.log > /dev/null
```

The earlier VLE/VLLE script probes remain as historical experiments in the
adjacent reports. They use different formulations/timing boundaries; their
numbers should not be substituted for this native public-solve comparison.
