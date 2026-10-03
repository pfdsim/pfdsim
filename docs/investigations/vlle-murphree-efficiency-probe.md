# VLLE Murphree-efficiency feasibility and boundary probes

This is a script-only numerical experiment. Production has not been given
stage-efficiency support. The maintained script is
`scripts/performance/probe_vlle_murphree_efficiency.py`; its VLE companion is
`scripts/performance/probe_vle_murphree_efficiency.py`.

## Model and verification

The two liquids are assumed to equilibrate immediately. Their common
equilibrium vapor is a reference state, while the actual outgoing vapor obeys
a scalar Murphree efficiency on each interior tray. Condenser and reboiler
boundaries remain unchanged. Fully vaporized feed is mixed with rising vapor
before defining the incoming-vapor composition. Actual vapor compositions and
enthalpies enter component/energy balances, including the vapor EOS departure
calculation for NRTL-RK. The script reuses the existing native liquid
equilibrium, stability checks, active-set solver, and projected-boundary logic.

Additional vapor compositions are independent normalized variables. They are
carried through topology rebuilds. The sparse Jacobian extends native core
derivatives, adding local vapor/enthalpy derivatives, efficiency equations,
and feed-mixing flow derivatives. Exact local property caching is confined to
one fixed-topology model and avoids duplicate evaluations in the added
derivative pass; it is not shared across benchmark solves.

Checks include full finite-difference Jacobian comparisons at unit efficiency
and a 0.5-to-0.9 stage profile; all component efficiency equations, including
the dependent component; external mass and energy balances; liquid fugacity
equality; and reference-vapor fugacity equality where vapor equilibrium is
enforced. Actual vapor is allowed to be out of equilibrium. Physical audits
also reject identical or negligible liquid phases and compare the solved
phases/lever fraction with independent LLE calculations, allowing label swaps.

## Cases and timing

Seven binary butanol/water cases use 20 stages, feed stage 10, 100 kmol/h
40/60 feed, and 1 bar:

- Selective total condenser, NRTL, D/F=0.6119275124597854: all aqueous liquid
  leaves as distillate; all butanol-rich liquid returns internally.
- The same case with 5 K subcooling.
- Co-routed total condenser, NRTL, RR=1.2 and D/F=0.773.
- Partial condenser, NRTL, RR=2 and D/F=0.65.
- Selective total condenser, NRTL, with a 430 K fully vaporized feed.
- Selective total condenser, NRTL-RK.
- Partial condenser, NRTL-RK, RR=2 and D/F=0.65, with a 430 K vapor feed.

Liquid feeds are at 298.15 K. A ternary UNIFAC case uses 16 stages, feed stage
8, 100 kmol/h 35/25/40 ethanol/water/benzene feed at 298.15 K, RR=2,
D/F=0.4, a total condenser, and 1 bar.

Every case tests native equilibrium, enlarged E=1, E=0.9, E=0.7, E=0.5,
E=0.3, and a stage profile increasing from 0.5 to 0.9. There are three timed
repetitions and one excluded warm-up per variant; order alternates. One CPU
thread and sequential workers prevent timing interference. A 90-second worker
timeout bounds stiff cases. No RNG is used.

All variants start with the same native initializer profile, captured before
the native coupled solve; the main comparison does not use converged-state
continuation. Timing includes active-set rebuilding, screening, projection,
and Newton work but excludes property hydration, initializer construction,
and audits. Thus it is broader than the VLE companion's raw Newton timing.
These are local three-repeat medians, not normalized performance-suite pins.

## Results on October 2, 2026

165 of 168 measured solves succeeded under the default phase policy, plus 55
of 56 warm-ups. The one failing configuration was co-routed butanol at E=0.7;
all its repetitions stopped on the same topology cycle. There were no worker
timeouts. The ternary and both NRTL-RK cases passed every efficiency.

| Case | Native, ms | Enlarged E=1, ms | E=0.7, ms |
|---|---:|---:|---:|
| Selective NRTL | 337.5 | 716.2 | 448.5 |
| Selective NRTL, subcooled | 171.5 | 408.0 | 356.2 |
| Co-routed NRTL | 687.7 | 711.3 | Topology cycle |
| Partial NRTL | 82.8 | 150.1 | 195.7 |
| Selective NRTL, vapor feed | 204.9 | 462.3 | 389.4 |
| Selective NRTL-RK | 464.9 | 900.7 | 483.3 |
| Partial NRTL-RK, vapor feed | 130.4 | 234.0 | 220.6 |
| Ternary UNIFAC | 161.7 | 319.3 | 307.3 |

Among successful default-policy cases, E=0.7 cost approximately 1.04–2.36
times native. The added variables and changed topology paths account for a
larger overhead than the VLE probe. Lower efficiency does not imply uniformly
more Newton iterations. All default-policy successful solves had scaled MESH
residual at most `6.3473e-8` against a strict `1e-7` target. Unit-efficiency
aggregate stage compositions agreed with native within `5.4927e-9`.

The partial NRTL case is substantially more expensive at lower efficiencies:
E=0.5 took 417 ms (5.03 times native), and E=0.3 took 778 ms (9.39 times
native). Both converged and passed the physical audits. The E=0.7 range alone
therefore does not bound overhead for all efficiencies and topology paths.

Separate physical-audit repeats passed 110 of 112 solves, reproducing the same
two co-routed E=0.7 failures (warm-up and measured repetition). Successful
solutions had:

- Maximum independent liquid-split composition/lever-fraction error:
  `4.0512e-8`.
- Maximum liquid fugacity mismatch: `5.7399e-13` in log units.
- Maximum reference-vapor fugacity mismatch: `3.4462e-13` in log units.
- Maximum Murphree residual: `1.2416e-8`.
- Minimum retained liquid fraction: `0.0011844`, far above the `1e-6` floor.
- Minimum liquid composition L1 distance: `0.71297`; no collapsed-phase
  solution was accepted.
- Maximum relative external energy imbalance: `6.45e-14`.

The primary binary runs had maximum external component imbalance
`2.633e-11 kmol/h` and energy imbalance `1.956e-6 kJ/h`. Finite-difference
Jacobian validation errors were below approximately `2.8e-6`.

## Phase-topology weakness

The default co-routed E=0.7 attempt follows homogeneous -> 13 split trays ->
12 split trays, then attempts to reactivate the 13th. Native cycle detection
rejects that revisit. Starting from the native converged equilibrium profile
does not cure the issue and causes additional premature-contraction cycles at
other efficiencies. Residual gating is also unsuitable for the original
homogeneous initializer: it stalls before the needed liquid phases appear.

Disabling the existing projected phase-contraction heuristic lets the
co-routed case converge at every efficiency in a diagnostic sweep. A second
targeted E=0.7 recovery passed the stronger independent liquid-split audits
with MESH residual `4.7422e-13`. Its single measured solve took 2.25 seconds
after warm-up; native equilibrium without projection took 10.56 seconds.
These diagnostic times are excluded from the default-policy comparison.

This isolates sensitivity to early projected phase removal rather than a
fundamental impossibility of solving the efficiency equations. It is a
robustness issue to address before production implementation. Globally turning
projection off would sacrifice the acceleration it provides to native VLLE.
No production change was made.

## Selective butanol behavior

At fixed pressure, the saturated binary two-liquid condenser pins the aqueous
distillate composition. Together with the fixed D/F, this pins bottoms purity
and recovery while both top liquids remain present. Reduced efficiency is
compensated by internal reflux:

| Efficiency | Bottoms butanol | Recovery | Effective RR | Split stages |
|---|---:|---:|---:|---|
| 1.0 | 99.73098 mol% | 96.75712% | 1.32861 | 1, 10, 11, 12 |
| 0.7 | 99.73098 mol% | 96.75712% | 1.32899 | 1 |
| 0.5 | 99.73098 mol% | 96.75712% | 1.37547 | 1 |
| 0.3 | 99.73098 mol% | 96.75712% | 2.91901 | 1 |

Thus unchanged external purity is not evidence that the efficiency is being
ignored. The internal vapor compositions, reflux, duties, and topology respond.

## Reproduction

```bash
set -o pipefail
python scripts/performance/probe_vlle_murphree_efficiency.py \
    --output /tmp/vlle-efficiency --repeats 3 --timeout 90 \
    2>&1 | tee /tmp/vlle-efficiency.log > /dev/null
```

For the specific recovery diagnosis, add
`--cases butanol_copooled --no-projection --variants native E0.7 --repeats 1`.
`--warm-seed` and `--topology-policy` are explicit diagnostic options; their
results must not be mixed with the default-policy, fresh-profile timings.

Artifacts are under `/tmp/pfdsim-vlle-efficiency-comparison-20261002/`,
`/tmp/pfdsim-vlle-efficiency-ternary-20261002/`,
`/tmp/pfdsim-vlle-efficiency-physical-audits-20261002/`, and
`/tmp/pfdsim-vlle-efficiency-recovery-audit-20261002/`.

## Scaling projection thresholds by efficiency

A subsequent script-only experiment multiplies both the predicted smaller
liquid-fraction gate and the predicted/current contraction-ratio limit by the
efficiency on that tray. For E=0.5, the limits are 0.075 and 0.25; for E=0.7,
0.105 and 0.35. Condenser and reboiler efficiencies remain one. Stability
checks and the two-assessment confirmation streak are unchanged. The script
uses an explicit copy of the native guard solely to apply per-tray thresholds
and record confirmed decisions. At unit scaling the copy reproduces the
unscaled failing path exactly.

The co-routed E=0.7 case still fails in all three measured repetitions, with
the same 13 -> 12 -> 13 split-stage cycle. An unscaled control in the same
instrumented script also fails in all three repetitions. The decisive stage
13 removal is identical in both runs:

| Quantity | Unscaled | E-scaled |
|---|---:|---:|
| Predicted-fraction gate | 0.15 | 0.105 |
| Predicted/current ratio limit | 0.50 | 0.35 |
| Current smaller liquid fraction | 0.0072064783 | 0.0072064783 |
| Raw predicted smaller fraction | 1.86449e-11 | 1.86449e-11 |
| Raw predicted/current ratio | 2.58724e-9 | 2.58724e-9 |
| MESH residual at removal | 0.0499735823 | 0.0499735823 |

Both predicted quantities are many orders below either set of limits, so
realistic efficiency scaling cannot screen this particular decision. Native,
enlarged E=1, E=0.9, E=0.5, E=0.3, and the nonuniform profile all converged
in the scaled sweep, but the proposal did not recover the problem point.

The underlying concern remains that the guard evaluates the raw Newton step
before the solver limits its maximum absolute coordinate change to 8 and
before line-search damping. Predicting from a bounded/accepted direction is
a separate hypothesis to investigate; it was not implemented here.

Reproduce the threshold experiment by adding `--scale-projection` to the
probe command. Artifacts are in
`/tmp/pfdsim-vlle-efficiency-scaled-projection-20261002/` and
`/tmp/pfdsim-vlle-efficiency-projection-control-20261002/`.
