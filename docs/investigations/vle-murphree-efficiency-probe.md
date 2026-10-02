# VLE Murphree vapor-efficiency probe

## Scope and formulation

This is a script-only experiment, not production support for tray efficiencies.
The maintained script is
`scripts/performance/probe_vle_murphree_efficiency.py`. It reuses native stage
thermodynamics, the total-condenser boundary, and sparse damped Newton solver.
Only interior trays receive efficiencies; the condenser and reboiler remain
unchanged. Liquid feeds, total condensers, and single-liquid VLE are considered.
Vapor feeds, side draws, partial/mixed condensers, gamma-phi/EOS methods, and
VLLE were not evaluated.

Each interior tray carries an independent outgoing vapor composition and obeys

\[
y_{i,j}=y_{i,j+1}+E_j\left(y^*_{i,j}-y_{i,j+1}\right).
\]

One efficiency applies to all components on a tray. The existing bubble-point
closure defines the equilibrium reference vapor; actual outgoing vapor
compositions and enthalpies enter component and energy balances. There are
`nc - 1` independent vapor-composition unknowns and efficiency equations per
interior tray; the omitted component equation follows from normalization. For
a 20-stage binary column this increases the state from 82 to 100 unknowns;
for a 20-stage ternary column, from 102 to 138.

The script assembles neighbor and flow derivatives analytically and evaluates
thermodynamic derivatives locally. Stage-only decoding avoids quadratic work
inside the Jacobian. This is an empirical efficiency model, not a rate-based
mass-transfer or equipment-sizing model.

## Cases and measurement method

All feeds are 100 kmol/h, with a 1 bar constant pressure profile. The feed is
introduced halfway down each column.

| Case | Stages | Method | Feed composition | Feed temperature | RR | D/F |
|---|---:|---|---|---:|---:|---:|
| Methanol/water | 20 | NRTL | 0.4/0.6 | 298.15 K | 2 | 0.35 |
| Methanol/water | 40 | NRTL | 0.4/0.6 | 298.15 K | 2 | 0.35 |
| Methanol/ethanol/water | 20 | IDEAL | 0.2/0.3/0.5 | 360 K | 2 | 0.40 |
| Ethanol/water | 20 | UNIFAC | 0.1/0.9 | 353.15 K | 4 | 0.11 |
| Subcooled methanol/water | 20 | NRTL | 0.4/0.6 | 298.15 K | 2 | 0.35 |

The last case specifies 5 K condenser subcooling. The ethanol/water operating
conditions follow `examples/ethanol_distillation_rigorous.pfd`; all cases use
the same generic estimate initialization to compare direct solver behavior.

Each case tests native equilibrium stages, the enlarged model with efficiencies
1.0, 0.9, 0.7, 0.5 and 0.3, and a linear interior-tray efficiency profile from
0.5 at the top to 0.9 at the bottom. All variants start from the same fresh
generic equilibrium estimate; no continuation or preceding solution is used.

Measurements use one CPU thread, sequential workers, three measured solves per
variant, one excluded warm-up per variant, and alternating forward/reverse
variant order. Timings cover the raw MESH solve, excluding imports, property
hydration, initialization, audits, and output. Every worker has a 90-second
timeout and saves its complete stdout/stderr and incremental results.
No random inputs are used. Timings are local medians, not normalized suite
baselines or precise forecasts for a production implementation.

Reproduce with a new output directory:

```bash
set -o pipefail
python scripts/performance/probe_vle_murphree_efficiency.py \
    --output /tmp/murphree-probe --repeats 3 --timeout 90 \
    2>&1 | tee /tmp/murphree-probe.log > /dev/null
```

## October 2, 2026 results

All 105 measured solves and 35 warm-ups converged, without timeouts or
continuation seeds. Each solve required a residual below `1e-7`; no relaxed
acceptance threshold was used. Newton counts ranged from 5 to 12.

| Case | Native, ms | Enlarged E=1, ms | E=0.7, ms | E=0.7/native |
|---|---:|---:|---:|---:|
| Methanol/water, 20 stages | 58.77 | 76.73 | 70.95 | 1.207 |
| Methanol/water, 40 stages | 149.02 | 217.74 | 167.91 | 1.127 |
| Ternary, 20 stages | 75.02 | 116.25 | 86.77 | 1.157 |
| Ethanol/water, 20 stages | 63.08 | 79.00 | 72.62 | 1.151 |
| Subcooled methanol/water, 20 stages | 58.55 | 79.84 | 63.24 | 1.080 |

At unit efficiency, the extra state/equations cost about 25–55% relative to
the native eliminated-vapor formulation. At 70% efficiency the observed cost
is 8–21% higher; lower efficiencies often reduce Newton iterations, partly
offsetting the larger system. At 50% efficiency the timing ratios range from
0.98 to 1.12. These results do not imply that lower efficiency generally makes
arbitrary columns easier to solve.

Numerical checks across the completed runs:

- Maximum scaled MESH residual: `7.7823e-8`.
- Maximum Murphree equation residual, including the dependent component:
  `4.7311e-9`.
- Maximum external component imbalance: `1.0717e-6 kmol/h`.
- Maximum external energy imbalance: `0.05270 kJ/h`, at most `1.9697e-9` of
  the solve's energy scale.
- Maximum stage-composition difference between E=1 and native VLE:
  `1.7847e-9`.
- Local Jacobians checked against full colored finite differences at both
  unit efficiency and a nonuniform efficiency profile; maximum scaled error
  approximately `2.597e-6`.

The 20-stage methanol/water column responds in the expected separation
direction: distillate methanol decreases from 99.7664 mol% at unit efficiency
to 99.0090% at E=0.7, 97.6169% at E=0.5, and 93.8785% at E=0.3.

Results were saved under
`/tmp/pfdsim-vle-murphree-comparison-validated-20261002/`. Earlier development
runs are excluded: one preceded stage-local decoding, and another caught an
incorrect reuse of vapor enthalpy for a perturbed equilibrium reboiler. The
reported run uses the corrected Jacobian and passed all validation gates.

The evidence supports numerical feasibility for these VLE cases, with a
moderate overhead. It is not evidence yet for VLLE efficiencies or all column
boundary conditions.
