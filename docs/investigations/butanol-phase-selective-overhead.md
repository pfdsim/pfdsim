# Butanol recovery with phase-selective overhead withdrawal

Investigation on 2026-10-01 using the production NRTL equilibrium/stability
implementation and script-only top-routing changes. The maintained harness is
[probe_butanol_phase_selective_distillation.py](../../scripts/probe_butanol_phase_selective_distillation.py).
It changes no production equations or runtime parameter databases.

## Column and feed

- 100 kmol/h feed, 40 mol% 1-butanol and 60 mol% water, 298.15 K.
- 20 equilibrium stages, feed on stage 10, 1 bar throughout.
- Total condenser and VLLE-capable stage model.
- No external recycle, purge, entrainer, or extra separation unit.
- Internal reflux may return the remaining amounts of both top liquids.

The reference is the existing near-pure-bottoms VLLE test, with D/F=0.773 and
reflux ratio 1.2. Its top liquids are withdrawn together in their equilibrium
proportions. The alternative independently specifies the withdrawn fractions
of the aqueous and butanol-rich liquids; its reflux ratio is then determined
by the coupled equilibrium and material balances.

## A feasible improvement at identical purity

Withdraw 100% of the aqueous top liquid and 0% of the butanol-rich top liquid.
Return all the butanol-rich liquid internally. Set D/F=0.61192751246.

| Quantity | Reference | Phase-selective |
|---|---:|---:|
| Bottoms butanol mole fraction | 0.997309789010 | 0.997309789009 |
| Butanol recovery in bottoms | 56.59733% | 96.75712% |
| Butanol lost in distillate, kmol/h | 17.361068 | 1.297151 |
| Distillate, kmol/h | 77.300000 | 61.192751 |
| Bottoms, kmol/h | 22.700000 | 38.807249 |
| Net internal reflux, kmol/h | 92.760000 | 81.301264 |
| Reflux/distillate ratio | 1.200000 | 1.328609 |
| Reboiler duty, kW | 2222.259 | 1888.496 |
| Condenser duty, kW | -1873.406 | -1569.735 |

Butanol loss falls by 92.53%; recovered butanol increases from 22.638932 to
38.702849 kmol/h. Reboiler duty falls by approximately 15.02%.

At the top, NRTL predicts a binary three-phase invariant at 93.03958 C with
aqueous liquid x_butanol=0.02119778743 and butanol-rich liquid
x_butanol=0.37768245054. The latter is butanol-rich relative to the other liquid;
it is still 62.23 mol% water. The distillate is therefore 97.88022 mol% water.

The selective column topology is `L........LLL........`: stages 1, 10, 11,
and 12 have two liquids; other stages have one. Stage 1 is the total condenser,
so there is no external vapor product. The reference topology is
`LLLLLLLLLLLLLL......`.

## Search and limiting recovery

The first scan compared 29 routing/cut combinations. It varied the aqueous
withdrawn fraction through 1.0, 0.8, 0.6, 0.454545, and 0.3, including selective
aqueous-only product and controls that also withdrew butanol-rich liquid.
Twenty continuation cases then used successful numerical profiles as starting
guesses. Each child process had a 45-second timeout, with at most five processes
and one numeric-library thread per process.

The combined refinement report contains 51 case records: 11 passed the audits
and 40 failed numerically. These counts include the reference and a tight
confirmation. Failed starts are not evidence of physical infeasibility.
In particular, some closer-to-99.5% cuts stalled even when nearby cases were
available as starting profiles. This does not affect the demonstrated solution.

Returning 20% or 40% of the aqueous condensate still yielded the same recovery
at matched purity, but reboiler duties increased to approximately 2280.93 and
2934.99 kW respectively. Complete aqueous withdrawal is preferable among those
validated matched-purity cases.

At this binary invariant the phase compositions are fixed. Any blend of both
liquids in the distillate has at least the aqueous phase's butanol fraction.
For feed fraction z, distillate fraction xD, target bottoms purity p, and cut d:

```text
d = (p-z)/(p-xD)
butanol recovery = p*(1-d)/z
```

For p=0.995, z=0.4, and aqueous-only xD=0.02119778743, the limiting cut is
0.61100703235 and limiting recovery is 96.76200%. With a strict greater-than
purity requirement this is a supremum, not a demonstrated optimum. The
matched-purity solution is within 0.00488 percentage points of that bound.
This bound applies to the fixed-pressure, equilibrium two-liquid total
condenser studied here; it is not a global bound across other pressures,
subcooled condensers, different thermodynamic models, or extra equipment.

A cold-start selective case at D/F=0.61180431066 also converged to 99.7 mol%
bottoms and 96.75778% recovery. Its independent confirmation reached a MESH
residual of 4.12e-13 and maximum log-fugacity mismatch of 4.44e-16.

## Numerical and physical validation

The matched-purity selective case has:

- MESH residual 1.55e-10.
- Maximum three-phase log-fugacity mismatch 4.44e-16.
- Maximum external component-balance error 3.28e-11 kmol/h.
- External energy-balance residual 3.89e-6 kJ/h.
- Phase-specific component and enthalpy routing into stage 2.

The probe retains the production topology assessment at every Jacobian
boundary. It replaces only derivative assembly with colored finite differences
for the modified routing equations. Skipping those topology checks can strand
nearly pure-bottoms problems in singular systems with ghost phases.

The production output adapter assumes co-routed distillate. The probe therefore
stops before that adapter and constructs products and balance audits from the
actual phase-specific solution. This is an experimental design, not a new
supported `.pfd` routing option.

## Reproduction and artifacts

```bash
set -o pipefail
python scripts/probe_butanol_phase_selective_distillation.py \
  --output /tmp/butanol-routing-next --workers 5 --timeout 45 \
  2>&1 | tee /tmp/butanol-routing-next.log > /dev/null
```

The complete scan writes per-case JSON and logs, a JSONL record, a manifest,
and summary. A further refinement can reuse successful profiles without
repeating the scan using `--refine-from` and a fresh output directory.

Investigation artifacts are in
`/tmp/pfdsim-butanol-phase-routing-topology-20261001` and
`/tmp/pfdsim-butanol-routing-refinement-20261001`. The matched-purity case is
`warm_a1.0000_pure0.99730979.json` in the latter directory. Its inputs refer to a
saved starting profile; that is only numerical initialization, not an external
recycle stream. Source/dependency provenance is retained in the manifests.
