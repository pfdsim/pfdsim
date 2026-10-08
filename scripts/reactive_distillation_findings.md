# Reactive distillation feasibility investigation

Date: 2026-10-08. Production code was not changed.

Reactive stage equations fit the existing rigorous column solver cleanly and
converged quickly in the tested esterification cases. The principal obstacle
to a physically credible implementation is existing thermodynamic reference
consistency, rather than Newton convergence. All conversions below are
**numerical feasibility results, not validated process predictions**.

## Experiment and equations

The maintained probe is `scripts/probe_reactive_distillation.py`. It calls the
production VLE MESH builder, state transforms, sparse Newton solver, reaction
parser, equilibrium constant calculator and reversible kinetic rate evaluator.
Only the stage reaction source terms, extent variables and their closures are
added in the script. No production hooks, duplicated column solver or empirical
property override was introduced.

Reaction:

    ethanol + acetic acid <=> ethyl acetate + water

Default conditions: 100 kmol/h feed, 50 mol% ethanol and 50 mol% acetic acid,
exactly zero ester/water in the feed, 350 K liquid feed, 1 bar throughout,
six stages including the total condenser and equilibrium reboiler, feed on
stage 3, reflux ratio 2, distillate 50 kmol/h. Stages 2–5 react; the condenser
and reboiler do not. The composition seed is 25 mol% of each component; this
seed does not add products to the actual material balance.

For stage j and species i, add `sum_r(nu_ir * xi_jr)` to the existing
`inflow - outflow` component balance. The reaction extent `xi_jr` is signed
and expressed in kmol/h; its solver variable is `xi_jr / feed_flow`.
Reverse reaction must be permitted: several equilibrium cases exhibit
hydrolysis on individual stages despite positive overall ester production.

Each active stage/reaction adds one unknown and one equation:

* Equilibrium: `sum_i(nu_ir * log(a_ij)) - log(K_r(T_j)) = 0`.
* Liquid kinetics: `xi_jr - liquid_holdup_j * rate_r(T_j, x_j, P_j) = 0`.
  Fixed-catalyst kinetics would instead use catalyst mass and the existing
  catalyst-mass rate units.

The prototype's reversible activity-based forward prefactor is A=2000 in
kmol/(m3 h), Ea=0 J/mol, with default liquid holdup 1 m3 per reactive stage.
Thus the existing evaluator supplies
`r = A*a_ethanol*a_acetic_acid*(1 - Q/K)`.
These values deliberately exercise coupling and stiffness; they are not a
fitted esterification/catalyst kinetic model. Activity references are the
backend's dimensionless 1-bar fugacity references, not simply mole fractions.

The existing energy equations already include species formation enthalpies.
An additional `-delta_H_reaction * xi` term would double-count reaction heat
on this basis. This fact does not repair the reference inconsistency found
below: conservation of the implemented enthalpy and thermodynamic consistency
are separate checks.

With C components and N stages, the current equilibrium-stage VLE core has
`N*(C+2)+2` unknowns. Four extra extents make the six-stage, four-component
experiment 42-by-42. The 20-stage model has 140 unknowns, including its 18
reaction extents. Reaction closures depend on local temperature/composition;
reaction sources depend on local extents, preserving sparse stage coupling.

Newly formed species receive component scales of 50 kmol/h rather than a tiny
zero-feed scale. Positive softmax composition seeds and signed linear extents
avoid artificial reaction-product feed and one-way reaction restrictions.

## Numerical results

All 32 successfully evaluated column cases met a strict scaled infinity-norm
residual tolerance of 1e-8. No loose residual acceptance was used. Numerical
success does not establish physical correctness, particularly for NRTL-VDM.

Representative six-stage results:

| Thermodynamics | Closure | Ethanol conversion, % | Distillate ester mole fraction | Newton iterations | Solve seconds | Residual infinity norm |
|---|---|---:|---:|---:|---:|---:|
| IDEAL | equilibrium | 85.7427 | 0.683782 | 6 | 0.1895 | 6.09e-13 |
| IDEAL | kinetic | 76.2971 | 0.605931 | 6 | 0.2340 | 2.72e-9 |
| NRTL | equilibrium | 75.8005 | 0.549442 | 6 | 0.2491 | 1.02e-12 |
| NRTL | kinetic | 63.8095 | 0.513749 | 7 | 0.3749 | 2.43e-10 |
| NRTL-VDM | equilibrium | 73.6805 | 0.546036 | 7 | 0.3440 | 1.25e-10 |
| NRTL-VDM | kinetic | 61.7770 | 0.507813 | 9 | 0.6006 | 3.58e-12 |
| NRTL-BV, HOC | equilibrium | 63.6313 | 0.517319 | 7 | 0.7536 | 2.99e-11 |
| NRTL-BV, HOC | kinetic | 50.7177 | 0.453401 | 8 | 0.8636 | 4.96e-10 |

Seconds measure the nonlinear solve, excluding imports, property construction,
initialization and post-solve audits. Worker calculation time including those
steps ranged from 3.36 to 12.37 seconds. Measurements were taken with at most
three concurrent cases, one numerical-library thread per process; they are
indicative timings, not controlled performance benchmarks.

Across all 32 cases, independent external audits found:

* Maximum reaction-adjusted species imbalance: 1.5361e-7 kmol/h.
* Maximum elemental imbalance: 1.8213e-7 kmol-atoms/h.
* Maximum energy imbalance: 6.9168e-6 kW.
* Maximum scaled equation residual: 9.2164e-9.

Tabulated molecular weights have a reaction mass discrepancy of 0.0013
kg/kmol extent. Consequently, the reported molecular-weight-based mass audit
can differ by about 0.056 kg/h even with tightly closed elemental balances.
This is a rounded-property-data discrepancy, not missing component flow.

The initial basic run had two IDEAL report-generation errors because the
report called an activity-coefficient method that IDEAL does not implement,
and two HOC initialization errors because isolated caches lack quantum
geometry dependencies. The reporting error was corrected. HOC was then
evaluated using the existing explicitly requested force-field radius estimator.
Those four setup/report errors are preserved in the initial logs and are not
classified as failed Newton solves.

### Robustness, derivatives and sensitivity

The following observations cover every successfully evaluated case; the full
case table follows.

* Independently seeded composition perturbations reproduced the same root
  for IDEAL seeds 17/42/123, NRTL seeds 17/42 and HOC seeds 17/42. These use
  Dirichlet(3,3,3,3) stage compositions and recomputed bubble temperatures.
* Twenty-stage equilibrium columns converged in seven iterations for IDEAL,
  NRTL and HOC. Solve times were 0.8344, 0.9452 and 2.2508 seconds respectively.
  Their numerical conversions were 83.1463%, 71.3188% and 59.1131%.
* NRTL kinetic/equilibrium continuation used strengths
  0.02/0.05/0.1/0.25/0.5/0.75/1. It reproduced the direct roots but required
  30/28 total Newton iterations and 1.3212/0.9835 seconds, versus 7/6
  iterations directly. Direct solving is adequate for these cases;
  continuation is a useful fallback, not evidence that every RD design is easy.
* IDEAL kinetic liquid holdups of 0.1, 1 and 10 m3/stage gave numerical
  conversions of 48.2565%, 76.2971% and 84.5232%, approaching the 85.7427%
  equilibrium-stage result. Holdup cannot be replaced by liquid throughput.
* Pure ester/water feed converged with net extent -7.3632 kmol/h, demonstrating
  reverse reaction without an artificial positivity constraint. The table's
  -14.7264% is signed extent divided by 50 kmol/h; hydrolysis conversion is
  positive 14.7264% of the ester feed.
* Separate 50 kmol/h acid and ethanol feeds on stages 2 and 9 of ten stages
  gave 77.0546% NRTL and 71.0486% HOC conversion. Each feed's own enthalpy was
  used; premixing enthalpy was not silently substituted.
* NRTL distillate specifications 40 and 60 kmol/h both converged in six
  iterations, at numerical conversions 76.1012% and 76.3424%.
* An augmented Jacobian retained the production semi-analytic/local-thermo
  MESH block, added exact extent-source derivatives, and finite-differenced
  only the local reaction closure blocks. Its maximum error against a separate
  dense central-difference Jacobian, scaled by `1+abs(reference)`, was
  2.906e-6 across four comparisons. Derivatives outside the declared sparsity
  were exactly zero in that audit. NRTL equilibrium/kinetic solve times were
  0.1139/0.2429 seconds, versus 0.2491/0.3749 for full colored differencing.
  HOC equilibrium/kinetic times were 0.7661/0.6288 versus 0.7536/0.8636;
  speedups should not be assumed for every package.
* Production spinodal checks and production LLE searches found no unstable
  liquid stages or liquid splits in the tested nonideal profiles. A negative
  search result is not a proof of global stability and does not establish
  suitability for a cooled overhead decanter.

The scalar homogeneous-liquid equilibrium comparator at 350 K gave forward
conversion 88.3574% IDEAL, 79.2562% NRTL and 67.9700% HOC. Thus these arbitrary
single-feed column specifications do **not** demonstrate that adding a column
necessarily improves conversion. More equilibrium stages improved overhead
ester enrichment while reducing conversion in the tested fixed-specification
cases. Feed placement, product routing and operating specifications must be
designed for the intended reactive separation.

## Physical defects exposed by the probes

### Caloric/equilibrium inconsistency

For the pure-liquid reaction standard state, thermodynamic consistency
requires `delta_H = R*T*T*d(log(K_liquid))/dT`, with pressure fixed. The audit
converted pfdsim's gas-standard K to its actual pure-liquid fugacity reference,
removed activity coefficients, used a central temperature difference of
0.01 K, and independently obtained reaction enthalpy from pure liquid stream
enthalpies. At 350 K and 1 bar:

| Backend | Current K_liquid | Enthalpy from van't Hoff, kJ/mol | Enthalpy from streams, kJ/mol | Stream minus van't Hoff, kJ/mol |
|---|---:|---:|---:|---:|
| IDEAL | 57.5951 | -12.7196 | -31.3687 | -18.6491 |
| NRTL | 57.6724 | -12.6902 | -31.3687 | -18.6785 |
| NRTL-BV, HOC | 18.2861 | -4.5946 | -31.3687 | -26.7741 |
| NRTL-VDM, current reaction interface | 57.6724 | -12.6902 | -3.5272 | +9.1629 |

This violates the caloric/equilibrium identity; the differences are much larger
than numerical differentiation noise. NRTL-VDM also has the separate reference
bug below, so its row audits the *current reaction interface*, not a corrected
VDM liquid standard state.

The underlying paths are `base.py::standard_chemical_potential`,
`reaction_log_equilibrium_constant`, `component_activities`, and
`enthalpy_liquid`, together with the activity/gamma-phi pure-liquid reference
functions. Chemical potential uses gas formation enthalpy/absolute entropy/
gas heat capacities plus the liquid fugacity reference. Liquid stream enthalpy
is independently anchored as gas enthalpy minus Hvap at 298.15 K and integrated
with liquid Cp. Those independent curves do not satisfy the same temperature
derivative. HOC additionally changes saturated fugacity through vapor
association. Passing an external energy balance does not resolve this defect.

A production correction must establish one authoritative phase chemical
potential/caloric reference and make its temperature derivative and stream
enthalpy consistent, including excess enthalpy and association contributions.
The alternatives are to derive compatible caloric curves from the fugacity
model, or supply a consistent fitted liquid Gibbs/enthalpy/Cp reference and
derive compatible fugacity/reaction properties. Selecting and validating that
change affects existing nonreactive column and reactor duties; a local RD heat
correction would leave competing thermodynamics and is not an adequate fix.

### NRTL-VDM reaction activities bypass the VLE reference

`base.py::component_activities` calls `_gamma_phi_reference_factors` directly
for activity-coefficient liquids. VDM overrides
`activity.py::_liquid_fugacity_reference_factors` to use saturated monomer
fugacity, but the reaction interface bypasses this authoritative method.
The vapor reaction activities do include VDM association; the liquid reaction
activities do not use its matching reference.

In the numerically converged VDM equilibrium profile, the maximum log
liquid/vapor reaction-activity mismatch is 1.41755, a factor of 4.127. Using the
VLE liquid reference, reactive stages have `ln(Q/K)` from 1.38803 to 1.40097,
or `Q/K` approximately 4.01–4.06, despite the solver's 1.25e-10 residual.
This is a violation of a consistent phase/reaction fugacity reference, not a
Newton issue. The focused correction belongs in the shared reaction activity
interface, using the existing authoritative liquid reference. The VDM numbers
above must not be presented as physically valid reactive equilibria.

### Parameter validity

All six NRTL binary pairs were available and no missing-pair ideal fallback
warnings occurred. Their provenance is mixed: acid/ester and acid/water fits
declare VDM vapor association, while ethanol/ester and ester/water fits declare
ideal vapor. Reusing that entire set with HOC is a sensitivity study, not a
validated quaternary HOC parameter set. The ester/water pair's stated fit range
is 308.15–348.15 K, so extrapolation also needs assessment for hotter stages.

HOC molecular radii came from the existing deterministic force-field estimator,
with reported quality 0.55: ethanol 2.22359 A, acid 2.51905 A, ester 3.40245 A,
water 0.621467 A. The script records these supporting-property estimates.
The installed environment lacks tblite, and the default isolated fixture
contains no suitable GFN2-xTB geometry. No fabricated quantum properties or
runtime JSON edits were used.

The experimental/design literature also treats the nonidealities as central:
[Tang, Huang and Chien (2003)](https://scholars.lib.ntu.edu.tw/entities/publication/671dde55-2301-4260-9203-8b26f0802a25)
used a fitted NRTL model, acetic-acid vapor association and an overhead
decanter, followed by another column for high-purity ester. That process is
not equivalent to this deliberately simple homogeneous-stage probe.

## Production incorporation plan

Extend the existing rigorous column and shared stage model instead of creating
another column solver or nesting a complete reactor solve inside every stage
residual evaluation.

1. Resolve and validate existing named reaction definitions with the existing
   stoichiometric and kinetic parsers. Keep stage allocation, phase and holdup
   metadata separate from the chemical reaction definition. Validate independent
   equilibrium reaction rank; do not add redundant equilibrium equations.
2. Expand `_component_order` to include all participating reaction species,
   including products absent from feeds and species involved in coupled reaction
   networks. Include these species in initializers/coarse grids and use
   reaction-aware component scales. Do not require tiny artificial product feeds.
3. Add signed local extents and closures in the shared stage equations. Append
   local reaction derivative blocks to the existing Jacobian. Keep reaction
   metadata and source accounting shared with the VLLE path.
4. Validate formation enthalpy/entropy availability rather than inheriting
   nonreactive zero-reference defaults. Resolve the two physical issues above
   before claiming physically rigorous reactive duties or equilibria.
5. Change external species audits to compare outlet flow with feed flow plus
   `nu * total_extent`; retain elemental and energy audits. Allow changing total
   molar flow for reactions with nonzero sum(nu), including flow-specification
   validation and initializer assumptions. Esterification's zero sum(nu) does
   not test that general case.
6. For VLLE equilibrium, impose one independent reaction affinity per reaction
   per stage: phase fugacity equality makes duplicate liquid-phase equilibrium
   closures redundant. For kinetics, sum the rates over actual liquid-phase
   holdups or catalyst allocations. A liquid *flow* split is not automatically
   the liquid *inventory* split. Preserve the production phase-topology checks
   and condenser routing. No reactive VLLE solve was attempted here.
7. Prefer direct Newton with meaningful positive species seeds, retaining
   reaction-strength continuation as a recovery path. Keep condenser/reboiler
   reaction locations explicit and require volume/catalyst loading for kinetics.
   Do not silently inherit reaction support into CMO/shortcut/McCabe-Thiele
   classes whose assumptions have not been extended.
8. Connect `.pfd` named reactions/stage settings through parser validation,
   `flowsheet_solver.py` parameter construction, unit reaction capability
   metadata and `static/js/configuration-editors.js`/`editor.js`.
   The parser currently restricts named reaction references to reactor units;
   the web inspector exposes reactions only for `supports_reactions` units.
   Stage reaction results should include extent, rate/affinity, conversion,
   holdup/catalyst basis and conservation diagnostics. Reject unsupported
   phase/model combinations descriptively instead of accepting ignored input.

Meaningful production validation would cover zero-product feeds, signed reverse
reaction, coupled reactions and rank deficiency, nonzero sum(nu), multiple feeds,
reaction-adjusted elemental/species/energy balances, equilibrium/phase fugacity
consistency, kinetic units/holdup, derivative sparsity and VLLE phase topology.
Frontend/PFD roundtrips and nonreactive-column regression checks belong in the
eventual implementation. No full suite was run for this script-only investigation.

## Complete evaluated case table

`nrtl-bv` here means correlation=HOC with estimated radii. Iterations and solve
seconds sum across all continuation steps where applicable.

| Case | Signed forward conversion, % | Newton iterations | Solve seconds | Residual |
|---|---:|---:|---:|---:|
| ideal-equilibrium-direct | 85.7427 | 6 | 0.1895 | 6.09e-13 |
| ideal-equilibrium-hydrolysis | -14.7264 | 6 | 0.1788 | 8.08e-13 |
| ideal-equilibrium-seed-123 | 85.7427 | 8 | 0.2877 | 4.11e-11 |
| ideal-equilibrium-seed-17 | 85.7427 | 8 | 0.2494 | 2.73e-13 |
| ideal-equilibrium-seed-42 | 85.7427 | 6 | 0.1957 | 9.22e-9 |
| ideal-equilibrium-stages-10 | 84.4277 | 7 | 0.4523 | 9.09e-13 |
| ideal-equilibrium-stages-20 | 83.1463 | 7 | 0.8344 | 2.06e-12 |
| ideal-kinetic-direct | 76.2971 | 6 | 0.2340 | 2.72e-9 |
| ideal-kinetic-holdup-0.1 | 48.2565 | 5 | 0.2331 | 1.99e-9 |
| ideal-kinetic-holdup-10.0 | 84.5232 | 7 | 0.3193 | 2.80e-12 |
| nrtl-bv-equilibrium-augmented | 63.6313 | 7 | 0.7661 | 2.99e-11 |
| nrtl-bv-equilibrium-direct | 63.6313 | 7 | 0.7536 | 2.99e-11 |
| nrtl-bv-equilibrium-seed-17 | 63.6313 | 9 | 0.9137 | 3.11e-10 |
| nrtl-bv-equilibrium-seed-42 | 63.6313 | 7 | 0.8478 | 9.76e-10 |
| nrtl-bv-equilibrium-split-feeds | 71.0486 | 6 | 0.9828 | 9.04e-13 |
| nrtl-bv-equilibrium-stages-20 | 59.1131 | 7 | 2.2508 | 2.31e-9 |
| nrtl-bv-kinetic-augmented | 50.7177 | 8 | 0.6288 | 4.96e-10 |
| nrtl-bv-kinetic-direct | 50.7177 | 8 | 0.8636 | 4.96e-10 |
| nrtl-equilibrium-augmented | 75.8005 | 6 | 0.1139 | 1.02e-12 |
| nrtl-equilibrium-continuation | 75.8005 | 28 | 0.9835 | 6.91e-13 |
| nrtl-equilibrium-direct | 75.8005 | 6 | 0.2491 | 1.02e-12 |
| nrtl-equilibrium-distillate-40.0 | 76.1012 | 6 | 0.2216 | 8.32e-13 |
| nrtl-equilibrium-distillate-60.0 | 76.3424 | 6 | 0.2303 | 6.37e-11 |
| nrtl-equilibrium-seed-17 | 75.8005 | 9 | 0.3798 | 2.43e-13 |
| nrtl-equilibrium-seed-42 | 75.8005 | 6 | 0.2162 | 3.27e-9 |
| nrtl-equilibrium-split-feeds | 77.0546 | 8 | 0.4895 | 8.83e-10 |
| nrtl-equilibrium-stages-20 | 71.3188 | 7 | 0.9452 | 7.13e-13 |
| nrtl-kinetic-augmented | 63.8095 | 7 | 0.2429 | 2.43e-10 |
| nrtl-kinetic-continuation | 63.8095 | 30 | 1.3212 | 3.42e-9 |
| nrtl-kinetic-direct | 63.8095 | 7 | 0.3749 | 2.43e-10 |
| nrtl-vdm-equilibrium-direct | 73.6805 | 7 | 0.3440 | 1.25e-10 |
| nrtl-vdm-kinetic-direct | 61.7770 | 9 | 0.6006 | 3.58e-12 |

## Reproduction and artifacts

Run from the repository root; output directories must be new. No randomness
is used except the explicitly seeded robustness cases. Source hashes and
NumPy/SciPy versions are included in each output directory.

```bash
set -o pipefail
.venv/bin/python scripts/probe_reactive_distillation.py \
  --suite extended --workers 3 --output /tmp/reactive-distillation-new \
  2>&1 | tee /tmp/reactive-distillation-new.log > /dev/null

.venv/bin/python scripts/probe_reactive_distillation.py \
  --audit-results /tmp/reactive-distillation-new \
  --output /tmp/reactive-distillation-new-audit \
  2>&1 | tee /tmp/reactive-distillation-new-audit.log > /dev/null
```

The current extended suite contains 38 cases: the 32 evaluated cases above
plus six additional continuation cases. `--cases` selects named cases without
repeating a completed expensive calculation. `--audit-results` checks saved
profiles without rerunning columns. The default per-case timeout is 90 seconds;
a timed-out worker is terminated and recorded, never automatically replaced.

Results from this investigation:

* `/tmp/pfdsim-reactive-distillation-basic/summary.json`: initial six cases,
  including the four setup/report errors and two successful NRTL cases.
* `/tmp/pfdsim-reactive-distillation-extended/summary.json`: 14 successful cases.
* `/tmp/pfdsim-reactive-distillation-robustness/summary.json`: 14 successful cases.
* `/tmp/pfdsim-reactive-distillation-vdm/summary.json`: two successful VDM solves.
* `/tmp/pfdsim-reactive-distillation-thermo-audit/thermodynamic_audit.json`:
  caloric, binary-parameter provenance and LLE audits of the first 16 successes.
* `/tmp/pfdsim-reactive-distillation-robustness-audit/thermodynamic_audit.json`:
  LLE/provenance/caloric audits of the 14 robustness cases.
* `/tmp/pfdsim-reactive-distillation-vdm-audit/thermodynamic_audit.json`:
  VDM caloric and reaction/VLE fugacity-reference mismatch audit.

Each solve directory also contains per-case profiles/logs, source provenance
and a sibling persisted command log. Earlier audit files predate the later
addition of phase/reaction activity comparison fields; the VDM audit includes
those fields. Solver outputs remain intact.

## Design Decisions

* Extend the production MESH equations in a standalone probe rather than writing
  an independent column implementation or modifying production during an
  investigation. This directly tests the intended integration point.
* Use signed scaled extents, one closure per active stage/reaction, and
  reaction-aware product scales. Directly substituting kinetic rates would
  remove some unknowns, but extents also support equilibrium and provide
  uniform source accounting and diagnostics.
* Compare full colored differencing with an augmented production Jacobian.
  Retaining the production MESH block avoids copying derivative machinery;
  independent central differencing verifies the added blocks and sparsity.
* Keep kinetics exploratory and use explicitly recorded HOC supporting-property
  estimates. Importing unverified empirical constants or pretending the geometry
  estimator was quantum data would obscure what was actually tested.
* Isolate runtime caches, persist all important results, refuse to overwrite
  output directories, and run at most three single-threaded worker processes.
  This preserves personal state and gives reproducible, bounded probes.
