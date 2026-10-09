# Runtime column recovery experiments — 2026-10-08/09

Both original alcohol/water +2% cases are now solved from cold starts, with the
original MESH equations and normal unit validation. The ethanol/water case is
no longer an unresolved feasibility question at that cut. A deterministic
fallback sequence recovers both; it is not yet a generally reliable solver,
because nearby cuts still fail. No production files were edited for this work.

The maintained experiment is
[probe_column_coordinate_recovery.py](probe_column_coordinate_recovery.py).
The numerical/injection checks are
[check_column_coordinate_recovery.py](check_column_coordinate_recovery.py).
The runtime patch is `RuntimePatch(budget, attempts, mode="portfolio")`; it
restores the original methods when its context exits. Experiments use fresh,
sequential worker processes and one BLAS thread. This global monkeypatch is an
experimental process-scoped tool, not a thread-safe application integration.

## Original settings and success criterion

The original fixtures are unchanged: 100 kmol/h liquid feed at 298.15 K,
30 stages, reflux ratio eight, one bar on every stage, no pressure drop, total
condenser. Ethanol/water uses UNIFAC with ethanol feed fraction 0.10 and feed
stage 21. IPA/water uses NRTL with IPA feed fraction 0.20 and feed stage 20.
Cut factors multiply the model azeotrope inventory capacity, not total feed:

| Case | Azeotropic inventory capacity D/F | +2% target D/F |
|---|---:|---:|
| Ethanol/water | 0.112041375871 | 0.114282203388 |
| IPA/water | 0.293704794235 | 0.299578890119 |

All reported successes require the **original** residual infinity norm below
1e-6 and successful normal outlet construction/unit validation. The same strict
threshold is used for relaxed acceptance in the experiment. Neither a reduced
residual, optimizer termination flag, nor improved cost counts as success.

## What worked from cold starts

The `portfolio` runtime patch performs the following deterministic sequence:

1. Run the actual native solver from the requested initializer. Return immediately
   if it meets the requested residual.
2. On a supported binary total-condenser model, generate a fresh cheap profile
   using the authoritative initializer. Try up to 100 fixed-K material-balance
   sweeps, with stable bottom-inventory coordinates, projected equilibrium
   temperatures/energy flows, and Anderson acceleration.
3. If necessary, generate a fresh azeotropic endpoint profile and interpolate
   its compositions logarithmically. Run exact dense TRF in the native logits
   for up to 150 Jacobians, retaining the original residual and model Jacobian.
4. Refine that approximate profile with the actual native solver. Apply a 1e-8
   mole-fraction floor only to this **initial** refinement profile. No floor is
   imposed on the final compositions or on the original equations.

No saved solution vectors or hand-selected successful profiles are inputs to this
sequence. The experiment reuses `dense_solve` and `profile_seed` from the earlier
diagnostic script instead of introducing a second implementation of those methods.
The dense diagnostic uses its declared 300–420 K temperature interval, flow
interval 1e-8–1e5 kmol/h, and composition-logit interval [-32,32]. Those limits
cover the tested alcohol/water conditions; they are not universal physical bounds.

| Cold-start case | Native first attempt | Recovery | Final original residual | Total solve seconds |
|---|---|---|---:|---:|
| Ethanol/water +2% | fails, residual 0.149866, 25 Jacobians | balance sweep, 74 iterations | **8.378e-7** | **6.679** |
| IPA/water +2% | fails, residual 0.110055, 78 Jacobians | logarithmic-profile dense solve, then native refinement | **1.368e-8** | **31.961** |

For ethanol, the sweep uses 148 tridiagonal component solves (two per iteration)
and 364 complete original-residual evaluations. Its linear component inventory
errors stay below 1.85e-13 kmol/h. It uses no Jacobian, so the total 25 Jacobians
reported for that recovered unit are solely its initial failed native attempt.

For IPA, the sweep does not converge: after 100 iterations its best residual is
0.074404. The dense stage then reaches 5.047e-5 in 150 Jacobians. That is only
an approximate profile and is **not accepted**. Native refinement converges to
1.368e-8 in 122 more Jacobians. The total is 350 Jacobians plus the sweep work.

Times include initialization, all recovery stages, ordinary unit completion, and
supplemental result checks. They exclude feed/model setup and azeotrope discovery.
These single-pass timings are not a controlled speedup benchmark. The portfolio
uses a 200-per-attempt native budget, larger than the earlier primary 100 budget;
it also spends work in additional algorithms. It demonstrates recovery rather
than an equal-work reliability comparison.

## Validated products

| Quantity | Ethanol/water +2% | IPA/water +2% |
|---|---:|---:|
| Distillate flow, kmol/h | 11.428220339 | 29.957889012 approximately |
| Distillate alcohol mole fraction | 0.875026881 | 0.66760378 approximately |
| Bottoms flow, kmol/h | 88.571779661 | 70.042110988 approximately |
| Bottoms alcohol mole fraction | 3.890e-11 | 5.560e-11 |
| Max external component imbalance / feed flow | 1.421e-16 | 3.894e-11 |
| Original MESH residual | 8.378e-7 | 1.368e-8 |

Both overhead compositions lie below their model azeotrope alcohol fractions,
0.892527419 and 0.680955857. They draw additional water rather than crossing a
prescribed azeotropic purity. Both pass all 30 liquid-stage spinodal checks, and
the LLE solver finds no split at the top, feed stage, or bottom. This sampled LLE
audit is not a proof of global stability on every stage. No physical-law violation
was established by these successful solutions.

## What the structured sweep changes

The failed native Newton iterates can drive a bottom composition toward zero,
making a logit Jacobian column weak. A uniform largest-coordinate step cap then
shrinks useful changes everywhere. Merely removing that particular cap is not
enough: the corrected coordinatewise-cap test still fails all four +2% cold
starts (two binaries, two initializers).

The successful ethanol sweep makes several related changes without changing the
equations' zero set:

- **Stable small inventories.** Parameterize component amounts remaining in the
  bottoms directly, rather than subtract an almost-complete distillate recovery
  from feed inventory. Logarithms retain relative precision of small positive
  amounts. A bounded allocation chart conserves each external component.
- **Exact total balances and specifications.** Distillate draw, reflux, bottoms
  flow, and total tray balances determine dependent phase flows. For a molar cut
  D and cumulative feed C_j, enforce L_j - V_(j+1) = C_j - D. Positive excess flow
  coordinates parameterize the remaining freedom in the less-reduced variants.
- **Temperature projection.** Compute each bubble temperature with the original
  stage-property residual and original temperature bounds. This is an elimination
  of an equation, not use of a different VLE model.
- **Reference-independent energy equations.** Subtract a constant pure-component
  enthalpy reference times the component balances from each energy balance:
  H_tilde_j = H_j - sum_i(h_ref_i M_ij). This invertible triangular row transform
  removes large formation-reference contributions. The reduced formulation solves
  the transformed interior energy balances for phase flows and the two end
  duties. At a component-balance root, original and transformed energy equations
  are equivalent. Original energy residuals remain part of final acceptance.
- **Global linear material solves.** Freeze the current equilibrium ratios and
  phase flows, and solve a tridiagonal material system for each component. Use
  normalized compositions and stable bottom inventories from those solutions to
  update the profile. The tridiagonal equations are the original component
  balances with fixed coefficients, not a separate physical column model.
- **Acceleration with safeguards.** Use up to six recent map updates for Anderson
  mixing, bounded by the composition region and physical-flow domain. Compare
  full original-residual merit over a short nonmonotone history, with damped
  alternatives if an accelerated proposal fails. This helps one case but is not
  by itself a globally convergent method.

For homogeneous constant-pressure binaries only, the optional region projection
uses the feed's interval between the model's azeotropes. Candidates with detected
LLE are excluded, and a 1e-7 fraction margin accommodates numerical boundary
error. This is deliberately not applied to arbitrary ternary/VLLE regions.

The small-inventory chart is numerically significant. Replaying an earlier IPA
root through a chart that computes bottoms by subtracting two nearly equal
amounts changes the bottom log coordinate by about 7.54, despite tiny absolute
material errors. The direct-small-inventory chart preserves its trace bottom
fraction to relative tolerance 1e-8 in the maintained checks.

## Nearby cuts: the limitation is substantial

The same fixed fallback sequence, same parameters, and cheap cold initializer
give these results. No method selection was changed between cuts.

| System | Cut relative to azeotropic inventory | Outcome | Final original residual | Seconds |
|---|---:|---|---:|---:|
| Ethanol/water | 0.98 | converges in native first attempt | 1.398e-11 | 0.255 |
| Ethanol/water | 1.005 | converges in native first attempt | 4.238e-8 | 0.312 |
| Ethanol/water | 1.02 | recovered by sweep | 8.378e-7 | 6.679 |
| Ethanol/water | 1.05 | **fails** | 1.293e-3 | 38.111 |
| IPA/water | 0.98 | converges in native first attempt | 9.582e-7 | 0.167 |
| IPA/water | 1.005 | **fails** | 3.483e-4 | 43.515 |
| IPA/water | 1.02 | recovered by dense profile + native refinement | 1.368e-8 | 31.961 |
| IPA/water | 1.05 | **fails** | 1.826e-3 | 47.306 |

That is five successes out of eight physical cut cases, compared with three
successful initial native attempts in the same collection. The latter is not a
general solver failure rate. It is a small deliberately difficult sample.
The failed cuts' residuals do not establish feasibility or infeasibility.
The nonmonotone success across cuts emphasizes sensitivity of the convergence
basins; it does not mean a smaller draw is inherently physically harder.

## Ordinary-column checks

The portfolio also completes these five existing fixtures at the strict 1e-6
tolerance. No recovery stage is needed. Unsupported layouts retain the native
solver, including event propagation, rather than receiving a partially compatible
projection.

| Existing fixture | Residual | Total Jacobians | Seconds | Path |
|---|---:|---:|---:|---|
| Benzene/toluene NRTL, 20 stages | 7.967e-8 | 6 | 0.092 | native first attempt |
| Hydrocarbon PR, 42 stages | 9.353e-7 | 12 | 1.328 | native delegate, including initializer |
| Partial-condenser methanol/water mass cut, 12 stages | 2.111e-11 | 7 | 0.077 | native delegate |
| UNIQUAC acetaldehyde absorber, 5 stages | 2.754e-8 | 11 | 0.207 | native delegate |
| UNIQUAC acetaldehyde stripper, 5 stages | 1.698e-7 | 4 | 0.154 | native delegate |

All complete normal unit validation and independent external component-balance
checks. Supplemental liquid stability/phase checks are reserved for the targeted
alcohol/water study; aqueous Henry gas systems require their own applicable phase
validation. Native delegates passing does not establish effectiveness of the new
recovery algorithms on those layouts.

## Other experiments: failures retained

Each four-start entry below is ethanol/water and IPA/water +2%, using cheap and
azeotropic starts. Budgets differ between methods, so this is an outcome inventory,
not a fair speed ranking.

| Variant | Per-attempt budget | Successes / starts |
|---|---:|---:|
| Scaled nonnegative liquid-component-flow and vapor-flow coordinates | 250 Jacobians | 0/4 |
| Exact total-material/specification projection | 200 Jacobians | 0/4 |
| Additional exact external component-inventory projection | 250 Jacobians | 0/4 |
| Additional bubble-temperature/end-duty elimination | 150 Jacobians | 0/4 |
| Coordinatewise clipped Newton, descent fallback | 200 Jacobians | 0/4 |
| Energy-flow elimination plus binary region bounds | 120 Jacobians | 0/4 |
| Direct small-bottom-inventory log coordinates | 100 Jacobians | 0/4 |
| Anderson-accelerated fixed-K material sweep | 180 sweeps | **1/4**, ethanol cheap |
| Constant-half-damped fixed-K sweeps | 300 sweeps | 0/2 cheap starts |
| Exact box-constrained Gauss–Newton trust subproblem | 180 Jacobians | 0/4 |
| TRF root of relative fixed-K update map | 90 outer iterations | 0/4 |
| Powell hybrid root of relative fixed-K update map | 25*(dimension+1) map calls | 0/4 |

The box-constrained subproblem tests a real alternative to uniformly shrinking
or clipping an unconstrained Newton direction. It minimizes the linearized
least-squares merit within per-coordinate bounds and uses actual/predicted
reduction to adapt the radius. It still does not cure the cold-start traps in
these four tests. Better geometry alone does not select the right profile.

Stage continuation was also investigated. The total-balance projected method
solved ethanol columns through 22 stages but failed at 24; IPA solved four and
six stages but failed at eight. The stable material sweep advanced IPA from four
through nine stages but failed at ten. These are incomplete chains, not solutions
of the 30-stage cases. An early stage pilot stopped on a script API-name error;
that run was diagnosed and a separate corrected experiment was used.

An early coordinate-cap pilot had an experimental dispatch bug that overwrote its
first solver result. It is excluded. The corrected run is the 0/4 entry above,
and a maintained synthetic check now verifies dispatch and method restoration.
An early inventory-chart pilot had inconsistent vector/bounds shapes and is also
excluded. Corrected projection invariants were verified before further solves.
All these logs are retained; no lost/failed process was silently restarted.

## Assessment

**Fixable examples: yes. General reliable algorithm: not established.** The two
original +2% equations now have validated solutions from an automatic cold-start
pipeline. The relevant strategy is to change the formulation and restart the
profile when local Newton steps become ineffective, rather than merely add more
backtracking or a larger global step cap.

The evidence supports keeping the native path for easy cases and investigating
structured recovery as a fallback. It does not support installing this prototype
as a universal replacement. It is currently limited to diagnostic-size,
two-component, total-condenser VLE recovery with no side draws, no stage-efficiency
extra variables, and no active-set events. Dense recovery, the declared diagnostic
bounds, binary region information, selection of a trace species, and finite-difference
accuracy near nearly pure stages all require broader validation before production
integration. A generally robust multicomponent implementation would need suitable
charts for more than one trace component and preservation of VLLE topology logic.

## Validation and artifacts

Maintained checks verify:

- runtime coordinate-cap dispatch and useful-update retention;
- restoration of the patched production methods on context exit;
- both recorded +2% roots against the original residual;
- stable trace-bottom inventory round trips;
- projected interior reference-adjusted energy equations;
- component conservation of the fixed-K linear solves;
- ten seeded random exact-material/specification projections per binary;
- ordinary unit completion, all-stage local stability, and top/feed/bottom LLE
  checks by independently replaying saved vectors, without nonlinear reruns.

Ruff passes for both added scripts. Final source-hash checks compare unchanged
column solver, thermodynamic activity model, and reused case/diagnostic scripts
with run manifests. Other working-tree edits belong to concurrent work and were
preserved.

Authoritative cold recovery results:

- `/tmp/pfdsim-coordinate-portfolio-20261009`: the two +2% cold solves.
- `/tmp/pfdsim-coordinate-portfolio-extended-20261009`: six nearby binary cuts and
  five ordinary-column checks.
- `/tmp/pfdsim-coordinate-portfolio-audit-20261009`: independent saved-root audits.
- `/tmp/pfdsim-coordinate-checks-20261009.log`: maintained invariant checks.

Every run records JSONL results, complete per-worker logs, a source snapshot, and
a manifest. Experimental failures remain in the similarly named coordinate run
directories. `--summarize-only` creates an exclusive new summary from existing
records without running any solvers.

Run from the repository root with new output directories:

```bash
set -o pipefail
.venv/bin/python scripts/performance/probe_column_coordinate_recovery.py \
  --output /tmp/coordinate-recovery-new --mode portfolio --budget 200 \
  --cases ethanol_water ipa_water --factors 1.02 --initializers cheap_estimate \
  2>&1 | tee /tmp/coordinate-recovery-new.log

.venv/bin/python scripts/performance/probe_column_coordinate_recovery.py \
  --output /tmp/coordinate-recovery-nearby-new --mode portfolio --budget 200 \
  --cases ethanol_water ipa_water benzene_toluene_20 hydrocarbon_pr_42 \
          partial_mass_12 absorption_5 stripping_5 \
  --factors .98 1.005 1.05 --initializers cheap_estimate \
  2>&1 | tee /tmp/coordinate-recovery-nearby-new.log

.venv/bin/python scripts/performance/probe_column_coordinate_recovery.py \
  --output /tmp/coordinate-recovery-audit-new --cases ethanol_water ipa_water \
  --audit-directory /tmp/coordinate-recovery-new \
  2>&1 | tee /tmp/coordinate-recovery-audit-new.log

OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 \
.venv/bin/python scripts/performance/check_column_coordinate_recovery.py \
  --records /tmp/coordinate-recovery-new \
  2>&1 | tee /tmp/coordinate-recovery-checks-new.log
```

## Design Decisions

- Use runtime patches in a standalone worker process. This honors the request
  to leave production unchanged and makes every algorithm change removable.
- Reuse authoritative residuals, property calculations, initializers, and the
  earlier dense diagnostic helpers. Only solver coordinates and execution policy
  change; successful outputs must pass the original unit checks.
- Eliminate dependent balances/specifications where possible and represent small
  inventories directly. This addresses conditioning and cancellation rather than
  hardcoding an alcohol-specific convergence exception.
- Keep binary-region projection scoped to homogeneous constant-pressure binaries.
  Arbitrary ternary/VLLE boundaries need a different validated treatment.
- Use a deterministic fallback sequence rather than silently select successful
  saved starts. It reproduces both +2% recoveries automatically and exposes its
  failures on nearby cuts.
- Retain native delegation for unsupported layouts and active-set events. Passing
  those checks preserves behavior; it is not claimed as new recovery capability.
- Report cumulative work and per-stage failures, not just the final successful
  refinement time. The added robustness has a substantial cost on difficult cases.
