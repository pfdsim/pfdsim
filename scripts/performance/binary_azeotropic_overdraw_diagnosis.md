# Why the binary alcohol/water overdraw probes failed — 2026-10-07

The immediate native line-search failure is identifiable: a nearly pure-water
bottom tray produces an enormous Newton correction in a log-composition
coordinate. The solver's uniform largest-coordinate cap then suppresses the
entire Newton step, including useful temperature, flow, and duty corrections.
The resulting physical change is indistinguishable from zero, so backtracking
cannot reduce the merit.

The IPA/water +2% specification is demonstrably feasible in the original MESH
model. A better profile allows even the unchanged native line search to solve it.
For ethanol/water +2%, the same stalled-step mechanism is established, but this
investigation has not recovered a converged solution or proved infeasibility.

Only [diagnose_binary_azeotropic_overdraw.py](diagnose_binary_azeotropic_overdraw.py)
and this report were added. The earlier experiment report was annotated with
these follow-up findings. Production equations and solvers remain unchanged.

## Scope and equations

The two original binary cases are retained: 100 kmol/h liquid feeds at 298.15 K,
30 stages, reflux ratio eight, total condenser, one-bar pressure, no pressure drop.
Feed stages are 21 for ethanol/water and 20 for IPA/water. Feed alcohol fractions
are 0.10 and 0.20. The target cut is 1.02 times each model's first-azeotrope
inventory capacity. Thus D/F is 0.114282203 for ethanol and 0.299578890 for IPA.

The diagnostic captures the actual production model and initial vector, rather
than implementing a second set of column equations. It examines a model
Jacobian, saved failed iterates, and native residual labels. Additional solves
use the same residual equations, with different globalization, coordinates,
initial profiles, or equivalent equation scaling. Most exploratory runs target
1e-8, stricter than the primary experiment's 1e-6; their budgets are declared in
each run manifest.

For a binary composition coordinate s,

    s = log(x / (1-x)),   x = logistic(s),   dx/ds = x(1-x).

Here the independent fraction x is alcohol. The chain rule gives

    dR/ds = (dR/dx) x(1-x).

As x approaches zero, that Jacobian column becomes numerically weak. A physical
composition correction that would be modest in x can correspond to an enormous
linearized correction in s. That large logit correction is also far outside the
region where its local linearization is meaningful.

The native solver solves J delta = -R, then applies

    delta_capped = delta * min(1, 8 / max(abs(delta))).

This factor multiplies every coordinate. One weak composition column can therefore
make all other corrections negligible. This behavior is shared column-solver
machinery; it is not an alcohol-specific special case.

## Quantitative evidence at the stalled native iterates

These are the azeotropically initialized native line-search failures at +2%
overdraw. Both largest Newton corrections are coordinate 59: the bottom tray's
alcohol logit in these 30-stage binary vectors.

| Quantity | Ethanol/water | IPA/water |
|---|---:|---:|
| Residual infinity norm | 0.151220 | 3.344497 |
| Bottom alcohol fraction | 2.493e-24 | 2.094e-20 |
| Jacobian condition estimate | 1.290e16 | 1.193e16 |
| Largest Newton coordinate correction | 3.323e20 | 1.317e17 |
| Uniform step-cap multiplier | 2.407e-20 | 6.076e-17 |
| Largest capped temperature-coordinate correction | 1.053e-20 | 1.767e-17 |
| Largest capped flow-log correction | 2.815e-21 | 2.411e-17 |
| Linear solve error, infinity norm | 1.422e-15 | 2.665e-15 |

The ethanol merit before and after the capped full trial is identical:
0.019662814243909876. IPA changes from 7.717106343429838 to
7.717106343429840, a tiny floating-point increase. These are actual native
residual evaluations at the proposed next step, not estimates from the quadratic
model. Further halving a physically negligible step does not repair it.

The largest failed ethanol residual is a tray-28 alcohol balance; the largest
IPA residual is a tray-29 alcohol balance. Three seeded directional central
difference checks agree with the model Jacobian to relative errors below 8.15e-7
and 8.02e-7, respectively. Those checks do not establish perfect accuracy of
every tiny individual column, but there is no evidence of a large general
Jacobian error. The linear systems themselves are solved to approximately
roundoff residual despite their conditioning. Returning a finite Newton vector
does not mean that vector gives a usable nonlinear step.

This diagnoses the immediate line-search stall. It does not by itself explain
every preceding step, every alternative solver failure, or the complete solution
structure of the ethanol case.

## IPA/water +2%: recovered and validated

The first useful approximate profile came from a logarithmically interpolated
azeotropic profile and exact dense TRF in the native coordinates. After 150
Jacobian evaluations it reached residual 4.878e-5. This was not accepted as a
converged solution; it supplied a better stage profile for recovery experiments.

A bounded solve using direct mole-fraction coordinates recovered a native-vector
residual of 6.340e-9. Its vector was subsequently passed through the normal unit
outlet construction and validation. This establishes that the original +2%
IPA failure was numerical rather than an impossible target cut.

A controlled comparison then used the same saved approximate profile for all
three algorithms, applying a 1e-8 composition floor only to the starting profile.
The final solutions are allowed to have smaller fractions: no composition floor
was added to the production equations. This also avoids an implicit startup
perturbation by SciPy's bounded solver at an almost-zero initial fraction.

| Method | Final native residual | Jacobians | Solver seconds |
|---|---:|---:|---:|
| Unchanged native line search | 5.151e-9 | 119 | 5.98 |
| Exact dense TRF, native logits | 2.969e-9 | 96 | 2.29 |
| Exact dense TRF, bounded mole fractions | 9.148e-9 | 142 | 3.77 |

Each targets 1e-8 and has a 200-Jacobian budget. The 119-Jacobian line solve uses
more work than the primary experiment's 100 limit. Dense native TRF reaches the
stricter target within 100. These times exclude construction of the saved profile
and subsequent unit/phase validation; they are not end-to-end speed claims.
The first original native IPA attempt failed after only five Jacobians, so its
immediate failure was not merely exhaustion of its 100-Jacobian budget.

These dense diagnostics use SciPy TRF with `tr_solver='exact'`. That route uses
SVD-based subproblem solves, whereas the earlier sparse TRF experiments used
LSMR and a two-dimensional subspace. See the [official SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).
They should not be treated as the same TRF configuration.

The unchanged line-search recovery produces:

- Distillate flow: 29.957889012 kmol/h.
- Distillate IPA fraction: 0.667603782, compared with model azeotrope 0.680955857.
- Bottoms flow: 70.042110988 kmol/h.
- Bottoms IPA fraction: 2.094e-11.
- External component imbalance: 1.466e-11 of feed flow.
- Native MESH residual: 5.151e-9.

All 30 liquid-stage local stability checks pass. The existing LLE solver finds
no split at the top, feed stage, or bottom. All three controlled recovered
solutions pass ordinary unit validation; their very small bottoms fractions
differ because the accepted residual tolerance does not tightly determine
fractions at that trace scale.

The previous failed IPA iterate had depleted the bottom alcohol fraction about
nine orders of magnitude below the recovered line-search solution, while leaving
large internal tray imbalances. It was an inappropriate nearly pure-water
state, not simply an accurate near-pure product awaiting a final small correction.

The controlled result also limits the attribution: changing coordinates is not
strictly necessary. A better initialized profile lets the existing logarithmic
solver converge. Initialization quality and handling of numerically inactive
composition directions both matter.

## Other diagnostics and unresolved ethanol result

The investigation records 50 nonlinear runs, plus saved-vector audits and the
internal coarse-initializer solve. They are exploratory diagnostics, not a tuned
solver comparison.

- Eighteen runs compared native line search, exact dense TRF, and bounded-logit
  dense TRF on the native, logarithmically interpolated, and feed-anchored
  logarithmic profiles, for both binaries at +2%. Each had a 150-Jacobian budget.
  No run reached 1e-8. Dense TRF substantially reduced residuals, without proving
  convergence. Best ethanol residual was 2.839e-4; best IPA residual was 4.878e-5.
- Four direct mole-fraction runs used native or saved profiles with a 500-Jacobian
  budget. IPA with the saved profile converged to 6.340e-9 in 141 Jacobians. Its
  native profile still failed at 1.644e-3. The best ethanol result was 2.280e-4.
- Ten attempted reflux-continuation entries covered reflux ratios 1, 2, 4, 6, and
  8. Neither case obtained a converged first root at reflux one, so no successful
  continuation chain was established. Higher entries remained cold starts rather
  than evidence of a tracked branch. They did not solve the original cases.
- Twelve row-preconditioned runs subtracted constant pure-component enthalpy
  references from the energy balances using the component-balance rows, and
  rescaled sensible/latent energy. This is an invertible row transformation and
  does not change the equations' zero set. It did not solve ethanol; best native
  ethanol residual was 1.991e-4. The IPA saved profile was already converged, so
  its rapid replay is not evidence that this scaling alone rescued the case.
- An independent physical-coordinate, numerical finite-difference TRF probe
  continued ethanol for up to 200 outer steps and reached 1.817e-4. That is still
  a failure, even though it removes dependence on the supplied model Jacobian.
  Early raw records use the callback step count as the FD Jacobian counter; the
  maintained script now records SciPy's actual `njev` for this mode. No early
  records were overwritten.
- The existing coarse rigorous initializer supplied another ethanol profile.
  Native line search failed at 0.314608; a bounded mole-fraction solve reached
  4.894e-4 after 300 Jacobians. It also did not establish a root.
- The three controlled IPA recovery runs above all converged.

Consequently, I can explain the stalled native ethanol step and diagnose serious
conditioning/initialization trouble. I cannot yet say the ethanol +2% target is
feasible for these exact settings, nor that it is infeasible. The remaining
nonzero least-squares residuals are not substitutes for a convergence or
feasibility proof.

## Implications

The alcohol/water results should not be interpreted as an inherent inability to
draw additional water with an azeotrope-rich overhead. The IPA solution explicitly
does that: its overhead alcohol fraction decreases below the azeotrope and its
bottoms retain a positive trace of alcohol.

Near-pure trays are a difficult numerical regime for logarithmic composition
coordinates. The uniform Newton cap makes this particularly damaging once a
weak coordinate dominates an otherwise solvable linear system. Trust regions
can avoid the full unstable Newton direction, but they do not automatically
recover a useful profile from every collapsed state. Increasing backtracking
attempts or simply increasing the global step limit would not address the
underlying coordinate-conditioning problem safely.

The production follow-up worth evaluating would handle near-boundary composition
directions and rank/scale information at the shared solver level, with a profile
restart when the nonlinear step becomes physically inactive. This investigation
does not implement a production policy, an alcohol-specific workaround, or an
unconditional mole-fraction floor.

## Artifacts and reproduction

All outputs are persisted under `/tmp`, including manifests, harness snapshots,
full logs, native vectors, singular values, residual labels, and validation data:

- `pfdsim-binary-overdraw-diagnostic-20261007`: the original 18 diagnostic solves.
- `pfdsim-binary-overdraw-physical-20261007`: four physical-coordinate runs.
- `pfdsim-binary-overdraw-ramp-20261007`: ten reflux attempts.
- `pfdsim-binary-overdraw-preconditioned-20261007`: twelve equivalent-row-scaling runs.
- `pfdsim-binary-overdraw-physical-fd-20261007`: independent numerical-FD ethanol run.
- `pfdsim-binary-overdraw-coarse-20261007`: two coarse-profile ethanol runs.
- `pfdsim-binary-overdraw-cap-audit-20261007`: reevaluation of all 18 original vectors,
  with explicit Newton-step cap diagnostics.
- `pfdsim-binary-overdraw-ipa-controlled-20261007`: three controlled IPA recoveries.
- `pfdsim-binary-overdraw-root-audit-20261007`: independent replay of the three
  controlled roots through the original unit and phase checks.

Examples from the repository root, with new directories:

```bash
set -o pipefail
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0 \
.venv/bin/python scripts/performance/diagnose_binary_azeotropic_overdraw.py \
  --output /tmp/binary-overdraw-new --factors 1.02 --budget 150 \
  2>&1 | tee /tmp/binary-overdraw-new.log

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0 \
.venv/bin/python scripts/performance/diagnose_binary_azeotropic_overdraw.py \
  --output /tmp/binary-ipa-recovery-new --cases ipa_water --factors 1.02 \
  --seeds saved --methods line dense physical --seed-composition-floor 1e-8 \
  --restart-data /tmp/binary-overdraw-new/results.jsonl --budget 200 \
  2>&1 | tee /tmp/binary-ipa-recovery-new.log

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0 \
.venv/bin/python scripts/performance/diagnose_binary_azeotropic_overdraw.py \
  --output /tmp/binary-cap-audit-new --factors 1.02 \
  --audit-data /tmp/binary-overdraw-new/results.jsonl \
  2>&1 | tee /tmp/binary-cap-audit-new.log
```

Ruff passes. The final audits verify the native residuals and normal unit checks;
source hashes in the final manifest verify unchanged production solver/thermo
and reused experimental scripts. Audits do not rerun the nonlinear solves.

## Design Decisions

- Capture the authoritative MESH model and saved vectors, rather than recreate
  column equations. This separates solver diagnostics from changes to physics.
- Inspect actual capped trial merits and linear solve errors, rather than infer
  the failure mechanism only from a large condition estimate.
- Use exact dense TRF only for these small 122-variable diagnostics. It supplies
  rank-aware comparison without proposing dense linear algebra for large columns.
- Compare bounded mole fractions with native logits, and use the same saved
  profile with a declared seed-only floor for the controlled comparison. This
  distinguishes coordinate effects from initial-profile and budget effects.
- Validate recovered vectors through ordinary unit outlet construction, material
  balances, and phase checks. A small optimizer cost alone is insufficient.
- Retain failed ethanol probes and state the unresolved feasibility explicitly,
  instead of treating improved residuals as successful solutions.
