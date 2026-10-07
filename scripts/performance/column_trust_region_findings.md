# Column trust-region experiments — 2026-10-06

Unscaled dogleg and a Newton/gradient subspace trust region improved convergence
from deliberately poor initial guesses. Neither gave a general speed improvement
with the project's normal initializers. The evidence favors retaining line search
as the default and considering a trust-region restart as a rescue strategy.

Only experimental scripts were added. The production solver sources and reused
case factory were verified against their hashes recorded before measurement.

## Implementations and measurement

The maintained harness is [benchmark_column_trust_region.py](benchmark_column_trust_region.py).
The read-only analysis is [analyze_column_trust_region.py](analyze_column_trust_region.py).
Case construction comes from the existing
[sparse-column benchmark](benchmark_sparse_column_solves.py), preserving its
examples, test fixtures, feed conventions, and offline property setup.

| Label | Implementation |
|---|---|
| line | Actual native sparse Newton solver and backtracking |
| dogleg | Sparse direct Newton direction plus Cauchy/Newton dogleg |
| dogleg_scaled | Same dogleg with monotone inverse Jacobian-column-norm scaling |
| subspace | Minimize the quadratic model in the span of the gradient and sparse direct Newton direction |
| trf | SciPy TRF, sparse LSMR, Jacobian column scaling |
| trf_unscaled | Same SciPy solver with `x_scale=1` |

The custom methods minimize the same merit, `0.5 * f.T @ f`. Their initial
Euclidean radius is `sqrt(number_of_variables)`, with a maximum of `8 * sqrt(n)`.
The scaled dogleg applies that radius in scaled coordinates. A trial is accepted
when actual/predicted reduction exceeds 0.1; the radius contracts below 0.25 and
expands above 0.75 when the step reaches the boundary. Rejected trials reuse the
Jacobian, with the native line-search trial limit serving as the rejection limit.
The native solver instead caps the largest coordinate step at eight before
backtracking. These are different globalization geometries, not identical step
constraints.

Sparse Newton directions use the native sequence of diagonal shifts, from zero
through `1e-2`. A regularized direction whose full quadratic-model reduction is
nonpositive is discarded as a dogleg endpoint; Cauchy remains available. This
guard matters because a shifted Newton direction need not minimize the original
least-squares model. Final results use the guarded version, rather than the more
optimistic recoveries observed during the initial pilot.

SciPy uses LSMR `atol=btol=1e-8`, `xtol=1e-12`, disabled `ftol`/`gtol`, and the
native Jacobian budget. Its optimizer status is not a convergence verdict:
the requested infinity-norm residual must actually be reached. Trial-point
`ThermodynamicsError` exceptions are rejected by contracting the trust radius;
initial-state and Jacobian failures still propagate. VLLE active-set/projection
events retain their existing untruncated Newton prediction and progress reporting.
See the [SciPy least-squares documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
for the library's TRF, LSMR, and scaling definitions.

Two experiment layers were used:

- **End-to-end:** fresh sequential processes retain the complete unit solver,
  initializer selection, coarse initialization, Jacobian retries, VLLE topology
  changes, outlet construction, and post-solve checks. Setup/import time is
  excluded; column initialization inside `unit.solve` is included. BLAS uses one
  thread. The primary five methods have three repetitions, with the first
  excluded as warm-up. Unscaled TRF uses its first two complete repetitions:
  compilation was already warm, and further repetitions were stopped after two
  PR-column worker timeouts. Its unused third-repetition partial output is retained.
- **Raw convergence:** one first nonlinear problem is captured per case and
  shared by every method. All methods get the exact same initial vectors;
  fingerprints are checked by the analyzer. There is no initializer or Jacobian
  retry rescue. This also avoids repeatedly constructing expensive models.

Raw counts measure equation convergence, not validated outlets or phase stability.
The end-to-end counts retain the unit's ordinary validation.

The final raw run fixes Python's hash seed to zero and seeds Python/NumPy with
20261006. Noise seeds are 11, 29, and 47. Independent Gaussian noise with standard
deviation 0.5 or 1.5 is added to the native transformed variables: temperature
logits, composition log ratios, log flows, and scaled duties where present.
For log flows, a one-standard-deviation displacement corresponds to factors
`exp(0.5)=1.65` or `exp(1.5)=4.48`. These are controlled numerical stress tests,
not an estimate of the distribution of real plant initial guesses.

Separate follow-up processes did not give bit-identical extractor initial-vector
fingerprints. Consequently, the authoritative stress results below were collected
again with all six methods sharing each captured problem, rather than combining
unmatched starts. Initial vectors are retained as readable numeric lists.

Environment: Python 3.14.4, NumPy 2.5.2, SciPy 1.18.1; production revision
`e36a5166a25941ab4c3536573c13eb2cfeb7b389`.

## Reliability from perturbed starts

Six numerical problems, two noise levels, and three seeds give 36 starts per
method and 216 final solves. The partial-mass cases share a physical fixture but
use different Jacobian implementations; these are not 36 independent flowsheets.

“Strict” means residual infinity norm below the unit's requested tolerance.
“Acceptable” means the native relaxed acceptance threshold. Requested tolerances
are `1e-6` for partial-mass distillation, `1e-7` for CMO, `2e-6` for absorber/
stripper, and `1e-5` for the extractor. Those criteria are unchanged across methods.

| Method | Strict | Acceptable | Noise 0.5 strict | Noise 1.5 strict | Rescues versus line | Lost line successes | Median time / line on common strict successes |
|---|---:|---:|---:|---:|---:|---:|---:|
| line | 17/36 | 17/36 | 13/18 | 4/18 | 0 | 0 | 1.000 |
| dogleg | 23/36 | 23/36 | 16/18 | 7/18 | 6 | 0 | 1.064 |
| dogleg_scaled | 15/36 | 15/36 | 13/18 | 2/18 | 1 | 3 | 1.334 |
| subspace | 23/36 | 23/36 | 16/18 | 7/18 | 6 | 0 | 1.139 |
| trf | 6/36 | 7/36 | 6/18 | 0/18 | 0 | 11 | 5.411 |
| trf_unscaled | 6/36 | 6/36 | 3/18 | 3/18 | 1 | 12 | 0.516 |

Dogleg and subspace improve strict success from 47.2% to 63.9%, or 16.7 percentage
points. Their successful-start runtime penalties are about 6% and 14% at the
median. The apparently favorable unscaled-TRF speed ratio comes from only five
common successes, all on the same three-stage extractor; it is not a general gain.

| Case | line | dogleg | dogleg_scaled | subspace | trf | trf_unscaled |
|---|---:|---:|---:|---:|---:|---:|
| Methanol/water partial condenser, mass cut, 12 stages | 3/6 | 3/6 | 3/6 | 3/6 | 2/6 | 0/6 |
| Same partial condenser, colored FD Jacobian | 3/6 | 3/6 | 3/6 | 3/6 | 2/6 | 0/6 |
| Methanol/water CMO, 20 stages | 1/6 | 2/6 | 1/6 | 3/6 | 0/6 | 0/6 |
| UNIQUAC acetaldehyde absorber, 5 stages | 2/6 | 5/6 | 1/6 | 5/6 | 0/6 | 0/6 |
| UNIQUAC acetaldehyde stripper, 5 stages | 3/6 | 4/6 | 2/6 | 3/6 | 0/6 | 0/6 |
| Acrylic-acid extractor, flow/semi-analytic Jacobian, 3 stages | 5/6 | 6/6 | 5/6 | 6/6 | 2/6 | 6/6 |

One useful rescue is the absorber with noise 0.5 and seed 47: line search exhausts
80 Jacobians in 3.6578 seconds and ends at residual 18.2851. Dogleg takes 16
Jacobians and 0.2270 seconds, reaching `3.63e-7`, below the requested `2e-6`.
The subspace method reaches essentially the same residual in 0.2419 seconds.

The direct methods rescue slightly different starts: subspace additionally
recovers a severe CMO start, while dogleg additionally recovers a severe stripper
start. The union of line/dogleg/subspace outcomes is 24/36. That is an inference
from separate outcomes; a combined hybrid pipeline was not measured.

## Normal initialization: full column runtime

Times below are medians in seconds from two retained repetitions per method.
`†` means the unit accepted at least one run above the requested residual
tolerance. Failed solves are not counted as speed improvements.

| Case | line | dogleg | dogleg_scaled | subspace | trf | trf_unscaled |
|---|---:|---:|---:|---:|---:|---:|
| Benzene/toluene NRTL, 20 stages | 0.1072 | 0.0983 | 0.0938 | 0.0943 | FAIL | FAIL |
| Hydrocarbon PR, 42 stages | 1.3692 | 1.3788 | 1.3305 | 1.3490 | 23.4136† | TIMEOUT |
| Methanol/water partial condenser, mass cut, 12 stages | 0.0796 | 0.0822 | 0.0843 | 0.1137 | 0.5359 | FAIL |
| Methanol/water CMO, 20 stages | 0.0633 | 0.0904 | 0.0736 | 0.0864 | 1.3207 | FAIL |
| Butanol/water VLLE, 20 stages | 0.8409 | 2.6693 | FAIL | 2.8513 | FAIL | FAIL |
| Acrylic-acid extractor, 20 stages | 0.2179 | 0.2102 | 0.2406 | 0.2205 | 2.2105 | 2.5007† |
| Acrylic-acid extractor, flow Jacobian, 3 stages | 0.0616 | 0.0553 | 0.0594 | 0.0557 | 0.0687 | 0.0644 |
| UNIQUAC acetaldehyde absorber, 5 stages | 0.1977 | 0.2311 | 1.4773 | 0.3974 | 2.7790† | 2.9110† |
| UNIQUAC acetaldehyde stripper, 5 stages | 0.1285 | 0.1573 | 0.1745 | 0.1884 | 1.4825† | FAIL |
| UNIFNIST trace-organic stripper, 5 stages | 0.6066 | 2.0121 | 2.0904 | 2.1274 | FAIL | FAIL |

Unscaled TRF's PR workers exceed the 180-second wall-clock limit, including
setup; their exact column-only solve times and evaluation counts are unavailable.
They are failures, not measured 180-second successful solves.

| Method | Accepted / runs | Strict / runs | Maximum accepted residual | Maximum external component error | Maximum external energy error |
|---|---:|---:|---:|---:|---:|
| line | 20/20 | 20/20 | 9.353e-7 | 6.930e-9 | 3.031e-7 |
| dogleg | 20/20 | 20/20 | 1.429e-6 | 1.642e-8 | 3.031e-7 |
| dogleg_scaled | 18/20 | 18/20 | 1.907e-6 | 4.840e-9 | 3.031e-7 |
| subspace | 20/20 | 20/20 | 1.428e-6 | 1.641e-8 | 3.031e-7 |
| trf | 14/20 | 8/20 | 2.507e-5 | 7.037e-5 | 8.931e-5 |
| trf_unscaled | 6/20 | 2/20 | 5.112e-5 | 8.676e-5 | 1.001e-4 |

Component error is the maximum external component molar imbalance divided by
total inlet flow. Energy error is the external enthalpy imbalance after heat
duty, divided by the largest absolute inlet/outlet enthalpy flow or heat duty.
For distillation, reported net duty is itself formed from the external enthalpy
difference, so that check is partly bookkeeping; the MESH energy residual remains
the relevant independent convergence constraint. CMO uses its existing
constant-molar-overflow approximation rather than a full MESH energy model.

Across the 20 common strict normal runs, median time ratios are 1.122 for dogleg
and 1.226 for subspace. Summing per-case median times over the ten cases gives
3.672596 seconds for line, 6.985086 for dogleg, and 7.484047 for subspace: about
1.90 and 2.04 times the baseline. These sums describe the case collection, not a
measured connected flowsheet.

The large penalties have concrete causes:

- VLLE takes 25 aggregate Jacobians with line search, versus 102 with dogleg or
  subspace. Both successful trust methods finish with the same 14-stage LLE
  topology as line search, but take about three times longer.
- The trace stripper's first initializer fails for all three methods. Line search
  gives up after 13 Jacobians; dogleg/subspace expend the full 80. All then converge
  from the second initializer in eight Jacobians, giving totals 21 versus 88 and
  essentially identical final residuals. An early failure/stall policy could be
  investigated separately; this comparison retains the supplied policies/budgets.
- Scaled dogleg's absorber takes 80 Jacobians on its failed first initializer and
  three on the successful second, versus line search's direct 11-Jacobian solve.
- The PR column takes 12 Jacobians with line and each direct trust method. Scaled
  TRF takes 120, lasts about 23 seconds, and still only reaches relaxed acceptance.

The direct trust methods reproduce the normal solutions closely: dogleg's largest
outlet mole-fraction difference from line is `8.49e-8`, temperature difference
`4.83e-6 K`, and flow difference divided by inlet flow `4.05e-9`. Subspace has
nearly identical bounds. Accepted scaled-TRF results differ by up to `1.34e-3` in
mole fraction and `0.406 K`, consistent with several missing the strict tolerance.
Unscaled-TRF output comparisons on the shared retained repetition show differences
up to `1.75e-3` and `0.435 K`; only the simple three-stage extractor reaches strict
convergence in its normal runs.

## Interpretation and limits

The useful result is the improvement from **unscaled, sparse-direct trust
regions**, especially on absorption and extraction stress cases. A line-search
step stays on the Newton direction; dogleg can move toward the Cauchy direction
when that prediction is poor. The Newton/gradient subspace method offers another
such direction choice, but does not beat dogleg's overall reliability here.

Jacobian scaling was detrimental in this sample. The native variables are already
transformed and equations are scaled, so another column-norm metric is not
automatically helpful. Removing that scaling from SciPy TRF did not rescue its
performance: the generic iterative least-squares route remains a poor fit in these
tests. This does not prove all TRF configurations or trust-region formulations
would behave the same way; there was no exhaustive radius, scaling, or LSMR tuning.

The practical choice supported by these measurements is to keep line search as
the default and investigate a dogleg restart **from the original seed** after
line-search failure. The observed rescue counts come from that original seed,
not from the line search's failed final iterate. Production integration and its
combined runtime have not been implemented or measured.

The sample is ten isolated column fixtures and 36 controlled starts, with only
two retained normal timing repetitions. Small timing differences are not strong
evidence of speed improvements. Recycle-flow convergence, different specifications,
stage efficiencies, and other thermodynamic packages are not characterized by
this experiment. No physical-law violation was established by these probes;
reported conservation errors should still be interpreted with the declared
strict/relaxed tolerances.

## Validation and reproducibility

Validation includes 200 seeded trust-step radius/descent checks, a regularized
Newton-endpoint check, six analytic Rosenbrock root solves, five two-variable
physical-domain rejection/root solves, Ruff, and source-hash checks. All final
216 stress records are standard JSON with finite terminal residuals and matched
initial vectors. A final colored-FD harness check confirms native and dogleg
reach the same `2.11e-11` residual in seven Jacobians, with nonzero Jacobian timing.
An earlier one-variable synthetic TRF/LSMR probe hit SciPy's internal two-dimensional
subspace `IndexError`; the benchmark column problems have more variables, and the
multi-variable checks pass. No library workaround was added.

The maintained harness now reuses the existing benchmark's native Jacobian timers,
including colored FD. Early native Jacobian-time fields were incomplete and are
not used in any comparison here: the reported speeds use complete elapsed solve
times, and work comparisons use evaluation counts. Non-finite rejected-trial ratios
are recorded as null; unknown timeout evaluation counts are likewise explicit.
Unscaled TRF is opt-in in the harness to avoid making its expensive timeout probes
part of every default run.

The complete generated analysis is
[/tmp/pfdsim-trust-analysis-final-20261006.md](/tmp/pfdsim-trust-analysis-final-20261006.md).
Authoritative data directories are:

- `/tmp/pfdsim-trust-stress-paired-20261006`: all six methods on shared captured problems.
- `/tmp/pfdsim-trust-end-final-20261006`: primary five methods; retain repeats 2 and 3.
- `/tmp/pfdsim-trust-unscaled-end-20261006`: unscaled TRF; retain completed repeats 1 and 2.

Each contains a manifest, saved harness source, incremental logs, and JSONL
records. Intermediate pilots/follow-ups are retained separately for audit; they
are not pooled into the authoritative stress results.

Run from the repository root and choose new output directories:

```bash
set -o pipefail
.venv/bin/python scripts/performance/benchmark_column_trust_region.py \
  --output /tmp/column-trust-normal-new --repeats 3 --warmup-repeats 1 \
  --cases benzene_toluene_20 hydrocarbon_pr_42 partial_mass_12 cmo_20 \
          vlle_butanol_20 acrylic_extraction_20 acrylic_extraction_flow_3 \
          absorption_5 stripping_5 trace_stripping_5 \
  2>&1 | tee /tmp/column-trust-normal-new.log

.venv/bin/python scripts/performance/benchmark_column_trust_region.py \
  --output /tmp/column-trust-stress-new --mode raw \
  --methods line dogleg dogleg_scaled subspace trf trf_unscaled \
  --noise .5 1.5 --seeds 11 29 47 \
  --cases partial_mass_12 partial_mass_fd_12 cmo_20 absorption_5 stripping_5 \
          acrylic_extraction_flow_3 \
  2>&1 | tee /tmp/column-trust-stress-new.log

.venv/bin/python scripts/performance/analyze_column_trust_region.py \
  --stress /tmp/pfdsim-trust-stress-paired-20261006 \
  --end-to-end /tmp/pfdsim-trust-end-final-20261006 \
               /tmp/pfdsim-trust-unscaled-end-20261006 \
  --warmup-repeats 1 --method-repeats trf_unscaled=1,2
```

To probe unscaled TRF separately, add `--methods trf_unscaled`; its PR case is
known to reach the worker timeout. The defaults avoid that extra diagnostic run.

## Design Decisions

- Keep new algorithms in experimental scripts and inject them only in worker
  processes. This makes the experiment reviewable without changing production
  behavior, unlike adding a new production solver before evidence supports it.
- Reuse the authoritative residuals, Jacobians, case factories, and native timing
  utilities. Reimplementing column physics would confound convergence comparisons.
- Measure both complete units and shared raw problems. Unit-only tests can hide
  failed attempts behind fallbacks; raw-only tests omit their practical cost.
- Retain sparse direct Newton solves for the custom trust methods. Dense Hessian
  approaches would change the existing sparse performance model; sparse LSMR is
  evaluated explicitly through the separate SciPy variants.
- Use requested residuals as the success criterion and report relaxed acceptance
  separately. Optimizer termination flags or successful outlet construction alone
  do not establish that the requested equations are solved.
- Run timing jobs sequentially and exclude compilation warm-up. Parallel execution
  would introduce contention into the speed comparison; raw model reuse reduces
  avoidable initialization work while keeping paired starts identical.
