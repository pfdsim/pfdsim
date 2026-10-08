# Azeotropic first-cut overdraw — 2026-10-07

Slightly exceeding an azeotropic first-cut inventory can produce severe column
convergence problems. An azeotropic initializer is decisive in two tested ternary
examples, but it is neither always necessary nor always sufficient. These cases
also give more targeted evidence for trust regions than random starting-vector
noise: trust regions can rescue a cheap start and accelerate an azeotropic start.

All experiments are maintained in
[probe_azeotropic_overdistillation.py](probe_azeotropic_overdistillation.py).
Production code was not modified. The script reuses the existing case factories,
native column residuals/Jacobians, and
[experimental trust-region solvers](benchmark_column_trust_region.py).

## What “beyond the azeotrope” means in these experiments

For feed mole fractions z and first azeotrope mole fractions a, define the
inventory capacity

    c_az = min(z_i / a_i), over participating components with a_i > 1e-8.

Then specify D/F = factor * c_az, with factors 0.98, 1.005, 1.02, and 1.05.
The last three draw 0.5%, 2%, or 5% more distillate than the largest amount the
feed could supply at exactly the azeotrope composition. These percentages are
relative to the first-cut capacity, not percentage points of the total feed.
No product purity is prescribed; the distillate is free to include material
from the next cut. No Gaussian perturbations were applied.

This capacity is an inventory benchmark, not a rigorous maximum column throughput.
Finite stages, reflux, feed condition, and phase behavior still matter. Azeotrope
compositions and capacities below are predictions of the project's thermodynamic
models, not experimental measurements. The “first” candidate is the lowest-boiling
minimum-boiling azeotrope identified by the existing candidate finder.

Insisting on an exactly azeotropic product at D/F > c_az would require a negative
bottoms inventory of the limiting component. Instead, the successfully solved
overdrawn columns change their product composition and satisfy material balance.
This distinction avoids confusing overdraw with a prescribed composition on the
other side of an azeotropic boundary. An azeotrope has equal liquid and vapor
composition at equilibrium; see [IUPAC's teaching explanation](https://list.iupac.org/didac/Didac05/Content/ST07.htm).

## Cases and controls

All feeds are 100 kmol/h; condensers are total, pressure drops are zero, and the
stage model is VLE. Binary feeds and the ethanol/water/propanol feed enter at
298.15 K. The nitrile feed enters at its model bubble point, 343.834904 K.

| Case | Thermo | Feed mole fractions | Stages / feed stage | Reflux ratio | Pressure bar | First azeotrope | Capacity D/F |
|---|---|---|---:|---:|---:|---|---:|
| Ethanol/water | UNIFAC | 0.10 / 0.90 | 30 / 21 | 8 | 1 | 0.892527 ethanol, 0.107473 water | 0.112041376 |
| Isopropanol/water | NRTL | 0.20 / 0.80 | 30 / 20 | 8 | 1 | 0.680956 IPA, 0.319044 water | 0.293704794 |
| Acrylonitrile/acetonitrile/water | UNIFNIST, existing test group assignments | 0.60 / 0.10 / 0.30 | 80 / 70 | 10 | 1.01325 | 0.696835 acrylonitrile, 0.303165 water | 0.861036056 |
| Ethanol/water/1-propanol | NRTL | 0.10 / 0.85 / 0.05 | 40 / 28 | 8 | 1 | 0.891618 ethanol, 0.108382 water | 0.112155680 |

The model azeotrope temperatures are respectively 350.972571, 353.015482,
343.094780, and 351.069460 K. All first candidates passed the local liquid
stability test and their participating components had K values approximately one.

The primary comparison has four cases, four cuts, two initializers, and three
methods: 96 full unit solves. Each uses tolerance 1e-6, identical strict acceptance
threshold, and a maximum of 100 iterations/Jacobian evaluations per nonlinear
attempt. Finite-difference step is fixed at 1e-6 and colored-Jacobian fallback is
disabled for that comparison. Every initializer/cut has matching starting-vector
fingerprints across the three methods. Complete outlet construction and normal
unit validation still run.

Workers are sequential, with single-threaded BLAS. Thermodynamic caches are
shared within each case. Python's hash seed is zero; Python/NumPy are seeded
20261007 for reproducibility, though the experiment adds no random noise.
Times include the initializer and full unit solve, but exclude imports, feed setup,
azeotrope discovery, and supplemental phase checks. This is an exploratory
single-pass comparison; Jacobian counts and failures are stronger evidence than
small timing differences.

## Native line search: initializer comparison

Each cell is the final infinity-norm residual. ✓ indicates < 1e-6; × indicates
failure. For the primary run there is one nonlinear attempt per cell.

| Case | Overdraw | Cheap start | Azeotropic start |
|---|---:|---:|---:|
| Ethanol/water | +0.5% | ✓ 4.24e-8 | ✓ 1.07e-10 |
| Ethanol/water | +2% | × 1.50e-1 | × 1.51e-1 |
| Ethanol/water | +5% | × 3.68e-1 | × 2.50 |
| Isopropanol/water | +0.5% | × 1.49e-1 | × 2.42e-1 |
| Isopropanol/water | +2% | × 1.10e-1 | × 3.34 |
| Isopropanol/water | +5% | × 2.41e-1 | × 3.40 |
| Nitrile ternary | +0.5% | × 4.09e-1 | ✓ 1.97e-9 |
| Nitrile ternary | +2% | × 9.29e-1 | ✓ 1.98e-7 |
| Nitrile ternary | +5% | × 1.27 | × 5.23 |
| Ethanol/water/propanol | +0.5% | × 2.28 | ✓ 9.37e-7 |
| Ethanol/water/propanol | +2% | × 2.29 | × 1.40e-1 |
| Ethanol/water/propanol | +5% | × 2.31 | × 1.89 |

At the control cut, 2% below inventory capacity, cheap/azeotropic line-search
outcomes are: ethanol/water both converge; IPA/water both converge; nitrile cheap
converges but azeotropic fails; ethanol/water/propanol cheap fails but azeotropic
converges. Difficulty is not monotonically determined by cut or initialization.

## Trust regions on the informative ternaries

| Nitrile overdraw | Initializer | Line residual / Jacobians / seconds | Dogleg residual / Jacobians / seconds | Subspace residual / Jacobians / seconds |
|---|---|---|---|---|
| +0.5% | cheap | FAIL 0.409 / 25 / 6.37 | 7.23e-10 / 87 / 7.31 | 7.08e-10 / 87 / 7.31 |
| +0.5% | azeotropic | 1.97e-9 / 8 / 0.85 | 1.87e-8 / 7 / 0.64 | 1.87e-8 / 7 / 0.67 |
| +2% | cheap | FAIL 0.929 / 93 / 27.06 | FAIL 0.463 / 25 / 2.17 | FAIL 0.289 / 100 / 8.83 |
| +2% | azeotropic | 1.98e-7 / 68 / 15.43 | 2.18e-8 / 34 / 3.47 | 1.79e-8 / 10 / 0.99 |
| +5% | cheap | FAIL 1.274 / 100 / 23.55 | 2.94e-12 / 57 / 5.83 | 3.93e-12 / 57 / 4.86 |
| +5% | azeotropic | FAIL 5.229 / 6 / 2.07 | FAIL 2.352 / 6 / 0.54 | FAIL 1.110 / 100 / 8.58 |

At +2%, azeotropic initialization enables all three methods to solve, and the
subspace method is about 15.6 times faster than line search in this pass. At
+0.5%, azeotropic initialization reduces trust-region Jacobian work from 87 to
seven, a factor of 12.4. At +5%, the cheap start with a trust region succeeds where
the azeotropic start fails, showing that initialization and globalization interact.

For ethanol/water/propanol at +0.5%, all cheap-start methods fail. All azeotropic
methods converge in 14 Jacobians: residuals 9.37e-7 for line, 5.33e-7 for dogleg,
and 5.32e-7 for subspace. At +2% and +5%, none of the tested combinations converge.

The two binaries do not show a universal rescue. Ethanol/water succeeds from
either initializer with every method at +0.5%; none succeeds at +2% or +5%.
IPA/water fails from both initializers with all three methods at each overdraw.
Those are failed numerical experiments, not proofs of infeasible specifications.

Subsequent [binary diagnostics](binary_azeotropic_overdraw_diagnosis.md) establish
a feasible IPA/water +2% solution, including recovery with unchanged native line
search from a better profile. The original failures above remain accurate for
their tested starts and budgets. The follow-up identifies a collapsed bottom
composition and uniform Newton-step cap as the immediate native stall mechanism;
ethanol/water +2% feasibility remains unresolved.

## Products and initialization mechanism

The nitrile +0.5% case has D/F = 0.865341237 and its azeotropically initialized
line-search solution has distillate mole fractions:

- Acrylonitrile: 0.693011045
- Acetonitrile: 0.003869012
- Water: 0.303119943

The +2% case has D/F = 0.878256778 and distillate:

- Acrylonitrile: 0.682865744
- Acetonitrile: 0.014142108
- Water: 0.302992148

The next component, acetonitrile, increasingly enters the overhead as the draw
exceeds the first azeotrope capacity. The overhead is not required to remain
exactly at the first azeotrope composition. Component balance errors in these
two line-search solutions are 3.89e-15 and 1.73e-13 of feed flow.

The ethanol/water/propanol +0.5% solution has D/F = 0.112716458 and overhead
ethanol/water/propanol fractions 0.879557188 / 0.120442777 / 3.52e-8. Additional
water enters the overhead; the bottoms retain propanol and have ethanol fraction
0.000968603. Its component balance error is 3.96e-9 of feed flow.

The cheap estimate starts with feed-like compositions, while the azeotropic
initializer builds product endpoints from azeotropic pseudo-components and
remaining component inventories, then constructs a stage profile. That supplies
the solver with a better representation of which material spills into the next
cut. This is a plausible explanation for the observed basin-of-convergence change,
not a proof that every failure has that cause.

Initial residual size alone does not identify the better seed. In the nitrile
+0.5% case, cheap initialization starts at norm 2.24 and merit 5.28; the azeotropic
start has the larger norm 5.03 and merit 17.75, yet converges easily. It is the
structure of the starting profile, not simply a smaller residual, that helps.

## Other initializers and continuation

The alternative initializer probe uses line search at +0.5% and +2%, with the
same strict residual criterion and per-attempt budget:

| Case | Overdraw | auto | cmo |
|---|---:|---|---|
| Ethanol/water | +0.5% | converges, 4.24e-8 | converges, 7.44e-8 |
| Ethanol/water | +2% | fails | fails |
| Nitrile ternary | +0.5% | converges, 3.77e-9 | fails |
| Nitrile ternary | +2% | fails | fails |

The nitrile auto seed uses a coarse rigorous column and consumes 19 total
Jacobians including initialization, versus eight for the explicit azeotropic
seed. Therefore a failure of the cheap start is not automatically a failure of
the project's default auto initializer.

Continuation was tested with the existing recycle-profile warm-start mechanism,
starting at factor 0.98 and increasing through 0.995, 1, 1.0025, 1.005, 1.01,
1.02, and 1.05. Profiles are retained only after successful solves; failed steps
do not replace the last successful profile.

- Ethanol/water reaches factors through 1.005, but not 1.01 or higher within the
  100-iteration budget. Its factor-1.005 continuation solve uses 80 Jacobians,
  compared with 12 from a fresh cheap start, so nearby warm starts are not
  automatically better.
- IPA/water only converges at 0.98; continuation fails even at 0.995.
- Nitrile converges at 0.98 and 0.995, but fails at 1 and above along this path,
  despite the independent azeotropic-start solutions at 1.005 and 1.02.
- Ethanol/water/propanol's cheap initial continuation seed fails at 0.98, so it
  never obtains a profile to continue. Subsequent cold attempts also fail.

These continuation results do not establish physical infeasibility. They do show
that simply reusing a nearby converged profile is not a reliable substitute for
an appropriate azeotropic seed in this sample.

## Native Jacobian retry policy

The follow-up enables all four native finite-difference step sizes and the
colored-Jacobian fallback, while retaining the strict 1e-6 criterion and a
100-iteration/Jacobian budget per attempt. Thus it can spend considerably more
total work than the controlled primary comparison.

For the nitrile ternary, retries do not rescue either cheap-start failure:

| Overdraw | Initializer | Outcome | Best residual | Attempts | Total Jacobians | Seconds |
|---|---|---|---:|---:|---:|---:|
| +0.5% | cheap | fails | 0.408734 | 8 | 214 | 62.36 |
| +0.5% | azeotropic | converges | 1.97e-9 | 1 | 8 | 0.91 |
| +2% | cheap | fails | 0.928705 | 8 | 777 | 303.49 |
| +2% | azeotropic | converges | 1.98e-7 | 1 | 68 | 15.58 |

For ethanol/water/propanol at +0.5%, the cheap start also fails after eight
attempts, 427 Jacobians, and 103.73 seconds, with best residual 2.280235.
The azeotropic seed converges in 14 Jacobians and 0.98 seconds to 9.37e-7.
At +2%, the azeotropic seed still fails after eight attempts and 484 Jacobians,
with best residual 0.140485, so retries do not make it universally sufficient.
The +2% cheap start also fails: eight attempts, 428 Jacobians, 107.48 seconds,
and best residual 2.285501.

Changing the Jacobian approximation repeatedly reproduces essentially the same
failure basin. For this fixture, changing the seed is much more effective than
spending those retries. The 100-per-attempt limit differs from the nitrile
regression's original 400 limit, so these are not claims about every possible
budget or termination policy.

## Phase and conservation checks

All 37 successful primary unit solves meet the requested residual below 1e-6.
The largest independent external component balance error is 1.126e-8 of inlet
flow. All 1,820 liquid-stage spinodal checks report local stability.

A separate audit invokes the existing liquid-liquid-equilibrium solver at each
first azeotrope and at the top, feed stage, and bottom of one representative
successful column per cut. All 31 sampled checks report no LLE split. This is
stronger than the local Hessian check alone but is still a sampled numerical audit,
not a global stability proof for every stage and every result. No physical-law
violation was established by these probes.

## Validation

The completed experiment contains 144 full-unit attempts: 96 primary, 32
continuation, eight alternative-initializer, and eight native-retry runs. All
records were recovered from persisted files and their expected counts verified.
The primary matrix has 32 matched starting-vector groups, each shared by line,
dogleg, and subspace. All accepted primary residuals, component balances, and
local stability results passed independent result checks. A final two-solve
harness check reproduces the ethanol/water +0.5% residuals 4.24e-8 and 1.07e-10
with 12 and nine Jacobians, respectively. Ruff and command-line checks pass.
Production solver/thermo files and reused benchmark scripts match the source
hashes in the native-retry run manifest. Only the new script and this report are
added to the repository.

## Interpretation

The targeted ternaries support PFDSim's intuition: drawing a little past the
first azeotropic inventory can make a naive initialized MESH solve fail even
when a valid solution is demonstrated by a better initializer. Azeotropic
initialization can be much more effective than replacing line search alone.

The practical conclusion remains conditional. Azeotropic initialization can fail,
automatic coarse initialization can sometimes succeed without it, and a trust
region can sometimes recover a cheap start. The strongest combination here is
an azeotropic seed plus a trust region in the nitrile +2% case. The four chosen
fixtures and cut sweep establish examples, not a population-wide failure rate.

## Reproduction and artifacts

Run from the repository root, choosing a new directory each time:

```bash
set -o pipefail
.venv/bin/python scripts/performance/probe_azeotropic_overdistillation.py \
  --output /tmp/aze-overdraw-new --budget 100 \
  2>&1 | tee /tmp/aze-overdraw-new.log

.venv/bin/python scripts/performance/probe_azeotropic_overdistillation.py \
  --output /tmp/aze-overdraw-continuation-new --initializers continuation \
  --methods line --factors .98 .995 1 1.0025 1.005 1.01 1.02 1.05 \
  2>&1 | tee /tmp/aze-overdraw-continuation-new.log

.venv/bin/python scripts/performance/probe_azeotropic_overdistillation.py \
  --output /tmp/aze-overdraw-other-init-new --initializers auto cmo \
  --methods line --factors 1.005 1.02 --cases ethanol_water nitrile_ternary \
  2>&1 | tee /tmp/aze-overdraw-other-init-new.log

.venv/bin/python scripts/performance/probe_azeotropic_overdistillation.py \
  --output /tmp/aze-overdraw-phase-new --audit-directory /tmp/aze-overdraw-new \
  2>&1 | tee /tmp/aze-overdraw-phase-new.log

.venv/bin/python scripts/performance/probe_azeotropic_overdistillation.py \
  --output /tmp/aze-overdraw-retries-new --native-retries --methods line \
  --initializers cheap_estimate azeotropic --factors 1.005 1.02 \
  --cases nitrile_ternary ethanol_water_propanol \
  2>&1 | tee /tmp/aze-overdraw-retries-new.log
```

The actual saved directories are:

- `/tmp/pfdsim-aze-overdraw-primary-20261007`: 96 primary solves, per-case metadata,
  raw profiles, attempt histories, logs, and [summary](/tmp/pfdsim-aze-overdraw-primary-20261007/summary.md).
- `/tmp/pfdsim-aze-overdraw-continuation-20261007`: 32 continuation attempts and
  [summary](/tmp/pfdsim-aze-overdraw-continuation-20261007/summary.md).
- `/tmp/pfdsim-aze-overdraw-alternative-init-20261007`: eight auto/CMO probes and
  [summary](/tmp/pfdsim-aze-overdraw-alternative-init-20261007/summary.md).
- `/tmp/pfdsim-aze-overdraw-phase-audit-20261007`: 31 supplemental phase checks.
- `/tmp/pfdsim-aze-overdraw-native-retries-20261007`: eight native retry-policy
  follow-ups and [summary](/tmp/pfdsim-aze-overdraw-native-retries-20261007/summary.md).

Each run retains the experimental harness and reused solver/case scripts. The
maintained script supports `--inspect`, `--summarize-only`, alternative budgets,
and `--native-retries`. Summary files use exclusive creation and do not overwrite
earlier reports. Later summary generation selects the best completed residual
for failed direct-initializer retry runs; early raw records can contain the last
retry residual instead, and those originals are retained unchanged.

## Design Decisions

- Define overdraw relative to azeotropic component inventory, with product
  composition free. Prescribing an unreachable azeotropic purity would test
  infeasibility instead of the convergence issue PFDSim described.
- Use real initialized profiles, without arbitrary starting-vector noise. This
  isolates the cut/initializer interaction in physically specified columns.
- Reuse the existing thermodynamic candidates, column models, and trust solvers.
  A separate implementation of column physics would confound the comparison.
- Hold tolerance, budget, and Jacobian configuration fixed in the primary matrix,
  then examine native retries separately. This distinguishes globalization from
  rescue provided by repeated Jacobian approximations.
- Use the existing recycle-profile machinery for continuation rather than adding
  a production warm-start path. Only successful profiles seed later cuts.
- Supplement residual and mass-balance checks with phase checks. A converged VLE
  residual alone does not establish physical stability of the liquid phases.
