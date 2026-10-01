# Adsorption performance and SQLite connection lifetimes

Investigation on 2026-10-01, comparing production at `a36931d` with the
changes accompanying this report. No adsorption parameters or thermodynamic
equations were changed.

## Measurement

The maintained harness is
[benchmark_adsorption.py](../../scripts/performance/benchmark_adsorption.py).
It pins numerical libraries to one thread, creates a fresh initialized
Simulator for each solve, excludes initialization and one warm-up solve, and
reports the median of three measured solves. Profiling and optional warning
allocation tracing run separately from timing. There is no randomness.
These are direct local wall times, not CPU-reference-normalized suite timings.

| Example | Before (ms) | After (ms) | Speedup |
|---|---:|---:|---:|
| Air 3A dryer | 233.789 | 10.570 | 22.12x |
| DCM 3A dryer | 233.482 | 12.912 | 18.08x |
| Ethanol 3A dryer | 255.444 | 13.907 | 18.37x |
| Competitive 13X | 242.494 | 146.867 | 1.65x |
| Propane/propylene 4A | 4900.486 | 2470.762 | 1.98x |

Artifacts from this investigation are in
`/tmp/pfdsim-adsorption-baseline-20261001` and
`/tmp/pfdsim-adsorption-after-20261001`, including result snapshots and
cProfile reports. The harness now writes code/dependency provenance for future
runs; those two earlier artifact directories predate the manifest addition.

For a future run, choose a fresh directory and persist both output streams:

```bash
set -o pipefail
python scripts/performance/benchmark_adsorption.py \
  --output /tmp/pfdsim-adsorption-next --repeats 3 \
  2>&1 | tee /tmp/pfdsim-adsorption-next.log > /dev/null
```

## Causes and changes

In the profiled air dryer, differential isosteric heat calculation consumed
0.427 s of a 0.499 s solve (86%). It numerically differentiated two inverse
IAST calculations. With only one adsorbate, each inverse unnecessarily solved
for spreading potential while repeatedly inverting that potential back to
pure fugacity. A pure inventory now directly inverts its loading. The same
fixed-inventory heat derivative is retained; no heat approximation replaces it.
Inventories at or above the pure saturation capacity are rejected rather than
returning an arbitrary fugacity from a floating-point saturation plateau.

Five-element GSTA reductions and small IAST reductions incurred substantial
array creation and general array-dispatch overhead. They now use a shared
max-shifted scalar log-sum-exp reduction and shared GSTA weights. GSTA spreading
still uses a stable softplus calculation, including positive trace potentials
when adding the terms to one would round to one.

The propane/propylene profile spent 8.832 s of 9.496 s in quadrature (93%),
inside repeated numerical spreading inversions. Inversions now build a verified
bracket near the previous solution, expanding geometrically as needed within
the existing representable log-fugacity bounds. Brent's method and its tolerance
remain unchanged. The previous solution is only an initial guess, never a
cached answer or interpolated property. Loading and spreading inversions share
the bracketing implementation.

Numerical Toth/custom spreading integrals remain the principal cost in the 4A
example. Replacing those integrals or introducing a coupled analytic Jacobian
would require a separate numerical validation; these changes retain the
existing quadrature and coupled balance solver.

## Correctness

Across the five example snapshots, the maximum absolute mole-fraction change
was `1.138e-13`, flow change `1.266e-14 kmol/h`, and molar stream enthalpy change
`4.108e-8 kJ/kmol`. The maximum relative heat-duty change was `2.606e-14`.
Differential heat values changed by at most `1.338e-10` relative. Relative mass
balance errors were at most `2.274e-16`; reported energy balance errors were zero.

All 164 focused tests passed across adsorption models, molecular-sieve units,
runtime caches, runtime locks, connection lifetimes, and property lookup
(162 in the main run, plus two subsequent connection-setup failure checks).
Added numerical checks cover all eight pure isotherm forms, fixed-inventory
heats against independent implicit derivatives, trace GSTA spreading,
unreachable capacities, and large changes in spreading-query order.

## ResourceWarnings

A separate tracemalloc probe reproduced eight unclosed-connection warnings:

| Allocation site before the fix | Count |
|---|---:|
| `SQLiteJSONCache._connect` | 3 |
| `SQLiteLeaseLock._connect` | 3 |
| `ChemicalDatabase._ensure_smiles_cache` | 1 |
| `ChemicalDatabase._cached_smiles` | 1 |

SQLite's connection context manager commits or rolls back but does not close
the connection. Garbage collection consequently emits a warning later, often
at an unrelated thermodynamic calculation or third-party lookup line.

The cache and lease connection helpers now own both transactions and connection
closure, including failures during setup. All five SMILES connection sites
explicitly close after transaction handling. Regression tests check immediate
closure, committed persistence, rollback, missing rows, lease contention,
renewal/release, and SQL errors. An expanded post-fix probe exercising these
paths reported zero ResourceWarnings.

The five adsorption examples alone did not reproduce ResourceWarnings in their
allocation probes; the warnings from the broader example suite are a separate
connection-lifetime problem rather than the cause of the dryer bottleneck.
