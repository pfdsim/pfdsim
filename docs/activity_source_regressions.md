# Numerical validation of activity source fits

Approved interaction JSONs and the runtime tables are publication artifacts.
Running tests or a regression check does not replace their coefficients. New
fits remain candidates until their data fit, held-out behavior, phase stability
and extrapolation have been reviewed and the normal builder is run explicitly.

The Moreau generators share `binary_regression.py` and the source equations in
`fit_moreau_ovejero_2012_2007_activity.py`. Excess enthalpy uses the exact
temperature derivative of the same excess Gibbs energy. Complex-step parameter
Jacobians avoid differencing a finite-difference enthalpy. Source pressure solves
must converge. The real-only Mohsen-Nia runtime calculations instead use a
scale-aware central Jacobian. Fendu's Antoine fit has an analytic Jacobian.
The multi-start regression checks finite results and a projected, column-scaled
gradient cosine below `1e-7`, including the correct sign at active bounds.

The scientific tests are in `tests/test_source_activity_regressions.py`:

- Raw observations, source citations, model choices, temperature bounds and
  continuation policies remain exact. Approved coefficients export exactly
  through the publication builder. The shared input assertion also checks
  Fendu's per-system observations and exclusions, both in tests and in the
  cross-environment checker.
- Fresh regressions must respect their original bounds, improve or retain the
  uncertainty-weighted objective, and satisfy the stationarity check. Fendu also
  retains its held-out composition-series checks.
- Reports evaluated at fixed approved parameters use numerical tolerances in
  observable units: `1e-6 K`, `1e-8` mole fraction, `1e-3 J/mol`, `1e-7 kPa`,
  `1e-6` pressure percentage points, `1e-9` infinite-dilution gamma, and
  `1e-10` derived interaction values.
  Raw data and coefficients never receive
  these allowances. These budgets cover root-solver accuracy and caloric
  differentiation while remaining below the source-data precision.
- Published-parameter reproduction reports remain checked for Fendu, Mohsen-Nia,
  toluene and hexane. Fendu pressure statistics allow `1e-6` percentage points
  and their logarithmic AICc allows `1e-4`; point counts remain exact.
- Fresh selected Moreau fits retain held-out checks (RMSE budgets `0.01 K` and
  `5e-5` vapor mole fraction), a dense liquid-Gibbs curvature audit over the
  published temperature range, and finite, unclipped continuation behavior.
  Changed extrapolated predictions are a publication review concern; fresh
  fits are not required to reproduce old extrapolated digits.

Install the optional geometry backend and run the focused checks:

```bash
uv sync --extra dipole-xtb --inexact
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python -m pytest tests/test_source_activity_regressions.py \
  tests/test_fit_source_comparison.py > /tmp/source-regressions.log 2>&1
```

The maintained numerical check records predictions, objectives, stationarity,
environment versions and CPU time against the same isolated source fixtures:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python scripts/activity_fitting/check_source_regressions.py \
  --output /tmp/source-fits-current.json > /tmp/source-fits-current.log 2>&1
```

Run that command under another installed Python environment with a different
output filename and `--reference /tmp/source-fits-current.json` to compare builds.
It refuses to overwrite an existing report and never writes approved data.
No library-version injection or compatibility layer is part of the generators.
