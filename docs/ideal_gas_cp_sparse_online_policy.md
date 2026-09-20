# Sparse online ideal-gas Cp policy

This document is the authoritative runtime policy for sparse online ideal-gas
heat-capacity data. It supersedes [Cp_matrix.md](Cp_matrix.md).

## Resolution order

Direct experimental and empirical correlations are exhausted first:

1. Explicit PFD/provided correlations.
2. Bundled canonical empirical curves, in their compiled source priority.
3. Native Shomate correlations provided directly by the online NIST WebBook.

Stored adjusted-Psi4 curves are computational rather than direct sources and
are deferred. Once the direct sources above are unavailable, clean tabulated
online gas-Cp points are combined with GFN2-xTB RRHO as follows:

| Accepted online points | Selected kernel | Quality |
| ---: | --- | ---: |
| 10 or more | Robust Shomate within the retained point range; independently boundary-matched affine-xTB continuations outside; exact piecewise kernel | `0.94` |
| 3–9 | Full-range affine xTB, `Cp=a+b Cp_xTB`, fitted to every accepted point | `0.93` |
| 1–2 | Full-range constant-residual xTB, `Cp=Cp_xTB+a`, using the mean observed residual | `0.91` |

If a validated ten-point-or-more Shomate fit itself covers the complete
273.15–1500 K ideal-gas Cp kernel domain, select it directly at quality `0.96`
without invoking xTB. Direct fitted Shomate curves must remain positive and
analytically monotonic nondecreasing across their retained range.

If xTB is unavailable or unusable and at least ten points remain, retain the
robust Shomate only on its source-supported range at quality `0.92`. Do not
extrapolate that data-only Shomate curve.

Only after these paths are exhausted, try:

1. Stored adjusted-Psi4 RRHO at quality `0.89`.
2. Plain GFN2-xTB RRHO at quality `0.89`.
3. The legacy in-range online linear or constant tabulated-data kernels.
4. The bundled formula atom-increment Shomate fallback.

Any absent dependency, missing structure, unsupported identity, failed xTB
optimization or Hessian, unresolved significant imaginary mode, nonpositive
correction, ill-conditioned calibration, or invalid kernel is a normal
provider miss. It must fall through without preventing a lower tier from
resolving Cp.

## Piecewise 10-point-or-more kernel

Let `[Tmin,data, Tmax,data]` be the retained robust-Shomate point range. Fit one
affine xTB calibration to the retained points. The executable kernel contains
up to three exact analytic segments:

1. Lower affine-xTB continuation, shifted to equal Shomate at `Tmin,data`.
2. Native fitted Shomate over the source-supported range.
3. Upper affine-xTB continuation, shifted to equal Shomate at `Tmax,data`.

The two continuation offsets are independent. This guarantees continuous Cp
without extrapolating the narrow Shomate fit. A derivative match is not
required. Enthalpy and entropy increments split at segment boundaries and sum
the segments' analytic primitives.

The piecewise kernel must preserve the source-supported range separately from
its larger evaluation domain in its serialized payload.

## xTB computation and caching

Plain xTB uses a GFN2-xTB optimized geometry and an ASE central finite-
difference Hessian. The resulting RRHO curve is fitted to the portable
rational-Chebyshev kernel contract, so point evaluation and enthalpy/entropy
integration never invoke xTB during simulation.

If the initial Hessian has a mode above the 20 cm^-1 imaginary-frequency
cutoff, the resolver displaces the geometry in both directions along the
worst mode, reoptimizes with tighter xTB accuracy and force convergence, and
recomputes a four-point Hessian. The lowest-energy retry without significant
imaginary modes is retained.

The shared optimized geometry is reused when available and generated when
missing. The expensive raw vibrational artifact is cached persistently with
method settings and dependency provenance. Fitted kernels use the ordinary
derived-kernel cache; an expired fitted kernel is reconstructed from the raw
frequencies without repeating the Hessian.

Failures are also cached persistently in the same SQLite artifact namespace.
Negative entries are keyed by canonical structure and a signature containing
the dependency versions and computation-policy version, so fresh worker
processes avoid repeating a known failure while relevant upgrades or policy
changes permit a retry. A renewable per-structure lease in the shared
`data/runtime/locks.sqlite` registry prevents concurrent workers from launching
the same uncached calculation. The xTB lease is renewed every 5 seconds and
expires 20 seconds after its last successful renewal, allowing prompt recovery
after a worker crash.

API calls accept `allow_computation=True` by default. The top-level PFD
directive `ALLOW_COMPUTATION: false` applies the same policy to every component.
Disabling computation blocks only generation on a cache miss; cached xTB/RRHO
artifacts and kernels remain usable.

## Validation basis

The adopted policy was tested on identity-clean canonical empirical curves.
In the difficult 20-compound cohort, with independent normally distributed
`0.3%` relative scatter on every observation:

- One 500 K constant-residual anchor gave about `2.11%` mean full-range MAPE.
- Two points at 500 and 600 K with affine xTB gave about `1.26%`.
- Five points from 500 through 600 K gave about `1.22%`.
- Ten points from 400 through 625 K gave about `1.03%`.
- Twenty points from 300 through 680 K gave about `1.16%` mean and `0.61%`
  median full-range MAPE for affine xTB.

A direct five-parameter Shomate fit to those twenty noisy points achieved about
`0.11%` mean error inside 300–680 K but about `11.7%` over the full range, with
nonpositive extrapolated curves in roughly `14.6%` of realizations. This is why
the Shomate segment remains strictly in range and xTB supplies both tails.
