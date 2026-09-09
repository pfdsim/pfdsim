# Ideal-gas Cp fitting matrix

> **Superseded (2026-09-09).** This atom-prior sparse-data policy is retained
> as historical design context. Runtime selection is now governed by
> [Sparse online ideal-gas Cp policy](ideal_gas_cp_sparse_online_policy.md),
> which uses GFN2-xTB RRHO as the curve-shape prior and was validated against
> sparse, narrow, noisy observation windows.

The atom-increment Shomate model is the default prior for sparse, noisy, or
poorly distributed ideal-gas heat-capacity data. Point count alone must not
select a fit: temperature coverage, independently supportable noise, and the
requested prediction interval matter as well.

| Data | Broad coverage, demonstrated low noise | Broad coverage, high or unknown noise | Narrow coverage, demonstrated low noise | Narrow coverage, high or unknown noise |
|---|---|---|---|---|
| 10+ points | Atom-prior regularized Shomate in the source range; boundary-scaled atom continuation outside it | Scaled atom unless robust validation demonstrates that a regularized Shomate correction improves prediction | Robust Shomate in the source range; boundary-scaled atom continuation outside it | Scaled atom |
| 3–9 points | Atom-prior regularized Shomate in the source range, with boundary-scaled atom continuation outside it, **only when low noise can be demonstrated** | Scaled atom | Atom-prior regularized Shomate in the source range, with boundary-scaled atom continuation outside it, **only when low noise can be demonstrated** | Scaled atom |
| 1–2 points | Scaled atom | Scaled atom | Scaled atom | Scaled atom |

For 3–5 points, residuals from a flexible fit do not provide credible evidence
of low noise. These cases should normally be classified as unknown noise and
use the scaled atom model. A low-noise classification requires independent
support such as reported experimental uncertainty, replicates, or consistent
independent sources. Six to nine points permit some predictive scatter
diagnostics, but do not automatically establish low noise.

## Model forms

### Scaled atom

Fit one positive multiplicative factor using all accepted observations:

\[
C_p(T) = s\,C_{p,\mathrm{atom}}(T)
\]

For a single observation at \(T_a\), this reduces to:

\[
s = \frac{C_{p,\mathrm{observed}}(T_a)}
         {C_{p,\mathrm{atom}}(T_a)}
\]

The scaled atom form is the conservative choice when shape information cannot
be distinguished reliably from noise. It preserves positivity and the
validated temperature dependence of the atom model.

### Atom-prior regularized Shomate

Fit a five-term Shomate correction while shrinking weakly identified
coefficient directions toward the atom-model coefficients. The regularization
strength must be selected without evaluating it on the compounds or points
used to report fit quality. Reject a fit that is nonpositive or numerically
unstable anywhere in its intended source-supported interval.

Point count does not establish identifiability. Three or four observations are
underdetermined relative to five Shomate coefficients, five are nominally
exactly determined, and six or more are overdetermined. The atom prior makes
the optimization well-posed, but it does not make clustered observations
informative about distant curvature.

### Boundary-scaled atom continuation

Outside a data-supported interval, continue with the atom model scaled to the
nearest **fitted** boundary value:

\[
C_{p,\mathrm{ext}}(T) = C_{p,\mathrm{atom}}(T)
\frac{C_{p,\mathrm{fit}}(T_b)}
     {C_{p,\mathrm{atom}}(T_b)}
\]

Here \(T_b\) is the nearest source-range boundary. The fitted, smoothed
boundary value is used instead of a literal noisy endpoint observation. Low
and high continuations may therefore have different scale factors.

Do not extrapolate a data-only Shomate fit outside its source-supported range.
Do not extrapolate a fitted residual slope unless a future validation study
shows a material and robust benefit over constant residual-ratio continuation.

## Temperature coverage

Coverage is measured against the temperature interval over which predictions
are requested, not against a fixed universal interval:

\[
f_{\mathrm{span}} =
\frac{T_{\max,\mathrm{obs}}-T_{\min,\mathrm{obs}}}
     {T_{\max,\mathrm{target}}-T_{\min,\mathrm{target}}}
\]

Also track the uncovered low and high tails separately:

\[
f_{\mathrm{low}} =
\frac{\max(0,T_{\min,\mathrm{obs}}-T_{\min,\mathrm{target}})}
     {T_{\max,\mathrm{target}}-T_{\min,\mathrm{target}}}
\]

\[
f_{\mathrm{high}} =
\frac{\max(0,T_{\max,\mathrm{target}}-T_{\max,\mathrm{obs}})}
     {T_{\max,\mathrm{target}}-T_{\min,\mathrm{target}}}
\]

A centered interval and an endpoint interval with the same span present very
different extrapolation risks. No production cutoff for “broad” should be
chosen until intermediate coverage fractions, including 40%, 60%, and 80%,
have been validated. The existing probes establish only that full-range data
are broad and a 20% observation window is narrow.

## Noise classification

Do not infer low noise from a small in-sample residual. A flexible Shomate fit
can absorb noise, especially when observations are clustered.

When enough points exist, use reported source uncertainty together with robust
leave-one-out/PRESS relative scatter, leverage, conditioning, and comparison
against the one-factor scaled-atom model. The following bands are provisional
interpretive guidance, not pruning targets:

| Robust predictive scatter | Interpretation |
|---:|---|
| at most 1% | Low noise |
| 1–2% | Moderate noise |
| 2–5% | Noisy |
| above 5% | High noise |

With one or two points, noise is unknowable from the data. With three to five
points it is normally too weakly determined to justify a low-noise label
without external evidence.

## Outliers

Prune only isolated observations identified by a robust, source-aware
criterion. Preserve every excluded point and its reason. Never continue
deleting observations merely to drive estimated scatter below a chosen
percentage; disagreement may represent real curvature, source inconsistency,
or underestimated experimental uncertainty.

## Supported range and evaluation domain

Always preserve two distinct concepts:

- **Source-supported range:** the interval covered by the observations or
  source correlation.
- **Evaluation domain:** the larger interval over which an atom-informed hybrid
  can return a numerical value.

For example, data from 300–800 K remain source-supported only from 300–800 K,
even if atom continuation permits evaluation from 273.15–1500 K. Inside the
source range, report the data-fit provenance and quality. Outside it, report
estimated/extrapolated provenance and continuation quality.

Do not force every model to 273.15–1500 K. Extend only where the validated atom
model connects meaningfully to the source-supported interval. Never bridge a
temperature gap silently.

## Executable representation

The logical hybrid may remain piecewise analytic or be approximated by one
Chebyshev kernel. A global Chebyshev refit is acceptable only if it retains the
original source-supported range in metadata and preserves the in-range
correlation substantially more tightly than the estimated continuation. Record
in-range and continuation errors separately. If a global refit cannot preserve
the supported correlation adequately, retain the exact composite kernel.

Regardless of representation, validate positivity, numerical conditioning,
and analytic enthalpy and entropy integrals across every segment of the
evaluation domain.
