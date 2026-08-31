# Perry Fixed-Enthalpy Dimer Entropy Inversion

## Question

With vapor dimerization enthalpy fixed at the current generic value,

```text
delta H dimer = -60.5 kJ/mol dimer
```

what dimerization entropy is required to make Perry's vapor-pressure and
heat-of-vaporization correlations satisfy differential Clapeyron when the
physical vapor/liquid volume correction is supplied by Peng-Robinson?

The practical target was a possible molecular-size correction to the previous
generic fallback:

```text
delta S dimer = -140 J/mol/K
```

Following this benchmark, the production generic fallback was changed to the
rounded compromise `-144 J/mol/K`.

## Method

The benchmark uses all `18` Perry monocarboxylic acids having overlapping
Table 2-8 Psat and Table 2-69 Hvap correlations. Each common temperature range
is sampled at `201` evenly spaced temperatures.

At every sample, Peng-Robinson supplies the saturated-state compressibility
difference using the Perry `Tc`, `Pc`, and acentric factor:

```text
H apparent = R T^2 deltaZ_PR dln(Psat)/dT
alpha required = 1 - H Perry / H apparent
```

For a physical inferred dimer extent, `0 < alpha < 0.5`, the fixed-enthalpy
dimer relation can be inverted directly:

```text
K = alpha (1-alpha) / ((1-2alpha)^2 P)
delta S = R ln(K) + delta H / T
```

This follows the existing lower-Psat dimer benchmark's convention: Perry Hvap
is treated as the nominal phase-change enthalpy represented by its published
curve, while fixed dimerization enthalpy controls the temperature dependence
of `K`. If Table 2-69 were instead interpreted as a strictly monomer-only
enthalpy, an explicit association-enthalpy term would enter the Clapeyron
balance and this inversion would need to be reformulated.

The inversion is round-trip checked by reconstructing `alpha` from the
inferred entropy. Samples with `alpha <= 0` are retained as diagnostics: no
positive association strength can reconcile the two Perry slopes there.

## Coverage

```text
Perry monoacids                         18
temperature samples                  3618
samples with a valid PR delta Z       3577
samples with 0 < alpha < 0.5          2826
```

Across the `14` acids whose CAS-keyed parameter lookup returns the generic
fallback:

```text
PR-valid samples                      2781
physical inferred samples             2088
no-positive-association samples         668
alpha >= 0.5 samples                     25
```

Thus about one quarter of the fallback-acid samples cannot be represented by
positive dimerization under the paired Perry correlations. This is an
important source-consistency warning, not evidence for an extreme entropy.

## Normal Saturated Homologous Series

The cleanest size test is the normal saturated C4-C10 series. Formic acid is
deliberately excluded from this regression because its association behavior is
atypical; C2 and C3 are also excluded because they have curated database
parameters. Regressing one inferred entropy per acid against carbon count gives:

| Basis | Delta S at C4 [J/mol/K] | Change per added carbon [J/mol/K] | 95% slope interval | R2 | RMSE [J/mol/K] |
|---|---:|---:|---:|---:|---:|
| Whole-overlap median | -140.69 | -1.469 | [-2.982, +0.043] | 0.555 | 2.63 |
| At 0.01 bar | -143.92 | -0.962 | [-2.468, +0.544] | 0.351 | 2.62 |
| At 0.1 bar | -143.79 | -0.583 | [-1.831, +0.666] | 0.224 | 2.17 |
| At 1.01325 bar | -141.67 | -0.936 | [-2.320, +0.449] | 0.376 | 2.41 |

Matched-pressure comparisons suggest a correction of roughly `-0.6` to
`-1.0 J/mol/K` per added carbon, but every 95% interval includes zero. The
whole-overlap result looks stronger partly because each acid has a different
temperature and pressure range.

At `0.1 bar`, the direct provisional relation would be:

```text
delta S(C) ~= -143.8 - 0.58 (C - 4) J/mol/K, C4-C10
```

The data do not distinguish that relation reliably from a simple value in the
`-143` to `-145 J/mol/K` range. A small constant shift from the current generic
fallback is better supported than the carbon-number slope.

Directly rescoring constant entropies over all valid C4-C10 points gives:

| Constant Delta S [J/mol/K] | Point median/p95 absolute slope error | Curve MARD median/p95/max |
|---:|---:|---:|
| -140 | 3.06% / 9.78% | 2.98% / 6.20% / 6.98% |
| -143 | 1.91% / 8.28% | 2.27% / 5.26% / 5.93% |
| -144 | 1.74% / 9.00% | 1.84% / 5.49% / 5.92% |
| -145 | 1.79% / 9.98% | 1.62% / 5.84% / 6.10% |

The continuous optima depend on the objective: about `-143.1` for point p95,
`-143.8` for point mean/median, and `-143.5` for the worst curve MARD. Thus
`-144 J/mol/K` is the best rounded compromise. `-145 J/mol/K` is close and
slightly better for median curve MARD, but gives weaker tail errors.

For these seven normal acids, the previous `-140 J/mol/K` fallback produces
per-acid median absolute Clapeyron-slope errors with a median of `2.74%`; the
largest is `7.31%`. Their per-acid p95 errors have a median of `5.87%` and a
maximum of `12.12%`.

## All Generic-Fallback Acids

Across all `14` fallback acids, the distribution of each acid's median
required entropy is:

```text
median / p05 / p95 = -147.56 / -187.31 / -141.02 J/mol/K
within 10 J/mol/K of the selected -144 = 9 / 14 acids
```

The broad lower tail is structural rather than a smooth size trend:

| Acid | Median required Delta S [J/mol/K] | Generic median/p95 absolute slope error |
|---|---:|---:|
| 2-Methylbutanoic acid | -192.24 | 23.34% / 34.71% |
| 2-Ethylbutanoic acid | -184.65 | 13.93% / 19.75% |
| 2-Ethylhexanoic acid | -172.63 | 5.50% / 8.96% |
| 2-Methyloctanoic acid | -172.18 | 3.52% / 11.04% |
| Benzoic acid | -160.98 | 2.79% / 4.09% |

Several of these acids also have more than half their samples demanding
`alpha <= 0`. Their extreme inferred entropies therefore should not be turned
into a branch or aromatic correction without independent data. They more
likely expose incompatibility among Perry Psat, Perry Hvap, PR delta Z, and
the one-reaction dimer model.

Over all fallback-acid PR-valid points, the selected `-144 J/mol/K` generic
entropy gives absolute slope errors of `3.37%` median and `23.94%` p95.

## Database Controls

The database acids provide a useful partial check, not size-fit evidence. With
the common fixed enthalpy, formic acid requires a median entropy of
`-155.04 J/mol/K`, nearly identical to its stored `-155.0 J/mol/K`; because
formic is atypical, that agreement is not used to calibrate the fallback. When
each database acid's stored enthalpy is substituted after the inversion, the
median required entropies are:

| Acid | Required Delta S with stored Delta H | Stored Delta S [J/mol/K] |
|---|---:|---:|
| Formic acid | -154.44 | -155.0 |
| Acetic acid | -152.18 | -155.5 |
| Propionic acid | -155.09 | -152.4 |
| Acrylic acid | -156.34 | -161.4 |

This is encouraging at the several-J/mol/K level, while still showing the
independent Perry curves are not exact thermodynamic identities.

## Conclusion

The Perry test supports the change from `-140` to `-144 J/mol/K` more clearly
than it supports a size-dependent rule.

A provisional normal-acid size correction could use about `-0.8 J/mol/K` per
carbon beyond C4, but the slope is not statistically resolved with only seven
homologues and independent Perry correlations. It should not yet be promoted
to the production fallback.

Branched and aromatic acids must remain separate. Their Perry inversions do
not form a credible carbon-number trend and frequently have no physical
positive-association solution.

## Reproduction

```bash
PYTHONPATH=.:scripts/psat_experiments \
python scripts/psat_experiments/benchmark_perry_dimer_entropy.py
```

Detailed point and per-acid summary tables are written to:

```text
/tmp/perry_dimer_entropy_required_points.csv
/tmp/perry_dimer_entropy_required_summary.csv
```
