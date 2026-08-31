# Hard-Segment Dimer Parameter Fit

## Question

Can a retained hard Psat segment and a qualified monomer Hvap relation identify
both the enthalpy and entropy of vapor-phase carboxylic-acid dimerization?

The current production route fixes:

```text
delta H dimer = -60.5 kJ/mol dimer
```

and calibrates `delta S dimer` from the incoming slope at the deep-completion
boundary.

## Method

For each hard-segment point:

```text
H apparent = R T^2 dln(P)/dT
alpha = 1 - H monomer vaporization / H apparent
```

The dimer equilibrium relation then gives:

```text
ln(K) = ln(alpha (1-alpha) / ((1-2alpha)^2 P))
      = delta S / R - delta H / (R T)
```

Therefore a rank-two linear fit of `ln(K)` against `1/T` supplies both
parameters without a nonlinear optimizer.

The benchmark uses all 18 Perry monoacids, four simulated hard lower
endpoints (`1.01325`, `0.75`, `0.50`, and `0.35 bar`), and hard fitting spans
of `0.15`, `0.30`, and `0.50` pressure decades. Predictions begin from the
existing lower-AW `0.25 bar` switch and extend to `0.001 bar` where the Perry
range permits.

## Results

All 72 hard-endpoint cases are formally rank-two for every tested span.
Formal rank is not sufficient evidence of practical identifiability.

For a `0.15`-decade hard span:

| Method | Point median/p95/max | Curve MARD median/p95/max |
|---|---:|---:|
| Fitted `H/S` | `1.679% / 30.221% / 66.321%` | `1.779% / 23.872% / 33.154%` |
| Fixed `H`, boundary-fitted `S` | `5.711% / 14.635% / 25.306%` | `5.710% / 11.264% / 13.610%` |

The free fit improves the median but more than doubles the p95 error.

The result depends strongly on how close the hard segment reaches to the
`0.25 bar` switch:

| Hard lower endpoint | Fitted curve MARD median/p95/max | Fixed curve MARD median/p95/max |
|---:|---:|---:|
| `1.01325 bar` | `2.62% / 32.81% / 33.15%` | `7.13% / 13.61% / 13.61%` |
| `0.75 bar` | `1.78% / 23.42% / 24.70%` | `6.34% / 11.62% / 11.71%` |
| `0.50 bar` | `1.12% / 14.74% / 16.56%` | `5.38% / 9.70% / 9.84%` |
| `0.35 bar` | `0.85% / 9.80% / 11.48%` | `4.45% / 8.93% / 9.23%` |

At `0.35 bar`, the fitted route improves 13 of 18 acids and keeps the worst
curve below `11.6%`. It is promising as a short-extrapolation specialization,
not as a general replacement.

## Identifiability

For the `0.15`-decade fit, fitted enthalpies span approximately
`-63.5` to `-6.4 kJ/mol`, with a median near `-38.0 kJ/mol`. The same acid's
fitted enthalpy changes across hard endpoints by a median of `6.8 kJ/mol`
and a p95 of `12.3 kJ/mol`.

The inferred extent changes by only about `0.006` to `0.008` across the
short hard window. A smooth `0.1%` affine drift in the hard derivative gives
fitted-enthalpy standard deviations around `4–5 kJ/mol` at the median and
`12–15 kJ/mol` at p95. Wider fitting spans improve numerical leverage but
produce worse deep extrapolation because the apparent dimer parameters are
not constant over the broader region.

High `R^2` is therefore not a sufficient acceptance criterion. It did not
identify the cases with bad deep tails.

## Conclusion

Do not replace the fixed-enthalpy route globally.

A two-parameter fit is worth retaining as a possible specialized route when:

- the hard segment reaches approximately `0.35 bar`;
- at least a narrow analytic hard derivative span is available above it;
- Hvap is broad, source-stable, and high quality;
- the fitted route receives explicit extrapolation and derivative-sensitivity
  diagnostics.

Even there, its median improves much more than its tail. A production change
should wait for validation on non-Perry sources or a conservative model
selection rule that predicts the five regressions without using target data.

## Reproduction

```bash
PYTHONPATH=.:scripts/psat_experiments \
python scripts/psat_experiments/benchmark_hard_dimer_parameter_fit.py
```

Detailed rows are written to:

```text
/tmp/hard_dimer_parameter_fit_benchmark.csv
```
