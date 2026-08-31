# Direct Lower Completion: Dynamic Omega Versus Nannoolal

## Scope

This experiment compares derivative-bearing hard Psat completion from:

```text
Tb, 0.75, 0.50, 0.25, 0.10, and 0.05 bar
```

Predictions are evaluated through `0.001 bar` wherever the reference curve
remains available. Nannoolal is capped at `0.8Tc`.

## Correct Nannoolal Construction

Nannoolal is treated as a slope relation integrated from the most reliable
known pressure endpoint.

The endpoint form is produced by:

```text
estimate_psat(smiles, psat_point=(Th, Ph))
```

This preserves Nannoolal's native structure-derived slope and makes the curve
pass exactly through the hard endpoint.

The only additional physically motivated diagnostic is multiplicative slope
calibration:

```text
scale_h = slope_hard(Th) / slope_Nannoolal(Th)
slope_corrected(T) = scale_h slope_Nannoolal(T)
```

Because the unscaled Nannoolal curve already passes through the hard endpoint,
the calibrated pressure is:

```text
ln(Pcorrected/Ph)
    = scale_h ln(PNannoolal/Ph)
```

No trusted-`Tb` forcing, value smoothing, additive C1 correction, or arbitrary
release function is part of the intended method.

## Handoff Slope Accuracy

Native Nannoolal slope mismatch at the hard endpoint:

| Reference | Median | p95 | Maximum |
|---|---:|---:|---:|
| CoolProp | `1.666%` | `5.403%` | `16.088%` |
| Perry | `1.534%` | `6.342%` | `18.462%` |
| Perry acids | `2.695%` | `9.058%` | `12.008%` |

The native slope is generally close, but its small bias accumulates over
several pressure decades.

## CoolProp

Aggregate point errors by hard endpoint:

| Endpoint | Dynamic omega median / p95 | Native endpoint Nannoolal | Slope-scaled Nannoolal |
|---:|---:|---:|---:|
| `Tb` | `0.697% / 8.160%` | `7.116% / 32.483%` | `2.677% / 32.988%` |
| `0.75 bar` | `0.546% / 10.894%` | `6.839% / 34.684%` | `2.525% / 31.887%` |
| `0.50 bar` | `0.362% / 8.453%` | `5.744% / 33.126%` | `1.906% / 26.302%` |
| `0.25 bar` | `0.247% / 5.938%` | `5.423% / 30.841%` | `1.541% / 20.785%` |
| `0.10 bar` | `0.134% / 3.505%` | `4.202% / 25.158%` | `1.133% / 13.401%` |
| `0.05 bar` | `0.113% / 2.180%` | `4.283% / 22.468%` | `1.097% / 10.385%` |

Dynamic omega is better at every handoff.

## Perry

Aggregate point errors by hard endpoint:

| Endpoint | Dynamic omega median / p95 | Native endpoint Nannoolal | Slope-scaled Nannoolal |
|---:|---:|---:|---:|
| `Tb` | `3.262% / 24.647%` | `5.913% / 35.095%` | `4.220% / 33.265%` |
| `0.75 bar` | `2.576% / 20.025%` | `5.341% / 33.838%` | `3.402% / 28.633%` |
| `0.50 bar` | `1.772% / 15.386%` | `4.440% / 31.193%` | `2.513% / 22.516%` |
| `0.25 bar` | `1.333% / 10.789%` | `4.631% / 29.042%` | `1.958% / 16.769%` |
| `0.10 bar` | `0.807% / 6.316%` | `4.188% / 25.339%` | `1.386% / 10.946%` |
| `0.05 bar` | `0.585% / 4.188%` | `4.017% / 22.331%` | `1.088% / 8.152%` |

Slope calibration materially improves Nannoolal, but dynamic omega remains
better at every handoff.

## Systematic Derivative-Error Function

The required Nannoolal slope correction was measured at every lower pressure.
A pressure-power continuation was fitted:

```text
correction(P)
    = correction(Ph) (P/Ph)^beta
```

Results:

| Reference | Fitted `beta` | Median correction-factor residual | p95 |
|---|---:|---:|---:|
| CoolProp | `-0.00262` | `1.57%` | `7.77%` |
| Perry non-acids | `-0.00076` | `1.47%` | `7.39%` |
| Perry acids | `-0.00978` | `1.60%` | `6.20%` |

For ordinary fluids, `beta` is essentially zero. There is no useful universal
pressure trend beyond freezing the endpoint slope ratio. The remaining drift
is compound-specific and too broad for a global correction.

Applying the fitted ordinary-fluid trend only marginally changes results and
does not make Nannoolal competitive with dynamic omega.

Acids show a stronger systematic trend:

- Native Nannoolal slope increasingly overpredicts the required correction as
  pressure falls.
- The fitted trend improves acid medians near `0.25-0.05 bar`.
- The trend was fitted and evaluated on the same small acid set and is not an
  independent predictive model.
- Its behavior is consistent with changing vapor association, for which the
  explicit monomer/dimer model is more physical.

## Conclusion

For a derivative-bearing hard Psat endpoint:

```text
linear dynamic omega
    > endpoint-integrated Nannoolal
    > raw unanchored structural estimates
```

Multiplicative endpoint slope calibration is physically defensible and
improves Nannoolal substantially, but not enough to overtake dynamic omega.

There is no broadly predictive ordinary-fluid derivative-error function beyond
the endpoint ratio itself. Acid derivative drift is systematic but should be
represented through dimerization rather than an empirical global exponent.

Nannoolal remains valuable when hard derivative or AW prerequisites are absent,
not as the preferred continuation from a strong derivative-bearing hard
segment.

## Reproduction

```bash
PYTHONPATH=.:scripts/psat_experiments \
python scripts/psat_experiments/benchmark_direct_lower_nannoolal.py
```

Detailed rows are written to:

```text
/tmp/direct_lower_nannoolal_benchmark.csv
```
