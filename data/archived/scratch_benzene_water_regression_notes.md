# Benzene/Water Interaction Regression Scratch Notes

These notes are exploratory. They record candidate interaction parameters fitted
to PFDSim's benzene/water mutual-solubility table plus the 1 atm
heteroazeotrope vapor composition.

Source for this dataset and the follow-on interaction-regression work:
J. Chem. Eng. Data 2003, 48, 750-752.

## Data Interpretation

- Component order in fitted parameters below: `water -> benzene`.
- Mutual-solubility table:
  - `xw_brich` = mole fraction water in benzene-rich liquid.
  - `xb_wrich` = mole fraction benzene in water-rich liquid.
  - Pressure column treated as total vapor pressure over the two-liquid split.
- Heteroazeotrope target:
  - `P = 1.01325 bar`
  - `T = 69.3 C`
  - `vapor = 91.1 wt% benzene`
  - Equivalent vapor mole fraction benzene: `0.7024485833`
- The tabulated solubility near `342.45 K` was treated as more reliable than
  the stated heteroazeotrope liquid compositions.

## Recommended NRTL Candidate

Runtime-supported form:

```text
tau_ij = tau_ij_c + tau_ij_d/T + tau_ij_e*((Tref - T)/T + ln(T/Tref))
Tref = 298.15 K
alpha = 0.2
```

Parameters as `water -> benzene`:

```text
tau12_c =  5.42158455
tau12_d =  170.495590
tau12_e = -9.18065486

tau21_c = -5.18998243
tau21_d =  2745.777555
tau21_e = -4.35588624

alpha12 = 0.2
```

If stored as `benzene -> water`, swap 12 and 21.

Composition residuals from runtime LLE splits, reported as
`ln(predicted/table)`:

```text
mean abs log residual, x_water in benzene-rich:  0.033764
mean abs log residual, x_benzene in water-rich:  0.021793
combined mean abs log residual:                  0.027779
max abs log residual:                            0.069704
```

Pressure residuals at the tabulated liquid compositions, using the mean of the
two phase vapor-pressure closures:

```text
mean abs relative pressure residual: 0.015333
max abs relative pressure residual:  0.046163
```

Heteroazeotrope validation through runtime VLLE helper:

```text
T = 69.3535 C
y_benzene = 0.701123281 mole fraction
vapor benzene = 91.0485 wt%
```

Pointwise runtime composition residuals:

| T K | log xw_brich | log xb_wrich | pred xw_brich | table xw_brich | pred xb_wrich | table xb_wrich |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 279.15 | -0.0140 | -0.0223 | 0.00165666 | 0.001680 | 0.000397035 | 0.000406 |
| 288.15 | -0.0112 | -0.0108 | 0.00225456 | 0.002280 | 0.000403616 | 0.000408 |
| 298.15 | -0.0236 | +0.0309 | 0.00312524 | 0.003200 | 0.000418759 | 0.000406 |
| 313.15 | +0.0082 | +0.0366 | 0.00496074 | 0.004920 | 0.000457443 | 0.000441 |
| 333.15 | -0.0255 | +0.0231 | 0.00878341 | 0.009010 | 0.000543411 | 0.000531 |
| 343.15 | +0.0171 | -0.0089 | 0.0114945 | 0.011300 | 0.000604607 | 0.000610 |
| 353.15 | -0.0527 | +0.0119 | 0.0148942 | 0.015700 | 0.000681058 | 0.000673 |
| 363.15 | -0.0646 | +0.0222 | 0.0191236 | 0.020400 | 0.000776011 | 0.000759 |
| 373.15 | -0.0581 | +0.0165 | 0.0243447 | 0.025800 | 0.000893651 | 0.000879 |
| 398.15 | -0.0697 | +0.0046 | 0.0429961 | 0.046100 | 0.00132612 | 0.001320 |
| 413.15 | -0.0633 | -0.0375 | 0.0591368 | 0.063000 | 0.00172415 | 0.001790 |
| 433.15 | +0.0120 | -0.0381 | 0.0881548 | 0.087100 | 0.00251238 | 0.002610 |
| 453.15 | -0.0189 | -0.0198 | 0.127562 | 0.130000 | 0.00376466 | 0.003840 |

Pointwise direct pressure residuals:

| T K | relative pressure residual |
| ---: | ---: |
| 279.15 | +1.281% |
| 288.15 | +0.497% |
| 298.15 | -2.420% |
| 313.15 | -2.811% |
| 333.15 | +1.762% |
| 343.15 | -0.197% |
| 353.15 | +0.919% |
| 363.15 | -0.425% |
| 373.15 | -0.675% |
| 398.15 | -0.641% |
| 413.15 | -0.723% |
| 433.15 | -2.965% |
| 453.15 | -4.616% |

## Recommended UNIQUAC Candidate

Runtime-supported form:

```text
ln(tau_ij) = tau_ij_a + tau_ij_b/T + tau_ij_c*T
```

Parameters as `water -> benzene`:

```text
tau12_a = -4.26152099
tau12_b =  410.757318
tau12_c =  0.00557446

tau21_a =  2.89513893
tau21_b = -1851.910186
tau21_c =  0.00151181
```

If stored as `benzene -> water`, swap 12 and 21.

Composition residuals from runtime LLE splits, reported as
`ln(predicted/table)`:

```text
mean abs log residual, x_water in benzene-rich:  0.041402
mean abs log residual, x_benzene in water-rich:  0.026324
combined mean abs log residual:                  0.033863
max abs log residual:                            0.090661
```

Pressure residuals at the tabulated liquid compositions, using the mean of the
two phase vapor-pressure closures:

```text
mean abs relative pressure residual: 0.015977
max abs relative pressure residual:  0.055082
```

Heteroazeotrope validation through runtime VLLE helper:

```text
T = 69.3534 C
y_benzene = 0.701124846 mole fraction
vapor benzene = 91.0486 wt%
```

Pointwise runtime composition residuals:

| T K | log xw_brich | log xb_wrich | pred xw_brich | table xw_brich | pred xb_wrich | table xb_wrich |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 279.15 | -0.0342 | -0.0396 | 0.00162355 | 0.001680 | 0.000390244 | 0.000406 |
| 288.15 | -0.0217 | -0.0130 | 0.00223102 | 0.002280 | 0.000402723 | 0.000408 |
| 298.15 | -0.0263 | +0.0389 | 0.00311693 | 0.003200 | 0.000422113 | 0.000406 |
| 313.15 | +0.0123 | +0.0497 | 0.00498077 | 0.004920 | 0.000463493 | 0.000441 |
| 333.15 | -0.0202 | +0.0302 | 0.00882939 | 0.009010 | 0.000547304 | 0.000531 |
| 343.15 | +0.0203 | -0.0079 | 0.0115313 | 0.011300 | 0.000605182 | 0.000610 |
| 353.15 | -0.0528 | +0.0058 | 0.0148924 | 0.015700 | 0.000676910 | 0.000673 |
| 363.15 | -0.0690 | +0.0090 | 0.0190394 | 0.020400 | 0.000765855 | 0.000759 |
| 373.15 | -0.0673 | -0.0029 | 0.0241199 | 0.025800 | 0.000876455 | 0.000879 |
| 398.15 | -0.0907 | -0.0219 | 0.0421044 | 0.046100 | 0.00129137 | 0.001320 |
| 413.15 | -0.0883 | -0.0589 | 0.0576783 | 0.063000 | 0.00168758 | 0.001790 |
| 433.15 | -0.0104 | -0.0370 | 0.0861989 | 0.087100 | 0.00251530 | 0.002610 |
| 453.15 | -0.0247 | +0.0273 | 0.126822 | 0.130000 | 0.00394621 | 0.003840 |

Pointwise direct pressure residuals:

| T K | relative pressure residual |
| ---: | ---: |
| 279.15 | +2.173% |
| 288.15 | +0.677% |
| 298.15 | -2.698% |
| 313.15 | -3.316% |
| 333.15 | +1.441% |
| 343.15 | -0.267% |
| 353.15 | +1.122% |
| 363.15 | +0.059% |
| 373.15 | +0.070% |
| 398.15 | +0.486% |
| 413.15 | +0.320% |
| 433.15 | -2.633% |
| 453.15 | -5.508% |

## Notes On Model Choice

- The three-term NRTL candidate is the best of the two candidates by runtime
  composition residuals and direct pressure residuals.
- The three-term UNIQUAC candidate is close and still very usable.
- The improvement from the third temperature term appears curvature-related,
  not just a single-point fit:
  - Two-term forms had systematic endpoint errors.
  - Three-term forms reduce residuals across the table.
- Treat both fits as range-limited to the source data: `279.15-453.15 K`.
- Coefficients should not be interpreted as uniquely physical; the predicted
  LLE/VLLE curve is the meaningful object.

# Toluene/Water Interaction Regression Scratch Notes

Source for this dataset and the follow-on interaction-regression work:
J. Chem. Eng. Data 2003, 48, 750-752.

These notes use the same interpretation as the benzene/water fit:

- Component order in fitted parameters below: `water -> toluene`.
- `xw_orich` = mole fraction water in toluene-rich liquid.
- `xo_wrich` = mole fraction toluene in water-rich liquid.
- Pressure column treated as total vapor pressure over the two-liquid split.
- The reported 1 atm heteroazeotrope was used as a soft validation target:
  - `T = 84.1 C`
  - `vapor = 79.8 wt% toluene`
  - Equivalent vapor mole fraction toluene: `0.4357930563`
- The heteroazeotrope temperature appears source-sensitive; a predicted value
  near `84.5 C` is acceptable for this scratch fit.

## Recommended NRTL Candidate

Runtime-supported form:

```text
tau_ij = tau_ij_c + tau_ij_d/T + tau_ij_e*((Tref - T)/T + ln(T/Tref))
Tref = 298.15 K
alpha = 0.2
```

Parameters as `water -> toluene`:

```text
tau12_c =  6.44255915
tau12_d =  230.691712
tau12_e = -13.97875967

tau21_c = -5.59819164
tau21_d =  2931.226806
tau21_e = -3.33064985

alpha12 = 0.2
```

If stored as `toluene -> water`, swap 12 and 21.

Composition residuals from runtime LLE splits, reported as
`ln(predicted/table)`:

```text
mean abs log residual, x_water in toluene-rich:  0.033639
mean abs log residual, x_toluene in water-rich:  0.042579
combined mean abs log residual:                  0.038109
max abs log residual:                            0.072061
```

Pressure residuals at the tabulated liquid compositions, using the mean of the
two phase vapor-pressure closures:

```text
mean abs relative pressure residual: 0.019286
max abs relative pressure residual:  0.047573
```

Heteroazeotrope validation through runtime VLLE helper:

```text
T = 84.5509 C
y_toluene = 0.439664267 mole fraction
vapor toluene = 80.0524 wt%
```

## Recommended UNIQUAC Candidate

Runtime-supported form:

```text
ln(tau_ij) = tau_ij_a + tau_ij_b/T + tau_ij_c*T
```

Parameters as `water -> toluene`:

```text
tau12_a = -5.02820679
tau12_b =  546.919292
tau12_c =  0.00691596

tau21_a =  4.86379009
tau21_b = -2243.653045
tau21_c = -0.00141886
```

If stored as `toluene -> water`, swap 12 and 21.

Composition residuals from runtime LLE splits, reported as
`ln(predicted/table)`:

```text
mean abs log residual, x_water in toluene-rich:  0.046140
mean abs log residual, x_toluene in water-rich:  0.054189
combined mean abs log residual:                  0.050165
max abs log residual:                            0.095098
```

Pressure residuals at the tabulated liquid compositions, using the mean of the
two phase vapor-pressure closures:

```text
mean abs relative pressure residual: 0.022247
max abs relative pressure residual:  0.060135
```

Heteroazeotrope validation through runtime VLLE helper:

```text
T = 84.5526 C
y_toluene = 0.439626951 mole fraction
vapor toluene = 80.0499 wt%
```

## Ternary Ethanol/Water/Toluene Note

Source for this ternary note: Fluid Phase Equilibria 266 (2008) 8-13.

That paper reports the ethanol/water/toluene azeotrope as homogeneous, while
common model compilations and standard NRTL/UNIQUAC/UNIFAC predictions can
incorrectly produce a heterogeneous azeotrope. A quick current-NRTL probe showed
the same pathology: the model predicts an oversized LLE region and a nearby
heterogeneous azeotrope instead of the reported homogeneous azeotrope. Do not
use this ternary azeotrope as a simple binary-pair regression target without
also constraining ternary LLE behavior.

# p-Xylene/Water Interaction Regression Scratch Notes

Source for this dataset and the follow-on interaction-regression work:
J. Chem. Eng. Data 2003, 48, 750-752.

These notes use only the mutual-solubility and pressure table; no trusted
p-xylene/water heteroazeotrope point was used in the regression.

- Component order in fitted parameters below: `water -> p-xylene`.
- `xw_orich` = mole fraction water in p-xylene-rich liquid.
- `xo_wrich` = mole fraction p-xylene in water-rich liquid.
- Pressure column treated as total vapor pressure over the two-liquid split.

## Recommended NRTL Candidate

Runtime-supported form:

```text
tau_ij = tau_ij_c + tau_ij_d/T + tau_ij_e*((Tref - T)/T + ln(T/Tref))
Tref = 298.15 K
alpha = 0.2
```

Parameters as `water -> p-xylene`:

```text
tau12_c =  6.31561607
tau12_d =  656.278436
tau12_e = -13.72537271

tau21_c = -4.15861986
tau21_d =  2511.056983
tau21_e = -8.55762685

alpha12 = 0.2
```

Composition residuals from runtime LLE splits, reported as
`ln(predicted/table)`:

```text
mean abs log residual, x_water in p-xylene-rich:  0.029182
mean abs log residual, x_p-xylene in water-rich:  0.039200
combined mean abs log residual:                   0.034191
max abs log residual:                             0.072752
```

Pressure residuals at the tabulated liquid compositions:

```text
mean abs relative pressure residual: 0.019416
max abs relative pressure residual:  0.038523
```

Leave-one-out direct validation:

```text
scaled RMS = 1.191
```

Predicted 1 atm heteroazeotrope:

```text
T = 92.475 C
y_p-xylene = 0.240460 mole fraction
vapor p-xylene = 65.10 wt%
```

## Recommended UNIQUAC Candidate

Runtime-supported form:

```text
ln(tau_ij) = tau_ij_a + tau_ij_b/T + tau_ij_c*T
```

Parameters as `water -> p-xylene`:

```text
tau12_a = -3.49453499
tau12_b =  263.479577
tau12_c =  0.00502872

tau21_a =  0.11693990
tau21_b = -1355.069752
tau21_c =  0.00443349
```

Composition residuals from runtime LLE splits, reported as
`ln(predicted/table)`:

```text
mean abs log residual, x_water in p-xylene-rich:  0.032031
mean abs log residual, x_p-xylene in water-rich:  0.032909
combined mean abs log residual:                   0.032470
max abs log residual:                             0.071220
```

Pressure residuals at the tabulated liquid compositions:

```text
mean abs relative pressure residual: 0.018362
max abs relative pressure residual:  0.045270
```

Leave-one-out direct validation:

```text
scaled RMS = 1.186
```

Predicted 1 atm heteroazeotrope:

```text
T = 92.477 C
y_p-xylene = 0.240403 mole fraction
vapor p-xylene = 65.10 wt%
```

## Rough Heteroazeotrope Sanity Check

Comparison data PFDSim had on hand:

```text
m-xylene/water:      92.0 C, 64.2 wt% xylene
old p-xylene/water:  92 C,   60 wt% p-xylene, one significant figure
```

The fitted p-xylene/water prediction is consistent with this rough range,
especially given the low precision of the old p-xylene source. The current
supplemental NRTL matrix row was much hotter/richer (`97.47 C`, `69.92 wt%`
p-xylene), so the fitted table-based rows look substantially more plausible.
