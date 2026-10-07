# 1-Butanol/water joint activity-fit investigation

## Data interpretation

Component 1 is 1-butanol and component 2 is water. Therefore
`gamma1_inf` is 1-butanol infinitely dilute in water and `gamma2_inf` is
water infinitely dilute in 1-butanol.

The imported 1.0133 bar table contains two VLE branches separated by a VLLE
discontinuity. Rows 74 and 75 have nearly equal temperature and vapor
composition but jump from 0.423 to 0.025 liquid butanol mole fraction:

| Quantity | Row 74 | Row 75 |
| --- | ---: | ---: |
| T / K | 365.95 | 365.85 |
| liquid x(butanol) | 0.423 | 0.025 |
| vapor y(butanol) | 0.250 | 0.248 |

They should be represented by one VLLE invariant near 365.9 K and
y(butanol) = 0.249, with the LLE data determining the two liquid endpoints.
Treating every plateau point as homogeneous VLE overconstrains metastable or
overall compositions and creates false phase-stability failures.

## Source and precision assessment

- The IUPAC LLE recommendations combine multiple studies and explicitly reject
  discordant observations. Their printed spreads are not statistical standard
  deviations. They are the strongest source for the binodal, although the
  butanol-rich branch remains difficult for simple activity models.
- Reported UCST values span 397.55–398.50 K and critical butanol mole fractions
  approximately 0.105–0.110. A soft target near 398.0 K and 0.107 is justified.
- The VLE table is a rounded secondary presentation of atmospheric data: mole
  fractions have three decimals and temperatures 0.1 °C precision. The pure
  butanol endpoint is 0.73 K below the fitter's qualified vapor-pressure basis,
  while the pure-water endpoint differs by only 0.02 K. A 1% fugacity residual
  scale is too aggressive for every point in this table.
- The entered butanol-in-water gamma-infinity series is the selected dataset in
  Islam and Kabadi (2011): most values are from Vrbka et al. (2005), while the
  298.15 K value 51.37 is from Islam et al. (2011). A broader literature table
  gives roughly 44.5–58.9 at 298.15 K. These are reputable selected data, but
  not a precision consensus with 1% uncertainty.
- The water-in-butanol series is also exactly the selected Islam and Kabadi
  dataset: 5.06 at 298.15 K is from Islam et al. (2011), and 3.27, 3.12, 3.07,
  and 2.97 at 343.15–372.15 K are from Loblein and Prausnitz (1982). The source
  explicitly selected the latter for consistency with the 25 °C value.

The rounded VLE data imply strong composition dependence near each pure end.
For example, at 384.65 K and x(butanol) = 0.961 they imply
gamma(water) = 4.60, whereas a linear extrapolation of the entered
water-in-butanol infinite-dilution series gives about 2.57. This is not a
thermodynamic contradiction: gamma can rise sharply away from infinite
dilution. It does require more composition flexibility than standard NRTL or
UNIQUAC supplied here.

## Standard-model screen

The maintained `probe_butanol_joint_models.py` script compared all supported
temperature forms for fixed-alpha NRTL, fitted-alpha NRTL, and UNIQUAC. The
best standard compromise was fitted-alpha NRTL with an A+B/T+D T law:

| Metric | Result |
| --- | ---: |
| VLE temperature RMSE | 0.70 K |
| VLE vapor-composition RMSE | 0.018 |
| gamma1 relative RMSE | 9.2% |
| gamma2 relative RMSE | 17.5% |
| water-rich LLE endpoint RMSE | 0.018 |
| butanol-rich LLE endpoint RMSE | 0.126 |
| UCST temperature error | 55.9 K |

No supported NRTL/UNIQUAC form fits all groups well. Fitting alpha removes the
3-bar rich-end maximum, but only by sacrificing the UCST and butanol-rich LLE
branch. Higher temperature order does not resolve a same-temperature
composition-shape limitation.

For an LLE-first standard model, the saved full-form UNIQUAC fit is a useful
exception to the joint-fit conclusion. It gives water-rich and butanol-rich LLE
RMSE values of 0.0034 and 0.0111, respectively, and predicts the UCST within
1.04 K and the critical composition within 0.0010 mole fraction. At 3 bar it
predicts only a stable minimum azeotrope near x(butanol) = 0.167 and 401.71 K;
no rich-end maximum occurs under any supported extrapolation policy.

That fit is not a quantitative joint VLE model. Against the later datasets,
without refitting, it gives a VLE temperature RMSE of 3.00 K, vapor-composition
RMSE of 0.070, butanol-in-water gamma-infinity relative RMSE of 9.7%, and
water-in-butanol gamma-infinity relative RMSE of 29.5%. Use it for LLE-focused
simulation and treat VLE/activity predictions as validation warnings unless
additional high-temperature or 3-bar data are fitted successfully.

## Flexible composition probe

`probe_butanol_redlich_kister.py` fits thermodynamically consistent
Redlich–Kister excess-Gibbs polynomials as a research probe. These are not
runtime models. A fifth-degree composition polynomial (six coefficients at each
temperature) with quadratic temperature dependence gives, after replacing only
the discontinuity pair by one VLLE invariant:

| Metric | Equal group weights |
| --- | ---: |
| VLE temperature RMSE | 0.49 K |
| VLE vapor-composition RMSE | 0.0099 |
| gamma1 relative RMSE | 1.50% |
| gamma2 relative RMSE | 2.91% |
| water-rich LLE endpoint RMSE | 0.0041 |
| butanol-rich LLE endpoint RMSE | 0.180 |
| predicted critical T / K | 397.21 |
| predicted critical x(butanol) | 0.1065 |

The probe eliminates the rich-end 3-bar maximum while closely fitting VLE,
both infinite-dilution directions, and the critical point. Increasing the LLE
weight in separate diagnostics reduces the rich-end error but shifts errors to
the other LLE endpoint; it does not remove the tradeoff. Its remaining large
butanol-rich LLE error shows that merely adding temperature terms is not the
answer. Even substantial composition flexibility leaves a material tradeoff.
This may reflect the limitations of low-order excess-Gibbs models for strongly
associated water/alcohol mixtures, inconsistent uncertainty weighting, or a
remaining source/standard-state mismatch.

## Published NRTL 3-bar reproduction

The currently published NRTL fit, extrapolated above its 397.85 K training
limit, predicts two stable homogeneous azeotropes at 3 bar:

| Type | x(butanol) | T / K |
| --- | ---: | ---: |
| Minimum | 0.09287 | 404.579 |
| Maximum | 0.80375 | 426.818 |

All existing extrapolation policies retain the rich-end maximum, moving it to
x(butanol) approximately 0.87–0.91. The issue originates in the fitted
composition law and endpoint constraints, not only the extrapolation policy.

The earlier LLE-focused NRTL fit with alpha = 0.45131325 behaves differently.
It fits the binodal with approximately 0.0105 tie-line RMS and predicts the UCST
near 399.41 K. At 3 bar, unrestricted extrapolation still produces a stable
rich-end maximum at x(butanol) = 0.8351 and 425.99 K. The `constant_inverse`
continuation merely moves it to x = 0.9471. The `inverse_linear_quadratic` and
`inverse_square_cubic` continuations remove the rich-end maximum, leaving only
the low-composition minimum near x = 0.124 and 403.3 K. This is the most direct
standard-NRTL option when LLE quality and absence of the extrapolated maximum
are the priorities.

A fresh fixed-alpha scan against the recovered LLE and soft UCST data improves
this result. With alpha = 0.451313 and the A+B/T+C h(T) law, the fitted
coefficients are:

| Parameter | Value |
| --- | ---: |
| tau12_c | 1.82083950 |
| tau12_d | -84.60042963 |
| tau12_e | -51.03945692 |
| tau21_c | 5.26856283 |
| tau21_d | -515.02512711 |
| tau21_e | -18.01287251 |

The water-rich and butanol-rich endpoint RMSE values are 0.00354 and 0.02242,
respectively. The predicted critical coordinates differ from the target by
0.96 K and 0.00075 mole fraction. At 3 bar, every supported extrapolation policy
predicts only a stable minimum near x(butanol) = 0.143 and 402.6–402.7 K; no
rich-end maximum appears. Alpha = 0.3 slightly improves the rich-end LLE RMSE
to 0.01778 but worsens the water-rich endpoint and UCST error (5.67 K). Negative
alpha values either fail the activity limit or give poor/absent stable LLE and
are not useful.

When LLE accuracy alone is prioritized, a fine alpha scan places the best
combined endpoint error near alpha = 0.36. With the same A+B/T+C h(T) law:

| Parameter | Value |
| --- | ---: |
| tau12_c | -0.1654130 |
| tau12_d | 132.9999370 |
| tau12_e | -23.5592891 |
| tau21_c | 6.5088084 |
| tau21_d | -811.5372637 |
| tau21_e | -18.1112524 |

Its water-rich and butanol-rich endpoint RMSE values are 0.00621 and 0.01723;
the combined endpoint RMSE is 0.01328. It predicts no 3-bar rich-end maximum
under any extrapolation policy. The cost is a 4.62 K UCST error and a less
accurate atmospheric VLLE invariant (367.90 K and y(butanol) = 0.186).
Therefore alpha near 0.36 is the LLE-only choice, while alpha near 0.45 is the
better LLE/UCST/VLLE compromise.

An unbounded ABCD probe shows that the default coefficient box does limit the
best attainable in-range fit, but it is not the root cause of the NRTL maximum:

| Model | alpha | lean LLE RMSE | rich LLE RMSE | UCST error / K | unrestricted 3-bar maximum |
| --- | ---: | ---: | ---: | ---: | --- |
| NRTL, alpha fitted | 0.44363 | 0.00343 | 0.01015 | 0.71 | x = 0.886, 425.61 K |
| NRTL, alpha fixed | 0.36 | 0.00359 | 0.01227 | 1.39 | x = 0.954, 425.38 K |
| NRTL, alpha fixed | 0.20 | 0.00397 | 0.00816 | 2.05 | x = 0.814, 426.66 K |
| UNIQUAC | structural | 0.00345 | 0.01262 | 1.01 | none |

The fitted-alpha NRTL maximum disappears under `constant_inverse`,
`inverse_linear_quadratic`, and `inverse_square_cubic` continuation, but not
under unrestricted extrapolation. The unbounded solutions are highly
correlated: for fitted-alpha NRTL at 298.15 K, directional terms of roughly
422, -207, and -214 cancel to tau12 = 1.38. They should therefore be treated as
research fits or regularized/reparameterized before publication. UNIQUAC is the
safer unbounded ABCD result because it remains maximum-free even under
unrestricted extrapolation.

The unbounded ABCD NRTL and UNIQUAC fits agree more closely with each other than
with the held-out atmospheric VLE data. Across the 25 tabulated compositions,
their pairwise bubble-temperature RMSE is 0.83 K (maximum difference 1.62 K)
and their pairwise vapor-composition RMSE is 0.0135 (maximum 0.0197). NRTL gives
2.31 K / 0.0626 RMSE against measured T/y; UNIQUAC gives 2.97 K / 0.0696.
Their predicted 1.0133-bar invariants have nearly identical liquid endpoints
(approximately 0.022 and 0.365 butanol), but differ in T/y: NRTL predicts
366.88 K / 0.215 and UNIQUAC 367.48 K / 0.198, versus the table discontinuity
near 365.9 K / 0.249. Thus LLE training determines a similar liquid free-energy
envelope in both models but does not determine the vapor-facing activities well
enough for quantitative VLE.

## Recommended fitting basis

1. Preserve the IUPAC LLE and a soft critical target.
2. Convert the branch discontinuity to one VLLE observation; do not fit the
   interior/duplicate plateau coordinates as homogeneous liquids.
3. Retain both infinite-dilution directions, with source-specific uncertainty
   rather than the same 1% scale for every value.
4. Treat the rounded atmospheric VLE table with realistic composition and
   temperature uncertainty.
5. Judge 3-bar behavior as extrapolation validation. Do not publish a law that
   produces a rich-end maximum unless independent high-pressure data support it.
6. If a standard activity model remains required, report its rich-branch and
   UCST compromise explicitly. For genuinely accurate joint behavior, evaluate
   an association-capable model or a carefully regularized higher-composition
   excess-Gibbs form with held-out validation.

## References consulted

- IUPAC Solubility Data Series 15, *Alcohols with Water*:
  https://iupac.github.io/SolubilityDataSeries/volumes/SDS-15.pdf
- Khazaeli et al., review and literature table for butanol-in-water
  infinite-dilution activity coefficients:
  https://jffhmt.avestia.com/2019/002.html
- High-temperature water/alcohol equilibrium review comparing the atmospheric
  water/1-butanol measurements with Boublik and other literature datasets:
  https://citeseerx.ist.psu.edu/document?doi=a120312ba5fd545e357e412914100699ed050a6e&repid=rep1&type=pdf
- Published discussion of water/1-butanol VLLE regression difficulty:
  https://skoge.folk.ntnu.no/prost/proceedings/distillation02/dokument/6-2.pdf
- Islam and Kabadi (2011), selected LLE/VLE/gamma-infinity datasets and joint
  NRTL regression: https://www.czasopisma.pan.pl/Content/84591?format_id=1
