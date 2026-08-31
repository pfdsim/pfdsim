# PSRK 2005 model validation

This note records validation of the numerical implementation in
`thermodynamics_models/psrk.py` against S. Horstmann et al., *Fluid Phase
Equilibria* **227** (2005) 157-164.

## Scope

The paper contains phase-equilibrium and Henry-coefficient plots, but no
tabulated departure-property examples. Validation is therefore split into:

1. thermodynamic identity checks for the new analytical departure properties;
2. reproduction of selected PSRK curves plotted in the 2005 paper.

The validated kernel is used by the integrated `PSRK` thermodynamic method.
Factory/parser/simulator wiring is covered separately by focused integration
tests; the published-curve results here remain kernel-level checks.

## Departure-property checks

The implemented molar properties are departure enthalpy, entropy, Gibbs
energy, and constant-pressure heat capacity. Enthalpy and entropy use the
analytical temperature derivative of the complete PSRK mixing rule, including
Mathias-Copeman alpha functions and temperature-dependent group interactions.

Focused tests verify:

- the analytical `dD/dT` against a centered numerical derivative;
- `G^R = H^R - T S^R`;
- `G^R/(RT) = sum(x_i ln(phi_i))`;
- `H^R = -R T^2 [d(G^R/RT)/dT]` at constant pressure and composition;
- the ideal-gas zero-pressure limit.

## Published-curve reproduction

Run:

```text
python scripts/validate_psrk2005.py
```

The reference anchors were digitized from the paper's solid PSRK curves, not
from the experimental symbols. Their precision is limited by the embedded
raster figures, so the script reports graphical tolerances rather than
pretending the anchors are exact source data.

| Paper example | Calculated comparison | Result |
|---|---:|---:|
| Fig. 2a, HCN/water bubble pressure at 291.15 K | 4 composition anchors | 0.277% AAD |
| Fig. 2c, oxygen/ozone bubble pressure at 90.25 K | 3 composition anchors | 0.355% AAD |
| Fig. 3b, chlorine/SO2 azeotropic composition | 228.15-323.15 K | 0.000490 maximum mole-fraction difference |
| Fig. 6, methane/water Henry coefficient | 300-580 K | 0.171% AAD |

All four comparisons pass their graphical-resolution tolerances. The methane
Henry calculation also reproduces the reported nonmonotonic temperature
behavior, with a maximum near 380 K.

Reproducing Figure 6 does not imply agreement with newer methane/water Henry
recommendations. The paper states that this interaction was revised against
the authors' measurements and literature data then held in the Dortmund Data
Bank. The close Figure 6 match therefore distinguishes faithful PSRK-2005
behavior from differences caused by regression-data vintage.

These comparisons establish that the JSON parameter transcription and the
PSRK fugacity kernel reproduce several independent published PSRK
examples. They do not replace a future experimental-data validation campaign.
