# PSRK thermodynamic method

Select the integrated published-parameter PSRK model with:

```text
THERMO_METHOD: PSRK
```

`PREDICTIVE-SRK` is accepted as an alias. The method uses the SRK equation of
state, the published PSRK first-order modified Huron-Vidal mixing rule
(`q1=-0.64663`), PSRK-UNIFAC groups and directional temperature-dependent
interactions, and the 2005 pure-component parameter supplement.

## Data policy

Components are identified by CAS number and require a complete subgroup
assignment in `data/psrk/pure_components.json`. Required group interactions
must exist in `data/psrk/group_parameters.json`; missing interactions are an
initialization error rather than silently becoming zero.

For a component with unstarred published `Tc` and `Pc`, PSRK uses the complete
published `Tc`/`Pc`/omega/Mathias-Copeman bundle. The model does not mix those
fitted alpha coefficients with unrelated modern critical constants.

If either published critical is starred, the integration layer supplies
normally resolved `Tc`, `Pc`, and omega and the component uses generalized
omega-based Soave alpha. This is explicitly warned as a hybrid fallback. A PFD
component override similarly replaces the bundle; explicit `mc_c1`, `mc_c2`,
and `mc_c3` values may provide Mathias-Copeman alpha for that override.

Separately from PSRK model construction, the generic critical-property
resolver offers independently unstarred PSRK-2005 `Tc` and `Pc` values at
quality `0.95`, behind CoolProp, ACS, Perry, and Smith. It does not expose the
PSRK table's starred estimates, `Vc`, or omega as generic sources.
Effective criticals scored at least `0.95` precede this fallback; that quality
band represents experimental criticals which coincide with the effective
parameterization. Lower-scored effective criticals remain after PSRK.

Ordinary `SRK-MC` also prefers the PSRK-2005 Mathias-Copeman coefficients over
the older ChemSep table when its selected `Tc` and `Pc` each agree with the
PSRK bundle within 1%. Explicit PFD coefficients remain highest priority.

## Implemented properties

The integrated method provides:

- phi-phi fugacity coefficients and composition-dependent K-values;
- ordinary TP flash, bubble-point, and dew-point paths used by EOS models;
- compressibility and un-translated EOS vapor molar volume;
- analytical departure enthalpy and entropy;
- departure Gibbs energy and constant-pressure departure heat capacity;
- normal process-simulator enthalpy, entropy, density, and Cp state paths;
- Henry liquid standard states combined with PSRK vapor fugacity where the
  simulator selects its aqueous Henry treatment.

Pure and mixture liquid densities deliberately use the shared liquid-volume
resolver. PSRK has no volume translation, and testing against Perry aqueous
ethanol densities showed that its incidental excess volume is much too small
and has the wrong temperature trend.

## Compiled numerical backend

When Numba is available, PSRK builds a dense fixed-component backend during
model initialization. It compiles the subgroup activity coefficients and
temperature derivatives, modified Huron-Vidal mixing rule, analytical SRK
cubic roots, fugacity coefficients, departure enthalpy, and iterative phi-phi
K-values. Temperature-dependent group interaction matrices are evaluated once
per state and reused for the mixture and all pure-component residual terms.

The readable Python implementation remains the numerical reference and the
automatic fallback when the compiled backend is unavailable. Parity tests
cover polar liquid, supercritical, and six-component light-gas/alcohol/water
states, including mixture derivatives, fugacity coefficients, and K-values.
The backend is compiled by explicit Numba signatures during
`Simulator.initialize()` without evaluating a sacrificial thermodynamic state.

## Current phase limitation

PSRK can represent liquid-liquid immiscibility mathematically, but the current
simulator VLLE engine is activity-model based. `PSRK` is therefore not yet
advertised to decanters, extractors, or `Flash3`; those require a future
fugacity-based stability and multiphase EOS flash rather than adapting the
low-pressure activity solver.

## Validation

The saved `scripts/validate_psrk2005.py` calculation reproduces selected 2005
paper curves for HCN/water, oxygen/ozone, chlorine/sulfur dioxide azeotropy,
and methane/water Henry coefficients. See
[`psrk2005_validation.md`](psrk2005_validation.md) for numerical results and
thermodynamic-identity checks.
