# Lyngby UNIFAC and RKSMHV2

Select `THERMO_METHOD: UNIFLBY` for the Larsen/Lyngby modified UNIFAC
activity-coefficient model with ideal vapor, or `THERMO_METHOD: RKSMHV2`
for SRK with the second-order modified Huron–Vidal mixing rule and the
Lyngby excess-Gibbs model. Both are available through `create_thermodynamics`
and in named thermodynamic scopes.
An executable input is provided in
[`examples/ethanol_water_mhv2.pfd`](../examples/ethanol_water_mhv2.pfd).

`UNIF-LBY`, `UNIFAC-LBY`, and `UNIFAC-LYNGBY` are aliases
for `UNIFLBY`; `RKS-MHV2` and `SRK-MHV2` are aliases for `RKSMHV2`.
Aliases are also accepted in `THERMO_SCOPES`.

## Parameter sources

The standalone model uses 45 subgroups, 21 main groups, and 307 directed
interactions from the open-source **thermo** project's transcription of
Larsen, Rasmussen and Fredenslund (1987),
[DOI 10.1021/ie00071a018](https://doi.org/10.1021/ie00071a018).
Source files were retrieved on 2026-09-19:

- [Subgroup definitions (`LUFSG`)](https://github.com/CalebBell/thermo/blob/master/thermo/unifac.py)
- [Interaction table](https://github.com/CalebBell/thermo/blob/master/thermo/Phase%20Change/UNIFAC%20Lyngby%20interaction%20parameters.tsv)
- The upstream MIT license is preserved in `data/source/lyngby_thermo_LICENSE.txt`.

RKSMHV2 additionally uses Tables I–III of Dahl, Fredenslund and Rasmussen
(1991), [DOI 10.1021/ie00056a041](https://doi.org/10.1021/ie00056a041),
verified against
`data/reference/thermodynamic-models/1991-dahl-fredenslund-rasmussen-mhv2-gas-solubility-vle.pdf`.
These supply 51 pure-component
Mathias–Copeman coefficient triples, 13 whole-molecule gas groups, and
gas–solvent interactions. Gas–gas interactions are explicitly zero in this
publication. Entries marked `na` remain unavailable. The Table III footnote
for H2/N2 with alkyl groups in alcohol molecules is included.

Human-readable source tables live in `data/source/lyngby_*` and
`data/source/mhv2_*`. `python scripts/build_lyngby_parameters.py` builds
`data/lyngby_unifac.json` and `data/mhv2_unifac.json`; edit source tables,
not runtime JSON. The builder does not fetch data or require network access.

## Equations and conventions

Lyngby uses `V_i = r_i^(2/3) / sum(x_j r_j^(2/3))` and
`ln(gamma_i^C) = 1 - V_i + ln(V_i)`, with the usual UNIFAC residual term.
Its interaction energy in kelvin is

```
a_ij(T) = a1 + a2*(T-T0) + a3*(T*ln(T0/T)+T-T0), T0=298.15 K
psi_ij = exp(-a_ij(T)/T)
```

This temperature convention differs from Dortmund's polynomial. Both the
Python and compiled UNIFAC paths use the same convention. The standalone
model supports the existing activity-model LLE/VLLE machinery.

RKSMHV2 uses linear covolume mixing, `b=sum(x_i b_i)`, and solves

```
q1*(D-sum(x_i D_i)) + q2*(D²-sum(x_i D_i²))
    = gE/(RT) + sum(x_i*ln(b/b_i))
D=a/(bRT), q1=-0.478, q2=-0.0047
```

The quadratic branch is continuous with the pure-component limit; a
nonreal or singular branch raises an error. Fugacity uses the partial molar
derivatives of `nD`. Enthalpy and entropy use the temperature derivative of
the same mixing rule. The excess-Gibbs temperature derivative uses a
fourth-order centered difference of the shared UNIFAC implementation.

Pure parameters use resolved Tc/Pc/omega. Explicit `mc_c1` (with optional
`mc_c2`/`mc_c3`, default zero) overrides Table I. Otherwise the Table I triple
is used when available; missing triples fall back to Soave alpha with a
warning. Above Tc the alpha polynomial uses only c1, as in Equation 5 of
Dahl et al.; it does not use the Boston–Mathias extrapolation. No volume
translation is applied. As with existing EOS packages, reported liquid
density uses the shared resolver; explicit EOS volume is available through
`molar_volume(T, P, composition, 'liquid')`.

## Groups and limits

Lyngby subgroup numbers differ from classic and Dortmund UNIFAC. For
example ethanol is `{1: 1, 2: 1, 12: 1}` and water is `{14: 1}`. Named groups
are also accepted. Ether CH-O (22) is distinct from aldehyde CHO (17).
Native assignments share the existing structural fragmenter and translate
only supported chemical identities. Unsupported chemistry is rejected.

RKSMHV2 assigns the 13 gases by resolved CAS identity; its gas subgroup
numbers are 46–58 (main groups 22–34). Alcohol alkyl subgroups 59–62
(main group 35) preserve the published alcohol-specific gas interactions.
Explicit groups are interpreted in the selected model's table.

Missing directed interactions raise an error in both new models. No
Dortmund/PSRK parameter substitutions or implicit zero interactions are
used. The standalone model retains the existing UNIFAC policy for gases
without a representable group assignment: these can be excluded from the
liquid group model. RKSMHV2 requires an assignment for every component.

RKSMHV2 supports phi–phi VLE and departure properties. It is not connected
to the current activity-model LLE/VLLE engine, just as PSRK is not. These
published tables are not a claim of bit-for-bit equivalence to proprietary
simulator databanks or later parameter revisions.

## Validation and timing

Tests cover published table values, reproducible data builds, compiled/Python
activity agreement, Gibbs–Duhem, partial-molar fugacity derivatives,
Gibbs–Helmholtz, pure-component limits, LLE, high-pressure VLE, property
overrides, scopes, and package imports.

`scripts/benchmark_phi_phi.py` provides seeded, uncached K-iteration timings
and fugacity residuals. The shared cubic/PSRK iteration now accepts a
single cubic root in either phase. Only a homogeneous `K=1` fixed point
falls back to the existing Wilson single-phase extrapolation; this is not
a replacement for a global EOS stability solver. Removing the repeated
root-count check inside the iteration reduces work in ordinary VLE cases.
