# Equilibrium warm washing in Filter

Set `washing_model = equilibrium` to wash a stationary cake with liquid at a
different temperature, including the pure melt of a crystallizing component.
The default `isothermal` model remains unchanged.

See [equilibrium_warm_melt_washing.pfd](../examples/equilibrium_warm_melt_washing.pfd)
for an ice/ethanol example using warm water as the wash melt. Its subcooled
viscosity correlations are illustrative inputs, not experimentally validated
fits. The example crystallizes a slurry first, then forms and washes the cake.

## What is solved

The existing filter capture curve first splits incoming particle classes
between the cake and primary filtrate. Cake formation uses the original
Darcy model and feed temperature. The subsequent washing stage is adiabatic:
heat arrives with wash liquid and leaves with displaced liquid. No external
washing duty is invented to close an energy imbalance.

The cake is represented by a sequence of saturated, mixed spatial cells.
Liquid and solid share a temperature within each cell (local thermal
equilibrium). Every wash increment passes through the cells in order. At
each contact the solver simultaneously determines temperature and pure-solid
phase amounts from component inventories, enthalpy, and solid–liquid
equilibrium. This includes sensible heat, fusion enthalpy, and the liquid
activity model supplied by the selected thermodynamics.

For each potentially crystallizing component the conditions are

```text
nsolid >= 0
ln(a_sat) - ln(a_liquid) >= 0
nsolid * [ln(a_sat) - ln(a_liquid)] = 0.
```

These conditions and the energy balance are solved together. In particular,
pure material can partially melt or freeze at its melting temperature; the
solver does not search for a unique temperature through a discontinuous
temperature-only phase split. An exact all-liquid enthalpy branch is tested
for stability to handle phase disappearance cleanly.

After equilibration, the cell releases enough liquid to restore its fixed
bulk volume. The released liquid has the cell's equilibrium composition and
temperature. Removing a portion of that liquid preserves equilibrium with
the retained solids. Each step conserves every component and its energy,
subject to the reported equilibrium-solver tolerance.

The instantaneous pore fraction is

```text
epsilon_j = 1 - Vsolid,j / Vcell,j.
```

Melting increases pore volume; recrystallization decreases it. Solid molar
volumes and liquid mixture volumes come from the property package. The model
rejects pore blockage or a contact that cannot remain saturated because of
volume contraction. It does not silently fill the missing volume with liquid
or discard an excess solid inventory.

## PSD and hydraulics

For surviving particles, isotropic growth/shrinkage scales class diameters
while preserving each class's particle number and sphericity. The diameter
scale accounts for both changing solid amount and thermal molar volume.
This is a homothetic population closure; it does not model preferential
surface recession, breakage, agglomeration, or growth kinetics.

New crystals without surviving seed particles require an explicit
`equilibrium_nucleus_diameter`. Their effective sphericity comes from the
component particle default, or the spherical default 1.0. This defines an
effective population for equilibrium calculations, not a nucleation rate.

Each cell's current PSD, shape, and porosity determine its Kozeny–Carman
resistance. Viscosity is resolved at the current liquid composition,
temperature, and pressure, rather than interpolated between feed viscosities.
A supplied `specific_cake_resistance` calibrates the initial geometric model;
its subsequent resistance still evolves with the geometry and population.
The original compressibility specification calibrates the formed bed at
the operating pressure; no mechanical consolidation model is added.

For every delivered wash increment, Darcy pressure losses through the cells
and medium determine the elapsed time. Inlet and outlet liquid volumes may
differ because of thermal expansion and phase change. The cell flux is
approximated by their average, and its hydraulic coefficient by the average
before and after contact. The medium uses its exiting liquid's viscosity.
Summed stage durations have the form `a/A^2 + b/A` at a fixed pressure drop.

The model supports the existing three specification modes:

- `P_drop` only: calculate required area.
- `area` only: solve the required pressure drop, searching within the
  liquid-only operating region.
- Both: rate the requested throughput and report whether it fits the cycle.

Local thermodynamics and transport use **filtrate pressure as the common
reference pressure**. The pressure drop is distributed through the hydraulic
resistances for the time calculation, but a separate local-pressure SLE field
is not solved. This incompressible-liquid approximation is inappropriate if
pressure-induced phase shifts across the bed are significant.

The SLE pressure correction retains temperature-dependent solid volume below
the normal melting point. For the hypothetical superheated solid used to
check liquid stability, it uses the solid volume at the melting point.
This explicit constant-volume extrapolation avoids requesting stable-solid
density data above melting; it is an approximation to the metastable solid
reference, not a claim that solid exists there.

## Deliquoring and collection

Optional deliquoring uses the existing capillary law and a common saturation
for the warm bed. Its resistance sums the final cell resistances weighted by
their local viscosities. Each cell releases the same fraction of its liquid
at its own temperature; the remaining material keeps its local equilibrium.
Gas is still a pressure utility, with no evaporation, gas consumption, or
gas/solid heat transfer calculation.

Cake contents are then collected adiabatically. Primary filtrate, wash
effluent, and drained liquid are collected separately. Each collection has
its own enthalpy/SLE solve, so further melting or crystallization caused by
mixing is included rather than reporting a frozen phase split at an averaged
temperature. PSD amounts and sizes follow the collected phase amounts;
unseeded precipitation again requires an effective nucleus diameter.
Collection phase changes are reported separately from the precollection
cell inventories and the initial filtration capture efficiency.

The unit heat duty is the original isothermal formation duty. Washing,
deliquoring, and collection are adiabatic material operations. The reported
energy residual verifies this independently against the final outlet states.

## Additional specifications

| Parameter | Meaning |
| --- | --- |
| `washing_model = equilibrium` | Enables this model; a wash inlet is required |
| `wash_cells` | Number of spatial mixed cells, integer 1–100; default 5 in this mode |
| `wash_steps` | Number of delivered-wash increments, integer 1–10000; default 100 |
| `equilibrium_tolerance` | Scaled energy/complementarity tolerance, 1e-12–1e-5; default 1e-8 |
| `T_equilibrium_min` | Optional lower temperature bound, standard temperature units |
| `T_equilibrium_max` | Optional upper temperature bound, standard temperature units |
| `equilibrium_nucleus_diameter` | Effective diameter for newly formed, unseeded solids; m, mm, or um |

Temperature bounds default to 25 K below the colder inlet (at least 1 K) and
25 K above the warmer inlet. Explicit bounds must bracket both inlet
temperatures. Widen them if large heats of mixing require it and the property
models are valid there. `wash_viscosity` is rejected in equilibrium mode;
provide temperature-dependent property correlations instead. The optional
`liquid_viscosity` override still applies to the initial formation stage.

`conventional_with_solid` components can melt, dissolve, or crystallize;
`permanent_solid` components retain their solid inventory and contribute
thermal mass and hydraulic surface. Fusion data, liquid and solid heat
capacities, volumes, and local liquid viscosities must be available. Thermal
conductivity is not needed for this local-thermal-equilibrium model.

Use wash-step refinement to check delivered-volume/time discretization
error. More cells resolve sharper temperature and composition fronts and
reduce mixed-cell dispersion; spatial refinement is a separate check. The
default discretization is an engineering starting point, not an error bound.

The solver reuses temperature-independent viscosity group assignments and
the preceding cell hydraulics. Seeded equilibrium contacts first try a
Newton continuation from the preceding state, with the bounded solver and
an exact all-liquid stability branch retained as fallbacks. These execution
optimizations do not relax phase checks, tolerances, or cell/increment counts.

For reproducible performance checks, run
`python scripts/benchmark_equilibrium_washing.py --acid-bottoms`.
This standalone UNIQUAC case uses AA/PA/water/MIBK, a 400 µm mass-lognormal
PSD, GSD 1.6, sphericity 0.75, and 10 cells × 100 increments. It assumes
280 K crystallization and 12 kg/h wash per 100 kg/h feed; it does not solve
for 60% crystal yield. It does not run the upstream lactic-acid process.
`--revision <commit>` compares the same inputs with that implementation;
`--stack-after <seconds>` provides a single diagnostic stack if a case stalls.

## Reports and limits

Performance reports include cell temperatures, liquid compositions, solid
inventories, porosities, viscosities, initial/final cake resistance, stage
times, pressure drop, area, equilibrium residuals, and component/energy
residuals. `net_solid_change_kmol_h` is the change from captured feed solids
to collected cake solids. `collection_solid_change_kmol_h` isolates the net
solid change in both collection operations. Standard outlet reports include
the resulting PSDs.

This is a stationary cake warm-washing model. It is not a countercurrent
moving-bed hydraulic wash column. Cell bulk volumes remain fixed: bed motion,
mechanical collapse, compressive settling, cracking, and channel formation
are not represented. The liquids must remain homogeneous and liquid;
detected local vapor or a second liquid phase is rejected. Solid solutions,
co-crystals, polymorph competition, kinetic melting, heat conduction, and
finite liquid/particle heat-transfer resistance are outside this model.

For physical context on pure-melt washing and recrystallization at a wash
front, see [Process for controlling a hydraulic wash column](https://patents.google.com/patent/US7637965B2/en).
That moving-bed process is not the equipment topology implemented here.
