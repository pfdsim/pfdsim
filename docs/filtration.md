# Pressure cake filtration

`Filter` represents repeated cake formation, optional saturated-cake washing,
and optional gas-pressure deliquoring on a continuous-equivalent basis. It
balances each component and each solid particle-size class. This is a
mechanistic engineering model with explicit empirical closures, not a resolved
two-phase porous-medium simulation.

This page describes the default `washing_model=isothermal`. For different
wash/feed temperatures, melting, and recrystallization with changing PSD and
hydraulics, use [`washing_model=equilibrium`](equilibrium_washing.md).

Run the maintained example:

```sh
python cli.py examples/cake_filtration_washing.pfd
```

## Connections and operating modes

| Port | Meaning | Numeric alias |
| --- | --- | --- |
| `in` | Slurry feed with one liquid phase and suspended solids | `1` |
| `wash` | Optional solids-free wash liquid | `2` |
| `filtrate` | Combined primary filtrate, wash effluent, drained liquid, and escaped solids | `3` |
| `cake` | Captured solids and retained pore liquid | `4` |

Both outlets must be connected. Feed and wash must have the same temperature.
Wash pressure must be at least slurry-feed pressure. Vapor-bearing and
two-liquid-phase feeds or calculated outlets are rejected. Wash and mother
liquor are assumed mutually miscible throughout the operation.

Feed and wash flows are cycle-average rates. A cycle handles `flow *
cycle_time / 3600` kmol, with times in seconds. The model describes one filter
operating repeatedly, with upstream buffering implicit in the cycle-average
flows. No flow is silently throttled.

| Specifications | Calculation |
| --- | --- |
| `P_drop`, omit `area` | Size area to complete formation and washing in the available cycle time |
| `area`, omit `P_drop` | Solve the required pressure drop using cake and medium resistance |
| Both `area` and `P_drop` | Rate the requested load; report cycle feasibility and capacity ratio |

The available formation/washing time is `cycle_time - downtime -
deliquoring_time`. It must be positive. Pressure drop must leave a positive
absolute filtrate pressure. A pressure solve fails if the requested throughput
requires more pressure than the feed supplies. Outlet streams are at feed
temperature and `feed.P - P_drop`; a calculated vapor phase is rejected.

An undersized rating returns the requested-load material balances with
`cycle_feasible=false` and a warning. These balances are not an achievable
steady operating point. `capacity_ratio` is the factor by which **both** feed
and wash rates can be scaled, preserving composition and cycle timing, to fit
the area at the specified pressure. It does not promise unchanged moisture
after scaling.

## Parameters

Numbers without units use the defaults below. Python callers supply pressure
values in bar, including when unit metadata is retained; PFD pressure tags are
converted to bar by the flowsheet solver. Other unit tags are converted by the
filter itself. Unsupported parameters and dimensional units are rejected.

| Parameter | Unit/default | Constraint or meaning |
| --- | --- | --- |
| `cycle_time` | s, required | Positive interval between batches |
| `porosity` | Required | Operating cake void fraction, strictly between 0 and 1 |
| `capture_cut_size` | m, required | Projected-equivalent diameter with 50% capture; explicit zero selects ideal complete capture |
| `capture_sharpness` | 4 | Positive slope of the calibrated capture curve |
| `area` | m², optional | Positive filtration area |
| `P_drop` | bar, optional | Positive formation/washing pressure drop |
| `kozeny_constant` | 5 | Positive geometric factor for PSD-derived resistance |
| `specific_cake_resistance` | m/kg, optional | Measured reference resistance; overrides Kozeny–Carman calculation |
| `medium_resistance` | 1/m, 0 | Nonnegative wet medium resistance |
| `compressibility` | 0 | Empirical resistance exponent, `0 <= s < 1` |
| `reference_pressure` | bar, 1 | Reference differential pressure for resistance |
| `downtime` | s, 0 | Loading, discharge, and other nonprocessing time |
| `liquid_viscosity` | Pa·s, optional | Override mother-liquor viscosity; otherwise use thermodynamic transport API |
| `wash_viscosity` | Pa·s, optional | Override wash viscosity; otherwise use transport API |
| `wash_cells` | 10 | Integer 1–200; physical washing/mixing parameter |
| `deliquoring_time` | s, 0 | Duration of post-wash drainage |
| `deliquoring_pressure` | bar, `P_drop` | Gas-to-filtrate pressure difference during drainage |
| `entry_pressure` | bar, required when draining | Positive capillary entry pressure |
| `residual_saturation` | Required when draining | Irreducible saturation, `0 <= Sr < 1` |
| `pore_index` | Required when draining | Positive Brooks–Corey pore-distribution index λ |
| `relative_permeability_exponent` | `3 + 2/λ` | Positive liquid relative-permeability exponent |

Time tags include `s`, `min`, `h`; diameter tags include `m`, `mm`, `um`;
viscosity accepts `Pa*s`, `cP`, and `mPa*s`. `kozeny_constant` and an explicit
`specific_cake_resistance` cannot both be supplied. Washing parameters require
a wash stream; drainage parameters require positive deliquoring time.

The medium's capture curve and hydraulic resistance are independent measured
inputs. A nominal manufacturer pore size is not automatically a capture cut
size. Example values are illustrative, not a material-specific calibration.

## PSD, shape, and solids slip

Each solid component can have its own extensive PSD and sphericity `φ`.
Particle diameters `d` are volume-equivalent diameters. A representative
particle diameter is accepted as a one-class PSD. Predictive resistance or
nonzero capture cut size requires particle data and sphericity for every
active solid component. No mean diameter silently replaces a supplied PSD.

For randomly oriented convex particles, projected-area scaling motivates the
closure `d_projected = d / sqrt(φ)`. Shape alone does not uniquely determine
passage through a pore, so this effective-size approximation and the capture
curve require calibration for the actual particles and medium. With cut size
`dc` and sharpness `n`, the captured fraction of each size class is

```text
η(d, φ) = 1 / [1 + (dc / d_projected)^n].
```

The cake receives `η * class_flow` and the filtrate receives the complement.
Both streams retain separate PSDs, including their absolute class flows.
Only actual solid inventory is classified: dissolved material follows the
liquid even when the same component also exists as conventional solid.

This is a calibrated cycle-average capture law. It does not predict the
evolution of capture as cake builds, pore blocking, bridging, particle
deformation, or erosion. The hydraulic model uses the resulting captured
population, so removal of fines changes both cake loading and resistance.

Let `vi` be the captured material volume of a particle class, including all
solid components, and `Vs = sum(vi)`. The actual surface per solid volume is

```text
Sv = sum[6 vi / (φi di)] / Vs
ρs = captured dry mass / Vs
αref = K Sv² (1 - ε) / (ρs ε³)
α(ΔP) = αref (ΔP / ΔPref)^s.
```

For monodisperse spheres and `K=5`, this gives the familiar coefficient 180.
For a single solid component it corresponds to the surface-volume mean
diameter, including its sphericity. Doubling every diameter quarters reference
resistance at unchanged capture and packing. A supplied measured resistance
overrides this prediction; PSD and shape still determine selective capture.

Porosity is the specified operating porosity. The empirical compressibility
law modifies average hydraulic resistance, and does not solve consolidation
or update porosity spatially or during drainage. It uses total filtration
pressure drop as its calibration pressure, including medium pressure loss.

## Formation and retained liquid

All hydraulic volumes are liquid or true solid volumes, not bulk slurry
volume inferred from total-stream density. Solid volume comes from the solid
property resolver. Liquid volume comes from the liquid mixture property API.

For one cycle, pore volume `Vp = Vs ε/(1-ε)`. Initially it retains the mother
liquor composition. The primary filtrate volume is feed-liquid volume minus
`Vp`; feeds without enough liquid to produce a saturated cake and positive
filtrate are rejected. Cake thickness is `Vs / [(1-ε) A]`.

For constant pressure and cycle-average capture, integrated Darcy filtration
gives, using captured dry mass `M`, primary filtrate volume `Vf`, medium
resistance `Rm`, and viscosity `μ`:

```text
tformation = μ/ΔP [α M Vf/(2 A²) + Rm Vf/A].
```

The model integrates from a clean, wetted medium with zero initial cake.
Medium resistance is supplied rather than updated by a fouling model.

## Washing

Wash amount is determined by the actual wash inlet, not an invented solvent
source. `W = wash volume / Vp`. A sequence of `N=wash_cells` equal mixed cells
represents washing dispersion. With initially uniform mother-liquor tracer:

```text
dr1/dW = -N r1
drj/dW = N (r[j-1] - rj)
rj(0) = 1.
```

The solver evaluates the analytic Poisson-sum solution. One cell gives fully
mixed washing, `r=exp(-W)`; increasing cell count approaches displacement
washing. Cell count is a calibrated mixing parameter, not a numerical grid
convergence setting. The residual profile and its average are reported.

Every liquid component is carried through the same conservative displacement
balance. Retained composition is the volume-weighted mixture of remaining
mother liquor and replacement wash liquid. Additive liquid volumes and no
solid dissolution, adsorption, or precipitation are assumed.

Washing time integrates Darcy resistance over delivered wash volume. Cake
resistance uses the average cell viscosity; the medium uses the last cell's
viscosity. Viscosity in each cell is linearly interpolated by volume between
resolved feed and wash viscosities. It is an explicit approximation for
strongly nonideal viscosity mixtures, not a full local composition-dependent
transport solve. The coefficients yield
`tformation + twashing = a/A² + b/A`, giving an analytic area solution and a
monotone numerical pressure solution for `s < 1`.

## Deliquoring

After washing (or directly after formation), the pore liquid is treated as a
uniformly mixed inventory with uniform saturation `S`. Define

```text
Se = (S-Sr)/(1-Sr)
Pc = Pb Se^(-1/λ)
krl = Se^n
Vp dS/dt = -A max(ΔPg-Pc, 0) / [μ (α M/(A krl) + Rm)].
```

`Pb` is entry pressure and `ΔPg` is the specified gas pressure difference.
Drainage stops asymptotically at
`Seq = Sr + (1-Sr) min[1, (Pb/ΔPg)^λ]`. Below entry pressure there is no
drainage. An implicit scalar ODE integration tracks saturation and checks the
equilibrium bound. Drained components enter the combined filtrate; their
complement remains in the cake. Residual saturation is not itself the final
saturation at finite pressure or finite time.

Gas is a utility pressure boundary with negligible flow resistance. This
implementation does not calculate gas consumption, gas heating, evaporation,
gas dissolution, a saturation front, or spatial capillary gradients. Cake
structure remains fixed during this stage. In particular, this reduced
drainage model should not be used as a substitute for a spatial two-phase
model when gas resistance, cracking, or a drainage front controls operation.

For context on integrated filtration/washing and spatial Darcy–capillary
models, see Destro et al., [Continuous integrated filtration, washing and
drying of aspirin (2021)](https://doi.org/10.1016/j.ifacol.2021.08.231).
The present implementation is an independently specified reduced model, not
a reproduction of that paper's spatial model or thermal drying stage.

## Energy and reporting

The isothermal duty equals outlet enthalpy flow minus all material-inlet
enthalpy flows, including solid sensible enthalpy and mixing effects from the
selected thermodynamics. Pump/compressor work and gas utility enthalpy are
not computed. Pressure equipment can be represented separately where the
simulator supports the corresponding inlet material.

`UnitResult.performance` includes operating mode, required and actual area,
pressure drop, cycle feasibility, capacity ratio, stage times, cake thickness,
resistance, captured and escaped solids, saturation, retained-liquid mass,
wet-basis cake moisture, wash tracer profile, viscosities, and duty. Standard
stream reports contain outlet compositions and the two classified PSDs.

`tests/test_filtration.py` checks analytic Darcy and washing limits, capillary
threshold and equilibrium, PSD-class and component conservation, sphericity
effects, pressure/size scaling, conventional solid routing, energy balances,
invalid specifications, and PFD execution. Constitutive parameters still need
experimental validation for each intended application.
