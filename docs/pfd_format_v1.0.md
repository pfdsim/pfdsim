# PFD File Format Specification

**Format version:** 1.0  
**Last updated:** 2026-08-24  
**Status:** Current

## Overview

The PFD (Process Flow Diagram) format is a human-readable text format for defining chemical process simulations. It uses a structured syntax that is both easy to write manually and parse programmatically.

## File Extension

- `.pfd` - Process Flow Diagram definition file

## Format Version

All PFD files should include a `VERSION` directive. If omitted, version 1.0 is
assumed. The value is retained as process metadata; the current parser does not
select a different grammar from it.

```
VERSION: 1.0
```

## File Structure

A PFD file consists of top-level directives and indented block sections:

1. **Metadata and policies** - Process identity, property method, lookup policy,
   and recycle controls
2. **Thermodynamic scopes** - Optional named property-method contexts
3. **Components** - Chemical species definitions
4. **Property correlations** - Optional pure-component correlation overrides
5. **Interaction parameters** - Optional binary-parameter overrides
6. **Streams** - Material flow definitions
7. **Units** - Unit operation, port, parameter, and reaction definitions

`COMPONENTS:` must precede `PROPERTY_CORRELATIONS:` because correlation rows
are resolved against components as they are parsed. Put `INTERACTION_PARAMETERS:`
after `COMPONENTS:` as well. Streams and units may appear in either order.

Top-level keywords and section headers are case-sensitive. Entries belonging to
`THERMO_SCOPES:`, `COMPONENTS:`, `PROPERTY_CORRELATIONS:`,
`INTERACTION_PARAMETERS:`, `STREAM`, or `UNIT` must be indented with spaces or
tabs. An unindented non-comment line ends the current block.

### Comments

Lines starting with `#` are comments and ignored. Inline comments are also
accepted when `#` begins a token outside a quoted string—that is, at the start
of a line or after whitespace.

```
# This is a comment
PROCESS: Example # This is an inline comment
    X | Hydrogen cyanide | SMILES=C#N # The SMILES hash is preserved
```

### Scalar and Collection Values

Component, correlation, and interaction fields use comma-separated `key=value`
pairs. Values may be numbers, `true`/`false`, bare strings, quoted strings,
lists such as `[1, 2, 3]`, or maps such as `{A:1,B:2}`. Quote strings containing
spaces, commas, `|`, brackets, or braces. Stream and unit parameter values are
kept as text and converted by the consuming operation; an optional unit is
written in square brackets.

## Section Specifications

### 1. Metadata

```
PROCESS: <process name>
VERSION: <version number>
DESCRIPTION: <description text>
AUTHOR: <author name>
DATE: <date>
```

All metadata fields are optional.

### 2. Online Lookup Policy

```
ONLINE_LOOKUP: true
```

Optional. Defaults to `true`. Set `ONLINE_LOOKUP: false` for self-contained
PFDs that should use only local/textbook/Perry/property-cache-independent data
plus properties supplied in the PFD. Local lookup still runs; the online and
cached-online component lookup path is skipped.

Accepted true values are `true`, `yes`, `1`, `on`, `enabled`, and `enable`;
accepted false values are `false`, `no`, `0`, `off`, `disabled`, and `disable`.
`ALLOW_ONLINE_LOOKUP:` and `FETCH_ONLINE:` are aliases.

### 3. Canonical Vapor-Pressure Minimum

```
PSAT_MINIMUM_PRESSURE: 1 [Pa]
```

Optional. Sets the lower pressure used when constructing canonical vapor-
pressure curves for every component in the PFD. If omitted, the default is
`0.001 bar`. Pressure units `bar`, `Pa`, `kPa`, `MPa`, `atm`, `psi`, `mmHg`,
and `torr` are accepted; a unitless value is interpreted as bar.

`MINIMUM_PRESSURE`, `MINIMUM_PSAT_PRESSURE`, and
`MINIMUM_VAPOR_PRESSURE` are accepted as aliases. Serialization always emits
`PSAT_MINIMUM_PRESSURE` in bar.

This changes the qualified lower Psat domain and enables the A-H inverse-tail
fit retry when the declared pressure is below the default. It does not impose
an exact lower-pressure fit constraint.

### 4. Thermodynamics Method

```
THERMO_METHOD: <method> | <option>=<value>, ...
```

`THERMO:` and `PROPERTY_METHOD:` are accepted aliases. Unsupported method names
are parse errors. Likely misspellings suggest an available spelling; recognized
methods from other simulators may instead suggest the nearest available model
and identify core model families intended for future implementation.

The recognized planned-method diagnostics currently include `PRWS` and
`RKSWS`/`SRKWS` (Wong-Sandler GE-EOS mixing rules), `PC-SAFT`, `CPA`, `LKP`
(Lee-Kesler-Plocker), `IAPWS-95`, and `AMINES` for amine/acidic-gas systems.
`IAWPS-95` is recognized as a common transposition and points to the standard
`IAPWS-95` name. These declarations remain parse errors until their models are
implemented; suggested available methods are approximations, not silent
substitutions.

- `IDEAL` - Ideal gas law + Raoult's law (default)
- `RK` or `REDLICH-KWONG` - Redlich-Kwong equation of state
- `SRK` or `SOAVE-REDLICH-KWONG` - Soave-Redlich-Kwong equation of state
- `PR` or `PENG-ROBINSON` - Peng-Robinson equation of state
- `RKS-BM`, `PR-BM` - SRK and PR with Boston-Mathias alpha function
- `SRK-MC`, `PR-MC` - SRK and PR with Mathias-Copeman alpha below the
  critical temperature and Boston-Mathias continuation above it
- `PSRK` or `PREDICTIVE-SRK` - Published 2005 predictive SRK model using the
  PSRK-UNIFAC excess-Gibbs mixing rule and Mathias-Copeman pure-component
  parameters. Components require a published PSRK subgroup assignment. See
  [`psrk.md`](psrk.md) for fallback and phase-capability details.
- `PRSV1`, `PRSV2` - Peng-Robinson-Stryjek-Vera alpha variants; `PRSV` is an
  alias for `PRSV1`
- `SRK-TWU`, `PR-TWU` - SRK and PR with Twu alpha parameters
- `NRTL`, `NRTL-VDM`, `NRTL-RK`, `NRTL-PR`, `NRTL-BV` - NRTL liquid activity
  coefficients, optionally with vapor-dimerization, RK, PR, or
  second-virial vapor fugacity corrections
- `UNIQUAC`, `UNIQUAC-RK`, `UNIQUAC-PR`, `UNIQUAC-BV` - UNIQUAC liquid activity
  coefficients, optionally with RK, PR, or second-virial vapor fugacity
  corrections
- `UNIFAC`, `UNIFAC-RK`, `UNIFAC-PR`, `UNIFAC-BV` - Original UNIFAC
- `UNIFDMD`, `UNIFDMD-RK`, `UNIFDMD-PR`, `UNIFDMD-BV` - Dortmund modified UNIFAC
- `UNIFNIST`, `UNIFNIST-RK`, `UNIFNIST-PR`, `UNIFNIST-BV` - NIST modified UNIFAC
- `NRTL-VDM`, `UNIQUAC-VDM`, `UNIFAC-VDM`, `UNIFDMD-VDM`, `UNIFNIST-VDM` - Activity
  models with vapor-dimerization fugacity corrections
- `STEAM` - CoolProp IF97 steam properties for water-only flowsheets

The `-BV` variants use a pressure-truncated second-virial vapor EOS while
retaining the selected activity model for the liquid. The default coefficient
provider is Tsonopoulos, including its supported polar-fluid corrections and
published binary interaction parameters. Binary-specific fitted values take
precedence over the published critical-volume correlations and molecular-family
defaults; `k_ij=0` is retained only for pairs without a supported rule. The
parameter record retains its source, fitted uncertainty, and temperature range
where those were reported. Explicit provider-level binary overrides have the
highest precedence. Component dipoles are resolved through the normal dipole
hierarchy, with published pure-component polar parameters used where required
by the fitted binary table. Vapor fugacity coefficients use

`ln(phi_i) = P/(R*T) * (2*sum_j(y_j*B_ij) - B_mix)`.

The same `B_mix(T)` and its temperature derivatives supply vapor residual
enthalpy, entropy, Gibbs energy, and heat capacity. The truncated virial form
is intended for low-to-moderate vapor densities; it is not a liquid EOS.
The provider can be selected explicitly. `PITZER-CURL` and `ABBOTT` use their
respective nonpolar corresponding-states forms with the same cross-property
combining rules; unlike the extended Tsonopoulos provider, they do not apply
dipole-based polar corrections. `HOC` implements the Hayden-O'Connell physical,
polar, association, and representative-group solvation contributions from the
1975 paper and its core supplementary tables. It resolves both molecular
dipole and Thompson's modified radius of gyration from the property system:

```pfd
THERMO_METHOD: NRTL-BV | correlation=TSONOPOULOS
THERMO_METHOD: NRTL-BV | correlation=PITZER-CURL
THERMO_METHOD: NRTL-BV | correlation=ABBOTT
THERMO_METHOD: NRTL-BV | correlation=HOC
```

HOC also has direct activity-model suffixes. These are normalized internally
to the corresponding `-BV` method with `correlation=HOC`:

```pfd
THERMO_METHOD: NRTL-HOC
THERMO_METHOD: UNIQUAC-HOC
THERMO_METHOD: UNIFAC-HOC
THERMO_METHOD: UNIFDMD-HOC
THERMO_METHOD: UNIFNIST-HOC
```

Canonical PFD serialization writes the expanded `-BV | correlation=HOC`
form. `UNIF-HOC` is accepted as an alias for `UNIFAC-HOC`.

HOC association and solvation parameters follow the generalized group values
in supplementary Tables IV and V. A parenthesized Table IV group label is
transferable to other members of that class, and Table V superscript-`e` rows
are transferred across their stated solvating groups. Unlabelled Table V
components remain identity-specific. The paper provides no additive rule for
multifunctional compounds, so multiple association groups are not summed and
receive no inferred association parameter. As specified by Table IV, water
and monohydric alcohols share the transferable Hydroxyl association group.

For systems without a carboxylic acid, HOC uses its total second virial
coefficients in the ordinary pressure-truncated backend. When one or more
monocarboxylic acids are present, the backend automatically uses the paper's
equation 31 chemical theory: acid-pair physical fugacities use `B_free`, while
`B_bound + B_metastable + B_chem` supplies the association equilibrium
constants. Every acid homodimer and heterodimer is solved on the nominal
monomer-equivalent composition basis. The unlike-acid equilibrium constant
includes the required statistical factor of two, and two acids use the same
organic-acid group value `eta=4.5` with HOC's ordinary cross molecular
parameters. Internal dimers do not appear as flowsheet components.

When Numba is available, the RK/SRK/PR family uses a shared dense compiled
backend for alpha functions, temperature-dependent binary interactions,
quadratic mixing, analytical cubic roots, departure properties, fugacity
coefficients, and the complete phi-phi K-value iteration. This covers `RK`,
`SRK`, `PR`, their Boston-Mathias, Mathias-Copeman, and Twu variants, and
`PRSV1`/`PRSV2`. The readable Python implementation remains the automatic
fallback and numerical reference.

`Simulator.initialize()` compiles or loads the Numba signatures for every
compiled backend reachable by the selected thermodynamic method and flowsheet
capabilities. Ordinary VLE does not compile optional LLE/VLLE kernels; LLE,
VLLE, and VDM kernels are prepared only when the phase model or unit operations
can call them. The compilation step does not perform a sacrificial property or
equilibrium calculation, and `run()` introduces no new compiled signatures.

#### Thermodynamic Scopes

Named scopes allow different units to use independent thermodynamic packages:

```pfd
THERMO_METHOD: UNIQUAC-VDM

THERMO_SCOPES:
    extraction | method=NRTL
    recovery   | method=UNIQUAC, inherit=global
    polishing  | method=UNIQUAC, inherit=recovery
```

`global` is reserved for the process-level `THERMO_METHOD`. A unit uses the
global package unless it explicitly selects a named scope:

```pfd
UNIT X-301 : RigorousLiquidLiquidExtractor
    thermo_scope = extraction
```

Scope names are unique identifiers. Multiple scopes may use the same method;
each scope still owns a distinct immutable package. Omitting `inherit` creates
an isolated scope. `inherit=global` or `inherit=<scope>` imports the parent's
effective interaction records. A child record for the same model/component
pair replaces the inherited records for that pair. Unknown parents and
inheritance cycles are errors.

Material streams are recomputed only where their source and destination scopes
differ. Adjacent units in the same scope share the existing state directly.
At a boundary, temperature, pressure, total flow, and component flows remain
fixed while context-owned properties (`H`, `S`, `Cp`, phase equilibrium,
density, and viscosity) are evaluated using the receiving package. This lets
scoped liquid-viscosity interactions—and future excess-volume interactions—
remain isolated to their intended units. The
enthalpy-flow difference is reported as a thermodynamic-scope correction and
is included in overall energy reconciliation without being assigned to a unit
duty.

#### Fluid Phase Model

```
FLUID_PHASE_MODEL: <VLE | VL(L)E | VLLE>
```

Optional. Defaults to `VLE`. `FLUID_PHASES:` is an accepted alias.

- `VLE` preserves the conventional one-vapor/one-liquid flash behavior. It
  performs no automatic liquid-liquid stability test. Dedicated LLE/VLLE
  units such as `Decanter` and `Flash3` retain their explicit behavior.
- `VL(L)E` first performs the conventional VLE flash, then applies a cheap
  local spinodal test to a meaningful liquid phase. A locally unstable liquid
  triggers the VLLE splitter. A locally stable result is retained without a
  global tangent-plane search, so this mode can intentionally retain a
  metastable liquid between the binodal and spinodal boundaries.
- `VLLE` performs the robust global liquid-stability/VLLE search for every
  unconstrained equilibrium state. It permits, but does not force, a vapor and
  two nonzero liquid phases.

`VL(L)E` and `VLLE` require an LLE-capable activity-coefficient method:
UNIFAC-family, NRTL, UNIQUAC, or their supported vapor-correction variants.
Selecting either mode with IDEAL, STEAM, or a cubic-EOS-only method is an
initialization error.

`SLE` and other solid-equilibrium phase-model declarations are not accepted as
near-spellings of `VLE`. Their parse diagnostic instead explains the available
component-level solid declarations. A nonparticipating process solid can use
`phase_behavior=permanent_solid` and optionally select a
`solid_material_form`. A conventional fluid component whose stream inventory
may also contain an explicitly assigned solid portion can use
`phase_behavior=conventional_with_solid` or its `three_phase` alias. This
declaration enables accounting only; it does not freeze the component during a
state calculation. Components that dissolve or precipitate will use the future
`phase_behavior=soluble_solid` declaration when soluble-solid/SLE equilibrium
support is implemented.

The current frozen pure-water Henry standard-state path supports `VLE` only.
If an aqueous unit activates Henry treatment while `VL(L)E` or `VLLE` is
selected, the calculation fails clearly rather than silently disabling Henry
or skipping the declared liquid-stability policy. Henry-aware LLE/VLLE and
non-water-solvent dilute standard states are future extensions.

The declaration governs unconstrained equilibrium state calculations. An
explicit `phase=liquid` or `phase=vapor` property/state probe remains
constrained and does not recursively invoke liquid stability calculations.

Ordinary equipment retains two discovered liquid phases in one multiphase
liquid outlet. `Flash3` and `Decanter` remain the units that physically route
the two liquids into distinct outlet streams.

Rigorous distillation inherits the global policy when no unit-local
`stage_phase_model`/`valid_phases` value is supplied. Under `VL(L)E`, it first
solves the VLE MESH equations, spinodal-checks every converged liquid stage,
and reruns with the VLLE MESH model only if a stage is locally unstable. An
explicit unit stage policy overrides the global default. Simpler stage models
that cannot retain two liquids reject an inherited LLE-aware policy rather
than silently reducing it to VLE.

Stream states reserve vapor, liquid-1, liquid-2, and solid phase inventory.
Explicit permanent-solid declarations populate the solid fraction,
composition, component-flow map, and optional particle defaults without
entering fluid equilibrium. Inactive liquid-2 and solid slots remain present
in memory but are omitted from serialized/PFR reporting until they contain a
fraction, composition, component flow, or particle value.

Particle-size distributions are stored on each stream's solid-component
inventory. A component declaration can provide the default population used
whenever that component enters the solid phase:

```text
COMPONENTS:
    salt | Sodium chloride | type=permanent_solid, PSD={diameters_um:[50, 100, 200], fractions:[0.2, 0.3, 0.5], basis:mass}
```

`diameters_m`, `diameters_mm`, or `diameters_um` supplies strictly increasing
representative volume-equivalent class diameters. `fractions` must be
nonnegative and have a positive sum; it is normalized during parsing. The
accepted input bases are `mole`, `mass`, `volume`, and `number`. Internally,
every distribution is converted to component molar flow in each size class,
so PSD material is extensive and conserved by mixers and non-size-selective
splits. For one pure component, mass, volume, and mole fractions are
equivalent; number fractions are converted using particle volume proportional
to diameter cubed.

Two conventional continuous distributions are also accepted:

```text
# Lognormal cumulative distribution on the declared basis.
PSD={distribution:lognormal, d50_um:200, GSD:1.6, basis:mass, classes:20}

# Weibull, also known as Rosin-Rammler or RRSB for particle sizing.
PSD={distribution:rosin_rammler, d63_2_um:500, shape:2.5, basis:volume, classes:20}
```

For `lognormal`, `d50` is the median diameter of the distribution on the
declared basis and `GSD` is the geometric standard deviation. The canonical
names are `d50_m` and `geometric_standard_deviation`; `d50`, `d50_mm`,
`d50_um`, `median_diameter`, `GSD`, and `sigma_g` are accepted input forms.
GSD must be at least one; exactly one represents a monodisperse population.

For `weibull`/`rosin_rammler`, the cumulative undersize fraction is

```text
Q(d) = 1 - exp(-(d / d63.2)^shape)
```

Consequently, the scale or characteristic diameter is the diameter at
`1 - exp(-1)`, approximately 63.2% cumulative undersize, on the declared
basis. The canonical names are `scale_diameter_m` and `shape`;
`characteristic_diameter`, `d63_2`, `spread_parameter`, and `exponent` are
accepted input forms, with the same `_m`, `_mm`, and `_um` diameter suffixes.

Parametric PSDs require an explicit basis because a median or percentile is
not meaningful without its weighting basis. `classes` is an integer from 2 to
1000 and defaults to 20. Parametric distributions are converted to the
canonical discrete population using equal-probability classes: every class
represents the same fraction on the declared basis and its representative
diameter is the inverse cumulative distribution at the class midpoint. This
accounts for the complete analytical distribution without selecting arbitrary
finite tail cutoffs. The normalized PFD representation retains the analytical
distribution and parameters; stream results contain the instantiated discrete
class flows.

A feed stream may override the component default independently for a
`permanent_solid` component:

```text
STREAM SaltFeed : FEED -> MIX.salt
    T = 25 [C]
    P = 1 [bar]
    F = 2 [kmol/h]
    x = salt:1
    PSD = {salt:{diameters_um:[25, 75], fractions:[0.6, 0.4], basis:number}}
```

`PSD` is a map because a stream can carry independent populations for several
solid components. Each attached population must account for the entire solid
flow of that component. The legacy `particle_diameter` field is interpreted
as a one-class monodisperse PSD when no explicit PSD is supplied; a component
cannot declare both fields.

A `conventional_with_solid` component may declare a component-level PSD
default, which is instantiated when a crystallizer or another solid-forming
calculation creates an explicit solid inventory. Stream-level PSD overrides
for such components are not currently accepted because the PFD stream format
cannot yet allocate the component between fluid and solid phases. Explicit
stream solid-component allocation is reserved for a future format extension;
when introduced, it will provide the solid flow to which the stream PSD is
attached.

### 5. Recycle Solver Method

```
RECYCLE_METHOD: <method> | <option>=<value>, ...
```

Optional. `RECYCLE_SOLVER:` is an alias. Valid methods are:

- `WEGSTEIN` - Default accelerated fixed-point tear solver
- `BROYDEN` - Quasi-Newton secant tear solver
- `DIRECT` - Direct substitution, useful as a conservative diagnostic

The option block is optional and method-specific. Omitting it preserves the
solver defaults:

| Method | Option | Default | Meaning |
|---|---|---:|---|
| `WEGSTEIN` | `max_acceleration` | `-5` | Lower bound applied to the Wegstein acceleration factor; more-negative values permit stronger extrapolation |
| `WEGSTEIN` | `stagnation_iterations` | `8` | Consecutive non-improving evaluations before direct-substitution fallback |
| `WEGSTEIN` | `fallback_damping` | `1` | Direct-substitution damping after fallback, in `(0, 1]` |
| `BROYDEN` | `stagnation_iterations` | `8` | Consecutive non-improving evaluations before direct-substitution fallback |
| `BROYDEN` | `divergence_factor` | `10` | Residual growth factor that resets the inverse-Jacobian estimate |
| `BROYDEN` | `fallback_damping` | `1` | Direct-substitution damping after fallback, in `(0, 1]` |
| `DIRECT` | `damping` | `1` | Direct-substitution damping, in `(0, 1]` |

For example, a high-recycle loop may permit stronger Wegstein extrapolation:

```
RECYCLE_METHOD: WEGSTEIN | max_acceleration=-20
```

`max_acceleration` must be between `-100` and `0`, iteration counts must be
positive integers, and `divergence_factor` must be greater than one. Unknown
or inapplicable options are validation errors rather than being ignored.

Recycle loops use Aspen-like tear variables by default: total molar flow,
component molar flows, pressure, and molar enthalpy. The solver automatically
chooses one tear stream per recycle loop, but you can override the location:

```
TEAR_STREAMS: <stream-id>, <stream-id>
RECYCLE_TRACE_TOLERANCE: <molar-flow-fraction>
```

`RECYCLE_TRACE_TOLERANCE` skips component-flow residuals when the component is
trace on both sides of a tear; if omitted, it defaults to one hundredth of the
requested recycle tolerance.

`RECYCLE_TEAR_STREAMS:` and singular `TEAR_STREAM:` are aliases for
`TEAR_STREAMS:`. `TRACE_TOLERANCE:` is an alias for
`RECYCLE_TRACE_TOLERANCE:`.

When a stream listed in `TEAR_STREAMS` includes normal stream specifications,
those specifications are used as the initial recycle guess. The stream remains
calculated by its source unit after each recycle evaluation.

For each recycle block, the solver evaluates acyclic upstream units once and
caches their boundary streams during iteration. Only units whose results depend
on that block's tear variables are repeated. Downstream-only units are deferred
until the final result pass. The resolved `invariant_units`, `repeated_units`,
and `deferred_units` are included in recycle result diagnostics.

```
STREAM Recycle : SEP-1.vap -> MIX-1.recycle
    T = 35 [C]
    P = 1 [bar]
    F_mass = 380 [kg/h]
    w = solvent:0.98, water:0.02
```

### 6. Components Section

```
COMPONENTS:
    <PFD symbol> | <lookup identifier> | <properties>
    <PFD symbol> | <lookup identifier> | <properties>
    ...
```

Each component line has:
- **PFD symbol**: Process-local alias used by streams, reactions, unit
  parameters, and results. It never participates in property lookup.
- **lookup identifier**: The sole external identity submitted to the property
  system. It may be a unique name, canonical database symbol, unique formula,
  CAS Registry Number, SMILES, InChI, InChIKey, or another uniquely resolvable
  identifier. InChI is converted locally through its molecular graph; InChIKey
  is matched against a local structure index or an online provider because the
  hash itself is not reversible. Ambiguous molecular formulas are rejected.
  A formula with one known local identity selects that identity; an otherwise
  unambiguous formula creates a custom component and emits a warning. When text
  is valid as both a molecular formula and SMILES, formula interpretation takes
  precedence; use a structurally explicit SMILES when structure lookup is
  intended.
- **properties**: Comma-separated `key=value` pairs

Optional properties are looked up if not specified. If a property is specified
in the PFD and the same value is found in the local, cached, or online property
system, the PFD value takes precedence.

For example, each of these alternatives uses `solvent` only inside the PFD
while resolving the same external compound through a different identifier:

```pfd
    solvent | Acetone
    solvent | 67-64-1
    solvent | CC(=O)C
```

`CAS=...`, `formula=...`, and `SMILES=...` in the third field are property
overrides; they do not replace the second-field lookup identifier.

Component property names form a closed schema. Unknown names are fatal parse
errors, with a `Did you mean ...?` suggestion when a close supported name can
be identified; they are never silently ignored.

Property names are matched case-insensitively. Common aliases are
`molecular_weight` for `MW`, `unifac` for `UNIFAC`, `T_rho` or `rho_T_K` for
`rho_T`, and `vapor_dimerization` for `VDM`. Serialization emits the canonical
spellings shown below. `Cps`/`solid_cp`, `rhos`/`solid_density`, `Vms`/
`solid_molar_volume`, and `solid_form` are aliases for the corresponding solid
property fields. `type` and `phase_model` are aliases for `phase_behavior`
inside the `COMPONENTS:` section. `dipole` and `dipole_D` are aliases for
`dipole_moment`. `eta_HOC` and `HOC_association_parameter` are aliases for
`HOC_eta`.

Identity and structure fields:
- `formula` - Molecular formula
- `CAS` - CAS Registry Number
- `SMILES` - SMILES string, used for identity and optional UNIFAC group inference
- `UNIFAC` - UNIFAC groups, either plus form such as `1CH3+2CH2+1OH` or dict form such as `{CH3:1,CH2:2,OH:1}`
- `henry_Hcp` - Pure-water Henry solubility constant at 298.15 K
  [mol/(m3*Pa)]. Overrides the bundled CAS-keyed Henry database value.
- `henry_B` - Henry temperature coefficient `d(ln Hcp)/d(1/T)` [K]. May
  independently override the database coefficient; if unavailable, `Hcp` is
  held constant with temperature and the simulator reports a warning when a
  Henry aqueous context uses the component.
- `henry_Tmin`, `henry_Tmax` - Optional full-quality temperature bounds [K]
  for the supplied Henry correlation. Missing bounds default to 278.15 and
  323.15 K when `henry_B` is available, or 298.15 K when it is not.
- `henry_Vinf` - Optional aqueous infinite-dilution partial molar volume
  [cm3/mol] used by the Krichevsky-Kasarnovsky pressure correction.
- `henry_Vinf_uncertainty` - Optional standard uncertainty of `henry_Vinf`
  [cm3/mol].

The bundled values are resolved from Sander Henry database version 5.0.0.
Seawater and other incompatible solvent/context-specific values are excluded;
source agreement, provenance, selected rows, and statistical outliers are
auditable in `data/henry_constants.sqlite`. See `docs/henry_database.md`.

Critical and phase-change fields:
- `MW` - Molecular weight [g/mol]
- `Tc` - Critical temperature [K]
- `Pc` - Critical pressure [bar]
- `Vc` - Critical volume [cm3/mol]
- `Zc` - Critical compressibility factor
- `omega` - Acentric factor
- `mc_c1`, `mc_c2`, `mc_c3` - Optional Mathias-Copeman pure-component alpha
  parameters for cubic EOS `SRK-MC`, `PR-MC`, or an explicit `PSRK` component
  bundle override. `SRK-MC` first uses compatible unstarred PSRK-2005
  coefficients (resolved `Tc` and `Pc` must each agree within 1%), then the
  CAS-keyed ChemSep table. Missing constants fall back to the parent
  Boston-Mathias alpha with a warning. In `PSRK`, an unavailable/starred
  published critical bundle instead uses normally resolved `Tc`, `Pc`, and
  `omega` with generalized Soave alpha unless explicit MC coefficients are
  supplied.
- `kappa1`, `kappa2`, `kappa3` - Optional PRSV pure-component alpha parameters
  for cubic EOS `PRSV1` or `PRSV2`. PFD values override
  `data/prsv_parameters_cas.json`; `PRSV1` uses `kappa1` only, while `PRSV2`
  accepts `kappa1` alone and defaults missing `kappa2`/`kappa3` to zero. If
  no fitted PRSV parameters are available, the method uses PRSV kappa0-only
  alpha and emits a warning.
- `twu_l`, `twu_m`, `twu_n` - Optional Twu pure-component alpha parameters for
  cubic EOS `SRK-TWU` or `PR-TWU`. PFD values override the CAS-keyed Twu JSON
  tables. All three alpha parameters are required; incomplete PFD Twu alpha
  overrides are ignored with a warning rather than mixed with table values.
- `twu_c` - Optional Twu volume-translation parameter parsed for future
  support, but currently ignored with a warning because volume translation is
  not applied.
- `Tb` - Normal boiling point [K]
- `Tt` - Triple-point temperature [K]
- `Pt` - Triple-point pressure [bar]
- `Tm` - Melting point [K]
- `Hvap` - Heat of vaporization [kJ/mol], usually at `Tb`; when a temperature-dependent value is needed, the resolver can Watson-scale it using `Tb` and `Tc`
- `Hfus` - Heat of fusion [kJ/mol]
- `dipole_moment` - Permanent gas-phase molecular dipole [Debye]. An explicit
  PFD value is authoritative and bypasses CCCBDB lookup and optional quantum
  calculations. The value must be finite and nonnegative.
- `R` - Conventional mass-weighted molecular radius of gyration [angstrom].
  `Rg`, `R_g`, and `radius_of_gyration` are aliases.
- `R_prime` - Thompson's modified mean radius `R'` [angstrom], used by the
  Hayden-O'Connell second-virial provider. `R_HOC`,
  `modified_radius_of_gyration`, and `thompson_radius_of_gyration` are aliases.
  When either radius is absent,
  both are evaluated from the same GFN2-xTB optimized geometry; a newly
  generated geometry is cached before radius evaluation. These computed
  radii carry quality `0.85`, consistent with approximately 4% MAE.
- `HOC_eta` - Optional finite nonnegative pure/self association parameter for
  the Hayden-O'Connell provider. It overrides the component's tabulated group
  value and participates in the pure HOC molecular-parameter calculation. It
  does not replace an unlike-pair `eta`; use an HOC interaction row for that.
  It is ignored with a warning when no active thermodynamic scope uses HOC.
- `rho` - Liquid mass-density reference [kg/m3], interpreted at `rho_T` if supplied, otherwise 298.15 K; this is used as a Rackett fit source when critical data is available
- `rho_T` - Temperature for `rho` [K]
- `rho_solid` - Explicit constant solid mass density [kg/m3]
- `Vm_solid` - Explicit constant solid molar volume [m3/kmol]. If both
  `rho_solid` and `Vm_solid` are supplied, they must agree with `MW` within 2%.
- `solid_material_form` - Optional `unspecified`, `crystalline`, `glass`,
  `hydrate`, or `solvate` selector. `amorphous` and `vitreous` normalize to
  `glass`.
- `solid_polymorph` - Optional free-text polymorph or allotrope selector. A
  form-labelled source is not guessed when no default form is available.
- `phase_behavior` - `conventional` by default; `permanent_solid` for an
  explicit process-model assertion that the component never enters vapor or
  liquid equilibrium; or `conventional_with_solid` for a conventional fluid
  component that may also have an explicitly assigned solid flow.
  `type=permanent_solid` is accepted as a compact alias, and `three_phase` is
  an alias for `conventional_with_solid`. The latter marker does not perform a
  freezing or SLE calculation. These settings are not inferred from
  `phase_at_STP`. `soluble_solid` is reserved for future
  dissolving/precipitating-component and SLE support and is not yet accepted.
- `particle_diameter` - Optional representative solid particle diameter [m]
  for a `permanent_solid` or `conventional_with_solid` component.
  `particle_diameter_m` is an alias.
- `particle_sphericity` - Optional solid-particle sphericity in `(0, 1]`.
  `sphericity` is an alias and the runtime default is 1.0. Particle defaults
  are invalid on a `conventional` component in the current model.
- `particle_size_distribution` / `PSD` - Optional normalized particle-size
  default for a `permanent_solid` or `conventional_with_solid` component. It
  may contain representative diameters and fractions, or a lognormal or
  Weibull/Rosin-Rammler analytical distribution, with a fraction basis as
  described above.
- `uniquac_r`, `uniquac_q` - UNIQUAC pure-component volume and surface-area parameters; if supplied, they take precedence over the built-in CAS-keyed r/q table and UNIFAC-based estimation
- `VDM` - Atomic vapor-dimerization parameter bundle for any component,
  including non-acids. It requires `delta_H` [J/mol dimer] and `delta_S`
  [J/mol/K]. The explicit aliases `delta_H_J_per_mol` and
  `delta_S_J_per_mol_K` are also accepted. A supplied bundle opts the component
  into association and takes precedence over curated acid data and the generic
  monoacid fallback. It is active only for `*-VDM` thermodynamic methods; other
  methods ignore it with a warning.

Formation, entropy, and combustion fields:
- `Hf`, `Gf`, `S` - Gas-standard enthalpy of formation [kJ/mol], Gibbs energy of formation [kJ/mol], and molar entropy [J/mol-K]
- `Hf_liquid`, `Gf_liquid`, `S_liquid` - Liquid-standard formation and entropy fields
- `Hf_solid`, `Gf_solid`, `S_solid` - Solid-standard formation and entropy fields
- `Hcomb` - Net heat of combustion [kJ/mol]
- `Hcomb_gross` - Gross heat of combustion [kJ/mol]

Heat-capacity and vapor-pressure fields:
- `Cp_coeffs` - Ideal-gas heat capacity polynomial coefficients, e.g. `Cp_coeffs=[33, 0.01, 0, 0]`
- `Cp_liquid` - Constant liquid heat capacity [J/mol-K]
- `Cp_solid` - Constant solid heat capacity [J/mol-K]. An explicit PFD value is
  authoritative and unbounded; non-PFD stored values are narrow 298.15 K
  reference points.
- `antoine_A`, `antoine_B`, `antoine_C` - Antoine coefficients for `log10(P_bar) = A - B/(C + T_C)`
- `antoine_Tmin`, `antoine_Tmax` - Antoine validity range [K]
- `antoine_source` - Text label for the Antoine data source

Other flags:
- `phase_at_STP` - `gas`, `liquid`, `solid`, or `unknown`
- `critical_properties_unavailable` - `true` or `false`

`Tt` and `Pt` are independent PFD overrides but are treated as an atomic source
pair: supplying only one does not retain the other value from CoolProp or a
hydrated component record. When both are supplied, they anchor the lower end of
canonical vapor-pressure resolution when physically eligible.

An Antoine override is also atomic: all of `antoine_A`, `antoine_B`,
`antoine_C`, `antoine_Tmin`, and `antoine_Tmax` must be supplied together.

Example:
```
COMPONENTS:
    H2O    | Water   | MW=18.02, Tc=647.1, Pc=220.6, Tt=273.16, Pt=0.00611657
    C2H5OH | Ethanol | MW=46.07, Tc=513.9, Pc=61.4, Tb=351.44, Hvap=38.56, dipole=1.44
    MIBK   | Methyl isobutyl ketone | CAS=108-10-1, SMILES=CC(C)CC(=O)C, UNIFAC=2CH3+1CH+1CH2+1CH3CO
    ACETONE | Acetone | CAS=67-64-1, VDM={delta_H:-50000,delta_S:-125}
    SALT | Sodium chloride | CAS=7647-14-5, type=permanent_solid, solid_material_form=crystalline, particle_diameter=0.0002, particle_sphericity=0.85
```

##### Permanent-Solid Semantics

`permanent_solid` is a modeling declaration, not a phase-equilibrium result.
The declared component never melts, dissolves, or vaporizes and therefore does
not require `Tm`, `Hfus`, `Tb`, `Hvap`, `Psat`, critical properties, acentric
factor, UNIFAC groups, or fluid interaction parameters. Molecular weight and
solid heat capacity are required when their corresponding balances are used;
solid molar volume or density is required for bulk-volume reporting.

Permanent solids are removed before every fluid flash. Reported phase
fractions use the total-stream mole basis:

```text
vapor_fraction + liquid1_fraction + liquid2_fraction + solid_fraction = 1
```

For a solid-bearing stream, `fluid_vapor_fraction` is also reported as the
vapor fraction of only the conventional-component subtotal. A dry-solid stream
has no `fluid_vapor_fraction`. Solid component flows are authoritative and
particle defaults propagate unchanged; no settling, voidage, slurry rheology,
or particle-size distribution is implied.

### 7. Property Correlations Section

Portable temperature-dependent property correlations can be supplied in an
optional `PROPERTY_CORRELATIONS:` section after `COMPONENTS:`. Lines use:

```
PROPERTY_CORRELATIONS:
    <component>.<property> | equation=<equation>, Tmin_K=<K>, Tmax_K=<K>, A=<...>, B=<...>
```

Supported property keys are the resolver-backed correlation keys:
- `Psat` - Vapor pressure [bar]
- `Hvap` - Heat of vaporization [kJ/mol]
- `Cpl` - Liquid heat capacity [J/mol-K]
- `Cpg` - Ideal-gas heat capacity [J/mol-K]
- `Cps` - Solid heat capacity [J/mol-K]
- `rhol` - Liquid mass density [kg/m3]
- `rhos` - Solid mass density [kg/m3]
- `mug` - Vapor viscosity; see property-resolution documentation for expected units
- `mul` - Liquid viscosity; see property-resolution documentation for expected units
- `sigma` or `surface_tension` - Pure-component liquid-vapor/air surface tension [N/m]

Equation compatibility by property:

| Property | Accepted equations |
|---|---|
| `Psat` | `poly_x`, `exp_poly_x`, `reduced_vapor_pressure`, `psat_mercury`, `canonical_psat`, `canonical_psat_af`, `canonical_psat_ag`, `canonical_psat_ah` |
| `Hvap` | `poly_x`, `exp_poly_x`, `reduced_hvap_log`, `dippr_eq106`, `eq106` |
| `Cpl`, `Cpg` | `poly_x`, `exp_poly_x`, `shomate` |
| `Cps` | `poly_x`, `shomate`, `perry_151` |
| `rhol` | `poly_x`, `exp_poly_x`, `density_reference` |
| `rhos` | `poly_x`, `exp_poly_x`, `density_reference` |
| `mug`, `mul` | `poly_x`, `exp_poly_x`, `poly_tp`, `exp_poly_tp`, `dippr_eq101`, `viscosity_exp_rhor` |
| `sigma`, `surface_tension` | `poly_x`, `exp_poly_x`, `dippr_eq106`, `eq106`, `vdi_ppds_11`, `constant`, `constant_surface_tension`, `surface_tension_reference`, `jasper`, `jasper_lange`, `somayajulu`, `somayajulu_revised`, `refprop_sigma`, `refprop`, `refprop_surface_tension` |

The general equation forms are:

- `poly_x` - `A + B*x + C*x^2 + ...`, with `x=(T-298.15 K)/100`
- `exp_poly_x` - `exp(A + B*x + C*x^2 + ...)`
- `poly_tp` - `A + B*x + C*p + D*x^2 + E*x*p + F*p^2`, with
  `p=P_bar-P_ref_bar`
- `exp_poly_tp` - The exponential of the `poly_tp` expression
- `shomate` - `A + B*t + C*t^2 + D*t^3 + E/t^2`, with `t=T/1000`
- `perry_151` - `4.184*(A + B*T + C/T^2 + D*T^2)`, the Perry
  solid-heat-capacity form
- `dippr_eq101` - `exp(A + B/T + C*ln(T) + D*T^E)`
- `dippr_eq106` or `eq106` -
  `A*(1-T/Tc)^(B+C*Tr+D*Tr^2+E*Tr^3)`, with `Tr=T/Tc`
- `reduced_hvap_log` - `exp(A + B*ln(tau) + C*tau + D*tau^2)`, with
  `tau=1-T/Tc`
- `reduced_vapor_pressure` - Four-coefficient reduced vapor pressure using
  `Tc_K` and `Pc_bar` or `Pc_Pa`
- `psat_mercury` - Six-coefficient Huber-Laesecke-Friend mercury form using
  `Tc_K` and `Pc_bar` or `Pc_Pa`
- `canonical_psat` or `canonical_psat_af` - Canonical A-F vapor-pressure form
  `ln(Psat/bar) = A + B/T + C*ln(T) + D*T + E*T^2 + F*T^5`;
  coefficients `A` through `F` are required.
- `canonical_psat_ag` - Adds `G*T^3` to the A-F form; coefficients `A`
  through `G` are required.
- `canonical_psat_ah` - Adds both `G*T^3` and
  `H*((T/Tc)^inverse_power - 1)`; coefficients `A` through `H` are required,
  and `inverse_power` must be `-3`, `-5`, or `-7`.
- `viscosity_exp_rhor` - Dilute exponential term plus a reduced-density
  pressure increment; coefficients are `A`-`F` and `x`/`y` (or `X`/`Y`)
- `density_reference` - Single density point for `rhol`, using `rho_kg_m3` and
  `T_ref_K`; inline `rho`/`rho_T` normally generates this form
- Surface-tension-specific forms are `constant` and its aliases (`sigma` or
  `value`), `jasper`/`jasper_lange` (`a`, `b`), `somayajulu` and its revised
  form (`A`, `B`, `C`, `Tc`), REFPROP forms (`Tc`, `sigma0`-`sigma2`,
  `n0`-`n2`), and `vdi_ppds_11` (DIPPR-106 coefficients)

Coefficients may be written as top-level `A=...`, `B=...` entries or as
`coefficients={A:...,B:...}`. Coefficient names are case-sensitive. Names that
are not top-level `A`-`H`, `x`, or `y` fields, such as Jasper's lowercase
`a`/`b` and REFPROP's `sigma0`/`n0`, belong in the `coefficients={...}` map.

Common optional fields are `Tmin_K`, `Tmax_K`, `Pmin_bar`, `Pmax_bar`,
`P_ref_bar`, `T_ref_K`, `Tc_K`, `Tb_K`, `Pc_bar`, `Pc_Pa`, `quality`,
`quality_note`, `source`, `selected_model`, and `units_note`. `Tmin`, `Tmax`,
`Tc`, `Tb`, `Pc`, `T_ref`, and `rho` are accepted field aliases for their
canonical unit-suffixed names. `Pmin_bar` and `Pmax_bar` bound pressure-specific
viscosity fits; such fits are not pressure-corrected a second time.

For `Cpl`, an omitted `Tmin_K` defaults to the resolved melting point and an
omitted `Tmax_K` defaults to the resolved normal boiling point. If the
corresponding phase point is unavailable or would make the completed interval
invalid, the fallback is 273.15 K for `Tmin_K` or 1500 K for `Tmax_K`.
Explicit endpoints always take precedence.

For `Cps`, an omitted `Tmax_K` defaults to the resolved melting point. An
omitted `Tmin_K` defaults to `min(100 K, 0.8*Tm)`, which keeps the lower bound
below low melting points. If the melting point is unavailable, the fallbacks
are 100 K and 1500 K. Explicit endpoints always take precedence. These
phase-derived bounds describe the assumed range of a provided correlation;
source correlations with known ranges should declare them explicitly.

Property keys, correlation fields, equation names, and coefficient names are
validated during parsing. Unknown names are fatal errors and include a close
match suggestion when one is available.

PFD-provided correlations take precedence over lookup sources for the same
property. Scalar `Hvap` takes precedence as the Watson reference when `Tb` and
`Tc` are available. Scalar `rho` is treated as a density reference and takes
precedence by fitting a Rackett liquid-volume path before Perry or online
density sources.

For `canonical_psat`, the canonical target domain runs from the selected Psat
lower bound through the component `Tc`. A row is used directly as the final
canonical curve when its declared range covers that entire interval:
`Tmin_K <= target Tmin` and `Tmax_K >= Tc`. Exact equality is not required.
Omitted `Tmin_K` defaults to the selected lower bound; omitted `Tmax_K` uses
`Tc_K`, then the component `Tc`. A narrower row remains the highest-priority
hard-pinned segment over only its declared range, and canonical completion
fills any uncovered ranges. Optional `Tc_K`, `Pc_bar` (or `Pc_Pa`), and `Tb_K`
record explicit anchors and must agree with the corresponding component values
when the row is used directly. All PFD Psat overrides have quality `1.0`.

Example:
```
PROPERTY_CORRELATIONS:
    MIBK.Psat | equation=canonical_psat_ag, Tmin_K=165.0, Tc_K=571.0, Pc_bar=32.7, Tb_K=389.8, A=12.3, B=-4200.0, C=-1.1, D=0.002, E=-1e-6, F=2e-14, G=2e-10
    MIBK.Hvap | equation=poly_x, Tmin_K=300, Tmax_K=420, A=34.0, B=-6.0
    MIBK.Cpl  | equation=poly_x, Tmin_K=250, Tmax_K=450, coefficients={A:150.0,B:12.0}
    MIBK.Cps  | equation=perry_151, Tmin_K=150, Tmax_K=300, A=10.0, B=0.01, C=0.0, D=0.0
    MIBK.rhol | equation=poly_x, Tmin_K=273.15, Tmax_K=350.0, A=800.0, B=-90.0, source="lab fit"
    MIBK.sigma | equation=DIPPR_EQ106, Tmin_K=250, Tmax_K=500, Tc_K=571.0, A=0.07, B=1.2, C=0.0, D=0.0, E=0.0
```

### 8. Interaction Estimation and Parameters

Missing molecular UNIQUAC or NRTL binary interactions may be estimated once
during thermodynamic initialization from a selected UNIFAC-family model. The
fitted molecular parameters are then frozen and reused for the entire
simulation; they are never refitted from a unit state or solver iterate.

```
INTERACTION_ESTIMATION:
    UNIQUAC | source=UNIFDMD, policy=missing_only,
               parameter_order=source, Tmin=293.15 [K], Tmax=423.15 [K]
    H2O/PD23 | model=UNIQUAC, Tmin=303.15 [K], Tmax=393.15 [K]
    UNIQUAC | scope=extraction, source=UNIFNIST,
               Tmin=293.15 [K], Tmax=373.15 [K]
```

Global rows begin with the destination molecular activity model (`UNIQUAC` or
`NRTL`). Pair-specific rows begin with a PFD component pair and require
`model=UNIQUAC` or `model=NRTL`. Pair rows inherit the global rule and may
override `source`, `Tmin`, `Tmax`, `T_ref`, `do_not_extrapolate`, and, for
NRTL, `alpha`.
Without `scope=...`, a rule belongs to the global thermodynamic context.
Records in different scopes may repeat the same model and component pair.
Within one scope, duplicate global rules or duplicate pair rules are errors.
A scope with `inherit=...` may inherit its parent's estimation rule and
override it locally.

Supported sources are `UNIFAC`, `UNIFAC2`, `UNIFDMD`, `UNIFM2`, and
`UNIFNIST`. `policy=missing_only` estimates only pairs absent from the bundled
database. `policy=always` freezes an estimate even when a bundled molecular
interaction exists; an explicit PFD `INTERACTION_PARAMETERS` row remains
authoritative. `parameter_order=source` is currently required. Precedence is
explicit PFD parameters, then an `always` estimate, bundled experimental
parameters, a `missing_only` estimate, and finally the model's ideal
residual-interaction fallback.

The estimator fixes resolved UNIQUAC `r`/`q` values or the selected NRTL
nonrandomness parameter, generates binary source-model `ln(gamma)` targets
over composition and temperature, and regresses the complete source activity
coefficients. Original UNIFAC sources fit one directional energy parameter in
each direction. Modified sources fit three coefficients in each direction:

```
UNIQUAC: ln(tau_ij) = a_ij + b_ij/T + c_ij*h(T,Tref) + d_ij*T + e_ij*T^2
NRTL:    tau_ij = c_ij + d_ij/T + e_ij*h(T,Tref) + f_ij*T + g_ij*T^2
h(T,Tref) = (Tref - T)/T + ln(T/Tref)
```

For NRTL, `alpha=0.3` is the default and may be overridden globally or for a
specific estimated pair:

```
INTERACTION_ESTIMATION:
    NRTL | source=UNIFDMD, policy=missing_only,
            parameter_order=source, alpha=0.3,
            Tmin=293.15 [K], Tmax=423.15 [K]
    H2O/PD23 | model=NRTL, alpha=0.2, Tmin=303.15 [K], Tmax=393.15 [K]
```

`Tmin` and `Tmax` define the regression/calibration domain. Frozen parameters
continue to evaluate outside that range by default. With
`do_not_extrapolate=true`, each generated binary record instead clamps its
interaction evaluation temperature to the inherited or pair-specific range.
Fit source, range, sample count, and
`ln(gamma)` error metrics are retained as thermodynamic provenance.

#### Explicit Interaction Parameters

Binary interaction parameters can be supplied in an optional
`INTERACTION_PARAMETERS:` section after `COMPONENTS:`. Lines use:

```
INTERACTION_PARAMETERS:
    <component1>/<component2> | model=<model>, <parameter>=<value>, ...
```

Component references may use PFD symbols, component names, or CAS numbers for
components present in the PFD. User-supplied interaction parameters override
database parameters for the same pair and model. For EOS `k_ij` rows with
`Tmin_K`/`Tmax_K`, the override applies only inside that temperature range; use
an unranged row for an all-temperature override.

`scope=<name>` assigns a row to a named thermodynamic scope. A row without a
scope belongs only to `global`, unless a child scope explicitly inherits that
context. Duplicate pair/model records are allowed across different scopes but
remain errors within the same scope. `LIQUID_VISCOSITY` records follow the same
scope and inheritance rules.

Each interaction model has a closed parameter schema. Unknown parameter names
are fatal parse errors, with a close match suggestion when one is available.

Supported models and fields:
- `NRTL` - `alpha` or `alpha12`, plus either scalar energy fields `a12`/`a21`
  [cal/mol] (`a12_cal_per_mol`/`a21_cal_per_mol` are aliases) or direct tau
  fields `tau12_c`, `tau12_d`, `tau12_e`, `tau12_f`, `tau12_g`, `tau21_c`,
  `tau21_d`, `tau21_e`, `tau21_f`, `tau21_g`, and optional
  `tau_tref`/`tref`. The direct form is
  `tau = c + d/T + e*((Tref-T)/T + ln(T/Tref)) + f*T + g*T^2`; omitted
  `d`, `e`, `f`, and `g` coefficients default to zero. Optional
  `do_not_extrapolate=true` requires `Tmin_K` and `Tmax_K` and clamps only this
  pair's interaction evaluation temperature to that range
- `UNIQUAC` - either scalar energy fields `a12`/`a21` [cal/mol] or direct tau
  fields `tau12_a`, `tau12_b`, `tau12_c`, `tau12_d`, `tau12_e`, `tau21_a`,
  `tau21_b`, `tau21_c`, `tau21_d`, `tau21_e`, and optional
  `tau_tref`/`tref`, `use_q_prime`, and `model_variant`. The direct form is
  `ln(tau) = a + b/T + c*((Tref-T)/T + ln(T/Tref)) + d*T + e*T^2`;
  omitted `b`, `c`, `d`, and `e` coefficients default to zero. Optional
  `do_not_extrapolate=true` requires `Tmin_K` and `Tmax_K` and clamps only this
  pair's interaction evaluation temperature to that range
- `PR` or `SRK` - constant `kij`/`k_ij`, optionally with `Tmin_K`/`Tmax_K`, or
  temperature-dependent `kij_a`, `kij_b`, `kij_c` using
  `k_ij = kij_a + kij_b/T + kij_c*T`; `T_ref_K` is accepted as provenance
  (`t_ref`, `t_ref_k`, `tref`, and `tref_k` are aliases)
- `TSONOPOULOS` - constant dimensionless `kij`/`k_ij` for the selected pair.
  It overrides the built-in fitted binary value, critical-volume correlation,
  or molecular-family default. `TSONOPOULOS-1974` and `TSONOPOULOS-74` are
  model aliases. The row is ignored with a warning unless the scope uses the
  Tsonopoulos vapor provider.
- `HOC` - constant dimensionless unlike-pair `eta`/`eta_ij`. It overrides
  HOC's exact-pair, same-group, group-pair, or correlated solvation rule. The
  row is ignored with a warning unless the scope uses HOC.
- `LIQUID_VISCOSITY` - binary liquid-mixture viscosity excess contribution for
  `ln(mu) = sum(x_i ln(mu_i)) + excess`. Supports multiple non-overlapping
  `Tmin_K`/`Tmax_K` rows for the same pair:
  - `form=grunberg_nissan`, with dimensionless `G` (aliases `G12` and
    `excess_G_over_RT`), contributes `x_i*x_j*G`
  - `form=jouyban_acree`, with `A0_K`, optional `A1_K`, `A2_K`, contributes
    `x_i*x_j/T * (A0_K + A1_K*(x_i-x_j) + A2_K*(x_i-x_j)^2)`
  - `form=excess_poly`, with `A`-`F` and optional `T_ref_K`, contributes
    `x_i*x_j*(A + B*d + C*d^2 + D*t + E*t*d + F*t^2)`, where
    `d=x_i-x_j` and `t=(T-T_ref_K)/100`
  `viscosity_form` is an alias for `form`; `t_ref`, `t_ref_k`, `tref`, and
  `tref_k` are aliases for `T_ref_K`.
- `VDM` - cross-dimer residual corrections for two distinct components that
  each have an active homodimer model. Requires `delta_H_residual` [J/mol] and
  `delta_S_residual` [J/mol/K]. The explicit aliases
  `delta_H_residual_J_per_mol` and `delta_S_residual_J_per_mol_K` are also
  accepted. These residuals replace the database cross-residual bundle and are
  added to the arithmetic means of the two homodimer `delta_H` and `delta_S`
  values. Cross-dimer entropy also includes the statistical `R*ln(2)` term for
  the two distinguishable pairings, making the uncorrected combining rule
  `K_AB = 2*sqrt(K_AA*K_BB)`. `VAPOR_DIMERIZATION` is accepted as a model-name
  alias.

`PENG-ROBINSON`, `PR-BM`, and `PENG-ROBINSON-BM` interaction rows normalize to
`PR`; `RKS`, `RKS-BM`, `SRK-BM`, and the long Soave names normalize to `SRK`.
`VISCOSITY`, `LIQUID-VISCOSITY`, and related mixture-viscosity spellings
normalize to `LIQUID_VISCOSITY`. Every model also accepts an optional `comment`.

Example:
```
INTERACTION_PARAMETERS:
    EtOH/H2O | model=NRTL, alpha=0.3, tau12_c=-0.801, tau12_d=246.2, tau21_c=3.458, tau21_d=-586.1
    EtOH/H2O | model=PR, k_ij=-0.078
    EtOH/H2O | model=TSONOPOULOS, kij=0.08
    EtOH/H2O | model=HOC, eta=1.40
    CO2/H2S | model=PR, kij=0.5, Tmin_K=290, Tmax_K=310
    EtOH/H2O | model=LIQUID_VISCOSITY, form=grunberg_nissan, G=0.81
    ACETONE/EtOH | model=VDM, delta_H_residual=1000, delta_S_residual=-2.5
```

### 9. Streams Section

```
STREAM <name> : <from> -> <to>
    <property> = <value> [<unit>]
    ...
```

External endpoints may be shortened without losing the arrow's direction:

```pfd
# Equivalent to FEED -> D-100.feed
STREAM ColumnFeed : -> D-100.f

# Equivalent to F-100.vapor_out -> PRODUCT
STREAM VaporProduct : F-100.v

# A trailing arrow is also accepted for an explicit product shorthand
STREAM LiquidProduct : F-100.l ->
```

A route with no arrow is always a unit outlet flowing to `PRODUCT`; a leading
arrow always means `FEED` flowing to the following unit inlet. Feed properties
remain indented beneath the stream exactly as in the full form.

Stream definition:
- **name**: Stream identifier
- **from**: Source (`FEED` for input streams, or `<unit>.<port>`)
- **to**: Destination (`PRODUCT` for output streams, or `<unit>.<port>`)

Stream properties:
- `T` - Temperature. `C`/`celsius`, `K`, and `F`/`fahrenheit` are accepted;
  a unitless value below 200 is interpreted as Celsius.
- `P` - Pressure. Accepted units are `bar`, `barg`, `atm`, `kPa`, `Pa`, `MPa`,
  `psi`, `psia`, `mmHg`, and `torr`; unitless pressure is bar.
- `F`, `flow`, or `molar_flow` - Molar flow. Supported conversions are
  `kmol/h` (default), `mol/h`, `mol/s`, and `kmol/s`. `F` with a unit
  containing `kg`, `lb`, or `mass` is treated as mass flow instead.
- `F_mass`, `mass_flow`, or `flow_mass` - Mass flow. Supported conversions are
  kg/h (default), g/h, kg/s, and lb-based mass flow.
- `x`, `mole_fraction(s)`, or `molar_fraction(s)` - Mole fractions in
  `comp1:frac1, comp2:frac2, ...` form.
- `w`, `mass_fraction(s)`, `weight_fraction(s)`, or `wt_fraction(s)` - Mass
  fractions; molecular weights are required for conversion to mole fractions.
- `VF`, `vap_frac`, `vapor_frac`, or `vapor_fraction` - Vapor fraction (0-1).

A feed stream must supply pressure, flow, composition, and either temperature
or vapor fraction. Composition is normalized before simulation. Calculated
inter-unit and product streams normally omit state properties; properties on a
tear stream provide its initial recycle guess.

Example:
```
STREAM Feed : FEED -> HX-1.in
    T = 25 [C]
    P = 1 [bar]
    F = 100 [kmol/h]
    x = H2O:0.6, C2H5OH:0.4

STREAM MassFeed : FEED -> HX-1.in
    T = 25 [C]
    P = 1 [bar]
    F_mass = 2500 [kg/h]
    w = H2O:0.7, CH3OH:0.3

STREAM S1 : HX-1.out -> FLASH-1.in

STREAM Product : FLASH-1.liq -> PRODUCT
```

### 10. Units Section

```
UNIT <id>
    TYPE: <unit_type>
    PORTS:
        <port_id> : <port_type>
        ...
    PARAMS:
        <param> = <value> [<unit>]
        ...
    REACTIONS:
        <reaction_equation> | <kinetics_params>
        ...
```

The compact form puts the type in the header, infers every referenced standard
port, and treats direct assignments as parameters:

```pfd
UNIT FLASH-1 : Flash
    T = 85 [C]
    P = 1 [bar]
```

This is equivalent to the expanded `TYPE:`, `PORTS:`, and `PARAMS:` form.
Serialization currently emits the expanded canonical form. A compact unit may
still declare `PORTS:` explicitly when custom names are needed; once it does,
those declarations are authoritative.

#### Compact Port Names

Port names are case-sensitive as stored, while standard aliases are matched
case-insensitively. Underscores, hyphens, spaces, and collapsed/camel-style
spellings are interchangeable for aliases (`hot_in`, `hot-in`, `hot in`, and
`HotIn`). Conservative fuzzy matching repairs an unambiguous misspelling such
as `vappr`. If a direction has only one possible port, an unfamiliar name is
accepted as that port. Multiport names still error when genuinely ambiguous.
Long and short spellings may be used on compact units or as semantic references
to compatible explicit ports.

| Unit family | Inlet names | Outlet names |
|---|---|---|
| Pump / compressor | Standard input aliases plus `suction`, `suct`, `lp_in`, `gas_in` | Standard output aliases plus `discharge`, `disch`, `hp_out`, `gas_out` |
| Expander | Standard input aliases plus `hp_in`, `expander_in`, `turbine_in` | Standard output aliases plus `exhaust`, `exh`, `lp_out`, `turbine_out` |
| Valve / pipe | Standard aliases plus `upstream`, `entry`, `entrance` | Standard aliases plus `downstream`, `exit` |
| Heater / cooler | Standard aliases plus `cold_in`/`hot_in`, `unheated`/`uncooled` | `heated`, `cooled`, `hot_out`, `cold_out`, and standard product aliases |
| Reactor, equilibrium reactor, CSTR, PFR, packed-bed reactor | `reactant`, `reactants`, `reactor_feed`, `charge`, and standard feed aliases | `effluent`, `products`, `reaction_products`, and standard product aliases |
| Batch reactor | `charge`, `reactant`, `initial_charge`, `addition`, `semi_batch_feed`, arbitrary named additions, and standard feed aliases | `effluent`, `products`, `reaction_products`, and standard product aliases |
| Mixer | Standard aliases plus arbitrary named inlets | `mixed`, `mix`, `blend`, and standard product aliases |
| Splitter | Standard single inlet aliases | `o1`/`out1`/`product1`/`branch1` through 10, plus arbitrary named outlets |
| Heat exchanger | Hot, cold, shell, and tube forms including `hi`, `ci`, `si`, `ti` and `*_inlet` | Corresponding `ho`, `co`, `so`, `to`, `*_outlet`, and `*_product` forms |
| Flash | Standard single inlet aliases | Vapor/gas/top/overhead/vent forms; liquid/bottoms/residue forms |
| Flash3 / three-phase flash | Standard single inlet aliases | Flash vapor aliases; `liquid1_out`/`liquid1`/`liq1`/`l1`; `liquid2_out`/`liquid2`/`liq2`/`l2` |
| Decanter | Standard feed/decanter/settler forms | Light/organic/oil/top forms; heavy/aqueous/water/bottom forms |
| Distillation columns | Feed/input/charge forms plus rigorous named feeds | Distillate/overhead/top/light forms; bottoms/residue/heavy forms; rigorous vapor/liquid distillates and side draws |
| Molecular-sieve dryer | Feed/input/wet/moisture-bearing forms | Dry/dried product forms; adsorbate/water/moisture/waste forms |
| Extractor | Process/solute/carrier feed forms; solvent/extractant forms | Raffinate/carrier-product forms; extract/solvent-rich forms |
| Absorber | Gas/vapor/sour-gas forms; liquid/solvent/lean-amine forms; rigorous named feeds | Treated/clean/sweet gas forms; rich/loaded-solvent and bottoms forms |
| Stripper | Rich/loaded-liquid forms; stripping-gas/steam forms; rigorous named feeds | Overhead/offgas/acid-gas forms; lean-solvent and bottoms forms |

Aliases are direction-aware. For example, `D-100.b` is a valid column stream
source but is rejected as a destination because bottoms is outlet-only. Use
`STREAM Feed : -> D-100.f` for the compact column-feed form.

##### Clockwise Numeric Ports

Compact units also accept positive integer port names. Numbering is interpreted
clockwise on a conventional left-to-right flowsheet: left-side inlets precede
right-side outlets. A direction that contradicts the numbered position is an
error.

```pfd
# Mixer: ports 1-3 enter from the left; port 4 leaves on the right.
STREAM A : -> M-1.1
STREAM B : -> M-1.2
STREAM C : -> M-1.3
STREAM Mixed : M-1.4
UNIT M-1 : Mixer

# Two-feed column: lower feed, higher feed, top product, bottom product.
STREAM LowerFeed : -> D-1.1
STREAM HigherFeed : -> D-1.2
STREAM Distillate : D-1.3
STREAM Bottoms : D-1.4
UNIT D-1 : RigorousDistillation
    N_stages = 12
```

For multiple numeric rigorous-column feeds, smaller numbers are lower on the
column. Unless `feed_stages` is supplied explicitly, their stages are spaced
evenly from lower to higher (`feed1:8,feed2:4` for two feeds on 12 stages). With one
feed and three outputs, a mixed condenser maps the clockwise outlets to vapor
distillate, liquid distillate, and bottoms. Without a mixed condenser,
intermediate outputs require matching `side_draws` specifications because a
port position cannot determine withdrawal stage, phase, or rate. They are then
canonicalized to `side_draw1`, `side_draw2`, and so on.

Other fixed layouts follow the same physical order: a flash is feed, vapor,
liquid; Flash3 adds liquid-1 then liquid-2; a decanter is feed, light, heavy;
and absorbers/strippers are lower gas feed, upper liquid feed, upper gas
product, lower liquid product. A heat exchanger uses lower cold inlet, upper
hot inlet, upper hot outlet, lower cold outlet. Splitter numeric outlet names
are canonicalized to `out`, `out2`, etc. and passed to its runtime outlet list
automatically. Mixer numeric inlets become `in1`, `in2`, etc.; rigorous-column
numeric feeds become `feed1`, `feed2`, etc. These canonical names are what
serialization and `.pfr` reporting expose.

#### Unit Types

**Mixing/Splitting:**
- `Mixer` - Combine multiple streams
- `Splitter` - Split stream by ratio

**Pressure Change:**
- `Pump` - Liquid pressure increase
- `Compressor` - Gas compression
- `Expander` - Gas expansion
- `Valve` - Pressure reduction
- `Pipe` - Integrated single-phase pipe pressure drop

**Heat Transfer:**
- `Heater` - Heat addition
- `Cooler` - Heat removal
- `HeatExchanger` - Two-stream heat exchange

**Separation:**
- `Flash` - VLE flash drum
- `Flash3` - Vapor-liquid-liquid flash drum; aliases: `ThreePhaseFlash`, `VLLEFlash`
- `Decanter` - Liquid-liquid separator; aliases: `FlashLLE`, `LLSeparator`, `Settler`
- `Crystallizer` - Pure-solid SLE cooling crystallizer with a retained slurry
  or optional ideal cake/mother-liquor split
- `ShortcutDistillation` - Shortcut distillation column
- `McCabeThieleDistillation` - Binary McCabe-Thiele column with optional latent-heat-corrected operating curves
- `CMODistillation` - Multicomponent stage-by-stage constant-molar-overflow column with total, partial, or mixed condenser
- `RigorousDistillation` - Sparse Newton MESH-equation distillation column with pressure profiles, side draws, total/partial/mixed condensers, and optional top decanter
- `MolecularSieveDryer` - Selective 3A adsorption dryer for trace water removal; alias: `Dryer`
- `ShortcutExtractor` - Shortcut multi-stage LLE extraction; aliases: `Extractor`, `LiquidLiquidExtractor`, `LLE`
- `RigorousExtractor` - Equilibrium-stage LLE extraction; alias: `RigorousLiquidLiquidExtractor`
- `Absorber` - Shortcut gas absorption column; alias: `AbsorptionColumn`
- `Stripper` - Shortcut gas stripping column; alias: `StrippingColumn`
- `RigorousAbsorber` - Equilibrium-stage absorber with MESH equations; alias: `RigorousAbsorptionColumn`
- `RigorousStripper` - Equilibrium-stage gas stripper with MESH equations; alias: `RigorousStrippingColumn`

**Reaction:**
- `Reactor` - Conversion reactor
- `EquilibriumReactor` - Equilibrium reactor
- `CSTR` - Continuous stirred tank with kinetics; alias: `KineticsCSTR`
- `BatchReactor` - Transient stirred batch/semi-batch reactor exposed on a
  continuous-equivalent basis; alias: `KineticsBatch`
- `PFR` - Plug flow reactor with kinetics; aliases: `KineticsPFR`, `PlugFlowReactor`
- `PackedBedReactor` - Fixed-catalyst packed-bed reactor; aliases: `PBR`,
  `KineticsPackedBed`

##### Permanent-Solid Unit Capability

The solid routing layer supports `Mixer`, `Splitter`, `Heater`, `Cooler`,
`HeatExchanger`, ordinary `Flash`, `Crystallizer`, and `Filter`. Feeds and
product sinks may be dry solids, slurries, or fluid/solid multiphase streams.

- A proportional `Splitter` preserves phase inventory and particle defaults;
  component-split mode may explicitly route solid components.
- `Heater` and `Cooler` include permanent-solid sensible enthalpy.
- `HeatExchanger` supports explicit duty or outlet-state specifications.
  Rating and auto-`U` modes reject solid-bearing streams because slurry/powder
  film coefficients are not modeled.
- Ordinary `Flash` sends no permanent solid to its vapor outlet. Its single
  nonvapor outlet retains all permanent solids and one or two liquid phases;
  it does not claim filtration or particle classification.
- `Crystallizer` preserves permanent solids and equilibrates only components
  declared `conventional_with_solid` against a homogeneous liquid mother phase.
- `Filter` performs cycle pressure cake filtration, optional washing, and
  capillary-limited deliquoring. Its capture curve classifies each PSD using
  particle size and sphericity; the captured PSD determines Kozeny–Carman cake
  resistance unless measured resistance is supplied. Specify `cycle_time`,
  `porosity`, `capture_cut_size`, and `P_drop` or `area`. Connections are `in`,
  optional `wash`, `cake`, and `filtrate`. See [filtration.md](filtration.md)
  for units, equations, empirical closures, and model limitations.
  `washing_model=equilibrium` enables adiabatic warm/melt washing with local
  enthalpy/SLE contacts, evolving PSD and porosity, and updated hydraulic
  resistance. See [equilibrium_washing.md](equilibrium_washing.md).

Other registered units reject permanent-solid-bearing inlets explicitly,
including pressure machines and pipes, `Flash3`, decanters/extractors,
distillation, absorbers/strippers, reactors, and `MolecularSieveDryer`.
Solid solutions, co-crystals, polymorph selection, all-solid topology,
particle-size prediction, settling, and slurry transport remain outside
format 1.0. Multiple independent pure solids may coexist with the retained
mother liquor.

#### Port Types

- `inlet` - Material input
- `outlet` - Material output
- `vapor_outlet` - Vapor output
- `liquid_outlet` - Liquid output
- `light_liquid_outlet`, `heavy_liquid_outlet` - Density-classified liquid outputs
- `liquid1_outlet`, `liquid2_outlet` - Two liquid outputs without density labels
- `solid_outlet` - Solid or adsorbate output
- `gas_inlet`, `solvent_inlet` - Specialized material inputs
- `extract_outlet`, `raffinate_outlet` - Liquid-extraction outputs

Ports are material-connection labels; heat and work are unit results rather than
connectable `heat_in`/`heat_out` ports in format 1.0.

#### Parameters

Every unit accepts the common optional parameter `thermo_scope=<name>`. When
omitted, the unit uses `global`. The named scope must be declared earlier in a
top-level `THERMO_SCOPES:` section.

Common parameters by unit type:

**Mixer:**
- `P_out`, `P`, or `pressure` - Optional outlet pressure. If omitted, the mixer
  automatically lets down higher-pressure inlets to the lowest inlet pressure
  before mixing; `P_drop` subtracts from that default pressure.
- `T_out`, `T`, or `temperature` - Optional outlet temperature. When specified,
  heat duty is calculated from the enthalpy balance.
- `Q`, `duty`, or `heat_duty` - Optional heat duty. Values are kW by default
  unless units are supplied. When specified, outlet temperature is solved from
  the enthalpy balance.
- `mode` - `adiabatic` by default. `isothermal` without `T_out` preserves the
  previous behavior of using the flow-weighted mixed inlet temperature.

**Splitter:**
- `outlets` or `outlet_names` - Optional comma-separated outlet port names.
- `split_frac`, `split_frac2`, ... - Legacy positional split fractions; the
  final outlet receives the remainder.
- `split_fracs` or `split_fractions` - Named split fractions, for example
  `product:0.9,purge:0.1`.
- `flow_<outlet>`, `<outlet>_flow`, `F_<outlet>`, or `<outlet>_F` - Direct
  molar outlet flow. Any unspecified outlet receives the remaining flow.
- `mass_flow_<outlet>`, `<outlet>_mass_flow`, `F_mass_<outlet>`, or
  `<outlet>_F_mass` - Direct mass outlet flow.
- `component_splits` - Optional arbitrary component split map for placeholder
  separator behavior, for example
  `ethanol:top=0.9,bottom=0.1; water:top=0.1,bottom=0.9`.
- `P_<outlet>`, `<outlet>_P`, `P_drop_<outlet>`, `T_<outlet>`,
  `<outlet>_T`, `phase_<outlet>` - Optional per-outlet state specs.

**Pump:**

- Specify at most one pressure target: `P_out`/`P`/`pressure`,
  `delta_P`/`dP`/`P_rise`/`pressure_rise`, or
  `pressure_ratio`/`P_ratio`/`ratio`. The default is a 5 bar rise.
- `eta` - Hydraulic efficiency, default `0.75`.
- `eta_mech` - Mechanical efficiency, default `0.95`.
- The inlet must be liquid and the outlet pressure must exceed the inlet
  pressure.

**Compressor / Expander:**

- Specify at most one pressure target: `P_out`/`P`/`pressure`, a positive
  `delta_P`/`dP` (also `P_rise` or `P_drop` as appropriate), or
  `pressure_ratio`/`P_ratio`/`ratio` greater than one. The default pressure
  ratio is 2.
- `eta_isen` or `eta` - Isentropic efficiency; defaults are `0.75` for a
  compressor and `0.80` for an expander.
- `eta_mech` - Mechanical efficiency, default `0.98`.
- `eta_poly` is recognized only to report that polytropic efficiency is not yet
  supported.

**Valve:**

- `P_out` or `P` - Required downstream pressure in practice; if omitted, half
  the inlet pressure is used. The valve performs an isenthalpic flash and
  requires the outlet pressure to be below the inlet pressure.

**Pipe:**
- Models a straight circular pipe by adaptive axial integration of the
  adiabatic energy and one-dimensional momentum balances. Density,
  viscosity, temperature, velocity, Reynolds number, and friction factor are
  recalculated from the active thermodynamic method along the pipe.
- `length` or `pipe_length` - Required pipe centerline length. Supported units
  include `m`, `cm`, `mm`, `km`, `ft`, and `in`.
- Specify exactly one inlet sizing method:
  - `diameter`, `pipe_diameter`, or `diameter_in` - Inside diameter.
  - `velocity`, `target_velocity`, or `inlet_velocity` - Desired inlet mean
    velocity; the inlet diameter is calculated automatically. Supported units
    include `m/s`, `ft/s`, and `km/h`.
- `diameter_out` or `outlet_diameter` - Optional outlet diameter.
  `diameter_profile = smooth` (the default when diameters differ) and `linear`
  use linear diameter interpolation, representing a conical transition.
  `smoothstep` uses a cubic taper with zero slope at both ends; `constant` is
  available when the inlet and outlet diameters are equal.
- `material` or `pipe_material` - Converts common materials to absolute
  roughness. Supported names are `smooth`, `glass`, `plastic`, `pvc`,
  `copper`, `drawn_tubing`, `stainless_steel`, `commercial_steel`,
  `carbon_steel`, `steel`, `wrought_iron`, `galvanized_steel`,
  `galvanized_iron`, `cast_iron`, and `concrete`. The default is
  `commercial_steel` with roughness 0.045 mm.
- `roughness`, `absolute_roughness`, or `epsilon` - Explicit absolute
  roughness. This overrides `material` when both are supplied.
- Elevation can be specified in exactly one of three ways:
  - `orientation = horizontal`, `vertical_up`, or `vertical_down`.
  - `angle` or `inclination` in degrees by default; positive is uphill.
  - `elevation_change` or `delta_z`; positive is uphill.
- `friction_model` - `churchill` (default), `haaland`, `swamee-jain`, or
  `colebrook`. Churchill is continuous through laminar, transitional, and
  turbulent flow.
- `phase_model` - `auto` (default), `single_phase`, or `two_phase`.
  `auto` uses the single-phase model for liquid/vapor states and switches to
  Beggs-Brill when the local state is two-phase. `single_phase` preserves the
  legacy single-phase-only behavior and returns the explicit zero-drop
  two-phase placeholder if two-phase flow is encountered. `two_phase` requires
  a two-phase inlet and forces the Beggs-Brill branch.
- `two_phase_model` - `beggs_brill` (default). Beggs-Brill is an empirical
  gas-liquid pipe correlation using phase densities, viscosities, mass quality,
  liquid-gas surface tension, pipe roughness, diameter, and inclination to
  calculate friction, holdup/elevation, and acceleration pressure-gradient
  terms.
- `surface_tension_method` - `auto` (default), `butler`, `butler-unifac`, or
  `wsd`, passed to the mixture surface-tension calculator for two-phase pipe
  flow. Pure-component two-phase flow uses the pure-component surface-tension
  resolver directly.
- Local energy-coupling controls:
  - `energy_tolerance` / `local_energy_tolerance` /
    `energy_iteration_tolerance` - static enthalpy/velocity iteration
    tolerance [kJ/kmol], default `1e-6`.
  - `energy_max_iterations` / `local_energy_max_iterations` /
    `energy_iteration_max_iterations` - maximum local energy-coupling
    iterations. The default is 12 for single-phase states and one pass for
    two-phase Beggs-Brill states; an explicit value applies regardless of
    phase regime.
- Numerical controls are `relative_tolerance`/`rtol`,
  `absolute_tolerance`/`atol`, `maximum_step`/`max_step`, and
  `profile_points` (2-201).
- Performance results include the integrated friction, gravity, and
  acceleration pressure-drop contributions plus an axial property profile.  In
  two-phase regions, profile rows include vapor fraction, vapor mass quality,
  surface tension, Beggs-Brill flow regime, liquid holdup, no-slip Reynolds
  number, and the two-phase friction multiplier.

**Heater/Cooler:**
- Specify exactly one of `T_out`/`T`, `Q`/`duty`/`heat_duty`, or
  `VF`/`vap_frac`/`vapor_fraction`.
- `P_drop` - Optional nonnegative pressure drop [bar]; default zero.
- The frozen-Henry controls documented under `Flash` are also accepted. This
  keeps a condensing Heater/Cooler and a downstream adiabatic separator on the
  same aqueous standard-state treatment.

**HeatExchanger:**
- Design mode: specify exactly one of `Q`/`duty`/`heat_duty`, one outlet
  temperature (`T_hot_out`, `T_cold_out`, `T_tube_out`, or `T_shell_out`), or
  one outlet vapor fraction (`hot_vapor_fraction`, `cold_vapor_fraction`,
  `tube_vapor_fraction`, or `shell_vapor_fraction`).
- Rating mode: omit outlet targets and specify `UA`, `UA_available`, numeric
  `U` plus `A`, or `A` plus explicit auto-U estimation.
- `U` - Overall heat-transfer coefficient [W/m2-K]. User-supplied values are
  preferred. Use `U = auto`/`estimated` or `estimate_U = true` only for
  preliminary service-class estimates.
- `A` or `area` - Heat-transfer area [m2].
- `UA` or `UA_available` - Overall conductance [W/K by default; kW/K if unit
  is supplied].
- `curve_segments` - Number of heating/cooling-curve segments, default 40.
- `flow_pattern` or `type` - `countercurrent` by default; `cocurrent` is also
  supported.
- `allow_temperature_cross` - Optional override for temperature-cross handling.
- If `U` and `A` are omitted in design mode, the simulator reports only
  `UA_required`; it does not invent a U value.

**Flash:**
- `T` - Temperature
- `P` - Pressure. When omitted with exactly one of `T`, `VF`, or `Q`, the
  flash inherits its inlet pressure.
- `VF`, `vapor_frac`, or `vapor_fraction` - Vapor fraction target [0-1]
- `Q`, `duty`, or `heat_duty` - Heat duty [kW by default]
- `henry_components` - `auto` by default. For water-rich condensates, freezes
  eligible noncondensables and sufficiently dilute solutes onto their
  pure-water Henry standard states for the complete flash solve. Use
  `none`/`off` to disable or provide a comma-separated explicit component list.
- `water_component` - Optional explicit aqueous solvent component.
- `henry_water_mole_fraction_min` / `henry_water_cutoff` - Minimum conservative
  pre-solve condensate water fraction for automatic selection; default `0.95`.
- `henry_dilute_mole_fraction_max` / `henry_dilute_cutoff` - Maximum estimated
  aqueous loading for automatically selected ordinary condensables; default
  `0.001`.
- `henry_pressure_warning_bar` - Pressure threshold above which an uncorrected
  Henry component emits a warning; default `20` bar. Corrected components
  report their pressure penalty and correction factor without this warning.

Specify exactly two flash variables, or specify one of `T`, `VF`, or `Q` and
omit `P` to use the inlet pressure as the second variable. Supported pairs are:
- `T` and `P` - Isothermal-isobaric flash
- `P` and `VF`/`vapor_fraction` - Find temperature for target vapor fraction
- `T` and `VF`/`vapor_fraction` - Find pressure for target vapor fraction
- `P` and `Q`/`duty`/`heat_duty` - Find temperature from heat duty
- `T` and `Q`/`duty`/`heat_duty` - Find pressure from heat duty
- `VF`/`vapor_fraction` and `Q`/`duty`/`heat_duty` - Find pressure and temperature from vapor fraction and heat duty

**Decanter / FlashLLE:**
- Separates a liquid feed into `light` and `heavy` liquid outlet phases using
  an LLE-capable activity-coefficient model (`NRTL`, `UNIQUAC`, `UNIFAC`, etc.).
- Multiple liquid inlets are mixed before decanting.
- Default mode is adiabatic; omit `T` to use the mixed inlet enthalpy state.
- `T`/`temperature` - Optional isothermal operating temperature.
- `P`/`pressure` - Optional operating pressure. Defaults to the lowest inlet
  pressure after any `P_drop`.
- `P_drop` - Optional nonnegative pressure drop [bar].
- `Q`/`duty`/`heat_duty` - Optional heat duty target [kW by default].
- `heavy_component`/`heavy_key` - Optional fallback selector when phase
  densities are unavailable. Normally phases are classified by density.
- Vapor-containing feeds are rejected; use `Flash3`/`ThreePhaseFlash` for VLLE.

**Flash3 / ThreePhaseFlash / VLLEFlash:**
- Vapor-liquid-liquid flash using a VLLE-capable activity-coefficient model
  (`NRTL`, `UNIQUAC`, `UNIFAC`, and gamma-phi variants).
- Specify exactly two of `T`/`temperature`, `P`/`pressure`,
  `VF`/`vapor_frac`/`vapor_fraction`, and `Q`/`duty`/`heat_duty`.
- Supported pairs mirror `Flash`: `T,P`; `P,VF`; `T,VF`; `P,Q`; `T,Q`;
  and `VF,Q`.
- Outlets are `vapor_out`, `liquid1_out`, and `liquid2_out`.
- Optional solver controls: `max_iter`/`flash_max_iter` and
  `tol`/`flash_tolerance`.
- Accepts the same frozen-Henry selection parameters as `Flash` when the
  result has at most one liquid phase. A state that simultaneously requires a
  pure-water Henry phase and a second liquid phase is rejected explicitly;
  coupled Henry/VLLE standard states are not currently implemented.

**Crystallizer:**

- `crystallization_mode` selects `suspension` (the default) or `layer`.
  Layer mode is an **equilibrium endpoint** calculation using the same rigorous
  pure-solid SLE and enthalpy model as suspension mode when used with
  `model=equilibrium`. It requires a feed without solids. All equilibrium crystals are
  assumed to deposit and are harvested through `layer` (alias of `cake`);
  drained liquid leaves through `mother_liquor`. Both outlets must be connected.
  Mother-liquor retention defaults to zero in layer mode; either retention
  specification below can override it. Retention is specified, not predicted.
  Deposited crystals carry no particle-size distribution, diameter, or
  sphericity; `outlet_sphericity` is rejected in layer mode.
  Flow rates represent equivalent throughput over complete harvest cycles.
  This mode does not predict wall growth, layer thickness, cycle duration,
  heat/mass-transfer limitations, impurity trapping, or sweating. Equal endpoint
  temperature, pressure, feed, and retention give the same phase amounts and
  duty as equilibrium suspension with a cake split.

  ```pfd
  UNIT C : Crystallizer
      model = equilibrium
      crystallization_mode = layer
      T = 250 [K]
  STREAM Layer : C.layer -> PRODUCT
  STREAM Mother : C.mother_liquor -> PRODUCT
  ```

- `model` selects `equilibrium` (the default), `MSMPR`, `layer_growth`, or
  `empirical_layer_growth`. The equilibrium
  model retains the existing pure-solid SLE behavior. `MSMPR` enables the
  steady kinetic population-balance model described below.
- One `in`/`solution` inlet is cooled to a specified `T_out`/`Tout`/`T`/`temperature`.
- `P_out`/`Pout`/`P`/`pressure` optionally specifies outlet pressure. Otherwise inlet
  pressure minus optional `P_drop` is used. Do not specify both an absolute
  outlet pressure and `P_drop`.
- Every feed component declared `phase_behavior=conventional_with_solid`
  (alias `three_phase`) participates in pure-solid SLE. Ordinary conventional
  components remain in the liquid, and permanent solids pass through unchanged.
- A declared candidate above its pure-component melting point is retained in
  the liquid without requesting metastable-solid properties. Other subcooled
  candidates continue through the SLE calculation normally.
- The equilibrium equation uses the resolved `Tm` and `Hfus`, integrated
  temperature-dependent liquid and solid heat capacities, liquid activity
  coefficients (or liquid EOS fugacity ratios), and a solid/liquid molar-volume
  pressure correction. Each crystallizing component forms its own pure solid;
  co-crystallization is not modeled.
- In suspension mode, by default the sole `out`/`slurry` outlet retains all
  mother liquor and solid.
- Specifying `mother_liquor_retention` (alias
  `mother_liquor_retention_fraction`) enables `cake` and `mother_liquor`
  outlets. It is the fraction from 0 to 1 of the equilibrium mother liquor
  sent with the cake. Alternatively, `mother_liquor_retention_rate` specifies
  retained mother-liquor mass per mass of conventional crystals in kg/kg.
  The two bases are mutually exclusive, and a mass rate requiring more mother
  liquor than exists is infeasible. All solid is sent to the cake. This is an
  ideal bookkeeping split, not a filtration or centrifuge transport model.
- `equilibrium_tolerance` and `max_iterations` control the equilibrium
  complementarity solve; defaults are `1e-8` and 500. In MSMPR mode,
  `max_iterations` limits the coupled kinetic material-balance root solve and
  `msmpr_tolerance` (default `1e-8` kmol/h) sets the absolute residual
  tolerance. `msmpr_relative_tolerance` optionally sets a tolerance relative
  to the inlet crystallizing-component flow and defaults to zero. The effective
  MSMPR tolerance is the larger of the absolute and relative tolerances.
- `outlet_sphericity` optionally specifies a component-level crystal
  sphericity in `(0, 1]` for the crystallizer outlet. It applies to every
  crystallizing component because the current stream model stores one
  sphericity per solid component rather than per PSD class. Without this
  override, a solid-bearing feed propagates its seed-crystal sphericity;
  newly formed crystals in a clear feed retain the component-level
  `particle_sphericity` default.
  Inert permanent solids preserve their inlet particle properties and PSD,
  including stream-specific sphericity, in either outlet mode.
- The reported duty is the enthalpy difference between the inlet and the
  equilibrium outlet stream or streams. Conventional-solid enthalpy and entropy
  are anchored to the liquid at `Tm` through `Hfus`, consistent with the SLE
  thermodynamic cycle.
- The first implementation assumes one homogeneous liquid mother phase and
  requires some liquid to remain. It supports multiple coexisting pure solids
  but fails explicitly for an all-solid topology rather than substituting an
  approximate result. In equilibrium mode, nucleation, growth, agglomeration,
  breakage, and a predictive PSD are not modeled.
- When global `FLUID_PHASE_MODEL` is `VL(L)E` or `VLLE` and the unit uses an
  activity-coefficient model, the calculated outlet mother liquor is checked
  using that fluid-equilibrium policy. `VL(L)E` uses its local spinodal test
  before escalating to a split calculation; it is not a global stability
  proof. `VLLE` uses the global phase search. If two liquid phases are found,
  a warning states that the crystallizer does not support crystallization
  with LLE present. This diagnostic does not alter the homogeneous-liquid
  crystallization result or split its mother liquor into equilibrium phases.
  The check runs once before the optional cake split, for both equilibrium
  and MSMPR modes. Unit performance records `outlet_lle_check`; outlet phase
  metadata records `crystallizer_lle_check`. A failed phase check produces an
  explicit warning and `checked=false`, rather than claiming no LLE.

**Finite-rate layer growth:**

`model = layer_growth` selects `crystallization_mode = layer` automatically.
It accepts a multicomponent liquid feed with exactly one present component
declared `conventional_with_solid` and at least one other liquid component.
All other components remain liquid, either in the free mother liquor or in
trapped inclusions. This is a planar, quasi-steady thermal model with
time-dependent deposition, bulk composition, and optional bulk cooling.
It is not a spatial transient conduction model.
The layer starts at zero thickness with instantaneous surface nucleation.

```pfd
UNIT C : Crystallizer
    model = layer_growth
    T = 270 [K]
    T_wall = 250 [K]
    cooled_area = 10 [m2]
    film_thickness = 1 [mm]
    growth_time = 6 [min]
    cycle_time = 60 [min]
STREAM Layer : C.layer -> PRODUCT
STREAM Mother : C.mother_liquor -> PRODUCT
```

Design/operating inputs:

- `thermal_mode = isothermal` (default): `T`/`T_out` specifies the controlled
  bulk-liquid and final harvested-product temperature, below the pure-component
  melting point.
- `thermal_mode = cooling`: omit `T`/`T_out`. The initial temperature is the
  liquid inlet temperature; outlet temperature is predicted by integrating the
  free-liquid enthalpy balance. Starting above the pure melting point is allowed.
- `T_wall`: constant solid-side cold-wall temperature, below bulk temperature
  and the pure-component melting point. Wall/coolant resistance upstream of
  this specified surface temperature is outside the model.
- `cooled_area`: planar deposition area in `m2` or `m^2`.
- `film_model = specified` (default): `film_thickness` is the effective
  mass-transfer film thickness in `m`, `mm`, or
  `um`. Optional `thermal_film_thickness` independently specifies the thermal
  film thickness in the same units; it defaults to `film_thickness`.
  These are supplied hydrodynamic parameters, not inferred vessel dimensions.
- `film_model = flat_plate`: supply `plate_length` (along the flow, in `m`,
  `mm`, or `um`) and `liquid_velocity` (`m/s`, `cm/s`, or `m/h`). Omit both film
  thickness inputs. The laminar flat-plate correlation calculates them instead.
- `growth_time`: active deposition duration in `h`, `min`, or `s`.
- `cycle_time`: full cycle duration in those same units, at least `growth_time`.
  Inlet kmol/h times cycle hours defines charge kmol per batch; deposited kmol
  divided by cycle hours defines the harvested flow. No separate batch charge
  or retained seed is added to this balance.

For example, cooling with correlated films and liquid inclusions:

```pfd
UNIT C : Crystallizer
    model = layer_growth
    thermal_mode = cooling
    T_wall = 250 [K]
    cooled_area = 10 [m2]
    film_model = flat_plate
    plate_length = 0.1 [m]
    liquid_velocity = 0.1 [m/s]
    growth_time = 6 [min]
    cycle_time = 60 [min]
    inclusion_max_fraction = 0.1
    effective_diffusivity = 1e-9 [m2/s]
```

The inclusion parameter and diffusivity here are illustrative inputs, not
recommended values for arbitrary compounds. Omitting diffusivity enables the
estimates described below; specifying it bypasses those estimates.

At each integration stage the interface solves pure-solid SLE and the Stefan
balance `integral[T_wall,T_interface](k_s(T) dT)/thickness = q_liquid + N_crystal ΔH_fusion`.
The heat of crystallization uses partial liquid molar enthalpy (including the
thermodynamic model's heat of mixing) and the fusion-consistent solid enthalpy.
Solid density is held at the wall-temperature value; solid conductivity is
resolved and numerically integrated across the wall/interface temperature range.
This is the steady planar Fourier-law integral, including nonlinear conductivity
variation rather than a mean-temperature approximation. The zero-thickness limit
sets the interface to wall temperature and obtains a finite rate from film
mass transfer.

Liquid-film heat transfer uses `h = k_liquid/thermal_film_thickness`. The shared
`mixture_liquid_thermal_conductivity` method uses Li's volume-fraction mixing
rule with harmonic pair conductivities:
`k_mix = sum_i sum_j phi_i phi_j 2/(1/k_i + 1/k_j)`.
Pure conductivities and pure liquid volumes come from existing resolver paths.
The empirical rule does not represent anomalous mixture conductivity.

Selective mass transfer uses a pseudo-binary solvent-blend film equation,
integrating `c D_MS Gamma/(1-x_crystal)` over interface-to-bulk mole fraction;
`Gamma = d ln(a_crystal)/d ln(x_crystal)` along a path with fixed relative
proportions of all noncrystallizing components. Activities and enthalpies are
evaluated for the full mixture. Film properties are evaluated at the
bulk temperature, and thermal diffusion (Soret effect) is omitted. Nonpositive
thermodynamic factors fail explicitly. For binary liquids this reduces to the
zero-solvent-flux Maxwell–Stefan film model. For multicomponent liquids it is
an effective approximation: differential diffusion among noncrystallizing
components and the full Maxwell–Stefan matrix are not resolved.

For each crystal-component/other-component pair, `D_MS` uses Vignes interpolation
of Wilke–Chang infinite-dilution estimates with mole fractions normalized within
that pair. The effective diffusivity is the harmonic average of those pair
values, weighted by the noncrystallizing components' solvent-blend fractions.
This mixture extension is a heuristic closure. Binary feeds retain the binary
Vignes result. Estimates are updated with bulk temperature and composition.
They use pure-solvent viscosity,
solute liquid volume at its normal boiling point, and solvent association
factors of 2.6 (water), 1.9 (methanol), 1.5 (ethanol), and 1 otherwise. Water
as solute uses four times its liquid molar volume in the diffusivity estimate.
These estimates are intended for molecular liquids, not electrolyte diffusion.
Optional `binary_diffusivity` (alias `effective_diffusivity`) supplies a constant
effective Maxwell–Stefan diffusivity instead (`m2/s`, `cm2/s`, or `m2/h`);
the thermodynamic factor still applies. This override bypasses all pair estimates.

The flat-plate option uses area-average `Nu=0.664 Re^0.5 Pr^(1/3)` and
`Sh=0.664 Re^0.5 Sc^(1/3)`, giving `thermal_film_thickness=L/Nu` and
`film_thickness=L/Sh`. Properties are evaluated at the instantaneous bulk
temperature/composition; heat capacity comes from the liquid-enthalpy derivative.
Schmidt number uses the effective Fick diffusivity `D_MS*Gamma` at bulk composition.
The model requires `Re<5e5`, `Pr>=0.6`, `Sc>=0.6`, and both thermal and solutal
Peclet numbers at least 100. Out-of-range conditions fail explicitly. This
correlation assumes forced laminar flow along a smooth plate, approximates the
surface by an area average, and does not describe stirred vessels, natural
convection, or the changing hydrodynamics of a thick/rough deposit.

`inclusion_max_fraction` enables a bounded mechanical-liquid-capture heuristic
when greater than zero; default zero disables capture. Its range is `[0,1)`.
At each step, `epsilon = inclusion_max_fraction * Pe_growth/(1+Pe_growth)`,
where `Pe_growth = G_dense*film_thickness/D_Fick` and `G_dense=N_crystal*V_solid`.
Captured volume rate is `epsilon/(1-epsilon)` times the newly formed solid
volume rate. This is an adjustable engineering heuristic, not a fitted universal
correlation. It vanishes with growth rate and is bounded by the specified fraction.

Captured liquid has the instantaneous bulk composition and enthalpy. Mechanical
engulfment is assumed to bypass the selective diffusion film, so it is separate
from lattice incorporation or an interfacial partition coefficient. Each
component's trapped amount is accumulated over time and removed from the free
liquid inventory. Layer thickness includes the captured volume. Effective layer
conductivity uses the harmonic solid/liquid mixing rule at the accumulated
porosity, with liquid conductivity evaluated at bulk conditions. Captured liquid
volumes remain at their capture values. Inclusions do not subsequently diffuse,
freeze, drain, or undergo sweating; their liquid state is a metastable approximation.

Cooling integrates free-liquid enthalpy, subtracting liquid-to-interface heat
transfer and the enthalpy carried by material leaving the free liquid. The
shared enthalpy-state solver recovers bulk temperature. An independent
deposition-enthalpy ledger accumulates crystal enthalpy at interface temperature
and trapped-liquid enthalpy at capture temperature. These equations obey
`d(H_bulk + H_deposited)/dt = Q_wall` within the quasi-steady model, and the
reported energy residual checks this identity. The ledger does not resolve
sensible-heat redistribution in the existing layer or a spatial temperature
field. In isothermal mode the ledger's bulk energy differs from actual bulk
enthalpy; their difference is reported as `temperature_control_energy_kJ`.

Solid conductivity is requested from the shared resolver with `phase='solid'`.
The current resolver does not yet support that phase, so a normal production
growth run will fail explicitly until that resolution chain is available.
Neither a liquid-conductivity substitute nor a hardcoded solid value is used.
Other missing transport properties also fail explicitly. Equilibrium layer
mode does not require conductivity or diffusivity.

`layer_relative_tolerance` (default `1e-6`) controls time integration;
`layer_profile_points` (integer >= 2, default 21) sets reported profile points.
Performance/phase details report thickness, deposited amount, interface
temperature/composition, SLE and Stefan residuals, and wall heat removal.
`wall_duty_kW` is cycle-averaged wall heat transfer (negative for cooling).
Total `duty_kW` remains the exact inlet/outlet enthalpy difference.
`conditioning_duty_kW = duty_kW - wall_duty_kW` accounts for thermostatted-bulk
operation and feed/product conditioning, including bringing the harvested
layer to the specified outlet temperature. Solid allocation is preserved during
that conditioning; it does not perform another equilibrium/remelting stage.

Mother-liquor retention/drainage applies only to the free liquid remaining after
inclusion capture. All captured liquid stays with the harvested layer, including
when drainage retention is zero. Its accumulated composition is preserved;
it is not reassigned the final free-liquid composition. The stream represents
trapped material as liquid, not as an additional crystalline impurity phase.
No solid solution, bulk nucleation, sweating, or remelting stage is predicted.
Significant reverse growth fails explicitly. The homogeneous-liquid limitation and outlet
LLE diagnostics still apply. Predictions rely on the stated film and
quasi-steady assumptions; the solver has not been calibrated to equipment data.

Transport references:
[Li (1976)](https://doi.org/10.1002/aic.690220520),
[Wilke and Chang (1955)](https://doi.org/10.1002/aic.690010222),
[Vignes (1966)](https://doi.org/10.1021/i160018a007), and the
[water-solute correction discussion](https://www.sciencedirect.com/science/article/pii/S0021967311010089).
For flat-plate transfer, see
[COMSOL external forced convection](https://doc.comsol.com/6.4/doc/com.comsol.help.heat/heat_ug_theory.07.102.html)
and [COMSOL mass-transfer theory](https://www.comsol.com/multiphysics/what-is-mass-transfer?parent=fluid-flow-heat-transfer-and-mass-transport-0402-372).

**Empirical finite-rate layer growth:**

`model = empirical_layer_growth` is a material-correlation alternative to the
transport-based `layer_growth` model. It integrates a supplied layer-thickness
growth law and optional, independent effective distribution coefficients as
the mother-liquor composition changes. It does not request thermal
conductivity, diffusivity, or a mechanistic inclusion parameter.

```pfd
UNIT C : Crystallizer
    model = empirical_layer_growth
    crystallizing_component = AA
    T = 11 [C]
    T_wall = 0 [C]
    cooled_area = 10 [m2]
    growth_time = 10 [min]
    cycle_time = 60 [min]
    layer_solid_density = 1050 [kg/m3]

    growth_model = undercooling_power_law
    growth_coefficient = 4.68e-7
    growth_exponent = 0.89
    growth_rate_unit = m/s

    keff_PA_expression = 0.99*(x_impurity/(1-x_impurity))**0.28*exp(59320*G_m_s)
    keff_H2O = 0
```

The unit requires `T`/`T_out`, `T_wall` (aliases `coolant_temperature` and
`T_coolant`), `cooled_area`, `growth_time`, and `cycle_time`. Wall temperature
is constant and must be below the controlled bulk/product temperature.
`cycle_time` must be at least `growth_time`; feed flow times cycle duration
defines the batch charge, and harvested amount divided by cycle duration gives
the equivalent continuous throughput. If the feed contains more than one
component declared `conventional_with_solid`, `crystallizing_component` must
select the one forming this layer; the others may have impurity `k_eff` laws
but do not form separate solids in this empirical model.

Growth can be supplied in four forms:

- Direct constant `growth_rate` (aliases `growth` and `G`), with a unit
  attached to that parameter.
- `growth_model = constant`, with `growth_value` and `growth_rate_unit`.
- `growth_model = undercooling_power_law` (aliases `power_law` and
  `delta_T_power_law`), with `growth_coefficient`, `growth_exponent`, and
  `growth_rate_unit`; it evaluates
  `G = coefficient*max(deltaT,0)**exponent`.
- `growth_model = custom`, with `growth_expression`, `growth_rate_unit`, and
  arbitrary finite constants declared as `growth_param_<name>`.

Growth units use the same explicit conversions as MSMPR growth: `m`, `mm`,
`um`, or `µm` per `h`, `min`, or `s`. `G` is total layer-thickness growth rate,
not particle-diameter or radius growth.

Each impurity law is named for its component symbol. Prefixes `keff_`,
`k_eff_`, `distribution_`, and `effective_distribution_` are equivalent. A
bare value specifies a constant coefficient:

```pfd
keff_PA = 0.2
```

A custom expression uses flattened fields and case-preserving `param_`
constants:

```pfd
keff_PA_model = custom
keff_PA_expression = alpha*(x_impurity/(1-x_impurity))**beta*exp(c*G_m_s)
keff_PA_param_alpha = 0.99
keff_PA_param_beta = 0.28
keff_PA_param_c = 59320
```

For direct Python construction, `effective_distributions`,
`distribution_coefficients`, `keff`, or `k_eff` may instead be a
component-to-definition mapping.
Each definition is either a number or a mapping containing `model=constant`
plus `value`, or `model=custom` plus `expression`, optional `parameters`,
and/or `param_*` values.

At every integration point, the coefficient uses the differential
complete-layer mole-fraction convention
`x_impurity,deposit = k_eff*x_impurity,liquid`. Coefficients must be finite and
nonnegative, and their implied impurity mole fractions must sum to less than
one. The remainder of each instantaneous deposit is the pure crystalline
component. Incorporated impurities are retained with the harvested layer and
reported as trapped material; this does not assert a solid solution or a
particular microscopic inclusion mechanism. An impurity without a coefficient
is completely rejected.

Custom growth and distribution expressions support the same arithmetic and
safe functions as MSMPR. They can use:

- Liquid mole- and mass-fraction mappings `x["component"]` and
  `w["component"]`; `x0` and `w0` are their initial values.
- `x_crystal`, `w_crystal`, and, inside an impurity law, `x_impurity` and
  `w_impurity`.
- `T`/`T_bulk`, `Twall`/`T_wall`/`Tcool`/`T_coolant`, `P`, and current
  saturation temperature `Tsat`/`Teq`/`T_eq`.
- `deltaT`/`dT = Tsat-Twall`, `deltaT_bulk = Tsat-T`, and
  `deltaT_wall`/`wall_undercooling = T-Twall`.
- Crystallizing-component `S`, `sigma`, `relative_supersaturation`, `lnS`,
  `activity`/`a`, and `a_sat`/`asat`, evaluated at bulk `T`.
- `time`/`t`/`time_h`, `time_s`, `duration`/`duration_h`,
  `thickness`/`L`, `area`, and crystallizing-component `recovery`.
- `Tm`, `Hfus` [J/mol], `rho_s`, `Vm_solid`, `pi`, and `R`.
- Evaluated growth `G`/`G_m_h` and `G_m_s` in distribution expressions.

`layer_solid_density` (`kg/m3`, `kg/m^3`, `g/cm3`, or `g/cm^3`) overrides the
resolved solid density. Without it, the shared solid-volume resolver is used.
`empirical_relative_tolerance` defaults to `1e-7`, and
`layer_profile_points` defaults to 21.

Growth is suppressed once wall temperature reaches the current liquid
saturation temperature; integration stops early if a required component
inventory is exhausted. The profile
reports liquid composition, saturation temperature, undercooling, growth rate,
layer composition, each `k_eff`, thickness, and recovery. Overall outlet duty
remains the exact inlet-to-outlet enthalpy difference at specified product
temperature; an empirical law does not separately predict coolant-side heat
transfer or wall duty.

The empirical laws are applied instantaneously while integrating changing
mother liquor. A correlation fitted to an overall layer composition at nearly
constant bulk composition is therefore a differential extrapolation at high
freezeout. Nucleation, sweating, remelting, layer failure, and temperature
ramps are not modeled.

**Steady kinetic MSMPR mode:**

The first kinetic implementation represents one well-mixed, steady-state
mixed-suspension, mixed-product-removal crystal population. It supports
deterministic size-independent or size-dependent growth, primary or
suspension-dependent secondary nucleation, and a same-component solid-bearing
inlet as a seed population. It excludes agglomeration, breakage, growth-rate
dispersion, classification, and spatial supersaturation gradients.

```text
UNIT C : Crystallizer
    model = MSMPR
    crystallizing_component = solute
    T = 20 [C]
    residence_time = 2 [h]
    quadrature_classes = 20

    growth_model = power_law
    growth_coefficient = 2.4e-8
    growth_g = 1.4
    growth_rate_unit = m/s

    nucleation_model = secondary_power_law
    nucleation_coefficient = 1.0e10
    nucleation_b = 2.1
    nucleation_j = 0.8
    nucleation_rate_unit = 1/m3/s
```

- Specify exactly one of `residence_time`/`tau` (hours by default) or
  `volume`/`V` (m3 by default). When residence time is specified, vessel volume
  is calculated from the mixed-product-removal volumetric rate; when volume is
  specified, residence time is calculated from that same outlet rate. The rate
  is coupled to the trial liquid density and solid molar volume rather than
  assumed equal to a constant-density feed rate.
- MSMPR mode currently requires exactly one
  `phase_behavior=conventional_with_solid` component in the feed. An optional
  `crystallizing_component`/`component` value may name it explicitly.
- A solid-bearing inlet is accepted only when its sole solid component is the
  crystallizing component and its PSD accounts for the complete inlet solid
  flow. Those particles are treated as continuously fed seeds. Direct PFD feed
  allocation to a conventional solid is still reserved for the future stream
  solid-allocation syntax; seeded MSMPR feeds can currently originate from an
  upstream solid-forming unit.
- `quadrature_classes`/`particle_classes` is an integer from 2 to 200 and
  defaults to 20. Gauss-Laguerre quadrature represents the complete exponential
  MSMPR residence-age distribution without a finite size cutoff.
- `maximum_output_classes`/`max_output_classes` is an integer from 2 to 1000
  and defaults to 200. Seeded cascades are conservatively rebinned to this
  limit on a number basis; each grouped representative diameter preserves both
  the group's particle count and solid volume.
- `nucleus_diameter`/`nucleation_diameter`/`L0` optionally supplies a finite
  birth diameter (m by default; `mm`, `um`, and `µm` are accepted). The default
  is zero. A size-dependent growth expression that has zero growth at `L=0`
  generally needs a positive birth diameter.

For a clear feed, no agglomeration/breakage/classification, and deterministic
linear growth, the steady population balance is

```text
d(G n)/dL + n/tau = 0,       G(0) n(0) = B0
```

For size-independent growth this gives
`n(L) = (B0/G) exp[-L/(G tau)]`. For size-dependent growth, the simulator
integrates `dL/dage = G(L, ...)` over the exponential residence-age
distribution. `L` is always volume-equivalent diameter [m], so `G` means
`dL/dt`, not a radius-growth rate.

The power-law presets are:

- Growth `power_law`: `G = coefficient * sigma^g`. Supply
  `growth_coefficient` (aliases `growth_k`, `growth_kg`) and `growth_g`
  (aliases `growth_exponent`, `growth_supersaturation_exponent`).
- Nucleation `primary_power_law`: `B0 = coefficient * sigma^b`. Supply
  `nucleation_coefficient` (aliases `nucleation_k`, `nucleation_kb`) and
  `nucleation_b`.
- Nucleation `secondary_power_law`:
  `B0 = coefficient * sigma^b * MT^j * G0^i`. Supply `nucleation_j`
  (aliases `nucleation_mt_exponent`, `nucleation_suspension_exponent`). This is
  suitable when secondary nucleation is empirically correlated with suspended
  crystal mass. `nucleation_i` (alias `nucleation_growth_exponent`) optionally
  adds birth-size growth-rate dependence and defaults to zero. This preset does
  not itself model an attrition mechanism or mixing field.

Growth units may be `m`, `mm`, `um`, or `µm` per `h`, `min`, or `s`.
Nucleation units may use `1`, `#`, or `particles` per `m3` per `h` or `s`, and
`1` or `#` per `L` per `h` or `s`. Units are mandatory because kinetic
coefficients are not magnitude-inferred.

Either law may instead use `model=custom` and an arithmetic expression:

```text
growth_model = custom
growth_expression = kg * sigma^g * (1 - exp(-alpha*(L + L0)))
growth_param_kg = 2.4e-8
growth_param_g = 1.4
growth_param_alpha = 25000
growth_param_L0 = 1e-6
growth_rate_unit = m/s
```

Custom expressions support `+`, `-`, `*`, `/`, powers, parentheses, and the
safe functions `abs`, `exp`, `log`, `log10`, `sqrt`, `min`, and `max`.
Arbitrary finite numeric constants are declared with `growth_param_<name>` or
`nucleation_param_<name>`. Available state variables are:

- `S = activity/a_sat`, `sigma = max(S - 1, 0)`, signed
  `relative_supersaturation = S - 1`, and `lnS = log(S)`.
- `Tsat` [K] is the temperature at which the current mother-liquor composition
  is saturated at `P`; `deltaT`/`dT = Tsat - T` [K], and
  `deltaT_reduced = deltaT/Tsat`.
- `deltaT_fusion = Hfus*deltaT/(R*T*Tm)` is the fusion-scaled dimensionless
  undercooling, with `Hfus` [J/mol] and `Tm` [K] also exposed. It is a common
  approximate kinetic driving-force convention; `lnS` remains the rigorous
  activity-based dimensionless chemical-potential driving force and can differ
  from `deltaT_fusion`, especially for nonideal solutions.
- `T` [K], `P` [bar], solute mole fraction `x`, solute concentration `C`
  [kmol/m3], liquid `activity`/`a`, and saturation activity
  `a_sat`/`asat`.
- `MT` [kg crystals/m3 slurry], `tau` [h], `V` [m3], and `Q` [m3/h].
- `L` [m] and `age` [h] for growth expressions only.
- `G0`/`G0_m_h` [m/h] for nucleation expressions only. This is the evaluated
  growth rate at the population-balance birth boundary (`L=L0`, `age=0`), so
  it remains unambiguous when `G` is size-dependent and supports common
  secondary-nucleation correlations such as `B0 = kb*MT**j*G0**i`.
- Constants `pi` and `R` [J/mol/K].

The steady supersaturation is calculated, not specified. The solver couples
the liquid activity model, pure-solid saturation activity, user kinetics,
third PSD moment, suspension density, and solute material balance. Positive
growth and nucleation are suppressed at `S <= 1`; seed dissolution is outside
this first kinetic model and an undersaturated seeded case fails explicitly.

The ideal MSMPR balance and boundary condition follow the standard continuous
crystallization formulation used in, for example,
`doi:10.1021/acs.oprd.7b00225`. Suspension-dependent secondary nucleation and
size-dependent growth are empirical extensions whose parameters must be
obtained for the actual solute, solvent, crystal form, and equipment.

**MolecularSieveDryer / Dryer:**

- Selective water adsorption with one `feed`/`in` inlet and `product` plus
  `adsorbate` outlets. Only 3A molecular sieve is currently supported.
- `water_component` - Component symbol to remove; otherwise `water` or `H2O`
  is detected.
- Specify at most one sizing/performance target:
  `target_water_mole_fraction` (default `1e-4`), `removal_fraction`, or an
  adsorbent mass flow (`adsorbent_mass_flow`, `sieve_mass_flow`,
  `molecular_sieve_mass_flow`, `adsorbent_flow`, `sieve_flow`,
  `adsorbent_mass`, or `sieve_mass`). Mass-flow values use kg/h by default.
- `sieve_type` or `molecular_sieve` - Must select `3A`.
- `initial_loading_kg_per_kg` or `initial_loading` - Initial water loading of
  regenerated sieve, default zero.

**ShortcutExtractor / Extractor / LiquidLiquidExtractor / LLE:**

- Requires `feed` and `solvent` inlets and produces `raffinate` and `extract`.
- `N_stages` - Number of ideal stages, default 5.
- `T` - Isothermal operating temperature; defaults to the mean inlet
  temperature. Feed pressure is used.

**RigorousExtractor / RigorousLiquidLiquidExtractor:**

- Counter-current equilibrium-stage LLE with `feed` and `solvent` inlets and
  `raffinate` and `extract` outlets.
- `N_stages` - Number of stages, default 5.
- `mode` - `isothermal` (default) or `adiabatic`.
- `T` - Isothermal temperature or initial adiabatic temperature.
- `solver_algorithm`/`algorithm` - `equation_oriented` (default) or
  `split_sweep`.
- Common controls include `mesh_tolerance`, `acceptable_mesh_residual`,
  `max_iterations`, `max_jacobian_evaluations`, `line_search_steps`,
  `finite_difference_rel_step`, `component_scale_floor`, and
  `semi_analytic_flow_jacobian`.
- Adiabatic bounds are `T_min`/`T_max` (aliases `adiabatic_T_min` and
  `adiabatic_T_max`). Initialization controls include `initializer`,
  `coarse_initial_stages`, and `mesh_initial_sweeps`.

**Absorber / Stripper:**
- Shortcut Kremser-style gas absorption or stripping estimate.
- `N_stages` - Number of ideal stages.
- `T` - Isothermal operating temperature.
- `P` - Operating pressure.

**RigorousAbsorber / RigorousStripper:**
- Sparse Newton equilibrium-stage column with vapor-liquid equilibrium,
  component balances, and either adiabatic stage energy balances or fixed
  isothermal stage temperatures.
- Inlets must include at least one vapor/gas stream and one liquid/solvent
  stream. Port names containing `gas`, `vapor`, `vapour`, or `air` are treated
  as vapor feeds; names containing `liquid`, `solvent`, `lean`, or `water` are
  treated as liquid feeds. `RigorousStripper` also treats `strip`/`steam` ports
  as vapor and `feed`/`rich`/`solution` ports as liquid.
- Outlets are `gas_out` and `liquid_out`.
- `N_stages` or `stages` - Number of equilibrium stages.
- `mode` - `adiabatic` by default; `isothermal` fixes each stage temperature
  and reports the implied heat duty.
- `T`, `temperature`, or `stage_temperature` - Scalar isothermal temperature.
  Supplying a scalar temperature without `mode` also selects isothermal mode.
- `stage_temperatures` or `temperature_profile` - Comma-separated isothermal
  stage-temperature profile, one value per stage.
- `P`, `P_top`, `P_bottom`, `P_drop_per_stage`, or `stage_pressures` - Pressure
  specification. If only `P_top`/`P` and `P_drop_per_stage` are supplied, stage
  pressure increases downward.
- `feed_stages` - Optional per-inlet feed stage map, for example
  `gas:5,solvent:1`.
- `gas_stage`/`vapor_stage` and `liquid_stage`/`solvent_stage` - Default feed
  stages for vapor and liquid feeds. `RigorousStripper` also accepts
  `strip_gas_stage`/`stripping_gas_stage`, `feed_stage`, and
  `rich_liquid_stage`.
- Solver controls include `mesh_tolerance`, `acceptable_mesh_residual`,
  `max_iterations`, `max_jacobian_evaluations`, `line_search_steps`,
  `finite_difference_rel_step`, `component_scale_floor`, and
  `component_solve_threshold`.
- `semi_analytic_flow_jacobian` - Enables exact liquid/vapor flow derivatives
  and the absorber's optimized Jacobian, default `true`. Set to `false` to use
  the full colored finite-difference Jacobian.
- `semi_analytic_local_thermo_jacobian` - Recalculates only the affected stage
  when differentiating temperature and composition, default `true`. Set to
  `false` to retain exact flow derivatives but use global colored thermo
  perturbations.
- `henry_components` - `auto` by default. Use `none`/`off` to disable aqueous
  Henry treatment, or provide a comma-separated explicit component list.
- `water_component` - Optional explicit aqueous solvent component. Otherwise
  water is identified from its symbol, formula, name, or CAS number.
- `henry_water_mole_fraction_min` / `henry_water_cutoff` - Minimum conservative
  pre-solve aqueous water mole fraction for automatic Henry selection, default
  `0.95`.
- `henry_dilute_mole_fraction_max` / `henry_dilute_cutoff` - Maximum
  conservative aqueous loading estimate for an ordinary condensable to use
  Henry's law automatically, default `0.001`.
- `henry_pressure_warning_bar` - Pressure above which the simulator warns for
  any Henry component lacking a high-pressure correction, default `20` bar.

Automatic Henry selection is performed once before initialization and remains
frozen throughout the solve. Components above their critical region at the
coolest estimated aqueous temperature use Henry data when available. Ordinary
condensables use it only when their maximum loading relative to water already
present in liquid feeds is below the dilute cutoff. The solved profile is
reported and warned if its water fraction falls below the selection cutoff, a
Henry solute rises above the dilute cutoff, or a component selected as
noncondensable falls below its critical temperature. The model never switches
branches mid-calculation.

For UNIQUAC-family methods, a component with valid Henry data may remain in the
thermodynamic component set even when ordinary-liquid UNIQUAC `r`/`q`
parameters are unavailable. A frozen aqueous Henry context removes its selected
Henry solutes, renormalizes the remaining liquid submixture, and evaluates
UNIQUAC only for that submixture. This is contextual rather than a permanent
component classification: using the same component at positive mole fraction
in an ordinary liquid calculation still requires `r`/`q` and fails clearly if
they are unavailable. Vapor-only state calculations do not require liquid
UNIQUAC parameters.

**ShortcutDistillation / McCabeThieleDistillation / CMODistillation:**
- `N_stages` - Number of stages
- `reflux_ratio` - Reflux ratio
- `D_to_F` - Distillate-to-feed molar flow ratio
- `D`, `distillate_flow`, or `top_flow` - Optional direct distillate flow specification
- `distillate_mass_flow`, `D_mass`, or `D_mass_flow` - Direct distillate mass flow specification; values are kg/h unless units say otherwise
- `D_mass_to_F_mass`, `D_mass_fraction`, or `distillate_mass_fraction` - Distillate-to-feed mass flow ratio
- `P_condenser` - Condenser pressure
- `P_drop_per_stage` - Pressure drop per stage

`ShortcutDistillation` uses the Fenske-Underwood-Gilliland method with a
Hengstebeck-Geddes component split and a Kirkbride feed-stage estimate. The
optional `light_key` and `heavy_key` parameters override automatic key
selection. Automatic keys are the adjacent volatility pair around the requested
distillate cut. Supplying only one key selects its adjacent counterpart;
supplying both requires an adjacent pair so the Underwood root interval contains
no component-volatility pole. Key recovery specifications are
`light_key_recovery_distillate` and either
`heavy_key_recovery_bottoms` or `heavy_key_recovery_distillate`. A distillate
flow specification can replace one key recovery. When `N_stages`,
`reflux_ratio`, and a distillate flow are all supplied without recoveries, both
key recoveries are backsolved to match the specified column.

For `ShortcutDistillation`, a total condenser produces a bubble-point liquid
distillate and rejects feeds whose predicted overhead contains non-condensables.
Use `condenser_type = partial` for those systems: the distillate is then a
dew-point vapor and the reported reflux liquid is its incipient equilibrium
liquid composition.

`McCabeThieleDistillation` accepts binary feeds only and automatically selects
the internal feed stage. Its default straight operating lines use constant
molar overflow. Set `latent_heat_correction = true` to use rational operating
curves based on constant component latent heats.

The McCabe-Thiele equilibrium construction evaluates relative volatility through the selected
thermodynamic model's `K_values()` at the reference temperature
`(Tsat,pure-LK + Tsat,pure-HK + 2 Tsat,feed) / 4`. This keeps cubic-EOS,
gamma-phi, and vapor-association corrections active in key selection and the
binary equilibrium construction.

`CMODistillation` solves stagewise component balances and bubble-point
equations with fixed section traffic. It supports multicomponent feeds and
`total`, `partial`, and `mixed` condensers. For a mixed condenser, set
`distillate_vapor_fraction` between zero and one; the additional
`distillate_liquid` and `distillate_vapor` outlets are exposed. Set
`latent_heat_correction = true` to use constant component latent heats when
calculating section traffic. CMO currently accepts one positive feed and does
not support side draws or a decanter condenser.

**RigorousDistillation:**
- `N_stages` - Number of stages
- `feed_stage` - Default feed stage for any inlet without an explicit stage
- `feed_stages` - Optional per-inlet feed stages, for example `feed:13,benzene:6`
- `{port}_stage_from_bottom` / `{port}_feed_stage_from_bottom` - Optional per-inlet stage counted from the reboiler/bottom stage upward
- `reflux_ratio` - Reflux ratio
- `D_to_F` - Distillate-to-feed molar flow ratio
- `D`, `distillate_flow`, or `top_flow` - Optional direct distillate flow specification
- `distillate_mass_flow`, `D_mass`, or `D_mass_flow` - Direct distillate mass flow specification; values are kg/h unless units say otherwise
- `D_mass_to_F_mass`, `D_mass_fraction`, or `distillate_mass_fraction` - Distillate-to-feed mass flow ratio
- `P_condenser` - Condenser pressure
- `P_drop_per_stage` - Pressure drop per stage
- `P_bottom` - Optional bottom pressure; creates a linear pressure profile from `P_condenser`
- `stage_pressures` - Optional comma-separated pressure profile with one value per stage
- `condenser_type` - `total`, `partial`, `mixed`, or `decanter`
- `stage_phase_model` - `VLE` (default) or `VLLE`. `VLLE` permits up to two
  liquid phases on each stage and retains the smaller VLE equation set on
  stages whose liquid remains stable.
- `vlle_seed` - `auto` (default), `cheap`, or `homogeneous`. Auto uses the
  inexpensive estimate directly when it already contains LLE and otherwise
  obtains a homogeneous MESH profile before activating VLLE stages.
- `vlle_max_topology_updates` - Maximum active-set topology solves, default `8`
- `vlle_initial_topology` - `screened` (default), `all_vle`, or `all_vlle`.
  The forced forms are intended for continuation studies and diagnostics;
  absent forced liquid phases are removed before equation assembly.
- `vlle_topology_change_residual` - Largest scaled MESH residual at which an
  in-Newton phase-topology rebuild is allowed, default `0.05`
- `vlle_phase_fraction_min` - Minimum retained fraction of either liquid
  phase, default `1e-6`
- `vlle_phase_distance_min` - Minimum L1 composition distance for retaining a
  second liquid, default `1e-3`
- `vlle_colored_jacobian_fallback` - Retry a failed fixed-topology local
  Jacobian solve with colored finite differences, default `true`
- `initializer` / `initialization` - Initial profile method: `auto` (default),
  `estimate`/`cheap_estimate`, `coarse_rigorous`, `cmo`, `cmo-hvap`, or
  `azeotropic`
- `coarse_initial_stages` - Number of stages used when `initializer = coarse_rigorous`
- `distillate_vapor_fraction` - Vapor fraction of the distillate for `mixed` condensers
- `decanter_reflux_component` - For `RigorousDistillation` top decanters, sends the liquid phase richer in this component back as reflux
- `decanter_distillate_component` - Optional component selector for the decanter distillate phase
- `decanter_reflux_purge_fraction` - Optional purge fraction taken from the selected reflux phase before it returns to the column
- `side_draws` - Optional semicolon-separated side draw specs for `RigorousDistillation`, for example `stage:7,phase:liquid,flow:5,port:side_liq`
- `mesh_tolerance` - Scaled MESH residual tolerance for rigorous solvers
- `semi_analytic_flow_jacobian` - Enables the optimized Jacobian, default
  `true`. Set to `false` to restore the original full colored
  finite-difference behavior.
- `semi_analytic_local_thermo_jacobian` - Uses stage-local temperature and
  composition perturbations, default `true`. Set to `false` to retain exact
  flow derivatives but use the older global thermo perturbations.
- `semi_analytic_dense_limit_mb` - Maximum temporary dense Jacobian size for
  local assembly, default `200` MB. Larger columns use the sparse global
  strategy.

For `RigorousDistillation`, the standard outlets are `distillate` and `bottoms`.
Mixed condensers also expose `distillate_liquid` and `distillate_vapor`. Top
decanters expose `decanter_purge` when `decanter_reflux_purge_fraction` is
positive; in this mode `D_to_F`/`D` specifies the distillate product rate, and
the decanter phase split determines the effective reflux ratio. Side draw ports
use the `port` names provided in `side_draws`.

With `stage_phase_model = VLLE`, coexisting equilibrium liquids are modeled as
co-routed phases with one aggregate downward flow. The MESH equations use separate L1/L2
compositions, a liquid phase fraction, phase-weighted liquid enthalpy, and a
vapor in equilibrium with both liquids. The reported performance includes
`stage_phase_counts`, `stage_liquid1_compositions`,
`stage_liquid2_compositions`, `stage_liquid2_fractions`, phase-specific liquid
flows, and the active-set topology history. Total, partial, and mixed
condensers are supported. Decanter condensers and side draws are rejected in
VLLE mode until their separate liquid-phase routing is specified. The selected
thermodynamic method must provide activity coefficients and LLE stability;
gamma-phi methods iterate one shared vapor EOS state from both liquid
fugacities and report a post-solve three-phase log-fugacity residual.
`CMODistillation`, shortcut columns, and McCabe-Thiele columns remain VLE-only.
Stage-local Jacobians support total, partial, and mixed condensers with molar
or mass distillate specifications. Decanter condensers and columns with side
draws retain the original colored finite-difference Jacobian. If an optimized
Jacobian does not converge, the solve automatically retries from the same
initial profile with the original Jacobian.

Use `initializer = auto` for most columns. Auto uses the inexpensive smooth
`estimate` path for binary columns and a coarse rigorous solve for
multicomponent columns. The coarse solve is seeded by the internal legacy CMO
estimate. If a coarse grid is inapplicable, multicomponent auto initialization
falls back to that legacy estimate; top-decanter columns use the decanter-aware
smooth estimate directly.

`estimate` and `cheap_estimate` are aliases for the same inexpensive smooth
profile and do not run a shortcut, coarse, CMO, or azeotrope solve.
`coarse_rigorous` solves a reduced-stage MESH column and interpolates its full
profile onto the requested stage grid. `cmo` and `cmo-hvap` solve
`CMODistillation` with the rigorous column's condenser and pressure profile,
then transfer the complete temperature, composition, and flow profiles;
`cmo-hvap` enables CMO's latent-heat traffic correction. These CMO initializers
can cost more than the rigorous iterations they save, so use them only after
they have been shown to improve the specific column.

Use `azeotropic` when the intended separation specifically sends an azeotrope
overhead, such as removing a known azeotropic mixture as the distillate. It is
not recommended as a general initializer. Explicit specialized initializers
fail if they cannot build their requested profile rather than silently changing
to another strategy.

**Reactor:**

- `Reactor` applies each reaction's `conversion` simultaneously on an inlet
  component-flow basis. Use `basis=<component>` on a reaction to select that
  basis; otherwise the first net-consumed reactant is used. Competing
  conversions therefore refer to the original inlet inventory and do not
  depend on reaction-list order. Jointly infeasible conversions are rejected.
- `selectivity` and `yield` are calculated results, not reaction inputs. Model
  side reactions explicitly.
- `mode` - `isothermal` (default), `adiabatic`, or `duty`.
- `T` or `T_out` - Isothermal outlet temperature; defaults to inlet
  temperature.
- `Q`, `duty`, or `heat_duty` - Specified reactor duty for `duty` mode.
- `P_drop` - Nonnegative pressure drop, default zero.
- `desired_product` - Optional component for product-specific molar yield and
  selectivity reporting. The common reaction basis is inferred when unique.
- `report_basis` - Required with `desired_product` only when reactions use
  multiple basis components.
- Every carried or reacting component requires standard formation enthalpy
  `Hf`; the reactor refuses thermal results with an arbitrary zero reference.

**EquilibriumReactor:**

- Requires explicit `phase=vapor` or `phase=liquid`; mixed reactive VLE, LLE,
  VLLE, VDM association, ionic reactions, and permanent solids are rejected.
- Every reaction must use `<=>` or `<->`. One-way arrows and the former
  `approach` parameter are invalid.
- Reactions must be stoichiometrically independent. Duplicate and linearly
  dependent reaction sets are rejected before solving.
- `mode` - `isothermal` (default), `adiabatic`, or `duty`.
- `T` or `T_out` - Isothermal outlet temperature; defaults to inlet
  temperature.
- `Q`, `duty`, or `heat_duty` - Specified duty for `duty` mode.
- `P_drop` - Nonnegative pressure drop, default zero.
- Vapor equilibrium uses `y_i phi_i P/P_standard`; liquid equilibrium uses
  activity-coefficient pure-liquid reference fugacities or EOS liquid
  fugacities, according to the selected thermodynamic method.
- Reaction equilibrium constants use the ideal-gas standard chemical
  potentials constructed from `Hf`, absolute `S`, and integrated ideal-gas
  heat capacity. `Gf` is required as an independent 298.15 K consistency
  check. Missing required thermochemistry is an error.
- Adiabatic and duty modes nest the constrained equilibrium solve inside the
  outlet enthalpy solve, so temperature and reaction extents remain coupled.
- The converged forced-phase outlet is checked with the unconstrained fluid
  flash. A phase-unstable homogeneous result is rejected.
- Interior reactions require `|ln(Q)-ln(K)| <= 1e-7`. When equilibrium lies
  beyond a `1e-14` mole-fraction reporting floor, the component is treated as
  completely converted and the reaction is reported as `boundary_limited`
  with its one-sided quotient evaluated at that floor.

See `examples/equilibrium_methanol_synthesis_recycle.pfd` for compact unit and
stream syntax around a PR vapor-equilibrium reactor, condenser flash, purge,
and recycle loop.

**CSTR / KineticsCSTR:**

- `volume` or `V` - Reactor volume [m3].
- `T` or `T_out` - Isothermal temperature.
- `mode=isothermal` uses `T`/`T_out` and calculates required duty.
- `mode=adiabatic` solves temperature at zero duty.
- `mode=duty` requires `Q`/`duty`/`heat_duty` and solves temperature.
- `mode=jacketed` requires `T_jacket` (aliases `T_coolant` and `coolant_T`)
  plus either `UA` or both `U` and `A_heat`/`heat_transfer_area`. `UA` is W/K
  by default; `U` is W/m2/K and area is m2. Supplying both forms is an error.
- `phase` - Required homogeneous reaction phase: `vapor` or `liquid`.
- `P_drop` - Pressure drop.
- `T_min`/`T_max` bound thermal root discovery; defaults are 200 and 2500 K.
- `thermal_scan_points` controls the bounded thermal-root scan; default 161.
- When multiple steady states exist, `thermal_branch=lowest|nearest_inlet|highest`
  is mandatory. Every bracketed root is reported with its energy-balance slope
  and local thermal-stability classification. Positive residual slope is
  dynamically thermally stable under the lumped energy balance.

The CSTR solves all reaction extents simultaneously from
`extent = volume * outlet_rate`, with exact nonnegative component inventories.
Concentrations and residence time use the selected thermodynamic model's
homogeneous molar density. The forced-phase outlet is rejected if an
unconstrained flash or liquid-stability calculation predicts phase splitting.
Every thermal mode resolves the complete CSTR material/rate balance at each
trial temperature before evaluating the full inlet/outlet stream-enthalpy
balance. Jacket duty is `UA*(T_jacket-T)`. Formation enthalpy is therefore
included once without a separate reaction-heat correction.

**BatchReactor / KineticsBatch:**

`BatchReactor` integrates one representative, perfectly mixed vessel cycle
and maps the terminal batch product back to a time-averaged continuous stream.
It currently supports homogeneous constant-pressure vapor or liquid operation.
`pressure_mode=constant_pressure` is the default and only current mechanical
mode; aliases `constant` and `isobaric` are accepted.
Rigid constant-volume vessels, variable-pressure internal-energy balances,
multiphase batches, vents, and continuous bleed are not supported.

Exactly two of the following scheduling specifications are required:

- `V_batch` or `batch_volume` - Nominal inlet working volume per batch [m3].
- `N`, `n_vessels`, or `n_tanks` - Positive integer number of staggered vessels.
- `t_rxn`, `t_reaction`, or `reaction_time` - Reaction-period duration [h].

The unit derives continuous-equivalent inlet volumetric flow from the selected
homogeneous thermodynamic density. The batch start interval is
`V_batch/Q_in`. By default, both filling and draining take one batch start
interval. Custom `t_fill`/`fill_time`, `t_drain`/`drain_time`, and
`t_turnaround`/`turnaround_time` may be supplied. The balanced scheduling
relation is:

```text
N * batch_start_interval
    = t_fill + t_rxn + t_drain + t_turnaround
```

When `N` is calculated it is rounded upward to a whole vessel, and the
resulting spare capacity is reported as fleet utilization. When `N` is
specified, the relation is exact. A drain duration shorter than the batch
start interval produces discharge gaps; the continuous-equivalent result
remains valid as an average, but `surge_buffer_required=true` is reported.

Every continuous inlet is apportioned into an exact amount per representative
batch. Unscheduled ports form the initial charge before reaction time begins.
Basic semi-batch additions use:

- `semi_batch_feeds` or `scheduled_feeds` - Comma-separated inlet port names.
- `feed_start_times`/`feed_starts` - Comma-separated `port:hours` entries.
- `feed_stop_times`/`feed_stops` - Comma-separated `port:hours` entries.

A scheduled feed defaults to the complete reaction interval. Its active feed
rate is chosen so that amount per batch multiplied by batch frequency equals
the attached continuous inlet flow exactly. At least one inlet must remain an
initial charge. Feed start and stop boundaries split the adaptive integration
into deterministic constant-policy segments. Initial filling and terminal
draining affect scheduling but do not yet form chemically reactive recipe
stages.

Thermal and numerical controls are:

- `mode=isothermal` uses `T`/`T_out`, defaulting to the flow-weighted initial
  charge temperature, and calculates heat per batch.
- `mode=adiabatic` integrates total batch enthalpy with zero external heat.
- `mode=duty` requires `Q_batch`/`batch_energy` [kJ/batch] and distributes it
  uniformly over the reaction period.
- `mode=jacketed` requires `T_jacket` and either `UA` or `U` plus `A_heat`.
  `UA` is per vessel, and local heat is `UA*(T_jacket-T)`.
- `solver=auto|RK45|BDF|Radau|LSODA`, `relative_tolerance`/`rtol`,
  `absolute_tolerance`/`atol`, `maximum_step`, `first_step`, and
  `profile_points` follow the kinetic PFR adaptive-control conventions, with
  time-step units instead of axial units. Feed start/stop boundaries are always
  added to the reporting grid, so a semi-batch profile may contain more rows
  than the requested base count.
- Optional `V_vessel`/`vessel_volume` and `fill_fraction` validate the maximum
  transient working volume against usable physical capacity with a terminal
  integration event.

The transient state contains component inventories, reaction extents, and—for
nonisothermal modes—complete total enthalpy and cumulative heat. Reaction heat
is included through formation enthalpy exactly once. Multiple reactions use
the same canonical rate states and explicit kinetic units as CSTR and PFR.
Terminal average component flow is `batch_frequency * terminal_batch_amount`;
reaction-induced total-mole changes are therefore preserved rather than
forcing outlet total flow to equal inlet total flow.

Performance results include the resolved schedule, fleet utilization,
discharge coverage, terminal state, reaction extents per batch and per hour,
heat per batch, average duty, transient profiles, independent material/energy
closure, and homogeneous phase-stability diagnostics. During recycle
iterations, full profiles and interior diagnostics use the same scheduled
expensive-diagnostic policy as PFR; outlet stability and exact closure remain
mandatory on every evaluation.

**PFR / KineticsPFR / PlugFlowReactor:**

- Geometry may be supplied as `volume`/`V`, `length` plus `diameter`, or any
  consistent two of those three values. `n_tubes` defaults to one. Diameter is
  per tube; volume and heat-transfer perimeter include every tube.
- `phase` is required and must be homogeneous `vapor` or `liquid`.
- `mode=isothermal` uses `T`/`T_out` and reports the exact required duty.
- `mode=adiabatic` integrates constant total enthalpy flow.
- `mode=duty` distributes `Q`/`duty`/`heat_duty` uniformly over reactor volume
  or length and integrates the resulting enthalpy flow.
- `mode=jacketed` requires `U`, `T_jacket` (aliases `T_coolant` and
  `coolant_T`), and resolvable length/diameter. Local heat transfer is
  `U * tube perimeter * (T_jacket - T)`.
- `pressure_drop_model=none` is the default.
- `pressure_drop_model=specified` distributes total `P_drop` linearly over
  physical length. A positive `P_drop` selects this model when no model is
  named.
- `pressure_drop_model=darcy` (aliases `pipe`, `unpacked`, and
  `darcy_weisbach`) uses local density, viscosity, superficial velocity,
  compressibility acceleration, optional `roughness`, and selectable
  `friction_model=churchill|colebrook|haaland|swamee_jain`.
- `pressure_drop_model=ergun` requires
  `void_fraction` and `particle_diameter`. It evaluates both viscous and
  inertial Ergun terms from the local reacting state and includes
  compressibility acceleration. Reported fluid residence time integrates
  void volume, while reaction rates remain on the declared total-reactor-volume
  basis.
- Use `PackedBedReactor` instead when kinetics are reported per catalyst mass;
  PFR's Ergun option retains fluid-volume kinetic units for compatibility.
- Calculated/distributed pressure drop requires resolvable length and diameter.
- `solver=auto|RK45|BDF|Radau|LSODA`; `auto` selects LSODA's automatic
  nonstiff/stiff switching for thermal, multi-reaction, reversible, or
  custom-rate problems and RK45 for simple isothermal power laws.
- `relative_tolerance`/`rtol`, `absolute_tolerance`/`atol`, `maximum_step`,
  and `first_step` control adaptive integration.
- `profile_points` controls reporting only. Legacy `n_segments` is accepted as
  a deprecated profile-point alias and no longer determines accuracy.

The PFR integrates reaction extents, pressure, total enthalpy flow, and
residence time in one adaptive system. Reaction rates use the same canonical
thermodynamic rate states as CSTRs. A minimum-pressure event remains fatal.
Component depletion instead activates a nonnegative-inventory continuation:
the state is projected exactly onto zero inventory, and only reaction
directions that would consume more of an absent component are constrained.
Independent and producing reactions continue, and constrained directions may
resume when production makes them feasible. Pressure drop, residence time,
specified duty, and jacket heat transfer continue through the full reactor.
Results report `component_depletion_events`; if every reaction becomes inactive
downstream, PFR additionally reports `inactive_remaining_reactor_volume_m3`
and PBR reports `inactive_remaining_catalyst_mass_kg`. Material and energy
closure are independently verified, and forced-phase stability is checked
throughout the reported profile with global liquid stability checks at the
inlet, midpoint, and outlet.

VDM thermodynamic methods are supported for axial PFR/PBR vapor calculations
on a nominal monomer-equivalent component basis. Stoichiometric flows and
`p[...] = y_nominal*P` kinetic variables remain nominal; `f[...]` and vapor
activities include the VDM monomer fugacity correction. Vapor volume and
density use the physical molecule count after all homo- and cross-dimerization
extents, so concentration, residence time, velocity, and hydraulic calculations
remain consistent with association. Vapor viscosity presently retains the
nominal-component mixing rule; a deduplicated warning is emitted only if a
Darcy/Ergun or other transport calculation actually requests it while
association is active.

VDM vapor enthalpy is the ideal-gas monomer-reference enthalpy plus the actual
association enthalpy at the local `T`, `P`, and nominal composition. Tabulated
`Hvap` values are apparent liquid-to-associated-saturated-vapor values, so the
pure saturated association reference is applied to the corresponding liquid
enthalpy instead of subtracted from every vapor state. Its temperature
derivative is included in liquid `Cp`. Consequently, pure saturated
liquid-vapor enthalpy differences preserve the selected apparent `Hvap`, while
dilute hot-vapor departures tend to zero with the physical association extent.

During recycle convergence, every PFR evaluation still verifies exact closure
and outlet phase admissibility. Full requested profiles and interior stability
sweeps follow the recycle solver's scheduled expensive-diagnostic cadence
(evaluations 1, 2, 5, then every tenth through 30 and every twenty-fifth
thereafter) and are always regenerated on the final pass. Intervening
evaluations retain inlet/outlet profile points only. Viscosity is evaluated only
for Darcy or Ergun hydraulic models.

**PackedBedReactor / PBR / KineticsPackedBed:**

`PackedBedReactor` is a distinct fixed-catalyst unit that shares the adaptive
axial thermodynamic, energy, hydraulic, event, and diagnostic core with PFR.
The catalyst belongs to the reactor and never enters or leaves in a material
stream. Reaction rates must use a catalyst-mass unit such as
`kmol/kg_cat/h` or `mol/kg_cat/s`; ordinary PFR, CSTR, and BatchReactor units
continue to require fluid-volume rate units.

Required bed specifications are:

- `diameter`/`D`/`bed_diameter` - Cylindrical bed diameter per tube.
- `bulk_catalyst_density`/`catalyst_bulk_density`/`rho_bulk` - Catalyst mass
  divided by total packed-bed volume [kg/m3], including bed voidage.
- `bed_void_fraction`/`void_fraction` - Interparticle fluid-volume fraction.
- At least one of `length`/`bed_length`, `bed_volume`/`volume`, or
  `catalyst_mass`/`W_cat`. The remaining quantities are derived. Redundant
  values must agree with cylindrical geometry and
  `catalyst_mass = bulk_catalyst_density * bed_volume`.

`n_tubes` defaults to one. Fluid residence time integrates
`bed_void_fraction * bed_volume`; catalyst-mass reaction extent instead uses:

```text
d(extent)/dz = (catalyst_mass/bed_length) * rate_per_catalyst_mass
```

Thermal modes, solver controls, homogeneous phase checks, recycle diagnostic
scheduling, and specified/jacketed heat inputs follow PFR. Optional
`pressure_drop_model=ergun` requires `particle_diameter` and evaluates local
density, viscosity, superficial velocity, viscous/inertial Ergun terms, and
compressibility acceleration. `pressure_drop_model=none` remains available
when pressure loss is intentionally neglected.

`kinetic_basis=apparent` is the default. Such rates are interpreted as measured
effective catalyst-mass rates and receive no pellet correction. Intrinsic rates
may use either:

- `diffusion_model=specified` with `effectiveness_factor` in `(0,1]`; or
- `diffusion_model=first_order_sphere` for the exact restricted analytical
  case; or
- `diffusion_model=generalized_power_law_sphere` with
  `effective_diffusivity` and `particle_diameter` or `pellet_radius`; or
- `diffusion_model=rigorous_power_law_sphere` for a numerical one-species
  spherical pellet BVP.

Supplying `effective_diffusivity` without naming a model selects
`generalized_power_law_sphere`. One diffusion-limiting component is inferred
when unique or selected with `diffusion_limiting_component`.

The exact `first_order_sphere` mode remains restricted to one irreversible,
first-order, concentration-basis power-law reaction with no other nonzero
orders. The generalized model supports one arbitrary positive-order power-law
reaction, or multiple irreversible power-law reactions when they all consume
the same limiting species with the same positive order. Other kinetic factors
are treated as spatially uniform within the pellet, so the reactions preserve
their relative rates and share one effectiveness factor. A single reversible
power-law reaction is supported using its local apparent order; multiple
reactions cannot share a calculated factor when any is reversible.

Pellet density is derived as
`bulk_catalyst_density/(1-bed_void_fraction)`. The generalized modulus is:

```text
phi_generalized = phi * sqrt((apparent_order + 1)/2)
eta = 3/phi_generalized
      * (1/tanh(phi_generalized) - 1/phi_generalized)
observed_rate = eta * intrinsic_rate
```

`effectiveness_factor_policy=constant_inlet` is the default: the inlet factor
is applied throughout the reactor. `effectiveness_factor_policy=local`
recalculates the inexpensive algebraic approximation at every axial state.
Results report inlet and minimum/maximum ordinary/generalized Thiele moduli,
apparent orders, and effectiveness factors.

The rigorous mode supports one diffusion-limiting species consumed by one or
more irreversible power-law reactions. The reactions may have different
positive orders in that species. It solves:

```text
u'' + 2*u'/x = sum(phi_j^2 * u^n_j)
u'(0) = 0
u(1) = 1
```

and integrates a separate `eta_j = 3*integral(u^n_j*x^2 dx)` for each reaction.
This retains diffusion-induced selectivity changes rather than applying one
factor to pathways with different orders. Fractional-order cases that form a
non-smooth dead core and do not converge are rejected explicitly.

`effectiveness_factor_resolves` controls rigorous axial sampling. Its default
is one inlet BVP solve. For `N>1`, the reactor performs a preliminary axial
pass, solves the pellet at exactly N evenly spaced support states, linearly
interpolates every reaction-specific factor, and performs the final axial
solve. Valid values are 1 through 101. Results report requested/actual solve
counts, support positions, factors, radial-node count, and pellet-center
concentration ratio. `pellet_relative_tolerance` defaults to `1e-6`, and
`pellet_maximum_nodes` defaults to `2000` with a minimum of 50.

Series networks that produce the nominated species, custom/signed calculated
diffusion, rigorous reversible networks,
external film transfer, nonisothermal pellets, catalyst deactivation, moving
catalyst, slurry catalyst, and multiphase trickle/fluidized beds are not
supported. Those arbitrary kinetics may still use a supplied
`effectiveness_factor`.

### 11. Reactions Section

Within a UNIT block:

```
REACTIONS:
    <equation> | <kinetics_params>
```

Equation format:
- Reactants and products separated by `->` or `<=>` (equilibrium)
- Stoichiometric coefficients before species
- Species separated by `+`

Example:
```
2 A + B -> 3 C + D
A + B <=> C | name=equilibrium_reaction
```

Kinetics parameters:
- `type` - `power_law` (default), `custom`, or `custom_net`.
- `A` - Arrhenius pre-exponential factor. Its derived units are the declared
  `rate_unit` divided by the declared rate-basis unit raised to the overall
  kinetic order.
- `Ea` and `Ea_unit` - Activation energy and its explicit unit. Supported
  units are `J/mol`, `kJ/mol`, `kJ/kmol`, and `J/kmol`.
- `rate_basis` - For power laws: `concentration`, `partial_pressure`,
  `fugacity`, or `activity`.
- `concentration_unit` - `kmol/m3`, `mol/m3`, or `mol/L`.
- `pressure_unit` - `bar`, `kPa`, `Pa`, or `atm`; it applies to both partial
  pressures and fugacities.
- `rate_unit` - Fluid-volume units `kmol/m3/h`, `kmol/m3/s`, `mol/m3/h`,
  `mol/m3/s`, `mol/L/h`, `mol/L/min`, or `mol/L/s`; or packed-bed
  catalyst-mass units `kmol/kg_cat/h`, `kmol/kg_cat/s`, `mol/kg_cat/h`, or
  `mol/kg_cat/s`.
- `order_X` - Reaction order for component `X`. Orders default to the
  magnitudes of the reactant stoichiometric coefficients.
- `expression` - Safe arithmetic expression for `type=custom` or
  `type=custom_net`.
- `param_X` - Declared scalar `X` available to a custom expression.
- `conversion` - Fixed conversion (for conversion reactor)
- `basis` - Conversion basis component; defaults to the first net-consumed
  reactant
- `name` or `label` - Optional equilibrium-reaction report label

Example:
```
REACTIONS:
    CH3OH -> CO + 2 H2 | A=1e8, Ea=80000, Ea_unit=J/mol, rate_basis=concentration, concentration_unit=kmol/m3, pressure_unit=bar, rate_unit=kmol/m3/h
    A + B -> C | type=custom, A=1, Ea=0, Ea_unit=J/mol, concentration_unit=mol/L, pressure_unit=kPa, rate_unit=mol/L/min, expression=k*C['A']*C['B']/(1+K*C['A']), param_K=0.5
    A -> B | type=custom_net, A=1, Ea=0, Ea_unit=J/mol, concentration_unit=kmol/m3, pressure_unit=bar, rate_unit=kmol/m3/h, expression=k*(C['A']-C['B'])
```

Custom expressions permit only `+`, `-`, `*`, `/`, `**`, unary signs,
`exp(...)`, and `log(...)`. Available values are temperature `T`, the shared
molar gas constant `R` in J/(mol K), Arrhenius
constant `k`, concentrations `C['component']`, partial pressures
`p['component']`, fugacities `f['component']`, activities `a['component']`, and
raw liquid volume percentages `volpct['component']`, and declared `param_X`
scalars. `volpct` is available only for homogeneous liquids and is calculated
as 100 times the component's hypothetical unmixed pure-liquid volume divided
by the current thermodynamic mixture volume. Pure-liquid molar volumes and the
dynamic mixture density come from the active thermodynamic model and property
resolver; these apparent percentages need not sum to 100 when excess volume is
present. Attribute access, comprehensions, imports, and other Python execution
are not allowed.

`type=custom` defines a nonnegative forward kinetic prefactor. For an
irreversible equation it is the reaction rate directly; for a reversible
equation the thermodynamic driving factor described below supplies the net
direction. `type=custom_net` instead defines the complete signed net rate: a
positive result follows the written stoichiometric direction and a negative
result follows it in reverse. `custom_net` requires an irreversible `->` or
`=>` equation and does not receive an additional `Q/K` correction. Component
inventories must remain nonnegative in either direction.

For `<=>` and `<->` kinetic equations, the forward kinetic prefactor is
multiplied by the thermodynamic driving factor `1-Q/K`. `Q` uses the selected
phase activities and `K(T)` uses the shared formation-property architecture.
Reaction heat is not entered separately; thermal reactor balances use complete
stream enthalpies containing formation contributions.

### Reusable Top-Level Reaction Catalog

Reactions shared by several units may be defined once at top level:

```pfd
REACTIONS:
    methanol_equilibrium : CO + 2 H2 <=> CH3OH
    methanol_rate : CO + 2 H2 -> CH3OH | A=2.0e6, Ea=65000, Ea_unit=J/mol, rate_basis=fugacity, concentration_unit=kmol/m3, pressure_unit=bar, rate_unit=kmol/m3/h
```

Applicable units load a definition with an `@name` row and may mix references
with inline reactions:

```pfd
UNIT R-100 : CSTR
    volume = 5 [m3]
    T = 250 [C]
    phase = vapor
    REACTIONS:
        @methanol_rate
```

References currently apply to `Reactor`, `EquilibriumReactor`, `CSTR`,
`BatchReactor`, `PFR`, and `PackedBedReactor`. Undefined names, duplicates,
local parameter overrides on a reference,
and references from other unit types are errors. Catalog names and `@name`
references are preserved during `.pfd` and dictionary round trips. Reactive
distillation will use the same catalog contract when that model is implemented.

## Complete Example

```
#==============================================================================
# PROCESS FLOW DIAGRAM - Ethanol Flash Separation
#==============================================================================

PROCESS: Ethanol-Water Flash
VERSION: 1.0
THERMO_METHOD: UNIFAC

COMPONENTS:
    C2H5OH | Ethanol | MW=46.07
    H2O    | Water   | MW=18.02

STREAM Feed : FEED -> FLASH-1.in
    T = 90 [C]
    P = 1.5 [bar]
    F = 100 [kmol/h]
    x = C2H5OH:0.3, H2O:0.7

STREAM Vapor : FLASH-1.vap -> PRODUCT
STREAM Liquid : FLASH-1.liq -> PRODUCT

UNIT FLASH-1
    TYPE: Flash
    PORTS:
        in  : inlet
        vap : vapor_outlet
        liq : liquid_outlet
    PARAMS:
        T = 85 [C]
        P = 1 [bar]
```

Equivalent flash specifications can use vapor fraction or heat duty:

```
UNIT FLASH-1
    TYPE: Flash
    PORTS:
        in  : inlet
        vap : vapor_outlet
        liq : liquid_outlet
    PARAMS:
        P = 1 [bar]
        vapor_fraction = 0.40

UNIT FLASH-2
    TYPE: Flash
    PORTS:
        in  : inlet
        vap : vapor_outlet
        liq : liquid_outlet
    PARAMS:
        P = 1 [bar]
        heat_duty = 250 [kW]
```

## Validation Rules

Parsing reports every independently recoverable parse-time problem in one
`ParseError`, with a source line for each diagnostic. Likely misspellings in
closed vocabularies include a conservative `Did you mean ...?` suggestion.
This includes unknown or malformed top-level directives, unsupported solver or
thermodynamics methods, malformed section rows, stream fields, unit fields,
ports, and reaction parameter records, as well as unknown component properties,
property-correlation fields/equations, interaction models, and known
interaction-model fields. Structural validation then:

1. Rejects duplicate component symbols, unit IDs, stream IDs, and port IDs.
2. Rejects nonpositive explicit molecular weights.
3. Rejects stream compositions that reference undefined component symbols.
4. Rejects stream endpoints that reference undefined units or undefined
   declared port IDs.
5. Warns when a composition differs from 1.0 by more than 0.001; feed
   compositions are normalized when the simulation state is built.
6. Warns when `Cpg`, `Cpl`, `Cps`, or legacy `Cp_coeffs` lacks a complete
   temperature range and applies the documented property-specific default.
7. Parses every declared reaction against the process component symbols and,
   when every participating component has a molecular formula, rejects
   elementally unbalanced stoichiometry. A missing or unparseable formula
   produces an explicit warning instead of silently claiming balance.

Unit types are checked against the registered vocabulary during parsing;
compact headers additionally use that type to select their inferred ports.
Degrees of freedom, required feed state, parameter units, thermodynamic
capability, and operation-specific physical constraints are checked when
`Simulator` constructs or runs the flowsheet, not by the basic PFD validator.

## Version History

- **1.0 (2026-08 reactor revision)** - Added canonical reaction parsing,
  elemental validation, explicit inlet-basis conversion semantics, calculated
  yield/selectivity reporting, and isothermal/adiabatic/duty conversion-reactor
  energy balances.
- **1.0 (2026-08 equilibrium-reactor revision)** - Replaced the fixed 95%
  approach placeholder with constrained homogeneous vapor/liquid reaction
  equilibrium, thermodynamic activities, coupled thermal modes, phase-stability
  checks, boundary-equilibrium reporting, and compact recycle example.
- **1.0 (2026-08 kinetic-CSTR revision)** - Added explicit kinetic units,
  concentration/pressure/fugacity/activity rate states, safe custom rate
  expressions, thermodynamically reversible rates, a constrained homogeneous
  isothermal CSTR, and reusable top-level reaction catalogs with `@name`
  references.
- **1.0 (2026-08 adaptive-PFR revision)** - Replaced fixed-segment RK4 with
  adaptive RK/stiff integration of reaction extents, exact enthalpy flow,
  residence time, distributed specified/Darcy/Ergun pressure profiles,
  jacket heat transfer, depletion events, phase-stability checks, and complete
  solver/profile diagnostics.
- **1.0 (2026-08 thermal-CSTR revision)** - Added coupled adiabatic,
  specified-duty, and jacketed kinetic CSTR balances, explicit UA semantics,
  bounded discovery of multiple ignition/extinction steady states, mandatory
  branch selection, and thermal-stability reporting.
- **1.0 (2026-08 batch-reactor revision)** - Added adaptive homogeneous
  constant-pressure batch and basic semi-batch kinetics, exact transient
  enthalpy balances, two-of-three staggered-vessel scheduling, time-averaged
  continuous outlet mapping, and batch/fleet performance diagnostics.

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-12-11 | Initial stable release |
| 1.0 documentation revision | 2026-08-20 | Synchronized directives, component overrides, correlations, interactions, streams, ports, and registered unit operations with the current parser and simulator |
| 1.0 documentation revision | 2026-08-23 | Added solid Cp, solid volume/density, material-form selectors, and transition-aware kernel semantics |
| 1.0 documentation revision | 2026-08-24 | Added explicit permanent-solid component behavior, total- and fluid-basis phase fractions, particle defaults, and supported-unit routing rules |
| 1.0 documentation revision | 2026-08-25 | Added canonical kinetic-rate and CSTR contracts plus reusable named reaction definitions |
| 1.0 documentation revision | 2026-08-25 | Added adaptive homogeneous PFR thermal, hydraulic, event, and solver contracts |
| 1.0 documentation revision | 2026-08-25 | Added coupled thermal CSTR modes and multiple-steady-state branch contracts |
