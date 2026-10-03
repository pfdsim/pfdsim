# PFDSim

PFDSim is a Python chemical-process simulator built around human-editable
`.pfd` flowsheets. It solves steady-state processes, including recycle loops,
and produces calculated stream states, equipment performance, material and
energy balances, and readable `.pfr` reports. The command line and Python
library are the primary interfaces; an independent renderer produces process
flow diagrams from the same input.

## Features

- **Text-based flowsheets:** explicit connections, engineering units, compact
  or expanded equipment definitions, reusable reactions, and input validation
  with degrees-of-freedom checks.
- **Steady-state flowsheet solving:** sequential modular execution, automatic
  or explicit tear streams, Wegstein/Broyden acceleration, and recycle
  diagnostics. Units independent of a recycle block are not repeatedly solved.
- **Thermodynamics:** ideal, cubic EOS, predictive EOS, and activity-coefficient
  methods; VLE, LLE, and VLLE where supported; aqueous Henry-law treatment;
  vapor association and nonideal vapor corrections; water-only steam properties.
- **A traceable property system:** explicit overrides, bundled correlations
  and databases, local CoolProp support, empirical and molecular estimators,
  optional online lookup, persistent caching, and source/quality diagnostics.
- **Separation equipment:** flashes and decanters; shortcut and rigorous
  distillation, extraction, absorption, and stripping; competitive molecular-
  sieve adsorption. 
- **Reaction engineering:** conversion and equilibrium reactors, kinetic
  CSTRs, batch/semi-batch reactors, adaptive PFRs, and catalyst packed beds
  with effectiveness and hydraulic models.
- **Solids processing:** explicit solid inventories and particle-size
  distributions, pure-solid crystallization, finite-rate crystal/layer growth,
  layer sweating, pressure cake filtration, washing, and deliquoring.
- **Analysis and presentation:** stream and stage profiles, reactor profiles,
  balance audits, warnings, property-quality reports, structured Python
  results, and SVG/PDF/PNG flowsheet rendering.

## Under development

> **PFDSim is under active development.** The current package version is
> `0.3.1`. Models, data coverage, numerical robustness, and documentation
> are evolving. Check property sources, model assumptions, convergence, and
> material/energy closure before relying on a result for a new application.

The Python API is **unstable throughout 0.x**, including between minor
versions, and may change without compatibility layers. The `.pfd` format has
its own version (`1.0`), separate from the package version. From package
version `0.3.0` onward, incompatible `.pfd` changes must reject affected old
input with a descriptive migration error. This input-compatibility policy now
applies; it does not guarantee unchanged numerical results.

The solver is steady state. Batch reactors and cyclic equipment expose
continuous-equivalent throughput and duties; their internal transient or
cycle models do not make PFDSim a general dynamic flowsheet simulator.

## Installation

PFDSim requires **Python 3.12 or newer**. Run the following commands from a
source checkout's `pfdsim/` directory.

### Using uv

```bash
uv sync
```

This installs the editable package with the repository's locked dependencies
and development tools. For a runtime-only environment:

```bash
uv sync --no-dev
```

Run commands through `uv run`, or activate `.venv` and use the installed
`pfdsim` command directly.

### Using pip

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in
PowerShell. Pip installation uses the dependency constraints in
[`pyproject.toml`](pyproject.toml); it does not reproduce the uv lockfile.

### Optional dependencies

| Extra | Purpose | uv command |
| --- | --- | --- |
| `render` | PDF/PNG export through CairoSVG | `uv sync --extra render` |
| `web` | Process laboratory, phase atlas, accounts, and HTTP API | `uv sync --extra web` |
| `dipole-xtb` | Optional xTB molecular calculations | `uv sync --extra dipole-xtb` |
| `dipole-pvdz` | Optional higher-level molecular calculations using PySCF | `uv sync --extra dipole-pvdz` |

With pip, use, for example, `python -m pip install -e '.[render]'`.
Extras can be combined. All diagram formats require the system **Graphviz
`dot` executable** on `PATH`; PDF/PNG export also requires the Cairo system
library where it is not already available. Ordinary simulation does not
require Graphviz or the web extra.

Some molecular property calculations require optional dependencies and can
be expensive on their first use. See the property controls below for how to
prevent new computational jobs.

## Quick start

### Run an existing flowsheet

From the repository directory:

```bash
# Print convergence, balances, final outlets, warnings, and equipment duties.
uv run pfdsim examples/simple_flash.pfd

# Save a full report in a separate output directory.
mkdir -p outputs
uv run pfdsim examples/simple_flash.pfd -o outputs/simple_flash_result.pfr

# Print the full report instead of the brief summary.
uv run pfdsim examples/simple_flash.pfd --pfr

# Show solver progress on standard error.
uv run pfdsim examples/simple_flash.pfd --verbose
```

`--output` with no path writes beside the input using the `.pfr` extension.
Reports overwrite an existing destination, so choose a separate path when
preserving a previous run or a bundled example report. The CLI returns `0`
for a converged simulation and a nonzero status for nonconvergence or an
input/simulation error. A converged result may still contain warnings.

### Write a small flowsheet

Save this as `flash.pfd`:

```pfd
PROCESS: Ethanol-Water Flash
VERSION: 1.0
ONLINE_LOOKUP: false
ALLOW_COMPUTATION: false
THERMO_METHOD: UNIFAC

COMPONENTS:
    water   | Water
    ethanol | Ethanol

STREAM Feed : FEED -> HEAT-1.in
    T = 25 [C]
    P = 1 [bar]
    F = 100 [kmol/h]
    x = water:0.6, ethanol:0.4
STREAM Heated : HEAT-1.out -> FLASH-1.in
STREAM Vapor : FLASH-1.vap -> PRODUCT
STREAM Liquid : FLASH-1.liq -> PRODUCT

UNIT HEAT-1 : Heater
    T_out = 80 [C]

UNIT FLASH-1 : Flash
    T = 80 [C]
    P = 1 [bar]
```

Then run `uv run pfdsim flash.pfd -o flash_result.pfr`. Feed conditions are
specified once; the connected equipment calculates the remaining streams.
The offline controls make this introductory input use local data without
starting new expensive property calculations.

### Use the Python API

```python
from pathlib import Path
from pfdsim.simulator import Simulator

sim = Simulator.from_file("examples/simple_flash.pfd")
result = sim.run(max_iterations=100, tolerance=1e-4)

print("Converged:", result.converged)
print("Mass balance error (%):", 100 * result.mass_balance_error)
print("Energy balance error (%):", 100 * result.energy_balance_error)
print("Vapor composition:", result.streams["Vapor"].composition)
print("Warnings:", result.warnings)

Path("outputs").mkdir(exist_ok=True)
sim.write_results("outputs/simple_flash_result.pfr")
results_dict = sim.get_results_dict()
```

`Simulator.from_string()` accepts PFD text directly. `initialize()` prepares
property packages, the execution graph, and applicable compiled backends;
`run()` calls it automatically. Initialization is idempotent for an unchanged
configuration. `simulate_pfd()` returns a result dictionary;
`simulate_file()` returns a `SimulationResult` and writes a report, using an
input-adjacent `.pfr` path when no output path is supplied.

## Property system

### Identity, overrides, and resolution

A component row separates the **process symbol** from the **lookup identity**:

```pfd
COMPONENTS:
    solvent | Acetone | MW=58.08
    water   | 7732-18-5
```

Streams and reactions use `solvent` and `water`. Only the second field is used
for external identity resolution. Supported identities include unique names,
CAS numbers, unambiguous formulas, SMILES, InChI, and resolvable InChIKeys.
Ambiguous molecular formulas are rejected rather than assigned to an arbitrary
isomer. Third-field identity/property overrides do not replace the lookup
identifier.

`PropertyResolver` is the shared authority behind `ChemicalProperties` and
the thermodynamic packages. Resolution follows a **property-specific** source
policy rather than one universal database order:

1. Explicit PFD values and correlations take precedence for their properties.
2. Supported local sources include curated component records, CoolProp,
   Perry and Smith textbook data, tabulated data, and specialized databases.
   For supported pure fluids, CoolProp supplies a preferred consistent bundle
   of critical and saturation properties unless the PFD overrides them.
3. Depending on the property and its applicability, missing data may be
   supplied by fitted correlations, corresponding-states or group-contribution
   methods, molecular calculations, or optional online sources such as
   PubChem and NIST.
4. Unavailable properties can produce warnings or errors when required by the
   chosen model. An estimate does not establish experimental validity.

The resolver covers critical and phase-transition properties, vapor pressure,
phase-specific heat capacities and their integrals, vaporization/fusion
enthalpies, formation thermochemistry, liquid/solid volumes, viscosity,
thermal conductivity, surface tension, dipoles, and molecular radii of gyration.
Mixture properties depend on the selected thermodynamic and transport models.

### Implemented property estimation methods

The main implemented estimators and predictive correlations include:

| Property | Methods |
| --- | --- |
| Normal boiling point and critical properties | Nannoolal group contribution; empirical formula/structure fallbacks; Lee–Kesler acentric-factor estimation |
| Vapor pressure | Nannoolal Part 3, Ambrose–Walton, and boundary-conditioned Clapeyron completion of canonical curves |
| Vaporization enthalpy | Watson scaling, Nannoolal vapor-pressure slopes with PR phase-volume correction, corresponding-states estimation, and Trouton's rule |
| Heat capacity | Empirical atom-increment Shomate and GFN2-xTB rigid-rotor/harmonic-oscillator (RRHO) estimates for ideal gases; Rowlinson–Bondi and hydrogen-bond-donor group estimates for liquids |
| Liquid volume/density | Rackett with fitted or Yamada–Gunn parameters, PTV/PR EOS fallbacks, and atomic-increment estimates |
| Viscosity | Hsu and Nannoolal Part 4 for liquids; Reichenberg and Yoon–Thodos for dilute vapors; Lucas liquid-pressure and Jossi–Stiel–Thodos dense-gas corrections |
| Thermal conductivity | Modified Pachaiyappan, selected-domain Baroncini, and locally refitted Govender for liquids; PFDSim-modified Stiel–Thodos for dilute gases |
| Formation thermochemistry | Domalski–Hearing group additivity for gas-phase formation enthalpy and absolute entropy |
| Surface tension | Knotts/Perry Parachor group contribution |
| Molecular properties | GFN2-xTB geometry/dipoles and optional PBE0/aug-cc-pVDZ dipoles; geometry-derived radii of gyration |
| Mixture properties | Grunberg–Nissan, Jouyban–Acree, and UNIFAC-VISCO liquid viscosity; Butler/UNIFAC surface tension |

Selection follows the property-specific resolution policy. Structural and
temperature/pressure applicability limits, local model refinements, and input
quality checks still apply; molecular calculations may require optional extras.

### Correlations and canonical vapor pressure

`PROPERTY_CORRELATIONS:` supplies component correlations with declared
units, equations, coefficients, and validity ranges. `INTERACTION_PARAMETERS:`
supplies explicit binary model parameters. These are different from scalar
component overrides; see the [PFD specification](docs/pfd_format_v1.0.md) for
accepted fields and equation forms.

Vapor-pressure resolution constructs and caches one canonical curve per
component and relevant input configuration. It selects source segments,
completes missing ranges where supported, and fits a common equation while
retaining provenance and fit diagnostics. Qualified normal-boiling and
critical anchors constrain the fit. Thermodynamic hot paths reuse its
coefficients rather than repeatedly choosing a provider at each temperature.

`PSAT_MINIMUM_PRESSURE` controls construction of the lower domain; its default
is `0.001 bar` (100 Pa). Numerical continuation outside the canonical interval
is not measured saturation data, and a supercritical continuation is not a
physical liquid-vapor coexistence curve. See the
[canonicalization guide](docs/vapor_pressure_canonicalization.md) and
[construction policy](docs/psat_construction_policy_and_justification.md).

### Lookup and computation controls

```pfd
ONLINE_LOOKUP: false
ALLOW_COMPUTATION: false
PSAT_MINIMUM_PRESSURE: 0.001 [bar]
```

- `ONLINE_LOOKUP` defaults to `true`. Setting it to `false` disables online
  property queries and the online component-lookup route while retaining
  bundled/local data and explicit input properties.
- `ALLOW_COMPUTATION` defaults to `true`. Setting it to `false` prevents new
  expensive computational-property jobs, such as geometry optimization and
  vibrational Hessians. Existing cached computational results and inexpensive
  empirical estimators remain eligible.
- The two controls are independent. Disabling online lookup alone does not
  prohibit local quantum-chemical work, and disabling computation alone does
  not prohibit network lookup.

Persistent property and computational caches reduce repeated work, including
repeated unsuccessful lookups where supported. Pin the package revision,
dependencies, property overrides, and relevant input data when reproducing a
calculation; an input file alone does not freeze every external data source.

### Provenance and quality

Resolved values carry `value`, `source`, `method`, `quality`, and `notes`.
Quality is a diagnostic ranking from 0 to 1, **not a probability, uncertainty
interval, or error bound**. Derived properties inherit input quality and
method penalties. Explicit input is authoritative and receives quality 1.0;
that records the override policy, not independent validation of its value.

For example, from the installed library:

```python
from pfdsim.property_resolver import PropertyResolver

resolver = PropertyResolver()
psat = resolver.resolve_vapor_pressure("Water", 373.15, allow_online=False)
print(psat.value, psat.source, psat.method, psat.quality, psat.notes)
```

The full `.pfr` includes a `QUALITY_REPORT` of flagged property resolutions,
with context and temperature information where available. Consult
[property resolution design](docs/property_resolution_design.md),
[heat-capacity coverage](docs/Cp_matrix.md), and the
[Henry database guide](docs/henry_database.md) for detailed policies. Some
historical design notes describe intermediate implementations; current runtime
behavior and the PFD reference determine available input behavior.

## Thermodynamics

Set `THERMO_METHOD:` explicitly for a new application. `IDEAL` is the default;
its presence does not imply that a real mixture behaves ideally.

| Family | Available methods and purpose |
| --- | --- |
| Ideal | `IDEAL`: ideal vapor and Raoult-law liquid equilibrium |
| Steam | `STEAM` / `IF97`: CoolProp IF97 for water-only flowsheets |
| Cubic EOS | `RK`, `SRK`, `PR`; Boston-Mathias (`RKS-BM`, `PR-BM`), Mathias-Copeman (`SRK-MC`, `PR-MC`), Twu (`SRK-TWU`, `PR-TWU`), and Stryjek-Vera (`PRSV1`, `PRSV2`) variants |
| Predictive GE–EOS | `PSRK`: published PSRK-2005 parameters; `RKSMHV2`: SRK with MHV2 mixing and Lyngby UNIFAC |
| Molecular activity | `NRTL`, `UNIQUAC`, with bundled or explicit binary parameters and optional initialization-time estimation |
| Group-contribution activity | `UNIFAC`, `UNIFAC2`, `UNIFDMD`, `UNIFM2`, `UNIFNIST`, `UNIFLBY` |
| Nonideal vapor corrections | Supported activity methods with `-RK`, `-PR`, `-BV`, `-HOC`, or `-VDM` variants; availability depends on the base model |

For `-BV`, `correlation=` selects `TSONOPOULOS` (default), `PITZER-CURL`,
`ABBOTT`, or Hayden–O'Connell (`HOC`). The `-HOC` aliases select the HOC
provider. HOC includes association and acid chemical-theory treatment where
applicable; `-VDM` provides the separate vapor-dimerization model. These
corrections operate on the nominal flowsheet component basis rather than
requiring internal dimers to be declared as new process components.

### Phase equilibrium policy

| `FLUID_PHASE_MODEL` | Behavior |
| --- | --- |
| `VLE` (default) | Conventional vapor/one-liquid flash without automatic liquid stability testing |
| `VL(L)E` | VLE followed by a local spinodal check; unstable liquid triggers VLLE, but a locally stable metastable liquid may be retained |
| `VLLE` | Global liquid-stability/VLLE search permitting vapor and two liquid phases |

Automatic LLE-aware modes require a compatible activity-coefficient method.
Cubic EOS, PSRK, RKSMHV2, IDEAL, and STEAM are not routed through the current
LLE/VLLE engine. `Flash3` and `Decanter` explicitly separate liquid phases;
an ordinary `Flash` retains both liquids in one nonvapor outlet. Discovering
a phase and routing it to a separate outlet are distinct operations.

Supported aqueous VLE calculations use a Henry-law standard state for dilute
noncondensables. The current pure-water Henry path is VLE-only; incompatible
Henry/LLE-aware calculations fail explicitly. Steam calculations require
water alone. Published group assignments, interaction coverage, and pure-
component parameter conventions constrain predictive model applicability.
See [PSRK](docs/psrk.md), [Lyngby/RKSMHV2](docs/lyngby.md), and
[Henry treatment](docs/henry_database.md).

### Thermodynamic scopes and interaction estimation

Different process sections can use independent packages:

```pfd
THERMO_METHOD: UNIQUAC-VDM
THERMO_SCOPES:
    extraction | method=NRTL
    recovery   | method=UNIQUAC, inherit=global

UNIT X-301 : RigorousExtractor
    thermo_scope = extraction
```

This fragment illustrates scope selection; the unit still needs its normal
connections and specifications. A scope without `inherit` is isolated.
Inheritance imports interaction records, and child records replace matching
parent records. Crossing a scope boundary preserves temperature, pressure,
and component flows while recalculating properties in the receiving package.
The report records the resulting enthalpy-flow correction separately from
physical equipment duty.

Missing NRTL/UNIQUAC interactions can be fitted once from a chosen
UNIFAC-family source and frozen for the run:

```pfd
INTERACTION_ESTIMATION:
    NRTL | source=UNIFDMD, policy=missing_only, parameter_order=source, alpha=0.3, Tmin=293.15 [K], Tmax=423.15 [K]
```

Explicit PFD parameters remain authoritative. Estimation has a stated fit
range, extrapolation policy, and fit diagnostics; it does not turn a predictive
estimate into measured binary data. Scope-specific estimation is also supported.

## Unit operations

The following names are the canonical PFD unit types. Common aliases are
listed in the [PFD reference](docs/pfd_format_v1.0.md).

| Area | Unit types | Main capabilities |
| --- | --- | --- |
| Mixing and splitting | `Mixer`, `Splitter` | Multiple feeds, proportional or component splits |
| Pressure and transport | `Pump`, `Compressor`, `Expander`, `Valve`, `Pipe` | Pressure changes, work, throttling, single-phase pipe hydraulics |
| Heat transfer | `Heater`, `Cooler`, `HeatExchanger` | Duty/outlet specifications; exchanger sizing/rating and supported transfer correlations |
| Phase separation | `Flash`, `Flash3`, `Decanter` | VLE and compatible LLE/VLLE separation with explicit phase routing |
| Distillation | `ShortcutDistillation`, `McCabeThieleDistillation`, `CMODistillation`, `RigorousDistillation` | Shortcut, binary graphical/stage, CMO, and rigorous MESH models |
| Liquid extraction | `ShortcutExtractor`, `RigorousExtractor` | Shortcut and multistage equilibrium LLE extraction |
| Gas–liquid contact | `Absorber`, `Stripper`, `RigorousAbsorber`, `RigorousStripper` | Shortcut and rigorous equilibrium-stage absorption/stripping |
| Adsorption | `MolecularSieveDryer` | Competitive equilibrium adsorption, built-in/custom isotherms, adsorbent-flow or target sizing |
| Reactions | `Reactor`, `EquilibriumReactor`, `CSTR`, `BatchReactor`, `PFR`, `PackedBedReactor` | Conversion, homogeneous equilibrium, kinetic stirred/batch/axial and catalytic models |
| Solids | `Crystallizer`, `LayerCrystallizer`, `Filter` | Pure-solid equilibrium/growth, deposited layers/sweating, PSD-dependent cake filtration and washing |

**Rigorous distillation** solves coupled material, equilibrium, summation, and
energy (MESH) equations, with pressure profiles, multiple feeds, detailed 
convergence settings, stage efficiencies, and total, partial, or mixed 
condensers. VLE supports side draws. VLLE stages retain two liquid compositions
with an aggregate downward liquid flow; the top liquids can have different 
withdrawal/reflux fractions. Specified-temperature or subcooled total condensers
use their actual liquid equilibrium and enthalpy in the coupled balances. VLLE
side draws remain unsupported. 

**Reactors** support unit-specific thermal modes and explicitly declared rate
bases/units. Kinetic PFRs provide adaptive axial integration and profiles;
packed beds add catalyst-mass coordinates, supported pellet-effectiveness
models, and Ergun pressure drop. Batch/semi-batch models report time histories,
cycle/fleet scheduling, and continuous-equivalent outlet and heat-duty bases.
Homogeneous equilibrium reactors do not provide arbitrary multiphase reactive
equilibrium.

**Molecular sieves** use competitive isotherms and IAST where supported, with
fugacities from the selected property package. They model an equilibrium
contact with fresh/regenerated adsorbent rather than a resolved breakthrough
or pressure-swing cycle. Adsorption enthalpy limitations are reported explicitly.
See [molecular sieves](docs/molecular_sieves.md).

**Solids** distinguish `permanent_solid` inventory from
`conventional_with_solid` (alias `three_phase`) accounting for a conventional
component with an assigned solid portion. These declarations do not create a
general soluble-solid equilibrium model. Crystallizers and equilibrium cake
washing supply their own pure-solid equilibrium contracts; layer sweating
has a restricted finite-rate solid-solution model.

Permanent solids are supported by mixers, splitters, heaters, coolers,
specification-driven heat exchangers, ordinary flashes, crystallizers, layer
crystallizers, and filters. Other equipment rejects them explicitly.
Heat-exchanger rating/automatic film-coefficient modes reject solid-bearing
streams. See [filtration](docs/filtration.md) and
[equilibrium warm/melt washing](docs/equilibrium_washing.md) for hydraulic,
particle, and equilibrium assumptions.

## Analysis and reporting

The default CLI summary shows convergence, recycle iterations, overall
balance errors, final outlet conditions/compositions, unit heat/work duties,
and warnings/errors. The full `.pfr` adds:

- The original flowsheet and simulation metadata.
- Stream thermodynamic states, mole/mass compositions, active phase inventories,
  and solid/particle information where present.
- Unit performance, column stage profiles, reactor axial/transient profiles,
  and model-specific diagnostics where produced.
- Material and energy reconciliation, including reaction contributions,
  thermodynamic-scope corrections, and additional equipment accounting terms.
- Recycle diagnostics, numerical warnings, and flagged property provenance.

Heat and work are reported in engineering units, including kW; internal
molar stream states use K, bar, kmol/h, kJ/kmol enthalpy, and kJ/(kmol·K)
entropy. Balance error attributes are fractional values; the CLI and report
convert them to percent. Check both solver convergence and physical closure.

`sim.get_results_dict()` provides metadata, summary, streams, units, recycle
information, scope corrections, warnings, and errors for further processing.
It does not include every section of the text report; the flagged
property-quality section is generated for the `.pfr`. See the
[PFR specification](docs/pfr_format_v1.0.md) for report conventions.

### Render a flowsheet

```bash
uv run python -m pfdsim.render examples/simple_flash.pfd -o outputs/flash.svg
uv run python -m pfdsim.render examples/simple_flash.pfd -o outputs/flash.pdf
uv run python -m pfdsim.render examples/simple_flash.pfd -o outputs/flash.png --dpi 200 --stream-table
```

Create `outputs/` first, as in the quick start. SVG is the default format;
PDF/PNG require the `render` extra. Existing diagram outputs require `--force`
to replace, and the input file is protected.

Rendering uses the flowsheet parser without initializing thermodynamics or
running a simulation. Its optional stream table contains **input
specifications**, not calculated results. Diagrams are engineering process
schematics, not P&IDs or a certified symbol library. See the
[rendering guide](docs/rendering.md).

The optional Flask process laboratory can be started with
`uv run --extra web python app.py` after installing `web`. It includes a visual
editor, tabbed process configuration, local/account autosaves, and a phase atlas
with selectable thermodynamics and binary/ternary LLE/VLLE views. See the
[web interface guide](docs/web_interface.md) for accounts, CPU budgets, and
the two-HTTP/two-calculation-worker deployment.

## Flowsheet specification and syntax

The [PFD format specification](docs/pfd_format_v1.0.md) is the detailed
reference for fields, supported units, equipment specifications, aliases,
reaction conventions, and validation rules. The essential structure is:

| Declaration | Purpose |
| --- | --- |
| `PROCESS:`, `VERSION:`, `DESCRIPTION:`, `AUTHOR:`, `DATE:` | Optional metadata; `VERSION: 1.0` identifies the input format |
| `THERMO_METHOD:` | Global property method and supported method options |
| `FLUID_PHASE_MODEL:` | Global unconstrained fluid-equilibrium policy |
| `ONLINE_LOOKUP:`, `ALLOW_COMPUTATION:`, `PSAT_MINIMUM_PRESSURE:` | Property-resolution policies |
| `RECYCLE_METHOD:`, `TEAR_STREAMS:`, `RECYCLE_TRACE_TOLERANCE:` | Recycle convergence controls |
| `THERMO_SCOPES:` | Named independent/inherited property contexts |
| `COMPONENTS:` | Process symbols, lookup identities, and scalar overrides |
| `PROPERTY_CORRELATIONS:` | Pure-component correlation overrides |
| `INTERACTION_PARAMETERS:`, `INTERACTION_ESTIMATION:` | Explicit or initialization-fitted interactions |
| `STREAM name : source -> destination` | Connections and feed/tear specifications |
| `UNIT id : Type` | Equipment and parameters in compact form |
| `REACTIONS:` | Reusable top-level catalog or unit-local reaction definitions |

Top-level keywords are case-sensitive. Indentation defines block contents;
an unindented noncomment line ends the block. `#` starts a full-line comment,
or an inline comment after whitespace outside quotes. Quote field strings
containing separators or spaces as required by the format. Define components
before their correlation/interaction records.

### Streams, units, and ports

```pfd
STREAM Feed : FEED -> F-1.in
    T = 25 [C]
    P = 1 [bar]
    F_mass = 1000 [kg/h]
    w = water:0.8, ethanol:0.2
STREAM Vapor : F-1.vap -> PRODUCT
STREAM Liquid : F-1.liq -> PRODUCT

UNIT F-1 : Flash
    T = 80 [C]
    P = 1 [bar]
```

This fragment assumes the components have already been declared. Feeds need
pressure, flow, composition, and temperature or vapor fraction. `F` and `x`
are molar flow and mole fractions; `F_mass` and `w` use a mass basis and need
component molecular weights for conversion. Fractions are normalized before
simulation. Use explicit units to avoid unitless-temperature heuristics.
Gauge pressure, when accepted, is converted to absolute pressure internally.

Internal and product streams normally have no state specifications. Values
on a declared tear stream seed the recycle guess. Compact endpoints include
`STREAM Feed : -> F-1.in` and `STREAM Product : F-1.liq`.

The expanded equivalent of a compact unit uses `TYPE:`, `PORTS:`, and
`PARAMS:`. Standard semantic port aliases and clockwise numeric ports are
supported; explicit `PORTS:` declarations are authoritative for custom names.
Port meaning and direction depend on equipment type, so consult the reference
for columns and other multiport units.

### Recycles and reactions

```pfd
RECYCLE_METHOD: BROYDEN | stagnation_iterations=8, fallback_damping=1
TEAR_STREAMS: Recycle
```

Available methods are `WEGSTEIN` (default), `BROYDEN`, and `DIRECT`. Automatic
tears are available when none are specified. Tear convergence uses flow,
component flows, pressure, and enthalpy; trace-component treatment and
method-specific options are documented in the reference.

Reactions may be declared once and referenced by name:

```pfd
REACTIONS:
    methanol_conversion : CO + 2 H2 -> CH3OH | conversion=0.25

UNIT R-1 : Reactor
    T = 250 [C]
    REACTIONS:
        @methanol_conversion
```

This fragment needs the component and stream declarations for a complete
flowsheet. Conversion, equilibrium, and kinetic reactions use different
required fields. Kinetic definitions explicitly declare activation-energy,
concentration/pressure, rate units, and the rate basis. The same catalog can
be referenced by the supported reactor types; undefined names, duplicate
names, and local parameter overrides on a reference are errors.

## Examples and tutorials

The repository contains **52 executable `.pfd` examples**. Run any of them with
`uv run pfdsim examples/<name>.pfd`; use a separate output path to preserve
bundled reports. Selected `.pfr` files are recorded worked results, not a
promise that every dependency revision produces identical output.

### Suggested tutorial path

There is no separate `tutorials/` directory. These examples and guides provide
a progression through the simulator:

1. **Basic syntax and reports:** [simple flash](examples/simple_flash.pfd),
   then [UNIFAC flash](examples/unifac_flash.pfd), with the
   [PFD](docs/pfd_format_v1.0.md) and [PFR](docs/pfr_format_v1.0.md) references.
2. **Equilibrium stages:**
   [ethanol distillation](examples/ethanol_distillation_rigorous.pfd), then
   [benzene/toluene](examples/benzene_toluene_20_stage_distillation_nrtl.pfd).
3. **Model choice and phase stability:**
   [NRTL/UNIQUAC comparison](examples/isopropanol_water_nrtl_uniquac_comparison.pfd),
   [local spinodal handling](examples/adaptive_spinodal_water_toluene.pfd),
   and [global VLLE](examples/global_vlle_water_methanol_benzene.pfd).
4. **Recycles and integrated separations:**
   [equilibrium methanol recycle](examples/equilibrium_methanol_synthesis_recycle.pfd)
   and [solvent-recovery extraction](examples/3methylpyridine_ether_extraction_recycle.pfd).
5. **Reaction models:** [CSTR saponification](examples/saponification_cstr.pfd),
   [PFR decomposition](examples/methanol_decomposition_pfr.pfd), and
   [batch esterification](examples/ethyl_acetate_batch_synthesis.pfd).
6. **Solids and adsorption:**
   [cake filtration](examples/cake_filtration_washing.pfd),
   [equilibrium warm washing](examples/equilibrium_warm_melt_washing.pfd), and
   [competitive adsorption](examples/competitive_13x_adsorption.pfd), alongside
   the filtration, washing, and molecular-sieve guides linked above.

[`scripts/extraction_example.py`](scripts/extraction_example.py) is a direct
Python example: it creates a UNIFAC package, constructs streams from mass
flows, solves a rigorous extractor, and prints outlet/component flows and
performance. Run it with `uv run python scripts/extraction_example.py`.

### Flash, thermodynamics, utilities, and transport

| Example | Demonstrates |
| --- | --- |
| [simple_flash.pfd](examples/simple_flash.pfd) | Heated ethanol/water flash with default ideal thermodynamics and expanded syntax |
| [unifac_flash.pfd](examples/unifac_flash.pfd) | Ethanol/water VLE using UNIFAC |
| [ethanol_water_mhv2.pfd](examples/ethanol_water_mhv2.pfd) | RKSMHV2 predictive EOS flash |
| [butanol_water_lle.pfd](examples/butanol_water_lle.pfd) | UNIFAC liquid–liquid decanting |
| [adaptive_spinodal_water_toluene.pfd](examples/adaptive_spinodal_water_toluene.pfd) | Local spinodal checking and multiphase-liquid retention through ordinary equipment |
| [global_vlle_water_methanol_benzene.pfd](examples/global_vlle_water_methanol_benzene.pfd) | Global stability search and vapor/two-liquid inventory in an ordinary flash |
| [cryogenic_air_separation_rks_bm.pfd](examples/cryogenic_air_separation_rks_bm.pfd) | Cryogenic air cooling and rigorous separation using RKS-BM |
| [methane_claude_liquefaction_pr.pfd](examples/methane_claude_liquefaction_pr.pfd) | PR methane liquefaction with heat exchange, expansion, and recycle |
| [simple_rankine_cycle_steam.pfd](examples/simple_rankine_cycle_steam.pfd) | Water-only Rankine cycle using IF97 steam properties |
| [ethanol_water_inclined_pipe_unifac.pfd](examples/ethanol_water_inclined_pipe_unifac.pfd) | Velocity-sized inclined pipe, pressure drop, and outlet expansion |

### Distillation and coupled recovery

| Example | Demonstrates |
| --- | --- |
| [ethanol_distillation_rigorous.pfd](examples/ethanol_distillation_rigorous.pfd) | Rigorous ethanol/water purification with UNIFAC |
| [benzene_toluene_20_stage_distillation_nrtl.pfd](examples/benzene_toluene_20_stage_distillation_nrtl.pfd) | Twenty-stage NRTL distillation |
| [biosteam_mesh_hydrocarbon_distillation.pfd](examples/biosteam_mesh_hydrocarbon_distillation.pfd) | Translation of the BioSTEAM hydrocarbon MESH example using PR |
| [isopropanol_water_nrtl_uniquac_comparison.pfd](examples/isopropanol_water_nrtl_uniquac_comparison.pfd) | Parallel scoped NRTL and UNIQUAC columns on the same feed |
| [isopropanol_diisopropyl_ether_distillation_nrtl_estimated.pfd](examples/isopropanol_diisopropyl_ether_distillation_nrtl_estimated.pfd) | NRTL parameters fitted from UNIFDMD at initialization and frozen |
| [methanol_diethyl_ether_5bar_nrtl_rk.pfd](examples/methanol_diethyl_ether_5bar_nrtl_rk.pfd) | Rigorous 5-bar distillation with NRTL liquid and RK vapor correction |
| [vinegar_concentration_uniquac_vdm.pfd](examples/vinegar_concentration_uniquac_vdm.pfd) | Acetic-acid concentration with vapor dimerization |
| [mixed_acid_dehydration_uniquac_vdm.pfd](examples/mixed_acid_dehydration_uniquac_vdm.pfd) | Multicomponent acid dehydration with vapor association |
| [ethanol_benzene_azeotropic_distillation_rigorous.pfd](examples/ethanol_benzene_azeotropic_distillation_rigorous.pfd) | Rigorous azeotropic columns, overhead decanting, and molecular-sieve polishing |
| [butanol_water_phase_selective_distillation.pfd](examples/butanol_water_phase_selective_distillation.pfd) | Coupled VLLE column with water-rich withdrawal and butanol-rich reflux |
| [ethanol_pressure_swing_recycle_wasteful.pfd](examples/ethanol_pressure_swing_recycle_wasteful.pfd) | Deliberately wasteful pressure-swing process illustrating Broyden recycle solving |
| [methanol_ethanol_light_gas_cleanup_compact.pfd](examples/methanol_ethanol_light_gas_cleanup_compact.pfd) | Compact port/unit syntax, PSRK degassing, and alcohol recovery |

### Extraction, absorption, and stripping

| Example | Demonstrates |
| --- | --- |
| [acrylic_acid_rigorous_extraction.pfd](examples/acrylic_acid_rigorous_extraction.pfd) | Multistage LLE extraction of acrylic acid |
| [pyridine_ether_extraction.pfd](examples/pyridine_ether_extraction.pfd) | Pyridine extraction with ether using UNIFNIST |
| [3methylpyridine_ether_extraction_recycle.pfd](examples/3methylpyridine_ether_extraction_recycle.pfd) | Rigorous extraction, solvent recovery/recycle, and polishing distillation |
| [ethylene_ethane_isoparaffin_absorption.pfd](examples/ethylene_ethane_isoparaffin_absorption.pfd) | RKSMHV2 equilibrium-stage gas absorption into mixed C6/alcohol solvent |
| [ethanol_ether_partial_condensation_absorption.pfd](examples/ethanol_ether_partial_condensation_absorption.pfd) | Henry-aware partial condensation followed by adiabatic water absorption |
| [trace_organic_water_stripping_isothermal_unifnist.pfd](examples/trace_organic_water_stripping_isothermal_unifnist.pfd) | Isothermal air stripping of dilute organics from water |

### Reactions and integrated synthesis

| Example | Demonstrates |
| --- | --- |
| [ammonia_synthesis.pfd](examples/ammonia_synthesis.pfd) | Single-pass conversion reactor, cooling, and ammonia flash recovery |
| [ammonia_oxidation.pfd](examples/ammonia_oxidation.pfd) | Adiabatic kinetic PFR for ammonia oxidation |
| [saponification_cstr.pfd](examples/saponification_cstr.pfd) | Liquid-phase kinetic CSTR |
| [methanol_decomposition_pfr.pfd](examples/methanol_decomposition_pfr.pfd) | Isothermal PFR with first-order kinetics |
| [rk_thermodynamics_pfr.pfd](examples/rk_thermodynamics_pfr.pfd) | Methanol-decomposition PFR with RK and power-law kinetics |
| [cstr_pfr_comparison.pfd](examples/cstr_pfr_comparison.pfd) | CSTR/PFR comparison for ethanol dehydration |
| [jacketed_cstr_ignition_extinction.pfd](examples/jacketed_cstr_ignition_extinction.pfd) | Low/high thermal branches with shared reaction definitions |
| [ethyl_acetate_batch_synthesis.pfd](examples/ethyl_acetate_batch_synthesis.pfd) | Batch esterification with NRTL and continuous-equivalent scheduling |
| [methanol_synthesis.pfd](examples/methanol_synthesis.pfd) | PR conversion-based synthesis, recycle/purge, and product degassing |
| [methanol_synthesis_psrk.pfd](examples/methanol_synthesis_psrk.pfd) | Corresponding synthesis flowsheet using published PSRK parameters |
| [equilibrium_methanol_synthesis_recycle.pfd](examples/equilibrium_methanol_synthesis_recycle.pfd) | Homogeneous reaction equilibrium, condensation, purge, and syngas recycle |
| [ethylene_oxide_simple.pfd](examples/ethylene_oxide_simple.pfd) | Single-pass oxidation and recovery with NRTL-PR |
| [ethylene_oxide.pfd](examples/ethylene_oxide.pfd) | Conversion-based ethylene-oxide process with recycle and purge |
| [haber_bosch_full.pfd](examples/haber_bosch_full.pfd) | Cryogenic nitrogen preparation, compression, staged kinetic ammonia synthesis, condensation, and recycle |
| [lactic_acid_dehydration_pbr.pfd](examples/lactic_acid_dehydration_pbr.pfd) | Catalytic packed bed, acid association, scoped extraction, and downstream recovery/recycle |

### Adsorption and solids

| Example | Demonstrates |
| --- | --- |
| [ethanol_3a_molecular_sieve_drying.pfd](examples/ethanol_3a_molecular_sieve_drying.pfd) | Liquid ethanol drying on regenerated 3A sieve |
| [dcm_3a_molecular_sieve_drying.pfd](examples/dcm_3a_molecular_sieve_drying.pfd) | Dichloromethane/water drying on 3A |
| [air_3a_molecular_sieve_drying.pfd](examples/air_3a_molecular_sieve_drying.pfd) | Humid-air drying on 3A with PR |
| [competitive_13x_adsorption.pfd](examples/competitive_13x_adsorption.pfd) | Competitive CO2/methane/nitrogen adsorption on 13X |
| [propane_propylene_4a_adsorption.pfd](examples/propane_propylene_4a_adsorption.pfd) | Impurity adsorption and hydrocarbon coadsorption on 4A |
| [permanent_solid_slurry_operations.pfd](examples/permanent_solid_slurry_operations.pfd) | Explicit permanent solids retained through ordinary fluid/solid operations |
| [permanent_solid_global_vlle_flash.pfd](examples/permanent_solid_global_vlle_flash.pfd) | Solid inventory retained alongside global fluid VLLE |
| [cake_filtration_washing.pfd](examples/cake_filtration_washing.pfd) | Pressure cake formation, particle capture, washing, and deliquoring |
| [equilibrium_warm_melt_washing.pfd](examples/equilibrium_warm_melt_washing.pfd) | Crystallization followed by coupled adiabatic warm/melt cake washing |

These examples illustrate syntax and models, including simplified assumptions,
empirical estimates, and deliberately challenging cases. They are not all
validated industrial designs; inspect each file's comments and property
choices before adapting it.

## Key design decisions and highlights

These choices explain PFDSim's workflow compared with GUI-centered,
fixed-database, or wholly equation-oriented simulator workflows. They are
architectural tradeoffs, not claims of better accuracy or broader coverage
than a particular commercial or open-source simulator.

| Choice | Alternative workflow | Reason and practical effect |
| --- | --- | --- |
| A text flowsheet is the process definition | GUI project files or constructing every process through Python | Inputs are readable, diffable, and scriptable, with the CLI, library, and renderer sharing one parser |
| Sequential modular solving with explicit recycle blocks | One simultaneous equation system for the whole process | Units have clear solve contracts; only recycle-dependent work repeats, while rigorous equipment can still solve coupled internal equations |
| One shared property resolver with provenance | Independent unit-specific lookups or a single opaque database | Explicit input authority, applicability checks, quality penalties, and source reporting are handled consistently |
| Canonical property kernels and cached compiled backends | Provider switching or interpreter-only evaluation in tight loops | Expensive resolution/preparation is separated from repeated numerical evaluation; sources and fit diagnostics remain inspectable |
| Immutable named thermodynamic scopes | One package for the whole process or mutable global settings | Different sections can use appropriate models; interaction data stay scoped and boundary enthalpy corrections remain visible |
| Explicit phase inventories and equilibrium policy | Assuming one homogeneous liquid everywhere | Local/global stability choices and phase routing are visible; unsupported combinations fail instead of silently reducing the declared model |
| Batch/cycle models embedded on a continuous-equivalent basis | A fully dynamic flowsheet or ignoring cyclic equipment | Scheduling and cycle performance can be studied within a steady-state material/energy framework, with explicit averaging assumptions |
| Reports and rendering independent of a modern GUI | Results accessible only through an editor | Runs can be automated and audited, and diagrams can be produced without performing property resolution or simulation |

## Limitations

- **Development maturity:** model/data coverage and numerical robustness are
  uneven. Convergence is not experimental validation; low-quality properties,
  extrapolation, poor specifications, or metastable solutions can still matter.
- **Steady-state scope:** no general dynamic plant simulation, control-system
  simulation, or resolved adsorption breakthrough/PSA cycle. Internal batch
  histories are averaged at their flowsheet boundaries.
- **Fluid phase coverage:** the current LLE/VLLE engine requires compatible
  activity models and represents at most two liquid phases and one vapor.
  `VL(L)E` can retain locally stable metastable liquid; select `VLLE` when a
  global liquid-stability search is required.
- **Henry and steam constraints:** Henry treatment currently uses a pure-water
  VLE standard state; non-water-solvent and Henry-aware LLE/VLLE extensions are
  absent. STEAM/IF97 is restricted to water-only systems.
- **Solid equilibrium:** pure-solid crystallization/washing and restricted
  layer-sweating models do not provide general SLE/SLLE/SVLLE, arbitrary solid
  solutions, co-crystals, polymorph selection, or electrolyte precipitation.
  Permanent-solid inventory is not a dissolution/precipitation calculation.
- **Solids routing and transport:** many units reject solid-bearing feeds;
  there is no general slurry pipe/pressure-machine model. Particle capture,
  growth, washing, and porosity use their documented closures rather than a
  general population-balance or particle-transport framework.
- **Column scope:** shortcut, McCabe–Thiele, and CMO columns are VLE-only.
  Rigorous VLLE columns co-route internal liquids and reject side draws.
  Reactive distillation is not implemented.
- **Reaction scope:** equilibrium reactor support is homogeneous; general
  multiphase/ionic reactive equilibrium is outside the present contract.
  Kinetics and catalyst models require application-specific data and valid
  reaction/rate units.
- **Thermodynamic model/data scope:** predictive methods need compatible
  subgroup and interaction coverage. Unavailable planned methods are rejected,
  not silently substituted. Virial vapor corrections have density/pressure
  applicability limits and are not liquid equations of state.
- **Interfaces and diagrams:** rendered diagrams
  are schematics, and large layouts can need manual drafting. Rendering does
  not produce a calculated stream table or a P&ID.

## Future work

The following are documented extension directions, **not available features
or release-date commitments**:

- Broader thermodynamic families: Wong–Sandler GE–EOS (`PRWS`/`RKSWS`),
  PC-SAFT, CPA, Lee–Kesler–Plöcker, IAPWS-95, and dedicated amine/acid-gas
  treatment. The parser already explains several of these unsupported names.
- Henry-aware LLE/VLLE and dilute standard states for non-water solvents.
- General soluble-solid equilibrium and wider solid-phase/model coverage,
  beyond explicit inventories and equipment-local pure-solid calculations.
- Reactive distillation using the shared reaction catalog, and wider supported
  phase/routing combinations in rigorous equipment.
- Mixture liquid excess-volume/density models and broader property/interaction
  coverage with experimental validation.
- Continued numerical robustness, performance, property diagnostics, worked
  examples, and documentation improvements.

These directions appear in the format/model design documents. General dynamic
simulation would require additional development and should not be inferred
from the web laboratory or internal batch models.

## Citation

When PFDSim contributes to published work, cite the software and record the
**package version and exact Git commit**, your thermodynamic methods, property
sources/overrides, and the flowsheet used. Cite the underlying model/data
publications separately where they materially support the calculation; the
[PSRK](docs/psrk.md), [Lyngby](docs/lyngby.md), and property guides identify
relevant sources.

A software citation for this version is:

> PFDSim. *PFDSim: a chemical-process simulator for text-based flowsheets*.
> Version 0.3.1, 2026. Software.

```bibtex
@software{pfdsim,
  author  = {PFDSim},
  title   = {{PFDSim}: A Chemical-Process Simulator for Text-Based Flowsheets},
  version = {0.3.1},
  year    = {2026},
  note    = {Software; specify the Git commit and source URL used}
}
```

Update the version, year, commit, and source URL to match the copy used in your
work. The repository does not currently provide a project citation DOI or a
`CITATION.cff`; the entry above is a software attribution, not a journal-paper
citation.

## License

PFDSim is licensed under **GNU Affero General Public License v3.0 only
(`AGPL-3.0-only`)**. See [LICENSE](LICENSE) for the complete terms.
Bundled third-party data and source transcriptions may also carry attribution
or licensing notices alongside their source files.
