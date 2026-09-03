# pfdsim

`pfdsim` is a command-line chemical process simulator. Its primary workflow is
to run human-editable `.pfd` flowsheet files with the `pfdsim` CLI, solve them
with sequential modular recycle handling, and write summaries or full `.pfr`
reports.

The project also includes a Flask editor/API, but that is secondary to the CLI
and simulation library.

## What It Can Model

- Steady-state process flowsheets with feeds, products, recycle loops, tear
  streams, and degrees-of-freedom validation.
- Material and energy balances for common process equipment.
- Vapor-liquid, liquid-liquid, and vapor-liquid-liquid equilibrium where the
  selected thermodynamic model supports it.
- Cubic-EOS, activity-coefficient, gamma-phi, vapor-dimerization, Henry-law,
  and steam-table property methods.
- Reaction systems ranging from conversion reactors to kinetic CSTR/PFR models.
- Property lookup and fallback from PFD overrides, local data, Perry/textbook
  sources, resolver caches, and optional online sources.

## Installation

Use Python 3.12 or newer.

```bash
cd pfdsim
uv sync
```

`uv sync` creates the project-local virtual environment, installs pfdsim in
editable mode, and installs the locked runtime and development dependencies.
Use `uv sync --no-dev` when only the runtime package is needed. The dependency
declarations in `pyproject.toml` and the resolved `uv.lock` are authoritative.

The optional web application dependencies are installed explicitly:

```bash
uv sync --extra web
```

## CLI Usage

Run a flowsheet and print a compact result summary:

```bash
uv run pfdsim examples/simple_flash.pfd
```

Write a `.pfr` result file next to the input:

```bash
uv run pfdsim examples/simple_flash.pfd --output
```

Choose an explicit output path:

```bash
pfdsim examples/simple_flash.pfd --output outputs/simple_flash_result.pfr
```

Print the complete `.pfr` report to stdout:

```bash
pfdsim examples/simple_flash.pfd --pfr
```

Print recycle/solver progress to stderr:

```bash
pfdsim examples/ethanol_pressure_swing_recycle_wasteful.pfd --verbose
```

## Python Usage

```python
from simulator import Simulator

sim = Simulator.from_file("examples/simple_flash.pfd")
result = sim.run()

print(result.converged)
print(result.mass_balance_error)
print(result.streams["Vapor"].composition)

sim.write_results("outputs/simple_flash_result.pfr")
```

For persistent web, application, or worker processes, initialization can be
made explicit before serving timed requests:

```python
sim = Simulator.from_file("examples/simple_flash.pfd")
sim.initialize()  # Idempotent: providers, databases, model kernels, solver graph
result = sim.run()  # Reuses initialization; run() initializes automatically otherwise
```

`initialize()` does not calculate feed states, flashes, recycle guesses, or
unit results. Repeated `run()` calls start with clean per-solve state while
retaining prepared thermodynamic and property backends.

Most internal modules still use source-root imports, so direct scripting is
simplest from the repository root. The installed `pfdsim` command handles that
path setup for CLI use.

## Minimal `.pfd` Example

```text
PROCESS: Simple Flash Separation
VERSION: 1.0
ONLINE_LOOKUP: false

COMPONENTS:
    H2O    | Water   | MW=18.02
    C2H5OH | Ethanol | MW=46.07

STREAM Feed : FEED -> HEAT-1.in
    T = 25 [C]
    P = 1 [bar]
    F = 100 [kmol/h]
    x = H2O:0.6, C2H5OH:0.4

STREAM S1 : HEAT-1.out -> FLASH-1.in
STREAM Vapor : FLASH-1.vap -> PRODUCT
STREAM Liquid : FLASH-1.liq -> PRODUCT

UNIT HEAT-1
    TYPE: Heater
    PORTS:
        in  : inlet
        out : outlet
    PARAMS:
        T_out = 80 [C]

UNIT FLASH-1
    TYPE: Flash
    PORTS:
        in  : inlet
        vap : vapor_outlet
        liq : liquid_outlet
    PARAMS:
        T = 80 [C]
        P = 1 [bar]
```

See [docs/pfd_format_v1.0.md](docs/pfd_format_v1.0.md) for the full input
format and [docs/pfr_format_v1.0.md](docs/pfr_format_v1.0.md) for result files.

## Thermodynamic Methods

Supported method names are defined in `thermodynamics_models/factory.py`.

| Family | Methods |
| --- | --- |
| Ideal and steam | `IDEAL`, `STEAM` / `IF97` |
| Cubic and predictive EOS | `RK`, `SRK`, `PR`, `PSRK`, `RKS-BM`, `PR-BM`, `SRK-MC`, `PR-MC`, `SRK-TWU`, `PR-TWU`, `PRSV1`, `PRSV2` |
| Activity coefficient | `NRTL`, `UNIQUAC`, `UNIFAC`, `UNIFDMD`, `UNIFNIST` |
| Gamma-phi | `NRTL-RK`, `NRTL-PR`, `UNIQUAC-RK`, `UNIQUAC-PR`, `UNIFAC-RK`, `UNIFAC-PR`, `UNIFDMD-RK`, `UNIFDMD-PR`, `UNIFNIST-RK`, `UNIFNIST-PR` |
| Vapor dimerization | `NRTL-VDM`, `UNIQUAC-VDM`, `UNIFAC-VDM`, `UNIFDMD-VDM`, `UNIFNIST-VDM` |

Activity-model methods are required for LLE/VLLE units such as decanters,
extractors, and three-phase flashes. `STEAM` is intended for water-only
flowsheets and uses CoolProp/IF97-style steam properties.

The optional top-level `FLUID_PHASE_MODEL` declaration selects constrained
`VLE` (default), locally spinodal-aware `VL(L)E`, or robust global `VLLE`
behavior for unconstrained stream flashes. The latter two require an
LLE-capable activity model. Stream states retain vapor, two liquid slots, and
an explicit permanent-solid inventory. Components declared with
`phase_behavior=permanent_solid` are excluded from fluid equilibrium and may
pass through the supported basic stream and thermal operations. See the PFD
format specification for exact semantics and unit limitations.

Pure-component solid properties are available independently of solid stream
routing. Form-aware solid Cp kernels provide scalar Cp plus analytic sensible
enthalpy/entropy increments; solid molar volume and density resolve from PFD
overrides, a bundled CRC solid-volume database, qualified PubChem observations,
or the restricted organic fallback. Permanent-solid streams consume these APIs
without claiming equilibrium solidification, melting, or dissolution.

`PSRK` uses the published 2005 parameter matrix for predictive phi-phi VLE,
including permanent gases, and supplies EOS departure enthalpy, entropy, and
heat capacity. See [docs/psrk.md](docs/psrk.md) for its data policy and current
limitations.

## Unit Operations

| Area | Unit types |
| --- | --- |
| Mixing and splitting | `Mixer`, `Splitter` |
| Pressure and flow | `Pump`, `Compressor`, `Expander`, `Valve`, `Pipe` |
| Heat transfer | `Heater`, `Cooler`, `HeatExchanger` |
| Flash and phase split | `Flash`, `Flash3`, `ThreePhaseFlash`, `FlashLLE`, `Decanter` |
| Distillation | `ShortcutDistillation`, `McCabeThieleDistillation`, `CMODistillation`, `RigorousDistillation` |
| Absorption and stripping | `Absorber`, `Stripper`, `RigorousAbsorber`, `RigorousStripper` |
| Extraction and drying | `ShortcutExtractor`, `Extractor`, `MolecularSieveDryer` |
| Crystallization | `Crystallizer` |
| Reaction | `Reactor`, `EquilibriumReactor`, `CSTR`, `PFR` |

The parser also recognizes some forward-looking unit names that are not fully
implemented in the simulator yet. Prefer the examples and tests as the current
source of truth for working units and parameters.

## Property Resolution

Component data can be supplied directly in a `.pfd` file:

```text
COMPONENTS:
    MIBK | Methyl isobutyl ketone | CAS=108-10-1, SMILES=CC(C)CC(=O)C, UNIFAC=2CH3+1CH+1CH2+1CH3CO
```

PFD-supplied values are treated as authoritative overrides. Missing properties
are resolved through the local chemical database, identity aliases, Perry and
textbook data, tabulated vapor-pressure data, correlation fallbacks, and
optional online lookup. Set `ONLINE_LOOKUP: false` in a `.pfd` when an example
or regression test should remain deterministic and self-contained.

Solid overrides use `Cp_solid`, `rho_solid`, `Vm_solid`, optional
`solid_material_form`/`solid_polymorph`, and `Cps` or `rhos` entries in
`PROPERTY_CORRELATIONS`. Canonical artifacts are
`data/solid_heat_capacity.sqlite` and `data/solid_volume.sqlite`; see
`scripts/heat_capacity/solid/README.md` for source and estimator audits.

For the detailed source order and quality model, see
[docs/property_resolution_design.md](docs/property_resolution_design.md).

## Examples

The `examples/` directory includes working flowsheets for:

- Basic flash calculations and UNIFAC flashes.
- Rigorous ethanol and benzene/toluene distillation.
- Ethanol dehydration, pressure-swing recycle, and azeotropic distillation.
- Liquid-liquid extraction and rigorous extraction.
- Absorbers, strippers, and partial-condensation recovery.
- Methane liquefaction and cryogenic air separation with cubic EOS methods.
- Steam Rankine cycle calculations.
- Kinetic CSTR and PFR reactor examples.
- Molecular-sieve drying examples.

Run any example with:

```bash
pfdsim examples/simple_rankine_cycle_steam.pfd
```

## Secondary Web App and API

The Flask app is still available for editing, validation, example loading, and
phase chart endpoints, but it is not the main interface:

```bash
python app.py
```

Then open `http://localhost:5000`.

Important endpoints include `/api/parse`, `/api/serialize`, `/api/validate`,
`/api/examples`, `/api/unit-templates`, `/api/chemicals`, and
`/api/chemicals/search`. Treat the frontend as older than the CLI and
simulation core.

## Project Layout

```text
pfdsim/
├── cli.py                         # Installed `pfdsim` command
├── simulator.py                   # High-level simulation interface
├── flowsheet_solver.py            # Sequential modular solver and recycles
├── pfd_parser.py                  # .pfd parser/serializer and validation
├── thermodynamics.py              # Compatibility facade
├── thermodynamics_models/         # Current thermodynamic model implementations
├── property_resolution/           # Resolver modules for physical properties
├── unit_operations_*.py           # Unit operation implementations
├── chemical_properties.py         # Component database and hydration layer
├── data/                          # Local interaction/property data
├── examples/                      # Runnable .pfd/.pfr examples
├── docs/                          # Format and design notes
├── tests/                         # Regression/unit tests
├── app.py                         # Flask app/API
├── templates/ and static/         # Web editor assets
└── scripts/                       # Data extraction and model probes
```

## Development

Run the default test suite from the repository root:

```bash
python -m pytest
```

The project-local pytest hook runs the default tests in parallel when pytest is
invoked without arguments. For a targeted test during development, pass the test
file or test name explicitly:

```bash
python -m pytest tests/test_steam_thermodynamics.py
```

The all-examples performance regression check is opt-in:

```bash
python -m unittest tests.performance_all_examples -q
```

## License

This project is licensed under the terms in [LICENSE](LICENSE).
