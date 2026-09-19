# PFDSim

PFDSim is a Python chemical-process simulator for human-editable `.pfd`
flowsheets. It validates and solves steady-state processes, including recycle
loops, and produces stream and equipment results, material and energy
balances, and human-readable `.pfr` reports.

Use it through the command line or Python API. A standalone renderer turns
flowsheets into SVG, PDF, or PNG diagrams without running a simulation.

## Main features

- **Flowsheet simulation:** sequential modular solving, automatic or explicit
  tear streams, accelerated recycle convergence, and degrees-of-freedom
  validation.
- **Phase equilibrium:** VLE, LLE, and VLLE with compatible thermodynamic
  methods; explicit phase routing through flashes, decanters, and extractors.
- **Process equipment:** pressure and flow handling, heat transfer, rigorous
  distillation and extraction, absorption, stripping, and molecular-sieve drying.
- **Reaction systems:** conversion and equilibrium reactors, kinetic CSTRs,
  batch/semi-batch reactors, PFRs, and packed beds with catalyst effectiveness
  and pressure-drop models.
- **Solids processing:** pure-solid crystallization, kinetic crystal growth,
  layer crystallization and sweating, particle-size distributions, cake
  filtration, and equilibrium warm/melt washing.
- **Physical properties:** explicit overrides, bundled data, property
  correlations, optional online resolution, and source/quality reporting.
- **Text-based inputs and reports:** reusable reaction declarations,
  thermodynamic scopes, executable examples, and engineering-style diagrams.

The flowsheet solver is steady state. Equipment that models a batch or cycle
reports continuous-equivalent flows and duties; PFDSim is not a general dynamic
flowsheet simulator.

## Installation

Use Python 3.12 or later. From a source checkout:

```bash
uv sync
```

This installs the package in editable mode with the locked runtime and
development dependencies. Use `uv sync --no-dev` for a runtime-only environment.

Optional extras:

```bash
uv sync --extra render       # PDF/PNG export; also requires system Cairo
uv sync --extra web          # legacy Flask application/API
uv sync --extra dipole-xtb   # optional molecular geometry/dipole calculations
uv sync --extra dipole-pvdz  # optional higher-level dipole calculations
```

All diagram formats require the Graphviz `dot` executable on `PATH`.
Dependency declarations and optional extras are defined in
[`pyproject.toml`](pyproject.toml).

## Run a flowsheet

```bash
# Print a result summary.
uv run pfdsim examples/simple_flash.pfd

# Write a .pfr report beside the input.
uv run pfdsim examples/simple_flash.pfd --output

# Choose a report path.
uv run pfdsim examples/simple_flash.pfd -o outputs/simple_flash_result.pfr

# Print the complete report.
uv run pfdsim examples/simple_flash.pfd --pfr

# Show solver progress on standard error.
uv run pfdsim examples/ethanol_pressure_swing_recycle_wasteful.pfd --verbose
```

The CLI exits nonzero when the simulation does not converge or an input or
simulation error occurs.

## Python API

```python
from pfdsim.simulator import Simulator

simulator = Simulator.from_file("examples/simple_flash.pfd")
result = simulator.run()

print(result.converged)
print(result.mass_balance_error)
print(result.streams["Vapor"].composition)

simulator.write_results("outputs/simple_flash_result.pfr")
```

`Simulator.from_string()` accepts PFD text directly. `initialize()` prepares
property packages and the solver graph without solving the flowsheet; `run()`
calls it automatically. Initialization is idempotent for an unchanged
configuration. Convenience functions `simulate_pfd()` and `simulate_file()`
are also available in `pfdsim.simulator`.

## PFD input format

A PFD declares components, stream connections and feed conditions, equipment,
reactions, and thermodynamic settings. Both compact and expanded unit syntax
are supported. This is a complete small flowsheet:

```pfd
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

UNIT HEAT-1 : Heater
    T_out = 80 [C]

UNIT FLASH-1 : Flash
    T = 80 [C]
    P = 1 [bar]
```

Explicit property values take precedence over resolved data. Set
`ONLINE_LOOKUP: false` to use local data and supplied values without online
property queries. Named thermodynamic scopes allow different parts of a
flowsheet to use different property packages.

See the [PFD format specification](docs/pfd_format_v1.0.md) for syntax,
parameters, units, and validation rules, and the
[PFR format specification](docs/pfr_format_v1.0.md) for reports.

## Thermodynamics

| Family | Methods |
| --- | --- |
| Ideal and steam | `IDEAL`; water-only `STEAM` / `IF97` |
| Cubic and predictive EOS | `RK`, `SRK`, `PR`, `PSRK`, `RKS-BM`, `PR-BM`, `SRK-MC`, `PR-MC`, `SRK-TWU`, `PR-TWU`, `PRSV1`, `PRSV2` |
| Activity coefficient | `NRTL`, `UNIQUAC`, `UNIFAC`, `UNIFAC2`, `UNIFDMD`, `UNIFM2`, `UNIFNIST` |
| Vapor corrections | Supported activity models with cubic-EOS, second-virial, Hayden-O'Connell, or vapor-dimerization corrections |

`FLUID_PHASE_MODEL` selects conventional `VLE`, locally spinodal-aware
`VL(L)E`, or global liquid-stability/VLLE handling. LLE/VLLE requires a
compatible activity model; cubic EOS and PSRK are not routed through the
current LLE/VLLE engine. Supported aqueous VLE calculations can use a Henry-law
standard state for dilute noncondensables.

Property resolution records sources and quality diagnostics. The available
models include thermal, transport, phase-change, and solid properties.
See [property resolution](docs/property_resolution_design.md) and
[PSRK model/data policy](docs/psrk.md) for details.

## Unit operations

| Area | Canonical unit names |
| --- | --- |
| Flow and pressure | `Mixer`, `Splitter`, `Pump`, `Compressor`, `Expander`, `Valve`, `Pipe` |
| Heat transfer | `Heater`, `Cooler`, `HeatExchanger` |
| Flash and phase separation | `Flash`, `Flash3`, `Decanter`, `MolecularSieveDryer` |
| Distillation | `ShortcutDistillation`, `McCabeThieleDistillation`, `CMODistillation`, `RigorousDistillation` |
| Extraction and gas–liquid contact | `ShortcutExtractor`, `RigorousExtractor`, `Absorber`, `RigorousAbsorber`, `Stripper`, `RigorousStripper` |
| Reactions | `Reactor`, `EquilibriumReactor`, `CSTR`, `BatchReactor`, `PFR`, `PackedBedReactor` |
| Solids | `Crystallizer`, `LayerCrystallizer`, `Filter` |

Rigorous distillation supports VLE and VLLE equilibrium stages, pressure
profiles, multiple feeds, and optional azeotropic initialization. Its VLE
configuration supports side draws and a top decanter; VLLE stages currently
co-route both liquids and reject side draws and separately routed decanter
condensers.

Crystallization and equilibrium cake washing use pure-solid equilibrium.
Layer sweating has a restricted finite-rate solid-solution model. These do
not provide general SLE/SLLE/SVLLE, polymorph selection, or a general
solid-solution equilibrium solver. Permanent-solid inventories are supported
only by equipment with an explicit solids contract.

Model-specific capabilities and limits are documented in the
[PFD specification](docs/pfd_format_v1.0.md),
[filtration guide](docs/filtration.md), and
[equilibrium washing guide](docs/equilibrium_washing.md).

## Render a flowsheet

```bash
uv run python -m pfdsim.render examples/simple_flash.pfd
uv run python -m pfdsim.render examples/simple_flash.pfd -o flash.pdf
uv run python -m pfdsim.render examples/simple_flash.pfd -o flash.png --stream-table
```

Rendering parses the source without initializing thermodynamics or solving
it. Existing outputs require `--force` to replace; the input PFD is protected.
Diagrams are process schematics, not P&IDs. See the
[rendering guide](docs/rendering.md).

## Examples

The [`examples/`](examples/) directory contains executable flowsheets, including:

| Example | Demonstrates |
| --- | --- |
| [Simple flash](examples/simple_flash.pfd) | Basic input syntax and heated VLE flash |
| [Rigorous azeotropic distillation](examples/ethanol_benzene_azeotropic_distillation_rigorous.pfd) | Multistage separation with VLLE |
| [Pressure-swing recycle](examples/ethanol_pressure_swing_recycle_wasteful.pfd) | Recycle convergence |
| [Acrylic-acid extraction](examples/acrylic_acid_rigorous_extraction.pfd) | Rigorous LLE extraction |
| [Cryogenic air separation](examples/cryogenic_air_separation_rks_bm.pfd) | Cubic-EOS thermodynamics |
| [Rankine cycle](examples/simple_rankine_cycle_steam.pfd) | Steam properties |
| [Lactic-acid dehydration](examples/lactic_acid_dehydration_pbr.pfd) | Packed-bed kinetics and downstream recovery |
| [Batch synthesis](examples/ethyl_acetate_batch_synthesis.pfd) | Batch reaction scheduling |
| [Cake filtration](examples/cake_filtration_washing.pfd) | Solids handling and washing |
| [Warm/melt washing](examples/equilibrium_warm_melt_washing.pfd) | Coupled energy and solid-equilibrium calculations |

Examples are worked inputs; their property estimates and engineering
assumptions still need validation for a new application.

## Development

Run the default tests from the repository root:

```bash
python -m pytest
```

The project hook dispatches a no-argument run in parallel. For focused work:

```bash
python -m pytest tests/test_render.py
python -m ruff check .
```

The all-examples performance check is opt-in:

```bash
python -m unittest tests.performance_all_examples -q
```

Simulation code lives in `simulator.py`, `flowsheet_solver.py`, and
`unit_operations_*.py`; thermodynamic models and property resolution live in
`thermodynamics_models/` and `property_resolution/`. See `docs/` for reference
material, `tests/` for regressions, and `scripts/` for maintained builders and
benchmarks.

## Optional web API

The legacy Flask application provides an editor and API for parsing,
validation, simulation, examples, and property/phase queries:

```bash
uv run python app.py
```

Install the `web` extra first. The CLI and simulation library are the primary
interfaces; the frontend is older than the simulation core.

## License

PFDSim is licensed under [AGPL-3.0-only](LICENSE).
