# PFR File Format Specification

**Version:** 1.0  
**Date:** 2024-12-11  
**Status:** Stable

## Overview

The PFR (Process Flow Results) format is a human-readable text format for storing simulation results from PFD files. It contains the original process definition plus calculated stream states and unit performance data.

## File Extension

- `.pfr` - Process Flow Results file

## Relationship to PFD

A PFR file is a superset of a PFD file. It contains:
1. The complete original PFD content
2. Calculated stream states
3. Unit operation results
4. Simulation metadata

## Format Version

```
PFR_VERSION: 1.0
```

## File Structure

```
================================================================================
                           PROCESS FLOW RESULTS
================================================================================

[Simulation Metadata]
[Original PFD Content]
[Stream Results]
[Unit Results]
[Material Balance Summary]
[Energy Balance Summary]
```

## Section Specifications

### 1. Header

```
================================================================================
                           PROCESS FLOW RESULTS
================================================================================
File: <original_filename>
Generated: <timestamp>
Simulator: PFD Editor v1.0
================================================================================
```

### 2. Simulation Summary

```
SIMULATION SUMMARY
--------------------------------------------------------------------------------
Status: <Converged/Not Converged>
Thermo Method: <IDEAL/RK/SRK/PR/PSRK/RKS-BM/PR-BM/UNIFAC/UNIFDMD/UNIFNIST/NRTL/UNIQUAC/...>
Fluid Phase Model: <VLE/VL(L)E/VLLE>
Max Iterations: <number>
Tolerance: <value>
Warnings: <count>
```

### 3. Original PFD

The complete original .pfd file content is included, marked with:

```
ORIGINAL PFD
================================================================================
<full PFD content>
================================================================================
```

### 4. Stream Results

Stream endpoints use the parser's canonical unit-port names. Compact aliases
and clockwise numeric ports therefore appear as names such as `vapor_out`,
`feed2`, or `side_draw1`; explicitly declared custom `PORTS:` names remain
authoritative and are preserved.

```
STREAM RESULTS
================================================================================

STREAM_RESULT <stream_name>:
  source = <source>
  destination = <destination>
  Temperature: <T> °C (<T> K)
  Pressure: <P> bar
  Molar Flow: <F> kmol/h
  Mass Flow: <M> kg/h
  Vapor Fraction: <VF>
  Fluid Vapor Fraction: <VF_fluid>  # solid-bearing streams with active fluid only
  Liquid-1 Fraction: <L1>
  Liquid-2 Fraction: <L2>
  Solid Fraction: <S>
  Phase Status: <status>
  Phase Stability: <stability method/status>
  Enthalpy: <H> kJ/kmol
  Entropy: <S> kJ/kmol-K
  Molar Volume: <V> m³/kmol
  Density: <rho> kg/m³
  MW (avg): <MW> g/mol
  
  Composition (mole fraction):
    <comp1>: <x1>
    <comp2>: <x2>
    ...
  
  Composition (mass fraction):
    <comp1>: <w1>
    <comp2>: <w2>
    ...

  SOLID_COMPONENT_FLOWS:
    <solid_comp>: F=<flow> [kmol/h]

  SOLID_PARTICLE_PROPERTIES:
    <solid_comp>: diameter=<diameter> [m], sphericity=<sphericity>

  SOLID_PARTICLE_SIZE_DISTRIBUTIONS:
    <solid_comp>: basis=component_molar_flow, D32=<Sauter mean diameter> [m]
        diameter=<representative diameter> [m], F=<class flow> [kmol/h], fraction=<component molar fraction>

  Distinct phase compositions are also reported when active. `x` remains the
  pooled liquid compatibility composition, while `LIQUID1_COMPOSITION` and
  `LIQUID2_COMPOSITION` retain an LLE/VLLE split. Solid-component flow,
  particle-property, and PSD sections are reported only when present. All
  primary phase fractions use total-stream mole basis; `fluid_vapor_fraction` separately
  reports vapor divided by vapor plus both liquid phases. Inactive liquid-2,
  solid, and particle fields are omitted rather than emitted as empty/null
  sections.
```

### 5. Unit Results

Unit types use their canonical registered names. For example, input aliases
such as `ThreePhaseFlash`, `Dryer`, `KineticsBatch`, `PlugFlowReactor`, and
`PBR` report as `Flash3`, `MolecularSieveDryer`, `BatchReactor`, `PFR`, and
`PackedBedReactor`.

Adsorption units can additionally report `unrepresented_enthalpy_change`
in kW: the net enthalpy change in adsorbed inventories minus the enthalpy
already represented by the adsorbate accounting stream. The overall energy
audit includes this term on the outlet side. When adsorption enthalpies are
identifiable, `heat_duty` includes the physical isothermal bed duty; otherwise
the unit reports an explicit warning and null `bed_heat_duty_kJ_h` performance
value. See [molecular sieve enthalpy](molecular_sieves.md).

```
UNIT RESULTS
================================================================================

Unit: <unit_id>
  Type: <unit_type>
  
  Inlet Streams:
    <stream1>: <flow> kmol/h at <T>°C
    ...
  
  Outlet Streams:
    <stream2>: <flow> kmol/h at <T>°C
    ...
  
  Performance:
    <param1>: <value1> <unit1>
    <param2>: <value2> <unit2>
    ...
  
  Heat Duty: <Q> kW
  Work: <W> kW
```

Unit-specific performance data:

**Crystallizer (MSMPR mode):**
- Calculated saturation ratio, relative supersaturation, and log saturation
  ratio, saturation temperature, dimensional/reduced/fusion-scaled
  undercooling, melting point, and heat of fusion for the crystallizing
  component
- Residence time, vessel volume, and operating volumetric flow
- Suspension density, nucleation rate, birth-size growth rate, growth-rate
  range, and seed/nucleated particle rates, plus the number-mean diameter
- Selected kinetic models and normalized expressions
- Population-balance quadrature class count, coupled material residual, and
  configured absolute, relative, and effective residual tolerances
- The extensive outlet PSD in the corresponding stream result

**Flash:**
- Vapor fraction
- K-values for each component
- Heat duty

**Distillation:**
- Number of stages
- Minimum stages (Fenske)
- Reflux ratio
- Minimum reflux (Underwood)
- Top/bottom temperatures
- Condenser/reboiler duties

**Reactor:**
- Component conversion, per-reaction extent, and canonical rate basis
- Isothermal/adiabatic/duty/jacket heat-duty and energy-closure diagnostics
- Thermal CSTR root temperatures, residual slopes, stability classifications,
  and the explicitly selected ignition/extinction branch
- Adaptive PFR temperature, pressure, composition, reaction-rate, velocity,
  Reynolds-number, and residence-time profiles when applicable
- Packed-bed catalyst-mass, effective/intrinsic rate, effectiveness-factor,
  reaction-specific rigorous pellet factors, support positions, bed-void
  residence-time, and Ergun hydraulic profiles when applicable
- PFR solver method, evaluation count, termination status, and material/energy
  residuals
- Batch fleet schedule, batch frequency, utilization, discharge coverage,
  heat per batch, average duty, transient time profile, and per-batch closure
- Semi-batch feed windows and time-averaged continuous outlet basis
- Homogeneous phase-stability diagnostics

**Heat Exchanger:**
- Heat duty
- LMTD
- UA value
- Temperature approach

### 6. Material Balance

```
MATERIAL BALANCE
================================================================================

Overall Mass Balance:
  Total In:  <mass_in> kg/h
  Total Out: <mass_out> kg/h
  Error: <error>%

Component Balances (kmol/h):
  Component     In        Out       Generated   Consumed   Error
  ---------     --        ---       ---------   --------   -----
  <comp1>       <in>      <out>     <gen>       <cons>     <err>%
  ...
```

### 7. Energy Balance

```
ENERGY BALANCE
================================================================================

Stream Enthalpies:
  Inlet Streams:  <H_in> kJ/h
  Outlet Streams: <H_out> kJ/h

Heat Duties:
  <unit1>: <Q1> kW
  <unit2>: <Q2> kW
  ...

Work:
  <unit3>: <W3> kW
  ...

Total:
  Heat In:  <Q_in> kW
  Heat Out: <Q_out> kW
  Work In:  <W_in> kW
  Work Out: <W_out> kW
  Error: <error>%
```

### 8. Warnings

```
WARNINGS
================================================================================
- <warning1>
- <warning2>
...
```

## Example PFR File

```
================================================================================
                           PROCESS FLOW RESULTS
================================================================================
File: simple_flash.pfd
Generated: 2024-12-11 10:30:45
Simulator: PFD Editor v1.0
================================================================================

SIMULATION SUMMARY
--------------------------------------------------------------------------------
Status: Converged
Thermo Method: UNIFAC
Max Iterations: 50
Tolerance: 1e-06
Warnings: 0

ORIGINAL PFD
================================================================================
PROCESS: Simple Flash
VERSION: 1.0
THERMO_METHOD: UNIFAC
...
================================================================================

STREAM RESULTS
================================================================================

Stream: Feed
  From: FEED
  To: FLASH-1.in
  Temperature: 90.00 °C (363.15 K)
  Pressure: 1.50 bar
  Molar Flow: 100.00 kmol/h
  Vapor Fraction: 0.000
  
  Composition (mole fraction):
    C2H5OH: 0.3000
    H2O: 0.7000

Stream: Vapor
  From: FLASH-1.vap
  To: PRODUCT
  Temperature: 85.00 °C (358.15 K)
  Pressure: 1.00 bar
  Molar Flow: 35.50 kmol/h
  Vapor Fraction: 1.000
  
  Composition (mole fraction):
    C2H5OH: 0.5430
    H2O: 0.4570

Stream: Liquid
  From: FLASH-1.liq
  To: PRODUCT
  Temperature: 85.00 °C (358.15 K)
  Pressure: 1.00 bar
  Molar Flow: 64.50 kmol/h
  Vapor Fraction: 0.000
  
  Composition (mole fraction):
    C2H5OH: 0.1660
    H2O: 0.8340

UNIT RESULTS
================================================================================

Unit: FLASH-1
  Type: Flash
  
  Inlet Streams:
    Feed: 100.00 kmol/h at 90.0°C
  
  Outlet Streams:
    Vapor: 35.50 kmol/h at 85.0°C
    Liquid: 64.50 kmol/h at 85.0°C
  
  Performance:
    Vapor Fraction: 0.355
    K_C2H5OH: 3.267
    K_H2O: 0.548
  
  Heat Duty: -125.3 kW

MATERIAL BALANCE
================================================================================

Overall Mass Balance:
  Total In:  2840.7 kg/h
  Total Out: 2840.7 kg/h
  Error: 0.00%

ENERGY BALANCE
================================================================================

Stream Enthalpies:
  Inlet:  2840.7 MJ/h
  Outlet: 2390.4 MJ/h

Heat Duties:
  FLASH-1: -125.3 kW

Error: 0.00%

================================================================================
                              END OF RESULTS
================================================================================
```

## Parsing Guidelines

When parsing PFR files:

1. Look for section headers (lines of `=` or `-`)
2. Parse metadata from header section
3. Extract original PFD between `ORIGINAL PFD` markers
4. Parse stream data from `Stream: <name>` blocks
5. Parse unit data from `Unit: <name>` blocks
6. Extract numerical values with units

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-12-11 | Initial stable release |
