# Rigorous Extraction Notes

## Current Pyridine/Ether Example

File: `examples/pyridine_ether_extraction.pfd`

- Thermo: `UNIFNIST`
- Unit: `RigorousExtractor`
- Mode: `adiabatic`
- Stages: 20
- Aqueous feed: 1000 kg/h, 10 wt% pyridine / 90 wt% water, 30 C, 1 bar
- Solvent feed: 500 kg/h diethyl ether, 20 C, 1 bar

Representative run after adding the internal enthalpy/Cp-only probe path:

```text
converged: true
solver: inverse_broyden
iterations: 66
component balance error: 9.519e-06
overall energy relative error: 1.050e-05

pyridine recovery: 98.290432%
extract mass flow: about 597.7 kg/h
raffinate mass flow: about 902.3 kg/h
```

Earlier detailed run before the density-skip optimization was numerically similar:

```text
pyridine recovery: 98.31%
extract T: 30.47 C
raffinate T: 24.50 C
extract composition: pyridine 0.14117, water 0.12356, diethyl ether 0.73527
raffinate composition: pyridine 0.00044, water 0.99402, diethyl ether 0.00554
```

Final product streams still calculate density; only internal temporary enthalpy/Cp
probe states skip density.

## Sanity Checks

Three comparison cases were run from the PFD text:

```text
Case                         Recovery    Extract T    Raffinate T   Iterations
adiabatic 26C / 24C           89.77%       26.70 C      23.08 C       55
adiabatic 25C / 25C           86.95%       25.72 C      22.75 C       49
isothermal 25C                84.96%       25.00 C      25.00 C       43
```

Balances looked sensible:

```text
adiabatic 26/24: mass error 1.20e-5, energy error 6.59e-6
adiabatic 25/25: mass error 3.93e-6, energy error 5.48e-6
isothermal 25:   mass error 2.90e-5, energy error 0.0
```

The adiabatic equal-inlet case does not keep both outlets at 25 C because
nonideal liquid mixture enthalpy/heat of solution redistributes energy between
the two liquid phases.

## Timing

One warmed Python process was used for timing: prewarm both modes once, then run
each mode three times.

Before skipping density for internal probes:

```text
isothermal avg: 2.603824 s
adiabatic avg: 25.681315 s
```

After skipping density for internal probes:

```text
isothermal avg: 2.548961 s
adiabatic avg: 19.655374 s
```

The density skip improved the adiabatic 20-stage case by about 23%.

## Profiling Takeaways

The compiled LLE backend is active. Multicomponent `liquid_liquid_equilibrium`
uses the Numba-backed `compiled_lle.split(...)` path for this example.

The current LLE split equates liquid activities, not Gibbs-minimization directly:

```text
x_i^phase1 * gamma_i^phase1 = x_i^phase2 * gamma_i^phase2
```

The remaining hot spots after the density skip are mostly real work:

```text
compiled_lle.split
excess_enthalpy
activity_coefficients
Cp_liquid / mixture_Cp
stage temperature solve evaluations
```

Full post-density-skip `cProfile` top lists from the warmed process:

### Isothermal Profile

```text
PROFILE_SUMMARY isothermal iters 43 converged True
         1596949 function calls (1596127 primitive calls) in 4.182 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    4.196    4.196 simulator.py:161(run)
        1    0.000    0.000    4.125    4.125 flowsheet_solver.py:1419(solve)
        1    0.000    0.000    4.118    4.118 flowsheet_solver.py:583(_calculate_unit)
        1    0.018    0.018    4.118    4.118 unit_operations_separation.py:738(solve)
      122    0.015    0.000    4.065    0.033 unit_operations_separation.py:1211(residual_for_vector)
      122    0.029    0.000    4.003    0.033 unit_operations_separation.py:1131(run_sweep)
     2440    0.012    0.000    3.198    0.001 unit_operations_separation.py:1055(split_stage)
     2440    0.031    0.000    2.011    0.001 unit_operations_separation.py:897(split_stage_at_T)
     9516    0.018    0.000    1.909    0.000 unit_operations_separation.py:878(enthalpy_flow)
     9516    0.032    0.000    1.890    0.000 unit_operations_separation.py:872(liquid_state)
     2442    0.017    0.000    1.836    0.001 thermodynamics.py:3380(liquid_liquid_equilibrium)
     2442    1.790    0.001    1.819    0.001 compiled_lle.py:99(split)
     9600    0.116    0.000    1.793    0.000 thermodynamics.py:2562(calculate_state)
     9600    0.061    0.000    1.514    0.000 thermodynamics.py:2147(mixture_enthalpy)
     9600    0.193    0.000    1.210    0.000 thermodynamics.py:2033(excess_enthalpy)
     2440    0.008    0.000    1.175    0.000 unit_operations_separation.py:928(outlet_enthalpy)
    19202    0.324    0.000    0.866    0.000 thermodynamics.py:3390(activity_coefficients)
     2440    0.007    0.000    0.749    0.000 unit_operations_separation.py:884(stage_input_enthalpy)
    86656    0.123    0.000    0.361    0.000 {built-in method builtins.sum}
    14349    0.224    0.000    0.322    0.000 compiled_unifac.py:115(activity_coefficients)
     9600    0.022    0.000    0.236    0.000 thermodynamics.py:1553(mixture_enthalpy)
   451975    0.227    0.000    0.227    0.000 {method 'get' of 'dict' objects}
    38397    0.032    0.000    0.186    0.000 thermodynamics.py:1575(<genexpr>)
    28797    0.125    0.000    0.154    0.000 thermodynamics.py:1533(enthalpy_liquid)
    14398    0.065    0.000    0.146    0.000 unit_operations_separation.py:844(normalize_positive)
```

### Adiabatic Profile

```text
PROFILE_SUMMARY adiabatic iters 66 converged True
         16355141 function calls (16354319 primitive calls) in 34.105 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000   34.227   34.227 simulator.py:161(run)
        1    0.000    0.000   34.161   34.161 flowsheet_solver.py:1419(solve)
        1    0.000    0.000   34.157   34.157 flowsheet_solver.py:583(_calculate_unit)
        1    0.028    0.028   34.157   34.157 unit_operations_separation.py:738(solve)
      193    0.028    0.000   34.088    0.177 unit_operations_separation.py:1211(residual_for_vector)
      193    0.074    0.000   33.959    0.176 unit_operations_separation.py:1131(run_sweep)
     3860    0.057    0.000   31.793    0.008 unit_operations_separation.py:1055(split_stage)
     3860    0.242    0.000   31.736    0.008 unit_operations_separation.py:956(solve_adiabatic_stage)
    19462    0.127    0.000   27.623    0.001 unit_operations_separation.py:967(evaluate)
    80408    0.225    0.000   20.054    0.000 unit_operations_separation.py:872(liquid_state)
    80492    0.954    0.000   19.127    0.000 thermodynamics.py:2562(calculate_state)
    46258    0.070    0.000   16.400    0.000 unit_operations_separation.py:878(enthalpy_flow)
    19462    0.057    0.000   14.425    0.001 unit_operations_separation.py:928(outlet_enthalpy)
    80492    0.474    0.000   13.000    0.000 thermodynamics.py:2147(mixture_enthalpy)
    19462    0.375    0.000   12.885    0.001 unit_operations_separation.py:897(split_stage_at_T)
    19464    0.092    0.000   11.479    0.001 thermodynamics.py:3380(liquid_liquid_equilibrium)
    19464   11.142    0.001   11.380    0.001 compiled_lle.py:99(split)
    80492    1.585    0.000    8.855    0.000 thermodynamics.py:2033(excess_enthalpy)
   160986    2.139    0.000    6.042    0.000 thermodynamics.py:3390(activity_coefficients)
   678563    1.119    0.000    4.787    0.000 {built-in method builtins.sum}
    80492    0.111    0.000    4.652    0.000 thermodynamics.py:1126(phase_weighted_mixture_Cp)
    80492    0.326    0.000    4.540    0.000 thermodynamics.py:1104(mixture_Cp)
   241473    0.642    0.000    4.189    0.000 thermodynamics.py:1013(Cp_liquid)
    17075    0.061    0.000    3.790    0.000 unit_operations_separation.py:940(outlet_heat_capacity_flow)
    80492    0.231    0.000    3.614    0.000 thermodynamics.py:1553(mixture_enthalpy)
```

The main future performance step-change would be an equation-oriented extractor,
similar in spirit to rigorous distillation, but solving direct LLE stage
equations instead of repeatedly calling nested LLE splits:

```text
variables:
  R_j, E_j, xR_j, xE_j, T_j

residuals:
  component balances
  ln(xR_i gammaR_i) - ln(xE_i gammaE_i)
  phase composition normalization
  adiabatic energy balance, or fixed-T/isothermal duty
```

Wrapping sparse Newton around the current black-box LLE split would probably not
help much; replacing the nested split with direct activity-equality residuals is
the likely faster architecture.

## Equation-Oriented Extractor Pass

Implemented a direct MESH-style rigorous LLE extractor path. It is now the
default rigorous extractor solver. It can also be selected explicitly with:

```text
solver_algorithm = equation_oriented
```

The legacy nested LLE split/stage-sweep path is deprecated but remains available
with:

```text
solver_algorithm = split_sweep
```

The equation-oriented variables are per-stage raffinate/extract flows, both
phase compositions, and stage temperature for adiabatic mode. The residuals are
component balances, all component activity equalities,
and stage energy balances for adiabatic mode. Isothermal mode omits the energy
equation from the Newton solve and still reports the implied heat duty.

Available MESH initializers:

```text
initializer = auto          # split_sweep for small N, coarse_grid for larger N
initializer = split_sweep   # shortcut estimate plus a few cheap LLE sweeps
initializer = coarse_grid   # solve fewer stages, interpolate to full N
initializer = homotopy      # adiabatic starts from an isothermal equation solve
```

`mesh_initializer` is accepted as an alias for `initializer`.
`auto_coarse_min_stages` controls the auto threshold; the default is 8 stages.

Speed mechanisms added:

```text
sparse damped Newton line search
structural sparsity pattern
colored finite-difference columns
semi-analytic derivatives for flow variables
log-flow variables for positivity
softmax composition variables for normalized positive phase compositions
logit temperature variables for bounded adiabatic stage temperatures
mixture_enthalpy residual calls instead of full calculate_state probes
```

Warmed in-process benchmark on the 20-stage pyridine/water + diethyl ether
example:

```text
case                         times (s)                         avg (s)
isothermal split_sweep        2.563933  2.274799  2.249418      2.362716
isothermal equation_oriented  0.432947  0.419434  0.388739      0.413707
adiabatic split_sweep        19.355973 20.852945 19.420786     19.876568
adiabatic equation_oriented   1.270049  1.260513  1.443808      1.324790
```

Representative outlet comparison:

```text
case                         recovery    extract T    raffinate T   mesh residual
isothermal split_sweep        84.9572%     25.000 C     25.000 C      1.18e-6
isothermal equation_oriented  84.9515%     25.000 C     25.000 C      7.54e-8
adiabatic split_sweep         98.2906%     30.472 C     24.502 C      7.02e-6
adiabatic equation_oriented   98.2968%     30.472 C     24.502 C      2.49e-7
```

Adiabatic equation-oriented final compositions:

```text
extract:    pyridine 0.1411575840, water 0.1235386075, diethyl ether 0.7353038085
raffinate:  pyridine 0.0004379738, water 0.9940189533, diethyl ether 0.0055430728
```

The updated `examples/pyridine_ether_extraction.pfd` relies on the new default
equation-oriented solver.
