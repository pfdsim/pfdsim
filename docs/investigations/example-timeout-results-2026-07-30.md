# Isolated example run with 60-second timeouts

- Examples: 32
- Concurrent child processes: 6
- Timeout: 60 seconds per process
- Total wall time: 87.770 seconds
- Passed: 27
- Failed before timeout: 3
- Killed at timeout: 2

| Example | Status | Wall time |
|---|---:|---:|
| `3methylpyridine_ether_extraction_recycle.pfd` | timeout | 60.116s |
| `acrylic_acid_rigorous_extraction.pfd` | passed | 11.425s |
| `air_3a_molecular_sieve_drying.pfd` | passed | 4.447s |
| `ammonia_oxidation.pfd` | passed | 9.045s |
| `ammonia_synthesis.pfd` | failed | 5.910s |
| `benzene_toluene_20_stage_distillation_nrtl.pfd` | passed | 10.803s |
| `biosteam_mesh_hydrocarbon_distillation.pfd` | passed | 27.310s |
| `butanol_water_lle.pfd` | passed | 5.507s |
| `cryogenic_air_separation_rks_bm.pfd` | passed | 6.289s |
| `cstr_pfr_comparison.pfd` | passed | 9.881s |
| `dcm_3a_molecular_sieve_drying.pfd` | passed | 5.580s |
| `ethanol_3a_molecular_sieve_drying.pfd` | passed | 12.059s |
| `ethanol_benzene_azeotropic_distillation_rigorous.pfd` | passed | 12.387s |
| `ethanol_distillation_rigorous.pfd` | passed | 9.249s |
| `ethanol_ether_partial_condensation_absorption.pfd` | passed | 13.069s |
| `ethanol_pressure_swing_recycle_wasteful.pfd` | passed | 18.027s |
| `ethanol_water_inclined_pipe_unifac.pfd` | passed | 9.019s |
| `ethylene_oxide.pfd` | timeout | 60.029s |
| `ethylene_oxide_simple.pfd` | failed | 15.161s |
| `methane_claude_liquefaction_pr.pfd` | passed | 5.538s |
| `methanol_decomposition_pfr.pfd` | passed | 23.292s |
| `methanol_diethyl_ether_5bar_nrtl_rk.pfd` | passed | 9.574s |
| `methanol_synthesis.pfd` | failed | 23.268s |
| `mixed_acid_dehydration_uniquac_vdm.pfd` | passed | 9.280s |
| `pyridine_ether_extraction.pfd` | passed | 9.534s |
| `rk_thermodynamics_pfr.pfd` | passed | 5.066s |
| `saponification_cstr.pfd` | passed | 9.053s |
| `simple_flash.pfd` | passed | 7.891s |
| `simple_rankine_cycle_steam.pfd` | passed | 3.376s |
| `trace_organic_water_stripping_isothermal_unifnist.pfd` | passed | 11.711s |
| `unifac_flash.pfd` | passed | 8.437s |
| `vinegar_concentration_uniquac_vdm.pfd` | passed | 6.013s |

## Timeout investigations

### `3methylpyridine_ether_extraction_recycle.pfd`

The initial six-process run repeatedly entered `COL-100` during Broyden
recycle evaluations. Its rigorous-distillation MESH solve repeatedly stopped at
residual `5.37e-01` because line search could not reduce the residual; the run
had reached recycle evaluation 6 at timeout.

The timeout was not reproducible with only the two timed-out examples active:
this example passed in 13.209 seconds. It is therefore primarily a
concurrency/cold-start performance timeout, amplified by repeated expensive
MESH attempts, rather than a deterministic hang.

### `ethylene_oxide.pfd`

This timeout reproduced with reduced concurrency. A 55-second traceback and
coefficient timing showed repeated `CO2` coefficient construction attempts
during recycle tear-state PH reconstruction. Each attempt takes roughly
0.5-1.7 seconds and fails, so no coefficient is cached and the next Psat call
starts canonical construction again.

The hot stack is:

`recycle tear PH reconstruction -> fallback flash -> K_values -> Psat ->
get_Psat_coefficients(CO2) -> canonical Psat assembly/fitting`.

The underlying CO2 construction error is that the canonical fitter rejects its
boiling-point/critical-point domain (`boiling point must lie below Tc`). The
timeout is repeated failure work, not slow recycle convergence.

## Non-timeout failures

- `ammonia_synthesis.pfd`: reports convergence but produces `NaN` mass and
  energy balances after a flash creates NaN vapor composition.
- `ethylene_oxide_simple.pfd`: fails coefficient construction for CO2 in
  `DIST-1`.
- `methanol_synthesis.pfd`: the flash produces NaN vapor composition and the
  subsequent recycle eventually fails mixer enthalpy inversion at 50 bar.

## Legacy pointwise comparison

The two timeouts and three ordinary failures were rerun concurrently in five
isolated processes with the exact preserved pointwise vapor-pressure resolver
and the pre-coefficient `thermo.Psat()` behavior. All five passed within the
same 60-second limit:

| Example | Legacy status | Wall time | Mass balance error | Energy balance error |
|---|---:|---:|---:|---:|
| `3methylpyridine_ether_extraction_recycle.pfd` | passed | 21.670s | 1.18408e-05 | 3.47231e-06 |
| `ammonia_synthesis.pfd` | passed | 8.839s | 1.46808e-05 | 2.52573e-16 |
| `ethylene_oxide.pfd` | passed | 13.337s | 1.58684e-06 | 1.98689e-07 |
| `ethylene_oxide_simple.pfd` | passed | 8.986s | 8.24655e-07 | 2.87187e-16 |
| `methanol_synthesis.pfd` | passed | 9.037s | 2.40854e-05 | 4.16889e-07 |

This confirms that all five failures are regressions in the canonical
coefficient path rather than pre-existing example defects. In particular, the
legacy route avoids the repeated CO2 coefficient-construction failure and the
unchecked infinities/NaNs that destabilize the synthesis flashes.
