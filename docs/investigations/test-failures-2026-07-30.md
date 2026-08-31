# pfdsim full-suite failures

- Date: 2026-07-30
- Command: `python -m pytest -n 6 --dist=load --durations=0 --junitxml=/tmp/pfdsim-test-results.xml -q`
- Parallelism: 6 pytest-xdist workers on 6 available CPUs
- Wall-clock runtime: 337.88s (5m 37.88s)
- Results: 114 failed outcomes, 864 passed tests, 855 passed subtests, 54 warnings
- Distinct failing test methods: 73

The time column is pytest's recorded duration for the complete test method. When a
method contains multiple failing subtests, pytest records one method duration rather
than a separate duration for each subtest; the failed-outcomes column shows how many
failures occurred inside that method.

| Failing test | Failed outcomes | Time |
|---|---:|---:|
| `tests/test_compiled_backends.py::CompiledBackendTests::test_compiled_unifac_vlle_matches_reference_binary_invariant` | 1 | 0.454s |
| `tests/test_compiled_backends.py::CompiledBackendTests::test_vlle_ps_skips_compiled_tp_for_binary_invariant_case` | 1 | 0.039s |
| `tests/test_compiled_backends.py::CompiledBackendTests::test_vlle_tv_skips_compiled_tp_for_binary_invariant_case` | 1 | 0.105s |
| `tests/test_examples.py::ExampleSimulationTests::test_3methylpyridine_ether_extraction_recycle_matches_expected_products` | 1 | 8.751s |
| `tests/test_examples.py::ExampleSimulationTests::test_all_examples_converge_and_close_balances` | 9 | 139.834s |
| `tests/test_examples.py::ExampleSimulationTests::test_dcm_3a_molecular_sieve_example_meets_ppm_limit` | 1 | 0.271s |
| `tests/test_examples.py::ExampleSimulationTests::test_ethanol_ether_partial_condensation_absorption_matches_expected_results` | 1 | 2.049s |
| `tests/test_examples.py::ExampleSimulationTests::test_ethanol_rigorous_distillation_is_physically_directional` | 1 | 0.058s |
| `tests/test_examples.py::ExampleSimulationTests::test_ethylene_oxide_distillation_temperatures_are_not_cryogenic` | 2 | 11.714s |
| `tests/test_examples.py::ExampleSimulationTests::test_recycle_solver_methods_converge_methanol_synthesis` | 3 | 5.695s |
| `tests/test_examples.py::ExampleSimulationTests::test_trace_organic_isothermal_stripper_matches_expected_results` | 1 | 0.680s |
| `tests/test_pfd_component_properties.py::PFDComponentPropertyTests::test_pfd_component_can_resolve_as_one_chemical_but_behave_like_another` | 1 | 4.858s |
| `tests/test_pfd_integration.py::PFDIntegrationTests::test_pressure_units_are_converted_for_feed_streams_and_unit_params` | 1 | 3.525s |
| `tests/test_pipe_unit.py::PipeUnitTests::test_commercial_steel_glycerol_45c_probe` | 1 | 7.997s |
| `tests/test_pipe_unit.py::PipeUnitTests::test_pipe_reports_where_liquid_exhausts_available_pressure` | 1 | 8.571s |
| `tests/test_quality_report_context.py::ContextualQualityReportTests::test_generated_methylpentane_report_hides_auxiliary_quality_noise` | 1 | 0.914s |
| `tests/test_quality_report_context.py::ContextualQualityReportTests::test_manual_dataset_properties_are_retained_high_quality_when_used` | 1 | 0.162s |
| `tests/test_surface_tension_resolution.py::SurfaceTensionResolutionTests::test_knotts_parachor_is_fallback_after_fitted_correlations` | 1 | 3.066s |
| `tests/test_thermodynamic_methods.py::ThermodynamicMethodTests::test_binary_invariant_vlle_spec_flashes_select_phase_amounts` | 2 | 1.770s |
| `tests/test_thermodynamic_methods.py::ThermodynamicMethodTests::test_bubble_scan_warns_when_pressure_exceeds_pure_critical_pressure` | 1 | 3.238s |
| `tests/test_thermodynamic_methods.py::ThermodynamicMethodTests::test_flash3_ph_recovers_efficient_expansion_across_seeded_vle_branch` | 1 | 0.004s |
| `tests/test_thermodynamic_methods.py::ThermodynamicMethodTests::test_flash3_tpd_reseeds_missed_water_rich_vle_branch` | 5 | 0.544s |
| `tests/test_thermodynamic_methods.py::ThermodynamicMethodTests::test_prsv_methods_improve_methanol_pure_eos_psat_at_80c` | 1 | 0.024s |
| `tests/test_thermodynamic_methods.py::ThermodynamicMethodTests::test_srk_mc_improves_methanol_pure_eos_psat_at_80c` | 1 | 0.020s |
| `tests/test_thermodynamic_methods.py::ThermodynamicMethodTests::test_twu_methods_methanol_pure_eos_psat_at_80c` | 1 | 0.023s |
| `tests/test_thermodynamic_methods.py::ThermodynamicMethodTests::test_vlle_bubble_dew_and_spec_flashes_recover_ternary_reference_state` | 1 | 0.047s |
| `tests/test_thermodynamic_methods.py::ThermodynamicMethodTests::test_vlle_bubble_point_detects_binary_heteroazeotrope` | 1 | 0.041s |
| `tests/test_thermodynamic_methods.py::ThermodynamicMethodTests::test_vlle_ph_prefers_three_phase_enthalpy_root_with_offset_guess` | 1 | 3.513s |
| `tests/test_thermodynamic_methods.py::ThermodynamicMethodTests::test_vlle_ps_prefers_three_phase_entropy_root_with_bad_guesses` | 1 | 9.962s |
| `tests/test_thermodynamic_methods.py::ThermodynamicMethodTests::test_vlle_tv_prefers_narrow_three_phase_pressure_branch` | 1 | 5.689s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_binary_rigorous_distillation_uses_robust_square_solver` | 1 | 6.279s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_cmo_rigorous_initializer_defaults_to_relaxed_tolerance_and_respects_override` | 1 | 6.495s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_coarse_rigorous_uses_legacy_cmo_estimate_seed` | 1 | 9.274s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_distillation_pressure_drop_profiles` | 1 | 29.491s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_every_registered_unit_class_has_a_smoke_case` | 8 | 32.004s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_expander_keeps_superheated_ethylene_vapor` | 1 | 5.823s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_explicit_rigorous_initializers_fail_hard` | 3 | 9.995s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_flash3_reports_binary_invariant_phase_amount_warning` | 1 | 3.023s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_heat_exchanger_auto_u_is_explicit_preliminary_estimate` | 1 | 4.635s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_heater_duty_uses_seeded_newton_before_bracketing` | 1 | 3.993s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_mccabe_thiele_uses_latent_heat_curved_operating_lines` | 1 | 11.170s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_mccabe_thiele_uses_model_k_values_at_weighted_saturation_temperature` | 1 | 5.114s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_methanol_water_hvap_variants_improve_mccabe_and_cmo_compositions` | 1 | 8.780s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_nitrile_column_azeotropic_initializer_converges_within_eight_iterations` | 1 | 3.609s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_absorber_adiabatic_mode_closes_energy_balance` | 1 | 4.703s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_absorber_and_stripper_report_initializer_paths` | 1 | 3.980s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_absorber_concentrated_uniquac_acetaldehyde_case` | 1 | 4.856s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_absorber_hot_wet_feed_uses_selective_henry_components` | 1 | 7.766s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_absorber_isothermal_mode_reports_heat_duty` | 1 | 4.118s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_absorber_temperature_spec_implies_isothermal_mode` | 1 | 4.179s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_cmo_initializers_preserve_condenser_and_pressure_profile` | 2 | 10.558s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_distillation_accepts_mass_feed_and_distillate_specs` | 2 | 7.793s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_distillation_accepts_multiple_feed_stages` | 1 | 7.016s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_distillation_ethanol_benzene_azeotropic_case` | 1 | 7.165s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_distillation_fails_fast_for_noncondensables` | 1 | 20.956s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_distillation_initializer_routing` | 7 | 13.041s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_distillation_local_jacobian_failure_retries_original` | 1 | 7.023s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_distillation_local_jacobian_matches_all_condenser_and_spec_types` | 6 | 12.649s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_distillation_local_jacobian_matches_thermo_families` | 2 | 18.616s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_distillation_methanol_trace_contaminants` | 2 | 14.850s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_distillation_mixed_condenser_has_two_distillate_phases` | 1 | 6.754s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_distillation_partial_condenser_and_side_draws` | 1 | 7.807s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_distillation_semi_analytic_jacobian_handles_partial_mass_spec` | 1 | 6.670s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_distillation_solves_mesh_balances` | 1 | 6.996s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_distillation_solves_multicomponent_mesh_balances` | 1 | 10.645s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_distillation_sparse_newton_solves_feed_stage_curve` | 1 | 7.478s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_distillation_top_decanter_selects_reflux_phase_and_purge` | 1 | 6.763s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_stripper_concentrated_uniquac_acetaldehyde_case` | 1 | 4.607s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_rigorous_stripper_trace_unifnist_case_reports_stripping` | 1 | 8.453s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_shortcut_distillation_uses_partial_condenser_for_noncondensables` | 1 | 6.348s |
| `tests/test_unit_operations.py::UnitOperationSmokeTests::test_trace_impurities_keep_effectively_binary_balance_rows` | 1 | 23.532s |
| `tests/test_vapor_pressure_adapter.py::VaporPressureCanonicalizationAdapterTests::test_curated_critical_qualities_override_dataset_default` | 1 | 0.006s |
| `tests/test_vapor_pressure_adapter.py::VaporPressureCanonicalizationAdapterTests::test_curated_reduced_psat_records_preserve_internal_criticals` | 2 | 0.222s |
