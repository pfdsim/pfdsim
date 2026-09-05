import math
import unittest

from chemical_properties import ChemicalDatabase
from dof_analyzer import SpecificationStatus, analyze_dof
from msmpr_models import (
    MSMPRConvergenceError,
    MSMPRDefinitionError,
    msmpr_rate_law_from_mapping,
    solve_steady_msmpr,
)
from pfd_parser import parse_pfd, validate_pfd
from simulator import Simulator
from thermodynamics_models.base import IdealThermodynamics
from thermodynamics_models.common import R
from thermodynamics_models.factory import create_thermodynamics
from thermodynamics_models.sle import (
    liquid_solution_activities,
    pure_solid_log_saturation_activity,
)
from unit_operations_base import UnitOperationError
from unit_operations_solids import Crystallizer


class MSMPRModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thermo = IdealThermodynamics(
            ['water', 'ethanol'],
            ChemicalDatabase(enable_online=False),
        )
        cls.thermo.configure_permanent_solids(
            ['water', 'ethanol'],
            [],
            {
                'water': {
                    'particle_size_distribution': {
                        'diameters_um': [50.0],
                        'fractions': [1.0],
                        'basis': 'mass',
                    },
                },
            },
            conventional_solid_components=['water'],
        )

    def test_constant_rates_recover_ideal_msmpr_third_moment(self):
        growth_rate = 5.0e-5
        nucleation_rate = 1.0e9
        residence_time = 2.0
        growth = msmpr_rate_law_from_mapping({
            'expression': 'kg',
            'param_kg': growth_rate,
            'rate_unit': 'm/h',
        }, 'growth')
        nucleation = msmpr_rate_law_from_mapping({
            'expression': 'b0',
            'param_b0': nucleation_rate,
            'rate_unit': '1/m3/h',
        }, 'nucleation')
        result = solve_steady_msmpr(
            self.thermo,
            component='water',
            total_component_flows_kmol_h={'water': 9.0, 'ethanol': 1.0},
            T=250.0,
            P=1.0,
            residence_time_h=residence_time,
            growth_law=growth,
            nucleation_law=nucleation,
            quadrature_classes=4,
        )
        molar_volume = self.thermo._solid_molar_volume('water', 250.0)
        volume = result.volume_m3
        expected_solid_flow = (
            nucleation_rate
            * volume
            * math.pi
            * (growth_rate * residence_time) ** 3
            / molar_volume
        )
        self.assertAlmostEqual(
            result.solid_flow_kmol_h,
            expected_solid_flow,
            places=11,
        )
        distribution = result.particle_size_distribution
        number_mean = sum(
            diameter * fraction
            for diameter, fraction in zip(
                distribution.diameters_m,
                distribution.number_fractions,
            )
        )
        self.assertAlmostEqual(
            number_mean, growth_rate * residence_time, places=14
        )
        self.assertAlmostEqual(
            result.number_mean_diameter_m,
            growth_rate * residence_time,
            places=14,
        )
        self.assertLess(abs(result.material_residual_kmol_h), 1.0e-10)

    def test_power_law_presets_and_unit_conversion(self):
        growth = msmpr_rate_law_from_mapping({
            'model': 'power_law',
            'kg': 2.0,
            'g': 2.0,
            'rate_unit': 'um/s',
        }, 'growth')
        nucleation = msmpr_rate_law_from_mapping({
            'model': 'secondary_power_law',
            'kb': 3.0,
            'b': 2.0,
            'j': 1.0,
            'i': 2.0,
            'rate_unit': '1/m3/s',
        }, 'nucleation')
        context = {
            'S': 1.5,
            'sigma': 0.5,
            'MT': 4.0,
            'G0': 2.0,
        }
        self.assertAlmostEqual(growth.evaluate(context), 1.8e-3)
        self.assertAlmostEqual(nucleation.evaluate(context), 43200.0)

    def test_custom_nucleation_can_use_birth_growth_rate(self):
        nucleation = msmpr_rate_law_from_mapping({
            'expression': 'kb * MT**j * G0**i',
            'param_kb': 2.0,
            'param_j': 1.0,
            'param_i': 2.0,
            'rate_unit': '1/m3/h',
        }, 'nucleation')
        self.assertAlmostEqual(nucleation.evaluate({
            'S': 1.1,
            'MT': 3.0,
            'G0': 4.0,
        }), 96.0)
        growth = msmpr_rate_law_from_mapping({
            'expression': '1e-5',
            'rate_unit': 'm/h',
        }, 'growth')
        coupled = msmpr_rate_law_from_mapping({
            'expression': 'kb * G0',
            'param_kb': 1.0e12,
            'rate_unit': '1/m3/h',
        }, 'nucleation')
        result = solve_steady_msmpr(
            self.thermo,
            component='water',
            total_component_flows_kmol_h={'water': 9.0, 'ethanol': 1.0},
            T=250.0,
            P=1.0,
            residence_time_h=1.0,
            growth_law=growth,
            nucleation_law=coupled,
        )
        self.assertAlmostEqual(result.birth_growth_rate_m_h, 1.0e-5)
        self.assertAlmostEqual(
            result.nucleation_rate_per_m3_h,
            1.0e12 * result.birth_growth_rate_m_h,
        )

    def test_zero_nucleation_clear_feed_returns_no_particle_population(self):
        growth = msmpr_rate_law_from_mapping({
            'expression': '1e-5',
            'rate_unit': 'm/h',
        }, 'growth')
        nucleation = msmpr_rate_law_from_mapping({
            'expression': '0',
            'rate_unit': '1/m3/h',
        }, 'nucleation')
        result = solve_steady_msmpr(
            self.thermo,
            component='water',
            total_component_flows_kmol_h={'water': 9.0, 'ethanol': 1.0},
            T=250.0,
            P=1.0,
            residence_time_h=1.0,
            growth_law=growth,
            nucleation_law=nucleation,
        )
        self.assertEqual(result.solid_flow_kmol_h, 0.0)
        self.assertIsNone(result.particle_size_distribution)

    def test_positive_nucleation_requires_growth_or_birth_size(self):
        growth = msmpr_rate_law_from_mapping({
            'expression': '0',
            'rate_unit': 'm/h',
        }, 'growth')
        nucleation = msmpr_rate_law_from_mapping({
            'expression': '1e9',
            'rate_unit': '1/m3/h',
        }, 'nucleation')
        with self.assertRaisesRegex(ValueError, 'positive nucleus_diameter'):
            solve_steady_msmpr(
                self.thermo,
                component='water',
                total_component_flows_kmol_h={'water': 9.0, 'ethanol': 1.0},
                T=250.0,
                P=1.0,
                residence_time_h=1.0,
                growth_law=growth,
                nucleation_law=nucleation,
            )

    def test_nonconvergence_uses_msmpr_exception(self):
        growth = msmpr_rate_law_from_mapping({
            'model': 'power_law',
            'kg': 1.0e-4,
            'g': 1.0,
            'rate_unit': 'm/h',
        }, 'growth')
        nucleation = msmpr_rate_law_from_mapping({
            'model': 'primary_power_law',
            'kb': 1.0e12,
            'b': 2.0,
            'rate_unit': '1/m3/h',
        }, 'nucleation')
        with self.assertRaisesRegex(
            MSMPRConvergenceError,
            'did not converge',
        ):
            solve_steady_msmpr(
                self.thermo,
                component='water',
                total_component_flows_kmol_h={
                    'water': 9.0,
                    'ethanol': 1.0,
                },
                T=250.0,
                P=1.0,
                residence_time_h=1.0,
                growth_law=growth,
                nucleation_law=nucleation,
                max_iterations=1,
            )

    def test_configured_material_tolerance_is_absolute(self):
        growth = msmpr_rate_law_from_mapping({
            'expression': '1e-5',
            'rate_unit': 'm/h',
        }, 'growth')
        nucleation = msmpr_rate_law_from_mapping({
            'expression': '1e12',
            'rate_unit': '1/m3/h',
        }, 'nucleation')
        tolerance = 1.0e-12
        try:
            result = solve_steady_msmpr(
                self.thermo,
                component='water',
                total_component_flows_kmol_h={
                    'water': 9.0e6,
                    'ethanol': 1.0e6,
                },
                T=250.0,
                P=1.0,
                residence_time_h=1.0,
                growth_law=growth,
                nucleation_law=nucleation,
                residual_tolerance=tolerance,
            )
        except MSMPRConvergenceError:
            pass
        else:
            self.assertLessEqual(
                abs(result.material_residual_kmol_h),
                tolerance,
            )

    def test_relative_material_tolerance_is_explicit_and_reported(self):
        growth = msmpr_rate_law_from_mapping({
            'expression': '1e-5',
            'rate_unit': 'm/h',
        }, 'growth')
        nucleation = msmpr_rate_law_from_mapping({
            'expression': '1e12',
            'rate_unit': '1/m3/h',
        }, 'nucleation')
        result = solve_steady_msmpr(
            self.thermo,
            component='water',
            total_component_flows_kmol_h={
                'water': 9.0e6,
                'ethanol': 1.0e6,
            },
            T=250.0,
            P=1.0,
            residence_time_h=1.0,
            growth_law=growth,
            nucleation_law=nucleation,
            residual_tolerance=1.0e-12,
            relative_residual_tolerance=1.0e-10,
        )
        self.assertEqual(result.absolute_residual_tolerance_kmol_h, 1.0e-12)
        self.assertEqual(result.relative_residual_tolerance, 1.0e-10)
        self.assertAlmostEqual(
            result.effective_residual_tolerance_kmol_h,
            9.0e-4,
        )
        self.assertLessEqual(
            abs(result.material_residual_kmol_h),
            result.effective_residual_tolerance_kmol_h,
        )
        with self.assertRaisesRegex(
            MSMPRDefinitionError,
            'relative_residual_tolerance',
        ):
            solve_steady_msmpr(
                self.thermo,
                component='water',
                total_component_flows_kmol_h={
                    'water': 9.0,
                    'ethanol': 1.0,
                },
                T=250.0,
                P=1.0,
                residence_time_h=1.0,
                growth_law=growth,
                nucleation_law=nucleation,
                relative_residual_tolerance=-1.0e-10,
            )

    def test_custom_growth_accepts_size_and_thermodynamic_variables(self):
        growth = msmpr_rate_law_from_mapping({
            'model': 'custom',
            'expression': (
                'max(0, kg*sigma*(1 + alpha*L))*exp(-Ea/(R*T))'
            ),
            'param_kg': 0.01,
            'param_alpha': 1000.0,
            'param_Ea': 500.0,
            'rate_unit': 'm/h',
        }, 'growth')
        self.assertTrue(growth.depends_on_size)
        rate = growth.evaluate({
            'sigma': 0.2,
            'L': 1.0e-4,
            'T': 300.0,
            'R': 8.314,
        })
        self.assertGreater(rate, 0.0)
        with self.assertRaisesRegex(MSMPRDefinitionError, 'Unknown'):
            msmpr_rate_law_from_mapping({
                'expression': 'kb * L',
                'param_kb': 1.0,
                'rate_unit': '1/m3/h',
            }, 'nucleation')
        with self.assertRaisesRegex(MSMPRDefinitionError, 'nonnegative'):
            msmpr_rate_law_from_mapping({
                'model': 'power_law',
                'kg': 1.0,
                'g': -1.0,
                'rate_unit': 'm/h',
            }, 'growth')

    def test_growth_can_use_fusion_scaled_undercooling(self):
        coefficient = 1.0e-4
        growth = msmpr_rate_law_from_mapping({
            'expression': 'kg * deltaT_fusion',
            'param_kg': coefficient,
            'rate_unit': 'm/h',
        }, 'growth')
        nucleation = msmpr_rate_law_from_mapping({
            'expression': '1e8',
            'rate_unit': '1/m3/h',
        }, 'nucleation')
        result = solve_steady_msmpr(
            self.thermo,
            component='water',
            total_component_flows_kmol_h={'water': 9.0, 'ethanol': 1.0},
            T=250.0,
            P=1.0,
            residence_time_h=1.0,
            growth_law=growth,
            nucleation_law=nucleation,
        )
        self.assertGreater(result.undercooling_K, 0.0)
        self.assertAlmostEqual(
            result.birth_growth_rate_m_h,
            coefficient * result.fusion_scaled_undercooling,
        )
        with self.assertRaisesRegex(MSMPRDefinitionError, 'model=custom'):
            msmpr_rate_law_from_mapping({
                'model': 'power_law',
                'expression': '1',
                'kg': 1.0,
                'g': 1.0,
                'rate_unit': 'm/h',
            }, 'growth')

    def test_seeded_secondary_nucleation_uses_mt_and_preserves_seed_source(self):
        feed = self.thermo.calculate_state_with_solid_flows(
            250.0,
            1.0,
            10.0,
            {'water': 0.9, 'ethanol': 0.1},
            {'water': 0.01},
            phase='liquid',
        )
        feed.solid_particle_properties['water'] = {'sphericity': 0.63}
        result = Crystallizer('C', self.thermo, {
            'model': 'MSMPR',
            'T': 250.0,
            'residence_time': 1.0,
            'growth': {
                'expression': 'kg * sigma',
                'param_kg': 5.0e-5,
                'rate_unit': 'm/h',
            },
            'nucleation': {
                'model': 'secondary_power_law',
                'kb': 1.0e9,
                'b': 1.0,
                'j': 1.0,
                'rate_unit': '1/m3/h',
            },
        }).solve({'in': feed})
        performance = result.performance
        self.assertGreater(
            performance['solid_component_flows_kmol_per_h']['water'],
            0.01,
        )
        self.assertGreater(performance['suspension_density_kg_m3'], 0.0)
        self.assertGreater(performance['nucleation_rate_per_m3_h'], 0.0)
        self.assertGreater(performance['seed_particle_rate_per_h'], 0.0)
        self.assertEqual(performance['particle_sphericity'], 0.63)
        self.assertEqual(
            result.outlet_streams['out']
            .solid_particle_properties['water']['sphericity'],
            0.63,
        )
        result.outlet_streams['out'].validate_particle_size_distributions()

    def test_nonideal_msmpr_uses_activity_not_mole_fraction(self):
        thermo = create_thermodynamics(['water', 'ethanol'], 'NRTL')
        thermo.configure_permanent_solids(
            ['water', 'ethanol'],
            [],
            conventional_solid_components=['water'],
        )
        growth = msmpr_rate_law_from_mapping({
            'expression': '1e-5', 'rate_unit': 'm/h',
        }, 'growth')
        nucleation = msmpr_rate_law_from_mapping({
            'expression': '1e8', 'rate_unit': '1/m3/h',
        }, 'nucleation')
        result = solve_steady_msmpr(
            thermo,
            component='water',
            total_component_flows_kmol_h={'water': 9.0, 'ethanol': 1.0},
            T=250.0,
            P=1.0,
            residence_time_h=1.0,
            growth_law=growth,
            nucleation_law=nucleation,
        )
        mole_fraction_ratio = (
            result.liquid_composition['water'] / result.saturation_activity
        )
        self.assertAlmostEqual(
            result.saturation_ratio,
            result.liquid_activity / result.saturation_activity,
        )
        self.assertNotAlmostEqual(
            result.saturation_ratio, mole_fraction_ratio, places=4
        )
        saturation_temperature = result.saturation_temperature_K
        saturation_liquid_activity = liquid_solution_activities(
            thermo,
            saturation_temperature,
            1.0,
            result.liquid_composition,
        )['water']
        saturation_solid_activity = math.exp(
            pure_solid_log_saturation_activity(
                thermo, 'water', saturation_temperature, 1.0
            )
        )
        self.assertAlmostEqual(
            saturation_liquid_activity,
            saturation_solid_activity,
            places=10,
        )

    def test_size_dependent_growth_changes_rate_over_population(self):
        feed = self.thermo.calculate_state(
            280.0,
            1.0,
            10.0,
            {'water': 0.9, 'ethanol': 0.1},
            phase='liquid',
        )
        result = Crystallizer('C', self.thermo, {
            'model': 'MSMPR',
            'T': 250.0,
            'residence_time': 1.0,
            'growth': {
                'expression': 'kg * sigma * (1 + alpha*L)',
                'param_kg': 5.0e-5,
                'param_alpha': 1000.0,
                'rate_unit': 'm/h',
            },
            'nucleation': {
                'model': 'primary_power_law',
                'kb': 1.0e10,
                'b': 1.0,
                'rate_unit': '1/m3/h',
            },
            'quadrature_classes': 8,
        }).solve({'in': feed})
        self.assertGreater(
            result.performance['growth_rate_max_m_h'],
            result.performance['growth_rate_min_m_h'],
        )
        self.assertGreater(result.performance['birth_growth_rate_m_h'], 0.0)
        self.assertEqual(
            len(result.outlet_streams['out']
                .solid_particle_size_distributions['water'].diameters_m),
            8,
        )

    def test_msmpr_rejects_a_different_inlet_solid(self):
        thermo = IdealThermodynamics(
            ['water', 'ethanol'], ChemicalDatabase(enable_online=False)
        )
        thermo.configure_permanent_solids(
            ['water', 'ethanol', 'NaCl'],
            ['NaCl'],
            conventional_solid_components=['water'],
        )
        feed = thermo.calculate_state(
            280.0,
            1.0,
            10.0,
            {'water': 0.8, 'ethanol': 0.1, 'NaCl': 0.1},
        )
        with self.assertRaisesRegex(UnitOperationError, 'unsupported solid'):
            Crystallizer('C', thermo, {
                'model': 'MSMPR',
                'T': 250.0,
                'residence_time': 1.0,
                'growth': {
                    'expression': '1e-5', 'rate_unit': 'm/h',
                },
                'nucleation': {
                    'expression': '1e8', 'rate_unit': '1/m3/h',
                },
            }).solve({'in': feed})


class MSMPRPFDTests(unittest.TestCase):
    PFD = """
PROCESS: MSMPR ice
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
COMPONENTS:
    water | Water | type=conventional_with_solid
    ethanol | Ethanol
STREAM Feed : FEED -> C.in
    T = 6.85 [C]
    P = 1 [bar]
    F = 10 [kmol/h]
    x = water:0.9, ethanol:0.1
STREAM Product : C.out -> PRODUCT
UNIT C : Crystallizer
    model = MSMPR
    T = 250 [K]
    residence_time = 1 [h]
    quadrature_classes = 8
    msmpr_relative_tolerance = 1e-9
    outlet_sphericity = 0.74
    growth_model = power_law
    growth_coefficient = 0.0001
    growth_g = 1
    growth_rate_unit = m/h
    nucleation_model = primary_power_law
    nucleation_coefficient = 1e12
    nucleation_b = 2
    nucleation_rate_unit = 1/m3/h
"""

    def test_complete_msmpr_pfd_validates_runs_and_reports_psd(self):
        pfd = parse_pfd(self.PFD)
        self.assertEqual(validate_pfd(pfd), ([], []))
        reparsed = parse_pfd(pfd.to_pfd())
        self.assertEqual(validate_pfd(reparsed), ([], []))
        self.assertEqual(
            analyze_dof(pfd).unit_results[0].status,
            SpecificationStatus.OK,
        )
        simulator = Simulator.from_string(self.PFD)
        result = simulator.run()
        self.assertTrue(result.converged, result.errors)
        performance = result.units['C'].performance
        self.assertEqual(performance['model'], 'steady_ideal_msmpr')
        self.assertEqual(performance['particle_sphericity'], 0.74)
        self.assertEqual(performance['relative_residual_tolerance'], 1.0e-9)
        self.assertGreaterEqual(
            performance['effective_residual_tolerance_kmol_per_h'],
            performance['absolute_residual_tolerance_kmol_per_h'],
        )
        self.assertGreater(performance['saturation_ratio'], 1.0)
        self.assertGreater(
            result.streams['Product'].solid_component_flows['water'], 0.0
        )
        product = result.streams['Product']
        self.assertEqual(
            product.solid_particle_properties['water']['sphericity'],
            0.74,
        )
        self.assertAlmostEqual(
            performance['volumetric_flow_m3_h'],
            product.F / product.rho,
            places=10,
        )
        self.assertEqual(
            len(result.streams['Product']
                .solid_particle_size_distributions['water'].diameters_m),
            8,
        )
        report = simulator._generate_pfr()
        self.assertIn('model = steady_ideal_msmpr', report)
        self.assertIn('saturation_ratio = ', report)
        self.assertIn('undercooling_K = ', report)
        self.assertIn('SOLID_PARTICLE_SIZE_DISTRIBUTIONS:', report)
        self.assertIn('sphericity=0.74', report)
        self.assertGreater(performance['saturation_temperature_K'], 250.0)
        self.assertAlmostEqual(
            performance['undercooling_K'],
            performance['saturation_temperature_K'] - 250.0,
        )
        self.assertAlmostEqual(
            performance['reduced_undercooling'],
            performance['undercooling_K']
            / performance['saturation_temperature_K'],
        )
        self.assertAlmostEqual(
            performance['fusion_scaled_undercooling'],
            performance['heat_of_fusion_J_mol']
            * performance['undercooling_K']
            / (
                R
                * 250.0
                * performance['melting_temperature_K']
            ),
        )

    def test_volume_spec_derives_residence_time(self):
        source = self.PFD.replace(
            '    residence_time = 1 [h]',
            '    volume = 0.21 [m3]',
        )
        result = Simulator.from_string(source).run()
        self.assertTrue(result.converged, result.errors)
        performance = result.units['C'].performance
        self.assertAlmostEqual(performance['volume_m3'], 0.21)
        self.assertGreater(performance['residence_time_h'], 0.0)

    def test_dimensioned_alias_units_are_case_insensitive(self):
        source = self.PFD.replace(
            '    residence_time = 1 [h]',
            '    v = 210 [L]\n'
            '    l0 = 50 [um]',
        )
        result = Simulator.from_string(source).run()
        self.assertTrue(result.converged, result.errors)
        self.assertAlmostEqual(result.units['C'].performance['volume_m3'], 0.21)
        distribution = result.streams[
            'Product'
        ].solid_particle_size_distributions['water']
        self.assertGreater(min(distribution.diameters_m), 50.0e-6)

    def test_cake_split_keeps_calculated_msmpr_population(self):
        source = self.PFD.replace(
            'STREAM Product : C.out -> PRODUCT',
            'STREAM Cake : C.cake -> PRODUCT\n'
            'STREAM MotherLiquor : C.mother_liquor -> PRODUCT',
        ).replace(
            '    quadrature_classes = 8',
            '    quadrature_classes = 8\n'
            '    mother_liquor_retention = 0.1',
        )
        result = Simulator.from_string(source).run()
        self.assertTrue(result.converged, result.errors)
        cake = result.streams['Cake']
        liquor = result.streams['MotherLiquor']
        self.assertIn('water', cake.solid_particle_size_distributions)
        self.assertEqual(
            cake.solid_particle_properties['water']['sphericity'],
            0.74,
        )
        cake.validate_particle_size_distributions()
        self.assertEqual(liquor.solid_particle_size_distributions, {})
        self.assertEqual(result.units['C'].performance['outlet_mode'], 'cake_split')

    def test_seeded_msmpr_cascade_bounds_output_class_count(self):
        source = self.PFD.replace(
            'STREAM Product : C.out -> PRODUCT',
            'STREAM Interstage : C.out -> C2.in\n'
            'STREAM Product : C2.out -> PRODUCT',
        ).replace(
            '    quadrature_classes = 8',
            '    quadrature_classes = 8\n'
            '    maximum_output_classes = 10',
            1,
        )
        second_unit = source[source.index('UNIT C : Crystallizer'):].replace(
            'UNIT C : Crystallizer', 'UNIT C2 : Crystallizer', 1
        )
        result = Simulator.from_string(source + '\n' + second_unit).run()
        self.assertTrue(result.converged, result.errors)
        interstage = result.streams[
            'Interstage'
        ].solid_particle_size_distributions['water']
        product = result.streams[
            'Product'
        ].solid_particle_size_distributions['water']
        self.assertEqual(len(interstage.diameters_m), 8)
        self.assertLessEqual(len(product.diameters_m), 10)
        result.streams['Product'].validate_particle_size_distributions()

    def test_msmpr_validation_requires_geometry_and_both_rate_laws(self):
        pfd = parse_pfd("""
UNIT C : Crystallizer
    model = MSMPR
    T = 250 [K]
STREAM Feed : -> C.in
STREAM Product : C.out
""")
        errors, _warnings = validate_pfd(pfd)
        self.assertTrue(any('exactly one' in error for error in errors), errors)
        self.assertTrue(any('growth kinetics' in error for error in errors), errors)
        self.assertTrue(any('nucleation kinetics' in error for error in errors), errors)
        result = analyze_dof(pfd).unit_results[0]
        self.assertEqual(result.status, SpecificationStatus.UNDER_SPECIFIED)

    def test_kinetic_parameters_require_explicit_msmpr_model(self):
        pfd = parse_pfd("""
UNIT C : Crystallizer
    T = 250 [K]
    residence_time = 1 [h]
STREAM Feed : -> C.in
STREAM Product : C.out
""")
        errors, _warnings = validate_pfd(pfd)
        self.assertTrue(any('require model=MSMPR' in error for error in errors))

    def test_equilibrium_remains_the_default_model(self):
        source = self.PFD.replace('    model = MSMPR\n', '').replace(
            '    residence_time = 1 [h]\n', ''
        )
        source = '\n'.join(
            line for line in source.splitlines()
            if not line.strip().startswith((
                'quadrature_classes', 'growth_', 'nucleation_', 'msmpr_',
            ))
        )
        result = Simulator.from_string(source).run()
        self.assertTrue(result.converged, result.errors)
        self.assertEqual(result.units['C'].performance['model'], 'equilibrium_pure_solids')


if __name__ == '__main__':
    unittest.main()
