import math
import unittest

from chemical_properties import ChemicalDatabase
from particle_size_distributions import (
    ParticleSizeDistribution,
    carry_particle_size_distributions,
    instantiate_particle_size_distribution,
    normalize_particle_size_distribution,
)
from pfd_parser import ProcessFlowDiagram, parse_pfd, validate_pfd
from simulator import Simulator
from thermodynamics_models.base import IdealThermodynamics
from unit_operations_basic import HeatExchanger, Mixer, Splitter
from unit_operations_solids import Crystallizer


class ParticleSizeDistributionTests(unittest.TestCase):
    def test_lognormal_uses_d50_and_geometric_standard_deviation(self):
        specification = normalize_particle_size_distribution({
            'distribution': 'lognormal',
            'd50': 200,
            'diameter_unit': 'um',
            'geometric_standard_deviation': 1.6,
            'basis': 'mass',
            'classes': 5,
        })
        self.assertEqual(specification['distribution'], 'lognormal')
        self.assertAlmostEqual(specification['d50_m'], 2.0e-4)
        self.assertEqual(specification['geometric_standard_deviation'], 1.6)
        self.assertEqual(specification['basis'], 'mass')
        self.assertEqual(specification['classes'], 5)
        distribution = instantiate_particle_size_distribution(
            specification, 10.0
        )
        self.assertEqual(len(distribution.diameters_m), 5)
        self.assertAlmostEqual(distribution.diameters_m[2], 2.0e-4)
        self.assertEqual(distribution.molar_flows_kmol_per_h, (2.0,) * 5)

    def test_lognormal_number_basis_has_equal_number_classes(self):
        distribution = instantiate_particle_size_distribution({
            'distribution': 'lognormal',
            'd50_um': 20,
            'gsd': 1.8,
            'basis': 'number',
            'classes': 5,
        }, 3.0)
        for fraction in distribution.number_fractions:
            self.assertAlmostEqual(fraction, 0.2)

    def test_weibull_alias_uses_d632_scale_and_shape(self):
        specification = normalize_particle_size_distribution({
            'distribution': 'rosin-rammler',
            'd63_2_um': 500,
            'exponent': 2.0,
            'basis': 'volume',
            'classes': 5,
        })
        self.assertEqual(specification, {
            'distribution': 'weibull',
            'scale_diameter_m': 5.0e-4,
            'shape': 2.0,
            'basis': 'volume',
            'classes': 5,
        })
        distribution = instantiate_particle_size_distribution(
            specification, 5.0
        )
        expected_median = 5.0e-4 * math.sqrt(math.log(2.0))
        self.assertAlmostEqual(distribution.diameters_m[2], expected_median)
        self.assertEqual(distribution.molar_flows_kmol_per_h, (1.0,) * 5)

    def test_zero_width_lognormal_is_monodisperse(self):
        distribution = instantiate_particle_size_distribution({
            'distribution': 'lognormal',
            'd50_mm': 0.25,
            'sigma_g': 1.0,
            'basis': 'mass',
            'classes': 20,
        }, 4.0)
        self.assertEqual(distribution.diameters_m, (2.5e-4,))
        self.assertEqual(distribution.molar_flows_kmol_per_h, (4.0,))

    def test_parametric_distributions_require_explicit_basis(self):
        with self.assertRaisesRegex(ValueError, 'explicit basis'):
            normalize_particle_size_distribution({
                'distribution': 'lognormal',
                'd50_um': 100,
                'gsd': 1.5,
            })
        with self.assertRaisesRegex(ValueError, 'positive'):
            normalize_particle_size_distribution({
                'distribution': 'weibull',
                'scale_diameter_um': 100,
                'shape': 0,
                'basis': 'mass',
            })

    def test_number_basis_converts_to_conserved_component_flow(self):
        distribution = instantiate_particle_size_distribution(
            {
                'diameters_um': [10, 20],
                'fractions': [0.5, 0.5],
                'basis': 'number',
            },
            9.0,
        )
        for actual, expected in zip(distribution.diameters_m, (1.0e-5, 2.0e-5)):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(distribution.total_molar_flow, 9.0)
        self.assertAlmostEqual(distribution.molar_flows_kmol_per_h[0], 1.0)
        self.assertAlmostEqual(distribution.molar_flows_kmol_per_h[1], 8.0)
        self.assertAlmostEqual(distribution.number_fractions[0], 0.5)
        self.assertAlmostEqual(distribution.number_fractions[1], 0.5)

    def test_distribution_scales_and_combines_as_extensive_inventory(self):
        first = ParticleSizeDistribution((1.0e-4,), (2.0,))
        second = ParticleSizeDistribution((1.0e-4, 2.0e-4), (1.0, 3.0))
        combined = ParticleSizeDistribution.combine((first, second))
        self.assertEqual(combined.diameters_m, (1.0e-4, 2.0e-4))
        self.assertEqual(combined.molar_flows_kmol_per_h, (3.0, 3.0))
        scaled = combined.with_total_molar_flow(2.0)
        self.assertEqual(scaled.molar_flows_kmol_per_h, (1.0, 1.0))
        self.assertAlmostEqual(scaled.sauter_mean_diameter_m, 4.0e-4 / 3.0)

    def test_numerical_reconstruction_rescales_without_blending_default(self):
        source = self._state_with_population(
            2.0,
            ParticleSizeDistribution((5.0e-5, 2.0e-4), (0.5, 1.5)),
        )
        target = self._state_with_population(
            8.0,
            ParticleSizeDistribution((1.0e-4,), (8.0,)),
        )
        carry_particle_size_distributions(source, target)
        distribution = target.solid_particle_size_distributions['salt']
        self.assertEqual(distribution.diameters_m, (5.0e-5, 2.0e-4))
        self.assertEqual(distribution.molar_flows_kmol_per_h, (2.0, 6.0))

    @staticmethod
    def _state_with_population(flow, distribution):
        from thermodynamics_models.base import StreamState

        return StreamState(
            T=298.15,
            P=1.0,
            F=flow,
            composition={'salt': 1.0},
            solid_fraction=1.0,
            solid_composition={'salt': 1.0},
            solid_component_flows={'salt': flow},
            solid_particle_size_distributions={'salt': distribution},
        )

    def test_equivalent_converted_diameters_combine_into_one_class(self):
        first = instantiate_particle_size_distribution({
            'diameters_um': [100],
            'fractions': [1],
            'basis': 'mass',
        }, 2.0)
        second = instantiate_particle_size_distribution({
            'diameters_m': [0.0001],
            'fractions': [1],
            'basis': 'mass',
        }, 3.0)
        combined = ParticleSizeDistribution.combine((first, second))
        self.assertEqual(combined.diameters_m, (0.0001,))
        self.assertEqual(combined.molar_flows_kmol_per_h, (5.0,))

    def test_invalid_distributions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            normalize_particle_size_distribution({
                'diameters_m': [2.0e-4, 1.0e-4],
                'fractions': [0.5, 0.5],
            })
        with self.assertRaisesRegex(ValueError, 'basis'):
            normalize_particle_size_distribution({
                'diameters_m': [1.0e-4],
                'fractions': [1.0],
                'basis': 'length',
            })


class ParticleSizePFDTests(unittest.TestCase):
    COMPONENTS = """
COMPONENTS:
    water | Water | MW=18.0153
    salt | Sodium chloride | CAS=7647-14-5, MW=58.4428, type=permanent_solid, PSD={diameters_um:[50, 100, 200], fractions:[0.2, 0.3, 0.5], basis:mass}, particle_sphericity=0.85
"""

    def test_component_default_and_stream_override_round_trip(self):
        pfd = parse_pfd(
            f"""
PROCESS: PSD parser
VERSION: 1.0
{self.COMPONENTS}
STREAM Feed : FEED -> PRODUCT
    T = 25 [C]
    P = 1 [bar]
    F = 2 [kmol/h]
    x = salt:1
    PSD = {{salt:{{diameters_um:[25, 75], fractions:[0.25, 0.75], basis:number}}}}
"""
        )
        component_psd = pfd.components[1].particle_size_distribution
        for actual, expected in zip(
            component_psd['diameters_m'],
            [5.0e-5, 1.0e-4, 2.0e-4],
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(component_psd['basis'], 'mass')
        self.assertEqual(
            pfd.streams[0].particle_size_distributions['salt']['basis'],
            'number',
        )
        reparsed = parse_pfd(pfd.to_pfd())
        self.assertEqual(
            reparsed.streams[0].particle_size_distributions,
            pfd.streams[0].particle_size_distributions,
        )
        restored = ProcessFlowDiagram.from_dict(pfd.to_dict())
        self.assertEqual(
            restored.components[1].particle_size_distribution,
            component_psd,
        )

    def test_parametric_component_and_feed_psds_round_trip(self):
        pfd = parse_pfd(
            """
PROCESS: parametric PSD parser
VERSION: 1.0
COMPONENTS:
    salt | Sodium chloride | type=permanent_solid, PSD={distribution:lognormal, d50_um:150, GSD:1.7, basis:mass, classes:21}
STREAM Feed : FEED -> PRODUCT
    T = 25 [C]
    P = 1 [bar]
    F = 2 [kmol/h]
    x = salt:1
    PSD = {salt:{distribution:rosin_rammler, d63_2_um:300, shape:2.5, basis:number, classes:25}}
"""
        )
        self.assertEqual(
            pfd.components[0].particle_size_distribution['distribution'],
            'lognormal',
        )
        stream_specification = pfd.streams[0].particle_size_distributions['salt']
        self.assertEqual(stream_specification['distribution'], 'weibull')
        self.assertEqual(stream_specification['classes'], 25)
        reparsed = parse_pfd(pfd.to_pfd())
        self.assertEqual(
            reparsed.components[0].particle_size_distribution,
            pfd.components[0].particle_size_distribution,
        )
        self.assertEqual(
            reparsed.streams[0].particle_size_distributions,
            pfd.streams[0].particle_size_distributions,
        )

    def test_psd_requires_a_solid_enabled_component(self):
        pfd = parse_pfd(
            """
PROCESS: invalid PSD
VERSION: 1.0
COMPONENTS:
    water | Water | PSD={diameters_um:[100], fractions:[1]}
"""
        )
        errors, _warnings = validate_pfd(pfd)
        self.assertTrue(any('Particle defaults' in error for error in errors))

    def test_conventional_solid_feed_psd_requires_future_allocation_syntax(self):
        pfd = parse_pfd(
            """
PROCESS: conventional-solid feed PSD
VERSION: 1.0
COMPONENTS:
    water | Water | phase_behavior=conventional_with_solid, PSD={diameters_um:[100], fractions:[1]}
STREAM Feed : FEED -> PRODUCT
    T = 25 [C]
    P = 1 [bar]
    F = 1 [kmol/h]
    x = water:1
    PSD = {water:{diameters_um:[200], fractions:[1]}}
"""
        )
        errors, _warnings = validate_pfd(pfd)
        self.assertTrue(any(
            'future explicit solid-allocation syntax' in error
            for error in errors
        ))

    def test_scalar_diameter_and_psd_are_mutually_exclusive(self):
        pfd = parse_pfd(
            """
PROCESS: conflicting PSD
VERSION: 1.0
COMPONENTS:
    salt | Sodium chloride | type=permanent_solid, particle_diameter=0.0001, PSD={diameters_um:[100], fractions:[1]}
"""
        )
        errors, _warnings = validate_pfd(pfd)
        self.assertTrue(any('cannot specify both' in error for error in errors))

    def test_feed_override_instantiates_extensive_stream_population(self):
        simulator = Simulator.from_string(
            f"""
PROCESS: feed PSD
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
{self.COMPONENTS}
STREAM Feed : FEED -> H.in
    T = 25 [C]
    P = 1 [bar]
    F = 10 [kmol/h]
    x = water:0.8, salt:0.2
    PSD = {{salt:{{diameters_um:[20, 80], fractions:[0.75, 0.25], basis:mass}}}}
STREAM Product : H.out -> PRODUCT
UNIT H : Heater
    T_out = 40 [C]
"""
        )
        result = simulator.run()
        feed_psd = result.streams['Feed'].solid_particle_size_distributions['salt']
        product_psd = result.streams['Product'].solid_particle_size_distributions['salt']
        for actual, expected in zip(feed_psd.diameters_m, (2.0e-5, 8.0e-5)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(feed_psd.molar_flows_kmol_per_h, (1.5, 0.5))
        self.assertEqual(product_psd, feed_psd)
        payload = result.streams['Product'].to_dict()
        self.assertEqual(
            payload['solid_particle_size_distributions']['salt'][
                'molar_flows_kmol_per_h'
            ],
            [1.5, 0.5],
        )
        report = simulator._generate_pfr()
        self.assertIn('SOLID_PARTICLE_SIZE_DISTRIBUTIONS:', report)
        self.assertIn('diameter=2e-05 [m], F=1.5 [kmol/h]', report)

    def test_scope_transition_preserves_feed_override_population(self):
        simulator = Simulator.from_string(
            f"""
PROCESS: scoped feed PSD
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
THERMO_SCOPES:
    scoped | method=IDEAL
{self.COMPONENTS}
STREAM Feed : FEED -> H.in
    T = 25 [C]
    P = 1 [bar]
    F = 10 [kmol/h]
    x = water:0.8, salt:0.2
    PSD = {{salt:{{diameters_um:[20, 80], fractions:[0.75, 0.25], basis:mass}}}}
STREAM Product : H.out -> PRODUCT
UNIT H : Heater
    T_out = 40 [C]
    thermo_scope = scoped
"""
        )
        result = simulator.run()
        feed = result.streams['Feed'].solid_particle_size_distributions['salt']
        product = result.streams['Product'].solid_particle_size_distributions['salt']
        self.assertEqual(product, feed)

    def test_recycle_converges_particle_class_flows(self):
        template = """
PROCESS: PSD recycle
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
RECYCLE_METHOD: __METHOD__
COMPONENTS:
    water | Water | MW=18.0153
    salt | Sodium chloride | CAS=7647-14-5, MW=58.4428, type=permanent_solid, PSD={diameters_um:[100], fractions:[1], basis:mass}
STREAM WaterFeed : FEED -> M.water
    T = 25 [C]
    P = 1 [bar]
    F = 8 [kmol/h]
    x = water:1
STREAM SaltFeed : FEED -> M.salt
    T = 25 [C]
    P = 1 [bar]
    F = 2 [kmol/h]
    x = salt:1
    PSD = {salt:{diameters_um:[50, 200], fractions:[0.25, 0.75], basis:mass}}
STREAM Recycle : SP.recycle -> M.recycle
STREAM Mixed : M.out -> SP.in
STREAM Product : SP.product -> PRODUCT
UNIT M : Mixer
    T_out = 25 [C]
    P = 1 [bar]
UNIT SP : Splitter
    outlets = product, recycle
    product_split_frac = 0.5
"""
        for method in ('DIRECT', 'WEGSTEIN', 'BROYDEN'):
            with self.subTest(method=method):
                result = Simulator.from_string(
                    template.replace('__METHOD__', method)
                ).run(max_iterations=100)
                self.assertTrue(result.converged, result.errors)
                for stream_name in ('Recycle', 'Product'):
                    distribution = result.streams[
                        stream_name
                    ].solid_particle_size_distributions['salt']
                    fractions = dict(zip(
                        distribution.diameters_m,
                        distribution.molar_fractions,
                    ))
                    small = min(fractions, key=lambda value: abs(value - 5.0e-5))
                    large = min(fractions, key=lambda value: abs(value - 2.0e-4))
                    self.assertAlmostEqual(small, 5.0e-5)
                    self.assertAlmostEqual(large, 2.0e-4)
                    self.assertAlmostEqual(fractions[small], 0.25, places=4)
                    self.assertAlmostEqual(fractions[large], 0.75, places=4)
                    default_fraction = sum(
                        fraction
                        for diameter, fraction in fractions.items()
                        if abs(diameter - 1.0e-4) < 1.0e-12
                    )
                    self.assertLess(default_fraction, 1.0e-4)

    def test_accelerated_high_recycle_does_not_inject_component_default(self):
        template = """
PROCESS: high PSD recycle
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
RECYCLE_METHOD: __METHOD__
COMPONENTS:
    water | Water | MW=18.0153
    salt | Sodium chloride | CAS=7647-14-5, MW=58.4428, type=permanent_solid, PSD={diameters_um:[100], fractions:[1], basis:mass}
STREAM WaterFeed : FEED -> M.water
    T = 25 [C]
    P = 1 [bar]
    F = 8 [kmol/h]
    x = water:1
STREAM SaltFeed : FEED -> M.salt
    T = 25 [C]
    P = 1 [bar]
    F = 2 [kmol/h]
    x = salt:1
    PSD = {salt:{diameters_um:[50, 200], fractions:[0.25, 0.75], basis:mass}}
STREAM Recycle : SP.recycle -> M.recycle
STREAM Mixed : M.out -> SP.in
STREAM Product : SP.product -> PRODUCT
UNIT M : Mixer
    T_out = 25 [C]
    P = 1 [bar]
UNIT SP : Splitter
    outlets = product, recycle
    product_split_frac = 0.05
"""
        for method in ('WEGSTEIN', 'BROYDEN'):
            with self.subTest(method=method):
                result = Simulator.from_string(
                    template.replace('__METHOD__', method)
                ).run(max_iterations=100)
                self.assertTrue(result.converged, result.errors)
                distribution = result.streams[
                    'Recycle'
                ].solid_particle_size_distributions['salt']
                self.assertEqual(distribution.diameters_m, (5.0e-5, 2.0e-4))
                self.assertAlmostEqual(distribution.molar_fractions[0], 0.25)
                self.assertAlmostEqual(distribution.molar_fractions[1], 0.75)


class ParticleSizeUnitPropagationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = ChemicalDatabase(enable_online=False)

    def make_thermo(self):
        thermo = IdealThermodynamics(['water'], self.db)
        thermo.configure_permanent_solids(
            ['water', 'NaCl'],
            ['NaCl'],
            {
                'NaCl': {
                    'sphericity': 0.9,
                    'particle_size_distribution': {
                        'diameters_m': [1.0e-4, 2.0e-4],
                        'fractions': [0.25, 0.75],
                        'basis': 'mass',
                    },
                },
            },
        )
        return thermo

    def test_mixer_combines_component_populations(self):
        thermo = self.make_thermo()
        first = thermo.calculate_state(298.15, 1.0, 2.0, {'NaCl': 1.0})
        second = thermo.calculate_state(298.15, 1.0, 4.0, {'NaCl': 1.0})
        second.solid_particle_size_distributions['NaCl'] = (
            ParticleSizeDistribution((3.0e-4,), (4.0,))
        )
        result = Mixer('M', thermo, {'T_out': 298.15, 'P': 1.0}).solve({
            'a': first,
            'b': second,
        })
        distribution = result.outlet_streams['out'].solid_particle_size_distributions[
            'NaCl'
        ]
        self.assertEqual(distribution.diameters_m, (1.0e-4, 2.0e-4, 3.0e-4))
        self.assertEqual(distribution.molar_flows_kmol_per_h, (0.5, 1.5, 4.0))

    def test_bulk_and_component_splits_conserve_each_size_class(self):
        thermo = self.make_thermo()
        slurry = thermo.calculate_state(
            298.15, 1.0, 10.0, {'water': 0.8, 'NaCl': 0.2}
        )
        bulk = Splitter(
            'B', thermo, {'outlets': 'a,b', 'split_frac_a': 0.3}
        ).solve({'in': slurry})
        for actual, expected in zip(
            bulk.outlet_streams['a'].solid_particle_size_distributions[
                'NaCl'
            ].molar_flows_kmol_per_h,
            (0.15, 0.45),
        ):
            self.assertAlmostEqual(actual, expected)
        component = Splitter('C', thermo, {
            'outlets': 'liquid,solids',
            'component_splits': {
                'water': {'liquid': 1.0},
                'NaCl': {'solids': 1.0},
            },
        }).solve({'in': slurry})
        self.assertNotIn(
            'NaCl',
            component.outlet_streams['liquid'].solid_particle_size_distributions,
        )
        self.assertEqual(
            component.outlet_streams['solids'].solid_particle_size_distributions[
                'NaCl'
            ],
            slurry.solid_particle_size_distributions['NaCl'],
        )

    def test_heat_exchanger_keeps_side_populations_separate(self):
        thermo = self.make_thermo()
        hot = thermo.calculate_state(
            350.0, 1.0, 10.0, {'water': 0.8, 'NaCl': 0.2}
        )
        cold = thermo.calculate_state(
            290.0, 1.0, 10.0, {'water': 0.8, 'NaCl': 0.2}
        )
        cold.solid_particle_size_distributions['NaCl'] = (
            ParticleSizeDistribution((3.0e-4,), (2.0,))
        )
        result = HeatExchanger(
            'HX', thermo, {'T_hot_out': 330.0, 'curve_segments': 4}
        ).solve({'hot_in': hot, 'cold_in': cold})
        self.assertEqual(
            result.outlet_streams['hot_out'].solid_particle_size_distributions[
                'NaCl'
            ],
            hot.solid_particle_size_distributions['NaCl'],
        )
        self.assertEqual(
            result.outlet_streams['cold_out'].solid_particle_size_distributions[
                'NaCl'
            ],
            cold.solid_particle_size_distributions['NaCl'],
        )

    def test_existing_scalar_diameter_becomes_monodisperse_population(self):
        thermo = IdealThermodynamics([], self.db)
        thermo.configure_permanent_solids(
            ['NaCl'],
            ['NaCl'],
            {'NaCl': {'diameter_m': 1.5e-4, 'sphericity': 0.8}},
        )
        state = thermo.calculate_state(298.15, 1.0, 3.0, {'NaCl': 1.0})
        distribution = state.solid_particle_size_distributions['NaCl']
        self.assertEqual(distribution.diameters_m, (1.5e-4,))
        self.assertEqual(distribution.molar_flows_kmol_per_h, (3.0,))

    def test_conventional_solid_default_instantiates_on_explicit_solid_flow(self):
        thermo = IdealThermodynamics(['water'], self.db)
        thermo.configure_permanent_solids(
            ['water'],
            [],
            {
                'water': {
                    'particle_size_distribution': {
                        'diameters_um': [100, 300],
                        'fractions': [0.4, 0.6],
                        'basis': 'mass',
                    },
                },
            },
            conventional_solid_components=['water'],
        )
        state = thermo.calculate_state_with_solid_flows(
            250.0,
            1.0,
            10.0,
            {'water': 1.0},
            {'water': 2.5},
            phase='liquid',
        )
        distribution = state.solid_particle_size_distributions['water']
        self.assertEqual(distribution.molar_flows_kmol_per_h, (1.0, 1.5))
        state.validate_particle_size_distributions()

    def test_crystallizer_assigns_default_psd_to_new_crystals(self):
        thermo = IdealThermodynamics(['water', 'ethanol'], self.db)
        thermo.configure_permanent_solids(
            ['water', 'ethanol'],
            [],
            {
                'water': {
                    'particle_size_distribution': {
                        'diameters_um': [100, 300],
                        'fractions': [0.4, 0.6],
                        'basis': 'mass',
                    },
                },
            },
            conventional_solid_components=['water'],
        )
        feed = thermo.calculate_state(
            280.0,
            1.0,
            10.0,
            {'water': 0.9, 'ethanol': 0.1},
            phase='liquid',
        )
        slurry = Crystallizer('C', thermo, {'T': 250.0}).solve({
            'in': feed,
        }).outlet_streams['out']
        distribution = slurry.solid_particle_size_distributions['water']
        self.assertAlmostEqual(
            distribution.total_molar_flow,
            slurry.solid_component_flows['water'],
        )
        self.assertEqual(len(distribution.molar_fractions), 2)
        for actual, expected in zip(
            distribution.molar_fractions, (0.4, 0.6)
        ):
            self.assertAlmostEqual(actual, expected, places=15)

    def test_state_rejects_partial_population(self):
        state = self.make_thermo().calculate_state(
            298.15, 1.0, 2.0, {'NaCl': 1.0}
        )
        state.solid_particle_size_distributions['NaCl'] = (
            ParticleSizeDistribution((1.0e-4,), (1.0,))
        )
        with self.assertRaisesRegex(ValueError, 'accounts for'):
            state.validate_particle_size_distributions()


if __name__ == '__main__':
    unittest.main()
