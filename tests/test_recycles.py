import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulator import Simulator
from pfd_parser import ProcessFlowDiagram


def simple_water_recycle_pfd(recycle_spec: str = ''):
    return (
        'PROCESS: Tear Initial Guess\n'
        'VERSION: 1.0\n'
        'RECYCLE_METHOD: DIRECT\n'
        'TEAR_STREAMS: Recycle\n'
        '\n'
        'COMPONENTS:\n'
        '    H2O | Water | MW=18.015\n'
        '\n'
        'STREAM Fresh : FEED -> MIX-1.fresh\n'
        '    T = 25 [C]\n'
        '    P = 1 [bar]\n'
        '    F = 10 [kmol/h]\n'
        '    x = H2O:1.0\n'
        '\n'
        'STREAM Mixed : MIX-1.out -> SPLIT-1.in\n'
        'STREAM Product : SPLIT-1.product -> PRODUCT\n'
        'STREAM Recycle : SPLIT-1.recycle -> MIX-1.recycle\n'
        f'{recycle_spec}'
        '\n'
        'UNIT MIX-1\n'
        '    TYPE: Mixer\n'
        '    PORTS:\n'
        '        fresh : inlet\n'
        '        recycle : inlet\n'
        '        out : outlet\n'
        '    PARAMS:\n'
        '        mode = adiabatic\n'
        '\n'
        'UNIT SPLIT-1\n'
        '    TYPE: Splitter\n'
        '    PORTS:\n'
        '        in : inlet\n'
        '        product : outlet\n'
        '        recycle : outlet\n'
        '    PARAMS:\n'
        '        split_fracs = product:0.5,recycle:0.5\n'
    )


def initialized_recycle_stream(stream_id: str = 'Recycle') -> str:
    return (
        '    T = 25 [C]\n'
        '    P = 1 [bar]\n'
        '    F = 10 [kmol/h]\n'
        '    x = H2O:1.0\n'
    )


class RecycleSolverTests(unittest.TestCase):
    def test_method_options_parse_validate_and_round_trip(self):
        pfd_text = simple_water_recycle_pfd(
            initialized_recycle_stream()
        ).replace(
            'RECYCLE_METHOD: DIRECT',
            'RECYCLE_METHOD: WEGSTEIN | max_acceleration=-20, '
            'stagnation_iterations=12, fallback_damping=0.5',
        )
        simulator = Simulator.from_string(pfd_text)
        metadata = simulator.pfd.metadata

        self.assertEqual(metadata.recycle_method, 'WEGSTEIN')
        self.assertEqual(metadata.recycle_options, {
            'max_acceleration': -20.0,
            'stagnation_iterations': 12,
            'fallback_damping': 0.5,
        })
        rendered = simulator.pfd.to_pfd()
        self.assertIn(
            'RECYCLE_METHOD: WEGSTEIN | max_acceleration=-20, '
            'stagnation_iterations=12, fallback_damping=0.5',
            rendered,
        )
        roundtrip = ProcessFlowDiagram.from_dict(simulator.pfd.to_dict())
        self.assertEqual(roundtrip.metadata.recycle_options, metadata.recycle_options)

    def test_method_specific_options_are_rejected_when_inapplicable(self):
        cases = {
            'BROYDEN | max_acceleration=-20': 'does not support option',
            'WEGSTEIN | divergence_factor=20': 'does not support option',
            'WEGSTEIN | max_acceleration=1': 'between -100 and 0',
            'BROYDEN | stagnation_iterations=2.5': 'positive integer',
            'DIRECT | damping=0': 'greater than 0 and at most 1',
        }
        for directive, message in cases.items():
            with self.subTest(directive=directive):
                pfd_text = simple_water_recycle_pfd().replace(
                    'RECYCLE_METHOD: DIRECT',
                    f'RECYCLE_METHOD: {directive}',
                )
                with self.assertRaisesRegex(Exception, message):
                    Simulator.from_string(pfd_text)

    def test_wegstein_max_acceleration_controls_high_recycle_convergence(self):
        base = simple_water_recycle_pfd(
            initialized_recycle_stream()
        ).replace(
            'split_fracs = product:0.5,recycle:0.5',
            'split_frac = 0.98',
        )
        default = Simulator.from_string(base.replace(
            'RECYCLE_METHOD: DIRECT',
            'RECYCLE_METHOD: WEGSTEIN',
        )).run(max_iterations=100, tolerance=1e-6)
        accelerated = Simulator.from_string(base.replace(
            'RECYCLE_METHOD: DIRECT',
            'RECYCLE_METHOD: WEGSTEIN | max_acceleration=-20',
        )).run(max_iterations=100, tolerance=1e-6)

        self.assertFalse(default.converged)
        self.assertTrue(accelerated.converged, accelerated.warnings)
        self.assertLess(accelerated.iterations, default.iterations)
        self.assertEqual(
            accelerated.recycle_info['method_options']['max_acceleration'],
            -20.0,
        )

    def test_broyden_and_direct_options_are_resolved_and_reported(self):
        cases = (
            (
                'BROYDEN | stagnation_iterations=4, '
                'divergence_factor=25, fallback_damping=0.4',
                {
                    'stagnation_iterations': 4,
                    'divergence_factor': 25.0,
                    'fallback_damping': 0.4,
                },
            ),
            ('DIRECT | damping=0.5', {'damping': 0.5}),
        )
        for directive, expected in cases:
            with self.subTest(method=directive):
                pfd_text = simple_water_recycle_pfd(
                    initialized_recycle_stream()
                ).replace(
                    'RECYCLE_METHOD: DIRECT',
                    f'RECYCLE_METHOD: {directive}',
                )
                result = Simulator.from_string(pfd_text).run(
                    max_iterations=20,
                    tolerance=1e-7,
                )
                self.assertTrue(result.converged, result.warnings)
                self.assertEqual(result.recycle_info['method_options'], expected)

    def test_runtime_method_override_uses_that_methods_defaults(self):
        pfd_text = simple_water_recycle_pfd(
            initialized_recycle_stream()
        ).replace(
            'RECYCLE_METHOD: DIRECT',
            'RECYCLE_METHOD: WEGSTEIN | max_acceleration=-20',
        )
        result = Simulator.from_string(pfd_text).run(
            max_iterations=20,
            tolerance=1e-7,
            recycle_method='BROYDEN',
        )

        self.assertTrue(result.converged, result.warnings)
        self.assertEqual(result.recycle_info['method'], 'BROYDEN')
        self.assertEqual(result.recycle_info['method_options'], {
            'stagnation_iterations': 8,
            'divergence_factor': 10.0,
            'fallback_damping': 1.0,
        })

    def test_tear_stream_specs_initialize_recycle_guess(self):
        pfd = simple_water_recycle_pfd(initialized_recycle_stream())

        messages = []
        result = Simulator.from_string(pfd).run(
            max_iterations=10,
            tolerance=1e-7,
            progress_callback=messages.append,
        )

        self.assertTrue(result.converged, result.warnings)
        initialized = [
            message for message in messages
            if message.startswith('tear_streams_initialized')
        ]
        self.assertTrue(initialized)
        self.assertIn('Recycle:', initialized[0])
        self.assertIn('F=10', initialized[0])
        self.assertAlmostEqual(result.streams['Recycle'].F, 10.0, places=6)

    def test_empty_tear_stream_still_uses_fallback_initial_guess(self):
        messages = []
        result = Simulator.from_string(simple_water_recycle_pfd()).run(
            max_iterations=10,
            tolerance=1e-3,
            progress_callback=messages.append,
        )

        self.assertTrue(result.converged, result.warnings)
        initialized = [
            message for message in messages
            if message.startswith('tear_streams_initialized')
        ]
        self.assertTrue(initialized)
        self.assertIn('Recycle:', initialized[0])

    def test_incomplete_tear_stream_initial_guess_reports_error(self):
        pfd = simple_water_recycle_pfd(
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    x = H2O:1.0\n'
        )

        result = Simulator.from_string(pfd).run()

        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertIn(
            "Tear stream 'Recycle' initial guess must specify P, F, and composition",
            result.errors,
        )

    def test_downstream_unit_outside_recycle_block_is_deferred_until_final_pass(self):
        pfd = (
            'PROCESS: Downstream Recycle Deferral\n'
            'VERSION: 1.0\n'
            'RECYCLE_METHOD: DIRECT\n'
            'TEAR_STREAMS: Recycle\n'
            '\n'
            'COMPONENTS:\n'
            '    H2O | Water | MW=18.015\n'
            '\n'
            'STREAM Fresh : FEED -> MIX-1.fresh\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 10 [kmol/h]\n'
            '    x = H2O:1.0\n'
            '\n'
            'STREAM Mixed : MIX-1.out -> SPLIT-1.in\n'
            'STREAM ToHeater : SPLIT-1.product -> HEAT-1.in\n'
            'STREAM Product : HEAT-1.out -> PRODUCT\n'
            'STREAM Recycle : SPLIT-1.recycle -> MIX-1.recycle\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 10 [kmol/h]\n'
            '    x = H2O:1.0\n'
            '\n'
            'UNIT MIX-1\n'
            '    TYPE: Mixer\n'
            '    PORTS:\n'
            '        fresh : inlet\n'
            '        recycle : inlet\n'
            '        out : outlet\n'
            '    PARAMS:\n'
            '        mode = adiabatic\n'
            '\n'
            'UNIT SPLIT-1\n'
            '    TYPE: Splitter\n'
            '    PORTS:\n'
            '        in : inlet\n'
            '        product : outlet\n'
            '        recycle : outlet\n'
            '    PARAMS:\n'
            '        split_fracs = product:0.5,recycle:0.5\n'
            '\n'
            'UNIT HEAT-1\n'
            '    TYPE: Heater\n'
            '    PORTS:\n'
            '        in : inlet\n'
            '        out : outlet\n'
            '    PARAMS:\n'
            '        T_out = 35 [C]\n'
        )

        messages = []
        result = Simulator.from_string(pfd).run(
            max_iterations=10,
            tolerance=1e-7,
            progress_callback=messages.append,
        )

        self.assertTrue(result.converged, result.warnings)
        self.assertIn('HEAT-1', result.recycle_info['deferred_units'])
        self.assertIn('HEAT-1', result.units)
        self.assertAlmostEqual(result.streams['Product'].T - 273.15, 35.0, places=6)
        self.assertTrue(any('unit_skipped HEAT-1' in message for message in messages))

    def test_invariant_upstream_units_are_cached_across_recycle_evaluations(self):
        pfd = simple_water_recycle_pfd(
            initialized_recycle_stream().replace(
                'F = 10 [kmol/h]',
                'F = 1 [kmol/h]',
            )
        ).replace(
            'STREAM Fresh : FEED -> MIX-1.fresh',
            'STREAM Fresh : FEED -> PRE-1.in',
        ).replace(
            'STREAM Mixed : MIX-1.out -> SPLIT-1.in',
            'STREAM Prepared : PRE-1.out -> MIX-1.fresh\n'
            'STREAM Mixed : MIX-1.out -> SPLIT-1.in',
        ).replace(
            'UNIT MIX-1\n',
            'UNIT PRE-1\n'
            '    TYPE: Heater\n'
            '    PORTS:\n'
            '        in : inlet\n'
            '        out : outlet\n'
            '    PARAMS:\n'
            '        T_out = 30 [C]\n'
            '\n'
            'UNIT MIX-1\n',
        )

        messages = []
        result = Simulator.from_string(pfd).run(
            max_iterations=40,
            tolerance=1e-7,
            progress_callback=messages.append,
        )

        self.assertTrue(result.converged, result.warnings)
        self.assertGreater(result.iterations, 2)
        block = result.recycle_info['recycle_blocks'][0]
        self.assertEqual(block['invariant_units'], ['PRE-1'])
        self.assertEqual(block['repeated_units'], ['MIX-1', 'SPLIT-1'])
        self.assertEqual(result.recycle_info['invariant_units'], ['PRE-1'])
        preheater_solves = [
            message for message in messages
            if message.startswith('unit_done PRE-1 ')
        ]
        self.assertEqual(
            len(preheater_solves),
            2,
            'Invariant unit should run once for recycle and once for final results',
        )
        self.assertTrue(any(
            message == 'unit_cached PRE-1 recycle_invariant'
            for message in messages
        ))

    def test_nested_recycle_tears_share_one_block_and_defer_only_after_outer_block(self):
        pfd = (
            'PROCESS: Nested Recycle Blocks\n'
            'VERSION: 1.0\n'
            'RECYCLE_METHOD: BROYDEN\n'
            'TEAR_STREAMS: InnerRecycle, OuterRecycle\n'
            '\n'
            'COMPONENTS:\n'
            '    H2O | Water | MW=18.015\n'
            '\n'
            'STREAM Fresh : FEED -> MIX-1.fresh\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 10 [kmol/h]\n'
            '    x = H2O:1.0\n'
            '\n'
            'STREAM Mixed1 : MIX-1.out -> SPLIT-1.in\n'
            'STREAM InnerProduct : SPLIT-1.product -> MIX-2.fresh\n'
            'STREAM InnerRecycle : SPLIT-1.recycle -> MIX-1.inner\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 4 [kmol/h]\n'
            '    x = H2O:1.0\n'
            'STREAM Mixed2 : MIX-2.out -> SPLIT-2.in\n'
            'STREAM ToHeater : SPLIT-2.product -> HEAT-1.in\n'
            'STREAM OuterRecycle : SPLIT-2.recycle -> MIX-1.outer\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 2 [kmol/h]\n'
            '    x = H2O:1.0\n'
            'STREAM Product : HEAT-1.out -> PRODUCT\n'
            '\n'
            'UNIT MIX-1\n'
            '    TYPE: Mixer\n'
            '    PORTS:\n'
            '        fresh : inlet\n'
            '        inner : inlet\n'
            '        outer : inlet\n'
            '        out : outlet\n'
            '    PARAMS:\n'
            '        mode = adiabatic\n'
            '\n'
            'UNIT SPLIT-1\n'
            '    TYPE: Splitter\n'
            '    PORTS:\n'
            '        in : inlet\n'
            '        product : outlet\n'
            '        recycle : outlet\n'
            '    PARAMS:\n'
            '        split_fracs = product:0.6,recycle:0.4\n'
            '\n'
            'UNIT MIX-2\n'
            '    TYPE: Mixer\n'
            '    PORTS:\n'
            '        fresh : inlet\n'
            '        out : outlet\n'
            '    PARAMS:\n'
            '        mode = adiabatic\n'
            '\n'
            'UNIT SPLIT-2\n'
            '    TYPE: Splitter\n'
            '    PORTS:\n'
            '        in : inlet\n'
            '        product : outlet\n'
            '        recycle : outlet\n'
            '    PARAMS:\n'
            '        split_fracs = product:0.7,recycle:0.3\n'
            '\n'
            'UNIT HEAT-1\n'
            '    TYPE: Heater\n'
            '    PORTS:\n'
            '        in : inlet\n'
            '        out : outlet\n'
            '    PARAMS:\n'
            '        T_out = 35 [C]\n'
        )

        messages = []
        result = Simulator.from_string(pfd).run(
            max_iterations=30,
            tolerance=1e-7,
            progress_callback=messages.append,
        )

        self.assertTrue(result.converged, result.warnings)
        blocks = result.recycle_info['recycle_blocks']
        self.assertEqual(len(blocks), 1)
        self.assertEqual(
            set(blocks[0]['tear_streams']),
            {'InnerRecycle', 'OuterRecycle'},
        )
        self.assertEqual(
            set(blocks[0]['required_units']),
            {'MIX-1', 'SPLIT-1', 'MIX-2', 'SPLIT-2'},
        )
        self.assertEqual(
            set(blocks[0]['repeated_units']),
            {'MIX-1', 'SPLIT-1', 'MIX-2', 'SPLIT-2'},
        )
        self.assertEqual(blocks[0]['invariant_units'], [])
        self.assertEqual(blocks[0]['deferred_units'], ['HEAT-1'])
        self.assertTrue(any('unit_skipped HEAT-1' in message for message in messages))

    def test_disconnected_recycle_loops_are_solved_as_separate_deferred_blocks(self):
        pfd = (
            'PROCESS: Disconnected Recycle Blocks\n'
            'VERSION: 1.0\n'
            'RECYCLE_METHOD: DIRECT\n'
            'TEAR_STREAMS: RecycleA, RecycleB\n'
            '\n'
            'COMPONENTS:\n'
            '    H2O | Water | MW=18.015\n'
            '\n'
            'STREAM FreshA : FEED -> MIX-A.fresh\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 10 [kmol/h]\n'
            '    x = H2O:1.0\n'
            'STREAM MixedA : MIX-A.out -> SPLIT-A.in\n'
            'STREAM ProductA : SPLIT-A.product -> PRODUCT\n'
            'STREAM RecycleA : SPLIT-A.recycle -> MIX-A.recycle\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 10 [kmol/h]\n'
            '    x = H2O:1.0\n'
            '\n'
            'STREAM FreshB : FEED -> MIX-B.fresh\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 8 [kmol/h]\n'
            '    x = H2O:1.0\n'
            'STREAM MixedB : MIX-B.out -> SPLIT-B.in\n'
            'STREAM ProductB : SPLIT-B.product -> PRODUCT\n'
            'STREAM RecycleB : SPLIT-B.recycle -> MIX-B.recycle\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 8 [kmol/h]\n'
            '    x = H2O:1.0\n'
            '\n'
            'UNIT MIX-A\n'
            '    TYPE: Mixer\n'
            '    PORTS:\n'
            '        fresh : inlet\n'
            '        recycle : inlet\n'
            '        out : outlet\n'
            '    PARAMS:\n'
            '        mode = adiabatic\n'
            'UNIT SPLIT-A\n'
            '    TYPE: Splitter\n'
            '    PORTS:\n'
            '        in : inlet\n'
            '        product : outlet\n'
            '        recycle : outlet\n'
            '    PARAMS:\n'
            '        split_fracs = product:0.5,recycle:0.5\n'
            '\n'
            'UNIT MIX-B\n'
            '    TYPE: Mixer\n'
            '    PORTS:\n'
            '        fresh : inlet\n'
            '        recycle : inlet\n'
            '        out : outlet\n'
            '    PARAMS:\n'
            '        mode = adiabatic\n'
            'UNIT SPLIT-B\n'
            '    TYPE: Splitter\n'
            '    PORTS:\n'
            '        in : inlet\n'
            '        product : outlet\n'
            '        recycle : outlet\n'
            '    PARAMS:\n'
            '        split_fracs = product:0.5,recycle:0.5\n'
        )

        messages = []
        result = Simulator.from_string(pfd).run(
            max_iterations=10,
            tolerance=1e-7,
            progress_callback=messages.append,
        )

        self.assertTrue(result.converged, result.warnings)
        blocks = result.recycle_info['recycle_blocks']
        self.assertEqual(len(blocks), 2)
        first, second = blocks
        self.assertEqual(first['tear_streams'], ['RecycleA'])
        self.assertEqual(set(first['repeated_units']), {'MIX-A', 'SPLIT-A'})
        self.assertEqual(first['invariant_units'], [])
        self.assertIn('MIX-B', first['deferred_units'])
        self.assertIn('SPLIT-B', first['deferred_units'])
        self.assertEqual(second['tear_streams'], ['RecycleB'])
        self.assertEqual(set(second['repeated_units']), {'MIX-B', 'SPLIT-B'})
        self.assertEqual(second['invariant_units'], [])
        self.assertIn('MIX-A', second['deferred_units'])
        self.assertIn('SPLIT-A', second['deferred_units'])
        self.assertTrue(any('unit_skipped MIX-B' in message for message in messages))
        self.assertTrue(any('unit_skipped MIX-A' in message for message in messages))

if __name__ == '__main__':
    unittest.main()
