import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tests import performance_all_examples as performance


class PerformanceCPUReferenceTests(unittest.TestCase):
    VALID_OUTPUT = (
        'elapsed_seconds=1.234567890 iterations=324000000 '
        'checksum=f36b0271f47e9160/5.3222442101923848\n'
    )

    def test_repeat_flag_defaults_to_six_and_preserves_unittest_arguments(self):
        repeats, argv = performance._parse_cli(['benchmark', '-q'])
        self.assertEqual(repeats, 6)
        self.assertEqual(argv, ['benchmark', '-q'])

        repeats, argv = performance._parse_cli([
            'benchmark',
            '--repeats',
            '11',
            '-q',
        ])
        self.assertEqual(repeats, 11)
        self.assertEqual(argv, ['benchmark', '-q'])

        with self.assertRaises(SystemExit):
            performance._parse_cli(['benchmark', '--repeats', '0'])

    def test_parses_and_validates_reference_output(self):
        self.assertEqual(
            performance._parse_cpu_reference_output(self.VALID_OUTPUT),
            1.23456789,
        )
        with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
            performance._parse_cpu_reference_output(
                self.VALID_OUTPUT.replace('f36b0271', '00000000')
            )
        with self.assertRaisesRegex(RuntimeError, 'Unexpected'):
            performance._parse_cpu_reference_output('elapsed_seconds=1.0\n')

    def test_scale_requires_exactly_six_valid_runs(self):
        samples = [1.0 + index * 0.1 for index in range(6)]
        self.assertEqual(performance._cpu_reference_scale(samples), 1.25)
        with self.assertRaisesRegex(RuntimeError, 'exactly 6'):
            performance._cpu_reference_scale(samples[:5])
        with self.assertRaisesRegex(RuntimeError, 'Invalid'):
            performance._cpu_reference_scale(samples[:-1] + [0.0])

    def test_runtime_normalization(self):
        self.assertAlmostEqual(
            performance._normalize_runtime(17.871, 1.156614515),
            15.451129,
            places=5,
        )
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            performance._normalize_runtime(1.0, 0.0)

    def test_reference_scales_interpolate_at_round_midpoints(self):
        self.assertEqual(
            performance._interpolated_cpu_reference_scales(1.0, 2.0, 4),
            [1.125, 1.375, 1.625, 1.875],
        )
        with self.assertRaisesRegex(ValueError, 'count must be positive'):
            performance._interpolated_cpu_reference_scales(1.0, 2.0, 0)

    def test_tight_initialized_example_limits_and_named_exception(self):
        self.assertEqual(performance.ALLOWED_WALL_RELATIVE_REGRESSION, 0.12)
        self.assertEqual(performance.ALLOWED_EXAMPLE_RELATIVE_REGRESSION, 0.12)
        self.assertEqual(performance.MIN_EXAMPLE_SLOWDOWN_SECONDS, 0.03)
        self.assertAlmostEqual(
            performance._example_runtime_limit('ordinary.pfd', 1.0),
            1.12,
        )
        self.assertAlmostEqual(
            performance._example_runtime_limit('fast.pfd', 0.01),
            0.04,
        )
        self.assertAlmostEqual(
            performance._example_runtime_limit(
                'ethanol_water_inclined_pipe_unifac.pfd',
                0.468,
            ),
            0.6318,
        )
        self.assertAlmostEqual(
            performance._example_runtime_limit(
                'trace_organic_water_stripping_isothermal_unifnist.pfd',
                0.501,
            ),
            0.62625,
        )
        self.assertAlmostEqual(
            performance._example_runtime_limit(
                'methane_claude_liquefaction_pr.pfd',
                1.151,
            ),
            1.32365,
        )
        self.assertAlmostEqual(
            performance._example_runtime_limit(
                'ethylene_oxide.pfd',
                0.227,
            ),
            0.26105,
        )

    @patch('simulator.Simulator.from_file')
    def test_example_timing_starts_after_explicit_initialization(self, from_file):
        events = []
        simulator = from_file.return_value
        simulator.initialize.side_effect = lambda: events.append('initialize')
        simulator.run.side_effect = lambda: (
            events.append('run')
            or SimpleNamespace(
                converged=True,
                errors=[],
                mass_balance_error=0.0,
                energy_balance_error=0.0,
            )
        )

        result = performance._run_example('/tmp/ready-example.pfd')

        self.assertEqual(events, ['initialize', 'run'])
        self.assertTrue(result['ok'])
        simulator.initialize.assert_called_once_with()
        simulator.run.assert_called_once_with()

    @patch('tests.performance_all_examples.subprocess.run')
    def test_measurement_compiles_once_and_runs_exactly_six_times(self, run):
        compile_result = subprocess.CompletedProcess([], 0, '', '')
        reference_result = subprocess.CompletedProcess(
            [], 0, self.VALID_OUTPUT, ''
        )
        run.side_effect = [compile_result] + [
            reference_result for _ in range(performance.CPU_REFERENCE_RUNS)
        ]

        scale, samples = performance._measure_cpu_reference()

        self.assertEqual(run.call_count, 7)
        self.assertEqual(len(samples), 6)
        self.assertEqual(scale, 1.23456789)
        compile_command = run.call_args_list[0].args[0]
        self.assertEqual(compile_command[1:3], ['-O3', '-march=native'])
        self.assertEqual(
            compile_command[-1],
            performance.CPU_REFERENCE_SOURCE,
        )


if __name__ == '__main__':
    unittest.main()
