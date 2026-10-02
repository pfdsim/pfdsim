"""
Default parallel test runner for pfdsim.

This module is used by project-local unittest/pytest hooks. It is not a test
module itself.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TESTS_DIR = os.path.join(ROOT, 'tests')
WORKER_COUNT = 6
RESULT_SENTINEL = '__PFDSIM_UNITTEST_RESULT__'

# Residual average runtimes per unittest method, derived from the six-worker
# JUnit run on 2026-08-07. Long indivisible methods are listed separately so
# longest-processing-time-first scheduling does not hide them in an average.
MODULE_SECONDS_PER_TEST = {
    'tests.test_unit_operations': 4.915,
    'tests.test_examples': 1.457,
    'tests.test_thermodynamic_methods': 0.713,
    'tests.test_compiled_backends': 1.273,
    'tests.test_pfd_component_properties': 0.923,
    'tests.test_vapor_pressure_adapter': 0.218,
    'tests.test_property_resolution_system': 0.136,
    'tests.test_pipe_unit': 1.054,
    'tests.test_property_lookup': 0.396,
    'tests.test_thermo_model_fixes': 0.642,
    'tests.test_pfd_integration': 1.727,
    'tests.test_enthalpy_resolution': 0.435,
    'tests.test_vapor_pressure_resolution': 0.900,
    'tests.test_perry_property_lookup': 0.665,
    'tests.test_henry_thermodynamics': 0.415,
    'tests.test_quality_report_context': 0.508,
    'tests.test_pressure_standards': 0.550,
    'tests.test_liquid_mixture_viscosity': 0.161,
    'tests.test_surface_tension_resolution': 0.128,
    'tests.test_interfacial_properties': 0.166,
    'tests.test_warning_reporting': 0.168,
    'tests.test_perry_properties': 0.074,
    'tests.test_vapor_pressure_canonical': 0.021,
    'tests.test_vapor_dimerization': 0.030,
    'tests.test_recycles': 0.052,
    'tests.test_nannoolal_method': 0.007,
    'tests.test_hsu_method': 0.010,
    'tests.test_entropy_support': 0.029,
    'tests.test_transport_primitives': 0.003,
    'tests.test_steam_unit_operations': 0.005,
    'tests.test_steam_thermodynamics': 0.002,
    'tests.test_compound_identity': 0.001,
    'tests.test_unit_conversions': 0.001,
}

TEST_SECONDS = {
    'tests.test_examples.ExampleSimulationTests.'
    'test_all_examples_converge_and_close_balances': 101.047,
    'tests.test_unit_operations.UnitOperationSmokeTests.'
    'test_rigorous_distillation_ethanol_benzene_azeotropic_case': 45.003,
    'tests.test_thermodynamic_methods.ThermodynamicMethodTests.'
    'test_added_property_methods_construct_and_return_directional_k_values': 31.762,
    'tests.test_unit_operations.UnitOperationSmokeTests.'
    'test_rigorous_distillation_local_jacobian_matches_thermo_families': 21.879,
    'tests.test_unit_operations.UnitOperationSmokeTests.'
    'test_flash3_vapor_fraction_duty_recovers_three_phase_branch': 19.603,
    'tests.test_unit_operations.UnitOperationSmokeTests.'
    'test_rigorous_distillation_initializer_routing': 14.423,
    'tests.test_unit_operations.UnitOperationSmokeTests.'
    'test_rigorous_distillation_methanol_trace_contaminants': 13.832,
    'tests.test_unit_operations.UnitOperationSmokeTests.'
    'test_trace_impurities_keep_effectively_binary_balance_rows': 11.240,
    'tests.test_unit_operations.UnitOperationSmokeTests.'
    'test_coarse_rigorous_uses_legacy_cmo_estimate_seed': 10.893,
    'tests.test_unit_operations.UnitOperationSmokeTests.'
    'test_rigorous_absorber_hot_wet_feed_uses_selective_henry_components': 10.038,
    'tests.test_unit_operations.UnitOperationSmokeTests.'
    'test_ethanol_water_azeotrope_and_benzene_entrainer': 10.002,
    'tests.test_unit_operations.UnitOperationSmokeTests.'
    'test_rigorous_distillation_solves_multicomponent_mesh_balances': 9.879,
    'tests.test_unit_operations.UnitOperationSmokeTests.'
    'test_rigorous_stripper_trace_unifnist_case_reports_stripping': 9.524,
    'tests.test_unit_operations.UnitOperationSmokeTests.'
    'test_mccabe_thiele_uses_latent_heat_curved_operating_lines': 9.507,
    'tests.test_examples.ExampleSimulationTests.'
    'test_pressure_swing_recycle_converges_with_two_phase_second_column_feed': 9.192,
    'tests.test_unit_operations.UnitOperationSmokeTests.'
    'test_rigorous_distillation_top_decanter_selects_reflux_phase_and_purge': 7.665,
}


@dataclass
class TestGroup:
    name: str
    tests: list[str]
    estimated_seconds: float = 0.0


def _test_modules() -> list[str]:
    modules = []
    for path in sorted(glob.glob(os.path.join(TESTS_DIR, 'test*.py'))):
        name = os.path.splitext(os.path.basename(path))[0]
        modules.append(f'tests.{name}')
    return modules


def _flatten_suite(suite) -> list:
    tests = []
    for item in suite:
        if hasattr(item, '__iter__') and not hasattr(item, 'id'):
            tests.extend(_flatten_suite(item))
        else:
            tests.append(item)
    return tests


def _all_test_ids() -> list[str]:
    import unittest

    loader = unittest.TestLoader()
    tests = []
    for module in _test_modules():
        suite = loader.loadTestsFromName(module)
        tests.extend(test.id() for test in _flatten_suite(suite))
    return tests


def _test_module(test_id: str) -> str:
    return '.'.join(test_id.split('.')[:2])


def _estimated_seconds(test_id: str) -> float:
    return TEST_SECONDS.get(
        test_id,
        MODULE_SECONDS_PER_TEST.get(_test_module(test_id), 0.05),
    )


def default_groups() -> list[TestGroup]:
    groups = [
        TestGroup(f'worker {index + 1}', [])
        for index in range(WORKER_COUNT)
    ]
    tests = sorted(
        _all_test_ids(),
        key=lambda test_id: (-_estimated_seconds(test_id), test_id),
    )
    for test_id in tests:
        group = min(
            groups,
            key=lambda item: (item.estimated_seconds, item.name),
        )
        seconds = _estimated_seconds(test_id)
        group.tests.append(test_id)
        group.estimated_seconds += seconds
    for group in groups:
        # Keep methods from the same module/class adjacent so unittest can
        # reuse module and class fixtures within each worker.
        group.tests.sort()
    return groups


def _command_for_group(group: TestGroup) -> list[str]:
    return [
        sys.executable,
        '-m',
        'tests.default_parallel',
        '--worker',
        *group.tests,
    ]


def _test_label(test) -> str:
    try:
        return test.id()
    except Exception:
        return str(test)


def _issue_payload(test, traceback_text: str) -> dict:
    lines = [
        line.strip()
        for line in str(traceback_text).splitlines()
        if line.strip()
    ]
    return {
        'test': _test_label(test),
        'message': lines[-1] if lines else 'no traceback message',
    }


def _worker_payload(result, elapsed: float) -> dict:
    return {
        'successful': result.wasSuccessful(),
        'tests_run': result.testsRun,
        'elapsed_seconds': elapsed,
        'failures': [
            _issue_payload(test, traceback_text)
            for test, traceback_text in result.failures
        ],
        'errors': [
            _issue_payload(test, traceback_text)
            for test, traceback_text in result.errors
        ],
        'skipped': [
            {'test': _test_label(test), 'reason': reason}
            for test, reason in result.skipped
        ],
        'expected_failures': [
            _issue_payload(test, traceback_text)
            for test, traceback_text in result.expectedFailures
        ],
        'unexpected_successes': [
            {'test': _test_label(test), 'message': 'unexpected success'}
            for test in result.unexpectedSuccesses
        ],
    }


def run_worker(test_ids: list[str]) -> int:
    import unittest
    from .cache_isolation import isolated_runtime_caches

    with isolated_runtime_caches():
        suite = unittest.TestLoader().loadTestsFromNames(test_ids)
        started = time.perf_counter()
        result = unittest.TextTestRunner(verbosity=1).run(suite)
    payload = _worker_payload(result, time.perf_counter() - started)
    print(
        RESULT_SENTINEL + json.dumps(payload, sort_keys=True),
        flush=True,
    )
    return 0 if result.wasSuccessful() else 1


def _split_worker_output(output: str) -> tuple[str, dict | None]:
    payload = None
    retained_lines = []
    for line in output.splitlines():
        if line.startswith(RESULT_SENTINEL):
            try:
                payload = json.loads(line[len(RESULT_SENTINEL):])
            except json.JSONDecodeError:
                retained_lines.append(line)
        else:
            retained_lines.append(line)
    return '\n'.join(retained_lines).rstrip(), payload


def _collect_process(item):
    group, process, started = item
    output, _ = process.communicate()
    output, payload = _split_worker_output(output)
    return (
        group,
        process.returncode,
        output,
        time.perf_counter() - started,
        payload,
    )


def run_parallel_default() -> int:
    groups = default_groups()
    print(
        "pfdsim default test runner: "
        f"spawning {WORKER_COUNT} timing-balanced workers",
        flush=True,
    )
    for group in groups:
        print(
            f"  {group.name}: {len(group.tests)} tests, "
            f"estimated {group.estimated_seconds:.1f}s",
            flush=True,
        )

    processes = []
    for group in groups:
        command = _command_for_group(group)
        env = dict(os.environ)
        env['PFDSIM_PARALLEL_DEFAULT_ACTIVE'] = '1'
        processes.append((
            group,
            subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            ),
            time.perf_counter(),
        ))

    failed_groups = []
    with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
        results = list(executor.map(_collect_process, processes))
    for group, returncode, output, elapsed, payload in results:
        status = (
            'PASS'
            if returncode == 0 and payload and payload.get('successful')
            else 'FAIL'
        )
        counts = ''
        if payload:
            counts = (
                f", ran {payload['tests_run']}, "
                f"{len(payload['failures'])} failures, "
                f"{len(payload['errors'])} errors, "
                f"{len(payload['skipped'])} skipped"
            )
        print(
            f"\n== {group.name}: {status} "
            f"({elapsed:.3f}s{counts}) ==",
            flush=True,
        )
        if output:
            print(output.rstrip(), flush=True)
        if status == 'FAIL':
            failed_groups.append((
                group,
                returncode,
                payload,
            ))

    tests_run = sum(
        payload['tests_run']
        for _group, _returncode, _output, _elapsed, payload in results
        if payload
    )
    skipped = sum(
        len(payload['skipped'])
        for _group, _returncode, _output, _elapsed, payload in results
        if payload
    )
    failures = [
        (group.name, issue)
        for group, _returncode, _output, _elapsed, payload in results
        if payload
        for issue in payload['failures']
    ]
    errors = [
        (group.name, issue)
        for group, _returncode, _output, _elapsed, payload in results
        if payload
        for issue in payload['errors']
    ]
    unexpected_successes = [
        (group.name, issue)
        for group, _returncode, _output, _elapsed, payload in results
        if payload
        for issue in payload['unexpected_successes']
    ]

    print(
        "\nParallel test summary: "
        f"{tests_run} methods run, {len(failures)} failure outcomes, "
        f"{len(errors)} errors, {skipped} skipped, "
        f"{len(unexpected_successes)} unexpected successes",
        flush=True,
    )
    for heading, issues in (
        ('Failures', failures),
        ('Errors', errors),
        ('Unexpected successes', unexpected_successes),
    ):
        if not issues:
            continue
        print(f"\n{heading}:", flush=True)
        for worker_name, issue in issues:
            print(
                f"  [{worker_name}] {issue['test']}: {issue['message']}",
                flush=True,
            )

    if failed_groups:
        print("\nFailed workers:", flush=True)
        for group, returncode, payload in failed_groups:
            detail = ''
            if payload is None:
                detail = ' (worker result payload missing)'
            print(
                f"  {group.name}: exit {returncode}{detail}",
                flush=True,
            )
        return 1
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        raise SystemExit(run_worker(sys.argv[2:]))
    raise SystemExit(run_parallel_default())
