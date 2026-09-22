"""
Opt-in all-examples performance regression test.

This file is intentionally not named test_*.py, so bare pytest and
`python -m unittest discover` skip it by default.

Run explicitly with:
    python -m unittest tests.performance_all_examples -q

Override the default six benchmark rounds with:
    python -m tests.performance_all_examples --repeats 11 -q

The suite brackets PFDsim timing with six runs of its fixed single-core C
reference before and six after. Each benchmark round is normalized with a
linearly interpolated reference scale, converting host seconds to 2026-08-08
reference-CPU seconds before comparison with the stored baselines.
"""

from __future__ import annotations

import concurrent.futures
import argparse
import glob
import multiprocessing
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from typing import Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DISABLED_EXAMPLES = {
    'lactic_acid_dehydration_pbr.pfd': (
        'Pending a dedicated normalized performance benchmark and baseline.'
    ),
    'saponification_cstr.pfd': (
        'Requires the deferred aqueous-electrolyte and salt-speciation model.'
    ),
}


MAX_WORKERS = 4
DEFAULT_REPEATS = 6
REPEATS = DEFAULT_REPEATS
CPU_REFERENCE_RUNS = 6
CPU_REFERENCE_CHECKSUM = 'f36b0271f47e9160/5.3222442101923848'
CPU_REFERENCE_SOURCE = os.path.join(
    os.path.dirname(__file__),
    'cpu_one_second_reference.c',
)
CPU_REFERENCE_PATTERN = re.compile(
    r'^elapsed_seconds=(?P<elapsed>[0-9]+(?:\.[0-9]+)?) '
    r'iterations=324000000 checksum=(?P<checksum>\S+)$'
)
ALLOWED_WALL_RELATIVE_REGRESSION = 0.12
ALLOWED_EXAMPLE_RELATIVE_REGRESSION = 0.15
PER_EXAMPLE_RELATIVE_REGRESSION = {
    'ethanol_water_inclined_pipe_unifac.pfd': 0.35,
    'trace_organic_water_stripping_isothermal_unifnist.pfd': 0.25,
    'vinegar_concentration_uniquac_vdm.pfd': 0.15,
    'methane_claude_liquefaction_pr.pfd': 0.15,
    'ethylene_oxide.pfd': 0.15,
}
# Per-example solve times are collected after explicit initialization inside a
# 4-worker pool. Keep the aggregate wall clock strict, and give individual rows
# enough absolute slack to avoid flagging scheduling noise as a regression.
MIN_EXAMPLE_SLOWDOWN_SECONDS = 0.03

BASELINE_WALL_SECONDS = 13.0
BASELINE_EXAMPLE_SECONDS = {
    '3methylpyridine_ether_extraction_recycle.pfd': 1.445,
    'adaptive_spinodal_water_toluene.pfd': 0.016,
    'acrylic_acid_rigorous_extraction.pfd': 0.288,
    'air_3a_molecular_sieve_drying.pfd': 0.007,
    'ammonia_oxidation.pfd': 0.037,
    'ammonia_synthesis.pfd': 0.021,
    'benzene_toluene_20_stage_distillation_nrtl.pfd': 0.086,
    'biosteam_mesh_hydrocarbon_distillation.pfd': 1.796,
    'butanol_water_lle.pfd': 0.007,
    'cryogenic_air_separation_rks_bm.pfd': 0.322,
    'cstr_pfr_comparison.pfd': 0.148,
    'dcm_3a_molecular_sieve_drying.pfd': 0.008,
    'ethanol_3a_molecular_sieve_drying.pfd': 0.006,
    'ethanol_benzene_azeotropic_distillation_rigorous.pfd': 0.410,
    'ethanol_distillation_rigorous.pfd': 0.089,
    'ethanol_ether_partial_condensation_absorption.pfd': 1.760,
    'ethanol_pressure_swing_recycle_wasteful.pfd': 6.941,
    'ethanol_water_mhv2.pfd': 0.027,
    'ethanol_water_inclined_pipe_unifac.pfd': 0.468,
    'ethylene_ethane_isoparaffin_absorption.pfd': 2.883,
    'ethylene_oxide.pfd': 0.227,
    'ethylene_oxide_simple.pfd': 0.354,
    'equilibrium_methanol_synthesis_recycle.pfd': 0.937,
    'global_vlle_water_methanol_benzene.pfd': 0.019,
    'haber_bosch_full.pfd': 4.504,
    'jacketed_cstr_ignition_extinction.pfd': 0.198,
    'methane_claude_liquefaction_pr.pfd': 1.151,
    'methanol_decomposition_pfr.pfd': 0.029,
    'methanol_diethyl_ether_5bar_nrtl_rk.pfd': 0.170,
    'methanol_ethanol_light_gas_cleanup_compact.pfd': 1.461,
    'methanol_synthesis.pfd': 0.366,
    'methanol_synthesis_psrk.pfd': 0.576,
    'mixed_acid_dehydration_uniquac_vdm.pfd': 0.744,
    'permanent_solid_global_vlle_flash.pfd': 0.019,
    'permanent_solid_slurry_operations.pfd': 0.003,
    'pyridine_ether_extraction.pfd': 0.256,
    'rk_thermodynamics_pfr.pfd': 0.310,
    'simple_flash.pfd': 0.005,
    'simple_rankine_cycle_steam.pfd': 0.001,
    'trace_organic_water_stripping_isothermal_unifnist.pfd': 0.501,
    'unifac_flash.pfd': 0.006,
    'vinegar_concentration_uniquac_vdm.pfd': 0.342,

    'cake_filtration_washing.pfd': 0.079,
    'equilibrium_warm_melt_washing.pfd': 0.355,
    'ethyl_acetate_batch_synthesis.pfd': 0.079,
    'isopropanol_diisopropyl_ether_distillation_nrtl_estimated.pfd': 0.098,
    'isopropanol_water_nrtl_uniquac_comparison.pfd': 0.147,
}


def _parse_cpu_reference_output(output: str) -> float:
    """Validate one reference run and return its elapsed wall time."""
    match = CPU_REFERENCE_PATTERN.fullmatch(output.strip())
    if match is None:
        raise RuntimeError(f'Unexpected CPU reference output: {output!r}')
    checksum = match.group('checksum')
    if checksum != CPU_REFERENCE_CHECKSUM:
        raise RuntimeError(
            'CPU reference checksum mismatch: '
            f'{checksum!r} != {CPU_REFERENCE_CHECKSUM!r}'
        )
    elapsed = float(match.group('elapsed'))
    if not 0.0 < elapsed < 60.0:
        raise RuntimeError(
            f'CPU reference returned invalid elapsed time {elapsed!r}'
        )
    return elapsed


def _cpu_reference_scale(samples: list[float]) -> float:
    """Return host seconds per historical reference-CPU second."""
    if len(samples) != CPU_REFERENCE_RUNS:
        raise RuntimeError(
            f'CPU reference requires exactly {CPU_REFERENCE_RUNS} runs; '
            f'got {len(samples)}'
        )
    if any(not 0.0 < value < 60.0 for value in samples):
        raise RuntimeError(f'Invalid CPU reference samples: {samples!r}')
    return statistics.median(samples)


def _measure_cpu_reference() -> tuple[float, list[float]]:
    """Compile and run one six-sample single-core reference bracket."""
    compiler = os.environ.get('CC', 'cc')
    with tempfile.TemporaryDirectory(prefix='pfdsim-cpu-reference-') as temp:
        executable = os.path.join(temp, 'cpu_one_second_reference')
        subprocess.run(
            [
                compiler,
                '-O3',
                '-march=native',
                '-o',
                executable,
                CPU_REFERENCE_SOURCE,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        samples = []
        for _ in range(CPU_REFERENCE_RUNS):
            completed = subprocess.run(
                [executable],
                check=True,
                capture_output=True,
                text=True,
            )
            samples.append(_parse_cpu_reference_output(completed.stdout))
    return _cpu_reference_scale(samples), samples


def _normalize_runtime(raw_seconds: float, cpu_reference_scale: float) -> float:
    """Convert host wall seconds to 2026-08-08 reference-CPU seconds."""
    if not 0.0 < cpu_reference_scale < 60.0:
        raise ValueError(
            f'Invalid CPU reference scale {cpu_reference_scale!r}'
        )
    return raw_seconds / cpu_reference_scale


def _interpolated_cpu_reference_scales(
    start_scale: float,
    end_scale: float,
    count: int,
) -> list[float]:
    """Return linearly interpolated scales at each timed round midpoint."""
    if count <= 0:
        raise ValueError(f'CPU reference interpolation count must be positive; got {count}')
    if not 0.0 < start_scale < 60.0 or not 0.0 < end_scale < 60.0:
        raise ValueError(
            f'Invalid bracketed CPU reference scales: {start_scale!r}, {end_scale!r}'
        )
    return [
        start_scale
        + (end_scale - start_scale) * ((index + 0.5) / count)
        for index in range(count)
    ]


def _rounded_samples(values: list[float]) -> list[float]:
    return [round(value, 3) for value in values]


def _parse_cli(argv: list[str]) -> tuple[int, list[str]]:
    """Extract performance-only flags and preserve unittest arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--repeats', type=int, default=DEFAULT_REPEATS)
    namespace, remaining = parser.parse_known_args(argv[1:])
    if namespace.repeats <= 0:
        parser.error('--repeats must be a positive integer')
    return namespace.repeats, [argv[0], *remaining]


def _example_runtime_limit(name: str, baseline: float) -> float:
    """Return the solve-time limit for one initialized example."""
    relative = PER_EXAMPLE_RELATIVE_REGRESSION.get(
        name,
        ALLOWED_EXAMPLE_RELATIVE_REGRESSION,
    )
    return max(
        baseline * (1.0 + relative),
        baseline + MIN_EXAMPLE_SLOWDOWN_SECONDS,
    )


def _trim_one_outlier(values: list[float]) -> tuple[list[float], Optional[float]]:
    """Remove at most one statistically obvious timing outlier."""
    values = list(values)
    if len(values) < 5:
        return values, None

    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations)

    outlier_index = None
    if mad > 1e-12:
        z_scores = [
            0.6745 * (value - median) / mad
            for value in values
        ]
        index, z_score = max(
            enumerate(z_scores),
            key=lambda item: abs(item[1]),
        )
        if abs(z_score) > 3.5:
            outlier_index = index
    else:
        mean = statistics.mean(values)
        stdev = statistics.pstdev(values)
        if stdev > 1e-12:
            index, value = max(
                enumerate(values),
                key=lambda item: abs(item[1] - mean),
            )
            z_score = (value - mean) / stdev
            if abs(z_score) > 3.0:
                outlier_index = index

    if outlier_index is None:
        return values, None

    outlier = values[outlier_index]
    trimmed = values[:outlier_index] + values[outlier_index + 1:]
    return trimmed, outlier


def _run_example(path: str) -> dict:
    from simulator import Simulator

    name = os.path.basename(path)
    start = time.perf_counter()
    try:
        simulator = Simulator.from_file(path)
        simulator.initialize()
        # Per-example timing measures the ready persistent-process solve.
        # The enclosing all-examples wall timer still includes parsing and
        # initialization, preserving the end-to-end cold-start regression.
        start = time.perf_counter()
        result = simulator.run()
        elapsed = time.perf_counter() - start
        ok = (
            result.converged
            and not result.errors
            and result.mass_balance_error < 1e-2
            and result.energy_balance_error < 6e-2
        )
        return {
            'name': name,
            'elapsed': elapsed,
            'ok': ok,
            'converged': result.converged,
            'errors': result.errors,
            'mass_balance_error': result.mass_balance_error,
            'energy_balance_error': result.energy_balance_error,
        }
    except Exception as exc:
        return {
            'name': name,
            'elapsed': time.perf_counter() - start,
            'ok': False,
            'converged': False,
            'errors': [repr(exc), traceback.format_exc()],
            'mass_balance_error': None,
            'energy_balance_error': None,
        }


def _run_all_examples_once() -> tuple[float, list[dict]]:
    paths = [
        path for path in sorted(glob.glob(os.path.join(ROOT, 'examples', '*.pfd')))
        if os.path.basename(path) not in DISABLED_EXAMPLES
    ]
    context = multiprocessing.get_context('fork')
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=context,
    ) as pool:
        results = list(pool.map(_run_example, paths))
    return time.perf_counter() - start, results


class AllExamplesPerformanceTests(unittest.TestCase):
    maxDiff = None

    def test_all_examples_average_runtime_stays_near_baseline(self):
        cpu_reference_start, cpu_reference_start_samples = _measure_cpu_reference()
        print(
            'CPU reference start: '
            f'median={cpu_reference_start:.6f}s '
            f'samples={_rounded_samples(cpu_reference_start_samples)}',
            flush=True,
        )
        raw_walls = []
        issues = []
        raw_new_example_timings = {}
        raw_example_timings = {
            name: []
            for name in BASELINE_EXAMPLE_SECONDS
        }

        for _ in range(REPEATS):
            raw_wall, results = _run_all_examples_once()
            raw_walls.append(raw_wall)
            for result in results:
                if not result['ok']:
                    issues.append(
                        f"{result['name']} did not converge/close balances: "
                        f"converged={result['converged']} "
                        f"mass={result['mass_balance_error']} "
                        f"energy={result['energy_balance_error']} "
                        f"errors={result['errors']}"
                    )
                if result['name'] in raw_example_timings:
                    raw_example_timings[result['name']].append(
                        result['elapsed']
                    )
                else:
                    raw_new_example_timings.setdefault(
                        result['name'], []
                    ).append(result['elapsed'])

        cpu_reference_end, cpu_reference_end_samples = _measure_cpu_reference()
        round_scales = _interpolated_cpu_reference_scales(
            cpu_reference_start,
            cpu_reference_end,
            REPEATS,
        )
        cpu_reference_scale = statistics.mean(round_scales)
        print(
            'CPU reference end: '
            f'median={cpu_reference_end:.6f}s '
            f'samples={_rounded_samples(cpu_reference_end_samples)} '
            f'interpolated_round_scales={_rounded_samples(round_scales)}',
            flush=True,
        )
        walls = [
            _normalize_runtime(value, round_scales[index])
            for index, value in enumerate(raw_walls)
        ]
        example_timings = {
            name: [
                _normalize_runtime(value, round_scales[index])
                for index, value in enumerate(values)
            ]
            for name, values in raw_example_timings.items()
        }
        new_example_timings = {
            name: [
                _normalize_runtime(value, round_scales[index])
                for index, value in enumerate(values)
            ]
            for name, values in raw_new_example_timings.items()
        }

        missing = [
            name for name, values in example_timings.items()
            if not values
        ]
        for name, values in sorted(new_example_timings.items()):
            issues.append(
                f"New example {name} needs a performance baseline; "
                f"normalized_samples={_rounded_samples(values)} "
                f"raw_samples={_rounded_samples(raw_new_example_timings[name])}"
            )
        if missing:
            issues.append(
                "Baselined examples did not run: "
                f"{', '.join(sorted(missing))}"
            )

        trimmed_walls, wall_outlier = _trim_one_outlier(walls)
        wall_average = statistics.mean(trimmed_walls)
        trimmed_raw_walls, raw_wall_outlier = _trim_one_outlier(raw_walls)
        raw_wall_average = statistics.mean(trimmed_raw_walls)
        print(
            'All examples wall: '
            f'normalized_average={wall_average:.6f}s '
            f'raw_average={raw_wall_average:.6f}s '
            f'cpu_reference_average_scale={cpu_reference_scale:.6f}s '
            f'normalized_samples={_rounded_samples(walls)} '
            f'raw_samples={_rounded_samples(raw_walls)} '
            f'normalized_outlier={
                None if wall_outlier is None else round(wall_outlier, 3)
            } '
            f'raw_outlier={
                None if raw_wall_outlier is None
                else round(raw_wall_outlier, 3)
            }',
            flush=True,
        )
        wall_limit = BASELINE_WALL_SECONDS * (
            1.0 + ALLOWED_WALL_RELATIVE_REGRESSION
        )
        if wall_average > wall_limit:
            issues.append(
                f"all examples wall average regressed: {wall_average:.3f}s "
                f"> {BASELINE_WALL_SECONDS:.3f}s baseline + "
                f"{100.0 * ALLOWED_WALL_RELATIVE_REGRESSION:.0f}%; "
                f"normalized_samples={_rounded_samples(walls)} "
                f"raw_samples={_rounded_samples(raw_walls)} "
                f"cpu_reference_average_scale={cpu_reference_scale:.6f}s "
                f"trimmed_outlier={None if wall_outlier is None else round(wall_outlier, 3)}"
            )

        for name, baseline in BASELINE_EXAMPLE_SECONDS.items():
            if not example_timings[name]:
                continue
            trimmed_timings, outlier = _trim_one_outlier(example_timings[name])
            average = statistics.mean(trimmed_timings)
            trimmed_raw_timings, raw_outlier = _trim_one_outlier(
                raw_example_timings[name]
            )
            raw_average = statistics.mean(trimmed_raw_timings)
            print(
                f'Example timing: {name} '
                f'normalized_average={average:.6f}s '
                f'raw_average={raw_average:.6f}s '
                f'normalized_samples={_rounded_samples(example_timings[name])} '
                f'raw_samples={_rounded_samples(raw_example_timings[name])} '
                f'normalized_outlier={
                    None if outlier is None else round(outlier, 3)
                } '
                f'raw_outlier={
                    None if raw_outlier is None else round(raw_outlier, 3)
                }',
                flush=True,
            )
            limit = _example_runtime_limit(name, baseline)
            if average > limit:
                issues.append(
                    f"{name}: avg={average:.3f}s limit={limit:.3f}s "
                    f"baseline={baseline:.3f}s "
                    f"normalized_samples={_rounded_samples(example_timings[name])} "
                    f"raw_samples={_rounded_samples(raw_example_timings[name])} "
                    f"trimmed_outlier={None if outlier is None else round(outlier, 3)}"
                )

        if issues:
            self.fail(
                "Performance benchmark issues:\n"
                + "\n".join(f"- {issue}" for issue in issues)
            )


if __name__ == '__main__':
    REPEATS, unittest_argv = _parse_cli(sys.argv)
    unittest.main(argv=unittest_argv)
