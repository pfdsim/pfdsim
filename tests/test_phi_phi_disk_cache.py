"""Disk-cache lifecycle of model-bound EOS entry points in fresh processes."""

import json
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys

import pytest


def cache_worker(phase):
    # Imports happen after the parent establishes the isolated cache and zero
    # dispatcher-retention environment. Never read serialized cache payloads.
    import gc
    import math
    import time
    import warnings
    import numpy as np
    from chemical_properties import ChemicalDatabase
    from compiled_phi_phi import kernels_for
    from numba import typeof
    from numba.core.dispatcher import _MemoMixin
    from numba.core.errors import NumbaWarning, NumbaIRAssumptionWarning
    import phi_phi
    from phi_phi_numerics import compile_kernels
    from thermodynamics import create_thermodynamics
    warnings.filterwarnings('error', message='Cannot cache compiled function.*', category=NumbaWarning)
    warnings.filterwarnings('error', category=NumbaIRAssumptionWarning)

    def fortran(value):
        if isinstance(value, tuple):
            return tuple(fortran(v) for v in value)
        return np.asfortranarray(value) if isinstance(value, np.ndarray) and value.ndim == 2 else value

    reports = []
    for method in ('PR', 'PSRK', 'RKSMHV2'):
        print('PREPARE', phase, method, flush=True)
        model = create_thermodynamics(['CH4', 'C2H6'], method, ChemicalDatabase(enable_online=False))
        state, data = phi_phi._native_provider(model, 300., record_warnings=False)
        kernels = kernels_for(state)
        before_hits = sum(sum(f._cache_hits.values()) for f in kernels)
        before_misses = sum(sum(f._cache_misses.values()) for f in kernels)
        start = time.perf_counter()
        model.initialize()
        _MemoMixin._recent.clear()
        gc.collect()
        extra = fortran(data)
        grew = False
        if phase in ('grow', 'reload-grown'):
            grew = typeof(extra) != typeof(data)
            assert grew, 'Matrix-layout change must exercise a new specialization'
            compile_kernels(state, extra, len(model.components))
            _MemoMixin._recent.clear()
            gc.collect()
        hits = sum(sum(f._cache_hits.values()) for f in kernels)-before_hits
        misses = sum(sum(f._cache_misses.values()) for f in kernels)-before_misses
        # Five independent numerical kernels per model. Signature growth adds
        # an alternate matrix layout, whose persistence is verified next run.
        expected_hits = {'seed': 0, 'reload': 5, 'grow': 5, 'reload-grown': 10}[phase]
        expected_misses = {'seed': 5, 'reload': 0, 'grow': 5, 'reload-grown': 0}[phase]
        assert hits == expected_hits, (method, phase, 'cache hits', hits)
        assert misses == expected_misses, (method, phase, 'cache misses', misses)
        for dispatcher in kernels:
            assert all('Dispatcher' not in str(signature) for signature in dispatcher.signatures)
        beta, x, y = model.flash_TP({'CH4': .664, 'C2H6': .336}, 245.65, 50.72)
        assert 0 < beta < 1
        pl = model.fugacity_coefficients(245.65, 50.72, x, 'liquid')
        pv = model.fugacity_coefficients(245.65, 50.72, y, 'vapor')
        residual = max(abs(math.log(x[c]*pl[c]/(y[c]*pv[c]))) for c in x)
        assert residual < 1e-8
        for c, z in {'CH4': .664, 'C2H6': .336}.items():
            assert abs((1-beta)*x[c]+beta*y[c]-z) < 1e-9
        pressure = model.bubble_point_P({'CH4': .5, 'C2H6': .5}, 200.)
        assert pressure > 0
        trace = model.calculate_state_PQ(10., 1e-8, 1., {'CH4': .5, 'C2H6': .5}, include=())
        for c in trace.composition:
            assert abs((1.-1e-8)*trace.x[c]+1e-8*trace.y[c]-.5) < 1e-9
        reports.append({'method': method, 'phase': phase, 'cache_hits': hits,
                        'cache_misses': misses, 'new_layout': grew,
                        'seconds': time.perf_counter()-start, 'log_fugacity_residual': residual})
        print('MODEL_RESULT', json.dumps(reports[-1]), flush=True)
    print(json.dumps(reports), flush=True)


def parallel_reloads(cache, output):
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, 'NUMBA_CACHE_DIR': str(cache), 'NUMBA_FUNCTION_CACHE_SIZE': '0',
                   'PYTHONPATH': str(root)}
    reports = []
    # The original failure appeared in the four-worker performance run. All
    # workers now read the grown cache concurrently with no retention crutch.
    with ExitStack() as stack:
        workers = []
        for index in range(4):
            path = output/f'parallel-reload-{index}.log'
            stream = stack.enter_context(path.open('x'))
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'reload-grown'],
                                       cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            workers.append((process, path))
        statuses = [(process.wait(), path) for process, path in workers]
    for status, path in statuses:
        assert status == 0, f'Concurrent cache reload failed; inspect {path}'
        report = json.loads(path.read_text().splitlines()[-1])
        reports.extend(report)
        print(path.stem, json.dumps(report), flush=True)
    return reports


def test_fixed_callback_cache_reloads_and_grows_without_dispatcher_retention(tmp_path):
    pytest.importorskip('numba')
    cache = tmp_path/'numba-cache'
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, 'NUMBA_CACHE_DIR': str(cache), 'NUMBA_FUNCTION_CACHE_SIZE': '0',
                   'PYTHONPATH': str(root)}
    reports = []
    for phase in ('seed', 'reload', 'grow', 'reload-grown'):
        path = tmp_path/f'{phase}.log'
        with path.open('x') as stream:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), phase],
                                    cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                    check=False)
        assert result.returncode == 0, f'{phase} failed; inspect {path}'
        report = json.loads(path.read_text().splitlines()[-1])
        reports.extend(report)
        print(phase, json.dumps(report), flush=True)
    reports.extend(parallel_reloads(cache, tmp_path))
    with (tmp_path/'summary.json').open('x') as stream:
        json.dump(reports, stream, indent=2)


if __name__ == '__main__':
    if sys.argv[1] == 'parallel':
        parallel_reloads(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        cache_worker(sys.argv[1])
