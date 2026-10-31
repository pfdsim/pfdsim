#!/usr/bin/env python3
"""Measure incremental steam/PV correctness-fix costs against a source snapshot.

Example (the output directory must be new)::

    .venv/bin/python scripts/performance/probe_inverse_quality_flash_cost.py \
        --baseline-dir /tmp/pfdsim-before-fixes --output /tmp/pfdsim-fix-cost \
        2>&1 | tee /tmp/pfdsim-fix-cost.log

The snapshot needs thermodynamics_models/{base,eos,steam}.py,
phi_phi_numerics.py, and compiled_phi_phi.py. Before/after workers are separate
fresh processes; only the before worker restores snapshot code in memory.
Initialization and one warmup per case are excluded from request timings.
Every measured input is distinct to avoid timing repeated flash-result hits.
No randomness, repository rewrites, or full/performance-suite runs are used.
Incorrect historical endpoint states are explicitly reported in the accuracy
columns: their timings are not performance-equivalent correct solutions.
--fixed-pressure varies mixture quality at a fixed pressure to exercise reused
phase boundaries; the pure-fluid, TP and steam control cases stay unchanged.
--paired-steam and --paired-tp alternate implementations in one process to
distinguish small overheads from between-process timing noise.
"""

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def restore_method(snapshot, module, owner, name):
    path = snapshot / (module.__name__.replace('.', '/')+'.py')
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    setattr(getattr(module, owner), name, namespace[name])


def worker(args):
    start = time.perf_counter()
    if args.variant == 'before':
        for name in ('phi_phi_numerics', 'compiled_phi_phi'):
            spec = importlib.util.spec_from_file_location(name, args.baseline_dir/(name+'.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    from chemical_properties import ChemicalDatabase
    from thermodynamics import create_thermodynamics
    from thermodynamics_models import base, eos, steam
    if args.variant == 'before':
        for module, owner, names in (
            (base, 'IdealThermodynamics', ('flash_PV',)),
            (eos, 'PhiPhiThermodynamics', ('flash_TP',)),
            (steam, 'SteamThermodynamics',
             ('_state_from_inputs', '_refine_inverse_temperature', 'temperature_at_PH')),
        ):
            for name in names:
                restore_method(args.baseline_dir, module, owner, name)
    report = {'variant': args.variant, 'import_seconds': time.perf_counter()-start,
              'fixed_pressure': args.fixed_pressure, 'cases': []}

    def measure(name, model, inputs, evaluate, audit, coexistence=False, tolerance=1e-9):
        start = time.perf_counter()
        model.initialize()
        evaluate(inputs[0])
        preparation = time.perf_counter()-start
        durations, residuals, calls = [], [], []
        valid = 0
        original = model.flash_TP
        count = 0
        def counted(*values, **options):
            nonlocal count
            count += 1
            return original(*values, **options)
        model.flash_TP = counted
        for sample in range(args.samples):
            batch = inputs[1+sample*args.batch:1+(sample+1)*args.batch]
            count = 0
            start = time.perf_counter()
            results = [evaluate(value) for value in batch]
            durations.append((time.perf_counter()-start)/len(batch))
            calls.append(count/len(batch))
            for value, result in zip(batch, results):
                residual = audit(value, result)
                residuals.append(residual)
                physical = residual <= tolerance
                if coexistence:
                    try:
                        model._check_diagram_vle(result.x, result.y, result.T, result.P,
                                                 fugacity_tolerance=1e-7)
                    except ValueError:
                        physical = False
                valid += physical
        model.flash_TP = original
        row = {'name': name, 'preparation_seconds': preparation,
               'median_ms': 1000*statistics.median(durations),
               'min_ms': 1000*min(durations), 'max_ms': 1000*max(durations),
               'TP_calls_per_request': statistics.median(calls),
               'max_residual': max(residuals), 'requests': len(residuals),
               'valid_requests': valid}
        if args.variant == 'after' and valid != len(residuals):
            raise RuntimeError(f'{name}: current solver failed its physical audit: {row}')
        report['cases'].append(row)
        print(json.dumps(row), flush=True)

    size = 1+args.samples*args.batch
    z = {'CH4': .5, 'C2H6': .5}
    for method in ('PR', 'LKP'):
        for quality in (.0001, .5, .9999):
            model = create_thermodynamics(list(z), method, ChemicalDatabase(enable_online=False))
            inputs = ([(10., quality+i*.001*min(quality, 1.-quality)) for i in range(size)]
                      if args.fixed_pressure else [(10.+i*.001, quality) for i in range(size)])
            measure(f'{method}_PQ_{quality}', model, inputs,
                    lambda value: model.calculate_state_PQ(value[0], value[1], 1., z, include=()),
                    lambda value, s: max(abs((1-value[1])*s.x[c]+value[1]*s.y[c]-z[c]) for c in z),
                    coexistence=True)
    model = create_thermodynamics(['CH4'], 'PR', ChemicalDatabase(enable_online=False))
    measure('PR_pure_PQ_0.5', model, [10.+i*.001 for i in range(size)],
            lambda P: model.calculate_state_PQ(P, .5, 1., {'CH4': 1.}, include=()),
            lambda P, s: abs(s.x['CH4']-1.)+abs(s.y['CH4']-1.), coexistence=True)
    model = create_thermodynamics(list(z), 'PR', ChemicalDatabase(enable_online=False))
    def audit_tp(value, result):
        beta, x, y = result
        return max(abs((1-beta)*x[c]+beta*y[c]-z[c]) for c in z)
    measure('PR_TP', model, [200.+i*.001 for i in range(size)],
            lambda T: model.flash_TP(z, T, 10.), audit_tp)
    for quantity in ('H', 'S'):
        model = create_thermodynamics(['water'], 'STEAM', ChemicalDatabase(enable_online=False))
        if args.variant == 'before':
            model._tmax = model._props('Tmax', '', 0., '', 0.)
        water, pressure = {'water': 1.}, 3.
        inputs = [(T, getattr(model.calculate_state(T, pressure, 1., water,
                   include=(quantity,)), quantity)) for T in (450.+i*.01 for i in range(size))]
        inverse = getattr(model, 'calculate_state_P'+quantity)
        measure('STEAM_P'+quantity, model, inputs,
                lambda value: inverse(pressure, value[1], 1., water, include=(quantity,)),
                lambda value, state: abs(state.T-value[0]), tolerance=1e-7)
    return report


def paired_requests(args, name, values, evaluate, audit, switch, tolerance):
    """Use identical batches and alternate ordering under the same CPU load."""
    durations = {'before': [], 'after': []}
    errors = {'before': [], 'after': []}
    for sample in range(-1, args.samples):
        batch = values[max(0, sample)*args.batch:(max(0, sample)+1)*args.batch]
        order = ('before', 'after') if sample % 2 else ('after', 'before')
        for variant in order:
            switch(variant)
            start = time.perf_counter()
            results = [evaluate(variant, value) for value in batch]
            elapsed = (time.perf_counter()-start)/len(batch)
            if sample >= 0:
                durations[variant].append(elapsed)
                errors[variant].extend(audit(value, result) for value, result in zip(batch, results))
    old, new = (statistics.median(durations[v]) for v in ('before', 'after'))
    if max(errors['after']) > tolerance:
        raise RuntimeError(f'{name}: paired current solver failed its physical audit')
    row = {'name': name, 'before_ms': 1000*old, 'after_ms': 1000*new,
           'change_percent': 100*(new/old-1), 'requests_per_variant': len(values),
           'before_max_residual': max(errors['before']), 'after_max_residual': max(errors['after'])}
    print(json.dumps(row), flush=True)
    return row


def paired_steam(args):
    from chemical_properties import ChemicalDatabase
    from thermodynamics import create_thermodynamics
    from thermodynamics_models import steam
    owner = steam.SteamThermodynamics
    names = ('_state_from_inputs', '_refine_inverse_temperature', 'temperature_at_PH')
    current = {name: getattr(owner, name) for name in names}
    previous = {}
    for name in names:
        restore_method(args.baseline_dir, steam, 'SteamThermodynamics', name)
        previous[name] = getattr(owner, name)
        setattr(owner, name, current[name])
    reports = []
    try:
        for quantity in ('H', 'S'):
            models = {variant: create_thermodynamics(['water'], 'STEAM', ChemicalDatabase(enable_online=False))
                      for variant in ('before', 'after')}
            for model in models.values():
                model._tmax = model._props('Tmax', '', 0., '', 0.)
                model.initialize()
            z, pressure = {'water': 1.}, 3.
            values = [(T, getattr(models['after'].calculate_state(T, pressure, 1., z,
                       include=(quantity,)), quantity))
                      for T in (450.+i*.0001 for i in range(args.samples*args.batch))]
            def switch(variant):
                for name, method in (previous if variant == 'before' else current).items():
                    setattr(owner, name, method)
            inverses = {v: getattr(m, 'calculate_state_P'+quantity) for v, m in models.items()}
            reports.append(paired_requests(
                args, 'STEAM_P'+quantity, values,
                lambda variant, value: inverses[variant](pressure, value[1], 1., z, include=(quantity,)),
                lambda value, state: abs(state.T-value[0]), switch, 1e-7))
    finally:
        for name, method in current.items():
            setattr(owner, name, method)
    with (args.output/'summary.json').open('x') as stream:
        json.dump(reports, stream, indent=2)
    return 0


def paired_tp(args):
    """Alternate both native numerical plans and the public TP adapters."""
    import phi_phi
    import compiled_phi_phi
    from chemical_properties import ChemicalDatabase
    from thermodynamics import create_thermodynamics
    from thermodynamics_models import eos
    owner = eos.PhiPhiThermodynamics
    current_method = owner.flash_TP
    modules = {'after': {'phi_phi_numerics': phi_phi.numerics, 'compiled_phi_phi': compiled_phi_phi}}
    modules['before'] = {}
    try:
        for name in ('phi_phi_numerics', 'compiled_phi_phi'):
            spec = importlib.util.spec_from_file_location(name, args.baseline_dir/(name+'.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules['before'][name] = module
        restore_method(args.baseline_dir, eos, 'PhiPhiThermodynamics', 'flash_TP')
        previous_method = owner.flash_TP
        def switch(variant):
            sys.modules.update(modules[variant])
            phi_phi.numerics = modules[variant]['phi_phi_numerics']
            owner.flash_TP = previous_method if variant == 'before' else current_method
        models = {}
        for variant in ('before', 'after'):
            switch(variant)
            models[variant] = create_thermodynamics(['CH4', 'C2H6'], 'PR', ChemicalDatabase(enable_online=False))
            models[variant].initialize()
        z = {'CH4': .5, 'C2H6': .5}
        def audit(value, result):
            beta, x, y = result
            return max(abs((1-beta)*x[c]+beta*y[c]-z[c]) for c in z)
        row = paired_requests(
            args, 'PR_TP', [200.+i*.0001 for i in range(args.samples*args.batch)],
            lambda variant, T: models[variant].flash_TP(z, T, 10.), audit, switch, 1e-9)
    finally:
        sys.modules.update(modules['after'])
        phi_phi.numerics = modules['after']['phi_phi_numerics']
        owner.flash_TP = current_method
    with (args.output/'summary.json').open('x') as stream:
        json.dump([row], stream, indent=2)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--variant', choices=('before', 'after'))
    parser.add_argument('--fixed-pressure', action='store_true')
    paired = parser.add_mutually_exclusive_group()
    paired.add_argument('--paired-steam', action='store_true')
    paired.add_argument('--paired-tp', action='store_true')
    args = parser.parse_args()
    if args.samples < 1 or args.batch < 1:
        parser.error('samples and batch must be positive')
    if args.variant:
        print('REPORT', json.dumps(worker(args)), flush=True)
        return 0
    if args.output is None:
        parser.error('--output is required')
    args.output.mkdir(parents=True, exist_ok=False)
    if args.paired_steam:
        return paired_steam(args)
    if args.paired_tp:
        return paired_tp(args)
    reports = {}
    for variant in ('before', 'after'):
        path = args.output/(variant+'.log')
        with path.open('x') as stream:
            command = [
                sys.executable, str(Path(__file__).resolve()), '--variant', variant,
                '--baseline-dir', str(args.baseline_dir), '--samples', str(args.samples),
                '--batch', str(args.batch),
            ]
            if args.fixed_pressure:
                command.append('--fixed-pressure')
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f'{variant} failed; inspect {path}')
        reports[variant] = json.loads(next(line[7:] for line in path.read_text().splitlines()
                                         if line.startswith('REPORT ')))
    comparison = []
    for old, new in zip(reports['before']['cases'], reports['after']['cases']):
        assert old['name'] == new['name']
        row = {'name': new['name'], 'before_ms': old['median_ms'], 'after_ms': new['median_ms'],
               'change_percent': 100*(new['median_ms']/old['median_ms']-1),
               'before_TP_calls': old['TP_calls_per_request'], 'after_TP_calls': new['TP_calls_per_request'],
               'before_max_residual': old['max_residual'], 'after_max_residual': new['max_residual'],
               'before_valid_requests': old['valid_requests'], 'after_valid_requests': new['valid_requests'],
               'requests': new['requests']}
        comparison.append(row)
        print(json.dumps(row), flush=True)
    with (args.output/'summary.json').open('x') as stream:
        json.dump({'reports': reports, 'comparison': comparison}, stream, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
