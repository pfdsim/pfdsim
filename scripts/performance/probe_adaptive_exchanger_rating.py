#!/usr/bin/env python3
"""Compare production adaptive duty rating against its full-grid reference.

Experimental patches are process-local. The reference rating solver supplies each duty
candidate; no film formula, property path, phase check or tolerance is changed.
An accepted candidate must pass the requested grid's actual area residual.
Adaptive mode advances failed trials to a finer grid. Local mode first tries
a bound-preserving Brent correction on the full-grid equation; a failed or
invalid correction falls back through grid refinement to the original solve.

Example (output directory must be new)::

    .venv/bin/python scripts/performance/probe_adaptive_exchanger_rating.py \
        --cases double_pipe shell_tube --modes original production --repeats 3 \
        --output /tmp/adaptive-rating \
        2>&1 | tee /tmp/adaptive-rating.log

Use --stress for deterministic narrow-property-feature cases and --tests for
the focused exchanger/phase-change tests. The script generates no randomness.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import json
import math
from pathlib import Path
import sys
import time
import traceback
import subprocess
from types import SimpleNamespace
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from chemical_properties import ChemicalDatabase
from simulator import Simulator
from thermodynamics import ThermodynamicsError, create_thermodynamics
from unit_operations_base import UnitOperationError
from unit_operations_basic import HeatExchanger
from tests.test_heat_exchanger_transport import ConstantLiquid, parameters
from tests.test_heat_transfer_extensions import shell_parameters

ORIGINAL_RATE = HeatExchanger._rate_exchanger_by_curves
PRODUCTION_RATE = ORIGINAL_RATE
RATE_SIGNATURE = inspect.signature(ORIGINAL_RATE)
TRIAL_ERRORS = (ValueError, UnitOperationError, ThermodynamicsError)


@contextmanager
def rating_scope(mode):
    attempts = []
    def adaptive(*args, **kwargs):
        arguments = RATE_SIGNATURE.bind(*args, **kwargs)
        arguments.apply_defaults()
        values = arguments.arguments
        owner = values['self']
        full = values['segments']
        transport, target = values['transport'], values['A_value']
        if transport is None or full <= 4:
            return ORIGINAL_RATE(*args, **kwargs)
        full_cache = {}
        def full_curve(duty):
            if duty not in full_cache:
                curve = owner._curve_metrics(
                    values['hot'], values['cold'], values['P_hot_out'], values['P_cold_out'],
                    duty, full, values['flow_pattern'], include_u_estimates=False,
                    hot_force_phase=values['hot_force_phase'], cold_force_phase=values['cold_force_phase'],
                    transport=transport)
                residual = curve['calculated_U_area_required_m2']-target
                if not math.isfinite(residual) or curve['temperature_cross']:
                    raise UnitOperationError('Full-grid verification has invalid area or temperature cross')
                full_cache[duty] = curve
            return full_cache[duty]
        def full_residual(duty):
            return full_curve(duty)['calculated_U_area_required_m2']-target
        def local_correction(duty, residual):
            correction = {'unit': owner.unit_id, 'status': 'full_grid_local_correction',
                          'requested_segments': full, 'trial_segments': full}
            attempts.append(correction)
            direction = -1 if residual > 0 else 1
            distance = max(1e-5, abs(duty)*1e-3)
            for _ in range(6):
                trial = duty+direction*distance
                if trial <= 0:
                    break
                try:
                    other_residual = full_residual(trial)
                except TRIAL_ERRORS:
                    break  # Never bracket across a trial's invalid domain.
                if residual*other_residual <= 0:
                    low, high = sorted((duty, trial))
                    root = brentq(full_residual, low, high, xtol=1e-5, rtol=1e-9, maxiter=100)
                    verified = full_residual(root)
                    correction.update(duty_kJ_h=root, full_area_relative_residual=verified/target,
                                      full_evaluations=len(full_cache))
                    if abs(verified) <= max(1e-7, target*1e-6):
                        correction['status'] = 'full_grid_local_correction_accepted'
                        return root
                    break
                distance *= 2
            correction.update(status='full_grid_local_correction_failed', full_evaluations=len(full_cache))
            return None
        count = 4
        while count < full:
            attempt = {'unit': owner.unit_id, 'trial_segments': count, 'requested_segments': full}
            attempts.append(attempt)
            values['segments'] = count
            try:
                duty = ORIGINAL_RATE(*arguments.args, **arguments.kwargs)
                attempt['duty_kJ_h'] = duty
                curve = full_curve(duty)
                area = curve['calculated_U_area_required_m2']
                residual = area-target
                attempt.update(full_area_residual_m2=residual,
                               full_area_relative_residual=residual/target,
                               min_approach_K=curve['min_approach_K'],
                               full_nodes=len(curve['calculated_U_nodes']))
                if (math.isfinite(residual) and not curve['temperature_cross']
                        and abs(residual) <= max(1e-7, target*1e-6)):
                    attempt['status'] = 'accepted'
                    return duty
                attempt['status'] = 'full_grid_rejected'
                if mode == 'local':
                    root = local_correction(duty, residual)
                    if root is not None:
                        return root
            except TRIAL_ERRORS as error:
                attempt.update(status='trial_or_verification_error', error=f'{type(error).__name__}: {error}')
            count *= 2
        values['segments'] = full
        attempts.append({'unit': owner.unit_id, 'status': 'original_full_grid_fallback',
                         'trial_segments': full, 'requested_segments': full})
        return ORIGINAL_RATE(*arguments.args, **arguments.kwargs)
    if mode in ('adaptive', 'local'):
        HeatExchanger._rate_exchanger_by_curves = adaptive
    elif mode == 'original':
        HeatExchanger._rate_exchanger_by_curves = ORIGINAL_RATE
    elif mode == 'production':
        HeatExchanger._rate_exchanger_by_curves = PRODUCTION_RATE
    try:
        yield attempts
    finally:
        HeatExchanger._rate_exchanger_by_curves = PRODUCTION_RATE


@dataclass
class Case:
    name: str
    model: str = 'STEAM'
    composition: dict | None = None
    hot_T: float = 360.
    cold_T: float = 300.
    hot_P: float = 3.
    cold_P: float = 3.
    hot_F: float = 200.
    cold_F: float = 300.
    hot_quality: float | None = None
    cold_quality: float | None = None
    duty_kW: float = 20.
    design_tube_out_K: float | None = None
    geometry: str = 'double_pipe'
    updates: dict | None = None
    feature_settings: dict | None = None

    def build(self, segments):
        composition = self.composition or {'water': 1.}
        if self.model == 'constant':
            thermo = ConstantLiquid()
        elif self.model.startswith('feature'):
            thermo = NarrowFeature(self.model)
            for name, value in (self.feature_settings or {}).items():
                setattr(thermo, name, value)
        else:
            thermo = create_thermodynamics(list(composition), self.model,
                                           ChemicalDatabase(enable_online=False))
            thermo.initialize()
        def state(T, P, F, quality):
            return (thermo.calculate_state(T, P, F, composition) if quality is None else
                    thermo.calculate_state_PQ(P, quality, F, composition))
        hot = state(self.hot_T, self.hot_P, self.hot_F, self.hot_quality)
        cold = state(self.cold_T, self.cold_P, self.cold_F, self.cold_quality)
        params = (shell_parameters() if self.geometry == 'shell_tube' else parameters())
        params.update(self.updates or {})
        params['curve_segments'] = segments
        return thermo, {'tube_in': hot, 'shell_in': cold}, params


class NarrowFeature(ConstantLiquid):
    """Positive transport perturbations test quadrature rejection/refinement."""

    def __init__(self, name):
        self.width = {'feature-wide': .6, 'feature-narrow': .08, 'feature-invalid': .15}[name]
        self.invalid = name == 'feature-invalid'
        self.center, self.amplitude = 302.1, .8

    def mixture_viscosity(self, composition, T, P, V):
        # A narrow property ridge is deliberately placed between coarse sample
        # points. It is applied only on the lower-pressure coolant side.
        if P != 4.:
            return .001
        if self.invalid and abs(T-self.center) < self.width:
            raise ThermodynamicsError('Deliberate narrow unsupported transport domain')
        return .001*(1+self.amplitude*math.exp(-((T-self.center)/self.width)**2))


def cases():
    return {
        'liquid_steam': Case('liquid_steam'),
        'liquid_ideal': Case('liquid_ideal', model='IDEAL'),
        'mixture_nrtl': Case('mixture_nrtl', model='NRTL', composition={'water': .9, 'ethanol': .1},
                             hot_T=350., cold_T=305.),
        'condense_tube': Case('condense_tube', hot_P=5., hot_F=20., hot_quality=.8,
                              cold_T=350., cold_F=200., duty_kW=20.),
        'condense_shell': Case('condense_shell', geometry='shell_tube', hot_T=320., hot_F=400.,
                               cold_P=5., cold_F=15., cold_quality=.95, duty_kW=10.,
                               updates={'baffle_count': 1}),
        'boil_and_condense': Case('boil_and_condense', hot_P=3., hot_F=100., hot_quality=.05,
                                  cold_P=5., cold_F=20., cold_quality=.8, duty_kW=20.),
        'subcool_after_condense': Case('subcool_after_condense', hot_T=450., hot_P=5., hot_F=1.,
                                       cold_T=310., cold_F=100., design_tube_out_K=380.),
        'pressure_drop': Case('pressure_drop', hot_T=380., hot_P=5., cold_T=300., cold_P=4.,
                              updates={'P_drop_tube': .5, 'P_drop_shell': .2}),
        'cocurrent': Case('cocurrent', updates={'flow_pattern': 'cocurrent'}),
        'tube_is_cold': Case('tube_is_cold', hot_T=300., cold_T=360., hot_F=300., cold_F=200.),
        'tight_pinch': Case('tight_pinch', model='constant', duty_kW=240.),
        'laminar_tube': Case('laminar_tube', model='constant', hot_F=1., duty_kW=.1),
        'transition_tube': Case('transition_tube', model='constant', hot_F=10., duty_kW=1.),
        'feature-wide': Case('feature-wide', model='feature-wide', cold_P=4., duty_kW=25.),
        'feature-narrow': Case('feature-narrow', model='feature-narrow', cold_P=4., duty_kW=25.),
        'feature-invalid': Case('feature-invalid', model='feature-invalid', cold_P=4., duty_kW=25.),
    }


def metrics(result, inlets):
    perf = result.performance
    nodes = perf['calculated_U_nodes']
    Q = perf['duty_kW']
    thermal_residuals = []
    phase_ranges = {}
    for side, inlet in inlets.items():
        outlet = result.outlet_streams[side.replace('_in', '_out')]
        thermal_residuals.append(abs(abs(inlet.F*(outlet.H-inlet.H)/3600)-Q))
        phase_ranges[side] = [inlet.vapor_fraction, outlet.vapor_fraction]
    return {'duty_kW': Q, 'area_m2': perf['area_m2'], 'area_required_m2': perf['area_required_m2'],
            'area_relative_residual': perf['area_required_m2']/perf['area_m2']-1,
            'max_wall_residual_K': max(n['wall_temperature_residual_K'] for n in nodes),
            'min_approach_K': perf['min_approach_K'], 'nodes': len(nodes),
            'max_energy_residual_kW': max(thermal_residuals), 'phase_fractions': phase_ranges,
            'outlet_T_K': {name: state.T for name, state in result.outlet_streams.items()},
            'warnings': result.warnings,
            'film_regimes': sorted({n[side]['flow_regime'] for n in nodes
                                   for side in ('tube', 'annulus', 'shell') if side in n})}


def run_case(case, segments, mode, area, output):
    thermo, inlets, params = case.build(segments)
    with rating_scope(mode) as attempts:
        start, cpu = time.perf_counter(), time.process_time()
        record = {'case': case.name, 'mode': mode, 'segments': segments, 'supplied_area': area}
        try:
            unit = HeatExchanger('HX', thermo, {**params, 'A': area})
            result = unit.solve(inlets)
            record.update(status='ok', metrics=metrics(result, inlets))
        except Exception as error:
            record.update(status='error', error=f'{type(error).__name__}: {error}',
                          traceback=traceback.format_exc())
        record.update(wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu,
                      attempts=attempts)
    with (output/f'{case.name}-{mode}-{segments}.json').open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in record.items() if k not in ('traceback',)}), flush=True)
    return record


def reference_area(case, segments):
    thermo, inlets, params = case.build(segments)
    design = ({'Q': case.duty_kW} if case.design_tube_out_K is None else
              {'T_tube_out': case.design_tube_out_K})
    return HeatExchanger('DESIGN', thermo, {**params, **design}).solve(inlets).performance['area_required_m2']


def run_example(name, mode, output, repeat):
    filename = {'double_pipe': 'double_pipe_calculated_u.pfd', 'shell_tube': 'shell_tube_calculated_u.pfd'}[name]
    simulator = Simulator.from_file(ROOT/'examples'/filename)
    simulator.initialize()
    with rating_scope(mode) as attempts:
        start, cpu = time.perf_counter(), time.process_time()
        result = simulator.run()
        record = {'case': name, 'mode': mode, 'repeat': repeat,
                  'wall_seconds': time.perf_counter()-start, 'cpu_seconds': time.process_time()-cpu,
                  'status': 'ok' if result.converged and not result.errors else 'error',
                  'errors': result.errors, 'attempts': attempts}
        perf = result.units['HX-100'].performance
        record['metrics'] = {'duty_kW': perf['duty_kW'], 'area_relative_residual':
            perf['area_required_m2']/perf['area_m2']-1, 'nodes': len(perf['calculated_U_nodes']),
            'min_approach_K': perf['min_approach_K'], 'max_wall_residual_K':
            max(n['wall_temperature_residual_K'] for n in perf['calculated_U_nodes'])}
    with (output/f'{name}-{mode}-{repeat}.json').open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps(record), flush=True)
    return record


def audit_guards(output):
    """Exercise acceptance guards independently of physical case selection."""
    class RatingFixture(HeatExchanger):
        def __init__(self, fault):
            self.unit_id, self.fault = 'GUARD', fault
        def _thermo_for_stream(self, stream):
            return None
        def _curve_metrics(self, hot, cold, P_hot, P_cold, duty, segments, *args, **kwargs):
            if segments == 16 and self.fault == 'domain':
                raise ThermodynamicsError('Manufactured full-grid domain rejection')
            area = duty*(1.001 if segments == 4 else 1.)
            if segments == 16 and self.fault == 'nan':
                area = math.nan
            return {'calculated_U_area_required_m2': area, 'min_approach_K': 1.,
                    'temperature_cross': segments == 16 and self.fault == 'cross',
                    'calculated_U_nodes': []}
    rows = []
    for fault in ('none', 'nan', 'cross', 'domain'):
        statuses = []
        for mode in ('original', 'adaptive', 'local', 'production'):
            with rating_scope(mode) as attempts:
                unit = RatingFixture(fault)
                try:
                    duty = unit._rate_exchanger_by_curves(
                        SimpleNamespace(T=2.), SimpleNamespace(T=1.), 1., 1., None, 1., None,
                        False, 16, 'countercurrent', transport=SimpleNamespace(inlet_capacity_W_K=lambda *a: 1.))
                    row = {'fault': fault, 'mode': mode, 'status': 'ok', 'duty': duty, 'attempts': attempts}
                    assert fault == 'none' and abs(duty-1.) < 1e-9
                except TRIAL_ERRORS as error:
                    row = {'fault': fault, 'mode': mode, 'status': 'error',
                           'error': f'{type(error).__name__}: {error}', 'attempts': attempts}
                    assert fault != 'none'
                statuses.append(row['status'])
                rows.append(row)
        assert len(set(statuses)) == 1
    with (output/'guard-audit.json').open('x') as stream:
        json.dump(rows, stream, indent=2, allow_nan=False)
    print('GUARD_AUDIT', len(rows), 'checks passed', flush=True)


def main():
    global ORIGINAL_RATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cases', nargs='+', default=['double_pipe', 'shell_tube'])
    parser.add_argument('--segments', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--stress', action='store_true')
    parser.add_argument('--tests', action='store_true')
    parser.add_argument('--feature-sweep', action='store_true')
    parser.add_argument('--modes', nargs='+', choices=('original', 'adaptive', 'local', 'production'),
                        default=('original', 'production'))
    parser.add_argument('--reference-revision', default='HEAD',
                        help='Git revision containing the full-grid reference rating method')
    args = parser.parse_args()
    if args.repeats < 1 or not 4 <= args.segments <= 400:
        parser.error('Positive repeats and segments in [4,400] are required')
    args.output.mkdir(parents=True, exist_ok=False)
    source = subprocess.run(['git', 'show', f'{args.reference_revision}:unit_operations_basic.py'],
                            cwd=ROOT, check=True, capture_output=True, text=True).stdout
    tree = ast.parse(source)
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HeatExchanger')
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '_rate_exchanger_by_curves')
    namespace = dict(vars(sys.modules[HeatExchanger.__module__]))
    exec(compile(ast.Module(body=[method], type_ignores=[]),
                 f'git:{args.reference_revision}:unit_operations_basic.py', 'exec'), namespace)
    ORIGINAL_RATE = namespace['_rate_exchanger_by_curves']
    audit_guards(args.output)
    failures = 0
    if args.tests:
        import pytest
        with rating_scope(args.modes[-1]) as attempts:
            status = pytest.main(['tests/test_heat_exchanger_transport.py', 'tests/test_heat_transfer_extensions.py'])
        with (args.output/'pytest-attempts.json').open('x') as stream:
            json.dump({'exit_code': int(status), 'attempts': attempts}, stream, indent=2)
        failures += int(status != 0)
    catalog = cases()
    names = list(args.cases)
    if args.stress:
        names.extend(name for name in catalog if name not in names)
    if args.feature_sweep:
        for index, (center, width) in enumerate((center, width)
                for center in (301., 302.1, 303.) for width in (.03, .08, .2, .6)):
            name = f'feature-sweep-{index:02d}'
            catalog[name] = Case(name, model='feature-wide', cold_P=4., duty_kW=25.,
                                 feature_settings={'center': center, 'width': width})
            names.append(name)
    for name in names:
        if name in ('double_pipe', 'shell_tube'):
            for repeat in range(args.repeats):
                for mode in args.modes:
                    failures += int(run_example(name, mode, args.output, repeat)['status'] != 'ok')
            continue
        case = catalog[name]
        try:
            design_case = (Case('feature-design', model='constant', cold_P=4., duty_kW=25.)
                           if name == 'feature-invalid' else case)
            area = reference_area(design_case, args.segments)
        except Exception as error:
            record = {'case': name, 'status': 'design_error', 'error': f'{type(error).__name__}: {error}'}
            with (args.output/f'{name}-design.json').open('x') as stream:
                json.dump(record, stream, indent=2)
            print(json.dumps(record), flush=True)
            failures += 1
            continue
        original = None
        for mode in args.modes:
            record = run_case(case, args.segments, mode, area, args.output)
            if mode == 'original':
                original = record
            elif original is not None:
                if original['status'] != record['status']:
                    failures += 1
                elif original['status'] == 'ok':
                    difference = abs(record['metrics']['duty_kW']/original['metrics']['duty_kW']-1)
                    if difference > 1e-5 or abs(record['metrics']['area_relative_residual']) > 1e-6:
                        failures += 1
                    print('COMPARISON', name, mode, 'relative_duty_difference', difference, flush=True)
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())
