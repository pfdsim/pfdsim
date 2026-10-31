#!/usr/bin/env python3
"""Measure exchanger optimization candidates without editing production code.

Use new output directories and persist both streams, for example::

    .venv/bin/python scripts/performance/probe_exchanger_acceleration.py \
        --segments 10 --variants current wall-pr prepared factored \
        --output /tmp/exchanger-probes 2>&1 | tee /tmp/exchanger-probes.log

Fresh simulators are initialized before timing. Candidates preserve existing
film correlations, root tolerances, phase checks, and final wall audits.
Prepared-property experiments are restricted to canonical pure-fluid models,
falling back outside the selected conductivity range or near saturation.
They are numerical probes; production provenance integration is not implied.
No random inputs are used. Reports include independent property spot checks.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import inspect
import json
import math
from pathlib import Path
import re
import sys
import subprocess
import textwrap
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import heat_exchanger_transport as transport_module
from heat_exchanger_transport import TubularHeatTransfer
from simulator import Simulator
from unit_operations_basic import HeatExchanger
from thermodynamics_models.base import IdealThermodynamics
from thermodynamics_models.steam import SteamThermodynamics
from thermodynamics_models.common import _property_lookup_identifier
from property_resolver import get_property_resolver
from scripts.performance.probe_heat_exchanger_transport import compact

ORIGINAL_SINGLE_PHASE = TubularHeatTransfer._single_phase
ORIGINAL_CURVE_STATES = HeatExchanger._curve_states
ORIGINAL_RATE = HeatExchanger._rate_exchanger_by_curves
ORIGINAL_STEAM_PH = SteamThermodynamics.calculate_state_PH
REFERENCE_SOURCE = None
EXAMPLES = {
    'double_pipe': 'double_pipe_calculated_u.pfd',
    'shell_tube': 'shell_tube_calculated_u.pfd',
}


class PropertyProbe:
    def __init__(self, variant):
        self.variant = variant
        self.evaluators = {}
        self.cache = {}
        self.requests = self.evaluations = self.fallbacks = 0
        self.prepared = []

    @staticmethod
    def public_pr(prepared, temperature):
        thermo, state = prepared['thermo'], prepared['state']
        vf = float(prepared['quality'] >= 1-1e-10)
        cp = thermo.phase_weighted_mixture_Cp(
            state.composition, temperature, vf, state.composition, state.composition, state.P)
        mu = thermo.mixture_viscosity(state.composition, temperature, state.P, vf)
        k = thermo.transport_mixture_thermal_conductivity(state.composition, temperature, state.P, vf)
        conductivity = k.vapor if vf else k.liquid
        return cp*1000/thermo.mixture_MW(state.composition)*mu/conductivity

    def evaluator(self, prepared):
        thermo, state = prepared['thermo'], prepared['state']
        phase = 'vapor' if prepared['quality'] >= 1-1e-10 else 'liquid'
        key = thermo, tuple(state.composition.items()), state.P, phase
        if key in self.evaluators:
            return self.evaluators[key]
        fallback = lambda T: self.public_pr(prepared, T)
        evaluate = fallback
        active = [c for c, v in state.composition.items() if v > 0.]
        mw = thermo.mixture_MW(state.composition)
        if type(thermo) is SteamThermodynamics and len(active) == 1 and prepared['sat']:
            import CoolProp
            saturation_T = prepared['sat']['liquid_state'].T
            def steam(T):
                if abs(T-saturation_T) <= max(1e-6, 1e-8*saturation_T):
                    self.fallbacks += 1
                    return fallback(T)
                # In this installed IF97 backend update() does not invalidate
                # cached transport values. A fresh state preserves PropsSI's
                # stateless semantics for viscosity and conductivity.
                water = CoolProp.AbstractState('IF97', 'Water')
                water.update(CoolProp.PT_INPUTS, state.P*1e5, T)
                cp = water.cpmass()*thermo._water_mw/mw
                return cp*water.viscosity()/water.conductivity()
            evaluate = steam
        elif type(thermo) is IdealThermodynamics and len(active) == 1 and phase == 'liquid':
            comp = active[0]
            # Resolve once with the actual authoritative source selection.
            thermo._pure_viscosity(comp, state.T, state.P, phase)
            viscosity = thermo._viscosity_kernels[(phase, comp)]
            cp = thermo._liquid_cp_kernel(comp)
            resolver = get_property_resolver()
            props = thermo._resolver_known_props.get(comp)
            identifier = _property_lookup_identifier(comp, thermo.props[comp])
            selected = resolver.resolve_thermal_conductivity(identifier, state.T, phase, props)
            library = resolver._get_perry_library()
            row = None
            for candidate in resolver._identifier_candidates(identifier, props):
                row = library.thermal_conductivity_correlation(candidate, state.T, phase) if library else None
                if row is not None:
                    break
            if cp is not None and row is not None and selected.method == f'perry_liquid_thermal_conductivity_eq{row["equation_id"]}':
                correlation = resolver._normalized_perry_conductivity_correlation(row)
                def ideal(T):
                    if not row['T_min_K'] <= T <= row['T_max_K']:
                        self.fallbacks += 1
                        return fallback(T)
                    conductivity = resolver._evaluate_correlation(correlation, T, props=props)[0]
                    return cp.cp(T)*1000/mw*viscosity.viscosity(T, P=state.P)/conductivity
                evaluate = ideal
        self.evaluators[key] = evaluate
        self.prepared.append((prepared, evaluate))
        return evaluate

    def wall_pr(self, prepared, temperature):
        self.requests += 1
        state = prepared['state']
        key = prepared['thermo'], tuple(state.composition.items()), state.P, prepared['quality'], temperature
        if key not in self.cache:
            self.evaluations += 1
            value = (self.public_pr(prepared, temperature) if self.variant == 'wall-pr'
                     else self.evaluator(prepared)(temperature))
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid wall Prandtl number')
            if len(self.cache) >= 4096:
                self.cache.clear()
            self.cache[key] = value
        return {'prandtl': self.cache[key]}

    def single_phase(self, owner, prepared, temperature):
        props = prepared['properties']
        if self.variant not in ('factored', 'homogeneous', 'coarse-rating', 'consistent-ph', 'coarse-consistent') or prepared['quality'] != 0:
            return self.modified(owner, prepared, temperature)
        wall_pr = self.wall_pr(prepared, temperature)['prandtl']
        reference = prepared.get('_probe_film_reference')
        if reference is None:
            if prepared['side'] == 'shell' and owner.shell_geometry:
                result = owner.shell_geometry.shell_coefficient(props, props['prandtl'])
                exponent = .25  # Nu_Zukauskas_Bejan's documented Pr_wall factor.
            else:
                transfer = transport_module.internal_flow_transfer(
                    props['reynolds'], props['prandtl'], props['prandtl'],
                    annulus_diameter_ratio=prepared['annulus_ratio'])
                result = {**transfer, 'h_W_m2_K': transfer['nusselt']*props['conductivity_W_m_K']/prepared['diameter']}
                exponent = .11 if transfer['flow_regime'] == 'turbulent' else None
            reference = prepared['_probe_film_reference'] = result, exponent
        result, exponent = reference
        if exponent is None or (exponent == .11 and not .5 <= wall_pr <= 1000):
            return self.modified(owner, prepared, temperature)
        correction = (props['prandtl']/wall_pr)**exponent
        updated = {**result, 'h_W_m2_K': result['h_W_m2_K']*correction,
                   'nusselt': result['nusselt']*correction}
        if 'wall_prandtl' in updated:
            updated['wall_prandtl'] = wall_pr
        return updated

    def audit(self):
        errors = []
        for prepared, evaluate in self.prepared:
            for T in (300.+60*i/8 for i in range(9)):
                expected = self.public_pr(prepared, T)
                errors.append(abs(evaluate(T)/expected-1.))
        return max(errors, default=0.)


@contextmanager
def variant_scope(variant, rating_segments=10):
    probe = PropertyProbe(variant)
    if variant == 'current':
        yield probe
        return
    previous = TubularHeatTransfer._single_phase
    source = textwrap.dedent(REFERENCE_SOURCE or inspect.getsource(ORIGINAL_SINGLE_PHASE))
    original = "wall = state.copy()\n    wall.T = temperature\n    wall_props = self._phase_properties(prepared['thermo'], wall, phase)"
    if source.count(original) != 1:
        raise RuntimeError('Inspect changed production wall-property lookup before probing')
    source = source.replace(original, 'wall_props = _wall_pr(prepared, temperature)')
    namespace = dict(vars(transport_module), _wall_pr=probe.wall_pr)
    exec(compile(source, '<probe-wall-prandtl>', 'exec'), namespace)
    probe.modified = namespace['_single_phase']
    if variant != 'current':
        TubularHeatTransfer._single_phase = lambda owner, prepared, T: probe.single_phase(owner, prepared, T)
    def curve_states(owner, hot, cold, P_hot_out, P_cold_out, Q, segments,
                     hot_force_phase=None, cold_force_phase=None, fractions=None):
        key = id(hot), id(cold), P_hot_out, P_cold_out
        saved = getattr(owner, '_probe_liquid_envelope', None)
        if saved is None or saved[0] != key:
            limit = 0.
            lo, hi = min(hot.T, cold.T), max(hot.T, cold.T)
            backends = [owner._thermo_for_stream(s) for s in (hot, cold)]
            eligible = lo >= 273.16 and all(
                type(t) in (IdealThermodynamics, SteamThermodynamics)
                and len(s.composition) == 1 and s.fluid_vapor_fraction == 0.
                and t.props[next(iter(s.composition))].CAS == '7732-18-5'
                and p == s.P
                for s, t, p in zip((hot, cold), backends, (P_hot_out, P_cold_out)))
            if eligible and all(t.calculate_state(hi, s.P, s.F, s.composition, include=()).fluid_vapor_fraction == 0.
                                for s, t in zip((hot, cold), backends)):
                hot_limit = backends[0].mixture_enthalpy(hot.composition, lo, 0., P=hot.P)
                cold_limit = backends[1].mixture_enthalpy(cold.composition, hi, 0., P=cold.P)
                limit = min(hot.F*(owner._stream_enthalpy(hot)-hot_limit),
                            cold.F*(cold_limit-owner._stream_enthalpy(cold)))
            saved = owner._probe_liquid_envelope = key, limit
        if saved[1] > 0. and 0. <= Q <= saved[1]*(1+1e-12):
            hot_force_phase = hot_force_phase or 'liquid'
            cold_force_phase = cold_force_phase or 'liquid'
        return ORIGINAL_CURVE_STATES(owner, hot, cold, P_hot_out, P_cold_out, Q, segments,
                                     hot_force_phase, cold_force_phase, fractions)
    if variant in ('homogeneous', 'coarse-rating', 'consistent-ph', 'coarse-consistent'):
        HeatExchanger._curve_states = curve_states
    def consistent_ph(thermo, P, H, F, composition, include=None, *, phase=None, T_guess=None):
        if phase is None and P < thermo._pcrit_bar:
            cache = getattr(thermo, '_probe_sat_enthalpies', None)
            if cache is None:
                cache = thermo._probe_sat_enthalpies = {}
            if P not in cache:
                cache[P] = tuple(thermo.calculate_state_PQ(P, q, F, composition, include=('H',)).H
                                 for q in (0., 1.))
            lower, upper = cache[P]
            phase = 'liquid' if H < lower else ('vapor' if H > upper else None)
        if phase is None:
            return ORIGINAL_STEAM_PH(thermo, P, H, F, composition, include)
        temperature, _ = thermo.temperature_at_PH(P, H, composition, phase=phase, T_guess=T_guess)
        return thermo.calculate_state(temperature, P, F, composition, phase=phase, flash=False, include=include)
    if variant in ('consistent-ph', 'coarse-consistent'):
        SteamThermodynamics.calculate_state_PH = consistent_ph
    def coarse_rating(*args, **kwargs):
        arguments = inspect.signature(ORIGINAL_RATE).bind(*args, **kwargs)
        arguments.arguments['segments'] = min(rating_segments, arguments.arguments['segments'])
        return ORIGINAL_RATE(*arguments.args, **arguments.kwargs)
    if variant in ('coarse-rating', 'coarse-consistent'):
        HeatExchanger._rate_exchanger_by_curves = coarse_rating
    try:
        yield probe
    finally:
        TubularHeatTransfer._single_phase = previous
        HeatExchanger._curve_states = ORIGINAL_CURVE_STATES
        HeatExchanger._rate_exchanger_by_curves = ORIGINAL_RATE
        SteamThermodynamics.calculate_state_PH = ORIGINAL_STEAM_PH


def simulate(example, segments, variant, output, repeat, rating_segments=10):
    source = (ROOT/'examples'/EXAMPLES[example]).read_text()
    source = re.sub(r'curve_segments = \d+', f'curve_segments = {segments}', source)
    with variant_scope(variant, rating_segments) as probe:
        simulator = Simulator.from_string(source)
        simulator.initialize()
        start, cpu = time.perf_counter(), time.process_time()
        result = simulator.run()
        seconds, cpu = time.perf_counter()-start, time.process_time()-cpu
        performance = result.units['HX-100'].performance
        forward_H = {}
        max_H_error = 0.
        for name, state in result.streams.items():
            thermo = simulator.thermo
            phase = 'vapor' if state.vapor_fraction >= 1-1e-10 else 'liquid'
            evaluated = thermo.calculate_state(state.T, state.P, state.F, state.composition,
                                                phase=phase, flash=False, include=('H',)).H
            forward_H[name] = evaluated
            max_H_error = max(max_H_error, abs(evaluated-state.H))
        hot_in, cold_in = ('HotWater', 'ColdWater') if example == 'double_pipe' else ('HotFeed', 'ColdFeed')
        q_hot = result.streams[hot_in].F*(forward_H[hot_in]-forward_H['CooledWater'])/3600
        q_cold = result.streams[cold_in].F*(forward_H['HeatedWater']-forward_H[cold_in])/3600
        record = {'example': example, 'segments': segments, 'variant': variant, 'repeat': repeat,
                  'rating_segments': min(segments, rating_segments) if variant.startswith('coarse') else segments,
                  'wall_seconds': seconds, 'cpu_seconds': cpu, 'converged': result.converged,
                  'errors': result.errors, 'mass_balance_error': result.mass_balance_error,
                  'energy_balance_error': result.energy_balance_error, 'performance': compact(performance),
                  'streams': {name: {key: getattr(state, key) for key in ('T','P','F','H','vapor_fraction')}
                              for name, state in result.streams.items()},
                  'wall_requests': probe.requests, 'wall_evaluations': probe.evaluations,
                  'property_fallbacks': probe.fallbacks, 'max_property_relative_error': probe.audit(),
                  'max_stream_H_consistency_error_kJ_kmol': max_H_error,
                  'forward_hot_duty_kW': q_hot, 'forward_cold_duty_kW': q_cold,
                  'max_forward_duty_residual_kW': max(abs(q_hot-performance['duty_kW']), abs(q_cold-performance['duty_kW'])),
                  'final_area_relative_residual': abs(performance['area_required_m2']/performance['area_m2']-1.)}
    with (output/f'{example}-{variant}-{segments}-{repeat}.json').open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in record.items() if k not in ('streams',)}), flush=True)
    return record


def main():
    global REFERENCE_SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--examples', nargs='+', choices=tuple(EXAMPLES), default=tuple(EXAMPLES))
    parser.add_argument('--segments', nargs='+', type=int, default=[40])
    parser.add_argument('--variants', nargs='+', choices=('current','wall-pr','prepared','factored','homogeneous','coarse-rating','consistent-ph','coarse-consistent'),
                        default=('current',))
    parser.add_argument('--reference-revision', default='HEAD',
                        help='Revision of the original wall method for historical experimental variants')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--rating-segments', type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1 or not 4 <= args.rating_segments <= 400 or any(not 4 <= n <= 400 for n in args.segments):
        parser.error('Positive repeats and segments in [4,400] are required')
    args.output.mkdir(parents=True, exist_ok=False)
    if any(variant != 'current' for variant in args.variants):
        source = subprocess.run(['git', 'show', f'{args.reference_revision}:heat_exchanger_transport.py'],
                                cwd=ROOT, check=True, capture_output=True, text=True).stdout
        tree = ast.parse(source)
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TubularHeatTransfer')
        method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '_single_phase')
        REFERENCE_SOURCE = ''.join(source.splitlines(keepends=True)[method.lineno-1:method.end_lineno])
    failures = 0
    for segments in args.segments:
        for variant in args.variants:
            for example in args.examples:
                for repeat in range(args.repeats):
                    record = simulate(example, segments, variant, args.output, repeat, args.rating_segments)
                    failures += int(not record['converged'] or bool(record['errors']) or
                                    record['max_property_relative_error'] > 1e-9 or
                                    record['final_area_relative_residual'] > 1e-6)
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())
