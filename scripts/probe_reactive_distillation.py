#!/usr/bin/env python3
"""Probe reactive MESH equations without modifying production code.

Reuse pfdsim's VLE MESH residual, variable transforms, thermodynamics, reaction
definitions, rate evaluator and sparse Newton solver. Add one signed, scaled
extent and one closure per reactive stage. No empirical kinetics are claimed:
A=2000, Ea=0 and the liquid volumes are numerical sensitivity parameters.

Run from the repository root, for example:
    set -o pipefail
    .venv/bin/python scripts/probe_reactive_distillation.py --workers 3 \
        --output /tmp/pfdsim-reactive-distillation \
        2>&1 | tee /tmp/pfdsim-reactive-distillation.log > /dev/null

Cases run in spawned, cache-isolated processes with a timeout. Reports include
failed solves, stage profiles, independent conservation audits and provenance.
The experiment assumes one homogeneous liquid on each stage; spinodal checks
are reported separately and are not hidden by the numerical success flag.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scipy.sparse import bmat, lil_matrix

from chemical_properties import ChemicalDatabase
from compound_identity import parse_formula_counts
from kinetic_models import (
    HomogeneousRateState,
    evaluate_reaction_rate,
    kinetic_reaction_from_mapping,
)
from reaction_models import equilibrium_reaction_from_mapping
from property_resolver import get_property_resolver
from tests.cache_isolation import isolated_runtime_caches
from thermodynamics import StreamState, create_thermodynamics
from unit_operations_distillation import RigorousDistillation

COMPONENTS = ['ethanol', 'acetic acid', 'ethyl acetate', 'water']
EQUATION = 'ethanol + acetic acid <=> ethyl acetate + water'
TOLERANCE = 1e-8


def write_json(path, value):
    # Outputs must be new; prevent accidental rewriting of prior reports.
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def make_thermodynamics(case):
    method = case['method']
    options = {'correlation': 'HOC'} if method == 'NRTL-BV' else None
    database = ChemicalDatabase(enable_online=False)
    supporting_estimates = {}
    if method == 'NRTL-BV' and case.get('estimate_hoc_geometry'):
        resolver = get_property_resolver()
        for component in COMPONENTS:
            props = database.get(component)
            known = vars(props).copy()
            radius = resolver.resolve_modified_radius_of_gyration(
                component, known, allow_online=False, allow_estimation=True,
            )
            props.modified_radius_of_gyration = float(radius.value)
            props.property_sources['modified_radius_of_gyration'] = {
                'source': radius.source, 'method': radius.method,
                'quality': radius.quality, 'notes': radius.notes,
            }
            supporting_estimates[component] = {
                'modified_radius_of_gyration_A': float(radius.value),
                'method': radius.method, 'quality': radius.quality,
            }
    thermo = create_thermodynamics(
        COMPONENTS, method, db=database,
        thermo_options=options,
    )
    return thermo, supporting_estimates


def make_case(case):
    started = time.perf_counter()
    method = case['method']
    thermo, supporting_estimates = make_thermodynamics(case)
    metadata = {c: thermo.props[c] for c in COMPONENTS}
    equilibrium = equilibrium_reaction_from_mapping(
        {'equation': EQUATION}, COMPONENTS, metadata,
    )
    kinetic = kinetic_reaction_from_mapping({
        'equation': EQUATION, 'type': 'power_law', 'A': 2000., 'Ea': 0.,
        'Ea_unit': 'J/mol', 'rate_basis': 'activity',
        'rate_unit': 'kmol/m3/h', 'concentration_unit': 'kmol/m3',
        'pressure_unit': 'bar',
    }, COMPONENTS, metadata)
    nu = equilibrium.reaction.stoichiometry
    stages = case['stages']
    active = list(range(1, stages - 1))
    nr = len(active)
    flow = 100.
    z = ({c: .25 for c in COMPONENTS} if case.get('reverse') else
         dict(zip(COMPONENTS, [.5, .5, 0., 0.])))
    if case.get('reverse'):
        z = dict(zip(COMPONENTS, [0., 0., .5, .5]))
    feed = StreamState(T=350., P=1., F=flow, composition=z, vapor_fraction=0.)
    feed.H = thermo.mixture_enthalpy(z, feed.T, 0., P=feed.P)
    unit = RigorousDistillation('probe', thermo, {'initializer': 'estimate'})
    feed_stage = stages // 2
    feed_specs = [{'stage': feed_stage - 1, 'F': flow, 'z': z, 'H': feed.H}]
    if case.get('split_feeds'):
        feed_specs = []
        for component, stage in (('acetic acid', 1), ('ethanol', stages - 2)):
            composition = {c: float(c == component) for c in COMPONENTS}
            feed_specs.append({'stage': stage, 'F': 50., 'z': composition,
                               'H': thermo.mixture_enthalpy(composition, feed.T, 0., P=1.)})
        feed.H = sum(f['F'] * f['H'] for f in feed_specs) / flow
    pressures = [1.] * stages
    t_min, t_max = unit._temperature_bounds(COMPONENTS)
    energy_scale = max(abs(flow * feed.H), flow * 50000.)
    # Newly generated products must have a meaningful source-based scale.
    scales = {c: 50. for c in COMPONENTS}
    spec = {'kind': 'molar', 'value': case.get('distillate', 50.)}
    reflux = case.get('reflux', 2.)
    base = unit._build_mesh_model(
        feed, feed_specs, COMPONENTS, z, stages, feed_stage - 1, reflux,
        pressures, 'total', 0., [], spec, flow, energy_scale, scales,
        t_min, t_max,
    )
    # Positive seed includes reaction products even when actual feed has none.
    seed_composition = {c: .25 for c in COMPONENTS}
    initial = unit._initial_guess(
        feed, COMPONENTS, seed_composition, stages, feed_stage, reflux, 1.,
        pressures, 'total', 0., spec, [], t_min, t_max,
    )
    if case.get('perturb_seed'):
        rng = np.random.default_rng(case['seed'])
        for j in range(stages):
            fractions = rng.dirichlet(np.ones(len(COMPONENTS)) * 3.)
            initial['x'][j] = dict(zip(COMPONENTS, map(float, fractions)))
            initial['T'][j] = unit._bubble_temperature_from_equation(
                initial['x'][j], 1., t_min, t_max,
            )
    packed = unit._pack_variables(
        initial['T'], initial['x'], initial['L'], initial['V'],
        initial['Q_cond'], initial['Q_reb'], COMPONENTS, t_min, t_max,
        energy_scale,
    )
    size = len(packed)
    direction = -1. if case.get('reverse') else 1.
    guess = np.concatenate([packed, np.full(nr, direction * .25 / nr)])
    source_pattern = lil_matrix((size, nr))
    closure_pattern = lil_matrix((nr, size))
    for k, j in enumerate(active):
        for i, c in enumerate(COMPONENTS):
            if nu.get(c, 0.):
                source_pattern[j * (len(COMPONENTS) + 2) + i, k] = 1
        closure_pattern[k, j] = 1
        start = stages + j * (len(COMPONENTS) - 1)
        closure_pattern[k, start:start + len(COMPONENTS) - 1] = 1
    pattern = bmat([
        [base['sparsity'], source_pattern],
        [closure_pattern, scipy.sparse.eye(nr)],
    ], format='csr')
    strength = [1.]

    def reaction_values(decoded, j):
        temperature = float(decoded['T'][j])
        composition = decoded['x'][j]
        a = thermo.component_activities(temperature, 1., composition, 'liquid')
        logq = sum(v * math.log(max(a[c], 1e-300)) for c, v in nu.items())
        logk = thermo.reaction_log_equilibrium_constant(nu, temperature)
        return temperature, composition, logq - logk

    def closures_for(vector, decoded):
        closures = []
        for k, j in enumerate(active):
            temperature, composition, affinity = reaction_values(decoded, j)
            if case['closure'] == 'equilibrium':
                closures.append((1. - strength[0]) * vector[size + k]
                                + strength[0] * affinity)
            else:
                state = HomogeneousRateState.from_flows(
                    thermo, composition, temperature, 1., 'liquid',
                )
                rate = evaluate_reaction_rate(kinetic, state, thermo)
                closures.append(vector[size + k] - strength[0] * case['holdup'] * rate / flow)
        return np.array(closures)

    def residual(vector):
        values = base['residual'](vector[:size]).copy()
        decoded = base['decode'](vector[:size])
        extents = vector[size:] * flow
        for k, j in enumerate(active):
            for i, c in enumerate(COMPONENTS):
                values[j * (len(COMPONENTS) + 2) + i] += (
                    nu.get(c, 0.) * extents[k] / scales[c]
                )
        return np.concatenate([values, closures_for(vector, decoded)])

    def augmented_jacobian(vector, _f0, rel_step):
        """Keep the production MESH Jacobian, append local reaction blocks."""
        base_f = base['residual'](vector[:size])
        core_jacobian, evaluations, name = base['jacobian'](vector[:size], base_f, rel_step)
        decoded = base['decode'](vector[:size])
        closure0 = closures_for(vector, decoded)
        lower_left = lil_matrix((nr, size))
        upper_right = source_pattern.copy()
        for k, j in enumerate(active):
            for i, c in enumerate(COMPONENTS):
                upper_right[j * (len(COMPONENTS) + 2) + i, k] = nu[c] * flow / scales[c]
            for col in closure_pattern.rows[k]:
                step = rel_step * max(abs(vector[col]), 1.)
                trial = vector.copy()
                trial[col] += step
                shifted = closures_for(trial, base['decode'](trial[:size]))
                lower_left[k, col] = (shifted[k] - closure0[k]) / step
                evaluations += 1
        extent_diagonal = (1. - strength[0]) if case['closure'] == 'equilibrium' else 1.
        assembled = bmat([[core_jacobian, upper_right],
                          [lower_left, extent_diagonal * scipy.sparse.eye(nr)]], format='csr')
        return assembled, evaluations + 1, name + '_reaction_blocks'

    solver_options = {
        'mesh_tolerance': TOLERANCE, 'acceptable_mesh_residual': TOLERANCE,
        'max_iterations': 70, 'max_jacobian_evaluations': 70,
        'line_search_steps': 18, 'finite_difference_rel_step': 1e-6,
        'stall_iterations': 8, 'stall_relative_tolerance': 1e-5,
    }
    schedule = ([1.] if case['strategy'] == 'direct' else
                [.02, .05, .1, .25, .5, .75, 1.])
    history = []
    for multiplier in schedule:
        strength[0] = multiplier
        tick = time.perf_counter()
        solved = unit._sparse_newton_solve(
            residual, pattern, guess, solver_options,
            jacobian=augmented_jacobian if case.get('jacobian') == 'augmented' else None,
        )
        history.append({
            'strength': multiplier, 'seconds': time.perf_counter() - tick,
            **{k: solved[k] for k in ('success', 'residual_norm', 'iterations',
                                      'function_evaluations', 'jacobian_evaluations', 'message')},
        })
        guess = solved['x']
        if not solved['success']:
            break
    decoded = base['decode'](guess[:size])
    extents = guess[size:] * flow
    total_extent = float(np.sum(extents))
    distillate = float(decoded['V'][0])
    bottoms = float(decoded['L'][-1])
    output_flows = {
        c: distillate * decoded['x'][0][c] + bottoms * decoded['x'][-1][c]
        for c in COMPONENTS
    }
    component_error = max(abs(output_flows[c] - flow * z[c] - nu[c] * total_extent)
                          for c in COMPONENTS)
    elements = {c: parse_formula_counts(metadata[c].formula) for c in COMPONENTS}
    element_errors = {
        atom: sum(elements[c].get(atom, 0) * (output_flows[c] - flow * z[c])
                  for c in COMPONENTS)
        for atom in sorted({a for formula in elements.values() for a in formula})
    }
    hout = (distillate * thermo.mixture_enthalpy(decoded['x'][0], float(decoded['T'][0]), 0., P=1.)
            + bottoms * thermo.mixture_enthalpy(decoded['x'][-1], float(decoded['T'][-1]), 0., P=1.))
    energy_error = hout - flow * feed.H - decoded['Q_cond'] - decoded['Q_reb']
    spinodal = getattr(thermo, 'liquid_spinodal_stability', None)
    unstable = []
    profile = []
    for j in range(stages):
        xi = float(extents[active.index(j)]) if j in active else 0.
        affinity = reaction_values(decoded, j)[2] if j in active else None
        stability = spinodal(float(decoded['T'][j]), decoded['x'][j]) if callable(spinodal) else None
        if stability is not None and not stability['locally_stable']:
            unstable.append(j + 1)
        profile.append({
            'stage': j + 1, 'T_K': float(decoded['T'][j]), 'x': decoded['x'][j],
            'L_kmol_h': float(decoded['L'][j]), 'V_kmol_h': float(decoded['V'][j]),
            'extent_kmol_h': xi, 'ln_Q_over_K': affinity,
            'locally_stable': None if stability is None else bool(stability['locally_stable']),
        })
    # Scalar liquid equilibrium comparator uses the identical reaction reference.
    from scipy.optimize import brentq

    def batch_residual(extent):
        composition = {c: z[c] + nu[c] * extent / flow for c in COMPONENTS}
        activities = thermo.component_activities(feed.T, 1., composition, 'liquid')
        return (sum(nu[c] * math.log(max(activities[c], 1e-300)) for c in COMPONENTS)
                - thermo.reaction_log_equilibrium_constant(nu, feed.T))

    lo, hi = (-50. + 1e-10, -1e-10) if case.get('reverse') else (1e-10, 50. - 1e-10)
    batch_extent = brentq(batch_residual, lo, hi, xtol=1e-11)
    references = []
    for temperature in (330., 350., 370., 390.):
        logk = thermo.reaction_log_equilibrium_constant(nu, temperature)
        quarter = {c: .25 for c in COMPONENTS}
        activities = thermo.component_activities(temperature, 1., quarter, 'liquid')
        gammas = (thermo.activity_coefficients(temperature, quarter)
                  if method != 'IDEAL' else {c: 1. for c in COMPONENTS})
        log_reference = sum(nu[c] * math.log(activities[c] / (.25 * gammas[c])) for c in COMPONENTS)
        references.append({'T_K': temperature, 'K_gas_standard': math.exp(logk),
                           'K_pure_liquid_standard': math.exp(logk - log_reference)})
    success = bool(solved['success'] and strength[0] == 1.)
    derivative_audit = None
    if case.get('jacobian') == 'augmented':
        analytic = augmented_jacobian(guess, residual(guess), 1e-6)[0].toarray()
        reference = np.zeros_like(analytic)
        for col in range(len(guess)):
            step = 2e-6 * max(abs(guess[col]), 1.)
            plus, minus = guess.copy(), guess.copy()
            plus[col] += step
            minus[col] -= step
            reference[:, col] = (residual(plus) - residual(minus)) / (2. * step)
        derivative_audit = {
            'max_abs_error': float(np.max(np.abs(analytic - reference))),
            'max_relative_scaled_error': float(np.max(np.abs(analytic - reference) / (1. + np.abs(reference)))),
            'max_derivative_outside_sparsity': float(np.max(np.abs(reference[pattern.toarray() == 0]))),
        }
    return {
        'case': case, 'success': success, 'seconds': time.perf_counter() - started,
        'history': history, 'variables': len(guess), 'sparsity_nnz': int(pattern.nnz),
        'residual_inf': float(np.max(np.abs(residual(guess)))),
        'extent_kmol_h': total_extent,
        'reactant_conversion_percent': 100. * total_extent / 50.,
        'batch_extent_at_350K_kmol_h': batch_extent,
        'distillate_kmol_h': distillate, 'bottoms_kmol_h': bottoms,
        'distillate_composition': decoded['x'][0], 'bottoms_composition': decoded['x'][-1],
        'Q_cond_kW': decoded['Q_cond'] / 3600., 'Q_reb_kW': decoded['Q_reb'] / 3600.,
        'component_balance_max_kmol_h': component_error,
        'element_balance_errors_kmol_atoms_h': element_errors,
        'energy_balance_error_kW': energy_error / 3600.,
        'molar_balance_error_kmol_h': distillate + bottoms - flow,
        'rounded_MW_mass_balance_error_kg_h': sum(
            metadata[c].MW * (output_flows[c] - flow * z[c]) for c in COMPONENTS),
        'unstable_liquid_stages': unstable, 'global_phase_stability_checked': False,
        'profile': profile, 'equilibrium_references': references,
        'reaction_validation_warnings': list(equilibrium.validation_warnings),
        'supporting_property_estimates': supporting_estimates,
        'jacobian_audit': derivative_audit, 'feed_specs': feed_specs,
    }


def worker(case, result_path, log_path):
    with log_path.open('x') as stream, redirect_stdout(stream), redirect_stderr(stream):
        try:
            with isolated_runtime_caches():
                result = make_case(case)
        except Exception:
            result = {'case': case, 'success': False, 'error': traceback.format_exc()}
        write_json(result_path, result)


def run_case(case, output, timeout):
    context = multiprocessing.get_context('spawn')
    result_path = output / (case['name'] + '.json')
    log_path = output / (case['name'] + '.log')
    process = context.Process(target=worker, args=(case, result_path, log_path))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        result = {'case': case, 'success': False, 'error': f'Case exceeded {timeout} seconds'}
        # The termination is deliberate and no replacement solve is launched.
        if not result_path.exists():
            write_json(result_path, result)
        return result
    if result_path.exists():
        return json.loads(result_path.read_text())
    result = {'case': case, 'success': False, 'error': f'Worker exit code {process.exitcode}'}
    write_json(result_path, result)
    return result


def cases_for(suite):
    cases = []
    methods = ['IDEAL', 'NRTL', 'NRTL-VDM', 'NRTL-BV']
    for method in methods:
        for closure in ('kinetic', 'equilibrium'):
            cases.append({'name': f'{method.lower()}-{closure}-direct', 'method': method,
                          'closure': closure, 'strategy': 'direct', 'stages': 6, 'holdup': 1.,
                          'estimate_hoc_geometry': method == 'NRTL-BV'})
    if suite == 'extended':
        for method in methods:
            for closure in ('kinetic', 'equilibrium'):
                cases.append({'name': f'{method.lower()}-{closure}-continuation', 'method': method,
                              'closure': closure, 'strategy': 'continuation', 'stages': 6, 'holdup': 1.,
                              'estimate_hoc_geometry': method == 'NRTL-BV'})
        for holdup in (.1, 10.):
            cases.append({'name': f'ideal-kinetic-holdup-{holdup}', 'method': 'IDEAL',
                          'closure': 'kinetic', 'strategy': 'direct', 'stages': 6, 'holdup': holdup})
        for stages in (10, 20):
            cases.append({'name': f'ideal-equilibrium-stages-{stages}', 'method': 'IDEAL',
                          'closure': 'equilibrium', 'strategy': 'direct', 'stages': stages, 'holdup': 1.})
        for seed in (17, 42, 123):
            cases.append({'name': f'ideal-equilibrium-seed-{seed}', 'method': 'IDEAL',
                          'closure': 'equilibrium', 'strategy': 'direct', 'stages': 6, 'holdup': 1.,
                          'perturb_seed': True, 'seed': seed})
        cases.append({'name': 'ideal-equilibrium-hydrolysis', 'method': 'IDEAL',
                      'closure': 'equilibrium', 'strategy': 'direct', 'stages': 6, 'holdup': 1.,
                      'reverse': True})
        for method in ('NRTL', 'NRTL-BV'):
            defaults = {'method': method, 'closure': 'equilibrium', 'strategy': 'direct',
                        'stages': 6, 'holdup': 1., 'estimate_hoc_geometry': method == 'NRTL-BV'}
            for seed in (17, 42):
                cases.append({**defaults, 'name': f'{method.lower()}-equilibrium-seed-{seed}',
                              'perturb_seed': True, 'seed': seed})
            cases.append({**defaults, 'name': f'{method.lower()}-equilibrium-stages-20', 'stages': 20})
            cases.append({**defaults, 'name': f'{method.lower()}-equilibrium-split-feeds',
                          'stages': 10, 'split_feeds': True})
            for closure in ('kinetic', 'equilibrium'):
                cases.append({**defaults, 'name': f'{method.lower()}-{closure}-augmented',
                              'closure': closure, 'jacobian': 'augmented'})
        for distillate in (40., 60.):
            cases.append({'name': f'nrtl-equilibrium-distillate-{distillate}', 'method': 'NRTL',
                          'closure': 'equilibrium', 'strategy': 'direct', 'stages': 6,
                          'holdup': 1., 'distillate': distillate})
    return cases


def audit_saved_results(directories):
    """Audit already solved profiles; never repeat the column calculation."""
    saved = []
    for directory in directories:
        saved.extend(r for r in json.loads((directory / 'summary.json').read_text()) if r['success'])
    records = []
    for method in sorted({r['case']['method'] for r in saved}):
        selected = [r for r in saved if r['case']['method'] == method]
        with isolated_runtime_caches():
            thermo, estimates = make_thermodynamics(selected[0]['case'])
            reaction = equilibrium_reaction_from_mapping({'equation': EQUATION}, COMPONENTS)
            nu = reaction.reaction.stoichiometry

            def pure_liquid_logk(t):
                a = thermo.component_activities(t, 1., {c: .25 for c in COMPONENTS}, 'liquid')
                gamma = (thermo.activity_coefficients(t, {c: .25 for c in COMPONENTS})
                         if method != 'IDEAL' else {c: 1. for c in COMPONENTS})
                return (thermo.reaction_log_equilibrium_constant(nu, t)
                        - sum(nu[c] * math.log(a[c] / (.25 * gamma[c])) for c in COMPONENTS))

            t = 350.
            slope = (pure_liquid_logk(t + .01) - pure_liquid_logk(t - .01)) / .02
            delta_h_reference = 8.31446261815324 * t**2 * slope
            delta_h_stream = sum(nu[c] * thermo.mixture_enthalpy({c: 1.}, t, 0., P=1.)
                                 for c in COMPONENTS)
            split_profiles = []
            for result in selected:
                splits = []
                phase_activity_errors = []
                liquid_reference_errors = []
                authoritative_affinities = []
                for stage in result['profile']:
                    temperature, x = stage['T_K'], stage['x']
                    kvals = (thermo.K_values(temperature, 1., x)
                             if hasattr(thermo, 'K_values') else
                             {c: thermo.K_value(c, temperature, 1.) for c in COMPONENTS})
                    yraw = {c: x[c] * kvals[c] for c in COMPONENTS}
                    y = {c: yraw[c] / sum(yraw.values()) for c in COMPONENTS}
                    liquid = thermo.component_activities(temperature, 1., x, 'liquid')
                    vapor = thermo.component_activities(temperature, 1., y, 'vapor')
                    phase_activity_errors.append(max(abs(math.log(liquid[c] / vapor[c]))
                                                     for c in COMPONENTS))
                    if method != 'IDEAL':
                        gamma = thermo.activity_coefficients(temperature, x)
                        refs = thermo._liquid_fugacity_reference_factors(temperature, 1.)
                        reference_a = {c: x[c] * gamma[c] * refs[c] for c in COMPONENTS}
                        liquid_reference_errors.append(max(abs(math.log(liquid[c] / reference_a[c]))
                                                           for c in COMPONENTS))
                        if stage['ln_Q_over_K'] is not None:
                            authoritative_affinities.append(
                                sum(nu[c] * math.log(reference_a[c]) for c in COMPONENTS)
                                - thermo.reaction_log_equilibrium_constant(nu, temperature))
                if method != 'IDEAL':
                    for stage in result['profile']:
                        two, x1, x2, beta = thermo.liquid_liquid_equilibrium(
                            stage['x'], stage['T_K'], max_iter=150, tol=1e-8,
                        )
                        if two:
                            splits.append({'stage': stage['stage'], 'x1': x1, 'x2': x2, 'beta': beta})
                split_profiles.append({
                    'case': result['case']['name'], 'liquid_splits': splits,
                    'max_log_liquid_vapor_reaction_activity_mismatch': max(phase_activity_errors),
                    'max_log_liquid_reaction_vs_vle_reference_mismatch': max(liquid_reference_errors, default=0.),
                    'reaction_ln_Q_over_K_using_vle_liquid_reference': authoritative_affinities,
                })
            pair_records = []
            for i, first in enumerate(COMPONENTS):
                for second in COMPONENTS[i + 1:]:
                    record = getattr(thermo, '_database_activity_interactions', {}).get((first, second))
                    if method != 'IDEAL':
                        pair_records.append({'components': [first, second], 'record': record})
            records.append({
                'method': method, 'T_K': t, 'K_pure_liquid': math.exp(pure_liquid_logk(t)),
                'van_t_Hoff_delta_H_kJ_kmol': delta_h_reference,
                'stream_delta_H_kJ_kmol': delta_h_stream,
                'delta_H_discrepancy_kJ_kmol': delta_h_stream - delta_h_reference,
                'supporting_estimates': estimates, 'profile_lle_checks': split_profiles,
                'interaction_records': pair_records, 'thermo_warnings': list(thermo.warnings),
            })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3, choices=range(1, 6))
    parser.add_argument('--timeout', type=float, default=90.)
    parser.add_argument('--suite', choices=('basic', 'extended'), default='basic')
    parser.add_argument('--cases', nargs='+', help='Run only these case names from the selected suite')
    parser.add_argument('--audit-results', nargs='+', type=Path,
                        help='Audit saved result directories without solving columns again')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {'numpy': np.__version__, 'scipy': scipy.__version__, 'sources': {}}
    for filename in ('scripts/probe_reactive_distillation.py', 'equilibrium_stage_column.py',
                     'unit_operations_distillation.py', 'reaction_models.py', 'kinetic_models.py',
                     'thermodynamics_models/base.py', 'thermodynamics_models/activity.py'):
        provenance['sources'][filename] = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
    write_json(args.output / 'provenance.json', provenance)
    if args.audit_results:
        audits = audit_saved_results(args.audit_results)
        write_json(args.output / 'thermodynamic_audit.json', audits)
        for record in audits:
            print(json.dumps({key: record[key] for key in (
                'method', 'K_pure_liquid', 'van_t_Hoff_delta_H_kJ_kmol',
                'stream_delta_H_kJ_kmol', 'delta_H_discrepancy_kJ_kmol',
            )}), flush=True)
        return
    results = []
    cases = cases_for(args.suite)
    if args.cases:
        unknown = set(args.cases) - {case['name'] for case in cases}
        if unknown:
            parser.error('Unknown case names: ' + ', '.join(sorted(unknown)))
        cases = [case for case in cases if case['name'] in args.cases]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_case, case, args.output, args.timeout): case
                   for case in cases}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({key: result.get(key) for key in (
                'case', 'success', 'seconds', 'residual_inf', 'reactant_conversion_percent',
                'component_balance_max_kmol_h', 'energy_balance_error_kW',
                'unstable_liquid_stages', 'error',
            )}), flush=True)
    results.sort(key=lambda result: result['case']['name'])
    write_json(args.output / 'summary.json', results)
    print(f"Completed {len(results)} cases; {sum(r['success'] for r in results)} numerical successes.", flush=True)


if __name__ == '__main__':
    main()
