"""Molecule-held-out solubility transfers from solvent offsets and cLogP slopes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_solubility_transfer import METHODS, SCOPES, write_csv
from benchmark_solvent_transfer_multipliers import (
    averages, contributions, difference_bootstrap, fit, reconstruct,
)
from solubility_paths import DATA_DIR, RESULTS_DIR, STUDY_DIR, verify_manifest, write_manifest


def accumulate_logp(blocks, logp, folds, n_solvents, n_folds):
    """Compact sufficient statistics preserve the original group/molecule weights."""
    matrices = np.zeros((n_folds, 3, n_solvents, n_solvents))
    vectors = np.zeros((n_folds, 2, n_solvents))
    for solute, indices, normal, rhs in blocks:
        if not np.isfinite(logp[solute]):
            continue
        fold, value = folds[solute], logp[solute]
        for power in range(3):
            matrices[fold, power][np.ix_(indices, indices)] += normal * value**power
        vectors[fold, 0, indices] += rhs
        vectors[fold, 1, indices] += rhs * value
    return matrices, vectors


def slope_support(training, logp, folds, held_out, n_solvents, minimum):
    """Count distinct training identities, never measurements or solvent pairs."""
    identities = [set() for _ in range(n_solvents)]
    for solute, indices, _ in training:
        if folds[solute] != held_out and np.isfinite(logp[solute]):
            for solvent in indices:
                identities[solvent].add(solute)
    counts = np.array([len(group) for group in identities])
    return counts, counts >= minimum


def fit_logp(matrices, vectors, mean, scale, anchor, slope_enabled=None):
    """Fit intercept/slope differences and retain nullspaces for estimability checks.

    Each connected component has both anchor coefficients fixed to zero.
    Diagonal equilibration improves conditioning. A singular fit may still
    identify some prediction contrasts; only those contrasts are accepted.
    """
    a0, a1, a2 = matrices
    b0, b1 = vectors
    cross = (a1 - mean * a0) / scale
    square = (a2 - 2 * mean * a1 + mean**2 * a0) / scale**2
    normal = np.block([[a0, cross], [cross, square]])
    rhs = np.concatenate([b0, (b1 - mean * b0) / scale])
    n = len(a0)
    if slope_enabled is None:
        slope_enabled = np.ones(n, dtype=bool)
    active = np.diag(a0) > 1e-12
    adjacency = np.abs(a0) > 1e-12
    np.fill_diagonal(adjacency, False)
    _, components = connected_components(adjacency, directed=False)
    components[~active] = -1
    beta = np.full((2, n), np.nan)
    fitted = []
    maximum_residual = 0.
    for component in np.unique(components[active]):
        indices = np.flatnonzero(components == component)
        fixed = anchor if anchor in indices else indices[0]
        free_solvents = indices[indices != fixed]
        free_slopes = free_solvents[slope_enabled[free_solvents]]
        free = np.concatenate([free_solvents, free_slopes + n])
        a = normal[np.ix_(free, free)]
        a = (a + a.T) / 2
        diagonal = np.sqrt(np.maximum(np.diag(a), 0))
        diagonal[diagonal < 1e-12] = 1.
        scaled = a / diagonal[:, None] / diagonal[None, :]
        values, basis = eigh(scaled, check_finite=False)
        tolerance = np.finfo(float).eps * max(len(free), 1) * 100 * max(float(values[-1]), 1.)
        if values[0] < -10 * tolerance:
            raise ArithmeticError('Negative eigenvalue in weighted least-squares normal matrix')
        identified = values > tolerance
        r = rhs[free] / diagonal
        solution = basis[:, identified] @ ((basis[:, identified].T @ r) / values[identified])
        coefficient = solution / diagonal
        residual = float(np.max(np.abs(a @ coefficient - rhs[free]))) / max(1., float(np.max(np.abs(rhs[free]))))
        maximum_residual = max(maximum_residual, residual)
        if residual > 1e-7:
            raise ArithmeticError(f'Normal-equation relative residual {residual:g}')
        flat = np.zeros(2 * n)
        flat[free] = coefficient
        beta[:, indices] = flat.reshape(2, n)[:, indices]
        null = np.zeros((2 * n, int((~identified).sum())))
        null[free] = basis[:, ~identified] / diagonal[:, None]
        qscale = np.zeros(2 * n)
        qscale[free] = 1 / diagonal
        fitted.append(dict(component=int(component), rank=int(identified.sum()), free_parameters=len(free),
                           null=null.reshape(2, n, -1), qscale=qscale.reshape(2, n), anchor=int(fixed)))
    return dict(beta=beta, components=components, fitted=fitted, mean=mean, scale=scale,
                slope_enabled=slope_enabled, maximum_relative_residual=maximum_residual)


def predict_logp(model, reference, target, logp):
    """Return only transfer contrasts identified by the training design."""
    components = model['components']
    prediction = np.full(len(reference), np.nan)
    z = (logp - model['mean']) / model['scale']
    for part in model['fitted']:
        selected = np.flatnonzero((components[reference] == part['component'])
                                  & (components[target] == part['component']) & np.isfinite(z))
        if not len(selected):
            continue
        r, t, v = reference[selected], target[selected], z[selected]
        null, qscale = part['null'], part['qscale']
        if null.shape[2]:
            projection = null[0, t] - null[0, r] + v[:, None] * (null[1, t] - null[1, r])
            magnitude = np.sqrt(qscale[0, t]**2 + qscale[0, r]**2
                                + v*v*(qscale[1, t]**2 + qscale[1, r]**2))
            estimable = np.linalg.norm(projection, axis=1) <= 1e-8 * np.maximum(1., magnitude)
            selected, r, t, v = selected[estimable], r[estimable], t[estimable], v[estimable]
        beta = model['beta']
        prediction[selected] = beta[0, t] - beta[0, r] + v*(beta[1, t] - beta[1, r])
    return prediction


def metrics(error, cases):
    """Equal-molecule metrics, including weighted predictive R² and correlation."""
    result, _ = averages(error, cases['solute'])
    _, inverse, counts = np.unique(cases['solute'], return_inverse=True, return_counts=True)
    weight = 1 / counts[inverse]
    weight /= weight.sum()
    y = cases['log_target_x']
    p = y[:, None] + error
    ymean = float(weight @ y)
    pmean = float(np.mean(weight @ p))
    variance = float(weight @ ((y - ymean)**2))
    pvariance = float(np.mean(weight @ ((p - pmean)**2)))
    covariance = float(np.mean(weight @ ((y - ymean)[:, None]*(p - pmean))))
    result['R2'] = 1 - result['RMSE_log10']**2 / variance if variance else None
    result['Pearson_r'] = covariance / np.sqrt(variance*pvariance) if variance*pvariance > 0 else None
    return result


def comparison_job(task):
    name, first, second, solutes, seed, draws, family = task
    _, a = averages(first, solutes)
    _, b = averages(second, solutes)
    result = difference_bootstrap(a - b, seed, draws, requested_family_size=family)
    return dict(comparison=name, difference=result['UNIFAC_minus_baseline_MAE_log10'],
                CI95=result['paired_95pct_CI'], familywise_CI95=result['paired_requested_family_95pct_CI'])


def render(output, summary):
    lines = ['# Solubility transfer: solvent offsets plus computed logP', '',
             'Fit two coefficient types per solvent: offset a_s and cLogP slope b_s. '
             'For a held-out solute i, log10(x_target / x_reference) = '
             '(a_target − a_reference) + (b_target − b_reference) cLogP_i. '
             'A single global solute logP term would cancel in this transfer.', '',
             '## Protocol', '',
             '- Reuse the multiplier benchmark’s exact molecule identities, cases, five folds '
             'and ten repeats. Match temperature and publication at 20–30 °C for the main training '
             'groups; all measurements of a molecule stay together in each split.',
             '- Fit least squares with the same equal-molecule, equal-group, equal-pair weighting '
             'as the multiplier benchmark. No ridge penalty, tuning, extra descriptors or fitted solid terms.',
             f'- A solvent slope is fixed to ZERO unless at least {summary["minimum_molecules"]} '
             'distinct training molecules were measured in that solvent. Repeated temperatures, '
             'publications and pairs cannot inflate support. Counts and eligibility are recalculated '
             'within each outer training fold. The offset remains fitted for every connected observed '
             'solvent. The threshold is a declared support rule, not selected from test errors.',
             '- cLogP is RDKit Wildman–Crippen MolLogP computed from the full standard InChI '
             'used for the original molecular identity. No measured logP, target solubility, '
             'ionization correction, largest-fragment substitution or learned molecular descriptor.',
             '- Center and scale cLogP using distinct training molecules only. Fix both water '
             'coefficients to zero in its connected component; anchor other components separately. '
             'Zero slopes for sparse solvents mean no logP dependence relative to that anchor; '
             'this is an explicit simplifying assumption, not a claim about their physical behavior.',
             '- Use a rank-revealing, diagonally equilibrated eigensolve. Reject disconnected, '
             'unseen-solvent, nonfinite-descriptor, or nullspace-dependent prediction contrasts. '
             'A connected graph alone does not identify solvent slopes. No arbitrary minimum-norm '
             'extrapolations are accepted where the transfer itself is not identifiable.',
             '- Refit the multiplier-only baseline on identical training groups, and reconcile '
             'it with the old held-out baseline when all cLogP descriptors are available. '
             'Successful UNIFAC predictions are reused without rerunning thermodynamics.',
             '- Main predicted log mole fractions are capped at zero (x ≤ 1), matching the old '
             'multiplier baseline. Literal uncapped errors and cap counts are retained; there '
             'is no arbitrary lower-solubility cutoff or clipping of large negative predictions.',
             '- Every score and test compares identical cases, requiring availability in every '
             'repeat. Score individual held-out losses, then weight each molecule equally. '
             'R² and Pearson r use those same molecule weights, not a fit through the plot.', '',
             '## Main and directional multiplier comparisons', '',
             '| Population | Model | Transfers | Molecules | MAE | RMSE | R² | Pearson r | Within ×10 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for population, models in summary['baseline_scores'].items():
        for name, row in models.items():
            lines.append(f'| {population} | {name} | {row["predictions"]:,} | {row["solutes"]:,} | '
                         f'{row["MAE_log10"]:.4f} | {row["RMSE_log10"]:.4f} | {row["R2"]:.4f} | '
                         f'{row["Pearson_r"]:.4f} | {100*row["within_factor_10"]:.1f}% |')
    lines += ['', '## UNIFAC on its own matched coverage', '',
              '| Variant | Transfers | Molecules | UNIFAC MAE | Multiplier MAE | Multiplier + logP MAE |',
              '|---|---:|---:|---:|---:|---:|']
    for name, row in summary['unifac_scores']['own_coverage'].items():
        u, b, p = row['UNIFAC'], row['multiplier'], row['multiplier_logP']
        lines.append(f'| {name} | {u["predictions"]:,} | {u["solutes"]} | '
                     f'{u["MAE_log10"]:.4f} | {b["MAE_log10"]:.4f} | {p["MAE_log10"]:.4f} |')
    for cohort in ('common_five', 'common_six'):
        lines += ['', f'## Identical-case cohort: {cohort}', '',
                  '| Variant | Transfers | Molecules | UNIFAC MAE | Multiplier MAE | Multiplier + logP MAE |',
                  '|---|---:|---:|---:|---:|---:|']
        for name, row in summary['unifac_scores'][cohort].items():
            u, b, p = row['UNIFAC'], row['multiplier'], row['multiplier_logP']
            lines.append(f'| {name} | {u["predictions"]:,} | {u["solutes"]} | '
                         f'{u["MAE_log10"]:.4f} | {b["MAE_log10"]:.4f} | {p["MAE_log10"]:.4f} |')
    lines += ['', '## Paired molecule/repetition uncertainty', '',
              'Differences use the order stated in each comparison name. '
              'The simultaneous intervals cover the seven primary comparisons: the new versus '
              'old baseline overall, and each of six UNIFAC variants versus the new baseline '
              'on its own matched coverage. Shared-cohort and directional scores are descriptive.', '',
              '| Comparison | MAE difference | Ordinary 95% CI | Familywise 95% CI |',
              '|---|---:|---|---|']
    for row in summary['paired']:
        lines.append(f'| {row["comparison"]} | {row["difference"]:+.4f} | '
                     f'{row["CI95"]} | {row["familywise_CI95"]} |')
    lines += ['', '## Audit', '', '```json', json.dumps(summary['audit'], indent=2), '```', '',
              '## Interpretation and reproducibility', '',
              'This predicts target solubility given a measured reference solubility; it is not '
              'an absolute aqueous-solubility regression. The scalar logP feature can express '
              'solute-dependent solvent preferences but cannot separately identify hydrogen bonding, '
              'specific functional groups, crystal forms, or aqueous ionization. Random molecular '
              'splits do not establish scaffold-held-out or external-dataset performance. '
              'Publication/temperature matching cannot guarantee identical solid form. '
              'Pooled-publication and AqSolDB-anchor scopes are separately labeled sensitivities; '
              'AqSolDB temperatures, pH, density conversion and species remain approximate.', '',
              'The paired bootstrap conditions on the fitted predictions and resamples molecules '
              f'and split repetitions ({summary["bootstrap_draws"]:,} draws, seed {summary["seed"]}). '
              'It does not include a complete refitting bootstrap.', '',
              '- errors_capped.npz / errors_uncapped.npz: every case/repeat prediction error.',
              '- baseline_errors.npz: the exactly training-matched multiplier-only errors.',
              '- fold_parameters.csv: per-fold solvent offsets/slopes, anchors and training scaling.',
              '- fold_audit.csv: numerical rank, support, extrapolation refusals and cap counts.',
              '- molecules.csv: exact identities, resolved structures and computed logP.',
              '- summary.json: full headline metrics, shared-case results and paired uncertainty.',
              '- manifest.json: input, implementation and output fingerprints; RDKit version.',
              '- No production code, datasets or previous predictions were changed.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR / 'unifac-transfer-results')
    parser.add_argument('--baseline-dir', type=Path, default=RESULTS_DIR / 'solvent-multiplier-results')
    parser.add_argument('--output-dir', type=Path, default=RESULTS_DIR / 'logp-transfer-supported-results')
    parser.add_argument('--minimum-molecules', type=int, default=20,
                        help='Distinct training molecules needed per fitted solvent slope; zero reproduces unrestricted fitting')
    parser.add_argument('--draws', type=int, default=20000)
    args = parser.parse_args()
    if args.draws < 1000:
        parser.error('Use at least 1000 bootstrap draws')
    if args.minimum_molecules < 0:
        parser.error('Minimum molecule support cannot be negative')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    started = time.monotonic()
    rdBase.DisableLog('rdApp.*')
    verify_manifest(args.baseline_dir / 'manifest.json')
    previous = json.loads((args.baseline_dir / 'summary.json').read_text())
    cases, training, records, molecules, solvents, data_audit = reconstruct(args.data_dir, args.results_dir)
    print('Original cases reconstructed and all UNIFAC case identities verified', flush=True)
    n_solutes, n_solvents = len(molecules), len(solvents)
    logp = np.full(n_solutes, np.nan)
    for row in molecules:
        molecule = Chem.MolFromInchi(row['inchi'])
        row['smiles'] = Chem.MolToSmiles(molecule) if molecule is not None else ''
        row['descriptor_status'] = 'ok'
        try:
            if molecule is None:
                raise ValueError('Could not reconstruct standard InChI')
            logp[row['index']] = Crippen.MolLogP(molecule)
            if not np.isfinite(logp[row['index']]):
                raise ValueError('Nonfinite computed logP')
        except (ValueError, RuntimeError) as error:
            row['descriptor_status'] = str(error)
        row['computed_logP'] = float(logp[row['index']]) if np.isfinite(logp[row['index']]) else ''
    repeats, n_folds, seed = previous['repeats'], previous['folds'], previous['seed']
    folds = np.full((repeats, n_solutes), -1, dtype=int)
    with (args.baseline_dir / 'folds.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            index = int(row['index'])
            if molecules[index]['inchi'] != row['inchi']:
                raise ValueError('Original fold identity mismatch')
            folds[int(row['repeat']), index] = int(row['fold'])
    if np.any(folds < 0):
        raise ValueError('Incomplete saved molecular folds')
    with np.load(args.baseline_dir / 'baseline_errors.npz') as saved:
        if not np.array_equal(saved['case'], cases['case']):
            raise ValueError('Baseline case mismatch')
        old_errors = saved['capped']
    blocks = contributions(training)
    anchor = next(row['index'] for row in solvents if row['name'] == 'water')
    errors = np.full((len(cases), repeats), np.nan)
    raw = np.full_like(errors, np.nan)
    baseline = np.full_like(errors, np.nan)
    parameters, audits = [], []
    training_solutes = np.array(sorted({solute for solute, _, _ in training}))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    for repeat in range(repeats):
        matrices, vectors = accumulate_logp(blocks, logp, folds[repeat], n_solvents, n_folds)
        all_a, all_b = matrices.sum(axis=0), vectors.sum(axis=0)
        for fold in range(n_folds):
            a, b = all_a - matrices[fold], all_b - vectors[fold]
            train = training_solutes[(folds[repeat, training_solutes] != fold) & np.isfinite(logp[training_solutes])]
            mean, scale = float(logp[train].mean()), max(float(logp[train].std()), 1e-8)
            support_counts, slope_enabled = slope_support(training, logp, folds[repeat], fold,
                                                           n_solvents, args.minimum_molecules)
            model = fit_logp(a, b, mean, scale, anchor, slope_enabled)
            beta, components, base_residual = fit(a[0].copy(), b[0].copy(), anchor)
            positions = np.flatnonzero(folds[repeat, cases['solute']] == fold)
            selected = cases[positions]
            ref, target = selected['reference'], selected['target']
            shift = predict_logp(model, ref, target, logp[selected['solute']])
            prediction = selected['log_reference_x'] + shift
            raw[positions, repeat] = prediction - selected['log_target_x']
            errors[positions, repeat] = np.minimum(prediction, 0) - selected['log_target_x']
            supported = (components[ref] >= 0) & (components[ref] == components[target])
            bprediction = selected['log_reference_x'][supported] + beta[target[supported]] - beta[ref[supported]]
            baseline[positions[supported], repeat] = np.minimum(bprediction, 0) - selected['log_target_x'][supported]
            graph_supported = (model['components'][ref] >= 0) & (model['components'][ref] == model['components'][target])
            audits.append(dict(repeat=repeat, fold=fold, training_molecules=len(train),
                               test_cases=len(positions), supported_cases=int(np.isfinite(shift).sum()),
                               graph_supported_but_unidentifiable=int(np.sum(graph_supported & ~np.isfinite(shift))),
                               predictions_capped=int(np.sum(prediction > 0)),
                               solvent_slopes_with_support=int(np.sum(slope_enabled & (model['components'] >= 0))),
                               observed_solvent_slopes_zeroed=int(np.sum(~slope_enabled & (model['components'] >= 0))),
                               maximum_relative_residual=model['maximum_relative_residual'],
                               baseline_absolute_residual=base_residual,
                               rank=sum(part['rank'] for part in model['fitted']),
                               free_parameters=sum(part['free_parameters'] for part in model['fitted'])))
            for part in model['fitted']:
                for solvent in np.flatnonzero(model['components'] == part['component']):
                    slope = model['beta'][1, solvent] / scale
                    parameters.append(dict(repeat=repeat, fold=fold, **solvents[solvent],
                                           component=part['component'], anchor=part['anchor'],
                                           intercept=model['beta'][0, solvent] - slope*mean,
                                           logP_slope=slope, training_logP_mean=mean, training_logP_scale=scale,
                                           distinct_training_molecules=int(support_counts[solvent]),
                                           slope_eligible=bool(slope_enabled[solvent]),
                                           slope_fitted=bool(slope_enabled[solvent] and solvent != part['anchor']),
                                           component_rank=part['rank'], free_parameters=part['free_parameters']))
        print(f'Completed repeat {repeat+1}/{repeats} in {time.monotonic()-started:.1f}s', flush=True)
    old_supported = np.isfinite(old_errors)
    if np.all(np.isfinite(logp)):
        if not np.array_equal(old_supported, np.isfinite(baseline)):
            raise ValueError('Refitted baseline support differs from saved baseline')
        if not np.allclose(old_errors[old_supported], baseline[old_supported], rtol=0, atol=1e-8):
            raise ValueError('Refitted multiplier predictions differ from original baseline')
    np.savez_compressed(output / 'errors_capped.npz', error=errors, case=cases['case'])
    np.savez_compressed(output / 'errors_uncapped.npz', error=raw, case=cases['case'])
    np.savez_compressed(output / 'baseline_errors.npz', error=baseline, case=cases['case'])
    write_csv(output / 'molecules.csv', molecules)
    write_csv(output / 'fold_parameters.csv', parameters)
    write_csv(output / 'fold_audit.csv', audits)
    print('Predictions saved; scoring matched populations', flush=True)
    common = np.all(np.isfinite(errors) & np.isfinite(baseline), axis=1)
    main = common & (cases['scope'] == 0)
    baseline_scores, jobs = {}, []
    selections = {'same_publication': main,
                  'water_to_nonwater': main & (cases['category'] == 0),
                  'nonwater_to_water': main & (cases['category'] == 1),
                  'nonwater_to_nonwater': main & (cases['category'] == 2)}
    selections.update({scope: common & (cases['scope'] == i) for i, scope in enumerate(SCOPES) if i})
    for name, mask in selections.items():
        if not np.any(mask):
            continue
        baseline_scores[name] = {model: metrics(values[mask], cases[mask])
                                 for model, values in (('multiplier', baseline), ('multiplier_logP', errors))}
    jobs.append(('logP_minus_multiplier', errors[main], baseline[main], cases['solute'][main], seed, args.draws, 7))
    scores = {cohort: {} for cohort in ('own_coverage', 'common_five', 'common_six')}
    method_rows = [records[(records['method'] == i) & (records['scope'] == 0) & (records['status'] == 0)]
                   for i in range(len(METHODS))]
    ids_five = set.intersection(*(set(rows['case']) for rows in method_rows[:5]))
    ids_six = ids_five & set(method_rows[-1]['case'])
    positions = {int(case): i for i, case in enumerate(cases['case'])}
    for method, rows in zip(METHODS, method_rows):
        location = np.array([positions[int(case)] for case in rows['case']])
        for cohort, allowed in (('own_coverage', None), ('common_five', ids_five), ('common_six', ids_six)):
            if cohort == 'common_five' and method == METHODS[-1]:
                continue
            keep = common[location]
            if allowed is not None:
                keep &= np.array([case in allowed for case in rows['case']])
            index, unifac = location[keep], rows['error_log10'][keep]
            if not len(index):
                continue
            scores[cohort][method] = {'UNIFAC': metrics(unifac[:, None], cases[index]),
                                      'multiplier': metrics(baseline[index], cases[index]),
                                      'multiplier_logP': metrics(errors[index], cases[index])}
            if cohort == 'own_coverage':
                jobs.append((method + '_minus_logP', unifac, errors[index], cases['solute'][index], seed, args.draws, 7))
    if len(jobs) != 7:
        raise ValueError('Expected seven primary paired comparisons')
    # These short bootstraps are sequential to avoid copying large case arrays to workers.
    paired = [comparison_job(job) for job in jobs]
    summary = dict(seed=seed, folds=n_folds, repeats=repeats, bootstrap_draws=args.draws,
                   minimum_molecules=args.minimum_molecules,
                   audit=dict(data_audit=data_audit, solutes=n_solutes, solvents=n_solvents,
                              descriptor_failures=int(np.sum(~np.isfinite(logp))),
                              computed_logP_min=float(np.nanmin(logp)), computed_logP_max=float(np.nanmax(logp)),
                              main_candidate_transfers=int(np.sum(cases['scope'] == 0)),
                              baseline_supported_main_transfers=int(np.sum(np.all(np.isfinite(baseline), axis=1) & (cases['scope'] == 0))),
                              new_model_supported_main_transfers=int(main.sum()),
                              baseline_reconciled_with_original=bool(np.all(np.isfinite(logp))),
                              maximum_relative_normal_residual=max(row['maximum_relative_residual'] for row in audits),
                              main_capped_metrics=metrics(errors[main], cases[main]),
                              main_uncapped_metrics=metrics(raw[main], cases[main]),
                              total_capped_predictions=sum(row['predictions_capped'] for row in audits),
                              fold_graph_identifiability_refusals=sum(row['graph_supported_but_unidentifiable'] for row in audits)),
                   baseline_scores=baseline_scores, unifac_scores=scores, paired=paired,
                   elapsed_seconds=time.monotonic()-started)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    render(output, summary)
    dependencies = [Path(__file__), STUDY_DIR / 'benchmark_solvent_transfer_multipliers.py',
                    STUDY_DIR / 'benchmark_solubility_transfer.py', args.results_dir / 'manifest.json',
                    args.results_dir / 'metric_records.npy', args.baseline_dir / 'manifest.json',
                    args.baseline_dir / 'summary.json', args.baseline_dir / 'folds.csv',
                    args.baseline_dir / 'baseline_errors.npz', *sorted(args.data_dir.glob('*.csv')),
                    *sorted(output.iterdir())]
    write_manifest(output / 'manifest.json', vars(args), dependencies, rdkit_version=rdBase.rdkitVersion)
    print(json.dumps(dict(main=baseline_scores['same_publication'], paired=paired, audit=summary['audit']), indent=2), flush=True)
    print(f'Completed {output} in {summary["elapsed_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
