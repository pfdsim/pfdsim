"""Summarize saved Nannoolal predictions and plot held-out correlation; no refits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmark_expanded_nannoolal import score
from benchmark_nannoolal_solubility import valid_predictions
from solubility_paths import RESULTS_DIR, verify_manifest, write_manifest

LABELS = {
    'native_adaptive': 'Native Nannoolal',
    'repaired_adaptive': 'Repaired Nannoolal',
    'expanded_adaptive': 'Expanded Nannoolal',
    'expanded_OLS': 'Expanded Nannoolal OLS',
    'element_adaptive': 'Element baseline',
}


def headline_metrics(prediction, target, ids):
    """Reuse study scoring; correlate individual held-out predictions, not an ensemble."""
    result = score(prediction, target, ids)
    keep = np.all(valid_predictions(prediction), axis=1)
    if not np.any(keep):
        raise ValueError('No molecules valid in every repeat')
    measured = np.repeat(target[keep], prediction.shape[1])
    predicted = prediction[keep].ravel()
    # Pearson correlation is undefined for constant targets or predictions.
    correlation = (float(np.corrcoef(measured, predicted)[0, 1])
                   if np.std(measured) > 0 and np.std(predicted) > 0 else None)
    result['metrics']['Pearson_r'] = correlation
    return result


def plot_panels(output, stem, names, target, predictions, masks, rows, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # Identical axes, bins and density scale within a figure. No outliers cropped.
    lower = min(min(target[mask].min(), predictions[name][mask].min())
                for name, mask in zip(names, masks)) - .25
    upper = max(max(target[mask].max(), predictions[name][mask].max())
                for name, mask in zip(names, masks)) + .25
    limit = [lower, upper]
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5),
                             sharex=True, sharey=True, layout='constrained')
    axes = np.atleast_1d(axes)
    bins = []
    for axis, name, mask in zip(axes, names, masks):
        x = np.repeat(target[mask], predictions[name].shape[1])
        y = predictions[name][mask].ravel()
        density = axis.hexbin(x, y, gridsize=55, extent=[*limit, *limit],
                              mincnt=1, norm=LogNorm(), cmap='Blues')
        bins.append(density)
        axis.plot(limit, limit, '--', color='#555555', linewidth=1)
        for offset in (-1, 1):
            axis.plot(limit, np.array(limit) + offset, ':', color='#999999', linewidth=.8)
        metrics = rows[name]['metrics']
        axis.set_title(f'{LABELS[name]} · n={mask.sum():,}\n'
                       f'MAE {metrics["MAE_log10"]:.3f} · RMSE {metrics["RMSE_log10"]:.3f}\n'
                       f'R² {metrics["R2"]:.3f} · Pearson r {metrics["Pearson_r"]:.3f}')
        axis.set(xlim=limit, ylim=limit, xlabel='Measured log₁₀(S / mol L⁻¹)')
        axis.set_aspect('equal')
        axis.spines[['top', 'right']].set_visible(False)
    maximum = max(float(item.get_array().max()) for item in bins)
    for item in bins:
        item.set_clim(1, maximum)
    fig.colorbar(bins[0], ax=list(axes), label='Held-out predictions per hexagon', shrink=.75)
    axes[0].set_ylabel('Held-out predicted log₁₀(S / mol L⁻¹)')
    fig.suptitle(title + '\nAll five CV repeats; dashed: equality; dotted: factor 10')
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'{stem}.{suffix}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path,
                        default=RESULTS_DIR / 'expanded-nannoolal-regression')
    parser.add_argument('--output-dir', type=Path,
                        default=RESULTS_DIR / 'nannoolal-headline-metrics')
    args = parser.parse_args()
    source, output = args.results_dir, args.output_dir
    verify_manifest(source / 'manifest.json')
    previous = json.loads((source / 'summary.json').read_text())
    with np.load(source / 'predictions.npz') as saved:
        target, ids = saved['target'], saved['index']
        predictions = {name: saved[name] for name in LABELS}
    own = {name: headline_metrics(p, target, ids) for name, p in predictions.items()}
    for name, row in own.items():
        for key, value in previous['scores'][name]['metrics'].items():
            if not np.isclose(row['metrics'][key], value, rtol=1e-12, atol=1e-12):
                raise ValueError(f'Saved score mismatch: {name}/{key}')
        for key in ('valid_all_repeats', 'supported_all_repeats', 'invalid_supported_predictions'):
            if row[key] != previous['scores'][name][key]:
                raise ValueError(f'Saved count mismatch: {name}/{key}')
    matched, masks = {}, {}
    for other in ('element_adaptive', 'native_adaptive'):
        mask = np.all(valid_predictions(predictions['expanded_adaptive'])
                      & valid_predictions(predictions[other]), axis=1)
        masks[other] = mask
        matched[other] = {name: headline_metrics(predictions[name][mask], target[mask], ids[mask])
                          for name in (other, 'expanded_adaptive')}
        pair = next(row for row in previous['paired']
                    if row['first'] == 'expanded_adaptive' and row['second'] == other)
        if not (mask.sum() == pair['molecules']
                and np.isclose(matched[other]['expanded_adaptive']['metrics']['MAE_log10'],
                               pair['MAE_first'], rtol=1e-12, atol=1e-12)):
            raise ValueError(f'Paired score mismatch: {other}')
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(own_coverage=own, matched=matched,
                   scoring='Pool individual held-out errors over repeats; no prediction averaging. '
                           'Each molecule has equal weight. R2 = 1 - MSE / variance(target). '
                           'Pearson r uses the same repeated targets and predictions; r squared is not R2. '
                           'Require valid predictions in every repeat; invalid range is outside [-30, 3].')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    rows = []
    for cohort, models in [('own_coverage', own), *matched.items()]:
        for name, row in models.items():
            rows.append(dict(cohort=cohort, model=name, **{k: v for k, v in row.items() if k != 'metrics'},
                             **row['metrics']))
    with (output / 'headline_metrics.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    names = ['native_adaptive', 'repaired_adaptive', 'expanded_adaptive']
    plot_panels(output, 'nannoolal_predictions', names, target, predictions,
                [np.all(valid_predictions(predictions[name]), axis=1) for name in names], own,
                'Native → repaired → expanded Nannoolal: own coverage (different molecules)')
    for other, mask in masks.items():
        plot_panels(output, f'matched_{other}', [other, 'expanded_adaptive'], target,
                    predictions, [mask, mask], matched[other],
                    f'Matched evaluation on identical {mask.sum():,} molecules and folds')
    files = [Path(__file__), Path(__file__).with_name('benchmark_expanded_nannoolal.py'),
             Path(__file__).with_name('benchmark_nannoolal_solubility.py'),
             source / 'manifest.json', source / 'summary.json', source / 'predictions.npz',
             *sorted(output.iterdir())]
    write_manifest(output / 'manifest.json', vars(args), files)
    print(f'Verified saved scores; wrote headline metrics and three correlation figures to {output}')


if __name__ == '__main__':
    main()
