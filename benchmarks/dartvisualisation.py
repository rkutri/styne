#!/usr/bin/env python3
"""Plot the latest DART benchmark results."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DART_COLOUR = '#377eb8'
REFERENCE_COLOUR = '0.35'
LOGISTIC_PREFIX = 'dartlogistic_production_jax_'
LOCALISATION_PREFIX = 'dartlocalisation_production_jax_'
EFFICIENCY_FIELD = (
    'minimum_laplace_hessian_eigendirection_ess_'
    'per_target_density_evaluation'
)
CONDITION_LABELS = {
    'laplace': 'original approximation',
    'mean_shift': 'wrong centre: moved 1.5 typical widths',
    'precision_distortion': 'wrong spread: too wide or narrow',
    'combined_misspecification': 'wrong centre and spread',
}


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        lines = (line for line in stream if not line.startswith('# '))
        return list(csv.DictReader(lines))


def latest_result(directory, prefix):
    candidates = [
        path for path in Path(directory).glob(f'{prefix}*.csv')
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def logistic_ratios(rows):
    paired = {}
    for row in rows:
        if row['analysis'] != 'logistic_comparison' \
                or row['method'] not in ('dart_laplace', 'mala'):
            continue
        value = float(row['slow_direction_ess_per_iteration'])
        if not np.isfinite(value) or value <= 0.0:
            continue
        key = (int(row['evaluation_seed']), int(row['dimension']))
        paired.setdefault(key, {})[row['method']] = value
    ratios = {}
    for (_, dimension), values in paired.items():
        if 'dart_laplace' in values and 'mala' in values:
            ratios.setdefault(dimension, []).append(
                values['dart_laplace'] / values['mala']
            )
    return {
        dimension: np.asarray(values)
        for dimension, values in sorted(ratios.items())
    }


def localisation_ratios(rows):
    paired = {}
    for row in rows:
        if row['analysis'] != 'localisation_ablation' \
                or row['method'] not in (
                    'localised_dart', 'unlocalised_surrogate'
                ):
            continue
        value = float(row[EFFICIENCY_FIELD])
        if not np.isfinite(value) or value <= 0.0:
            continue
        key = (
            int(row['evaluation_seed']),
            int(row['dimension']),
            row['surrogate_condition'],
        )
        paired.setdefault(key, {})[row['method']] = value

    grouped = {}
    for key, values in paired.items():
        if 'localised_dart' not in values \
                or 'unlocalised_surrogate' not in values:
            continue
        dimension = key[1]
        condition = key[2]
        group = grouped.setdefault(
            (dimension, condition),
            {'localised_dart': [], 'unlocalised_surrogate': []},
        )
        for method in group:
            group[method].append(values[method])
    return {
        key: float(
            np.mean(values['localised_dart'])
            / np.mean(values['unlocalised_surrogate'])
        )
        for key, values in grouped.items()
    }


def localisation_trend(ratios):
    dimensions = sorted({dimension for dimension, _ in ratios})
    conditions = sorted({condition for _, condition in ratios})
    completeConditions = [
        condition for condition in conditions
        if all((dimension, condition) in ratios for dimension in dimensions)
    ]
    if len(dimensions) < 2 or not completeConditions:
        return np.asarray([]), np.asarray([]), np.nan

    dimensions = np.asarray(dimensions, dtype=float)
    coordinates = np.log(dimensions)
    logRatios = np.asarray([
        [np.log(ratios[(dimension, condition)])
         for condition in completeConditions]
        for dimension in dimensions
    ])
    centeredCoordinates = coordinates - np.mean(coordinates)
    centeredRatios = logRatios - np.mean(logRatios, axis=0)
    slope = np.sum(
        centeredRatios * centeredCoordinates[:, None]
    ) / (
        len(completeConditions)
        * centeredCoordinates @ centeredCoordinates
    )
    intercepts = np.mean(logRatios, axis=0) - slope * np.mean(coordinates)
    fittedRatios = np.exp(np.mean(intercepts) + slope * coordinates)
    return dimensions, fittedRatios, float(np.exp(np.log(2.0) * slope))


def style_axis(axis):
    axis.tick_params(direction='out', length=3.0, width=0.8)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    for side in ('bottom', 'left'):
        axis.spines[side].set_linewidth(0.8)
    axis.grid(True, which='major', linestyle=':', linewidth=0.6, alpha=0.7)
    axis.axhline(1.0, color=REFERENCE_COLOUR, linewidth=0.8, linestyle='--')
    axis.text(
        0.99,
        1.0,
        'equal efficiency',
        color=REFERENCE_COLOUR,
        fontsize=8,
        ha='right',
        va='bottom',
        transform=axis.get_yaxis_transform(),
    )


def plot_logistic(axis, ratios):
    dimensions = sorted(ratios)
    positions = np.arange(len(dimensions))
    estimates = []
    for position, dimension in zip(positions, dimensions):
        values = ratios[dimension]
        offsets = np.linspace(-0.12, 0.12, len(values))
        axis.scatter(
            position + offsets,
            values,
            s=12,
            color=DART_COLOUR,
            alpha=0.22,
            edgecolors='none',
        )
        estimates.append(float(np.median(values)))
    axis.plot(
        positions, estimates, color=DART_COLOUR, marker='o', linewidth=1.5
    )
    axis.set_xticks(positions, dimensions)
    axis.set_yscale('log')
    axis.set_xlabel('dimension')
    axis.set_ylabel('DART / MALA ESS ratio')
    axis.set_title('Sampler comparison')
    style_axis(axis)


def plot_localisation(axis, ratios, colourMap):
    dimensions = sorted({dimension for dimension, _ in ratios})
    conditions = [
        condition for condition in CONDITION_LABELS
        if any(candidate == condition for _, candidate in ratios)
    ]
    colours = colourMap(
        np.linspace(0.45, 0.9, len(conditions))
    )
    for condition, colour in zip(conditions, colours):
        values = [ratios.get((dimension, condition), np.nan)
                  for dimension in dimensions]
        axis.plot(
            dimensions,
            values,
            color=colour,
            marker='o',
            linewidth=1.3,
            label=CONDITION_LABELS[condition],
        )
    trendDimensions, trendRatios, trendMultiplier = localisation_trend(ratios)
    if trendDimensions.size:
        axis.plot(
            trendDimensions,
            trendRatios,
            color='0.15',
            linestyle='--',
            linewidth=1.2,
            label=f'power-law trend: {trendMultiplier:.2f}x per doubling',
        )
    axis.set_xticks(dimensions)
    axis.set_xlabel('model dimension')
    axis.set_ylabel('sampling efficiency ratio\n(localised / unlocalised)')
    axis.set_title('Does localisation help?')
    axis.legend(
        title='Gaussian approximation', frameon=False, fontsize=8
    )
    style_axis(axis)


def make_parser():
    resultDirectory = Path(__file__).parent / 'results'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logistic', type=Path)
    parser.add_argument('--localisation', type=Path)
    parser.add_argument(
        '--output', type=Path,
        default=resultDirectory / 'dartbenchmarks.pdf',
    )
    return parser


def main(arguments=None):
    parsed = make_parser().parse_args(arguments)
    resultDirectory = Path(__file__).parent / 'results'
    logisticPath = parsed.logistic or latest_result(
        resultDirectory, LOGISTIC_PREFIX
    )
    localisationPath = parsed.localisation or latest_result(
        resultDirectory, LOCALISATION_PREFIX
    )
    panels = []
    if logisticPath is not None:
        panels.append(('logistic', logisticPath))
    if localisationPath is not None:
        panels.append(('localisation', localisationPath))
    if not panels:
        raise FileNotFoundError('No DART production CSVs found.')

    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(4.5 * len(panels), 3.6),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, (benchmark, path) in zip(axes, panels):
        rows = read_rows(path)
        if benchmark == 'logistic':
            plot_logistic(axis, logistic_ratios(rows))
        else:
            plot_localisation(
                axis,
                localisation_ratios(rows),
                plt.get_cmap('Blues'),
            )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(parsed.output, bbox_inches='tight')
    plt.close(figure)
    print(f'saved: {parsed.output}')  # noqa: T201
    return parsed.output


if __name__ == '__main__':
    main()
