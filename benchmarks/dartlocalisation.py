#!/usr/bin/env python3
"""Compare localised DART with its unlocalised Gaussian limit."""

import argparse
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from . import dartbenchmark as common
else:
    import dartbenchmark as common

import numpy as np  # noqa: E402


BENCHMARK_SLUG = 'dartlocalisation'


@dataclass(frozen=True)
class LocalisationConfiguration:
    name: str
    ablationDimensions: tuple
    ablationConditions: tuple
    ablationEvaluationSeeds: tuple
    ablationChains: int
    ablationBurnin: int
    ablationRetained: int
    ablationPilotSeeds: tuple
    ablationPilotChains: int
    ablationPilotBurnin: int
    ablationPilotRetained: int


SMOKE_CONFIGURATION = LocalisationConfiguration(
    name='smoke',
    ablationDimensions=(2,),
    ablationConditions=common.SURROGATE_CONDITIONS,
    ablationEvaluationSeeds=(41_000,),
    ablationChains=2,
    ablationBurnin=20,
    ablationRetained=80,
    ablationPilotSeeds=(21_000,),
    ablationPilotChains=2,
    ablationPilotBurnin=10,
    ablationPilotRetained=40,
)

PRODUCTION_CONFIGURATION = LocalisationConfiguration(
    name='production',
    ablationDimensions=(4, 6, 8, 10, 12, 14, 16),
    ablationConditions=common.SURROGATE_CONDITIONS,
    ablationEvaluationSeeds=tuple(range(43_000, 45_000, 100)),
    ablationChains=4,
    ablationBurnin=5_000,
    ablationRetained=30_000,
    ablationPilotSeeds=(23_000, 23_100, 23_200, 23_300),
    ablationPilotChains=4,
    ablationPilotBurnin=1_000,
    ablationPilotRetained=4_000,
)


def run_benchmark(
        configuration, backend, checkpointState=None, checkpointCallback=None):
    common.validate_production_runtime(configuration, backend)
    common.generate_logistic_problem.cache_clear()
    print(  # noqa: T201
        f'=== {BENCHMARK_SLUG} ({configuration.name}, {backend.name}) ==='
    )
    checkpointRows = list((checkpointState or {}).get('rows', ()))
    pilotRows = [
        row for row in checkpointRows
        if row['analysis'] == common.LOCALISATION_PILOT
    ]
    evaluationRows = [
        row for row in checkpointRows
        if row['analysis'] == common.LOCALISATION_ABLATION
    ]
    if len(pilotRows) + len(evaluationRows) != len(checkpointRows):
        raise ValueError('Checkpoint contains an unknown row.')

    def pilot_checkpoint(rows):
        if checkpointCallback is not None:
            checkpointCallback({'rows': rows + evaluationRows})

    selectedGamma, pilotRows, pilotScores = common.select_ablation_gamma(
        configuration,
        backend,
        initialRows=pilotRows,
        checkpointCallback=pilot_checkpoint,
    )
    if selectedGamma is None:
        raise RuntimeError('No finite localisation pilot candidate.')

    def evaluation_checkpoint(rows):
        if checkpointCallback is not None:
            checkpointCallback({'rows': pilotRows + rows})

    evaluationRows = common.run_ablation(
        configuration,
        backend,
        selectedGamma,
        initialRows=evaluationRows,
        checkpointCallback=evaluation_checkpoint,
    )
    return pilotRows + evaluationRows, selectedGamma, pilotScores


def configuration_ratios(rows):
    paired = {}
    for row in rows:
        if row['analysis'] != common.LOCALISATION_ABLATION \
                or not common.is_usable(row):
            continue
        value = float(row[common.PRIMARY_ABLATION_METRIC])
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
        if common.LOCALISED not in values or common.UNLOCALISED not in values:
            continue
        dimension = key[1]
        condition = key[2]
        group = grouped.setdefault(
            (dimension, condition),
            {common.LOCALISED: [], common.UNLOCALISED: []},
        )
        group[common.LOCALISED].append(values[common.LOCALISED])
        group[common.UNLOCALISED].append(values[common.UNLOCALISED])
    return {
        key: float(
            np.mean(values[common.LOCALISED])
            / np.mean(values[common.UNLOCALISED])
        )
        for key, values in grouped.items()
    }


def print_summary(rows, selectedGamma):
    ratios = configuration_ratios(rows)
    print(f'\nselected gamma/L: {selectedGamma:g}')  # noqa: T201
    for dimension in sorted({key[0] for key in ratios}):
        values = [
            value for (candidateDimension, _), value in ratios.items()
            if candidateDimension == dimension
        ]
        print(  # noqa: T201
            f'  d={dimension}: median ratio {np.median(values):.3f}'
        )


def make_parser():
    parser = argparse.ArgumentParser(
        description='DART localisation ablation.'
    )
    parser.add_argument(
        '--production', action='store_true',
        help='Run the JAX production workload.',
    )
    parser.add_argument(
        '--backend', choices=('numpy', 'pytorch', 'jax'), default=None,
        help='Smoke backend; production defaults to JAX.',
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=Path(__file__).parent / 'results',
    )
    parser.add_argument('--resume', action='store_true')
    return parser


def parse_args(arguments=None):
    parser = make_parser()
    parsed = parser.parse_args(arguments)
    parsed.backend = parsed.backend or (
        'jax' if parsed.production else 'numpy'
    )
    if parsed.production and parsed.backend != 'jax':
        parser.error('--production requires the JAX backend')
    return parsed


def main(arguments=None):
    arguments = parse_args(arguments)
    if arguments.backend == 'jax':
        import jax
        jax.config.update('jax_enable_x64', True)
    backend = common.get_backend(arguments.backend)
    configuration = (
        PRODUCTION_CONFIGURATION
        if arguments.production else SMOKE_CONFIGURATION
    )
    identity = common.checkpoint_identity(
        configuration, backend, BENCHMARK_SLUG
    )
    checkpointPath = arguments.output_dir / (
        f'.{BENCHMARK_SLUG}_{configuration.name}_{backend.name}.checkpoint.json'
    )
    if arguments.resume:
        checkpointState = common.load_checkpoint(checkpointPath, identity)
    else:
        if checkpointPath.exists():
            raise FileExistsError(
                f'Checkpoint exists: {checkpointPath}. Use --resume.'
            )
        checkpointState = {}
        common.write_checkpoint(checkpointPath, identity, checkpointState)

    def checkpoint(state):
        common.write_checkpoint(checkpointPath, identity, state)

    rows, selectedGamma, _ = run_benchmark(
        configuration,
        backend,
        checkpointState=checkpointState,
        checkpointCallback=checkpoint,
    )
    outputPath = common.write_results(
        arguments.output_dir, BENCHMARK_SLUG, configuration, backend, rows
    )
    common.remove_checkpoint(checkpointPath)
    print_summary(rows, selectedGamma)
    print(f'output: {outputPath}')  # noqa: T201
    return outputPath


if __name__ == '__main__':
    main()
