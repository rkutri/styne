#!/usr/bin/env python3
"""Pilot-tuned DART, MALA, and untempered MLDA logistic comparison."""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from . import dartbenchmark as common
else:
    import dartbenchmark as common

import numpy as np  # noqa: E402


BENCHMARK_SLUG = 'dartlogistic'
DART_GAMMA_OVER_L_CANDIDATES = tuple(
    float(value) for value in 10.0 ** np.linspace(-5.0, 1.0, 25)
)
MALA_STEP_TIMES_SQRT_L_CANDIDATES = tuple(
    float(value) for value in 10.0 ** np.linspace(-1.5, 0.5, 21)
)
MLDA_ROOT_STEP_TIMES_SQRT_D_CANDIDATES = tuple(
    float(value) for value in 10.0 ** np.linspace(-1.0, 1.0, 21)
)
MLDA_SUBCHAIN_LENGTH = 10


@dataclass(frozen=True)
class LogisticComparisonConfiguration:
    name: str
    dimensions: tuple
    evaluationSeeds: tuple
    pilotChains: int
    pilotBurnin: int
    pilotRetained: int
    reportingChains: int
    reportingBurnin: int
    reportingRetained: int
    dartGammaCandidates: tuple
    malaNormalisedStepCandidates: tuple
    mldaNormalisedRootStepCandidates: tuple
    mldaSubchainLength: int


SMOKE_CONFIGURATION = LogisticComparisonConfiguration(
    name='smoke',
    dimensions=(2,),
    evaluationSeeds=(31_000,),
    pilotChains=2,
    pilotBurnin=10,
    pilotRetained=40,
    reportingChains=2,
    reportingBurnin=20,
    reportingRetained=80,
    dartGammaCandidates=(0.01, 0.1),
    malaNormalisedStepCandidates=(0.25, 1.0),
    mldaNormalisedRootStepCandidates=(0.5, 2.0),
    mldaSubchainLength=MLDA_SUBCHAIN_LENGTH,
)

PRODUCTION_CONFIGURATION = LogisticComparisonConfiguration(
    name='production',
    dimensions=(2, 4, 8, 16),
    evaluationSeeds=tuple(range(33_000, 35_000, 100)),
    pilotChains=4,
    pilotBurnin=2_000,
    pilotRetained=10_000,
    reportingChains=4,
    reportingBurnin=25_000,
    reportingRetained=100_000,
    dartGammaCandidates=DART_GAMMA_OVER_L_CANDIDATES,
    malaNormalisedStepCandidates=MALA_STEP_TIMES_SQRT_L_CANDIDATES,
    mldaNormalisedRootStepCandidates=(
        MLDA_ROOT_STEP_TIMES_SQRT_D_CANDIDATES
    ),
    mldaSubchainLength=MLDA_SUBCHAIN_LENGTH,
)


def comparison_seeds(evaluationSeed, dimension, method=None, candidate=0):
    methodCode = {
        None: 0,
        common.DART: 1,
        common.MALA: 2,
        common.MLDA: 3,
    }[method]

    def seed(purpose, extra=0):
        return int(np.random.SeedSequence((
            int(evaluationSeed), int(dimension), int(purpose),
            int(methodCode), int(extra),
        )).generate_state(1)[0])

    return {
        'dataset': seed(1),
        'pilot_start': seed(2),
        'pilot_chain': seed(3, candidate),
        'reporting_start': seed(4),
        'reporting_chain': seed(5),
    }


def prior_starts(seed, chainCount, dimension):
    randomGenerator = np.random.default_rng(seed)
    return randomGenerator.normal(
        scale=1.0 / np.sqrt(common.ALPHA_PRIOR),
        size=(chainCount, dimension),
    )


def target_components(backend, problem):
    target = common.LogisticPosterior(
        problem.features, problem.responses, common.ALPHA_PRIOR
    )
    surrogate = common.backend_gaussian(
        backend,
        problem.mapCoordinate,
        np.linalg.inv(problem.hessian),
    )
    return target, surrogate


def pilot_candidates(configuration, method):
    if method == common.DART:
        return configuration.dartGammaCandidates
    if method == common.MALA:
        return configuration.malaNormalisedStepCandidates
    if method == common.MLDA:
        return configuration.mldaNormalisedRootStepCandidates
    raise ValueError(f'Unknown comparison method: {method}')


def make_comparison_sampler(configuration, method, backend, target, surrogate,
                            problem, tuningValue):
    if method == common.DART:
        gamma = tuningValue * problem.smoothness
        return common.make_dart_sampler(
            target, backend, surrogate, problem.hessian, gamma
        )
    if method == common.MALA:
        stepSize = tuningValue / np.sqrt(problem.smoothness)
        return common.make_mala_sampler(target, stepSize)
    if method == common.MLDA:
        rootCovariance = (
            tuningValue ** 2 / problem.hessian.shape[0]
        ) * np.linalg.inv(problem.hessian)
        return common.make_mlda_sampler(
            target,
            backend,
            surrogate,
            rootCovariance,
            configuration.mldaSubchainLength,
        )
    raise ValueError(f'Unknown comparison method: {method}')


def tuning_fields(method, tuningValue, problem, configuration):
    if method == common.DART:
        return {
            'tempering': common.TEMPERING,
            'gamma_over_l': tuningValue,
            'mala_step_size': '',
            'tuning_parameter': 'gamma_over_l',
            'tuning_value': tuningValue,
            'mlda_root_step_times_sqrt_d': '',
            'mlda_subchain_length': '',
        }
    if method == common.MALA:
        stepSize = tuningValue / np.sqrt(problem.smoothness)
        return {
            'tempering': '',
            'gamma_over_l': '',
            'mala_step_size': stepSize,
            'tuning_parameter': 'step_size_times_sqrt_l',
            'tuning_value': tuningValue,
            'mlda_root_step_times_sqrt_d': '',
            'mlda_subchain_length': '',
        }
    if method == common.MLDA:
        return {
            'tempering': 1.0,
            'gamma_over_l': '',
            'mala_step_size': '',
            'tuning_parameter': 'root_step_times_sqrt_d',
            'tuning_value': tuningValue,
            'mlda_root_step_times_sqrt_d': tuningValue,
            'mlda_subchain_length': configuration.mldaSubchainLength,
        }
    raise ValueError(f'Unknown comparison method: {method}')


def run_tuning_cell(
        configuration, backend, targetIndex, evaluationSeed, dimension, method,
        candidateIndex, tuningValue):
    baseSeeds = comparison_seeds(evaluationSeed, dimension)
    methodSeeds = comparison_seeds(
        evaluationSeed, dimension, method, candidateIndex
    )
    problem = common.generate_logistic_problem(
        baseSeeds['dataset'], dimension
    )
    starts = prior_starts(
        baseSeeds['pilot_start'], configuration.pilotChains, dimension
    )
    runId = f'target_{targetIndex}:{method}_{candidateIndex}'
    fields = tuning_fields(method, tuningValue, problem, configuration)
    setupSeconds = ''
    try:
        setupStarted = time.perf_counter()
        target, surrogate = target_components(backend, problem)
        sampler = make_comparison_sampler(
            configuration,
            method,
            backend,
            target,
            surrogate,
            problem,
            tuningValue,
        )
        setupSeconds = time.perf_counter() - setupStarted
        metrics = common.run_sampler(
            sampler,
            backend,
            starts,
            problem.eigenvectors,
            problem.hessian,
            configuration.pilotBurnin,
            configuration.pilotRetained,
            methodSeeds['pilot_chain'],
        )
        row = common.metrics_row(
            configuration,
            backend.name,
            common.LOGISTIC_TUNING,
            method,
            runId,
            evaluationSeed,
            baseSeeds['dataset'],
            baseSeeds['pilot_start'],
            methodSeeds['pilot_chain'],
            dimension,
            'laplace',
            configuration.pilotChains,
            configuration.pilotBurnin,
            configuration.pilotRetained,
            fields['gamma_over_l'],
            fields['mala_step_size'],
            metrics,
            setupSeconds=setupSeconds,
            mldaSubchainLength=(
                configuration.mldaSubchainLength
                if method == common.MLDA else None
            ),
        )
    except Exception as error:
        row = common.failure_row(
            configuration,
            backend.name,
            common.LOGISTIC_TUNING,
            method,
            runId,
            evaluationSeed,
            baseSeeds['dataset'],
            baseSeeds['pilot_start'],
            methodSeeds['pilot_chain'],
            dimension,
            'laplace',
            configuration.pilotChains,
            configuration.pilotBurnin,
            configuration.pilotRetained,
            error,
        )
        row['setup_seconds'] = setupSeconds
    row.update(fields)
    row['selected_for_evaluation'] = False
    row['boundary_selection'] = False
    return row


def select_tuning_candidate(configuration, rows, method, evaluationSeed, dimension):
    candidates = pilot_candidates(configuration, method)
    candidateRows = [
        row for row in rows
        if row['method'] == method
        and int(row['evaluation_seed']) == evaluationSeed
        and int(row['dimension']) == dimension
        and common.is_usable(row)
    ]
    if not candidateRows:
        return None
    selected = max(
        candidateRows,
        key=lambda row: (
            float(row['slow_direction_ess_per_iteration']),
            -float(row['tuning_value']),
        ),
    )
    selectedValue = float(selected['tuning_value'])
    boundary = selectedValue in (candidates[0], candidates[-1])
    selected['selected_for_evaluation'] = True
    selected['boundary_selection'] = boundary
    return {
        'value': selectedValue,
        'score': float(selected['slow_direction_ess_per_iteration']),
        'boundary': boundary,
        'status': selected['status'],
    }


def run_reporting_cell(
        configuration, backend, targetIndex, evaluationSeed, dimension, method,
        selection):
    baseSeeds = comparison_seeds(evaluationSeed, dimension)
    methodSeeds = comparison_seeds(evaluationSeed, dimension, method)
    problem = common.generate_logistic_problem(
        baseSeeds['dataset'], dimension
    )
    starts = prior_starts(
        baseSeeds['reporting_start'], configuration.reportingChains, dimension
    )
    if selection is None:
        error = RuntimeError('No eligible pilot candidate.')
        row = common.failure_row(
            configuration,
            backend.name,
            common.LOGISTIC_COMPARISON,
            method,
            targetIndex,
            evaluationSeed,
            baseSeeds['dataset'],
            baseSeeds['reporting_start'],
            methodSeeds['reporting_chain'],
            dimension,
            'laplace',
            configuration.reportingChains,
            configuration.reportingBurnin,
            configuration.reportingRetained,
            error,
        )
        row['tempering'] = (
            common.TEMPERING if method == common.DART
            else 1.0 if method == common.MLDA else ''
        )
        if method == common.MLDA:
            row['mlda_subchain_length'] = configuration.mldaSubchainLength
        return row

    fields = tuning_fields(method, selection['value'], problem, configuration)
    setupSeconds = ''
    try:
        setupStarted = time.perf_counter()
        target, surrogate = target_components(backend, problem)
        sampler = make_comparison_sampler(
            configuration,
            method,
            backend,
            target,
            surrogate,
            problem,
            selection['value'],
        )
        setupSeconds = time.perf_counter() - setupStarted
        metrics = common.run_sampler(
            sampler,
            backend,
            starts,
            problem.eigenvectors,
            problem.hessian,
            configuration.reportingBurnin,
            configuration.reportingRetained,
            methodSeeds['reporting_chain'],
        )
        row = common.metrics_row(
            configuration,
            backend.name,
            common.LOGISTIC_COMPARISON,
            method,
            targetIndex,
            evaluationSeed,
            baseSeeds['dataset'],
            baseSeeds['reporting_start'],
            methodSeeds['reporting_chain'],
            dimension,
            'laplace',
            configuration.reportingChains,
            configuration.reportingBurnin,
            configuration.reportingRetained,
            fields['gamma_over_l'],
            fields['mala_step_size'],
            metrics,
            setupSeconds=setupSeconds,
            mldaSubchainLength=(
                configuration.mldaSubchainLength
                if method == common.MLDA else None
            ),
        )
    except Exception as error:
        row = common.failure_row(
            configuration,
            backend.name,
            common.LOGISTIC_COMPARISON,
            method,
            targetIndex,
            evaluationSeed,
            baseSeeds['dataset'],
            baseSeeds['reporting_start'],
            methodSeeds['reporting_chain'],
            dimension,
            'laplace',
            configuration.reportingChains,
            configuration.reportingBurnin,
            configuration.reportingRetained,
            error,
        )
        row['setup_seconds'] = setupSeconds
    row.update(fields)
    row['boundary_selection'] = selection['boundary']
    return row


def descriptive_ratio(rows, denominator, dimensions=None):
    if dimensions is None:
        dimensions = {
            int(row['dimension']) for row in rows
            if row['analysis'] == common.LOGISTIC_COMPARISON
        }
    paired = {}
    for row in rows:
        if row['analysis'] != common.LOGISTIC_COMPARISON \
                or row['method'] not in (common.DART, denominator) \
                or int(row['dimension']) not in dimensions \
                or not common.is_usable(row):
            continue
        value = float(row['slow_direction_ess_per_iteration'])
        if np.isfinite(value) and value > 0.0:
            key = (int(row['evaluation_seed']), int(row['dimension']))
            paired.setdefault(key, {})[row['method']] = value
    ratios = np.asarray([
        values[common.DART] / values[denominator]
        for values in paired.values()
        if common.DART in values and denominator in values
    ])
    if ratios.size == 0:
        return np.nan, 0, 0
    return (
        float(np.median(ratios)),
        int(ratios.size),
        int(np.sum(ratios > 1.0)),
    )


def comparison_target_key(evaluationSeed, dimension):
    return f'{int(evaluationSeed)}:d{int(dimension)}'


def expected_checkpoint_row_keys(configuration, targetIndex):
    keys = {
        (
            common.LOGISTIC_TUNING,
            method,
            f'target_{targetIndex}:{method}_{candidateIndex}',
        )
        for method in (common.DART, common.MALA, common.MLDA)
        for candidateIndex, _ in enumerate(pilot_candidates(configuration, method))
    }
    keys.update({
        (common.LOGISTIC_COMPARISON, method, str(targetIndex))
        for method in (common.DART, common.MALA, common.MLDA)
    })
    return keys


def run_benchmark(
        configuration, backend, checkpointState=None, checkpointCallback=None):
    common.validate_production_runtime(configuration, backend)
    common.generate_logistic_problem.cache_clear()
    print(  # noqa: T201
        f'=== {BENCHMARK_SLUG} ({configuration.name}, {backend.name}) ==='
    )
    checkpointState = checkpointState or {}
    rows = list(checkpointState.get('rows', ()))
    completedTargets = set(checkpointState.get('completed_targets', ()))
    expectedTargets = {
        comparison_target_key(evaluationSeed, dimension)
        for evaluationSeed in configuration.evaluationSeeds
        for dimension in configuration.dimensions
    }
    if not completedTargets.issubset(expectedTargets):
        raise ValueError('Checkpoint contains an unknown target.')

    selections = {}
    for targetIndex, evaluationSeed in enumerate(configuration.evaluationSeeds):
        for dimension in configuration.dimensions:
            targetKey = comparison_target_key(evaluationSeed, dimension)
            targetRows = [
                row for row in rows
                if comparison_target_key(
                    row['evaluation_seed'], row['dimension']
                ) == targetKey
            ]
            expectedKeys = expected_checkpoint_row_keys(
                configuration, targetIndex
            )
            actualKeys = {
                (row['analysis'], row['method'], str(row['run_id']))
                for row in targetRows
            }
            if targetKey in completedTargets:
                if len(targetRows) != len(expectedKeys) \
                        or actualKeys != expectedKeys:
                    raise ValueError(
                        f'Checkpoint target {targetKey} has unexpected rows.'
                    )
            elif targetRows:
                raise ValueError(f'Checkpoint target {targetKey} is partial.')
            else:
                for method in (common.DART, common.MALA, common.MLDA):
                    for candidateIndex, tuningValue in enumerate(
                            pilot_candidates(configuration, method)):
                        print(  # noqa: T201
                            f'[{configuration.name}] tuning '
                            f'target={targetIndex} d={dimension} '
                            f'method={method} candidate={candidateIndex}'
                        )
                        rows.append(run_tuning_cell(
                            configuration,
                            backend,
                            targetIndex,
                            evaluationSeed,
                            dimension,
                            method,
                            candidateIndex,
                            tuningValue,
                        ))

            for method in (common.DART, common.MALA, common.MLDA):
                selections[(evaluationSeed, dimension, method)] = (
                    select_tuning_candidate(
                        configuration, rows, method, evaluationSeed, dimension
                    )
                )

            if targetKey not in completedTargets:
                for method in (common.DART, common.MALA, common.MLDA):
                    print(  # noqa: T201
                        f'[{configuration.name}] reporting '
                        f'target={targetIndex} d={dimension} method={method}'
                    )
                    rows.append(run_reporting_cell(
                        configuration,
                        backend,
                        targetIndex,
                        evaluationSeed,
                        dimension,
                        method,
                        selections[(evaluationSeed, dimension, method)],
                    ))
                completedTargets.add(targetKey)
                if checkpointCallback is not None:
                    checkpointCallback({
                        'rows': rows,
                        'completed_targets': sorted(completedTargets),
                    })
    return rows


def print_summary(rows, dimensions):
    print('\nMedian DART / MALA ESS ratios')  # noqa: T201
    for dimension in dimensions:
        ratio, pairs, wins = descriptive_ratio(
            rows, common.MALA, (dimension,)
        )
        print(  # noqa: T201
            f'  d={dimension}: {ratio:.3f} ({wins}/{pairs} wins)'
        )
    ratio, pairs, wins = descriptive_ratio(rows, common.MALA)
    print(f'  overall: {ratio:.3f} ({wins}/{pairs} wins)')  # noqa: T201
    ratio, pairs, wins = descriptive_ratio(rows, common.MLDA)
    print(  # noqa: T201
        f'DART / MLDA: {ratio:.3f} ({wins}/{pairs} wins)'
    )


def make_parser():
    parser = argparse.ArgumentParser(
        description='Pilot-tuned DART, MALA, and MLDA comparison.'
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

    rows = run_benchmark(
        configuration,
        backend,
        checkpointState=checkpointState,
        checkpointCallback=checkpoint,
    )
    outputPath = common.write_results(
        arguments.output_dir, BENCHMARK_SLUG, configuration, backend, rows
    )
    common.remove_checkpoint(checkpointPath)
    print_summary(rows, configuration.dimensions)
    print(f'output: {outputPath}')  # noqa: T201
    return outputPath


if __name__ == '__main__':
    main()
