from dataclasses import replace

import numpy as np

from benchmarks import dartbenchmark as benchmark
from benchmarks import dartlocalisation
from styne.backend import get_backend


def test_logistic_problem_is_deterministic():
    first = benchmark.generate_logistic_problem(31_000, 4)
    benchmark.generate_logistic_problem.cache_clear()
    repeated = benchmark.generate_logistic_problem(31_000, 4)

    assert np.array_equal(first.features, repeated.features)
    assert np.array_equal(first.responses, repeated.responses)
    assert np.array_equal(first.mapCoordinate, repeated.mapCoordinate)
    assert np.array_equal(first.hessian, repeated.hessian)


def test_ablation_seed_streams_are_distinct_and_reproducible():
    first = benchmark.ablation_seeds(41_000, 4)
    repeated = benchmark.ablation_seeds(41_000, 4)

    assert first == repeated
    assert len(set(first.values())) == len(first)


def test_split_rhat_detects_shifted_chains():
    chains = np.zeros((4, 100))
    chains[0] = 3.0

    assert benchmark.split_rhat(chains) > 1.1


def test_finite_metrics_are_reported_without_claim_gates():
    metrics = benchmark.SamplingMetrics(
        essPerIteration=np.asarray([1e-6, 0.2]),
        splitRhat=np.asarray([1.5, 1.0]),
        acceptanceRate=0.5,
        maximumRhat=1.5,
        samplingSeconds=1.0,
        hessianEsjdPerEvaluation=0.1,
        fineEvaluations=100,
    )

    assert benchmark.evaluation_status(metrics) == 'ok'


def test_checkpoint_round_trip(tmp_path):
    configuration = replace(
        dartlocalisation.SMOKE_CONFIGURATION,
        ablationEvaluationSeeds=(1,),
    )
    backend = get_backend('numpy')
    identity = benchmark.checkpoint_identity(
        configuration, backend, 'test_benchmark'
    )
    path = tmp_path / 'checkpoint.json'
    payload = {'rows': [{'value': np.inf}]}

    benchmark.write_checkpoint(path, identity, payload)
    restored = benchmark.load_checkpoint(path, identity)

    assert np.isposinf(restored['rows'][0]['value'])
    benchmark.remove_checkpoint(path)
    assert not path.exists()


def test_write_results_creates_one_csv(tmp_path):
    configuration = dartlocalisation.SMOKE_CONFIGURATION
    backend = get_backend('numpy')
    row = benchmark.empty_raw_row(
        configuration, backend.name, benchmark.LOCALISATION_PILOT,
        benchmark.LOCALISED,
    )

    path = benchmark.write_results(
        tmp_path, 'test_benchmark', configuration, backend, [row]
    )

    assert path.suffix == '.csv'
    assert len(list(tmp_path.glob('*.csv'))) == 1
