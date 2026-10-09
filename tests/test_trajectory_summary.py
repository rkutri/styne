import numpy as np
import pytest

from styne.mcmc import MetropolisedRandomWalk
from styne.mcmc.diagnostics import DummyDiagnostics
from styne.parameter import Vector
from styne.statistics import Gaussian, IIDCovarianceMatrix

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')


def random_walk():
    target = Gaussian(IIDCovarianceMatrix(2, jnp.asarray(1.0)),
                      Vector(jnp.zeros(2)))
    return MetropolisedRandomWalk(
        target.density, IIDCovarianceMatrix(2, jnp.asarray(1.5)), DummyDiagnostics()
    )


@pytest.mark.parametrize('batched', [False, True])
def test_summary_and_thinning_match_every_eager_transition(batched):
    sampler = random_walk()
    start = Vector(jnp.zeros((4, 2) if batched else (2,)))
    key = jax.random.key(5)
    result = sampler.transformed_trajectory(
        12, start, key, thinning=3,
        readout=lambda parameter: jnp.sum(parameter.coordinate ** 2, axis=-1),
    )
    state, rng = sampler.initial_state(start), key
    coordinates, outcomes = [], []
    for _ in range(12):
        state, record, rng = sampler.step(state, rng)
        coordinates.append(state.parameter.coordinate)
        outcomes.append(record.outcome)

    coordinates = np.stack(coordinates)
    np.testing.assert_allclose(result.coordinates, coordinates[2::3], atol=1e-6)
    np.testing.assert_allclose(result.readouts, np.sum(coordinates ** 2, axis=-1),
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(result.sums['accepted'], np.sum(outcomes, axis=0))
    np.testing.assert_array_equal(
        jax.random.key_data(result.rng), jax.random.key_data(rng)
    )
    assert sampler.chain.length == 0
    assert sampler.lastState is None


def test_chunks_reuse_compilation_and_equal_one_trajectory():
    sampler = random_walk()
    start, key = Vector(jnp.zeros(2)), jax.random.key(11)
    traces = []

    def readout(parameter):
        traces.append(1)
        return parameter.coordinate.sum(axis=-1)

    run = sampler.compiled_trajectory(6, thinning=3, readout=readout)
    first = run(start, key)
    firstTraces = len(traces)
    second = run(first.final, first.rng)
    assert firstTraces > 0 and len(traces) == firstTraces
    whole = sampler.transformed_trajectory(12, start, key, thinning=3, readout=readout)

    np.testing.assert_allclose(np.concatenate([first.coordinates, second.coordinates]),
                               whole.coordinates, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.concatenate([first.readouts, second.readouts]),
                               whole.readouts, rtol=1e-5, atol=1e-6)
    assert first.sums['accepted'] + second.sums['accepted'] == whole.sums['accepted']


@pytest.mark.parametrize('nSteps,thinning', [(0, 1), (2.5, 1), (True, 1),
                                            (10, 4), (10, 0), (10, 2.5)])
def test_invalid_trajectory_lengths_are_rejected(nSteps, thinning):
    with pytest.raises(ValueError):
        random_walk().compiled_trajectory(nSteps, thinning)
