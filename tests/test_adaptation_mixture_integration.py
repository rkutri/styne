import numpy as np
import pytest

from styne.mcmc.diagnostics import DummyDiagnostics
from styne.mcmc.method.mrw import MetropolisedRandomWalk
from styne.mcmc.mixture import Mixture
from styne.parameter import Vector
from styne.utility import FullCovarianceAdaptation
from tests.test_mcmc_mixture import array, setup_mixture


@pytest.mark.parametrize("backendName, batchSize", [
    ("numpy", None), ("pytorch", None), ("jax", None), ("jax", 4),
])
def test_full_covariance_mixture_retains_private_history_and_resumes(
        backendName, batchSize):
    backend, start, target, base = setup_mixture(backendName)

    def cholesky_of(covariance):
        if batchSize is not None:
            return backend.vectorize(lambda item: item.to_cholesky())(covariance)
        return covariance.to_cholesky()

    adaptive = MetropolisedRandomWalk(
        target.density, base.components[0].proposal.covariance,
        DummyDiagnostics(), adaptation=FullCovarianceAdaptation(interval=2, nugget=0.1),
    )
    mixture = Mixture([adaptive, base.components[1]], [0.4, 0.6])
    rng = backend.random_state(39)
    if backendName == "jax":
        if batchSize is not None:
            start = Vector(backend.zeros((batchSize, 2), dtype="float32"))
        whole = mixture.compiled_trajectory(24)(start, rng)
        halfRun = mixture.compiled_trajectory(12)
        half = halfRun(start, rng)
        resumed = halfRun(half.finalState, half.rng)
        np.testing.assert_allclose(
            whole.coordinates,
            np.concatenate([half.coordinates, resumed.coordinates]),
        )
        state, selected = whole.finalState, array(whole.sums["selected"])
        np.testing.assert_allclose(
            cholesky_of(state.componentStates[0].adaptationState.covariance),
            cholesky_of(resumed.finalState.componentStates[0].adaptationState.covariance),
        )
    else:
        state = mixture.initial_state(start)
        selected = np.zeros(2, dtype=int)
        for _ in range(24):
            state, transition, rng = mixture.step(state, rng)
            selected[int(array(transition.component))] += 1

    full, scalar = (saved.adaptationState for saved in state.componentStates)
    np.testing.assert_array_equal(array(full.moments.nData), selected[..., 0])
    np.testing.assert_array_equal(array(scalar.stepCount), selected[..., 1])
    np.testing.assert_array_equal(selected.sum(axis=-1), 24)
    assert np.all(selected > 1)
    factor = array(cholesky_of(full.covariance))
    assert np.isfinite(factor).all()
    assert np.all(np.diagonal(factor, axis1=-2, axis2=-1) > 0)
