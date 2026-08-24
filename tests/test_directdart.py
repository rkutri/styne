import numpy as np
import pytest

from styne.backend import BackendUnavailableError, get_backend
from styne.mcmc.diagnostics import DummyDiagnostics
from styne.mcmc.method.dartdirect import DirectDART, DirectDARTProposal
from styne.mcmc.transition import TransitionData
from styne.parameter import Vector
from styne.statistics import DenseCovarianceMatrix, Gaussian


def test_directdart_proposal_mean():
    tempering = 0.5
    gamma = 0.1
    surrogateMean = Vector(np.array([1.0, -1.0]))
    covarianceArray = np.array([[2.0, 0.5], [0.5, 1.0]])
    surrogate = Gaussian(
        DenseCovarianceMatrix(covarianceArray), surrogateMean
    )
    precision = np.linalg.inv(covarianceArray)
    proposalPrecision = tempering * precision + gamma * np.eye(2)
    proposalCovariance = np.linalg.inv(proposalPrecision)
    proposal = DirectDARTProposal(
        tempering,
        gamma,
        surrogate,
        DenseCovarianceMatrix(proposalCovariance),
    )
    state = Vector(np.array([0.5, 0.5]))

    transition, _ = proposal.propose(state, np.random.default_rng(0))

    expectedB = (
        tempering * precision @ surrogateMean.coordinate
        + gamma * state.coordinate
    )
    expectedMean = proposalCovariance @ expectedB
    assert np.allclose(transition.auxiliary['bx'], expectedB)
    assert np.allclose(transition.auxiliary['mux'], expectedMean)


def test_directdart_mh_ratio():
    tempering = 0.5
    gamma = 0.1
    surrogateMean = Vector(np.array([1.0, -1.0]))
    covarianceArray = np.array([[2.0, 0.5], [0.5, 1.0]])
    surrogate = Gaussian(
        DenseCovarianceMatrix(covarianceArray), surrogateMean
    )
    precision = np.linalg.inv(covarianceArray)
    proposalCovariance = np.linalg.inv(
        tempering * precision + gamma * np.eye(2)
    )
    sampler = DirectDART(
        surrogate.density,
        tempering,
        gamma,
        surrogate,
        DenseCovarianceMatrix(proposalCovariance),
        DummyDiagnostics(),
    )
    state = Vector(np.array([0.5, 0.5]))
    proposed = Vector(np.array([0.2, 0.8]))
    bx = (
        tempering * precision @ surrogateMean.coordinate
        + gamma * state.coordinate
    )
    mux = proposalCovariance @ bx
    transition = TransitionData(
        current=sampler.evaluate_state(state),
        proposed=sampler.evaluate_state(proposed),
        auxiliary={'bx': bx, 'mux': mux},
    )
    bz = (
        tempering * precision @ surrogateMean.coordinate
        + gamma * proposed.coordinate
    )
    muz = proposalCovariance @ bz
    stateDifference = state.coordinate - surrogateMean.coordinate
    proposalDifference = proposed.coordinate - surrogateMean.coordinate
    expected = (
        surrogate.density.evaluate_log(proposed)
        - surrogate.density.evaluate_log(state)
        + 0.5 * tempering * (
            proposalDifference @ precision @ proposalDifference
            - stateDifference @ precision @ stateDifference
        )
        + 0.5 * (mux @ bx - muz @ bz)
        - 0.5 * gamma * (
            state.coordinate @ state.coordinate
            - proposed.coordinate @ proposed.coordinate
        )
    )

    assert sampler._log_mh_ratio(transition) == pytest.approx(expected)


def make_directdart(backend):
    dtype = 'float32' if backend.name == 'jax' else 'float64'
    covariance = backend.asarray(np.eye(2), dtype=dtype)
    surrogate = Gaussian(
        DenseCovarianceMatrix(covariance),
        Vector(backend.asarray(np.zeros(2), dtype=dtype)),
    )
    proposalCovariance = DenseCovarianceMatrix(
        backend.asarray(np.eye(2) / 0.6, dtype=dtype)
    )
    return DirectDART(
        surrogate.density,
        0.5,
        0.1,
        surrogate,
        proposalCovariance,
        DummyDiagnostics(),
    )


@pytest.mark.parametrize('backendName', ('numpy', 'jax', 'pytorch'))
def test_directdart_supports_batched_backends(backendName):
    if backendName == 'jax':
        pytest.importorskip('jax')
    elif backendName == 'pytorch':
        pytest.importorskip('torch')
    try:
        backend = get_backend(backendName)
    except BackendUnavailableError as error:
        pytest.skip(str(error))
    sampler = make_directdart(backend)
    dtype = 'float32' if backend.name == 'jax' else 'float64'
    initial = Vector(backend.asarray(
        [[1.0, -1.0], [0.5, 0.25]], dtype=dtype
    ))

    nextState, transition, _ = sampler.step(
        sampler.initial_state(initial), backend.random_state(13)
    )

    assert nextState.parameter.coordinate.shape == (2, 2)
    assert transition.outcome.shape == (2,)
