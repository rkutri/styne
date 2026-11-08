import numpy as np
import pytest

from styne.mcmc.method.mrw import MRWFactory
from styne.mcmc.method.pcn import PCNFactory
from styne.statistics.bayes import UnnormalisedPosterior
from styne.statistics.covariance import IIDCovarianceMatrix, DiagonalCovarianceMatrix
from styne.statistics.gaussian import Gaussian
from styne.statistics.likelihood import RegressionLikelihood
from styne.statistics.response import GaussianResponse
from tests.testSetup import (
    LotkaVolterraParameter, LotkaVolterraSolver, generate_synthetic_data,
)


@pytest.fixture
def inverse_problem():
    config = {
        'T': 6.0, 'alpha': 0.8, 'gamma': 0.4, 'nData': 5,
        'dataDim': 2, 'solver': 'LSODA', 'rtol': 1e-4,
    }
    design = np.array([[0.1, 0.9], [0.5, 0.5], [1.0, 0.5],
                       [0.5, 1.0], [2.5, 1.5]])
    solver = LotkaVolterraSolver(design, config)
    truth = LotkaVolterraParameter.from_interpolation(np.array([0.4, 0.6]))
    noiseVariance = 0.05
    priorVariance = 0.02
    data = generate_synthetic_data(
        truth, solver, noiseVariance, rng=np.random.default_rng(16)
    )
    prior = Gaussian(
        IIDCovarianceMatrix(2, priorVariance), LotkaVolterraParameter(np.zeros(2))
    )
    likelihood = RegressionLikelihood(
        data, solver, GaussianResponse(IIDCovarianceMatrix(10, noiseVariance))
    )
    return (
        UnnormalisedPosterior(prior, likelihood), solver, data,
        noiseVariance, priorVariance,
    )


@pytest.mark.parametrize('method', ['iid', 'diagonal', 'pcn'])
def test_ode_transition_matches_explicit_posterior(inverse_problem, method):
    target, solver, data, noiseVariance, priorVariance = inverse_problem
    if method == 'pcn':
        builder = PCNFactory()
        builder.beta = 0.2
    else:
        builder = MRWFactory()
        builder.proposalCovariance = (
            IIDCovarianceMatrix(2, 0.02) if method == 'iid'
            else DiagonalCovarianceMatrix(np.array([0.02, 0.01]))
        )
    builder.target = target
    rng = np.random.default_rng(17)
    builder.rng = rng
    sampler = builder.create()
    current = sampler.initial_state(LotkaVolterraParameter(np.array([-0.6, -0.3])))

    for _ in range(5):
        nextState, transition, rng = sampler.step(current, rng)
        likelihoods, posteriors = [], []
        for evaluated in (transition.current, transition.proposed):
            parameter = evaluated.parameter
            residual = (data.measurement - solver(parameter)).ravel()
            logLikelihood = -0.5 * np.dot(residual, residual) / noiseVariance
            coordinate = parameter.coordinate
            logPrior = -0.5 * np.dot(coordinate, coordinate) / priorVariance
            assert evaluated.logDensity == pytest.approx(logLikelihood)
            assert target.evaluate_log(parameter) == pytest.approx(
                logLikelihood + logPrior
            )
            likelihoods.append(logLikelihood)
            posteriors.append(logLikelihood + logPrior)

        values = likelihoods if method == 'pcn' else posteriors
        expected = min(0.0, values[1] - values[0])
        assert transition.logAcceptanceProbability == pytest.approx(expected)
        selected = transition.proposed if transition.outcome else transition.current
        np.testing.assert_array_equal(
            nextState.parameter.coordinate, selected.parameter.coordinate
        )
        current = nextState
