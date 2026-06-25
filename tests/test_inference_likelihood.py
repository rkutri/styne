import numpy as np
import pytest

from styne.statistics.data import Data
from styne.statistics.likelihood import RegressionLikelihood
from styne.statistics.response import GaussianResponse
from styne.statistics.gaussian import Gaussian
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.parameter.vector import Vector
from styne.model.model import Model

# Local fixtures were moved to conftest.py



def test_initialisation(mock_likelihood, mock_data, mock_forward_model):

    assert mock_likelihood.data is mock_data
    assert mock_likelihood.model is mock_forward_model
    assert isinstance(mock_likelihood.response, GaussianResponse)


@pytest.mark.skip()
def test_memoisation(mock_likelihood):

    parameter = Vector(np.array([0.5, 0.5]))
    logLFirst = mock_likelihood.evaluate_log(parameter)
    logLCached = mock_likelihood.evaluate_log(parameter)

    assert logLFirst == logLCached
    assert mock_likelihood._logLikelihoodCache.contains(parameter)

    assert mock_likelihood._logLikelihoodCache.retrieve(parameter) == logLFirst


@pytest.mark.skip()
def test_cache_eviction():

    from styne.utility.memoisation import EvaluationCache

    cache = EvaluationCache(2)

    param1 = Vector(np.array([0.1, 0.1]))
    param2 = Vector(np.array([0.2, 0.2]))
    param3 = Vector(np.array([0.3, 0.3]))

    cache.add(param1, 1.0)
    cache.add(param2, 2.0)

    assert cache.contains(param1)
    assert cache.contains(param2)

    cache.add(param3, 3.0)

    assert not cache.contains(param1)
    assert cache.contains(param2)
    assert cache.contains(param3)
    assert cache.retrieve(param2) == 2.0
    assert cache.retrieve(param3) == 3.0


@pytest.mark.skip()
def test_stress_test_memoisation(mock_noise, mock_forward_model):

    np.random.seed(19)

    numTests = 10000
    cacheSize = 2
    paramSize = 2

    mockData = Data(2, np.zeros((1, 2)))
    mockData.measurement = np.zeros((1, 2))
    likelihood = RegressionLikelihood(
        mockData, mock_forward_model, GaussianResponse(mock_noise.density.covariance)
    )

    from styne.utility.memoisation import EvaluationCache
    likelihood._logLikelihoodCache = EvaluationCache(cacheSize)

    cacheHits = 0

    for i in range(numTests):

        if (i == 0):

            paramOld = Vector(np.random.rand(paramSize))
            likelihood.evaluate_log(paramOld)

        paramNew = Vector(np.random.rand(paramSize))

        likelihood.evaluate_log(paramNew)

        if likelihood._logLikelihoodCache.contains(paramOld):

            cacheHits += 1

            logLCached = likelihood._logLikelihoodCache.retrieve(paramOld)
            logL = likelihood.evaluate_log(paramOld)

            assert logLCached == logL

        if likelihood._logLikelihoodCache.contains(paramNew):

            cacheHits += 1

            logLCached = likelihood._logLikelihoodCache.retrieve(paramNew)
            logL = likelihood.evaluate_log(paramNew)

            assert logLCached == logL

        paramOld = paramNew

    cacheHitPerc = 50. * cacheHits / numTests
    print(
        f"number of cache hits: {cacheHits}. Corresponds to {cacheHitPerc} %")

    cacheTOL = 1e-3
    assert cacheHits <= 2. * (1. + cacheTOL) * numTests
