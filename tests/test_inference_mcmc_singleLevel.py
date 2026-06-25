import pytest
import tests.testSetup as setup

import numpy as np

from numpy.random import seed
from styne.statistics.covariance import IIDCovarianceMatrix, DiagonalCovarianceMatrix
from styne.statistics.gaussian import Gaussian
from styne.statistics.likelihood import RegressionLikelihood
from styne.statistics.response import GaussianResponse
from styne.mcmc.method.mrw import MRWFactory
from styne.mcmc.method.pcn import PCNFactory
from styne.statistics.bayes import UnnormalisedPosterior


def check_mean(means, trueParam):

    MTOL = 0.5

    meanState = means[0]
    posteriorMean = means[1]

    np.allclose(posteriorMean.coordinate, trueParam.coordinate, atol=MTOL)
    np.allclose(meanState.coordinate, trueParam.coordinate, atol=2. * MTOL)


# define forward problem
config = {
    'T': 6.,
    'alpha': 0.8,
    'gamma': 0.4,
    'nData': 5,
    'dataDim': 2,
    'solver': 'LSODA',
    'rtol': 1e-4}
design = np.array([
    np.array([0.1, 0.9]),
    np.array([0.5, 0.5]),
    np.array([1., 0.5]),
    np.array([0.5, 1.]),
    np.array([2.5, 1.5])
])

PDIM = 2

solver = setup.LotkaVolterraSolver(design, config)

# define problem parameters
groundTruth = setup.LotkaVolterraParameter.from_interpolation(
    np.array([0.4, 0.6]))

dataNoiseVariance = 0.05
data = setup.generate_synthetic_data(groundTruth, solver, dataNoiseVariance)

# start with a prior centred around the true parameter coefficient
priorMean = setup.LotkaVolterraParameter.from_coefficient(np.zeros(2))

priorMargVar = 0.02
priorCovariance = IIDCovarianceMatrix(PDIM, priorMargVar)

# set up prior: Gaussian(covariance, mean)
prior = Gaussian(priorCovariance, priorMean)

# define a noise model: Gaussian with no mean (set automatically by likelihood)
noiseMVar = dataNoiseVariance
noiseCov = IIDCovarianceMatrix(PDIM, noiseMVar)
noiseModel = Gaussian(noiseCov)

# define the statistical inverse problem
likelihood = RegressionLikelihood(
    data, solver, GaussianResponse(IIDCovarianceMatrix(10, noiseMVar)))
statModel = UnnormalisedPosterior(prior, likelihood)


@pytest.mark.parametrize("mcmcProposal", ["iid", "indep"])
def test_mrw(mcmcProposal):

    seed(16)

    chainBuilder = MRWFactory()

    if (mcmcProposal == 'iid'):

        proposalMargVar = 0.02
        proposalCov = IIDCovarianceMatrix(PDIM, proposalMargVar)

    elif (mcmcProposal == 'indep'):

        proposalMargVar = np.array([0.02, 0.01])
        proposalCov = DiagonalCovarianceMatrix(proposalMargVar)

    else:
        raise Exception("Proposal " + mcmcProposal + " not implemented")

    chainBuilder.proposalCovariance = proposalCov
    chainBuilder.target = statModel

    mcmc = chainBuilder.create()

    # run mcmc
    nSteps = 200
    initState = setup.LotkaVolterraParameter.from_coefficient(
        np.array([-0.6, -0.3]))
    mcmc.run(nSteps, initState)

    states = mcmc.chain.trajectory

    burnIn = 200
    thinningStep = 5

    mcmcSamples = states[burnIn::thinningStep]
    meanState = setup.LotkaVolterraParameter.from_coefficient(
        np.mean(states, axis=0))
    posteriorMean = setup.LotkaVolterraParameter.from_coefficient(
        np.mean(mcmcSamples, axis=0))

    check_mean([meanState, posteriorMean], groundTruth)


@pytest.mark.skip(reason="PCN is not maintained currently")
def test_pcn():

    seed(17)

    chainBuilder = PCNFactory()
    chainBuilder.beta = 0.001
    chainBuilder.target = statModel

    mcmc = chainBuilder.create()

    # run mcmc
    nSteps = 200
    initState = setup.LotkaVolterraParameter.from_coefficient(
        np.array([-0.6, -0.3]))
    mcmc.run(nSteps, initState)

    states = mcmc.chain.trajectory

    burnIn = 200
    thinningStep = 5

    mcmcSamples = states[burnIn::thinningStep]
    meanState = setup.LotkaVolterraParameter.from_coefficient(
        np.mean(states, axis=0))
    posteriorMean = setup.LotkaVolterraParameter.from_coefficient(
        np.mean(mcmcSamples, axis=0))

    check_mean([meanState, posteriorMean], groundTruth)


if __name__ == "__main__":
    pytest.main()
