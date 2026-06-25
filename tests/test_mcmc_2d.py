import pytest
import numpy as np

from numpy.random import default_rng
from tests.testSetup import GaussianTargetDensity
from styne.statistics.covariance import IIDCovarianceMatrix, DiagonalCovarianceMatrix
from styne.mcmc.method.mrw import MetropolisedRandomWalk
from styne.mcmc.diagnostics import *
from styne.parameter.vector import Vector


@pytest.mark.parametrize("mcmcProposal", ["iid", "indep"])
@pytest.mark.parametrize("Diagnostics",
                         [DummyDiagnostics, AcceptanceRateDiagnostics,
                          FullDiagnostics])
def test_moments(mcmcProposal, Diagnostics):

    rng = default_rng(116)

    # Target distribution setup
    tgtMean = Vector(np.array([1., 1.5]))
    tgtCov = np.array(
        [[1.2, -0.2],
         [-0.2, 0.4]]
    )
    tgtDensity = GaussianTargetDensity(tgtMean, tgtCov)

    # Proposal distribution setup based on mcmcProposal
    if mcmcProposal == 'iid':

        proposalMargVar = 0.25
        proposalCov = IIDCovarianceMatrix(tgtMean.dimension, proposalMargVar)

    elif mcmcProposal == 'indep':

        proposalMargVar = np.array([tgtCov[0, 0], tgtCov[1, 1]])
        proposalCov = DiagonalCovarianceMatrix(proposalMargVar)

    else:
        raise Exception(f"Proposal {mcmcProposal} not implemented")

    diagnostics = Diagnostics()

    mcmc = MetropolisedRandomWalk(tgtDensity, proposalCov, diagnostics, rng=rng)

    # MCMC run setup
    nSteps = 2000
    initState = Vector(np.array([-2., 0.]))
    mcmc.run(nSteps, initState)

    assert len(mcmc.chain.trajectory) == nSteps + 1

    states = np.array(mcmc.chain.trajectory)

    # Postprocessing
    burnin = 200
    thinningStep = 5

    mcmcSamples = states[burnin::thinningStep]

    # Moment tests
    meanState = np.mean(states, axis=0)
    meanEst = np.mean(mcmcSamples, axis=0)

    stateCov = np.cov(states, rowvar=False)
    sampleCov = np.cov(mcmcSamples, rowvar=False)

    MTOL = 0.5
    CTOL = 0.5

    assert np.allclose(meanState, tgtMean.coordinate, atol=MTOL), \
        f"Mean state does not match target mean with " \
        "mcmcProposal='{mcmcProposal}'"
    assert np.allclose(meanEst, tgtMean.coordinate, atol=MTOL), \
        f"Mean estimate does not match target mean with " \
        "mcmcProposal='{mcmcProposal}'"

    assert np.allclose(sampleCov, tgtCov, atol=CTOL), \
        f"Sample covariance does not match target covariance with " \
        "mcmcProposal='{mcmcProposal}'"
    assert np.allclose(stateCov, tgtCov, atol=2. * CTOL), \
        f"State covariance does not match target covariance with " \
        "mcmcProposal='{mcmcProposal}'"



if __name__ == "__main__":
    pytest.main()
