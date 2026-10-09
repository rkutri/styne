import pytest

import numpy as np

from unittest.mock import MagicMock
from styne.mcmc.method.mlda import MLDAFactory
from styne.statistics.bayes import UnnormalisedPosterior
from styne.statistics.covariance import IIDCovarianceMatrix
from tests.testSetup import GaussianTargetDensity
from styne.parameter.vector import Vector


@pytest.fixture
def setup_mlda_test_data():
    """
    Fixture to provide shared setup data for MLDA tests.
    """
    tgtMean = np.array([1.0, 1.5])
    tgtCov = np.array([[2.4, -0.5], [-0.5, 0.7]])

    baseSurrMean = tgtMean + np.array([-0.05, 0.01])
    fineSurrMean = tgtMean + np.array([0.0, -0.01])
    baseSurrCov = 3.0 * np.array([[2.8, -0.1], [-0.1, 1.7]])
    fineSurrCov = 1.5 * np.array([[2.4, -0.3], [-0.3, 1.1]])

    tgtDensity = GaussianTargetDensity(Vector(tgtMean), tgtCov)
    baseSurrDensity = GaussianTargetDensity(
        Vector(baseSurrMean), baseSurrCov
    )
    fineSurrDensity = GaussianTargetDensity(
        Vector(fineSurrMean), fineSurrCov
    )

    basePropCov = IIDCovarianceMatrix(len(tgtMean), 1.0)

    return {
        "tgtDensity": tgtDensity,
        "baseSurrDensity": baseSurrDensity,
        "fineSurrDensity": fineSurrDensity,
        "basePropCov": basePropCov,
        "tgtMean": tgtMean,
    }


@pytest.fixture
def mlda_chain_builder(setup_mlda_test_data):
    """
    Fixture to build the MLDA chain with the shared setup data.
    """
    data = setup_mlda_test_data
    chainBuilder = MLDAFactory()
    chainBuilder.target = data["tgtDensity"]
    chainBuilder.surrogate = [
        data["baseSurrDensity"],
        data["fineSurrDensity"]]
    chainBuilder.root.proposalCovariance = data["basePropCov"]
    chainBuilder.nChain = [6, 6]
    return chainBuilder


@pytest.mark.slow
def test_mlda_chain(setup_mlda_test_data, mlda_chain_builder):

    np.random.seed(42)

    # Build the MLDA method
    chainBuilder = mlda_chain_builder
    mcmc = chainBuilder.create()

    # Run the chain
    nChain = 3000
    initState = Vector(np.array([-8.0, -7.0]))
    mcmc.run(nChain, initState)

    # Extract results
    states = np.array(mcmc.chain.trajectory)
    diagnostics = mcmc.diagnostics

    # Assertions
    assert len(states) == nChain + 1, "Chain length mismatch with expected steps."
    assert 0.1 < diagnostics.global_acceptance_rate() < 0.9, \
        "Pathological acceptance rate for given parameters."

    burnin = 500
    fixedThinning = 5

    # Test convergence of mean estimate
    mean_estimate = np.mean(states[burnin::fixedThinning], axis=0)
    tgtMean = setup_mlda_test_data["tgtMean"]
    assert np.allclose(mean_estimate, tgtMean, atol=0.3), \
        f"Mean estimate {mean_estimate} deviates from target mean {tgtMean}."


def test_mlda_perfect_surrogate(setup_mlda_test_data):
    """
    Test MLDA chain where the base and fine surrogates match the target measure.
    Expect an acceptance rate of 1.
    """
    # Set seed for reproducibility
    np.random.seed(123)

    # Extract shared setup data
    data = setup_mlda_test_data

    # Surrogates are set to the exact target
    tgtDensity = data["tgtDensity"]
    baseSurrDensity = tgtDensity
    fineSurrDensity = tgtDensity
    basePropCov = data["basePropCov"]

    # Build MLDA chain
    chainBuilder = MLDAFactory()
    chainBuilder.target = tgtDensity
    chainBuilder.surrogate = [baseSurrDensity, fineSurrDensity]
    chainBuilder.root.proposalCovariance = basePropCov
    chainBuilder.nChain = [2, 10]

    mcmc = chainBuilder.create()

    # Run the chain
    nChain = 500
    initState = Vector(np.array([-8.0, -7.0]))
    mcmc.run(nChain, initState)

    # Extract diagnostics
    diagnostics = mcmc.diagnostics

    # Assertions
    assert np.abs(diagnostics.global_acceptance_rate() - 1.) < 1e-3, \
        "Proposals are rejected although the surrogates match the target."


@pytest.mark.slow
def test_mlda_two_level(setup_mlda_test_data):

    np.random.seed(456)

    # Extract shared setup data
    data = setup_mlda_test_data
    tgtDensity = data["tgtDensity"]

    # Create a single surrogate density
    surrogateMean = data["tgtMean"] + np.array([0.1, -0.2])
    surrogateCov = 1.5 * np.array([[2.5, -0.3], [-0.3, 0.9]])
    surrogateDensity = GaussianTargetDensity(
        Vector(surrogateMean), surrogateCov)

    basePropCov = data["basePropCov"]

    # Build two-level MLDA chain
    chainBuilder = MLDAFactory()
    chainBuilder.target = tgtDensity
    chainBuilder.surrogate = [surrogateDensity]
    chainBuilder.root.proposalCovariance = basePropCov
    chainBuilder.nChain = [10]

    mcmc = chainBuilder.create()

    # Run the chain
    nChain = 3000
    initState = Vector(np.array([-5.0, 4.0]))
    mcmc.run(nChain, initState)

    # Extract diagnostics
    diagnostics = mcmc.diagnostics
    acceptance_rate = diagnostics.global_acceptance_rate()

    # Postprocessing
    states = np.array(mcmc.chain.trajectory)
    burnin = 500
    thinningStep = 5
    mcmcSamples = states[burnin::thinningStep]

    meanEst = np.mean(mcmcSamples, axis=0)

    # Assertions
    assert len(
        mcmc.chain.trajectory) == nChain + 1, "Chain length mismatch for two-level method."

    assert 0.1 < acceptance_rate < 0.9, \
        f"Acceptance rate {acceptance_rate} for two-level method is outside " \
        "expected range."

    np.testing.assert_allclose(
        meanEst, data["tgtMean"], atol=0.3,
        err_msg="Estimated mean from two-level method deviates significantly from target mean."
    )


@pytest.mark.slow
def test_mlda_five_level_method(setup_mlda_test_data):
    """
    Test a five-level MLDA method with reasonable surrogate densities.
    """
    # Set seed for reproducibility
    np.random.seed(789)

    # Extract shared setup data
    data = setup_mlda_test_data
    tgtDensity = data["tgtDensity"]

    # Create surrogate densities for five levels
    surrogateMeans = [
        data["tgtMean"] + np.array([0.2, -0.3]),
        data["tgtMean"] + np.array([-0.1, 0.1]),
        data["tgtMean"] + np.array([0.05, -0.05]),
        data["tgtMean"] + np.array([0.0, 0.0]),
    ]
    surrogateCovs = [
        8. * np.array([[3.0, -0.2], [-0.2, 1.5]]),
        6. * np.array([[2.7, -0.25], [-0.25, 1.3]]),
        4. * np.array([[2.5, -0.3], [-0.3, 1.1]]),
        2. * np.array([[2.4, -0.35], [-0.35, 1.0]]),
    ]
    surrogateDensities = [
        GaussianTargetDensity(Vector(mean), cov)
        for mean, cov in zip(surrogateMeans, surrogateCovs)
    ]

    basePropCov = data["basePropCov"]

    # Build five-level MLDA chain
    chainBuilder = MLDAFactory()
    chainBuilder.target = tgtDensity
    chainBuilder.surrogate = surrogateDensities
    chainBuilder.root.proposalCovariance = basePropCov
    chainBuilder.nChain = [6, 4, 4, 3]

    mcmc = chainBuilder.create()

    # Run the chain
    nChain = 3000
    initState = Vector(np.array([2.0, -3.0]))
    mcmc.run(nChain, initState)

    # Extract diagnostics
    diagnostics = mcmc.diagnostics
    acceptance_rate = diagnostics.global_acceptance_rate()

    # Postprocessing
    states = np.array(mcmc.chain.trajectory)
    burnin = 200
    thinningStep = 3
    mcmcSamples = states[burnin::thinningStep]

    meanState = np.mean(states, axis=0)
    meanEst = np.mean(mcmcSamples, axis=0)

    # Assertions
    assert len(
        mcmc.chain.trajectory) == nChain + 1, "Chain length mismatch for five-level method."
    assert 0.1 < acceptance_rate < 0.9, \
        f"Acceptance rate {acceptance_rate} for five-level method is " \
        "outside expected range."
    np.testing.assert_allclose(
        meanEst, data["tgtMean"], atol=0.3,
        err_msg="Estimated mean from five-level method deviates significantly from target mean."
    )


def test_mlda_multilevel_levels_are_pure():
    from styne.mcmc.method.mlda import MLDAFactory
    from styne.statistics.gaussian import GaussianDensity
    from styne.statistics.covariance import IIDCovarianceMatrix
    from styne.mcmc.diagnostics import PersistentAcceptanceRateDiagnostics
    from styne.parameter.vector import Vector
    import numpy as np

    targetDensity = GaussianDensity(IIDCovarianceMatrix(2, 1.0), Vector(np.zeros(2)))
    surrogate1 = GaussianDensity(IIDCovarianceMatrix(2, 2.0), Vector(np.zeros(2)))
    surrogate2 = GaussianDensity(IIDCovarianceMatrix(2, 3.0), Vector(np.zeros(2)))

    factory = MLDAFactory(root="mrw")
    factory.target = targetDensity
    factory.surrogate = [surrogate1, surrogate2]
    factory.nChain = [2, 2]
    factory.root.proposalCovariance = IIDCovarianceMatrix(2, 1.0)
    factory.subDiagnostics = PersistentAcceptanceRateDiagnostics

    mainChain = factory.create()
    assert len(mainChain.subsamplers) == 2   # one per surrogate

    mainChain.run(5, Vector(np.zeros(2)))
    for subsampler in mainChain.subsamplers:
        assert subsampler.diagnostics._total == 0
        assert len(subsampler.chain.trajectory) == 0
