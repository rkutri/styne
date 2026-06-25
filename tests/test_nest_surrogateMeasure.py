import pytest
import numpy as np
from math import isclose
from numpy.random import default_rng
from numpy.linalg import norm
from scipy.stats import gaussian_kde

from tests.testSetup import (
    GaussianTargetDensity,
    covariance_matrix,
    RosenbrockTargetDensity
)
from styne.parameter.vector import Vector
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.mcmc.method.mrw import MRWFactory
from styne.utility.postprocessing import integrated_autocorrelation
from styne.mcmc.localised import (
    LocalisedSurrogateDensity,
    LocalisedSurrogateTransitionMeasure,
)
from styne.statistics.gaussian import Gaussian


@pytest.mark.parametrize("dim", [1, 16, 256])
@pytest.mark.parametrize("h", [500., 50.0, 0.5, 0.005])
@pytest.mark.parametrize("cond", [1, 100, 1000])
@pytest.mark.parametrize("target", ["gaussian", "rosenbrock"])
def test_density(dim, h, cond, target):
    """
    Compare log densities point-wise.
    """

    rng_setup = default_rng(1111)

    modelMode = Vector(np.ones(dim))

    if target == "gaussian":
        modelCov = covariance_matrix(dim, cond)
        baseDensity = GaussianTargetDensity(modelMode, modelCov)
    elif target == "rosenbrock":
        baseDensity = RosenbrockTargetDensity(modelMode, cond)
    else:
        raise Exception(f"Unknown target density: {target}")

    # draw random surrogate Measure locations
    nLocs = 5
    rng = np.random.default_rng(1234)
    locations = [
        Vector(rng.normal(modelMode.coordinate, 10., size=dim))
        for _ in range(nLocs)
    ]

    initLoc = Vector(np.zeros(dim))
    surrogateDensity = LocalisedSurrogateDensity(h, 1.0, baseDensity)
    surrogateDensity.location = initLoc

    # deliberately invalid covariance (not used in sampling)
    rootBuilder = MRWFactory()
    rootBuilder.rng = rng_setup
    rootBuilder.proposalCovariance = IIDCovarianceMatrix(dim, 1e-12)
    rootBuilder.target = surrogateDensity

    iC = IIDCovarianceMatrix(dim, 1e-12)
    # Wrap in Gaussian to satisfy ProbabilityMeasure interface
    initMeas = Gaussian(iC, Vector(np.zeros(dim)))
    rC = rootBuilder.create()
    nC = 0

    surrogateMeasure = LocalisedSurrogateTransitionMeasure(rC, nC, initMeas)

    nEval = 100
    for loc in locations:

        def true_density_log(z: Vector):
            y = z.coordinate - loc.coordinate
            return -0.5 * h * np.dot(y, y) + baseDensity.evaluate_log(z)

        surrogateMeasure.location = loc

        xEval = [
            Vector(rng.normal(np.zeros(dim), 20., size=dim))
            for _ in range(nEval)
        ]

        for x in xEval:
            smEval = surrogateMeasure.density.evaluate_log(x)
            trueEval = true_density_log(x)
            assert isclose(smEval, trueEval, rel_tol=1e-5, abs_tol=1e-8)


@pytest.mark.slow
@pytest.mark.parametrize("kappa", [1.0, 5.0])
@pytest.mark.parametrize("h", [1.0, 0.1])
def test_samples(kappa, h):
    """
    Overkill the subchain length and verify that the invariant measure
    approximates the correct density.
    """

    DIM = 2
    RTOL = 0.5

    modelMean = Vector(np.zeros(DIM))
    modelCov = covariance_matrix(DIM, kappa)
    print(modelCov)

    rng = np.random.default_rng(4349)
    nLocs = 2
    locations = [
        Vector(rng.uniform(-8.0, 8.0, size=DIM))
        for _ in range(nLocs)
    ]

    baseDensity = GaussianTargetDensity(modelMean, modelCov)
    surrogateDensity = LocalisedSurrogateDensity(h, 1.0, baseDensity)
    surrogateDensity.location = locations[0]

    rootBuilder = MRWFactory()
    rootBuilder.rng = rng
    precision = 1.0 / kappa + h
    rootBuilder.proposalCovariance = IIDCovarianceMatrix(
        DIM, float(0.5 / precision)
    )
    rootBuilder.target = surrogateDensity
    rootChain = rootBuilder.create()

    initMeasCov = IIDCovarianceMatrix(DIM, h / kappa)
    initMeas = Gaussian(initMeasCov, Vector(np.zeros(DIM)))
    nChain = int(kappa) * int(5000)

    surrogateMeasure = LocalisedSurrogateTransitionMeasure(
        rootChain,
        nChain,
        initMeas,
    )

    for loc in locations:
        surrogateMeasure.location = loc
        surrogateMeasure.generate_realisation()
        subchain = np.array(surrogateMeasure.chain.trajectory)

        burnin = 1000
        thinning = int(np.ceil(integrated_autocorrelation(subchain[burnin:], "max")))
        samples = np.array(subchain[burnin::thinning])

        nGridPerDim = 50
        axes = [
            np.linspace(
                loc.coordinate[i] - 5.0,
                loc.coordinate[i] + 5.0,
                nGridPerDim,
            )
            for i in range(DIM)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        mesh = np.vstack([m.ravel() for m in mesh])

        kde = gaussian_kde(samples.T)
        densityEst = kde(mesh)

        targetDensity = np.exp(
            np.array(
                [
                    surrogateMeasure.density.evaluate_log(
                        Vector(coord))
                    for coord in mesh.T
                ]
            )
        )

        densityEst /= np.sum(densityEst) * (10.0 / nGridPerDim) ** DIM
        targetDensity /= np.sum(targetDensity) * (10.0 / nGridPerDim) ** DIM

        error = norm(densityEst - targetDensity) / norm(targetDensity)
        assert error < RTOL


def test_localised_density_iid_backward_compatible():
    """No-weights constructor is backward-compatible: penalty is IID in xi-space."""
    from tests.testSetup import GaussianTargetDensity

    dim = 4
    gamma = 1.0
    modelMode = Vector(np.zeros(dim))
    baseDensity = GaussianTargetDensity(modelMode, np.eye(dim))

    # Old-style construction: no spectralWeights argument
    density = LocalisedSurrogateDensity(gamma, 1.0, baseDensity)

    location = Vector(np.array([1.0, 0.5, -0.5, 0.0]))
    density.location = location

    d = np.array([1.0, 0.0, 0.0, 0.0])
    pt = Vector(location.coordinate + d)

    # IID: penalty is -gamma/2 * ||d||^2 = -0.5
    expected = -gamma / 2.0 * np.dot(d, d)
    # access the regularisation Gaussian density directly
    actual = (
        density._regGaussian.density.evaluate_log(pt)
        - density._regGaussian.density.evaluate_log(location)
    )
    assert np.isclose(actual, expected, rtol=1e-10)
    assert density.spectralWeights is None

