import pytest
import numpy as np
from numpy.linalg import norm
from scipy.stats import gaussian_kde

from tests.testSetup import (
    GaussianTargetDensity,
    covariance_matrix,
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
        _, _, subchain, rng = surrogateMeasure.transition_trajectory(loc, rng)
        subchain = np.asarray(subchain)
        density = surrogateMeasure.density.with_location(loc)

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
                    density.evaluate_log(Vector(coord))
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
