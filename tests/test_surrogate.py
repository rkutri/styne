import pytest
import numpy as np
from math import isclose
from numpy.random import seed
from numpy.linalg import norm
from scipy.stats import ks_2samp

from styne.parameter.vector import Vector
from styne.statistics.gaussian import GaussianDensity
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.mcmc.method.mrw import MRWFactory
from styne.mcmc.localised import (
    LocalisedSurrogateDensity,
    LocalisedSurrogateTransitionMeasure,
)


@pytest.mark.parametrize("dim", [1, 16, 256])
@pytest.mark.parametrize("h", [500., 50.0, 0.5, 0.005])
@pytest.mark.parametrize("cond", [1, 100, 1000])
def test_density(dim, h, cond):
    seed(1111)

    modelCov = IIDCovarianceMatrix(dim, float(cond))
    modelMean = Vector(np.ones(dim))
    baseDensity = GaussianDensity(modelCov, modelMean)

    nLocs = 5
    rng = np.random.default_rng(1234)
    locations = [
        Vector(rng.normal(modelMean.coordinate, 10., size=dim))
        for _ in range(nLocs)
    ]

    initLoc = Vector(np.zeros(dim))
    surrogateDensity = LocalisedSurrogateDensity(h, 1.0, baseDensity)

    rootBuilder = MRWFactory()
    rootBuilder.proposalCovariance = IIDCovarianceMatrix(dim, 1e-10)
    rootBuilder.target = surrogateDensity

    rC = rootBuilder.create()
    nC = 1

    surrogateMeasure = LocalisedSurrogateTransitionMeasure(rC, nC)

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


@pytest.mark.parametrize("kappa", [1.0, 5.0])
@pytest.mark.parametrize("h", [1.0, 0.1])
def test_samples(kappa, h):
    """
    Overkill the subchain length and verify that the invariant measure
    approximates the correct density via two-sample KS test.
    """

    DIM = 2

    modelMean = Vector(np.zeros(DIM))
    modelCov = IIDCovarianceMatrix(DIM, float(kappa))

    rng = np.random.default_rng(4321)
    nLocs = 2
    locations = [
        Vector(rng.uniform(-8.0, 8.0, size=DIM))
        for _ in range(nLocs)
    ]

    baseDensity = GaussianDensity(modelCov, modelMean)
    surrogateDensity = LocalisedSurrogateDensity(h, 1.0, baseDensity)

    rootBuilder = MRWFactory()
    # A robust proposal scales with the target variance (1 / precision)
    precision = 1.0 / kappa + h
    rootBuilder.proposalCovariance = IIDCovarianceMatrix(
        DIM, float(0.5 / precision)
    )
    rootBuilder.target = surrogateDensity
    rootBuilder.rng = rng
    rootChain = rootBuilder.create()

    nChain = 5000

    surrogateMeasure = LocalisedSurrogateTransitionMeasure(
        rootChain,
        nChain,
    )

    for loc in locations:
        surrogateMeasure.mcmc.clear()
        surrogateMeasure.location = loc
        surrogateMeasure.generate_realisation()
        subchain = np.array(surrogateMeasure.chain.trajectory)

        burnin = 2000
        thinning = 5
        samples = np.array(subchain[burnin::thinning])

        # the target density represents the surrogate transition measure target
        # which is pi_k(u)^theta * N(u; loc, (1/gamma)*I)
        # Here theta=1.0, gamma=h, pi_k=N(0, kappa*I)
        # So pi_k = N(0, kappa)  (var=kappa, prec=1/kappa)
        # L2 term prec = h
        precision = 1.0 / kappa + h
        std = np.sqrt(1.0 / precision)
        mu = loc.coordinate * h / precision

        refSamples = rng.normal(mu, std, size=(len(samples), DIM))

        for dim in range(DIM):
            _, pval = ks_2samp(samples[:, dim], refSamples[:, dim])
            assert pval >= 1e-5, f"KS test failed for dim {dim}: p={pval:.5f}"


def test_surrogate_decomposition():
    from styne.statistics.radonnikodym import RadonNikodym
    from styne.utility.densityarithmetic import ProductWrapper
    dim = 2
    rng = np.random.default_rng(999)
    y = Vector(rng.normal(0, 1, size=dim))
    loc = Vector(rng.normal(0, 1, size=dim))
    
    from styne.statistics.gaussian import Gaussian
    baseCov = IIDCovarianceMatrix(dim, 1.0)
    baseMean = Vector(np.zeros(dim))
    baseMeasure = Gaussian(baseCov)
    baseMeasure.mean = baseMean
    
    from styne.statistics.interface import DensityInterface
    class DummyDerivative(DensityInterface):
        @property
        def domainDimension(self):
            return dim
            
        @property
        def domainType(self):
            return Vector
            
        def evaluate_log(self, v):
            return np.sum(v.coordinate)
    
    rnDensity = RadonNikodym(baseMeasure, DummyDerivative())
    
    gamma = 2.0
    theta = 0.5
    
    def check_density(density_obj):
        density_obj.location = loc
        full_eval = density_obj.evaluate_log(y)
        surr_eval = density_obj.evaluate_log_surrogate(y)
        reg_eval = density_obj._regGaussian.density.evaluate_log(y)
        from math import isclose
        assert isclose(full_eval - surr_eval, reg_eval, rel_tol=1e-10)

    # Case 1: Plain surrogate
    plain = LocalisedSurrogateDensity(gamma, theta, baseMeasure.density)
    check_density(plain)
    
    # Case 2: RN surrogate, temperFullDensity=False
    rn_false = LocalisedSurrogateDensity(
        gamma, theta, rnDensity, temperFullDensity=False
    )
    check_density(rn_false)
    
    # Case 3: RN surrogate, temperFullDensity=True
    rn_true = LocalisedSurrogateDensity(
        gamma, theta, rnDensity, temperFullDensity=True
    )
    check_density(rn_true)

    # Case 4: Weighted regularisation
    weights = rng.uniform(0.5, 2.0, size=dim)
    weighted = LocalisedSurrogateDensity(
        gamma, theta, baseMeasure.density, spectralWeights=weights
    )
    check_density(weighted)

