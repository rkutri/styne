import pytest
import numpy as np
from numpy.linalg import norm
from scipy.stats import ks_2samp

from tests.testSetup import (
    GaussianTargetDensity, RosenbrockTargetDensity, covariance_matrix,
)

from styne.parameter.vector import Vector
from styne.statistics.gaussian import GaussianDensity
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.mcmc.method.mrw import MRWFactory
from styne.mcmc.localised import (
    LocalisedSurrogateDensity,
    LocalisedSurrogateTransitionMeasure,
)


@pytest.mark.parametrize('target, dim, gamma, condition', [
    ('iid', 1, 500.0, 1),
    ('iid', 16, 50.0, 100),
    ('iid', 256, 0.5, 1000),
    ('iid', 16, 0.005, 1),
    ('gaussian', 1, 0.005, 1),
    ('gaussian', 16, 0.5, 100),
    ('gaussian', 256, 50.0, 1000),
    ('gaussian', 16, 500.0, 1000),
    ('rosenbrock', 2, 0.5, 1),
    ('rosenbrock', 16, 50.0, 100),
    ('rosenbrock', 256, 0.005, 1000),
    ('rosenbrock', 16, 500.0, 1000),
])
def test_localised_density_matches_quadratic_penalty(target, dim, gamma, condition):
    mode = Vector(np.ones(dim))
    if target == 'iid':
        baseDensity = GaussianDensity(IIDCovarianceMatrix(dim, float(condition)), mode)
    elif target == 'gaussian':
        baseDensity = GaussianTargetDensity(mode, covariance_matrix(dim, condition))
    else:
        baseDensity = RosenbrockTargetDensity(mode, condition)

    builder = MRWFactory()
    builder.target = LocalisedSurrogateDensity(gamma, 1.0, baseDensity)
    builder.proposalCovariance = IIDCovarianceMatrix(dim, 1.0)
    measure = LocalisedSurrogateTransitionMeasure(builder.create(), nChain=1)
    rng = np.random.default_rng(1234)

    for location in (Vector(np.zeros(dim)), Vector(rng.normal(1.0, 10.0, dim))):
        measure.location = location
        for coordinate in (location.coordinate, np.ones(dim), rng.normal(0.0, 20.0, dim)):
            parameter = Vector(coordinate)
            displacement = coordinate - location.coordinate
            penalty = -0.5 * gamma * np.dot(displacement, displacement)
            expected = baseDensity.evaluate_log(parameter) + penalty
            assert measure.density.evaluate_log(parameter) == pytest.approx(
                expected, rel=1e-5, abs=1e-8
            )


@pytest.mark.parametrize("kappa", [1.0, 5.0])
@pytest.mark.parametrize("h", [1.0, 0.1])
@pytest.mark.slow
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
        _, _, subchain, rng = surrogateMeasure.transition_trajectory(loc, rng)
        subchain = np.asarray(subchain)

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
    baseMeasure = Gaussian(baseCov, baseMean)
    
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
