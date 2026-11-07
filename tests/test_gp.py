import importlib

import numpy as np
import pytest

from numpy.random import default_rng

import styne.gp as gp_module
from styne.statistics.stationary import (
    ExponentialCovariance1D, MaternCovariance1D, MaternCovariance2D,
    matern_covariance, matern_log_rho_gradient
)
from styne.statistics.gaussian import Gaussian
from styne.model.representation.bspline import BSpline1D, BSpline2D
from styne.gp.gaussianprocess import GaussianProcess
from styne.gp.direct import DirectExpansion
from styne.utility.grid import UniformGrid


LB, RB = 0., 1.


def test_gp_package_exports_expansions_without_legacy_realisations():
    assert gp_module.DirectExpansion is DirectExpansion
    assert "DNAFourierExpansion" in gp_module.__all__
    for removed_name in (
            "DirectRealisation", "BSplineRealisation1D",
            "BSplineRealisation2D", "DNAFourierRealisation",
            "GPEngine", "DirectGPEngine", "BSplineGPEngine",
            "DNAFourierEngine"):
        assert removed_name not in gp_module.__all__
        assert not hasattr(gp_module, removed_name)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("styne.gp.engine")
    assert not hasattr(
        importlib.import_module("styne.gp.direct"), "DirectGPEngine"
    )
    assert not hasattr(
        importlib.import_module("styne.gp.bspline"), "BSplineGPEngine"
    )
    assert not hasattr(
        importlib.import_module("styne.gp.dna"), "DNAFourierEngine"
    )


# ---------------------------------------------------------------------------
# GaussianProcess.dense
# ---------------------------------------------------------------------------

def test_dense_gp_sampler_shape():
    n = 10
    grid = UniformGrid(LB, RB, n)
    cov = ExponentialCovariance1D(alpha=5., marginalVariance=1.)
    gp = GaussianProcess.direct(grid, cov)
    function = gp.sampler.generate_realisation(seed=0)
    assert isinstance(function.expansion, DirectExpansion)
    assert function.coordinate.shape == (n,)


def test_evaluate_uses_explicit_coefficient_without_mutating_coefficient():
    grid = UniformGrid(0., 1., 5)
    gp = GaussianProcess.direct(
        grid, MaternCovariance1D(0.3, 1.5, 1.0))
    coefficient = np.linspace(-0.5, 0.5, gp.parameterDimension)
    before = coefficient.copy()

    values = gp.evaluate(coefficient, grid)

    assert values.shape == (len(grid),)
    np.testing.assert_array_equal(coefficient, before)


def test_dense_gp_measure_is_gaussian():
    grid = UniformGrid(LB, RB, 5)
    cov = ExponentialCovariance1D(alpha=5., marginalVariance=1.)
    gp = GaussianProcess.direct(grid, cov)
    assert isinstance(gp.measure, Gaussian)


def test_dense_gp_zero_mean():
    n, nSamples, sigma2 = 10, 500, 0.7
    rng = default_rng(42)
    grid = UniformGrid(LB, RB, n)
    cov = ExponentialCovariance1D(alpha=5., marginalVariance=sigma2)
    gp = GaussianProcess.direct(grid, cov)
    sampler = gp.sampler

    samples = np.array(
        [sampler.draw(rng).evaluate(gp.nativeGrid) for _ in range(nSamples)])
    mean = np.mean(samples, axis=0)

    assert np.max(np.abs(mean)) < 3. * np.sqrt(sigma2 / nSamples), (
        f"Empirical mean too large: max |mean| = {np.max(np.abs(mean)):.3f}"
    )


def test_dense_gp_marginal_variance():
    n, nSamples, sigma2 = 10, 500, 0.7
    rng = default_rng(43)
    grid = UniformGrid(LB, RB, n)
    cov = ExponentialCovariance1D(alpha=5., marginalVariance=sigma2)
    gp = GaussianProcess.direct(grid, cov)
    sampler = gp.sampler

    samples = np.array(
        [sampler.draw(rng).evaluate(gp.nativeGrid) for _ in range(nSamples)])
    variances = np.var(samples, axis=0)

    rel_err = np.max(np.abs(variances - sigma2) / sigma2)
    assert rel_err < 0.4, (
        f"Marginal variance mismatch: max rel. error = {rel_err:.3f}"
    )


def test_dense_gp_parameter_reconstruction_shares_representation():
    n = 8
    grid = UniformGrid(LB, RB, n)
    cov = ExponentialCovariance1D(alpha=5., marginalVariance=1.)
    gp = GaussianProcess.direct(grid, cov)

    coords = np.arange(n, dtype=float)
    replacement = gp.function(coords)

    np.testing.assert_array_equal(
        replacement.coordinate, coords,
        err_msg="replacement did not preserve its coordinate"
    )
    assert replacement.expansion is gp.expansion


def test_dense_gp_covariance_update_is_functional():
    n = 6
    grid = UniformGrid(LB, RB, n)
    cov1 = ExponentialCovariance1D(alpha=5., marginalVariance=1.)
    cov2 = ExponentialCovariance1D(alpha=2., marginalVariance=0.5)

    gp = GaussianProcess.direct(grid, cov1)
    measureRef = gp.measure

    updated = gp.with_covariance_function(cov2)

    assert gp.measure is measureRef
    assert updated.measure is not measureRef
    assert gp.covarianceFunction is cov1
    assert updated.covarianceFunction is cov2


# ---------------------------------------------------------------------------
# GaussianProcess.bspline_1d
# ---------------------------------------------------------------------------

def test_bspline_1d_gp_measure_dimension():
    nx = 8
    cov = MaternCovariance1D(lengthScale=0.3, smoothness=1.5, marginalVariance=1.)
    expansion = BSpline1D(nx, degree=3, boundary=[LB, RB])

    gp = GaussianProcess.bspline(cov, expansion)

    assert gp.measure.density.domainDimension == nx


def test_bspline_1d_gp_sampler_returns_function():
    n, nx = 32, 6
    grid = UniformGrid(LB, RB, n)
    cov = MaternCovariance1D(lengthScale=0.3, smoothness=1.5, marginalVariance=1.)
    expansion = BSpline1D(nx, degree=3, boundary=[LB, RB])

    gp = GaussianProcess.bspline(cov, expansion)
    function = gp.sampler.generate_realisation(seed=7)

    assert function.expansion is expansion
    assert function.evaluate(grid).shape == (n,)


# ---------------------------------------------------------------------------
# GaussianProcess.bspline_2d
# ---------------------------------------------------------------------------

def make_bspline_2d_gp(nx=5, ny=5):
    cov = MaternCovariance2D(lengthScale=0.3, smoothness=1.5, marginalVariance=1.)
    expansion = BSpline2D([nx, ny], degree=3, boundary=[[LB, RB], [LB, RB]])
    return GaussianProcess.bspline(cov, expansion)


def test_bspline_2d_gp_measure_dimension():
    nx, ny = 5, 5
    gp = make_bspline_2d_gp(nx, ny)
    assert gp.measure.density.domainDimension == nx * ny


def test_bspline_2d_gp_measure_is_gaussian():
    gp = make_bspline_2d_gp()
    assert isinstance(gp.measure, Gaussian)


def test_bspline_2d_gp_sampler_shares_expansion():
    gp = make_bspline_2d_gp()
    function = gp.sampler.generate_realisation(seed=13)
    assert function.expansion is gp.expansion


def test_bspline_2d_gp_sampler_evaluate():
    nx, ny = 5, 5
    gp = make_bspline_2d_gp(nx, ny)
    function = gp.sampler.generate_realisation(seed=14)

    pts = UniformGrid((LB, RB, 4), (LB, RB, 4))
    values = function.evaluate(pts)
    assert values.shape == (16,)


def test_bspline_2d_gp_parameter_reconstruction_shares_representation():
    nx, ny = 5, 5
    gp = make_bspline_2d_gp(nx, ny)

    coords = np.ones(nx * ny)
    replacement = gp.function(coords)

    np.testing.assert_array_almost_equal(
        replacement.coordinate, coords,
        err_msg="replacement did not preserve its coordinate"
    )
    assert replacement.expansion is gp.expansion


def test_bspline_2d_gp_covariance_update_is_functional():
    gp = make_bspline_2d_gp()
    measureRef = gp.measure
    cov2 = MaternCovariance2D(lengthScale=0.1, smoothness=1.5, marginalVariance=2.)
    updated = gp.with_covariance_function(cov2)
    assert gp.measure is measureRef
    assert updated.measure is not measureRef
    assert updated.covarianceFunction is cov2


# ---------------------------------------------------------------------------
# DNA Fourier GP
# ---------------------------------------------------------------------------

def test_dna_fourier_1d_shape():
    q = 30
    rng = default_rng(1)
    cov = MaternCovariance1D(lengthScale=0.3, smoothness=1.5, marginalVariance=1.)
    gp = GaussianProcess.dna(cov, q=q, d=1)
    sites = gp.nativeGrid
    sample = gp.sampler.draw(rng)
    val = sample.evaluate(sites)
    assert val.shape == (32,)


def test_dna_fourier_1d_marginal_isotropy():
    """Interior marginal variances should be approximately uniform (DNA property)."""
    q = 30
    nSamples = 300
    sigma2 = 1.0
    rng = default_rng(99)
    cov = MaternCovariance1D(lengthScale=0.3, smoothness=1.5, marginalVariance=sigma2)
    gp = GaussianProcess.dna(cov, q=q, d=1)
    sites = gp.nativeGrid
    sampler = gp.sampler

    samples = np.array([
        sampler.draw(rng).evaluate(sites) for _ in range(nSamples)
    ])
    variances = np.var(samples, axis=0)

    rel_err = np.max(np.abs(variances[2:-2] - sigma2) / sigma2)
    assert rel_err < 0.4, (
        f"DNA isotropy check failed: max rel. variance deviation = {rel_err:.2f}"
    )


# ---------------------------------------------------------------------------
# Matern Fast Paths Regression
# ---------------------------------------------------------------------------

def test_matern_fast_paths():
    """Verify that closed-form fast paths match the generic Bessel implementation."""
    from scipy.special import gamma, kv
    
    def generic_matern_covariance(x, lengthScale, smoothness, variance):
        kappa = np.sqrt(2. * smoothness) / lengthScale
        scaledDistance = np.abs(x) * kappa
        covariance = np.zeros_like(scaledDistance)
        validMask = (scaledDistance > 1e-8) & (scaledDistance < 800.0)
        covariance[~validMask & (scaledDistance < 1e-8)] = variance
        covariance[validMask] = (variance * (2. ** (1. - smoothness)) / gamma(smoothness)) * \
            (scaledDistance[validMask] ** smoothness) * kv(smoothness, scaledDistance[validMask])
        return covariance

    def generic_matern_gradient(x, lengthScale, smoothness, variance):
        kappa = np.sqrt(2. * smoothness) / lengthScale
        scaledDistance = np.abs(x) * kappa
        gradient = np.zeros_like(scaledDistance)
        validMask = (scaledDistance > 1e-8) & (scaledDistance < 800.0)
        gradient[validMask] = (variance * (2. ** (1. - smoothness)) / gamma(smoothness)) * \
            (scaledDistance[validMask] ** (smoothness + 1.)) * \
            kv(smoothness - 1., scaledDistance[validMask])
        return gradient

    x = np.linspace(0, 2, 100)
    ls, var = 0.5, 1.5
    
    for nu in [1.5, 2.5]:
        # Test Covariance
        fast_cov = matern_covariance(x, ls, nu, var)
        gen_cov = generic_matern_covariance(x, ls, nu, var)
        np.testing.assert_allclose(fast_cov, gen_cov, rtol=1e-10, atol=1e-10)
        
        # Test Gradient
        fast_grad = matern_log_rho_gradient(x, ls, nu, var)
        gen_grad = generic_matern_gradient(x, ls, nu, var)
        np.testing.assert_allclose(fast_grad, gen_grad, rtol=1e-10, atol=1e-10)
