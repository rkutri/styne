import numpy as np
import pytest

from numpy.random import default_rng

from styne.gp.bspline import (
    BSplineRealisation1D, BSplineRealisation2D,
    BSplineGPEngine
)
from styne.gp.gaussianprocess import GaussianProcess
from styne.model.representation.bspline import BSpline1D, BSpline2D
from styne.statistics.gaussian import Gaussian
from styne.statistics.covariance import CovarianceMatrix
from styne.statistics.stationary import MaternCovariance1D, MaternCovariance2D


# ---- helpers ----

def _make_bsp1d(n=6):
    bsp = BSpline1D(n, degree=3, boundary=[0., 1.])
    bsp.project(np.zeros(n))
    return bsp


def _make_bsp2d(nx=5, ny=5):
    bsp = BSpline2D([nx, ny], degree=3, boundary=[[0., 1.], [0., 1.]])
    bsp.project(np.zeros(nx * ny))
    return bsp


def _cov1d(ell=0.3, nu=1.5, variance=0.7):
    return MaternCovariance1D(ell, nu, variance)


def _cov2d(ell=0.3, nu=1.5, variance=0.7):
    return MaternCovariance2D(ell, nu, variance)


# ---- BSplineRealisation API sanity ----

class TestBSplineRealisation1DAPI:

    def setup_method(self):
        self.bsp = _make_bsp1d(n=6)
        self.r = BSplineRealisation1D(self.bsp)

    def test_dimension(self):
        assert self.r.dimension == 6

    def test_coefficient_round_trip(self):
        coeff = np.arange(6, dtype=float)
        self.r.coefficient = coeff
        np.testing.assert_allclose(self.r.coefficient, coeff)


class TestBSplineRealisation2DAPI:

    def test_dimension(self):
        bsp = _make_bsp2d(nx=5, ny=4)
        r = BSplineRealisation2D(bsp)
        assert r.dimension == 20


# ---- GaussianProcess.bspline API sanity ----

class TestBSpline1DGPMeasure:

    def setup_method(self):
        self.grid = np.linspace(0., 1., 20)
        self.bsp = _make_bsp1d(n=6)
        self.gp = GaussianProcess.bspline(_cov1d(), self.bsp)

    def test_measure_dimension(self):
        assert self.gp.measure.covariance.to_dense().shape == (6, 6)

    def test_measure_is_gaussian(self):
        assert isinstance(self.gp.measure, Gaussian)

    def test_sampler_returns_realisation(self):
        rng = default_rng(0)
        sample = self.gp.sampler.draw(rng)
        assert isinstance(sample.function, BSplineRealisation1D)
        assert sample.coordinate.shape == (6,)


class TestBSpline2DGPMeasure:

    def setup_method(self):
        grid2d = np.array([[x, y]
                           for x in np.linspace(0., 1., 5)
                           for y in np.linspace(0., 1., 5)])
        self.bsp2d = _make_bsp2d(nx=5, ny=5)
        self.gp = GaussianProcess.bspline(_cov2d(), self.bsp2d)
        self.dim = 25

    def test_measure_dimension(self):
        assert self.gp.measure.covariance.to_dense().shape == (self.dim, self.dim)

    def test_measure_is_gaussian(self):
        assert isinstance(self.gp.measure, Gaussian)

    def test_sampler_returns_realisation(self):
        rng = default_rng(0)
        sample = self.gp.sampler.draw(rng)
        assert isinstance(sample.function, BSplineRealisation2D)
        assert sample.coordinate.shape == (self.dim,)


# ---- GaussianProcess.bspline correctness ----

class TestBSpline1DGPCorrectness:

    def setup_method(self):
        self.n = 6
        self.grid = np.linspace(0., 1., 20)
        self.bsp = _make_bsp1d(n=self.n)
        self.gp = GaussianProcess.bspline(_cov1d(), self.bsp)
        self.nSamples = 1000
        self.rng = default_rng(7)

    def _draw_coefficients(self):
        sampler = self.gp.sampler
        return np.array([
            sampler.draw(self.rng).coordinate
            for _ in range(self.nSamples)
        ])

    def test_zero_mean(self):
        coeffs = self._draw_coefficients()
        empiricalMean = coeffs.mean(axis=0)
        marginalStd = np.sqrt(
            np.diag(self.gp.measure.covariance.to_dense())
        )
        stdErr = marginalStd / np.sqrt(self.nSamples)
        assert np.all(np.abs(empiricalMean) < 4. * stdErr)

    def test_coeff_covariance(self):
        coeffs = self._draw_coefficients()
        empiricalCov = np.cov(coeffs.T)
        trueCov = self.gp.measure.covariance.to_dense()
        frobTrue = np.linalg.norm(trueCov, 'fro')
        frobErr = np.linalg.norm(empiricalCov - trueCov, 'fro')
        assert frobErr / frobTrue < 0.3


class TestBSpline2DGPCorrectness:

    def setup_method(self):
        self.nx = 5
        self.ny = 5
        self.dim = self.nx * self.ny
        grid2d = np.array([[x, y]
                           for x in np.linspace(0., 1., 5)
                           for y in np.linspace(0., 1., 5)])
        self.bsp2d = _make_bsp2d(nx=self.nx, ny=self.ny)
        self.gp = GaussianProcess.bspline(_cov2d(), self.bsp2d)
        self.nSamples = 1000
        self.rng = default_rng(13)

    def _draw_coefficients(self):
        sampler = self.gp.sampler
        return np.array([
            sampler.draw(self.rng).coordinate
            for _ in range(self.nSamples)
        ])

    def test_zero_mean(self):
        coeffs = self._draw_coefficients()
        empiricalMean = coeffs.mean(axis=0)
        marginalStd = np.sqrt(
            np.diag(self.gp.measure.covariance.to_dense())
        )
        stdErr = marginalStd / np.sqrt(self.nSamples)
        assert np.all(np.abs(empiricalMean) < 3. * stdErr)

    def test_coeff_covariance(self):
        coeffs = self._draw_coefficients()
        empiricalCov = np.cov(coeffs.T)
        trueCov = self.gp.measure.covariance.to_dense()
        frobTrue = np.linalg.norm(trueCov, 'fro')
        frobErr = np.linalg.norm(empiricalCov - trueCov, 'fro')
        assert frobErr / frobTrue < 0.3
