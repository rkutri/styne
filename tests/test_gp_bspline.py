import numpy as np

from numpy.random import default_rng

from styne.gp.gaussianprocess import GaussianProcess
from styne.model.representation.bspline import BSpline1D, BSpline2D
from styne.statistics.gaussian import Gaussian
from styne.statistics.stationary import MaternCovariance1D, MaternCovariance2D


# ---- helpers ----

def make_bsp1d(n=6):
    return BSpline1D(n, degree=3, boundary=[0., 1.])


def make_bsp2d(nx=5, ny=5):
    return BSpline2D(
        [nx, ny], degree=3, boundary=[[0., 1.], [0., 1.]]
    )


def cov1d(ell=0.3, nu=1.5, variance=0.7):
    return MaternCovariance1D(ell, nu, variance)


def cov2d(ell=0.3, nu=1.5, variance=0.7):
    return MaternCovariance2D(ell, nu, variance)


# ---- Static B-spline API sanity ----

class TestBSplineExpansion1DAPI:

    def setup_method(self):
        self.bsp = make_bsp1d(n=6)
    def test_dimension(self):
        assert self.bsp.dimension == 6

    def test_explicit_coefficient_evaluation(self):
        coeff = np.arange(24, dtype=float).reshape(2, 2, 6)
        grid = np.linspace(0., 1., 10)
        np.testing.assert_allclose(
            self.bsp.evaluate(coefficient=coeff, grid=grid),
            coeff @ self.bsp.design_matrix(grid).T,
        )


class TestBSplineExpansion2DAPI:

    def test_dimension(self):
        bsp = make_bsp2d(nx=5, ny=4)
        assert bsp.dimension == 20


# ---- GaussianProcess.bspline API sanity ----

class TestBSpline1DGPMeasure:

    def setup_method(self):
        self.grid = np.linspace(0., 1., 20)
        self.bsp = make_bsp1d(n=6)
        self.gp = GaussianProcess.bspline(cov1d(), self.bsp)

    def test_measure_dimension(self):
        assert self.gp.measure.covariance.to_dense().shape == (6, 6)

    def test_measure_is_gaussian(self):
        assert isinstance(self.gp.measure, Gaussian)

    def test_sampler_shares_expansion(self):
        rng = default_rng(0)
        sample = self.gp.sampler.draw(rng)
        assert sample.expansion is self.bsp
        assert sample.coordinate.shape == (6,)


class TestBSpline2DGPMeasure:

    def setup_method(self):
        self.bsp2d = make_bsp2d(nx=5, ny=5)
        self.gp = GaussianProcess.bspline(cov2d(), self.bsp2d)
        self.dim = 25

    def test_measure_dimension(self):
        assert self.gp.measure.covariance.to_dense().shape == (self.dim, self.dim)

    def test_measure_is_gaussian(self):
        assert isinstance(self.gp.measure, Gaussian)

    def test_sampler_shares_expansion(self):
        rng = default_rng(0)
        sample = self.gp.sampler.draw(rng)
        assert sample.expansion is self.bsp2d
        assert sample.coordinate.shape == (self.dim,)


# ---- GaussianProcess.bspline correctness ----

class TestBSpline1DGPCorrectness:

    def setup_method(self):
        self.n = 6
        self.grid = np.linspace(0., 1., 20)
        self.bsp = make_bsp1d(n=self.n)
        self.gp = GaussianProcess.bspline(cov1d(), self.bsp)
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
        self.bsp2d = make_bsp2d(nx=self.nx, ny=self.ny)
        self.gp = GaussianProcess.bspline(cov2d(), self.bsp2d)
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
