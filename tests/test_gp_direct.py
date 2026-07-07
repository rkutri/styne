import numpy as np
import pytest

from numpy.random import default_rng

from styne.gp.direct import DirectRealisation, DirectGPEngine
from styne.gp.gaussianprocess import GaussianProcess
from styne.statistics.gaussian import Gaussian
from styne.statistics.covariance import CovarianceMatrix, DenseCovarianceMatrix
from styne.statistics.stationary import MaternCovariance1D
from styne.utility.grid import Grid, UniformGrid


# ---- DirectRealisation API sanity ----

class TestDirectRealisationAPI:

    def setup_method(self):
        self.grid1d = UniformGrid(0., 1., 5)
        self.n = 5

    def test_dimension(self):
        r = DirectRealisation(self.grid1d, self.n)
        assert r.dimension == self.n

    def test_coefficient_unset_raises(self):
        r = DirectRealisation(self.grid1d, self.n)
        with pytest.raises(ValueError):
            _ = r.coefficient

    def test_coefficient_round_trip(self):
        r = DirectRealisation(self.grid1d, self.n)
        coeff = np.array([1., 2., 3., 4., 5.])
        r.coefficient = coeff
        np.testing.assert_allclose(r.coefficient, coeff)

    def test_coefficient_returns_copy(self):
        r = DirectRealisation(self.grid1d, self.n)
        coeff = np.ones(self.n)
        r.coefficient = coeff
        c = r.coefficient
        c[0] = 999.
        np.testing.assert_allclose(r.coefficient[0], 1.)

    def test_size_mismatch_raises(self):
        r = DirectRealisation(self.grid1d, self.n)
        with pytest.raises(ValueError):
            r.coefficient = np.ones(self.n + 1)

    def test_evaluate_1d(self):
        grid_vals = self.grid1d.to_array().ravel()
        r = DirectRealisation(self.grid1d, self.n)
        r.coefficient = grid_vals * 2.
        result = r.evaluate(self.grid1d)
        np.testing.assert_allclose(result, grid_vals * 2., atol=1e-12)

    def test_evaluate_2d(self):
        gridX = np.array([0., 0.5, 1.])
        gridY = np.array([0., 0.5, 1.])
        xx, yy = np.meshgrid(gridX, gridY, indexing='ij')
        grid2d = Grid(np.column_stack([xx.ravel(), yy.ravel()]))
        values = xx.ravel() + yy.ravel()
        r = DirectRealisation(grid2d, len(grid2d))
        r.coefficient = values
        with pytest.raises(NotImplementedError):
            r.evaluate(grid2d)

    def test_clone_is_independent(self):
        r = DirectRealisation(self.grid1d, self.n)
        coeff = np.ones(self.n)
        r.coefficient = coeff
        r2 = r.clone()
        r2.coefficient = np.zeros(self.n)
        np.testing.assert_allclose(r.coefficient, np.ones(self.n))


# ---- DirectGPEngine API sanity ----

class TestDirectGPEngineAPI:

    def setup_method(self):
        self.n = 10
        self.grid = UniformGrid(0., 1., self.n)
        self.variance = 0.7
        self.covFcn = MaternCovariance1D(0.3, 1.5, self.variance)
        self.engine = DirectGPEngine(self.grid)

    def test_realisation_type(self):
        r = self.engine.build_realisation()
        assert isinstance(r, DirectRealisation)

    def test_realisation_dimension(self):
        r = self.engine.build_realisation()
        assert r.dimension == self.n

    def test_covariance_type(self):
        cov = self.engine.build_covariance(self.covFcn)
        assert isinstance(cov, CovarianceMatrix)

    def test_covariance_dimension(self):
        cov = self.engine.build_covariance(self.covFcn)
        assert cov.dimension == self.n
        assert self.engine._shapeCovariance.to_dense().shape == (self.n, self.n)


# ---- GaussianProcess.direct correctness ----

class TestDirectGPCorrectness:

    def setup_method(self):
        self.n = 10
        self.grid = UniformGrid(0., 1., self.n)
        self.variance = 0.7
        self.covFcn = MaternCovariance1D(0.3, 1.5, self.variance)
        self.gp = GaussianProcess.direct(self.grid, self.covFcn)
        self.nSamples = 2000
        self.rng = default_rng(42)

    def _draw_samples(self):
        sampler = self.gp.sampler
        samples = []
        for _ in range(self.nSamples):
            sample = sampler.draw(self.rng)
            val = sample.function.evaluate(self.gp.engine.grid)
            samples.append(val)
        return np.array(samples)

    def test_measure_is_gaussian(self):
        assert isinstance(self.gp.measure, Gaussian)

    def test_sampler_shape(self):
        samples = self._draw_samples()
        assert samples.shape == (self.nSamples, self.n)

    def test_zero_mean(self):
        samples = self._draw_samples()
        empiricalMean = samples.mean(axis=0)
        stdErr = np.sqrt(self.variance / self.nSamples)
        assert np.all(np.abs(empiricalMean) < 3. * stdErr)

    def test_marginal_variance(self):
        samples = self._draw_samples()
        empiricalVar = samples.var(axis=0)
        relErr = np.abs(empiricalVar - self.variance) / self.variance
        assert np.all(relErr < 0.3)

    def test_sample_covariance(self):
        samples = self._draw_samples()
        empiricalCov = np.cov(samples.T)
        trueCov = self.gp.engine._shapeCovariance.to_dense()
        frobTrue = np.linalg.norm(trueCov, 'fro')
        frobErr = np.linalg.norm(empiricalCov - trueCov, 'fro')
        assert frobErr / frobTrue < 0.3

    def test_covariance_update_inplace(self):
        newCovFcn = MaternCovariance1D(0.5, 1.5, 1.2)
        self.gp.covarianceFunction = newCovFcn
        newK = self.gp.engine._shapeCovariance.to_dense()
        expected = newCovFcn.evaluate_covariance(
            self.grid, self.grid
        )
        np.testing.assert_allclose(newK, expected, atol=1e-10)


# ---- DirectSampler evaluate behaviour ----

class TestDirectSamplerEvaluate:
    """Regression tests verifying DirectRealisation shapes itself when evaluated."""

    def setup_method(self):
        self.gridSize = 15
        self.grid = UniformGrid(0.0, 1.0, self.gridSize)
        self.variance = 0.8
        self.covariance = MaternCovariance1D(0.3, 1.5, self.variance)
        self.process = GaussianProcess.direct(self.grid, self.covariance)
        self.randomGenerator = default_rng(2026)

    def test_sampler_evaluate_matches_at_sites(self):
        sample = self.process.sampler.generate_realisation(rng=self.randomGenerator)
        evaluated = sample.function.evaluate(self.grid)
        atSites = self.process.engine.at_sites(sample.function, self.grid)
        np.testing.assert_allclose(evaluated, atSites, atol=1e-12)

    def test_sampler_evaluate_marginal_variance(self):
        sampleCount = 2000
        samples = np.array([
            self.process.sampler.generate_realisation(
                rng=self.randomGenerator).function.evaluate(self.grid)
            for _ in range(sampleCount)
        ])
        empiricalVariance = samples.var(axis=0)
        relativeError = np.max(np.abs(empiricalVariance - self.variance) / self.variance)
        assert relativeError < 0.25, (
            f"Marginal variance mismatch: max rel. error = {relativeError:.3f}"
        )
