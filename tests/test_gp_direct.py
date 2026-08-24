import numpy as np
import pytest

from numpy.random import default_rng

from styne.gp.direct import DirectExpansion, DirectGPEngine
from styne.gp.gaussianprocess import GaussianProcess
from styne.statistics.gaussian import Gaussian
from styne.statistics.covariance import CovarianceMatrix
from styne.statistics.stationary import MaternCovariance1D
from styne.utility.grid import Grid, UniformGrid


# ---- DirectExpansion API sanity ----

class TestDirectExpansionAPI:

    def setup_method(self):
        self.grid1d = UniformGrid(0., 1., 5)
        self.n = 5

    def test_dimension(self):
        r = DirectExpansion(self.grid1d, self.n)
        assert r.dimension == self.n

    def test_has_no_coefficient_state(self):
        r = DirectExpansion(self.grid1d, self.n)
        for name in ("coefficient", "project", "clone"):
            assert not hasattr(r, name)

    def test_size_mismatch_raises(self):
        r = DirectExpansion(self.grid1d, self.n)
        with pytest.raises(ValueError):
            r.evaluate(np.ones(self.n + 1), self.grid1d)

    def test_evaluate_1d(self):
        grid_vals = self.grid1d.to_array().ravel()
        r = DirectExpansion(self.grid1d, self.n)
        result = r.evaluate(grid_vals * 2., self.grid1d)
        np.testing.assert_allclose(result, grid_vals * 2., atol=1e-12)

    def test_evaluate_batch(self):
        gridValues = self.grid1d.to_array().ravel()
        coefficient = np.stack([
            gridValues, 2. * gridValues
        ]).reshape(1, 2, self.n)
        expansion = DirectExpansion(self.grid1d, self.n)

        result = expansion.evaluate(
            coefficient=coefficient, grid=self.grid1d
        )

        np.testing.assert_allclose(result, coefficient, atol=1e-12)

    def test_evaluate_2d(self):
        gridX = np.array([0., 0.5, 1.])
        gridY = np.array([0., 0.5, 1.])
        xx, yy = np.meshgrid(gridX, gridY, indexing='ij')
        grid2d = Grid(np.column_stack([xx.ravel(), yy.ravel()]))
        values = xx.ravel() + yy.ravel()
        r = DirectExpansion(grid2d, len(grid2d))
        with pytest.raises(NotImplementedError):
            r.evaluate(values, grid2d)


# ---- DirectGPEngine API sanity ----

class TestDirectGPEngineAPI:

    def setup_method(self):
        self.n = 10
        self.grid = UniformGrid(0., 1., self.n)
        self.variance = 0.7
        self.covFcn = MaternCovariance1D(0.3, 1.5, self.variance)
        self.engine = DirectGPEngine(self.grid)

    def test_expansion_type(self):
        r = self.engine.build_expansion()
        assert isinstance(r, DirectExpansion)

    def test_expansion_dimension(self):
        r = self.engine.build_expansion()
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
            val = sample.evaluate(self.gp.engine.grid)
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
    """Regression tests for bound direct-function evaluation."""

    def setup_method(self):
        self.gridSize = 15
        self.grid = UniformGrid(0.0, 1.0, self.gridSize)
        self.variance = 0.8
        self.covariance = MaternCovariance1D(0.3, 1.5, self.variance)
        self.process = GaussianProcess.direct(self.grid, self.covariance)
        self.process.sites = self.grid
        self.randomGenerator = default_rng(2026)

    def test_sampler_evaluate_matches_at_sites(self):
        sample = self.process.sampler.generate_realisation(rng=self.randomGenerator)
        evaluated = sample.evaluate(self.grid)
        atSites = self.process.at_sites(sample.coordinate)
        np.testing.assert_allclose(evaluated, atSites, atol=1e-12)

    def test_sampler_evaluate_marginal_variance(self):
        sampleCount = 2000
        samples = np.array([
            self.process.sampler.generate_realisation(
                rng=self.randomGenerator).evaluate(self.grid)
            for _ in range(sampleCount)
        ])
        empiricalVariance = samples.var(axis=0)
        relativeError = np.max(np.abs(empiricalVariance - self.variance) / self.variance)
        assert relativeError < 0.25, (
            f"Marginal variance mismatch: max rel. error = {relativeError:.3f}"
        )
