import numpy as np
import pytest

from numpy.random import default_rng
from scipy.linalg import toeplitz

from styne.gp.ce import (
    CirculantEmbeddingEngine1D,
    CirculantEmbeddingEngine2D,
    ApproximateCirculantEmbeddingEngine1D,
    ApproximateCirculantEmbeddingEngine2D,
)
from styne.statistics.stationary import matern_covariance
from styne.utility.exceptions import NotPositiveDefinite


def _matern_cov(r, lengthScale=0.3, smoothness=1.5, variance=1.0):
    return matern_covariance(r, lengthScale, smoothness, variance)


class TestCirculantEmbeddingConstruction:

    def setup_method(self):
        self.vertPerDim = 10
        self.domExt = 1.0

    def test_initial_padding_exceeds_max_raises(self):
        with pytest.raises(RuntimeError):
            CirculantEmbeddingEngine1D(
                _matern_cov, self.vertPerDim, self.domExt, padding=1024, maxPadding=512
            )

    def test_draw_shape_1d(self):
        engine = CirculantEmbeddingEngine1D(
            _matern_cov, self.vertPerDim, self.domExt
        )
        rng = default_rng(42)
        sample = engine.draw(rng)
        assert sample.shape == (self.vertPerDim,)

    def test_draw_shape_2d(self):
        engine = CirculantEmbeddingEngine2D(
            _matern_cov, self.vertPerDim, self.domExt
        )
        rng = default_rng(42)
        sample = engine.draw(rng)
        assert sample.shape == (self.vertPerDim, self.vertPerDim)

    def test_eigenvalues_are_real_and_nonneg(self):
        engine = CirculantEmbeddingEngine1D(
            _matern_cov, self.vertPerDim, self.domExt
        )
        assert np.all(np.isreal(engine._eigenvalues))
        assert engine._eigenvalues.min() >= -engine._tol

    def test_draw_deterministic_with_seed(self):
        engine = CirculantEmbeddingEngine1D(
            _matern_cov, self.vertPerDim, self.domExt
        )
        sample1 = engine.draw(default_rng(123))
        sample2 = engine.draw(default_rng(123))
        np.testing.assert_allclose(sample1, sample2)


class TestCirculantEmbeddingPadding:

    def test_indefinite_initial_triggers_padding_increase(self):
        covFcn = lambda r: _matern_cov(r, lengthScale=2.0)
        engine = CirculantEmbeddingEngine1D(
            covFcn, vertPerDim=16, domExt=1.0, autotunePadding=False
        )
        assert engine._padding > 0

    def test_autotune_reduces_padding(self):
        covFcn = lambda r: _matern_cov(r, lengthScale=2.0)
        engineNoTune = CirculantEmbeddingEngine1D(
            covFcn, vertPerDim=16, domExt=1.0, autotunePadding=False
        )
        engineTune = CirculantEmbeddingEngine1D(
            covFcn, vertPerDim=16, domExt=1.0, autotunePadding=True
        )
        assert engineTune._padding <= engineNoTune._padding

    def test_max_padding_exhaustion_raises(self):
        covFcn = lambda r: _matern_cov(r, lengthScale=5.0)
        with pytest.raises(NotPositiveDefinite):
            CirculantEmbeddingEngine1D(
                covFcn, vertPerDim=16, domExt=1.0, maxPadding=2
            )


class TestApproximateCirculantEmbedding:

    def setup_method(self):
        self.vertPerDim = 16
        self.domExt = 1.0
        self.covFcn = lambda r: _matern_cov(r, lengthScale=2.0)

    def test_approximate_1d_eigenvalues_nonneg(self):
        engine = ApproximateCirculantEmbeddingEngine1D(
            self.covFcn, self.vertPerDim, self.domExt
        )
        assert engine._eigenvalues.min() >= 0.0

    def test_approximate_2d_eigenvalues_nonneg(self):
        engine = ApproximateCirculantEmbeddingEngine2D(
            self.covFcn, self.vertPerDim, self.domExt
        )
        assert engine._eigenvalues.min() >= 0.0

    def test_approximate_1d_draw_shape(self):
        engine = ApproximateCirculantEmbeddingEngine1D(
            self.covFcn, self.vertPerDim, self.domExt
        )
        sample = engine.draw(default_rng(42))
        assert sample.shape == (self.vertPerDim,)

    def test_approximate_2d_draw_shape(self):
        engine = ApproximateCirculantEmbeddingEngine2D(
            self.covFcn, self.vertPerDim, self.domExt
        )
        sample = engine.draw(default_rng(42))
        assert sample.shape == (self.vertPerDim, self.vertPerDim)


class TestCirculantEmbeddingCorrectness:

    def setup_method(self):
        self.vertPerDim = 10
        self.domExt = 1.0
        self.variance = 0.8
        self.covFcn = lambda r: _matern_cov(
            r, lengthScale=0.2, smoothness=1.5, variance=self.variance
        )
        self.nSamples = 2500
        self.rng = default_rng(42)

    def _draw_samples_1d(self):
        engine = CirculantEmbeddingEngine1D(
            self.covFcn, self.vertPerDim, self.domExt
        )
        samples = [engine.draw(self.rng) for _ in range(self.nSamples)]
        return np.array(samples)

    def _draw_samples_2d(self):
        engine = CirculantEmbeddingEngine2D(
            self.covFcn, self.vertPerDim, self.domExt
        )
        samples = [engine.draw(self.rng) for _ in range(self.nSamples)]
        return np.array(samples)

    def test_zero_mean_1d(self):
        samples = self._draw_samples_1d()
        empiricalMean = samples.mean(axis=0)
        stdErr = np.sqrt(self.variance / self.nSamples)
        assert np.all(np.abs(empiricalMean) < 4.0 * stdErr)

    def test_marginal_variance_1d(self):
        samples = self._draw_samples_1d()
        empiricalVar = samples.var(axis=0)
        relErr = np.abs(empiricalVar - self.variance) / self.variance
        assert np.all(relErr < 0.2)

    def test_spatial_covariance_1d(self):
        samples = self._draw_samples_1d()
        empiricalCov = np.cov(samples.T)
        h = self.domExt / self.vertPerDim
        firstRow = np.array([self.covFcn(k * h) for k in range(self.vertPerDim)])
        trueCov = toeplitz(firstRow)
        frobTrue = np.linalg.norm(trueCov, "fro")
        frobErr = np.linalg.norm(empiricalCov - trueCov, "fro")
        assert frobErr / frobTrue < 0.25

    def test_zero_mean_2d(self):
        samples = self._draw_samples_2d()
        empiricalMean = samples.mean(axis=0)
        stdErr = np.sqrt(self.variance / self.nSamples)
        assert np.all(np.abs(empiricalMean) < 4.0 * stdErr)

    def test_marginal_variance_2d(self):
        samples = self._draw_samples_2d()
        empiricalVar = samples.var(axis=0)
        relErr = np.abs(empiricalVar - self.variance) / self.variance
        assert np.all(relErr < 0.25)

    def test_spatial_covariance_2d_slice(self):
        samples = self._draw_samples_2d()
        rowSamples = samples[:, 0, :]
        empiricalCov = np.cov(rowSamples.T)
        h = self.domExt / self.vertPerDim
        firstRow = np.array([self.covFcn(k * h) for k in range(self.vertPerDim)])
        trueCov = toeplitz(firstRow)
        frobTrue = np.linalg.norm(trueCov, "fro")
        frobErr = np.linalg.norm(empiricalCov - trueCov, "fro")
        assert frobErr / frobTrue < 0.3

