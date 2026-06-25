import numpy as np
import pytest

from numpy.random import default_rng

from styne.gp.dna import (
    BC, BoundaryCondition,
    DNAFourierRealisation, DNAFourierEngine
)
from styne.gp.gaussianprocess import GaussianProcess, GPSampler
from styne.parameter.function import Function
from styne.statistics.gaussian import Gaussian
from styne.statistics.covariance import DiagonalCovarianceMatrix
from styne.statistics.stationary import MaternCovariance1D, MaternCovariance2D
from styne.utility.grid import Grid, UniformGrid


# ---- helpers ----

def _cov1d():
    return MaternCovariance1D(0.3, 1.5, 1.0)


def _cov2d():
    return MaternCovariance2D(0.3, 1.5, 1.0)


# ---- DNAFourierRealisation API (1D) ----

class TestDNAFourierRealisation1D:

    def setup_method(self):
        self.q = 5
        self.r = DNAFourierRealisation(self.q, d=1)

    def test_dimension(self):
        assert self.r.dimension == 2 * self.q + 1

    # (Removed coefficient unset test since DNAFourierRealisation initializes to zero)

    def test_coefficient_round_trip(self):
        coeff = np.arange(2 * self.q + 1, dtype=float)
        self.r.coefficient = coeff
        np.testing.assert_allclose(self.r.coefficient, coeff)

    def test_coefficient_propagates_to_components(self):
        coeff = np.arange(self.r.dimension, dtype=float)
        self.r.coefficient = coeff
        offset = 0
        for i in range(self.r._param.nBlocks):
            dim = self.r._param._dims[i]
            np.testing.assert_allclose(
                self.r._param.block(i).function.coefficient,
                coeff[offset:offset + dim]
            )
            offset += dim

    def test_zero_coefficient_zero_field(self):
        self.r.coefficient = np.zeros(self.r.dimension)
        pts = UniformGrid(0.1, 0.9, 10)
        np.testing.assert_allclose(self.r.evaluate(pts), 0.)

    def test_evaluate_shape(self):
        self.r.coefficient = np.zeros(self.r.dimension)
        pts = UniformGrid(0.1, 0.9, 15)
        assert self.r.evaluate(pts).shape == (15,)

    def test_clone_independence(self):
        self.r.coefficient = np.ones(self.r.dimension)
        clone = self.r.clone()
        clone.coefficient = np.zeros(self.r.dimension)
        np.testing.assert_allclose(self.r.coefficient, 1.)


# ---- DNAFourierRealisation API (2D) ----

class TestDNAFourierRealisation2D:

    def setup_method(self):
        self.q = 4
        self.r = DNAFourierRealisation(self.q, d=2)

    def test_dimension(self):
        assert self.r.dimension == (2 * self.q + 1)**2

    # (Removed coefficient unset test since DNAFourierRealisation initializes to zero)

    def test_coefficient_round_trip(self):
        coeff = np.arange((2 * self.q + 1)**2, dtype=float)
        self.r.coefficient = coeff
        np.testing.assert_allclose(self.r.coefficient, coeff)

    def test_zero_coefficient_zero_field(self):
        self.r.coefficient = np.zeros(self.r.dimension)
        pts = Grid(np.stack(
            [np.linspace(0.1, 0.9, 6),
             np.linspace(0.1, 0.9, 6)],
            axis=1))
        np.testing.assert_allclose(self.r.evaluate(pts), 0.)

    def test_evaluate_shape(self):
        self.r.coefficient = np.zeros(self.r.dimension)
        pts = Grid(np.stack(
            [np.linspace(0.1, 0.9, 8),
             np.linspace(0.1, 0.9, 8)],
            axis=1))
        assert self.r.evaluate(pts).shape == (8,)


# ---- DNAFourierEngine API ----

class TestDNAFourierEngine1D:

    def setup_method(self):
        self.q = 5
        self.engine = DNAFourierEngine(self.q, d=1)
        self.covFcn = _cov1d()

    def test_build_realisation_type(self):
        assert isinstance(
            self.engine.build_realisation(),
            DNAFourierRealisation)

    def test_covariance_type(self):
        assert isinstance(
            self.engine.build_covariance(self.covFcn), DiagonalCovarianceMatrix
        )

    def test_covariance_dimension(self):
        assert self.engine.build_covariance(
            self.covFcn).dimension == 2 * self.q + 1

    def test_spectral_densities_positive(self):
        cov = self.engine.build_covariance(self.covFcn)
        assert np.all(cov.marginalVariance > 0.)

    def test_covariance_matches_realisation_dimension(self):
        assert (self.engine.build_covariance(self.covFcn).dimension
                == self.engine.build_realisation().dimension)


class TestDNAFourierEngine2D:

    def setup_method(self):
        self.q = 4
        self.engine = DNAFourierEngine(self.q, d=2)
        self.covFcn = _cov2d()

    def test_build_realisation_type(self):
        assert isinstance(
            self.engine.build_realisation(),
            DNAFourierRealisation)

    def test_covariance_dimension(self):
        assert self.engine.build_covariance(
            self.covFcn).dimension == (
            2 * self.q + 1) ** 2

    def test_spectral_densities_positive(self):
        cov = self.engine.build_covariance(self.covFcn)
        assert np.all(cov.marginalVariance > 0.)

    def test_covariance_matches_realisation_dimension(self):
        assert (self.engine.build_covariance(self.covFcn).dimension
                == self.engine.build_realisation().dimension)


# ---- GPSampler equivalence ----

class TestDNAFourierGPSampler:

    def setup_method(self):
        self.seed = 42
        self.pts1d = UniformGrid(0.05, 0.95, 20)
        self.pts2d = Grid(np.stack(
            [np.linspace(0.1, 0.9, 8), np.linspace(0.1, 0.9, 8)], axis=1
        ))

    def _check_equivalence(self, q, d, covFcn, pts):
        engine = DNAFourierEngine(q, d)
        cov = engine.build_covariance(covFcn)
        realisation = engine.build_realisation()
        measure = Gaussian(cov)
        mean = Function(realisation.clone())
        mean.coordinate = np.zeros(realisation.dimension)
        measure.mean = mean

        sqrtVars = np.sqrt(cov.marginalVariance)

        rng = default_rng(self.seed)
        realisation.coefficient = sqrtVars * \
            rng.standard_normal(realisation.dimension)
        result1 = realisation.evaluate(pts)

        rng = default_rng(self.seed)
        result2 = GPSampler(realisation, measure).draw(rng).function.evaluate(pts)

        np.testing.assert_allclose(result1, result2, rtol=1e-12)

    def test_gp_sampler_equivalence_1d(self):
        self._check_equivalence(q=6, d=1, covFcn=_cov1d(), pts=self.pts1d)

    def test_gp_sampler_equivalence_2d(self):
        self._check_equivalence(q=5, d=2, covFcn=_cov2d(), pts=self.pts2d)


# ---- Spectral structure ----

class TestDNASpectralStructure:
    """
    Three nested checks on the joint distribution of spectral coefficients:

    1. Aggregate chi-squared: Σ_i (N-1)*S²_i/σ²_i ~ chi²(D*(N-1)).
       Detects any systematic over- or under-dispersion across all modes.

    2. Per-coefficient max relative error with a principled threshold.
       Each |S²_i/σ²_i - 1| / relErrStd is approximately half-normal(1);
       threshold is set so P(spurious failure) ≈ 1e-4.

    3. Off-diagonal empirical correlations ~ N(0, 1/N) under independence.
       Threshold at P(spurious failure) ≈ 1e-4 over all pairs.
       Detects any coupling between spectral modes introduced by the sampling
       pipeline (would indicate the covariance is not truly diagonal).
    """

    def _sample_coefficients(self, engine, covFcn, nSamples, seed=7):
        cov = engine.build_covariance(covFcn)
        realisation = engine.build_realisation()
        measure = Gaussian(cov)
        mean = Function(realisation.clone())
        mean.coordinate = np.zeros(realisation.dimension)
        measure.mean = mean
        sampler = GPSampler(realisation, measure)
        rng = default_rng(seed)
        return cov, np.array([sampler.draw(rng).coordinate
                              for _ in range(nSamples)])

    def _assert_spectral_variances(self, cov, coefficients):
        N, D = coefficients.shape
        empiricalVars = np.var(coefficients, axis=0, ddof=1)

        # --- Aggregate chi-squared ---
        # sum_i (N-1)*S²_i/σ²_i ~ chi²(D*(N-1)); mean = D*(N-1), std = sqrt(2*D*(N-1))
        chiSq = (N - 1) * np.sum(empiricalVars / cov.marginalVariance)
        chiMean = float(D * (N - 1))
        chiStd = np.sqrt(2. * chiMean)
        zAgg = (chiSq - chiMean) / chiStd
        assert abs(
            zAgg) < 5., f"Aggregate spectral chi-squared z-score: {zAgg:.2f}"

        # --- Per-coefficient max ---
        # relErr[i] ≈ half-normal with scale relErrStd = sqrt(2/(N-1))
        # Threshold: P(max of D such values > threshold) ≈ 1e-4
        relErr = np.abs(
            empiricalVars - cov.marginalVariance) / cov.marginalVariance
        relErrStd = np.sqrt(2. / (N - 1))
        maxThreshold = relErrStd * np.sqrt(2. * np.log(2. * D / 1e-4))
        assert relErr.max() < maxThreshold, (
            f"Max per-coefficient relative error {relErr.max():.4f} "
            f"exceeds threshold {maxThreshold:.4f}"
        )

    def _assert_independence(self, cov, coefficients):
        N, D = coefficients.shape
        corrMatrix = np.corrcoef(coefficients.T)
        offDiag = np.abs(corrMatrix[np.triu_indices(D, k=1)])

        # Each off-diagonal correlation ~ N(0, 1/N) under independence
        # Threshold: P(max of nPairs such values > threshold) ≈ 1e-4
        nPairs = D * (D - 1) // 2
        corrStd = 1. / np.sqrt(N)
        maxThreshold = corrStd * np.sqrt(2. * np.log(2. * nPairs / 1e-4))
        assert offDiag.max() < maxThreshold, (
            f"Spurious off-diagonal correlation {offDiag.max():.4f} "
            f"exceeds threshold {maxThreshold:.4f}"
        )

    def test_spectral_structure_1d(self):
        engine = DNAFourierEngine(10, 1)
        cov, coefficients = self._sample_coefficients(
            engine, _cov1d(), nSamples=5000)
        self._assert_spectral_variances(cov, coefficients)
        self._assert_independence(cov, coefficients)

    def test_spectral_structure_2d(self):
        engine = DNAFourierEngine(7, 2)
        cov, coefficients = self._sample_coefficients(
            engine, _cov2d(), nSamples=4000)
        self._assert_spectral_variances(cov, coefficients)
        self._assert_independence(cov, coefficients)


# ---- Isotropy ----

class TestDNAIsotropy2D:

    def test_interior_variance_uniformity(self):
        q = 10
        nSamples = 2000
        rng = default_rng(42)
        gp = GaussianProcess.dna(_cov2d(), q=q, d=2)

        xs = np.linspace(0.1, 0.9, 10)
        ys = np.linspace(0.1, 0.9, 10)
        pts = Grid(np.array([[x, y] for x in xs for y in ys]))

        fields = np.array([gp.sampler.draw(rng).function.evaluate(pts)
                           for _ in range(nSamples)])
        variances = np.var(fields, axis=0)

        relSpread = (variances.max() - variances.min()) / variances.mean()
        assert relSpread < 0.35, f"Variance non-uniformity too large: {relSpread:.3f}"


    def test_zero_mean(self):
        q = 10
        nSamples = 2000
        rng = default_rng(43)
        gp = GaussianProcess.dna(_cov2d(), q=q, d=2)

        xs = np.linspace(0.2, 0.8, 6)
        ys = np.linspace(0.2, 0.8, 6)
        pts = Grid(np.array([[x, y] for x in xs for y in ys]))

        fields = np.array([gp.sampler.draw(rng).function.evaluate(pts)
                           for _ in range(nSamples)])
        variances = np.var(fields, axis=0)
        means = np.mean(fields, axis=0)

        maxStdUnits = np.max(
            np.abs(means)) / np.sqrt(variances.mean() / nSamples)
        assert maxStdUnits < 4.0, f"Empirical mean too large: {maxStdUnits:.2f} std units"

