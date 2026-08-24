import numpy as np
import pytest

from styne.mcmc.method.pmala import (
    PMALAFactory,
    PMALAProposal,
    PreconditionedMALA,
    validate_pmala_target,
)
from styne.mcmc.diagnostics import AcceptanceRateDiagnostics
from styne.statistics.radonnikodym import RadonNikodym
from styne.statistics.gaussian import Gaussian, GaussianDensity
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.interface import DensityInterface
from styne.statistics.measure import AbsolutelyContinuousProbabilityMeasure
from styne.parameter.vector import Vector
from styne.utility.postprocessing import integrated_autocorrelation
from styne.utility.tuning import PMALATuner


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_valid_target(dim=2):
    refCov = IIDCovarianceMatrix(dim, 1.0)
    refMean = Vector(np.zeros(dim))
    prior = Gaussian(refCov, refMean)
    likCov = IIDCovarianceMatrix(dim, 0.5)
    likMean = Vector(np.ones(dim))
    derivative = GaussianDensity(likCov, likMean)
    return RadonNikodym(prior, derivative)


class NonDifferentiableDensity(DensityInterface):
    @property
    def domainType(self): return Vector
    @property
    def domainDimension(self): return 2
    def evaluate_log(self, p): return 0.0


class NonGaussianMeasure(AbsolutelyContinuousProbabilityMeasure):
    """Stub that looks like an AbsolutelyContinuousProbabilityMeasure but is not Gaussian."""
    class _Density(DensityInterface):
        @property
        def domainType(self): return Vector
        @property
        def domainDimension(self): return 2
        def evaluate_log(self, p): return 0.0

    _density = _Density()

    @property
    def density(self) -> DensityInterface:
        return self._density

    def draw(self, rng):
        return Vector(np.zeros(2))



# ---------------------------------------------------------------------------
# 1. Setup validity
# ---------------------------------------------------------------------------

class TestPMALASetup:

    def test_rejects_non_radon_nikodym_target(self):
        dim = 2
        density = GaussianDensity(IIDCovarianceMatrix(dim, 1.0),
                                  Vector(np.zeros(dim)))
        with pytest.raises(TypeError):
            PreconditionedMALA(density, 0.5, AcceptanceRateDiagnostics())

    def test_rejects_non_differentiable_derivative(self):
        refCov = IIDCovarianceMatrix(2, 1.0)
        prior = Gaussian(refCov, Vector(np.zeros(2)))
        target = RadonNikodym(prior, NonDifferentiableDensity())
        with pytest.raises(ValueError):
            PreconditionedMALA(target, 0.5, AcceptanceRateDiagnostics())

    def test_rejects_beta_zero(self):
        with pytest.raises(ValueError):
            PreconditionedMALA(make_valid_target(), 0.0,
                               AcceptanceRateDiagnostics())

    def test_rejects_beta_above_one(self):
        with pytest.raises(ValueError):
            PreconditionedMALA(make_valid_target(), 1.1,
                               AcceptanceRateDiagnostics())

    def test_rejects_non_gaussian_reference(self):
        stub = NonGaussianMeasure()
        deriv = GaussianDensity(IIDCovarianceMatrix(2, 1.0),
                                Vector(np.zeros(2)))
        target = RadonNikodym(stub, deriv)
        with pytest.raises(NotImplementedError):
            PreconditionedMALA(target, 0.5, AcceptanceRateDiagnostics())

    def test_factory_rejects_non_radon_nikodym_target(self):
        dim = 2
        factory = PMALAFactory()
        factory.target = GaussianDensity(IIDCovarianceMatrix(dim, 1.0),
                                         Vector(np.zeros(dim)))
        factory.beta = 0.5
        with pytest.raises(TypeError):
            factory.create()

    def test_factory_rejects_missing_beta(self):
        factory = PMALAFactory()
        factory.target = make_valid_target()
        with pytest.raises(ValueError):
            factory.create()

    def test_factory_rejects_invalid_beta(self):
        factory = PMALAFactory()
        factory.target = make_valid_target()
        factory.beta = -0.1
        with pytest.raises(ValueError):
            factory.create()

    def test_factory_creates_correctly(self):
        factory = PMALAFactory()
        factory.target = make_valid_target()
        factory.beta = 0.5
        sampler = factory.create()
        assert isinstance(sampler, PreconditionedMALA)


# ---------------------------------------------------------------------------
# 2. Proposal step correctness
# ---------------------------------------------------------------------------

class TestPMALAProposalStep:
    """
    Analytical setup:
      dim=2, beta=0.5, reference N(0, I), derivative GaussianDensity(I, [1,1]).
      For state x=[2,3]:
        grad log Psi(x) = -C_lik^{-1}(x - mu_lik) = [1,1] - [2,3] = [-1,-2]
        drift = sqrt(0.75)*[2,3] + 0.125*[-1,-2]
    """

    DIM = 2
    BETA = 0.5
    STATE_COORD = np.array([2.0, 3.0])

    def setup_method(self):
        refCov = IIDCovarianceMatrix(self.DIM, 1.0)
        refMean = Vector(np.zeros(self.DIM))
        prior = Gaussian(refCov, refMean)
        likCov = IIDCovarianceMatrix(self.DIM, 1.0)
        likMean = Vector(np.ones(self.DIM))
        deriv = GaussianDensity(likCov, likMean)
        self.target = RadonNikodym(prior, deriv)
        self.proposal = PMALAProposal(self.target, self.BETA)
        self.state = Vector(self.STATE_COORD.copy())

        gradLogPsi = np.array([1., 1.]) - np.array([2., 3.])  # [-1, -2]
        self.expectedDrift = (
            np.sqrt(1. - self.BETA**2) * self.STATE_COORD
            + 0.5 * self.BETA**2 * gradLogPsi
        )

    def test_drift_is_correct(self):
        computed = self.proposal._drift(self.state)
        assert np.allclose(computed, self.expectedDrift, atol=1e-12)

    def test_proposal_mean_matches_drift(self):
        rng = np.random.default_rng(7)
        proposals = np.array([
            self.proposal.propose(self.state, rng)[0].proposal.coordinate
            for _ in range(5000)
        ])
        assert np.allclose(proposals.mean(axis=0), self.expectedDrift, atol=0.05)

    def test_proposal_covariance_is_beta2_times_C(self):
        rng = np.random.default_rng(8)
        proposals = np.array([
            self.proposal.propose(self.state, rng)[0].proposal.coordinate
            for _ in range(5000)
        ])
        sampleCov = np.cov(proposals, rowvar=False)
        expected = self.BETA**2 * np.eye(self.DIM)
        assert np.allclose(sampleCov, expected, atol=0.05)


# ---------------------------------------------------------------------------
# 3. Invariant measure
# ---------------------------------------------------------------------------

class TestPMALAInvariantMeasure:
    """
    Conjugate Gaussian posterior.
      Prior: N(0, 4I)    Likelihood: N([2,-1], 0.5I)
      Posterior: C_post = (0.25 + 2.0)^{-1} I = (4/9) I
                 mu_post = C_post * (1/0.5) * [2,-1] = [16/9, -8/9]
    """

    DIM = 2
    PRIOR_VAR = 4.0
    LIK_VAR = 0.5
    OBS = np.array([2.0, -1.0])

    N_STEPS = 5000
    BURNIN = 500
    BETA = 0.5
    SEED = 42

    @classmethod
    def _analytical_posterior(cls):
        precision = 1.0 / cls.PRIOR_VAR + 1.0 / cls.LIK_VAR
        postVar = 1.0 / precision
        muPost = postVar * (cls.OBS / cls.LIK_VAR)
        return muPost, postVar * np.eye(cls.DIM)

    def setup_method(self):
        np.random.seed(self.SEED)

        refCov = IIDCovarianceMatrix(self.DIM, self.PRIOR_VAR)
        prior = Gaussian(refCov, Vector(np.zeros(self.DIM)))

        likCov = IIDCovarianceMatrix(self.DIM, self.LIK_VAR)
        deriv = GaussianDensity(likCov, Vector(self.OBS.copy()))

        target = RadonNikodym(prior, deriv)

        factory = PMALAFactory()
        factory.target = target
        factory.beta = self.BETA
        sampler = factory.create()
        sampler.run(self.N_STEPS, Vector(np.zeros(self.DIM)))

        self.traj = np.array(sampler.chain.trajectory)
        self.muPost, self.covPost = self._analytical_posterior()

    def test_posterior_mean(self):
        samples = self.traj[self.BURNIN:]
        assert np.allclose(samples.mean(axis=0), self.muPost, atol=0.1)

    def test_posterior_covariance(self):
        rawSamples = self.traj[self.BURNIN:]
        iat = integrated_autocorrelation(rawSamples, method='max')
        thinning = max(1, int(iat)) if np.isfinite(iat) else 1
        samples = rawSamples[::thinning]
        assert np.allclose(np.cov(samples, rowvar=False), self.covPost, atol=0.1)


# ---------------------------------------------------------------------------
# 4. PMALATuner
# ---------------------------------------------------------------------------

class TestPMALATuner:

    DIM = 2
    SEED = 0

    def setup_method(self):
        np.random.seed(self.SEED)

        refCov = IIDCovarianceMatrix(self.DIM, 4.0)
        prior = Gaussian(refCov, Vector(np.zeros(self.DIM)))
        likCov = IIDCovarianceMatrix(self.DIM, 0.5)
        deriv = GaussianDensity(likCov, Vector(np.array([2.0, -1.0])))
        self.target = RadonNikodym(prior, deriv)

    def test_tuner_returns_preconditioned_mala(self):
        factory = PMALAFactory()
        factory.target = self.target
        init = Vector(np.zeros(self.DIM))
        tuner = PMALATuner(factory, init)
        sampler = tuner.tune()
        assert isinstance(sampler, PreconditionedMALA)

    def test_tuned_acceptance_rate_in_range(self):
        factory = PMALAFactory()
        factory.target = self.target
        init = Vector(np.zeros(self.DIM))
        tuner = PMALATuner(factory, init)
        sampler = tuner.tune()
        sampler.run(1000, init)
        rate = sampler.diagnostics.global_acceptance_rate()
        assert 0.1 <= rate <= 0.9
