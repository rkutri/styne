import numpy as np
import pytest

from styne.mcmc.method.pcn import (
    PCNFactory,
    PCNProposal,
    PreconditionedCrankNicolson,
)
from styne.mcmc.transition import TransitionData
from styne.mcmc.diagnostics import AcceptanceRateDiagnostics
from styne.mcmc.acceptance import AcceptanceProbability, BarkerAcceptance
from styne.statistics.radonnikodym import RadonNikodym
from styne.statistics.gaussian import Gaussian, GaussianDensity
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.interface import DensityInterface
from styne.statistics.measure import AbsolutelyContinuousProbabilityMeasure
from styne.parameter.vector import Vector
from styne.utility.postprocessing import integrated_autocorrelation
from styne.utility.tuning import PCNTuner


def make_valid_target(dim=2):
    refCov = IIDCovarianceMatrix(dim, 1.0)
    refMean = Vector(np.zeros(dim))
    prior = Gaussian(refCov, refMean)
    likCov = IIDCovarianceMatrix(dim, 0.5)
    likMean = Vector(np.ones(dim))
    derivative = GaussianDensity(likCov, likMean)
    return RadonNikodym(prior, derivative)


class ConstantDensity(DensityInterface):

    def __init__(self, dimension):
        self._dimension = dimension

    @property
    def domainType(self):
        return Vector

    @property
    def domainDimension(self):
        return self._dimension

    def evaluate_log(self, parameter):
        return np.sum(parameter.coordinate * 0.0, axis=-1)


class RejectAllAcceptance(AcceptanceProbability):

    def log_probability(self, logMHRatio):
        return np.asarray(float("-inf"))


class NonGaussianMeasure(AbsolutelyContinuousProbabilityMeasure):
    class _Density(DensityInterface):
        @property
        def domainType(self):
            return Vector

        @property
        def domainDimension(self):
            return 2

        def evaluate_log(self, p):
            return 0.0

    _density = _Density()

    @property
    def density(self) -> DensityInterface:
        return self._density

    def draw(self, rng):
        return Vector(np.zeros(2))


class TestPCNSetup:

    def test_rejects_non_gaussian_reference(self):
        stub = NonGaussianMeasure()
        with pytest.raises(NotImplementedError):
            PCNProposal(stub, 0.5)

    def test_sampler_rejects_non_gaussian_rn_reference(self):
        target = RadonNikodym(NonGaussianMeasure(), ConstantDensity(2))

        with pytest.raises(
                NotImplementedError, match="Gaussian reference measure"):
            PreconditionedCrankNicolson(
                target, 0.5, AcceptanceRateDiagnostics()
            )

    def test_rejects_beta_zero(self):
        refCov = IIDCovarianceMatrix(2, 1.0)
        prior = Gaussian(refCov, Vector(np.zeros(2)))
        with pytest.raises(ValueError):
            PCNProposal(prior, 0.0)

    def test_rejects_beta_above_one(self):
        refCov = IIDCovarianceMatrix(2, 1.0)
        prior = Gaussian(refCov, Vector(np.zeros(2)))
        with pytest.raises(ValueError):
            PCNProposal(prior, 1.1)

    def test_accepts_beta_one(self):
        refCov = IIDCovarianceMatrix(2, 1.0)
        prior = Gaussian(refCov, Vector(np.zeros(2)))
        proposal = PCNProposal(prior, 1.0)
        assert proposal.beta == 1.0

    def test_rejects_non_radon_nikodym_target(self):
        density = GaussianDensity(
            IIDCovarianceMatrix(2, 1.0), Vector(np.zeros(2))
        )
        with pytest.raises(TypeError):
            PreconditionedCrankNicolson(density, 0.5, AcceptanceRateDiagnostics())

    def test_proposal_requires_explicit_state(self):
        refCov = IIDCovarianceMatrix(2, 1.0)
        prior = Gaussian(refCov, Vector(np.zeros(2)))
        proposal = PCNProposal(prior, 0.5)
        rng = np.random.default_rng(42)
        transition, nextRng = proposal.propose(
            Vector(np.zeros(2)), rng
        )
        assert transition.state.dimension == 2
        assert nextRng is rng

    def test_name_attribute(self):
        assert PreconditionedCrankNicolson.name == "pCN"

    def test_properties_round_trip(self):
        refCov = IIDCovarianceMatrix(2, 1.0)
        prior = Gaussian(refCov, Vector(np.zeros(2)))
        proposal = PCNProposal(prior, 0.42)
        assert proposal.referenceMeasure is prior
        assert proposal.beta == 0.42


class TestPCNFactorySetup:

    def test_factory_rejects_missing_target(self):
        factory = PCNFactory()
        factory.beta = 0.5
        with pytest.raises(ValueError):
            factory.create()

    def test_factory_rejects_non_radon_nikodym_target(self):
        factory = PCNFactory()
        factory.target = GaussianDensity(
            IIDCovarianceMatrix(2, 1.0), Vector(np.zeros(2))
        )
        factory.beta = 0.5
        with pytest.raises(TypeError):
            factory.create()

    def test_factory_rejects_missing_beta(self):
        factory = PCNFactory()
        factory.target = make_valid_target()
        with pytest.raises(ValueError):
            factory.create()

    def test_factory_rejects_invalid_beta(self):
        factory = PCNFactory()
        factory.target = make_valid_target()
        factory.beta = 0.0
        with pytest.raises(ValueError):
            factory.create()

    def test_factory_creates_correctly(self):
        factory = PCNFactory()
        factory.target = make_valid_target()
        factory.beta = 0.5
        sampler = factory.create()
        assert isinstance(sampler, PreconditionedCrankNicolson)

    def test_factory_acceptance_round_trip(self):
        factory = PCNFactory()
        strategy = BarkerAcceptance()
        factory.acceptance = strategy
        assert factory.acceptance is strategy


class TestPCNProposalStep:

    DIM = 2
    BETA = 0.5
    STATE_COORD = np.array([2.0, 3.0])

    def setup_method(self):
        refCov = IIDCovarianceMatrix(self.DIM, 1.0)
        refMean = Vector(np.zeros(self.DIM))
        self.prior = Gaussian(refCov, refMean)
        self.proposal = PCNProposal(self.prior, self.BETA)
        self.state = Vector(self.STATE_COORD.copy())
        self.expectedDrift = np.sqrt(1.0 - self.BETA**2) * self.STATE_COORD

    def test_proposal_mean_matches_analytical_drift(self):
        rng = np.random.default_rng(7)
        proposals = np.array([
            self.proposal.propose(self.state, rng)[0].proposal.coordinate
            for _ in range(5000)
        ])
        assert np.allclose(
            proposals.mean(axis=0), self.expectedDrift, atol=0.05
        )

    def test_proposal_covariance_is_beta_squared_times_c(self):
        rng = np.random.default_rng(8)
        proposals = np.array([
            self.proposal.propose(self.state, rng)[0].proposal.coordinate
            for _ in range(5000)
        ])
        sampleCov = np.cov(proposals, rowvar=False)
        expected = self.BETA**2 * np.eye(self.DIM)
        assert np.allclose(sampleCov, expected, atol=0.05)

    def test_proposal_compiles_with_jax(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        from styne.backend import get_backend

        prior = Gaussian(
            IIDCovarianceMatrix(self.DIM, jnp.array(1.0)),
            Vector(jnp.zeros(self.DIM)),
        )
        proposal = PCNProposal(prior, self.BETA)

        transition, _ = jax.jit(proposal.propose)(
            Vector(jnp.array(self.STATE_COORD)), jax.random.key(7)
        )

        assert get_backend("jax").is_array(transition.proposal.coordinate)


class TestPCNLogMHRatio:

    def test_log_mh_ratio_equals_derivative_difference(self):
        target = make_valid_target(dim=2)
        sampler = PreconditionedCrankNicolson(
            target, 0.5, AcceptanceRateDiagnostics()
        )
        state = Vector(np.array([1.0, 0.5]))
        proposal = Vector(np.array([1.2, 0.3]))

        expected = (
            target.derivative.evaluate_log(proposal)
            - target.derivative.evaluate_log(state)
        )

        transition = TransitionData(
            current=sampler.evaluate_state(state),
            proposed=sampler.evaluate_state(proposal),
        )
        assert np.isclose(sampler._log_mh_ratio(transition), expected)

    def test_retarget_rebuilds_proposal_from_new_reference(self):
        oldReference = Gaussian(
            IIDCovarianceMatrix(1, 1.0), Vector([0.0])
        )
        newReference = Gaussian(
            IIDCovarianceMatrix(1, 4.0), Vector([10.0])
        )
        sampler = PreconditionedCrankNicolson(
            RadonNikodym(oldReference, ConstantDensity(1)),
            1.0,
            AcceptanceRateDiagnostics(),
        )

        sampler.target = RadonNikodym(newReference, ConstantDensity(1))
        expected, _ = newReference.sample(np.random.default_rng(7))
        current = sampler.initial_state(Vector([10.0]))
        nextState, transition, _ = sampler.step(
            current, np.random.default_rng(7)
        )

        assert sampler.proposal.referenceMeasure is newReference
        assert bool(transition.outcome)
        np.testing.assert_allclose(
            transition.proposed.parameter.coordinate, expected.coordinate
        )
        np.testing.assert_allclose(
            nextState.parameter.coordinate, expected.coordinate
        )

    def test_retarget_uses_new_derivative_for_ratio_and_rejection(self):
        oldReference = Gaussian(
            IIDCovarianceMatrix(1, 1.0), Vector([0.0])
        )
        newReference = Gaussian(
            IIDCovarianceMatrix(1, 4.0), Vector([10.0])
        )
        derivative = GaussianDensity(
            IIDCovarianceMatrix(1, 0.5), Vector([9.0])
        )
        sampler = PreconditionedCrankNicolson(
            RadonNikodym(oldReference, ConstantDensity(1)),
            1.0,
            AcceptanceRateDiagnostics(),
            acceptance=RejectAllAcceptance(),
        )
        sampler.target = RadonNikodym(newReference, derivative)
        current = sampler.initial_state(Vector([10.0]))

        nextState, transition, _ = sampler.step(
            current, np.random.default_rng(4)
        )
        expectedRatio = (
            derivative.evaluate_log(transition.proposed.parameter)
            - derivative.evaluate_log(current.parameter)
        )

        np.testing.assert_allclose(
            sampler._log_mh_ratio(transition), expectedRatio
        )
        assert not bool(transition.outcome)
        np.testing.assert_array_equal(nextState.parameter.coordinate, [10.0])


class TestPCNInvariantMeasure:

    DIM = 2
    PRIOR_VAR = 4.0
    LIK_VAR = 0.5
    OBS = np.array([2.0, -1.0])

    N_STEPS = 5000
    BURNIN = 500
    BETA = 0.4
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

        factory = PCNFactory()
        factory.target = target
        factory.beta = self.BETA
        sampler = factory.create()
        sampler.run(self.N_STEPS, Vector(np.zeros(self.DIM)))

        self.trajectory = np.array(sampler.chain.trajectory)
        self.muPost, self.covPost = self._analytical_posterior()

    def test_posterior_mean(self):
        samples = self.trajectory[self.BURNIN:]
        assert np.allclose(samples.mean(axis=0), self.muPost, atol=0.1)

    def test_posterior_covariance(self):
        rawSamples = self.trajectory[self.BURNIN:]
        iat = integrated_autocorrelation(rawSamples, method='max')
        thinning = max(1, int(iat)) if np.isfinite(iat) else 1
        samples = rawSamples[::thinning]
        assert np.allclose(
            np.cov(samples, rowvar=False), self.covPost, atol=0.1
        )


class TestPCNTuner:

    DIM = 2
    SEED = 0

    def setup_method(self):
        np.random.seed(self.SEED)

        refCov = IIDCovarianceMatrix(self.DIM, 4.0)
        prior = Gaussian(refCov, Vector(np.zeros(self.DIM)))
        likCov = IIDCovarianceMatrix(self.DIM, 0.5)
        deriv = GaussianDensity(likCov, Vector(np.array([2.0, -1.0])))
        self.target = RadonNikodym(prior, deriv)

    def test_tuner_returns_preconditioned_crank_nicolson(self):
        factory = PCNFactory()
        factory.target = self.target
        init = Vector(np.zeros(self.DIM))
        tuner = PCNTuner(factory, init)
        sampler = tuner.tune()
        assert isinstance(sampler, PreconditionedCrankNicolson)

    def test_tuned_acceptance_rate_in_range(self):
        factory = PCNFactory()
        factory.target = self.target
        init = Vector(np.zeros(self.DIM))
        tuner = PCNTuner(factory, init)
        sampler = tuner.tune()
        sampler.run(1000, init)
        rate = sampler.diagnostics.global_acceptance_rate()
        assert 0.1 <= rate <= 0.9
