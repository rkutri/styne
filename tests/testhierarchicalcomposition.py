import numpy as np
import pytest

from styne.gp import GaussianProcess
from styne.gp.dnautility import DNACoarseFinePartition
from styne.mcmc.acceptance import AcceptanceProbability
from styne.mcmc.diagnostics import DummyDiagnostics
from styne.mcmc.method.dart import DARTFactory
from styne.mcmc.method.gibbs import GibbsBuilder
from styne.mcmc.method.mrw import MetropolisedRandomWalk
from styne.mcmc.method.pcn import PreconditionedCrankNicolson
from styne.mcmc.method.pmala import PreconditionedMALA
from styne.model import SGLMM
from styne.parameter import BlockParameter, Vector
from styne.statistics.bayes import HierarchicalBayesModelBuilder
from styne.statistics.conditional import MetropolisWithinGibbsConditional
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.data import Data
from styne.statistics.gaussian import Gaussian
from styne.statistics.hierarchical import (
    SGLMMHyperConditionalDensity,
    SGLMMLatentConditional,
)
from styne.statistics.interface import DensityInterface
from styne.statistics.likelihood import RegressionLikelihood
from styne.statistics.radonnikodym import RadonNikodym
from styne.statistics.response import PoissonResponse
from styne.statistics.stationary import MaternCovariance1D
from styne.utility.grid import UniformGrid


class FixedAcceptance(AcceptanceProbability):

    def __init__(self, accept):
        self._logProbability = 0.0 if accept else float("-inf")

    def log_probability(self, logMHRatio):
        return np.asarray(self._logProbability)


class RecordingAcceptance(FixedAcceptance):

    def log_probability(self, logMHRatio):
        self.logMHRatio = logMHRatio
        return super().log_probability(logMHRatio)


class LogGaussianPullback(DensityInterface):

    def __init__(self, gaussian):
        self._gaussian = gaussian

    @property
    def domainType(self):
        return Vector

    @property
    def domainDimension(self):
        return self._gaussian.density.domainDimension

    def evaluate_log(self, parameter):
        logCoordinate = np.log(parameter.coordinate)
        return (
            self._gaussian.density.evaluate_log(Vector(logCoordinate))
            - np.sum(logCoordinate)
        )


def spatial_components():
    sites = UniformGrid(0.0, 1.0, 5)
    data = Data(1, sites.to_array())
    data.measurement = np.zeros((5, 1))
    covariance = MaternCovariance1D(0.3, 1.5, 1.0)
    gp = GaussianProcess.dna(covariance, q=3, d=1)
    coarseGP = GaussianProcess.dna(covariance, q=1, d=1)
    model = SGLMM(gp, sites)
    coarseModel = SGLMM(coarseGP, sites)
    likelihood = RegressionLikelihood(data, model, PoissonResponse())
    coarseLikelihood = RegressionLikelihood(
        data, coarseModel, PoissonResponse()
    )
    target = RadonNikodym(gp.measure, likelihood)
    surrogate = RadonNikodym(coarseGP.measure, coarseLikelihood)
    partition = DNACoarseFinePartition(gp, qC=1, d=1)
    return (
        gp,
        coarseGP,
        model,
        likelihood,
        target,
        surrogate,
        partition,
        partition.fine_measure(),
    )


def fresh_latent_target(gp, likelihood, hyperState):
    rho, sigma = np.exp(hyperState.coordinate)
    covariance = MaternCovariance1D(
        rho, gp.covarianceFunction._smoothness, sigma**2
    )
    replacementGP = gp.with_covariance_function(covariance)
    replacementLikelihood = likelihood.with_model(
        likelihood.model.with_gp(replacementGP)
    )
    return RadonNikodym(replacementGP.measure, replacementLikelihood)


def test_latent_conditioning_reconstructs_independent_spatial_graphs():
    (
        gp,
        coarseGP,
        _,
        likelihood,
        target,
        surrogate,
        partition,
        finePrior,
    ) = spatial_components()
    from styne.mcmc.localised import LocalisedSurrogateDensity

    localised = LocalisedSurrogateDensity(
        2.0,
        1.0,
        surrogate,
        spectralWeights=coarseGP.expansion.spectralWeights,
    )
    template = SGLMMLatentConditional(
        target,
        gp,
        coarseGP=coarseGP,
        partition=partition,
        finePrior=finePrior,
        localisedDensity=localised,
    )
    latent = Vector(np.full(gp.parameterDimension, 0.2))
    firstState = BlockParameter([
        latent, Vector(np.log([0.2, 1.5]))
    ])
    secondState = BlockParameter([
        latent, Vector(np.log([0.6, 0.7]))
    ])

    first = template.condition(firstState)
    second = template.condition(secondState)
    firstFresh = fresh_latent_target(gp, likelihood, firstState.block(1))
    secondFresh = fresh_latent_target(gp, likelihood, secondState.block(1))

    np.testing.assert_allclose(
        first.evaluate_log(latent), firstFresh.evaluate_log(latent)
    )
    np.testing.assert_allclose(
        second.evaluate_log(latent), secondFresh.evaluate_log(latent)
    )
    assert first.reference is first.gp.measure
    assert first.derivative.model._gp is first.gp
    assert first.localisedDensity.surrogateDensity.reference \
        is first.coarseGP.measure
    assert first.localisedDensity.surrogateDensity.derivative.model._gp \
        is first.coarseGP
    np.testing.assert_allclose(
        first.localisedDensity.spectralWeights,
        first.coarseGP.expansion.spectralWeights,
    )
    assert first.gp is not second.gp
    assert first.localisedDensity is not second.localisedDensity
    assert template.gp is gp
    assert template.reference is gp.measure
    assert template.derivative.model._gp is gp


def test_spatial_pcn_conditions_and_executes_independent_updates():
    gp, _, _, likelihood, target, _, _, _ = spatial_components()
    template = SGLMMLatentConditional(target, gp)
    directSampler = PreconditionedCrankNicolson(
        template, 0.45, DummyDiagnostics()
    )
    sampler = PreconditionedCrankNicolson(
        target,
        0.45,
        DummyDiagnostics(),
        acceptance=FixedAcceptance(True),
    )
    sampler.target = template
    update = MetropolisWithinGibbsConditional(sampler, blockIdx=0)
    latent = Vector(np.linspace(-0.4, 0.5, gp.parameterDimension))
    firstState = BlockParameter([
        latent, Vector(np.log([0.2, 1.5]))
    ])
    secondState = BlockParameter([
        latent, Vector(np.log([0.6, 0.7]))
    ])

    first = update.condition(firstState)
    second = update.condition(secondState)
    firstTarget = first._sampler.target
    secondTarget = second._sampler.target
    firstFresh = fresh_latent_target(gp, likelihood, firstState.block(1))
    secondFresh = fresh_latent_target(gp, likelihood, secondState.block(1))
    firstValue = firstTarget.evaluate_log(latent)
    secondValue = secondTarget.evaluate_log(latent)

    np.testing.assert_allclose(firstValue, firstFresh.evaluate_log(latent))
    np.testing.assert_allclose(secondValue, secondFresh.evaluate_log(latent))
    np.testing.assert_allclose(
        first._sampler.initial_state(latent).logDensity,
        firstFresh.derivative.evaluate_log(latent),
    )
    assert directSampler.proposal.referenceMeasure is template.reference
    assert first._sampler.proposal.referenceMeasure is firstTarget.reference
    assert second._sampler.proposal.referenceMeasure is secondTarget.reference
    assert firstTarget.reference is firstTarget.gp.measure
    assert secondTarget.reference is secondTarget.gp.measure

    firstSample, _ = first.sample(np.random.default_rng(5))
    secondSample, _ = second.sample(np.random.default_rng(7))

    assert firstSample.dimension == gp.parameterDimension
    assert secondSample.dimension == gp.parameterDimension
    np.testing.assert_allclose(firstTarget.evaluate_log(latent), firstValue)
    np.testing.assert_allclose(secondTarget.evaluate_log(latent), secondValue)
    assert firstTarget.gp is not secondTarget.gp
    assert sampler.target is template
    assert template.gp is gp
    assert template.reference is gp.measure
    assert template.derivative.model._gp is gp


@pytest.mark.parametrize(('accept', 'seed'), ((True, 8), (False, 9)))
def test_spatial_pcn_acceptance_uses_only_rn_factor(accept, seed):
    gp, _, _, _, target, _, _, _ = spatial_components()
    latent = Vector(np.linspace(-0.3, 0.4, gp.parameterDimension))
    joint = BlockParameter([
        latent, Vector(np.log([0.45, 1.8]))
    ])
    conditioned = SGLMMLatentConditional(target, gp).condition(joint)
    acceptance = RecordingAcceptance(accept)
    sampler = PreconditionedCrankNicolson(
        conditioned,
        0.65,
        DummyDiagnostics(),
        acceptance=acceptance,
    )
    current = sampler.initial_state(latent)

    nextState, transition, _ = sampler.step(
        current, np.random.default_rng(seed)
    )
    proposal = transition.proposed.parameter
    expectedRatio = (
        conditioned.derivative.evaluate_log(proposal)
        - conditioned.derivative.evaluate_log(latent)
    )
    fullRatio = (
        conditioned.evaluate_log(proposal)
        - conditioned.evaluate_log(latent)
    )

    np.testing.assert_allclose(current.logDensity,
                               conditioned.derivative.evaluate_log(latent))
    np.testing.assert_allclose(acceptance.logMHRatio, expectedRatio)
    assert not np.isclose(expectedRatio, fullRatio)
    assert bool(transition.outcome) is accept
    expectedCoordinate = proposal.coordinate if accept else latent.coordinate
    np.testing.assert_array_equal(
        nextState.parameter.coordinate, expectedCoordinate
    )


def test_spatial_pmala_conditions_rn_factor_gradient():
    gp, _, _, likelihood, target, _, _, _ = spatial_components()
    template = SGLMMLatentConditional(target, gp)
    latent = Vector(np.linspace(-0.2, 0.3, gp.parameterDimension))
    firstState = BlockParameter([
        latent, Vector(np.log([0.25, 1.4]))
    ])
    secondState = BlockParameter([
        latent, Vector(np.log([0.55, 0.8]))
    ])

    with pytest.raises(ValueError, match="gradient for the NumPy backend"):
        PreconditionedMALA(template, 0.2, DummyDiagnostics())

    sampler = PreconditionedMALA(
        target,
        0.2,
        DummyDiagnostics(),
        acceptance=FixedAcceptance(True),
        gradient=target.derivative.evaluate_log_gradient,
    )
    sampler.target = template
    update = MetropolisWithinGibbsConditional(sampler, blockIdx=0)
    first = update.condition(firstState)
    second = update.condition(secondState)

    for conditionedUpdate, state in (
            (first, firstState), (second, secondState)):
        conditionedTarget = conditionedUpdate._sampler.target
        proposal = conditionedUpdate._sampler.proposal
        fresh = fresh_latent_target(gp, likelihood, state.block(1))

        assert proposal._target is conditionedTarget
        assert proposal.referenceMeasure is conditionedTarget.reference
        assert proposal._gradient.__self__ is conditionedTarget.derivative
        np.testing.assert_allclose(
            proposal._gradient(latent),
            fresh.derivative.evaluate_log_gradient(latent),
        )
        np.testing.assert_allclose(
            conditionedUpdate._sampler.initial_state(latent).logDensity,
            fresh.evaluate_log(latent),
        )
        sample, _ = conditionedUpdate.sample(np.random.default_rng(6))
        assert sample.dimension == gp.parameterDimension

    assert first._sampler.target.gp is not second._sampler.target.gp
    assert sampler.target is template
    assert template.gp is gp


def test_spatial_latent_conditional_rejects_non_rn_target():
    gp, _, _, _, _, _, _, _ = spatial_components()

    with pytest.raises(TypeError, match="RadonNikodym instance"):
        SGLMMLatentConditional(LogGaussianPullback(gp.measure), gp)


def test_conditioned_dart_rebinds_coarse_and_fine_proposals():
    (
        gp,
        coarseGP,
        _,
        _,
        target,
        surrogate,
        partition,
        finePrior,
    ) = spatial_components()
    latent = Vector(np.zeros(gp.parameterDimension))
    factory = DARTFactory(root="pcn")
    factory.target = target
    factory.surrogate = [surrogate]
    factory.regularisation = [2.0]
    factory.nChain = [2]
    factory.burnin = 0
    factory.root.beta = 0.5
    factory.partition = partition
    factory.finePrior = finePrior
    factory.spectralWeights = coarseGP.expansion.spectralWeights
    factory.crankUpInitialState = latent
    sampler = factory.create()
    wrapper = SGLMMLatentConditional(
        target,
        gp,
        coarseGP=coarseGP,
        partition=partition,
        finePrior=finePrior,
        localisedDensity=factory.localisedDensity,
    )
    sampler.target = wrapper
    transition = MetropolisWithinGibbsConditional(
        sampler, blockIdx=0, nSteps=1
    )
    state = BlockParameter([
        latent, Vector(np.log([0.45, 1.8]))
    ])

    conditioned = transition.condition(state)
    conditionedTarget = conditioned._sampler.target
    proposal = conditioned._sampler.proposal
    coarseProposal = proposal._coarseProposal

    assert proposal._pFinePrior is conditionedTarget.finePrior
    assert proposal._fineKernel.referenceMeasure is conditionedTarget.finePrior
    assert coarseProposal.density is conditionedTarget.localisedDensity
    assert coarseProposal.density.surrogateDensity.reference \
        is conditionedTarget.coarseGP.measure
    assert coarseProposal.correction._surrogateMeasure \
        is coarseProposal.measure
    assert sampler.target is wrapper
    assert sampler.proposal._pFinePrior is finePrior


def test_hyper_target_matches_fresh_models_after_accept_and_reject():
    gp, _, model, likelihood, _, _, _, _ = spatial_components()
    root = Gaussian(
        IIDCovarianceMatrix(2, 1.0), Vector(np.log([0.3, 1.0]))
    )
    prior = LogGaussianPullback(root)
    latent = Vector(np.full(gp.parameterDimension, 0.1))
    initialHyper = Vector(np.log([0.3, 1.0]))
    joint = BlockParameter([latent, initialHyper])
    target = SGLMMHyperConditionalDensity(
        prior, gp, model, likelihood, latent
    ).condition(joint)

    acceptedSampler = MetropolisedRandomWalk(
        target,
        IIDCovarianceMatrix(2, 0.02),
        DummyDiagnostics(),
        acceptance=FixedAcceptance(True),
    )
    current = acceptedSampler.initial_state(initialHyper)
    accepted, acceptedTransition, _ = acceptedSampler.step(
        current, np.random.default_rng(4)
    )
    rejectedSampler = MetropolisedRandomWalk(
        target,
        IIDCovarianceMatrix(2, 0.02),
        DummyDiagnostics(),
        acceptance=FixedAcceptance(False),
    )
    rejectedCurrent = rejectedSampler.initial_state(accepted.parameter)
    rejected, rejectedTransition, _ = rejectedSampler.step(
        rejectedCurrent, np.random.default_rng(9)
    )

    for transition in (acceptedTransition, rejectedTransition):
        candidate = transition.proposed.parameter
        rho, sigma = np.exp(candidate.coordinate)
        candidateGP = gp.with_covariance_function(
            MaternCovariance1D(rho, 1.5, sigma**2)
        )
        candidateLikelihood = likelihood.with_model(
            model.with_gp(candidateGP)
        )
        expected = prior.evaluate_log(Vector(np.exp(candidate.coordinate)))
        expected += candidateLikelihood.evaluate_log(latent)
        expected += np.sum(candidate.coordinate)
        np.testing.assert_allclose(
            transition.proposed.logDensity, expected
        )

    assert bool(acceptedTransition.outcome)
    assert not bool(rejectedTransition.outcome)
    np.testing.assert_array_equal(
        accepted.parameter.coordinate,
        acceptedTransition.proposed.parameter.coordinate,
    )
    np.testing.assert_array_equal(
        rejected.parameter.coordinate, accepted.parameter.coordinate
    )
    np.testing.assert_allclose(
        rejected.logDensity, target.evaluate_log(accepted.parameter)
    )


def test_public_spatial_hierarchy_sweeps_both_named_blocks():
    gp, _, model, likelihood, target, _, _, _ = spatial_components()
    latent = Vector(np.linspace(-0.2, 0.2, gp.parameterDimension))
    hyper = Vector(np.log([0.3, 1.0]))
    root = Gaussian(IIDCovarianceMatrix(2, 1.0), hyper)
    prior = LogGaussianPullback(root)
    latentFactor = SGLMMLatentConditional(target, gp)
    latentSampler = PreconditionedCrankNicolson(
        latentFactor,
        0.35,
        DummyDiagnostics(),
        acceptance=FixedAcceptance(True),
    )
    latentUpdate = MetropolisWithinGibbsConditional(
        latentSampler, blockIdx=0
    )
    hyperTarget = SGLMMHyperConditionalDensity(
        prior, gp, model, likelihood, latent
    )
    hyperSampler = MetropolisedRandomWalk(
        hyperTarget,
        IIDCovarianceMatrix(2, 0.01),
        DummyDiagnostics(),
        acceptance=FixedAcceptance(True),
    )
    hyperUpdate = MetropolisWithinGibbsConditional(
        hyperSampler, blockIdx=1
    )
    hierarchy = (
        HierarchicalBayesModelBuilder()
        .set_root(root, name="hyperparameters")
        .add_conditional(latentFactor, name="latent")
        .add_update(latentUpdate)
        .add_update(hyperUpdate)
        .build()
    )
    samplerBuilder = GibbsBuilder()
    samplerBuilder.model = hierarchy
    sampler = samplerBuilder.build()
    state = BlockParameter(
        [latent, hyper],
        names={"latent": 0, "hyperparameters": 1},
    )

    nextState, _, _ = sampler.step(state, np.random.default_rng(12))

    assert hierarchy.nBlocks == 2
    assert nextState.nBlocks == 2
    assert nextState.names == state.names
    assert nextState.block(0).dimension == gp.parameterDimension
    assert nextState.block(1).dimension == 2
    assert not np.array_equal(
        nextState.block(0).coordinate, state.block(0).coordinate
    )
    assert not np.array_equal(
        nextState.block(1).coordinate, state.block(1).coordinate
    )
