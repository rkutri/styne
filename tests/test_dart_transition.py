import numpy as np
import pytest

from styne.mcmc.diagnostics import AcceptanceRateDiagnostics
from styne.mcmc.localised import (
    LocalisedSurrogateDensity,
    LocalisedSurrogateTransitionMeasure,
)
from styne.mcmc.method.dart import LocalisedSurrogateTransition
from styne.mcmc.method.mrw import MetropolisedRandomWalk, MRWProposal
from styne.parameter.vector import Vector
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.gaussian import Gaussian, GaussianDensity


def test_dart_proposal_keeps_localised_surrogate_template_unchanged():
    baseDensity = GaussianDensity(IIDCovarianceMatrix(1, 1.0), Vector([0.0]))
    density = LocalisedSurrogateDensity(1.0, 1.0, baseDensity)
    sampler = MetropolisedRandomWalk(
        density, IIDCovarianceMatrix(1, 0.5), AcceptanceRateDiagnostics()
    )
    measure = LocalisedSurrogateTransitionMeasure(sampler, 2)
    proposal = LocalisedSurrogateTransition(measure, burnin=0, thinning=1)

    transition, _ = proposal.propose(Vector([1.0]), np.random.default_rng(7))

    assert transition.proposed.parameter.dimension == 1
    assert set(transition.auxiliary) == {"logCorrection"}
    np.testing.assert_array_equal(measure.location.coordinate, [0.0])
    assert sampler.lastState is None
    assert sampler.chain.length == 0


def test_zero_subchain_uses_localised_initial_measure():
    baseDensity = GaussianDensity(IIDCovarianceMatrix(1, 1.0), Vector([0.0]))
    density = LocalisedSurrogateDensity(1e-10, 1.0, baseDensity)
    sampler = MetropolisedRandomWalk(
        density, IIDCovarianceMatrix(1, 0.5), AcceptanceRateDiagnostics()
    )
    proposalCovariance = IIDCovarianceMatrix(1, 5.0)
    initialMeasure = Gaussian(proposalCovariance)
    measure = LocalisedSurrogateTransitionMeasure(
        sampler, 0, initialMeasure
    )
    proposal = LocalisedSurrogateTransition(measure, burnin=0, thinning=1)
    state = Vector([-3.0])

    transition, _ = proposal.propose(state, np.random.default_rng(12345))
    mrwTransition, _ = MRWProposal(proposalCovariance).propose(
        state, np.random.default_rng(12345)
    )

    np.testing.assert_array_equal(
        transition.proposed.parameter.coordinate,
        mrwTransition.proposed.parameter.coordinate,
    )
    assert transition.auxiliary["logCorrection"] == 0.0
    assert proposal.reference is None
    with pytest.raises(RuntimeError, match="Mean not set"):
        initialMeasure.mean


@pytest.mark.parametrize("radonNikodym", (False, True))
@pytest.mark.parametrize("diracStart", (True, False))
def test_dart_correction_matches_tempered_surrogate_and_ratio_estimate(
        radonNikodym, diracStart):
    from styne.mcmc.proposal import log_reference_ratio
    from styne.statistics.radonnikodym import RadonNikodym

    reference = Gaussian(IIDCovarianceMatrix(2, 1.0), Vector(np.zeros(2)))
    likelihood = GaussianDensity(
        IIDCovarianceMatrix(2, 0.5), Vector(np.array([1.0, -0.5]))
    )
    surrogate = RadonNikodym(reference, likelihood) if radonNikodym else likelihood
    density = LocalisedSurrogateDensity(0.7, 0.5, surrogate)
    sampler = MetropolisedRandomWalk(
        density, IIDCovarianceMatrix(2, 0.3), AcceptanceRateDiagnostics()
    )
    initialMeasure = None if diracStart else Gaussian(IIDCovarianceMatrix(2, 0.01))
    measure = LocalisedSurrogateTransitionMeasure(sampler, 4, initialMeasure)
    proposal = LocalisedSurrogateTransition(measure, burnin=0, thinning=1)
    state = Vector(np.array([0.2, -0.1]))

    transition, _ = proposal.propose(state, np.random.default_rng(3))
    _, _, trajectory, _ = measure.transition_trajectory(
        state, np.random.default_rng(3)
    )
    proposed = transition.proposed.parameter
    logRatio = proposal.correction.log_ratio_estimate(
        state, proposed, trajectory
    )
    expected = -(
        density.evaluate_log_surrogate(proposed)
        - density.evaluate_log_surrogate(state)
    ) - logRatio

    assert (proposal.reference is reference) is radonNikodym
    np.testing.assert_allclose(
        transition.auxiliary["logCorrection"]
        - log_reference_ratio(proposal.reference, state, proposed),
        expected,
    )
