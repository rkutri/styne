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

    assert transition.proposal.dimension == 1
    assert len(transition.auxiliary["surrogateTrajectory"]) == 3
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
        transition.proposal.coordinate, mrwTransition.proposal.coordinate
    )
    np.testing.assert_array_equal(
        transition.auxiliary["surrogateTrajectory"],
        mrwTransition.proposal.coordinate[None, :],
    )
    correction = proposal.log_acceptance_correction(
        state,
        transition.proposal,
        transition.auxiliary["surrogateTrajectory"],
        transition.auxiliary["proposalTrajectory"],
    )
    assert correction == 0.0
    with pytest.raises(RuntimeError, match="Mean not set"):
        initialMeasure.mean
