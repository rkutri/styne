import numpy as np
import pytest

from styne.mcmc.diagnostics import AcceptanceRateDiagnostics, DummyDiagnostics
from styne.mcmc.method.mlda import (
    MLDAProposal,
    MultilevelDelayedAcceptanceMCMC,
)
from styne.mcmc.method.mrw import MetropolisedRandomWalk
from styne.mcmc.surrogate import SurrogateTransitionMeasure
from styne.parameter.vector import Vector
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.dirac import DiracMeasure
from styne.statistics.gaussian import GaussianDensity


def test_mlda_proposal_keeps_surrogate_template_state_unchanged():
    target = GaussianDensity(IIDCovarianceMatrix(1, 1.0), Vector([0.0]))
    sampler = MetropolisedRandomWalk(
        target, IIDCovarianceMatrix(1, 0.5), AcceptanceRateDiagnostics()
    )
    initialMeasure = DiracMeasure()
    surrogate = SurrogateTransitionMeasure(sampler, 2, initialMeasure)
    proposal = MLDAProposal(surrogate)

    transition, nextRng = proposal.propose(
        Vector([1.0]), np.random.default_rng(7)
    )

    assert transition.proposal.dimension == 1
    assert nextRng is not None
    assert initialMeasure.location is None
    assert sampler.lastState is None
    assert sampler.chain.length == 0


def test_mlda_transformed_trajectory_compiles_with_jax():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    density = GaussianDensity(
        IIDCovarianceMatrix(1, jnp.array(1.)), Vector(jnp.zeros(1))
    )
    surrogateSampler = MetropolisedRandomWalk(
        density, IIDCovarianceMatrix(1, jnp.array(0.5)), DummyDiagnostics()
    )
    sampler = MultilevelDelayedAcceptanceMCMC(
        density,
        SurrogateTransitionMeasure(surrogateSampler, 2),
        DummyDiagnostics(),
    )

    finalState, trajectory, nextRng = sampler.transformed_trajectory(
        2, Vector(jnp.zeros(1)), jax.random.key(3)
    )

    assert finalState.coordinate.shape == (1,)
    assert trajectory.shape == (2, 1)
    assert isinstance(nextRng, jax.Array)
