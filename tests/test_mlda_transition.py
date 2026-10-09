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
from styne.statistics.gaussian import Gaussian, GaussianDensity
from styne.utility.partition import Partition, PartitionRule


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

    result = sampler.transformed_trajectory(
        2, Vector(jnp.zeros(1)), jax.random.key(3)
    )

    assert result.final.coordinate.shape == (1,)
    assert result.coordinates.shape == (2, 1)
    assert isinstance(result.rng, jax.Array)


def test_partitioned_mlda_compiles_with_jax_and_matches_eager_steps():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    rule = PartitionRule([[0, 2], [1]])
    target = GaussianDensity(
        IIDCovarianceMatrix(3, jnp.array(1.)), Vector(jnp.zeros(3))
    )
    surrogate = GaussianDensity(
        IIDCovarianceMatrix(2, jnp.array(1.)), Vector(jnp.zeros(2))
    )
    finePrior = Gaussian(
        IIDCovarianceMatrix(1, jnp.array(1.)), Vector(jnp.zeros(1))
    )
    root = MetropolisedRandomWalk(
        surrogate, IIDCovarianceMatrix(2, jnp.array(0.5)), DummyDiagnostics()
    )
    sampler = MultilevelDelayedAcceptanceMCMC(
        target,
        SurrogateTransitionMeasure(root, 2),
        DummyDiagnostics(),
        partition=Partition(rule, Vector(jnp.zeros(3))),
        finePrior=finePrior,
    )
    initial = Vector(jnp.array([0.3, -0.2, 0.1]))

    result = sampler.transformed_trajectory(
        3, initial, jax.random.key(5)
    )

    state, rng = sampler.initial_state(initial), jax.random.key(5)
    for index in range(3):
        state, _, rng = sampler.step(state, rng)
        np.testing.assert_allclose(
            result.coordinates[index], state.parameter.coordinate, rtol=1e-5
        )
    assert result.coordinates.shape == (3, 3)


def test_surrogate_trajectory_preserves_jax_arrays():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    density = GaussianDensity(
        IIDCovarianceMatrix(1, jnp.array(1.)), Vector(jnp.zeros(1))
    )
    measure = SurrogateTransitionMeasure(
        MetropolisedRandomWalk(
            density,
            IIDCovarianceMatrix(1, jnp.array(0.5)),
            DummyDiagnostics(),
        ),
        2,
    )

    _, _, trajectory, _ = measure.transition_trajectory(
        Vector(jnp.zeros(1)), jax.random.key(3)
    )

    assert isinstance(trajectory, jax.Array)
    assert trajectory.shape == (3, 1)


def test_compiled_root_chain_is_scanned_with_the_eager_random_stream():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    class CountingRandomWalk(MetropolisedRandomWalk):
        calls = 0

        def step(self, currentState, rng):
            type(self).calls += 1
            return super().step(currentState, rng)

    density = GaussianDensity(
        IIDCovarianceMatrix(2, jnp.array(1.)), Vector(jnp.zeros(2))
    )
    measure = SurrogateTransitionMeasure(
        CountingRandomWalk(
            density, IIDCovarianceMatrix(2, jnp.array(0.5)), DummyDiagnostics()
        ),
        6,
    )

    def trajectory(coordinate, key):
        return measure.transition_trajectory(Vector(coordinate), key)[2]

    compiled = jax.jit(trajectory)(jnp.array([0.2, -0.4]), jax.random.key(4))
    tracedSteps = CountingRandomWalk.calls
    eager = trajectory(jnp.array([0.2, -0.4]), jax.random.key(4))

    assert tracedSteps < 6
    assert compiled.shape == (7, 2)
    np.testing.assert_allclose(compiled, eager, rtol=1e-5)


def shared_prior_problem():
    from styne.statistics.radonnikodym import RadonNikodym

    prior = Gaussian(IIDCovarianceMatrix(3, 1.0), Vector(np.zeros(3)))
    target = RadonNikodym(prior, GaussianDensity(
        IIDCovarianceMatrix(3, 0.4), Vector(np.array([1.0, 0.5, -0.2]))
    ))
    return prior, target


def test_mlda_correction_is_relative_to_the_shared_prior():
    from styne.mcmc.transition import TransitionData
    from styne.statistics.radonnikodym import RadonNikodym

    prior, target = shared_prior_problem()
    surrogate = RadonNikodym(prior, GaussianDensity(
        IIDCovarianceMatrix(3, 0.6), Vector(np.array([0.8, 0.4, 0.0]))
    ))
    root = MetropolisedRandomWalk(
        surrogate, IIDCovarianceMatrix(3, 0.5), DummyDiagnostics()
    )
    sampler = MultilevelDelayedAcceptanceMCMC(
        target, SurrogateTransitionMeasure(root, 3), DummyDiagnostics()
    )
    state = Vector(np.array([0.1, -0.2, 0.3]))

    record, _ = sampler.proposal.propose(state, np.random.default_rng(4))
    proposed = record.proposed.parameter
    transition = TransitionData(
        current=sampler.initial_state(state),
        proposed=sampler.initial_state(proposed),
        auxiliary=record.auxiliary,
    )

    assert sampler.proposal.reference is prior
    np.testing.assert_allclose(
        record.auxiliary["logCorrection"],
        surrogate.derivative.evaluate_log(state)
        - surrogate.derivative.evaluate_log(proposed),
    )
    np.testing.assert_allclose(
        sampler._log_mh_ratio(transition, sampler.proposal.reference),
        target.evaluate_log(proposed) - target.evaluate_log(state)
        - surrogate.evaluate_log(proposed) + surrogate.evaluate_log(state),
    )


def test_partitioned_mlda_correction_matches_its_block_densities():
    from styne.mcmc.transition import TransitionData
    from styne.statistics.radonnikodym import RadonNikodym

    _, target = shared_prior_problem()
    rule = PartitionRule([[0, 2], [1]])
    coarsePrior = Gaussian(IIDCovarianceMatrix(2, 1.0), Vector(np.zeros(2)))
    surrogate = RadonNikodym(coarsePrior, GaussianDensity(
        IIDCovarianceMatrix(2, 0.6), Vector(np.array([0.8, 0.0]))
    ))
    finePrior = Gaussian(IIDCovarianceMatrix(1, 1.0), Vector(np.zeros(1)))
    root = MetropolisedRandomWalk(
        surrogate, IIDCovarianceMatrix(2, 0.5), DummyDiagnostics()
    )
    sampler = MultilevelDelayedAcceptanceMCMC(
        target,
        SurrogateTransitionMeasure(root, 3),
        DummyDiagnostics(),
        partition=Partition(rule, Vector(np.zeros(3))),
        finePrior=finePrior,
    )
    state = Vector(np.array([0.1, -0.2, 0.3]))

    record, _ = sampler.proposal.propose(state, np.random.default_rng(6))
    proposed = record.proposed.parameter
    transition = TransitionData(
        current=sampler.initial_state(state),
        proposed=sampler.initial_state(proposed),
        auxiliary=record.auxiliary,
    )
    coarse = [Vector(rule.extract(0, p.coordinate)) for p in (state, proposed)]
    fine = [Vector(rule.extract(1, p.coordinate)) for p in (state, proposed)]

    assert sampler.proposal.reference is None
    np.testing.assert_allclose(
        sampler._log_mh_ratio(transition, sampler.proposal.reference),
        target.evaluate_log(proposed) - target.evaluate_log(state)
        - surrogate.evaluate_log(coarse[1]) + surrogate.evaluate_log(coarse[0])
        - finePrior.density.evaluate_log(fine[1])
        + finePrior.density.evaluate_log(fine[0]),
    )
