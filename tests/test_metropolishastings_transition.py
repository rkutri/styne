import numpy as np
import pytest

from styne.mcmc.diagnostics import DummyDiagnostics
from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.proposal import ProposalMethod
from styne.mcmc.transition import EvaluatedState, TransitionData
from styne.parameter.scalar import Scalar
from styne.parameter.vector import Vector


class QuadraticDensity:

    def __init__(self):
        self.evaluations = 0

    def evaluate_log(self, parameter):
        self.evaluations += 1
        return -(parameter.coordinate @ parameter.coordinate)


class GraphQuadraticDensity:

    def evaluate_log(self, parameter):
        return -(parameter.coordinate @ parameter.coordinate)


class FixedOffsetProposal(ProposalMethod):

    def propose(self, state, rng):
        proposal = state.with_coordinate(state.coordinate + 10.)
        return TransitionData(state, proposal), rng


class SymmetricMetropolisHastings(MetropolisHastings):

    def _log_mh_ratio(self, transition):
        return transition.proposed.logDensity - transition.current.logDensity


def test_step_reuses_current_log_density_after_rejection():
    density = QuadraticDensity()
    sampler = SymmetricMetropolisHastings(
        density, FixedOffsetProposal(), DummyDiagnostics(),
        rng=np.random.default_rng(4),
    )
    current = sampler.evaluate_state(Scalar(0.))

    nextState, transition, rng = sampler.step(current, sampler._rng)
    repeatedState, repeatedTransition, _ = sampler.step(nextState, rng)

    assert density.evaluations == 3
    assert not transition.outcome
    assert not repeatedTransition.outcome
    np.testing.assert_allclose(nextState.parameter.coordinate, [0.])
    np.testing.assert_allclose(repeatedState.parameter.coordinate, [0.])


def test_runner_initializes_current_density_once():
    density = QuadraticDensity()
    sampler = SymmetricMetropolisHastings(
        density, FixedOffsetProposal(), DummyDiagnostics(),
        rng=np.random.default_rng(5),
    )

    sampler.run(2, Scalar(0.))

    assert density.evaluations == 3


def test_runner_reuses_evaluated_state_when_continued():
    density = QuadraticDensity()
    sampler = SymmetricMetropolisHastings(
        density, FixedOffsetProposal(), DummyDiagnostics(),
        rng=np.random.default_rng(6),
    )

    sampler.run(1, Scalar(0.))
    sampler.continue_run(1)

    assert density.evaluations == 3


def test_step_compiles_with_jax():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from styne.backend import get_backend

    get_backend("jax")
    density = GraphQuadraticDensity()
    sampler = SymmetricMetropolisHastings(
        density, FixedOffsetProposal(), DummyDiagnostics(),
    )
    current = EvaluatedState(Vector(jnp.array([0.])), jnp.array(0.))

    nextState, transition, _ = jax.jit(sampler.step)(
        current, jax.random.key(3)
    )

    assert get_backend("jax").is_array(nextState.parameter.coordinate)
    assert get_backend("jax").is_array(transition.outcome)


def test_runner_preserves_jax_chain_arrays():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from styne.backend import get_backend

    backend = get_backend("jax")
    sampler = SymmetricMetropolisHastings(
        GraphQuadraticDensity(), FixedOffsetProposal(), DummyDiagnostics(),
        rng=jax.random.key(5),
    )

    sampler.run(1, Vector(jnp.array([0.])))

    assert backend.is_array(sampler.lastState.coordinate)
    assert backend.is_array(sampler.chain.trajectory[-1])


def test_runner_preserves_pytorch_chain_arrays():
    torch = pytest.importorskip("torch")
    from styne.backend import get_backend

    backend = get_backend("pytorch")
    sampler = SymmetricMetropolisHastings(
        GraphQuadraticDensity(), FixedOffsetProposal(), DummyDiagnostics(),
        rng=backend.random_state(5),
    )

    sampler.run(1, Vector(torch.tensor([0.])))

    assert backend.is_array(sampler.lastState.coordinate)
    assert backend.is_array(sampler.chain.trajectory[-1])


def test_transformed_trajectory_compiles_with_jax():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    sampler = SymmetricMetropolisHastings(
        GraphQuadraticDensity(), FixedOffsetProposal(), DummyDiagnostics(),
    )
    finalState, coordinates, _ = sampler.transformed_trajectory(
        2, Vector(jnp.array([0.])), jax.random.key(9)
    )

    np.testing.assert_allclose(coordinates, [[0.], [0.]])
    np.testing.assert_allclose(finalState.coordinate, [0.])


def test_transformed_trajectory_rejects_pytorch_generator_compilation():
    torch = pytest.importorskip("torch")
    from styne.backend import get_backend

    backend = get_backend("pytorch")
    sampler = SymmetricMetropolisHastings(
        GraphQuadraticDensity(), FixedOffsetProposal(), DummyDiagnostics(),
    )

    with pytest.raises(RuntimeError, match="transformed loops"):
        sampler.transformed_trajectory(
            2, Vector(torch.tensor([0.])), backend.random_state(9)
        )
