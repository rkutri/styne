import numpy as np
import pytest

from styne.mcmc.diagnostics import DummyDiagnostics
from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.method.mala import (
    MALAProposal,
    MetropolisAdjustedLangevinAlgorithm,
)
from styne.mcmc.proposal import ProposalMethod, zero_correction
from styne.mcmc.transition import EvaluatedState
from styne.parameter.scalar import Scalar
from styne.parameter.vector import Vector
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.gaussian import GaussianDensity


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
        return self.record(state, proposal, zero_correction(state)), rng



def test_step_reuses_current_log_density_after_rejection():
    density = QuadraticDensity()
    sampler = MetropolisHastings(
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
    sampler = MetropolisHastings(
        density, FixedOffsetProposal(), DummyDiagnostics(),
        rng=np.random.default_rng(5),
    )

    sampler.run(2, Scalar(0.))

    assert density.evaluations == 3


def test_runner_reuses_evaluated_state_when_continued():
    density = QuadraticDensity()
    sampler = MetropolisHastings(
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
    sampler = MetropolisHastings(
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
    sampler = MetropolisHastings(
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
    sampler = MetropolisHastings(
        GraphQuadraticDensity(), FixedOffsetProposal(), DummyDiagnostics(),
        rng=backend.random_state(5),
    )

    sampler.run(1, Vector(torch.tensor([0.])))

    assert backend.is_array(sampler.lastState.coordinate)
    assert backend.is_array(sampler.chain.trajectory[-1])


def test_transformed_trajectory_compiles_with_jax():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    sampler = MetropolisHastings(
        GraphQuadraticDensity(), FixedOffsetProposal(), DummyDiagnostics(),
    )
    result = sampler.transformed_trajectory(
        2, Vector(jnp.array([0.])), jax.random.key(9)
    )

    np.testing.assert_allclose(result.coordinates, [[0.], [0.]])
    np.testing.assert_allclose(result.final.coordinate, [0.])


def test_transformed_trajectory_rejects_pytorch_generator_compilation():
    torch = pytest.importorskip("torch")
    from styne.backend import get_backend

    backend = get_backend("pytorch")
    sampler = MetropolisHastings(
        GraphQuadraticDensity(), FixedOffsetProposal(), DummyDiagnostics(),
    )

    with pytest.raises(RuntimeError, match="transformed loops"):
        sampler.transformed_trajectory(
            2, Vector(torch.tensor([0.])), backend.random_state(9)
        )


def test_batched_mala_step_uses_jax_autodiff():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from styne.backend import get_backend

    density = GaussianDensity(
        IIDCovarianceMatrix(1, jnp.array(1.)),
        Vector(jnp.zeros(1)),
    )
    sampler = MetropolisAdjustedLangevinAlgorithm(
        density, 0.2, DummyDiagnostics()
    )
    current = sampler.initial_state(
        Vector(jnp.array([[0.5], [-0.5]]))
    )

    nextState, transition, _ = jax.jit(sampler.step)(
        current, jax.random.key(8)
    )

    assert nextState.parameter.coordinate.shape == (2, 1)
    assert get_backend("jax").is_array(transition.outcome)


def test_mala_drift_compiles_with_pytorch_autodiff():
    torch = pytest.importorskip("torch")
    from styne.backend import get_backend

    backend = get_backend("pytorch")
    density = GaussianDensity(
        IIDCovarianceMatrix(1, torch.tensor(1.)),
        Vector(torch.zeros(1)),
    )
    sampler = MetropolisAdjustedLangevinAlgorithm(
        density, 0.2, DummyDiagnostics()
    )
    compiled = backend.compile(
        lambda coordinate: sampler.proposal._drift(Vector(coordinate)),
        backend="eager",
        fullgraph=True,
    )

    drift = compiled(torch.tensor([0.5]))

    torch.testing.assert_close(drift, torch.tensor([0.49]))


def test_mala_numpy_requires_an_explicit_gradient():
    density = GaussianDensity(
        IIDCovarianceMatrix(1, 1.), Vector(np.zeros(1))
    )
    proposal = MALAProposal(1, 0.2, logDensityCallable=density.evaluate_log)

    with pytest.raises(ValueError, match="NumPy backend"):
        proposal._drift(Vector(np.array([0.5])))


def test_mala_numpy_explicit_gradient_controls_the_drift():
    proposal = MALAProposal(
        1, 0.2, logGradientCallable=lambda state: -state.coordinate
    )

    drift = proposal._drift(Vector(np.array([0.5])))

    np.testing.assert_allclose(drift, [0.49])


def test_mala_retarget_rebinds_density_and_bound_gradient():
    oldDensity = GaussianDensity(
        IIDCovarianceMatrix(1, 1.0), Vector(np.zeros(1))
    )
    newDensity = GaussianDensity(
        IIDCovarianceMatrix(1, 2.0), Vector(np.array([3.0]))
    )
    sampler = MetropolisAdjustedLangevinAlgorithm(
        oldDensity,
        0.2,
        DummyDiagnostics(),
        gradient=oldDensity.evaluate_log_gradient,
    )

    sampler.target = newDensity
    state = Vector(np.array([1.0]))
    expected = state.coordinate + 0.5 * 0.2 ** 2 * (
        newDensity.evaluate_log_gradient(state)
    )

    assert sampler.proposal._logGradient.__self__ is newDensity
    np.testing.assert_allclose(sampler.proposal._drift(state), expected)


def test_mala_drift_retains_pytorch_gradient_connectivity():
    torch = pytest.importorskip("torch")

    density = GaussianDensity(
        IIDCovarianceMatrix(1, torch.tensor(1.)),
        Vector(torch.zeros(1)),
    )
    sampler = MetropolisAdjustedLangevinAlgorithm(
        density, 0.2, DummyDiagnostics()
    )
    coordinate = torch.tensor([0.5], requires_grad=True)

    drift = sampler.proposal._drift(Vector(coordinate))
    gradient, = torch.autograd.grad(drift.sum(), coordinate)

    torch.testing.assert_close(gradient, torch.tensor([0.98]))


def radon_nikodym_target():
    from styne.statistics.gaussian import Gaussian
    from styne.statistics.radonnikodym import RadonNikodym

    reference = Gaussian(IIDCovarianceMatrix(2, 1.0), Vector(np.zeros(2)))
    likelihood = GaussianDensity(
        IIDCovarianceMatrix(2, 0.3), Vector(np.array([1.0, -0.5]))
    )
    return RadonNikodym(reference, likelihood)



def test_radon_nikodym_target_is_stored_relative_to_its_reference():
    from styne.mcmc.method.mrw import MetropolisedRandomWalk

    target = radon_nikodym_target()
    sampler = MetropolisedRandomWalk(
        target, IIDCovarianceMatrix(2, 0.5), DummyDiagnostics()
    )
    current = sampler.initial_state(Vector(np.array([0.3, 0.1])))

    _, transition, _ = sampler.step(current, np.random.default_rng(3))
    x, y = transition.current.parameter, transition.proposed.parameter

    np.testing.assert_allclose(
        current.logDensity, target.derivative.evaluate_log(x)
    )
    np.testing.assert_allclose(
        sampler._log_mh_ratio(transition, sampler.proposal.reference),
        target.evaluate_log(y) - target.evaluate_log(x),
    )


def test_target_ratio_changes_reference_to_that_of_the_proposal():
    from styne.mcmc.method.pcn import PCNProposal
    from styne.statistics.gaussian import Gaussian

    target = radon_nikodym_target()
    proposalReference = Gaussian(
        IIDCovarianceMatrix(2, 2.0), Vector(np.array([0.5, 0.0]))
    )
    sampler = MetropolisHastings(
        target, PCNProposal(proposalReference, 0.6), DummyDiagnostics()
    )
    current = sampler.initial_state(Vector(np.array([0.3, 0.1])))

    _, transition, _ = sampler.step(current, np.random.default_rng(5))
    x, y = transition.current.parameter, transition.proposed.parameter
    referenceDensity = proposalReference.density

    np.testing.assert_allclose(
        sampler._log_mh_ratio(transition, sampler.proposal.reference),
        target.evaluate_log(y) - target.evaluate_log(x)
        - referenceDensity.evaluate_log(y) + referenceDensity.evaluate_log(x),
    )
