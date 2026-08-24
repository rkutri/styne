from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from styne.mcmc import EvaluatedState, TransitionData
from styne.parameter.vector import Vector


def test_transition_data_owns_immutable_evaluated_states():
    current = EvaluatedState(Vector(np.array([0.2, -0.1])), np.array(-1.3))
    proposed = EvaluatedState(Vector(np.array([0.4, 0.3])), np.array(-0.9))
    transition = TransitionData(
        current=current,
        proposed=proposed,
        outcome=np.array(True),
        logAcceptanceProbability=np.array(-0.2),
    )

    assert transition.current is current
    assert transition.proposed is proposed
    assert transition.state is current.parameter
    assert transition.proposal is proposed.parameter
    with pytest.raises(FrozenInstanceError):
        transition.outcome = np.array(False)


def test_transition_data_wraps_legacy_parameter_arguments():
    state = Vector(np.array([0.2]))
    proposal = Vector(np.array([0.4]))
    transition = TransitionData(state, proposal, TransitionData.ACCEPTED)

    assert transition.current == EvaluatedState(state)
    assert transition.proposed == EvaluatedState(proposal)
    assert transition.outcome is True


def test_jax_transition_records_flatten_reconstruct_and_compile():
    jax = pytest.importorskip("jax", reason="JAX is optional")
    jnp = pytest.importorskip("jax.numpy", reason="JAX is optional")
    from styne.backend import get_backend

    get_backend("jax")
    transition = TransitionData(
        current=EvaluatedState(
            Vector(jnp.array([0.2, -0.1])), jnp.array(-1.3)
        ),
        proposed=EvaluatedState(
            Vector(jnp.array([0.4, 0.3])), jnp.array(-0.9)
        ),
        outcome=jnp.array(True),
        logAcceptanceProbability=jnp.array(-0.2),
    )
    leaves, structure = jax.tree_util.tree_flatten(transition)
    reconstructed = jax.tree_util.tree_unflatten(structure, leaves)

    @jax.jit
    def update_log_densities(record):
        return TransitionData(
            current=EvaluatedState(
                record.current.parameter,
                record.current.logDensity + 1.0,
            ),
            proposed=record.proposed,
            outcome=record.outcome,
            logAcceptanceProbability=record.logAcceptanceProbability,
            auxiliary=record.auxiliary,
        )

    updated = update_log_densities(reconstructed)
    np.testing.assert_allclose(updated.current.logDensity, -0.3, atol=1e-6)
    np.testing.assert_allclose(
        updated.proposed.parameter.coordinate, [0.4, 0.3]
    )


def test_pytorch_transition_records_flatten_reconstruct_and_transform():
    torch = pytest.importorskip("torch", reason="PyTorch is optional")
    from torch.utils import _pytree
    from styne.backend import get_backend

    get_backend("pytorch")
    transition = TransitionData(
        current=EvaluatedState(
            Vector(torch.tensor([0.2, -0.1])), torch.tensor(-1.3)
        ),
        proposed=EvaluatedState(
            Vector(torch.tensor([0.4, 0.3])), torch.tensor(-0.9)
        ),
        outcome=torch.tensor(1.0),
        logAcceptanceProbability=torch.tensor(-0.2),
        auxiliary=torch.tensor(0.0),
    )
    leaves, structure = _pytree.tree_flatten(transition)
    reconstructed = _pytree.tree_unflatten(leaves, structure)

    gradient = torch.func.grad(
        lambda record: torch.sum(record.current.parameter.coordinate ** 2)
    )(reconstructed)

    torch.testing.assert_close(
        gradient.current.parameter.coordinate, torch.tensor([0.4, -0.2])
    )
