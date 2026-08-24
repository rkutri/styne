import numpy as np
import pytest

from styne.parameter.vector import Vector
from styne.statistics.pc import JointMaternPCPrior


def make_prior():
    return JointMaternPCPrior(0.5, 0.05, 3.0, 0.05)


def test_joint_pc_prior_constructs_and_preserves_batches():
    prior = make_prior()
    parameter = Vector(np.array([[0.8, 1.2], [1.1, 0.7]]))

    assert prior.evaluate_log(parameter).shape == (2,)
    assert prior.evaluate_log_gradient(parameter).shape == (2, 2)


def test_joint_pc_prior_jax_autodiff_matches_manual_gradient():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    prior = make_prior()
    coordinate = jnp.array([0.8, 1.2])

    gradient = jax.jit(jax.grad(
        lambda value: prior.evaluate_log(Vector(value))
    ))(coordinate)

    np.testing.assert_allclose(
        gradient,
        prior.evaluate_log_gradient(Vector(coordinate)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_joint_pc_prior_pytorch_autodiff_matches_manual_gradient():
    torch = pytest.importorskip("torch")
    prior = make_prior()
    coordinate = torch.tensor(
        [0.8, 1.2], dtype=torch.float64, requires_grad=True
    )

    value = prior.evaluate_log(Vector(coordinate))
    gradient = torch.autograd.grad(value, coordinate)[0]

    torch.testing.assert_close(
        gradient,
        prior.evaluate_log_gradient(Vector(coordinate)),
    )
