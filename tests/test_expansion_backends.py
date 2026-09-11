import numpy as np
import pytest

from styne.gp.direct import DirectExpansion
from styne.gp.dna import DNAFourierExpansion
from styne.model.representation.bspline import BSpline1D
from styne.utility.grid import UniformGrid


def bound_expansions():
    anchorGrid = UniformGrid(0.0, 1.0, 5)
    evaluationGrid = UniformGrid(0.1, 0.9, 7)
    return (
        DirectExpansion(anchorGrid, 5).bind(evaluationGrid),
        BSpline1D(5, degree=3, boundary=[0.0, 1.0]).bind(
            evaluationGrid
        ),
        DNAFourierExpansion(3, d=1).bind(evaluationGrid),
    )


@pytest.mark.parametrize("evaluation", bound_expansions())
def test_jax_expansions_are_jittable_and_automatically_differentiable(
        evaluation):
    jax = pytest.importorskip("jax", reason="JAX is an optional backend")
    jnp = pytest.importorskip("jax.numpy")
    coefficient = jnp.linspace(-0.5, 0.8, evaluation.dimension)
    direction = jnp.linspace(0.2, 0.7, evaluation.dimension)
    cotangent = jnp.linspace(-0.3, 0.6, 7)

    evaluate = jax.jit(evaluation.evaluate)
    directional = jax.jit(evaluation.directional_derivative)
    adjoint = jax.jit(evaluation.adjoint_derivative)

    value = evaluate(coefficient)
    derivative = directional(coefficient, direction)
    pullback = adjoint(coefficient, cotangent)

    assert value.shape == (7,)
    np.testing.assert_allclose(derivative, evaluate(direction), rtol=1e-6)
    np.testing.assert_allclose(
        jnp.vdot(derivative, cotangent),
        jnp.vdot(direction, pullback),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("evaluation", bound_expansions())
def test_pytorch_expansions_preserve_graph_and_use_automatic_derivatives(
        evaluation):
    torch = pytest.importorskip(
        "torch", reason="PyTorch is an optional backend"
    )
    coefficient = torch.linspace(
        -0.5, 0.8, evaluation.dimension, requires_grad=True
    )
    direction = torch.linspace(0.2, 0.7, evaluation.dimension)
    cotangent = torch.linspace(-0.3, 0.6, 7)

    value = evaluation.evaluate(coefficient)
    derivative = evaluation.directional_derivative(
        coefficient, direction
    )
    pullback = evaluation.adjoint_derivative(coefficient, cotangent)

    assert value.shape == (7,)
    torch.testing.assert_close(derivative, evaluation.evaluate(direction))
    torch.testing.assert_close(
        torch.vdot(derivative, cotangent),
        torch.vdot(direction, pullback),
    )
    value.sum().backward()
    assert coefficient.grad is not None
