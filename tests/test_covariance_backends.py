import numpy as np
import pytest

from styne.statistics.covariance import (
    DenseCovarianceMatrix,
    DiagonalCovarianceMatrix,
    IIDCovarianceMatrix,
)


def test_numpy_covariance_scalars_are_rank_zero_arrays():
    covariance = DenseCovarianceMatrix(
        np.array([[2.0, 0.3], [0.3, 1.1]]), scaling=1.25
    )

    assert covariance.log_determinant().shape == ()
    assert covariance.quadratic_form(np.array([0.4, -1.2])).shape == ()
    assert covariance.dual_quadratic_form(
        np.array([0.4, -1.2])
    ).shape == ()


def test_pytorch_covariances_preserve_graph_and_batches():
    torch = pytest.importorskip("torch", reason="PyTorch is optional")
    matrix = torch.tensor(
        [[2.0, 0.3], [0.3, 1.1]], dtype=torch.float64, requires_grad=True
    )
    scaling = torch.tensor(1.25, dtype=torch.float64, requires_grad=True)
    points = torch.tensor([[0.4, -1.2], [1.0, 0.5]], dtype=torch.float64)

    covariance = DenseCovarianceMatrix(matrix, scaling)
    value = covariance.log_determinant() + covariance.dual_quadratic_form(
        points
    ).sum()
    matrixGradient, scalingGradient = torch.autograd.grad(
        value, (matrix, scaling)
    )

    assert value.shape == ()
    assert covariance.apply(points).shape == points.shape
    assert covariance.apply_inverse(points).shape == points.shape
    assert torch.isfinite(matrixGradient).all()
    assert torch.isfinite(scalingGradient)

    diagonal = DiagonalCovarianceMatrix(
        torch.tensor([1.5, 0.6], dtype=torch.float32)
    )
    iid = IIDCovarianceMatrix(2, torch.tensor(0.5, dtype=torch.float32))
    assert diagonal.to_cholesky().dtype == torch.float32
    assert iid.apply(points.to(dtype=torch.float32)).shape == points.shape


def test_jax_covariances_preserve_graph_and_batches():
    jax = pytest.importorskip("jax", reason="JAX is optional")
    jnp = pytest.importorskip("jax.numpy", reason="JAX is optional")
    matrix = jnp.array([[2.0, 0.3], [0.3, 1.1]], dtype=jnp.float32)
    points = jnp.array([[0.4, -1.2], [1.0, 0.5]], dtype=jnp.float32)

    def objective(covarianceEntries, scaling):
        covariance = DenseCovarianceMatrix(covarianceEntries, scaling)
        return covariance.log_determinant() + jnp.sum(
            covariance.dual_quadratic_form(points)
        )

    matrixGradient, scalingGradient = jax.grad(objective, argnums=(0, 1))(
        matrix, jnp.array(1.25, dtype=jnp.float32)
    )
    covariance = DenseCovarianceMatrix(matrix, jnp.array(1.25))

    assert covariance.apply(points).shape == points.shape
    assert covariance.apply_inverse(points).shape == points.shape
    assert jnp.all(jnp.isfinite(matrixGradient))
    assert jnp.isfinite(scalingGradient)

    diagonal = DiagonalCovarianceMatrix(jnp.array([1.5, 0.6]))
    iid = IIDCovarianceMatrix(2, jnp.array(0.5))
    assert diagonal.to_cholesky().dtype == jnp.float32
    assert iid.apply(points).shape == points.shape
