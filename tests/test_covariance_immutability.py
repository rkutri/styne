import numpy as np
import pytest

from styne.statistics.covariance import (
    DenseCovarianceMatrix,
    DiagonalCovarianceMatrix,
)


def test_with_scaling_returns_independent_covariance():
    covariance = DiagonalCovarianceMatrix([1.5, 0.6])

    scaled = covariance.with_scaling(1.25)

    assert covariance.scaling == 1.0
    assert scaled.scaling == 1.25
    np.testing.assert_allclose(
        covariance.apply(np.array([1.0, 2.0])), [1.5, 1.2]
    )
    np.testing.assert_allclose(
        scaled.apply(np.array([1.0, 2.0])), [1.875, 1.5]
    )


def test_covariance_public_state_cannot_be_reassigned():
    values = np.array([1.5, 0.6])
    covariance = DiagonalCovarianceMatrix(values)
    values[0] = 5.0

    with pytest.raises(AttributeError):
        covariance.scaling = 1.25
    with pytest.raises(AttributeError):
        covariance.marginalVariance = values

    np.testing.assert_allclose(covariance.marginalVariance, [1.5, 0.6])


def test_dense_covariance_does_not_share_constructor_state():
    matrix = np.array([[2.0, 0.3], [0.3, 1.1]])
    covariance = DenseCovarianceMatrix(matrix)
    matrix[0, 0] = 5.0

    np.testing.assert_allclose(
        covariance.to_dense(), [[2.0, 0.3], [0.3, 1.1]]
    )
