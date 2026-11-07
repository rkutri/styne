import numpy as np
import pytest

from styne.statistics.covariance import (
    DenseCovarianceMatrix,
    DiagonalCovarianceMatrix,
    IIDCovarianceMatrix,
    PrecisionCovarianceMatrix,
    SemidefiniteCovarianceMatrix,
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


@pytest.mark.parametrize("covariance", [
    IIDCovarianceMatrix(2, 1.5),
    DiagonalCovarianceMatrix([1.5, 0.6]),
    DenseCovarianceMatrix(np.array([[2.0, 0.3], [0.3, 1.1]])),
    SemidefiniteCovarianceMatrix(np.array([[1.0, 1.0], [1.0, 1.0]])),
    PrecisionCovarianceMatrix(np.array([[2.0, 0.3], [0.3, 1.1]])),
])
def test_with_scaling_keeps_the_class_and_shares_the_factorisation(covariance):
    # A metric gamma B is built for every localisation precision gamma, so
    # rescaling must not factorise B again.
    vector = np.array([1.0, -2.0])

    scaled = covariance.with_scaling(2.5)

    assert type(scaled) is type(covariance)
    assert scaled.scaling == 2.5 and covariance.scaling == 1.0
    np.testing.assert_allclose(scaled.apply(vector), 2.5 * covariance.apply(vector))
    for name, value in vars(covariance).items():
        if name != "_scaling":
            assert vars(scaled)[name] is value
    with pytest.raises(ValueError, match="positive"):
        covariance.with_scaling(0.0)


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


@pytest.mark.parametrize("covarianceType", [
    DenseCovarianceMatrix, SemidefiniteCovarianceMatrix, PrecisionCovarianceMatrix,
])
def test_covariance_isolates_mutable_constructor_inputs(covarianceType):
    matrix = np.array([[2.0, 0.3], [0.3, 1.1]])
    scaling = np.array(1.5)
    covariance = covarianceType(matrix, scaling)
    expected = covariance.apply(np.ones(2))

    matrix[:] = 0.0
    scaling[...] = 10.0

    np.testing.assert_allclose(covariance.apply(np.ones(2)), expected)


def test_precision_factor_getter_does_not_expose_mutable_internal_factor():
    covariance = PrecisionCovarianceMatrix(np.array([[2.0, 0.3], [0.3, 1.1]]))
    original = covariance.precision.copy()

    covariance.precisionFactor[:] = 0.0

    np.testing.assert_allclose(covariance.precision, original)


def test_rescaling_dense_covariance_performs_no_factorisations(monkeypatch):
    covariance = DenseCovarianceMatrix(np.eye(32))

    def unexpected_factorisation(matrix):
        pytest.fail("Rescaling an existing covariance must reuse its factor.")

    monkeypatch.setattr(covariance._backend, "cholesky", unexpected_factorisation)
    for scaling in (0.1, 1.0, 5.0):
        scaled = covariance.with_scaling(scaling)
        np.testing.assert_allclose(scaled.to_cholesky(), np.sqrt(scaling) * np.eye(32))


def test_precision_from_dense_operator_reuses_validated_factor(monkeypatch):
    operator = DenseCovarianceMatrix(np.array([[2.0, 0.3], [0.3, 1.1]]), 2.0)

    def unexpected_factorisation(matrix):
        pytest.fail("An existing dense operator already owns a validated factor.")

    monkeypatch.setattr(operator._backend, "cholesky", unexpected_factorisation)
    covariance = PrecisionCovarianceMatrix(operator, scaling=3.0)

    assert covariance._precisionFactor is operator._cholFactor
    assert covariance.scaling == 3.0
    np.testing.assert_allclose(
        covariance.apply(np.ones(2)), 3.0 * operator.apply_inverse(np.ones(2))
    )
