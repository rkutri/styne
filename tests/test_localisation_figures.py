"""Tests for the model behind the localisation figure.

The figure module is not part of the styne package, so it is imported via a
path insertion. Its plotted settings are imported rather than restated:
retuning a theta or gamma cannot then leave these tests guarding values no
figure draws.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "figures"))
import localisationModel as lm  # noqa: E402
import dartProposalFamily as figureA  # noqa: E402


def finite_difference_gradient(fn, points, eps=1e-6):
    gradient = np.zeros_like(points)
    for i in range(points.shape[-1]):
        step = np.zeros(points.shape[-1])
        step[i] = eps
        gradient[..., i] = (fn(points + step) - fn(points - step)) / (2 * eps)
    return gradient


def finite_difference_hessian(gradient_fn, point, eps=1e-6):
    n = point.shape[-1]
    hessian = np.zeros((n, n))
    for i in range(n):
        step = np.zeros(n)
        step[i] = eps
        hessian[:, i] = (
            gradient_fn(point + step) - gradient_fn(point - step)
        ) / (2 * eps)
    return hessian


def figure_settings():
    """Every (theta, gamma, state) pair the figure actually draws."""
    return [(theta, gamma, figureA.CURRENT_STATE)
            for theta, gamma in figureA.SETTINGS]


@pytest.fixture
def random_points():
    rng = np.random.default_rng(0)
    return rng.uniform(-2.5, 2.5, size=(20, 2))


def test_target_gradient_matches_finite_differences(random_points):
    analytic = lm.target_gradient(random_points)
    numeric = finite_difference_gradient(lm.target_potential, random_points)
    assert np.max(np.abs(analytic - numeric)) < 1e-7


def test_surrogate_gradient_matches_finite_differences(random_points):
    analytic = lm.surrogate_gradient(random_points)
    numeric = finite_difference_gradient(lm.surrogate_potential, random_points)
    assert np.max(np.abs(analytic - numeric)) < 1e-7


def test_surrogate_hessian_matches_finite_differences(random_points):
    for point in random_points[:8]:
        analytic = lm.surrogate_hessian(point[None, :])[0]
        numeric = finite_difference_hessian(
            lambda p: lm.surrogate_gradient(p[None, :])[0], point)
        assert np.max(np.abs(analytic - numeric)) < 1e-7


def test_integration_by_parts_identity():
    """mean(Pi_x) - x == -(theta/gamma) * E_{Pi_x}[grad g(u)]."""
    rng = np.random.default_rng(1)
    points, cellArea = lm.build_grid((-12, 12, -12, 12), 700)

    for _ in range(6):
        theta = rng.uniform(0.05, 1.0)
        gamma = rng.uniform(0.5, 12.0)
        x = rng.uniform(-2.0, 2.0, size=2)

        density = lm.localised_proposal(theta, gamma, x)
        mean, _, probabilities, _ = lm.quadrature_moments(density, points, cellArea)
        expectedGradient = probabilities @ lm.surrogate_gradient(points)

        lhs = mean - x
        rhs = -(theta / gamma) * expectedGradient
        assert np.max(np.abs(lhs - rhs)) < 1e-10


@pytest.mark.parametrize("theta,gamma,state", figure_settings())
def test_plotted_settings_give_a_usable_proposal(theta, gamma, state):
    """Finite moments, a positive-definite covariance, and mass the
    quadrature domain actually contains."""
    points, cellArea = lm.build_grid(
        figureA.QUADRATURE_BOUNDS, figureA.QUADRATURE_RESOLUTION)
    density = lm.localised_proposal(theta, gamma, state)
    mean, covariance, _, mass = lm.quadrature_moments(density, points, cellArea)

    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(covariance))
    assert np.linalg.eigvalsh(covariance)[0] > 0.0

    widePoints, wideCellArea = lm.build_grid((-45.0, 45.0, -45.0, 45.0), 500)
    _, _, _, wideMass = lm.quadrature_moments(density, widePoints, wideCellArea)
    assert mass / wideMass > 1.0 - 1e-6


def identity_grid(gamma):
    """Quadrature grid wide enough to test the identity at 'gamma'.

    The identity needs the boundary term of the integral of grad(Pi_x) to
    vanish, so truncation leaves a residual that dividing by gamma amplifies:
    at gamma = 0.025 a (-20, 20) domain leaves 3e-7 against 1e-12 at
    (-30, 30). Plotted means and covariances are insensitive to this, not
    being tail-weighted the way E[grad g] is.
    """
    halfWidth = max(20.0, 6.0 / np.sqrt(gamma))
    return lm.build_grid(
        (-halfWidth, halfWidth, -halfWidth, halfWidth),
        int(2.0 * halfWidth / 0.1))


@pytest.mark.parametrize("theta,gamma,state", figure_settings())
def test_plotted_settings_satisfy_the_identity(theta, gamma, state):
    """The drift identity holds at the settings actually drawn."""
    points, cellArea = identity_grid(gamma)
    density = lm.localised_proposal(theta, gamma, state)
    mean, _, probabilities, _ = lm.quadrature_moments(density, points, cellArea)
    expectedGradient = probabilities @ lm.surrogate_gradient(points)

    lhs = mean - np.asarray(state, dtype=float)
    rhs = -(theta / gamma) * expectedGradient
    assert np.max(np.abs(lhs - rhs)) < 1e-10


def test_newton_metric_is_positive_definite():
    """The Newton column inverts grad^2 g at the state, which is indefinite
    over much of this model's domain."""
    hessian = lm.surrogate_hessian(figureA.CURRENT_STATE[None, :])[0]
    assert np.linalg.eigvalsh(hessian)[0] > 0.0
