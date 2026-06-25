import numpy as np
import pytest
from styne.parameter.vector import Vector
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.gaussian import Gaussian, GaussianDensity
from styne.statistics.radonnikodym import RadonNikodym
from styne.mcmc.localised import LocalisedSurrogateDensity


def _make_rn_surrogate(priorVar, likVar, dim):
    priorCov = IIDCovarianceMatrix(dim, priorVar)
    prior = Gaussian(priorCov)
    prior.mean = Vector(np.zeros(dim))
    likCov = IIDCovarianceMatrix(dim, likVar)
    likelihood = GaussianDensity(likCov, Vector(np.zeros(dim)))
    return RadonNikodym(prior, likelihood)


# ---------------------------------------------------------------------------
# Section 1 — evaluate_log equals theta*g + l2Reg in both branches
# ---------------------------------------------------------------------------

class TestEvaluateLogUnchanged:

    def test_rn_case_matches_formula(self):
        gamma, theta = 2.0, 0.6
        dim = 3
        surrogate = _make_rn_surrogate(priorVar=1.5, likVar=0.5, dim=dim)
        loc = LocalisedSurrogateDensity(gamma, theta, surrogate, temperFullDensity=False)
        x = np.array([1., -1., 0.5])
        u = np.array([0.5, 0.2, -0.3])
        loc.location = Vector(x)

        # temperFullDensity=False means prior is UNTOUCHED.
        expected = (theta * surrogate.derivative.evaluate_log(Vector(u))
                    + surrogate.reference.density.evaluate_log(Vector(u))
                    - gamma / 2. * np.dot(u - x, u - x))
        assert abs(loc.evaluate_log(Vector(u)) - expected) < 1e-10

    def test_rn_case_full_tempering(self):
        gamma, theta = 2.0, 0.6
        dim = 3
        surrogate = _make_rn_surrogate(priorVar=1.5, likVar=0.5, dim=dim)
        loc = LocalisedSurrogateDensity(gamma, theta, surrogate, temperFullDensity=True)
        x = np.array([1., -1., 0.5])
        u = np.array([0.5, 0.2, -0.3])
        loc.location = Vector(x)

        # temperFullDensity=True means BOTH prior and likelihood are tempered by theta.
        expected = (theta * surrogate.evaluate_log(Vector(u))
                    - gamma / 2. * np.dot(u - x, u - x))
        assert abs(loc.evaluate_log(Vector(u)) - expected) < 1e-10

    def test_non_rn_case_matches_formula(self):
        gamma, theta, tau2 = 1.5, 0.7, 2.0
        dim = 3
        cov = IIDCovarianceMatrix(dim, tau2)
        surrogate = GaussianDensity(cov, Vector(np.zeros(dim)))
        loc = LocalisedSurrogateDensity(gamma, theta, surrogate)
        x = np.array([0.5, -0.3, 1.0])
        u = np.array([1.0, 0.5, -0.5])
        loc.location = Vector(x)

        expected = (theta * surrogate.evaluate_log(Vector(u))
                    - gamma / 2. * np.dot(u - x, u - x))
        assert abs(loc.evaluate_log(Vector(u)) - expected) < 1e-10


# ---------------------------------------------------------------------------
# Section 2 — location only moves the regularisation, never the prior
# ---------------------------------------------------------------------------

class TestLocationOnlyMovesRegularisation:

    def test_prior_mean_unchanged_after_location_set(self):
        surrogate = _make_rn_surrogate(priorVar=1.0, likVar=1.0, dim=2)
        prior_mean_before = surrogate.reference.mean.coordinate.copy()

        loc = LocalisedSurrogateDensity(1.0, 0.8, surrogate)
        loc.location = Vector(np.array([99., -99.]))

        np.testing.assert_array_equal(
            surrogate.reference.mean.coordinate, prior_mean_before)

    def test_location_getter_returns_reg_centre(self):
        surrogate = _make_rn_surrogate(priorVar=1.0, likVar=1.0, dim=2)
        loc = LocalisedSurrogateDensity(1.0, 1.0, surrogate)
        x = np.array([3., -2.])
        loc.location = Vector(x)
        np.testing.assert_array_equal(loc.location.coordinate, x)

    def test_evaluate_log_moves_with_location(self):
        gamma, theta = 1.0, 1.0
        dim = 2
        surrogate = _make_rn_surrogate(priorVar=1.0, likVar=1.0, dim=dim)
        loc = LocalisedSurrogateDensity(gamma, theta, surrogate)
        u = Vector(np.array([1., 1.]))

        loc.location = Vector(np.zeros(dim))
        v0 = loc.evaluate_log(u)
        loc.location = Vector(np.array([5., 5.]))
        v5 = loc.evaluate_log(u)

        assert abs(v0 - v5) > 0.1


# ---------------------------------------------------------------------------
# Section 3 — reference covariance is prior's covariance scaled by 1/theta
# ---------------------------------------------------------------------------

class TestReferenceCovariance:

    def test_reference_covariance_is_scaled_copy_when_full_tempering(self):
        priorVar, theta = 2.0, 0.5
        surrogate = _make_rn_surrogate(priorVar=priorVar, likVar=1.0, dim=2)
        loc = LocalisedSurrogateDensity(1.0, theta, surrogate, temperFullDensity=True)

        priorCov = surrogate.reference.covariance
        refCov = loc.reference.covariance

        assert refCov is not priorCov
        x = np.array([1., -1.])
        np.testing.assert_allclose(
            refCov.apply(x), priorCov.apply(x) / theta, rtol=1e-10)

    def test_reference_covariance_is_untouched_when_not_full_tempering(self):
        priorVar, theta = 2.0, 0.5
        surrogate = _make_rn_surrogate(priorVar=priorVar, likVar=1.0, dim=2)
        loc = LocalisedSurrogateDensity(1.0, theta, surrogate, temperFullDensity=False)

        priorCov = surrogate.reference.covariance
        refCov = loc.reference.covariance

        assert refCov is priorCov
        x = np.array([1., -1.])
        np.testing.assert_allclose(
            refCov.apply(x), priorCov.apply(x), rtol=1e-10)

    def test_prior_covariance_not_mutated(self):
        priorVar, theta = 1.5, 0.4
        surrogate = _make_rn_surrogate(priorVar=priorVar, likVar=1.0, dim=2)
        priorScalingBefore = surrogate.reference.covariance.scaling

        LocalisedSurrogateDensity(1.0, theta, surrogate)

        assert surrogate.reference.covariance.scaling == priorScalingBefore

    def test_non_rn_reference_mean_tracks_location(self):
        gamma, theta, tau2 = 2.0, 0.5, 1.0
        dim = 2
        cov = IIDCovarianceMatrix(dim, tau2)
        surrogate = GaussianDensity(cov, Vector(np.zeros(dim)))
        loc = LocalisedSurrogateDensity(gamma, theta, surrogate)
        x = np.array([1., 2.])
        loc.location = Vector(x)
        np.testing.assert_array_equal(loc.reference.mean.coordinate, x)
