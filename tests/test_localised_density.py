import pytest
import numpy as np

from styne.mcmc.localised import LocalisedSurrogateDensity
from styne.mcmc.localised import LocalisedSurrogateTransitionMeasure
from styne.mcmc.diagnostics import DummyDiagnostics
from styne.mcmc.method.pcn import PreconditionedCrankNicolson
from styne.mcmc.method.ratio import log_dot_product_weights
from styne.statistics.covariance import (
    IIDCovarianceMatrix, DiagonalCovarianceMatrix
)
from styne.statistics.gaussian import Gaussian
from styne.statistics.interface import DensityInterface
from styne.statistics.radonnikodym import RadonNikodym
from styne.parameter.vector import Vector


class ConstantDensity(DensityInterface):

    def __init__(self, dimension):
        self._dimension = dimension

    @property
    def domainType(self):
        return Vector

    @property
    def domainDimension(self):
        return self._dimension

    def evaluate_log(self, parameter):
        return np.sum(parameter.coordinate * 0.0, axis=-1)


def gaussian_surrogate(dim):
    """Minimal RadonNikodym surrogate: N(0, I) prior, trivial likelihood."""
    cov = IIDCovarianceMatrix(dim, 1.0)
    prior = Gaussian(cov, Vector(np.zeros(dim)))
    # derivative is a uniform (zero-gradient) density; use prior.density itself
    # as a stand-in — the test only cares about the regularisation term.
    return RadonNikodym(prior, prior.density)


def test_localised_trajectory_retargets_centre_dependent_reference():
    density = LocalisedSurrogateDensity(4.0, 1.0, ConstantDensity(1))
    sampler = PreconditionedCrankNicolson(
        density, 1.0, DummyDiagnostics()
    )
    measure = LocalisedSurrogateTransitionMeasure(sampler, nChain=1)
    centre = Vector([3.0])
    expectedReference = density.with_location(centre).reference
    expected, _ = expectedReference.sample(np.random.default_rng(7))

    proposal, trajectory, _ = measure.transition_trajectory(
        centre, np.random.default_rng(7)
    )

    np.testing.assert_allclose(proposal.coordinate, expected.coordinate)
    np.testing.assert_allclose(trajectory[1], expected.coordinate)
    np.testing.assert_array_equal(sampler.target.location.coordinate, [0.0])
    np.testing.assert_array_equal(
        sampler.proposal.referenceMeasure.mean.coordinate, [0.0]
    )


def test_localised_rn_target_keeps_fixed_gaussian_reference():
    reference = Gaussian(IIDCovarianceMatrix(1, 2.0), Vector([1.0]))
    surrogate = RadonNikodym(reference, ConstantDensity(1))
    density = LocalisedSurrogateDensity(4.0, 1.0, surrogate)
    sampler = PreconditionedCrankNicolson(
        density, 1.0, DummyDiagnostics()
    )

    sampler.target = density.with_location(Vector([3.0]))

    assert sampler.proposal.referenceMeasure is reference
    np.testing.assert_array_equal(
        sampler.proposal.referenceMeasure.mean.coordinate, [1.0]
    )


# ──────────────────────────────────────────────────────────────────
# Regression: IID behaviour (spectralWeights=None)
# ──────────────────────────────────────────────────────────────────

def test_iid_regularisation_penalty():
    """IID: log-ratio between location and off-location equals -gamma/2 * ||d||^2."""
    dim = 8
    gamma = 0.5
    surr = gaussian_surrogate(dim)
    density = LocalisedSurrogateDensity(gamma, 1.0, surr)  # no spectralWeights

    rng = np.random.default_rng(42)
    location = Vector(rng.standard_normal(dim))
    density.location = location

    d = rng.standard_normal(dim)
    pt = Vector(location.coordinate + d)

    # The regularisation Gaussian is N(location, (1/gamma)*I)
    # log density at pt - log density at location = -gamma/2 * ||d||^2
    expected_diff = -gamma / 2.0 * np.dot(d, d)
    actual_diff = (
        density._regGaussian.density.evaluate_log(pt)
        - density._regGaussian.density.evaluate_log(location)
    )
    assert np.isclose(actual_diff, expected_diff, rtol=1e-10)


# ──────────────────────────────────────────────────────────────────
# Unit: weighted regularisation (spectralWeights provided)
# ──────────────────────────────────────────────────────────────────

def test_weighted_regularisation_penalty():
    """Weighted: log-ratio equals -gamma/2 * ||W*d||^2."""
    dim = 8
    gamma = 0.5
    surr = gaussian_surrogate(dim)

    rng = np.random.default_rng(7)
    weights = np.abs(rng.standard_normal(dim)) + 0.1

    density = LocalisedSurrogateDensity(gamma, 1.0, surr, spectralWeights=weights)

    location = Vector(rng.standard_normal(dim))
    density.location = location

    d = rng.standard_normal(dim)
    pt = Vector(location.coordinate + d)

    # With weights W = diag(weights), reg covariance = diag(1 / (gamma * w^2))
    # log N(pt | loc, diag(1/(gamma*w^2))) - log N(loc | loc, ...) = -gamma/2 * ||W*d||^2
    expected_diff = -gamma / 2.0 * np.dot(weights * d, weights * d)
    actual_diff = (
        density._regGaussian.density.evaluate_log(pt)
        - density._regGaussian.density.evaluate_log(location)
    )
    assert np.isclose(actual_diff, expected_diff, rtol=1e-10)


def test_weighted_less_penalising_than_iid():
    """For spectral weights << 1 (high-frequency modes), weighted penalty is smaller."""
    dim = 16
    gamma = 2.0
    surr = gaussian_surrogate(dim)

    # Simulate DNA-like weights: mostly small, a few larger
    weights = np.full(dim, 0.01)
    weights[:2] = 0.5

    density_iid = LocalisedSurrogateDensity(gamma, 1.0, surr)
    density_wt = LocalisedSurrogateDensity(
        gamma, 1.0, surr, spectralWeights=weights
    )

    rng = np.random.default_rng(99)
    location = Vector(np.zeros(dim))
    density_iid.location = location
    density_wt.location = location

    # A large displacement that is purely in high-frequency (small-weight) directions
    displacement = np.zeros(dim)
    displacement[2:] = 1.0  # modes with weight 0.01
    pt = Vector(displacement)

    log_iid = density_iid._regGaussian.density.evaluate_log(pt)
    log_wt = density_wt._regGaussian.density.evaluate_log(pt)

    # Weighted should penalise this displacement much less
    assert log_wt > log_iid


def test_spectralweights_property():
    """spectralWeights property returns the stored weights (or None)."""
    dim = 4
    surr = gaussian_surrogate(dim)
    weights = np.array([1.0, 0.5, 0.1, 0.01])

    density_iid = LocalisedSurrogateDensity(1.0, 1.0, surr)
    density_wt = LocalisedSurrogateDensity(
        1.0, 1.0, surr, spectralWeights=weights
    )

    assert density_iid.spectralWeights is None
    assert np.array_equal(density_wt.spectralWeights, weights)


# ──────────────────────────────────────────────────────────────────
# Unit: sync_weights
# ──────────────────────────────────────────────────────────────────

def test_sync_weights_updates_covariance():
    """sync_weights replaces the regularisation covariance in place."""
    dim = 4
    gamma = 1.0
    surr = gaussian_surrogate(dim)

    w1 = np.array([1.0, 0.5, 0.1, 0.05])
    density = LocalisedSurrogateDensity(
        gamma, 1.0, surr, spectralWeights=w1
    )

    location = Vector(np.zeros(dim))
    density.location = location

    # Record penalty at a fixed point before sync
    d = np.ones(dim)
    pt = Vector(d)
    log_before = density._regGaussian.density.evaluate_log(pt)

    # Now sync with different weights
    w2 = np.array([0.5, 0.5, 0.5, 0.5])
    density.sync_weights(w2)

    log_after = density._regGaussian.density.evaluate_log(pt)

    # Expected after: -gamma/2 * ||w2 * d||^2 = -gamma/2 * dim * 0.25
    expected_after = -gamma / 2.0 * np.dot(w2 * d, w2 * d)
    assert np.isclose(log_after, expected_after, rtol=1e-10)
    # Ensure it actually changed
    assert not np.isclose(log_before, log_after)

    # Stored weights are updated
    assert np.array_equal(density.spectralWeights, w2)


def test_sync_weights_none_is_noop():
    """sync_weights(None) does not change anything."""
    dim = 4
    gamma = 1.0
    surr = gaussian_surrogate(dim)
    weights = np.array([1.0, 0.5, 0.1, 0.05])
    density = LocalisedSurrogateDensity(
        gamma, 1.0, surr, spectralWeights=weights
    )
    location = Vector(np.zeros(dim))
    density.location = location

    pt = Vector(np.ones(dim))
    log_before = density._regGaussian.density.evaluate_log(pt)
    density.sync_weights(None)
    log_after = density._regGaussian.density.evaluate_log(pt)

    assert np.isclose(log_before, log_after)
    assert np.array_equal(density.spectralWeights, weights)


# ──────────────────────────────────────────────────────────────────
# Unit: log_dot_product_weights
# ──────────────────────────────────────────────────────────────────

def test_log_dot_product_weights_iid():
    """Without spectral weights: result is gamma * (samples - mid) @ (x - z)."""
    rng = np.random.default_rng(11)
    dim = 6
    gamma = 2.0
    samples = rng.standard_normal((10, dim))
    x = rng.standard_normal(dim)
    z = rng.standard_normal(dim)

    result = log_dot_product_weights(gamma, samples, x, z)
    mid = 0.5 * (x + z)
    expected = gamma * ((samples - mid) @ (x - z))
    assert np.allclose(result, expected)


def test_log_dot_product_weights_weighted():
    """With spectral weights: diff is pre-multiplied by w^2."""
    rng = np.random.default_rng(22)
    dim = 6
    gamma = 2.0
    samples = rng.standard_normal((10, dim))
    x = rng.standard_normal(dim)
    z = rng.standard_normal(dim)
    w = np.abs(rng.standard_normal(dim)) + 0.1

    result = log_dot_product_weights(gamma, samples, x, z, spectralWeights=w)
    mid = 0.5 * (x + z)
    expected = gamma * ((samples - mid) @ (w**2 * (x - z)))
    assert np.allclose(result, expected)


def test_log_dot_product_weights_none_matches_no_arg():
    """Passing spectralWeights=None is identical to not passing it."""
    rng = np.random.default_rng(33)
    dim = 5
    gamma = 1.5
    samples = rng.standard_normal((8, dim))
    x = rng.standard_normal(dim)
    z = rng.standard_normal(dim)

    result_default = log_dot_product_weights(gamma, samples, x, z)
    result_none = log_dot_product_weights(gamma, samples, x, z, spectralWeights=None)
    assert np.allclose(result_default, result_none)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
