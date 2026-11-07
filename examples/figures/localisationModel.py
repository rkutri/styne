"""Shared model and quadrature helpers for the DART localisation figures.

Target f and surrogate g on R^2 are exposed as DensityInterface instances
whose evaluate_log returns the negative potential, the convention
'styne.mcmc.localised' expects: LocalisedSurrogateDensity then evaluates to
-(theta*g(y) + gamma/2*||y-x||^2) directly.
"""
import numpy as np

from styne.statistics.interface import DensityInterface
from styne.parameter.vector import Vector
from styne.mcmc.localised import LocalisedSurrogateDensity


# Frozen. Do not retune.
RIDGE_A = 1.60
RIDGE_B = 0.20
RIDGE_C = 0.55

WAVE_1 = dict(amplitude=0.55, kappa=1.4, phi=0.4, psi=0.0)
WAVE_2 = dict(amplitude=0.62, kappa=2.2, phi=-1.1, psi=0.8)
WAVE_3 = dict(amplitude=0.36, kappa=3.1, phi=1.9, psi=0.3)

# Rotates both potentials' inputs, turning the ridge portrait.
ROTATION_ALPHA = np.pi / 2.0

# Tilts the surrogate only, displacing its mode from the target's without
# touching grad^2 g.
BIAS_VECTOR = np.array([0.30, -0.20])


def _ridge_terms(points):
    x1 = points[..., 0]
    x2 = points[..., 1]

    r = RIDGE_B * (x1 ** 2 - RIDGE_A ** 2)
    rPrime = 2.0 * RIDGE_B * x1
    rDoublePrime = 2.0 * RIDGE_B
    u = x2 - r

    value = x1 ** 2 / (2.0 * RIDGE_A ** 2) + u ** 2 / (2.0 * RIDGE_C ** 2)

    dx1 = x1 / RIDGE_A ** 2 - u * rPrime / RIDGE_C ** 2
    dx2 = u / RIDGE_C ** 2
    gradient = np.stack([dx1, dx2], axis=-1)

    hxx = 1.0 / RIDGE_A ** 2 + rPrime ** 2 / RIDGE_C ** 2 \
        - u * rDoublePrime / RIDGE_C ** 2
    hxy = -rPrime / RIDGE_C ** 2
    hyy = np.full_like(hxx, 1.0 / RIDGE_C ** 2)
    hessian = np.stack([
        np.stack([hxx, hxy], axis=-1),
        np.stack([hxy, hyy], axis=-1),
    ], axis=-2)

    return value, gradient, hessian


def _wave_terms(points, wave):
    x1 = points[..., 0]
    x2 = points[..., 1]

    amplitude, kappa, phi, psi = (
        wave["amplitude"], wave["kappa"], wave["phi"], wave["psi"]
    )
    cosPhi, sinPhi = np.cos(phi), np.sin(phi)
    phase = kappa * (cosPhi * x1 + sinPhi * x2) + psi
    cosPhase = np.cos(phase)
    sinPhase = np.sin(phase)

    value = amplitude * cosPhase

    dx1 = -amplitude * kappa * cosPhi * sinPhase
    dx2 = -amplitude * kappa * sinPhi * sinPhase
    gradient = np.stack([dx1, dx2], axis=-1)

    common = -amplitude * kappa ** 2 * cosPhase
    hxx = common * cosPhi ** 2
    hxy = common * cosPhi * sinPhi
    hyy = common * sinPhi ** 2
    hessian = np.stack([
        np.stack([hxx, hxy], axis=-1),
        np.stack([hxy, hyy], axis=-1),
    ], axis=-2)

    return value, gradient, hessian


def _sum_terms(points, waves):
    value, gradient, hessian = _ridge_terms(points)
    for wave in waves:
        v, g, h = _wave_terms(points, wave)
        value = value + v
        gradient = gradient + g
        hessian = hessian + h
    return value, gradient, hessian


def _target_terms_model_frame(points):
    return _sum_terms(points, [WAVE_1, WAVE_2, WAVE_3])


def _surrogate_terms_model_frame(points):
    return _sum_terms(points, [WAVE_1])


# y = R(alpha) x in column-vector convention, so a model-frame quantity is
# evaluated at figure-frame y via x = R(alpha)^T y.
def _rotation_matrix(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s], [s, c]])


def _rotate_vector(vectors, matrix):
    return np.einsum("ij,...j->...i", matrix, vectors)


def _rotate_matrix(matrices, matrix):
    return np.einsum("ij,...jk,lk->...il", matrix, matrices, matrix)


def _to_model_frame(points):
    rotation = _rotation_matrix(ROTATION_ALPHA)
    return _rotate_vector(points, rotation.T)


def _rotate_terms_to_figure_frame(value, gradient, hessian):
    rotation = _rotation_matrix(ROTATION_ALPHA)
    return value, _rotate_vector(gradient, rotation), _rotate_matrix(hessian, rotation)


def target_potential(points):
    """Potential f(y) = ridge + wave1 + wave2 + wave3, rotated by ROTATION_ALPHA."""
    points = np.asarray(points, dtype=float)
    modelPoints = _to_model_frame(points)
    value, _, _ = _target_terms_model_frame(modelPoints)
    return value


def target_gradient(points):
    """Analytic gradient of target_potential."""
    points = np.asarray(points, dtype=float)
    modelPoints = _to_model_frame(points)
    value, gradient, hessian = _target_terms_model_frame(modelPoints)
    _, gradient, _ = _rotate_terms_to_figure_frame(value, gradient, hessian)
    return gradient


def surrogate_potential(points):
    """Potential g(y) = ridge + wave1 + bias.y, rotated, with a linear tilt."""
    points = np.asarray(points, dtype=float)
    modelPoints = _to_model_frame(points)
    value, _, _ = _surrogate_terms_model_frame(modelPoints)
    return value + points @ BIAS_VECTOR


def surrogate_gradient(points):
    """Analytic gradient of surrogate_potential."""
    points = np.asarray(points, dtype=float)
    modelPoints = _to_model_frame(points)
    value, gradient, hessian = _surrogate_terms_model_frame(modelPoints)
    _, gradient, _ = _rotate_terms_to_figure_frame(value, gradient, hessian)
    return gradient + BIAS_VECTOR


def surrogate_hessian(points):
    """Analytic Hessian of surrogate_potential (the tilt has zero Hessian)."""
    points = np.asarray(points, dtype=float)
    modelPoints = _to_model_frame(points)
    value, gradient, hessian = _surrogate_terms_model_frame(modelPoints)
    _, _, hessian = _rotate_terms_to_figure_frame(value, gradient, hessian)
    return hessian


class PotentialDensity(DensityInterface):
    """Log-density -potential(y) for an arbitrary R^2 potential.

    'potential' and 'gradient' both take points of shape (..., 2).
    """

    def __init__(self, potential, gradient):
        self._potential = potential
        self._gradient = gradient

    @property
    def domainType(self):
        return Vector

    @property
    def domainDimension(self) -> int:
        return 2

    def evaluate_log(self, state):
        return -self._potential(state.coordinate)

    def evaluate_log_gradient(self, state):
        return -self._gradient(state.coordinate)


target_density = PotentialDensity(target_potential, target_gradient)
surrogate_density = PotentialDensity(surrogate_potential, surrogate_gradient)


def build_grid(bounds, resolution):
    """Flattened regular grid and its per-cell area, over (xMin, xMax, yMin,
    yMax) at 'resolution' points per axis."""
    xMin, xMax, yMin, yMax = bounds
    xGrid = np.linspace(xMin, xMax, resolution)
    yGrid = np.linspace(yMin, yMax, resolution)
    cellArea = (xGrid[1] - xGrid[0]) * (yGrid[1] - yGrid[0])
    gridX, gridY = np.meshgrid(xGrid, yGrid, indexing="ij")
    points = np.stack([gridX.ravel(), gridY.ravel()], axis=-1)
    return points, cellArea


def grid_mesh(bounds, resolution):
    """As 'build_grid', but also returning the two axis vectors a contour
    plot needs to reshape a flattened field."""
    xMin, xMax, yMin, yMax = bounds
    xGrid = np.linspace(xMin, xMax, resolution)
    yGrid = np.linspace(yMin, yMax, resolution)
    points, cellArea = build_grid(bounds, resolution)
    return xGrid, yGrid, points, cellArea


def localised_proposal(theta, gamma, location):
    """LocalisedSurrogateDensity for Pi_x, centred at `location`."""
    density = LocalisedSurrogateDensity(
        regularisation=gamma, tempering=theta, surrogateDensity=surrogate_density
    )
    return density.with_location(Vector(np.asarray(location, dtype=float)))


def quadrature_moments(density, points, cellArea):
    """Mean, covariance, normalised weights and unnormalised mass, by
    quadrature on 'points'. The mass is over the given domain only, so it
    doubles as a truncation check against a wider grid."""
    logValues = np.asarray(density.evaluate_log(Vector(points)))
    shift = logValues.max()
    weights = np.exp(logValues - shift) * cellArea
    mass = weights.sum() * np.exp(shift)
    probabilities = weights / weights.sum()

    mean = probabilities @ points
    diff = points - mean
    covariance = (diff * probabilities[:, None]).T @ diff

    return mean, covariance, probabilities, mass


def isotropic_sd(covariance):
    """sqrt(trace(covariance)/2), the isotropic-equivalent standard deviation."""
    return np.sqrt(np.trace(covariance) / 2.0)


def hpd_threshold(probabilities, level):
    """Largest per-cell probability whose super-level set still captures
    'level' of the mass."""
    order = np.argsort(probabilities)[::-1]
    cumulative = np.cumsum(probabilities[order])
    cutoffIndex = np.searchsorted(cumulative, level)
    cutoffIndex = min(cutoffIndex, len(probabilities) - 1)
    return probabilities[order[cutoffIndex]]


def hpd_log_density_level(logValues, cellArea, level):
    """The same threshold as 'hpd_threshold', converted back to a level of
    'logValues' so it can contour a density sampled on a different grid."""
    shift = logValues.max()
    weights = np.exp(logValues - shift) * cellArea
    probabilities = weights / weights.sum()
    thresholdProb = hpd_threshold(probabilities, level)
    return np.log(thresholdProb * weights.sum() / cellArea) + shift
