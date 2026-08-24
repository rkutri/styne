import numpy as np
from styne.gp.gaussianprocess import GaussianProcess
from styne.statistics.stationary import matern_fourier


class StubFourierCov:
    """
    A stub Fourier covariance class for testing DNA Gaussian Processes.
    """

    def __init__(self, lengthScale: float):
        self.lengthScale = lengthScale

    def evaluate_fourier(self, freqs: np.ndarray) -> np.ndarray:
        d = freqs.shape[1] if freqs.ndim > 1 else 1
        return matern_fourier(freqs, self.lengthScale, 1.5, 1.0, d=d)

ELL = 0.2
K = 1000          # MC draws
SEED = 12345


def _center_variance(q, alpha, d):
    gp = GaussianProcess.dna(StubFourierCov(ELL), q, d, alpha)
    expansion = gp.expansion
    n = expansion.dimension

    nG = tuple(qj + 2 for qj in (q if not np.isscalar(q) else (q,) * d))
    centre = tuple(g // 2 for g in nG)

    rng = np.random.default_rng(SEED)
    acc = 0.0
    for _ in range(K):
        coefficient = rng.standard_normal(n)
        field = np.asarray(
            expansion.evaluate_native(coefficient)
        ).reshape(nG)
        acc += field[centre]**2
    return acc / K


def test_square_variance_independent_of_alpha():
    # Matched resolution: q scales with alpha so max frequency q/(2 alpha) is fixed.
    v1 = _center_variance(20, 1.0, 2)
    v2 = _center_variance(40, 2.0, 2)
    ratio = v2 / v1
    assert 0.8 < ratio < 1.2, (
        f"interior variance scales with alpha (ratio={ratio:.3f}); "
        "marginal-variance normalisation is alpha-dependent -- STOP and report")


def test_rectangular_variance_independent_of_alpha():
    v_sq = _center_variance((30, 30), (1.5, 1.5), 2)
    v_rect = _center_variance((20, 40), (1.0, 2.0), 2)
    ratio = v_rect / v_sq
    assert 0.8 < ratio < 1.2, (
        f"rectangular interior variance off by ratio={ratio:.3f}; "
        "per-axis alpha broke the normalisation")
