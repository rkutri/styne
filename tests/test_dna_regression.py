import os
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

SNAP = os.path.join(os.path.dirname(__file__), "dna_regression_snapshot.npz")
Q, D, ALPHA = 16, 2, 1.0
ELL = 0.2


def outputs():
    """Deterministic outputs for a fixed square configuration. No RNG."""
    cov = StubFourierCov(ELL)
    gp = GaussianProcess.dna(cov, Q, D, ALPHA)
    weights = np.asarray(gp.expansion.spectralWeights)

    expansion = gp.expansion
    coeff = np.linspace(-1.0, 1.0, weights.size)   # fixed, not random
    native = expansion.evaluate_native(coeff)

    return {"weights": weights, "native": np.asarray(native)}


def make_snapshot():
    np.savez(SNAP, **outputs())


def test_scalar_path_unchanged():
    assert os.path.exists(SNAP), f"Missing committed DNA snapshot: {SNAP}"
    ref = np.load(SNAP)
    out = outputs()
    for key, value in out.items():
        np.testing.assert_allclose(
            value, ref[key], rtol=0, atol=1e-12,
            err_msg=f"scalar regression broken in '{key}'")


if __name__ == "__main__":
    import sys
    if "--snapshot" in sys.argv:
        make_snapshot()
        print(f"wrote {SNAP}")
