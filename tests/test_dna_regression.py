import os
import numpy as np
import pytest

from styne.gp.dna import DNAFourierEngine
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


def _outputs():
    """Deterministic outputs for a fixed square configuration. No RNG."""
    eng = DNAFourierEngine(Q, D, ALPHA)
    cov = StubFourierCov(ELL)
    eng.build_covariance(cov)                      # sets eng.spectralWeights
    weights = np.asarray(eng.spectralWeights)

    real = eng.build_realisation()
    real.spectralWeights = eng.spectralWeights
    coeff = np.linspace(-1.0, 1.0, weights.size)   # fixed, not random
    real.coefficient = coeff
    native = real.evaluate_native()

    mult = eng.compute_log_length_multiplier(nu=1.5, lengthScale=ELL)
    return {"weights": weights, "native": np.asarray(native), "mult": np.asarray(mult)}


def make_snapshot():
    np.savez(SNAP, **_outputs())


def test_scalar_path_unchanged():
    if not os.path.exists(SNAP):
        make_snapshot()
    ref = np.load(SNAP)
    out = _outputs()
    for key in ref.files:
        np.testing.assert_allclose(
            out[key], ref[key], rtol=0, atol=1e-12,
            err_msg=f"scalar regression broken in '{key}'")


if __name__ == "__main__":
    import sys
    if "--snapshot" in sys.argv:
        make_snapshot()
        print(f"wrote {SNAP}")
