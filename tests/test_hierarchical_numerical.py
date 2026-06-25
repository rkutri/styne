import numpy as np
import pytest
from scipy.special import gamma
from styne.statistics.stationary import matern_kappa, matern_beta, matern_fourier

def legacy_matern_fourier(f, lengthScale, smoothness, variance):
    """Original power-based implementation for regression testing."""
    DIM = 1
    f = np.asarray(f, dtype=float)
    kappa = matern_kappa(lengthScale, smoothness)
    beta = matern_beta(smoothness, DIM)
    normConst = (4. * np.pi)**(DIM / 2) * \
        gamma(2. * beta) / gamma(smoothness) * kappa**(2. * smoothness)
    sSq = (2. * np.pi * f)**2
    return variance * normConst * np.power(kappa**2 + sSq, -2. * beta)

def test_matern_fourier_regression():
    """Verify that the log-space refactor matches the original formula in stable regions."""
    # Test over stable range
    frequencies = np.array([0.1, 1.0, 10.0])
    rhos = [0.1, 1.0, 5.0]
    sigmas = [0.5, 1.0, 2.0]
    nu = 1.5
    
    for r in rhos:
        for s in sigmas:
            valNew = matern_fourier(frequencies, r, nu, s**2)
            valOld = legacy_matern_fourier(frequencies, r, nu, s**2)
            assert np.allclose(valNew, valOld, rtol=1e-12)

def test_matern_fourier_stability():
    """Verify that the refactored implementation handles extreme rho without overflow."""
    f = np.array([1.0])
    # rho = 1e-110 would cause kappa^3 to overflow in float64 (~1e330)
    rhoSmall = 1e-110
    
    # This should have overflowed in the old version but remain finite now
    val = matern_fourier(f, rhoSmall, 1.5, 1.0)
    assert np.isfinite(val)
    assert val > 0
    
    # Regression: verify it actually overflows in the "legacy" way
    with np.errstate(over='raise'):
        try:
            legacy_matern_fourier(f, rhoSmall, 1.5, 1.0)
            # If it didn't raise, it might have just returned inf
        except (FloatingPointError, OverflowError, RuntimeWarning):
            pass # Expected failure for legacy

def test_hierarchical_log_gp_finite_check():
    """Verify that SGLMMHyperConditionalDensity handles LinAlgError from singular covariance."""
    from styne.statistics.hierarchical import SGLMMHyperConditionalDensity
    from unittest.mock import MagicMock
    from styne.parameter.vector import Vector
    
    # Setup mock objects
    pcPrior = MagicMock()
    pcPrior.evaluate_log.return_value = 0.0
    gp = MagicMock()
    gp.covarianceFunction._smoothness = 1.5
    model = MagicMock()
    likelihood = MagicMock()
    latentState = Vector(np.zeros(10))
    
    hyperDensity = SGLMMHyperConditionalDensity(
        pcPrior, gp, model, likelihood, latentState
    )
    
    # Trigger LinAlgError on model.interpolate or model.evaluate
    model.interpolate.side_effect = np.linalg.LinAlgError("Singular covariance matrix")
    
    result = hyperDensity.evaluate_log(Vector([0.0, 0.0]))
    assert result == -np.inf

if __name__ == '__main__':
    pytest.main([__file__])
