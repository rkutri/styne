import numpy as np
from scipy.optimize import minimize, approx_fprime
from typing import Tuple

from styne.parameter.parameter import Parameter
from styne.statistics.interface import DensityInterface, DifferentiableDensity, TwiceDifferentiableDensity


def determine_map(density: DensityInterface, initial_guess: Parameter, method: str = 'L-BFGS-B') -> Tuple[Parameter, np.ndarray]:
    """
    Compute the Maximum A Posteriori (MAP) estimate and the Hessian at the mode.

    Parameters
    ----------
    density : DensityInterface
        The target density (must satisfy DifferentiableDensity).
    initial_guess : Parameter
        The initial guess for the optimization.
    method : str
        The optimization method to use (default 'L-BFGS-B').

    Returns
    -------
    tuple[Parameter, np.ndarray]
        A tuple containing the MAP estimate (as a Parameter) and the exact or 
        finite-difference Hessian matrix at the mode.
    """
    if not isinstance(density, DifferentiableDensity):
        raise TypeError("Density must satisfy DifferentiableDensity protocol to compute MAP.")

    def objective(x: np.ndarray) -> float:
        p = initial_guess.clone()
        p.coordinate = x
        return -density.evaluate_log(p)

    def jacobian(x: np.ndarray) -> np.ndarray:
        p = initial_guess.clone()
        p.coordinate = x
        return -density.evaluate_log_gradient(p)

    res = minimize(
        fun=objective,
        x0=initial_guess.coordinate,
        jac=jacobian,
        method=method
    )

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    x_map = initial_guess.clone()
    x_map.coordinate = res.x

    if isinstance(density, TwiceDifferentiableDensity):
        hessian = -density.evaluate_log_hessian(x_map)
    else:
        # Fallback to finite differences on the gradient
        hessian = approx_fprime(res.x, jacobian, epsilon=1e-8)
        # Symmetrize to reduce numerical errors
        hessian = 0.5 * (hessian + hessian.T)

    return x_map, hessian
