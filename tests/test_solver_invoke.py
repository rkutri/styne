import pytest
import numpy as np

from scipy.integrate import solve_ivp
from tests.testSetup import (
    LotkaVolterraSolver, LotkaVolterraParameter, EvaluationStatus
)

# Sample configuration
config = {
    'T': 10,
    'alpha': 0.1,
    'gamma': 0.1,
    'nData': 1,
    'dataDim': 2,
    'solver': 'LSODA',
    'rtol': 1e-8
}

# initial conditions
design = np.array([np.array([1., 0.8])])

# Sample parameters
coefficients = np.array([0.2, 0.3])
parameter = LotkaVolterraParameter.from_coefficient(coefficients)


def reference_lotka_volterra_solver(alpha, beta, gamma, delta, y0, t_span):

    def lotka_volterra(t, y):

        prey, predator = y
        dprey_dt = alpha * prey - beta * prey * predator
        dpredator_dt = delta * prey * predator - gamma * predator

        return [dprey_dt, dpredator_dt]

    result = solve_ivp(lotka_volterra, t_span, y0, method='LSODA')

    if result.status != 0:
        raise RuntimeError(f"Reference solver failed: {result.message}")

    return result.y[:, -1].reshape(1, 2)


def reference_lotka_volterra_full_solution(
        alpha, beta, gamma, delta, y0, t_span):

    def lotka_volterra(t, y):

        prey, predator = y
        dprey_dt = alpha * prey - beta * prey * predator
        dpredator_dt = delta * prey * predator - gamma * predator

        return [dprey_dt, dpredator_dt]

    result = solve_ivp(lotka_volterra, t_span, y0, method='LSODA')

    if result.status != 0:
        raise RuntimeError(f"Reference solver failed: {result.message}")

    return result.t, result.y


def test_invoke():

    solver = LotkaVolterraSolver(design, config)
    evaluation = solver(parameter)

    # Check if solution succeeds
    assert solver.status == EvaluationStatus.SUCCESS

    # Check the shape of the evaluation array
    assert evaluation.shape == (config['nData'], config['dataDim'])
    assert evaluation is not None

    for res in evaluation:
        assert res[0] >= 0
        assert res[1] >= 0

    # Reference implementation
    alpha = config['alpha']
    gamma = config['gamma']
    beta = np.exp(coefficients[0])
    delta = np.exp(coefficients[1])
    t_span = (0, config['T'])
    y0 = design[0]

    ref_result = reference_lotka_volterra_solver(
        alpha, beta, gamma, delta, y0, t_span)

    # Compare results
    np.testing.assert_allclose(evaluation, ref_result,
                               rtol=1e-3, atol=1e-6)


def test_full_solution():

    solver = LotkaVolterraSolver(design, config)
    y0 = np.array([1., 0.5])

    # Get the full solution from the solver
    t_solver, y_solver = solver.full_solution(parameter, y0)

    # Reference implementation
    alpha = config['alpha']
    gamma = config['gamma']
    beta = np.exp(coefficients[0])
    delta = np.exp(coefficients[1])
    t_span = (0, config['T'])

    t_ref, y_ref = reference_lotka_volterra_full_solution(
        alpha, beta, gamma, delta, y0, t_span)

    # Compare solutions
    np.testing.assert_allclose(y_solver, y_ref, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    pytest.main()
