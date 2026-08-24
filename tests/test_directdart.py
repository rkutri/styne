import numpy as np

from styne.mcmc.method.dartdirect import DirectDART, DirectDARTProposal
from styne.mcmc.transition import TransitionData
from styne.parameter.vector import Vector
from styne.statistics.gaussian import Gaussian, GaussianDensity
from styne.statistics.covariance import DenseCovarianceMatrix
from styne.mcmc.diagnostics import DummyDiagnostics


def test_directdart_proposal():
    dim = 2
    tempering = 0.5
    gamma = 0.1

    x_hat = Vector(np.array([1.0, -1.0]))
    # surrogate covariance:
    cov_arr = np.array([[2.0, 0.5], [0.5, 1.0]])
    surrogate_cov = DenseCovarianceMatrix(cov_arr)
    surrogate = Gaussian(surrogate_cov, mean=x_hat)

    A = np.linalg.inv(cov_arr)
    P = tempering * A + gamma * np.eye(dim)
    P_inv = np.linalg.inv(P)
    proposal_cov = DenseCovarianceMatrix(P_inv)

    proposal_method = DirectDARTProposal(tempering, gamma, surrogate, proposal_cov)

    x = np.array([0.5, 0.5])
    state = Vector(x)
    rng = np.random.default_rng(0)
    transition, _ = proposal_method.propose(state, rng)
    
    # Hand-calculate expected values
    bx_expected = tempering * (A @ x_hat.coordinate) + gamma * x
    mux_expected = P_inv @ bx_expected

    assert np.allclose(transition.auxiliary['bx'], bx_expected)
    assert np.allclose(transition.auxiliary['mux'], mux_expected)


def test_directdart_mh_ratio():
    dim = 2
    tempering = 0.5
    gamma = 0.1

    x_hat = Vector(np.array([1.0, -1.0]))
    cov_arr = np.array([[2.0, 0.5], [0.5, 1.0]])
    surrogate_cov = DenseCovarianceMatrix(cov_arr)
    surrogate = Gaussian(surrogate_cov, mean=x_hat)

    A = np.linalg.inv(cov_arr)
    P = tempering * A + gamma * np.eye(dim)
    P_inv = np.linalg.inv(P)
    proposal_cov = DenseCovarianceMatrix(P_inv)

    # Use the surrogate itself as the target for simplicity
    target = surrogate.density

    sampler = DirectDART(target, tempering, gamma, surrogate, proposal_cov, DummyDiagnostics())

    x = np.array([0.5, 0.5])
    z = np.array([0.2, 0.8])
    state = Vector(x)
    prop = Vector(z)

    # Mock the transition
    bx = tempering * (A @ x_hat.coordinate) + gamma * x
    mux = P_inv @ bx
    transition = TransitionData(
        current=sampler.evaluate_state(state),
        proposed=sampler.evaluate_state(prop),
        auxiliary={'bx': bx, 'mux': mux},
    )

    # Call _log_mh_ratio
    log_mh_ratio = sampler._log_mh_ratio(transition)

    # Hand-calculate expected log MH ratio
    # 1. logDiffTarget
    log_target_x = target.evaluate_log(state)
    log_target_z = target.evaluate_log(prop)
    log_diff_target = log_target_z - log_target_x

    # 2. Quad diff
    diff_z = z - x_hat.coordinate
    diff_x = x - x_hat.coordinate
    quad_z = diff_z @ A @ diff_z
    quad_x = diff_x @ A @ diff_x
    quad_surrogate = 0.5 * tempering * (quad_z - quad_x)

    # 3. Norm diff
    bz = tempering * (A @ x_hat.coordinate) + gamma * z
    muz = P_inv @ bz
    norm_diff = 0.5 * (mux.dot(bx) - muz.dot(bz)) - 0.5 * gamma * (x.dot(x) - z.dot(z))

    expected_log_mh_ratio = log_diff_target + quad_surrogate + norm_diff

    assert np.isclose(log_mh_ratio, expected_log_mh_ratio)


def test_directdart_execution():
    dim = 2
    tempering = 0.5
    gamma = 0.1

    x_hat = Vector(np.array([0.0, 0.0]))
    cov_arr = np.eye(dim)
    surrogate_cov = DenseCovarianceMatrix(cov_arr)
    surrogate = Gaussian(surrogate_cov, mean=x_hat)

    A = np.eye(dim)
    P = tempering * A + gamma * np.eye(dim)
    P_inv = np.linalg.inv(P)
    proposal_cov = DenseCovarianceMatrix(P_inv)

    target = surrogate.density

    sampler = DirectDART(target, tempering, gamma, surrogate, proposal_cov, DummyDiagnostics())

    init_state = Vector(np.array([1.0, 1.0]))
    sampler.run(nSteps=10, initialState=init_state)

    assert len(sampler.chain.trajectory) == 11 # init + 10 steps

if __name__ == '__main__':
    test_directdart_proposal()
    test_directdart_mh_ratio()
    test_directdart_execution()
    print("All tests passed.")
