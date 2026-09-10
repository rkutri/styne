"""
Tests for the IS, geometric bridge, and cumulant ratio estimators.

Test setup
----------
Surrogate density: pi_k = N(v; mu, (1/gamma) * I) with mu != x and mu != z.
Localised density: Pi_x proportional to pi_k(v)^theta * N(v; x, (1/gamma)*I).

The product of two Gaussians is Gaussian, so Pi_x and Pi_z have known
normalising constants. With theta=1 the analytical ground truth is

    log(N_z / N_x) = -gamma / 4 * (||mu - z||^2 - ||mu - x||^2)

and samples come from

    Pi_x = N((mu + x)/2, (1/(2*gamma)) * I)
    Pi_z = N((mu + z)/2, (1/(2*gamma)) * I).

Concrete parameter values used in all accuracy tests
-----------------------------------------------------
  x     = [0, ..., 0]       (state at origin)
  z     = [1, 0, ..., 0]    (proposal, delta = 1 along first axis)
  mu    = [3, 0, ..., 0]    (surrogate mean, different from both x and z)
  gamma = 1.0,  theta = 1.0

True log(N_z/N_x) = -1/4 * (||[3]-[1]||^2 - ||[3]-[0]||^2)
                  = -1/4 * (4 - 9) = 5/4 = 1.25.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from styne.parameter.vector import Vector
from styne.mcmc.method.ratio import RatioEstimator


SEED = 20240607

class State:
    def __init__(self, coordinate):
        self.coordinate = coordinate

class StubMeasure:
    """Minimal surrogate-measure stand-in for the one-sided estimators.
    Deliberately has NO location and NO generate_realisation, so any code
    path that is secretly two-sided will raise instead of passing silently."""
    class _Chain:
        def __init__(self, trajectory):
            self.trajectory = trajectory
    class _Density:
        spectralWeights = None
    def __init__(self, trajectory, gamma):
        self.chain = StubMeasure._Chain(trajectory)
        self.density = StubMeasure._Density()
        self.regularisation = gamma


def trajectory_with_terminal(samples):
    samples = np.asarray(samples)
    return np.concatenate([samples, samples[-1:]], axis=0)


@pytest.mark.parametrize('estimatorType', ('is', 'bridge', 'cumulant'))
def test_coincident_states_are_an_explicit_exact_identity(estimatorType):
    estimator = RatioEstimator(
        StubMeasure([], 1.0), 5, 2, estimatorType
    )
    state = Vector(np.array([1.5]))

    result = estimator.log_ratio_estimate(
        state, state, trajectory=np.empty((0, 1))
    )

    assert result.shape == ()
    assert result == 0.0


@pytest.mark.parametrize('estimatorType', ('is', 'bridge', 'cumulant'))
def test_distinct_states_reject_empty_state_trajectory(estimatorType):
    estimator = RatioEstimator(
        StubMeasure([], 1.0), 0, 1, estimatorType
    )

    with pytest.raises(ValueError, match="state trajectory"):
        estimator.log_ratio_estimate(
            Vector(np.array([0.0])),
            Vector(np.array([1.0])),
            trajectory=np.empty((0, 1)),
        )


def test_is_accepts_one_retained_state_but_cumulant_rejects_it():
    trajectory = trajectory_with_terminal([[0.25]])
    state = Vector(np.array([0.0]))
    proposal = Vector(np.array([1.0]))

    estimate = RatioEstimator(
        StubMeasure(trajectory, 1.0), 0, 1, 'is'
    ).log_ratio_estimate(state, proposal, trajectory=trajectory)

    assert np.isfinite(estimate)
    with pytest.raises(ValueError, match="at least 2 retained"):
        RatioEstimator(
            StubMeasure(trajectory, 1.0), 0, 1, 'cumulant'
        ).log_ratio_estimate(state, proposal, trajectory=trajectory)


def test_bridge_accepts_one_retained_state_in_each_trajectory():
    stateTrajectory = trajectory_with_terminal([[0.25]])
    proposalTrajectory = trajectory_with_terminal([[0.75]])
    estimator = RatioEstimator(
        StubMeasure(stateTrajectory, 1.0), 0, 1, 'bridge'
    )

    estimate = estimator.log_ratio_estimate(
        Vector(np.array([0.0])),
        Vector(np.array([1.0])),
        proposalTrajectory=proposalTrajectory,
    )

    assert np.isfinite(estimate)


@pytest.mark.parametrize('burnin', (3, 4))
def test_burnin_at_or_beyond_trajectory_length_is_rejected(burnin):
    trajectory = trajectory_with_terminal([[0.0], [0.5], [1.0]])
    estimator = RatioEstimator(
        StubMeasure(trajectory, 1.0), burnin, 1, 'is'
    )

    with pytest.raises(ValueError, match="got 0"):
        estimator.log_ratio_estimate(
            Vector(np.array([0.0])),
            Vector(np.array([1.0])),
            trajectory=trajectory,
        )


def test_thinning_uses_only_the_retained_states():
    trajectory = trajectory_with_terminal(
        [[-1.0], [-0.5], [0.0], [0.5], [1.0], [1.5]]
    )
    state = Vector(np.array([0.0]))
    proposal = Vector(np.array([2.0]))
    retained = trajectory[:-1][1::2]
    weights = 1.0 * (retained[:, 0] - 1.0) * -2.0
    expected = np.log(np.mean(np.exp(-weights)))

    result = RatioEstimator(
        StubMeasure(trajectory, 1.0), 1, 2, 'is'
    ).log_ratio_estimate(state, proposal, trajectory=trajectory)

    np.testing.assert_allclose(result, expected)


def test_bridge_validates_each_trajectory_independently():
    sufficient = trajectory_with_terminal([[0.0], [0.5]])
    empty = np.empty((0, 1))
    estimator = RatioEstimator(
        StubMeasure(sufficient, 1.0), 0, 1, 'bridge'
    )
    state = Vector(np.array([0.0]))
    proposal = Vector(np.array([1.0]))

    with pytest.raises(ValueError, match="state trajectory"):
        estimator.log_ratio_estimate(
            state,
            proposal,
            trajectory=empty,
            proposalTrajectory=sufficient,
        )
    with pytest.raises(ValueError, match="proposal trajectory"):
        estimator.log_ratio_estimate(
            state,
            proposal,
            trajectory=sufficient,
            proposalTrajectory=empty,
        )
    with pytest.raises(ValueError, match="explicit proposal trajectory"):
        estimator.log_ratio_estimate(
            state, proposal, trajectory=sufficient
        )


def test_deterministic_estimator_algebra():
    state = Vector(np.array([0.0]))
    proposal = Vector(np.array([2.0]))
    stateSamples = np.array([[0.0], [1.0], [2.0]])
    proposalSamples = np.array([[1.0], [2.0], [3.0]])
    stateTrajectory = trajectory_with_terminal(stateSamples)
    proposalTrajectory = trajectory_with_terminal(proposalSamples)
    measure = StubMeasure(stateTrajectory, 1.0)
    stateWeights = -2.0 * (stateSamples[:, 0] - 1.0)
    proposalWeights = -2.0 * (proposalSamples[:, 0] - 1.0)

    isEstimate = RatioEstimator(
        measure, 0, 1, 'is'
    ).log_ratio_estimate(state, proposal)
    cumulantEstimate = RatioEstimator(
        measure, 0, 1, 'cumulant'
    ).log_ratio_estimate(state, proposal)
    bridgeEstimate = RatioEstimator(
        measure, 0, 1, 'bridge'
    ).log_ratio_estimate(
        state, proposal, proposalTrajectory=proposalTrajectory
    )

    np.testing.assert_allclose(
        isEstimate, np.log(np.mean(np.exp(-stateWeights)))
    )
    np.testing.assert_allclose(
        cumulantEstimate,
        -np.mean(stateWeights) + 0.5 * np.var(stateWeights, ddof=1),
    )
    np.testing.assert_allclose(
        bridgeEstimate,
        np.log(np.mean(np.exp(-0.5 * stateWeights)))
        - np.log(np.mean(np.exp(0.5 * proposalWeights))),
    )


def test_cumulant_ratio_compiles_with_jax():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    estimator = RatioEstimator(
        StubMeasure(jnp.zeros((3, 1)), 1.0), 0, 1, 'cumulant'
    )

    compiled = jax.jit(lambda state, proposal, trajectory:
        estimator.log_ratio_estimate(
            Vector(state), Vector(proposal), trajectory=trajectory
        )
    )
    result = compiled(
        jnp.array([0.0]),
        jnp.array([1.0]),
        jnp.array([[0.0], [0.5], [1.0]]),
    )

    assert isinstance(result, jax.Array)
    assert result.shape == ()


@pytest.mark.parametrize(
    ('estimatorType', 'minimum', 'invalidLength', 'invalidBurnin'),
    (
        ('is', 1, 1, 1),
        ('bridge', 1, 1, 1),
        ('cumulant', 2, 1, 0),
    ),
)
def test_configured_trajectory_requirements(
        estimatorType, minimum, invalidLength, invalidBurnin):
    measure = StubMeasure([], 1.0)
    measure.subchainLength = minimum

    estimator = RatioEstimator(measure, 0, 1, estimatorType)

    assert estimator.minimumSamples == minimum
    measure.subchainLength = invalidLength
    with pytest.raises(ValueError, match="configuration retains"):
        RatioEstimator(measure, invalidBurnin, 1, estimatorType)


def test_cumulant_ratio_preserves_jax_scalar():
    jnp = pytest.importorskip("jax.numpy")

    trajectory = jnp.array([[0.], [1.], [1.]])
    estimate = RatioEstimator(
        StubMeasure(trajectory, 1.), 0, 1, 'cumulant'
    ).log_ratio_estimate(Vector(jnp.array([0.])), Vector(jnp.array([1.])))

    assert isinstance(estimate, type(jnp.array(0.)))


def test_cumulant_ratio_preserves_pytorch_scalar():
    torch = pytest.importorskip("torch")

    trajectory = torch.tensor([[0.], [1.], [1.]])
    estimate = RatioEstimator(
        StubMeasure(trajectory, 1.), 0, 1, 'cumulant'
    ).log_ratio_estimate(Vector(torch.tensor([0.])), Vector(torch.tensor([1.])))

    assert isinstance(estimate, torch.Tensor)


# ---------------------------------------------------------------------------
# Analytical reference
# ---------------------------------------------------------------------------

def analytical_log_ratio(x, z, mu, gamma, theta):
    """Exact log(N_z / N_x) for the Gaussian-product localised density."""
    norm_diff = np.sum((mu - z) ** 2) - np.sum((mu - x) ** 2)
    return -gamma * theta / (2.0 * (theta + 1.0)) * norm_diff


# ---------------------------------------------------------------------------
# Sample helpers
# ---------------------------------------------------------------------------

def draw_pi(loc, mu, gamma, theta, n, rng):
    """Draw n samples from Pi_loc = N((theta*mu + loc)/(theta+1), 1/((theta+1)*gamma)*I)."""
    d = len(loc)
    mean = (theta * mu + loc) / (theta + 1.0)
    std = 1.0 / np.sqrt((theta + 1.0) * gamma)
    return rng.normal(0.0, std, size=(n, d)) + mean


# ---------------------------------------------------------------------------
# Mock surrogate factory
# ---------------------------------------------------------------------------

def make_surrogate(gamma, samplesX=None, samplesZ=None):
    """
    Minimal surrogate mock.

    samplesX populates chain.trajectory before log_ratio_estimate is called.
    samplesZ populates chain.trajectory via a generate_realisation side effect
    (bridge estimator only).
    """
    measure = MagicMock()
    measure.regularisation = gamma
    measure.density.regularisation = gamma
    measure.density.spectralWeights = None
    if samplesX is not None:
        # trajectory has n+1 elements; [:-1] trimming yields exactly samplesX
        measure.chain.trajectory = list(samplesX) + [samplesX[-1]]
    if samplesZ is not None:
        def realise():
            measure.chain.trajectory = list(samplesZ) + [samplesZ[-1]]
        measure.generate_realisation.side_effect = realise
    return measure


def test_bridge_uses_explicit_proposal_trajectory_without_mutating_measure():
    samplesX = np.array([[0.], [1.]])
    samplesZ = np.array([[1.], [2.]])
    measure = make_surrogate(1., samplesX=samplesX)
    estimator = RatioEstimator(measure, 0, 1, 'bridge')

    result = estimator.log_ratio_estimate(
        Vector(np.array([0.])),
        Vector(np.array([1.])),
        list(samplesX) + [samplesX[-1]],
        list(samplesZ) + [samplesZ[-1]],
    )

    assert np.isfinite(result)
    measure.generate_realisation.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: IS accuracy
# ---------------------------------------------------------------------------

class TestISCorrectionAccuracy:
    """IS estimate against known analytical value (non-zero ground truth)."""

    GAMMA = 1.0
    THETA = 1.0
    DELTA = 1.0
    MU_VAL = 3.0
    TOL = 0.15

    def _run(self, d, n, seed=SEED):
        rng = np.random.default_rng(seed)
        x = np.zeros(d)
        z = np.zeros(d)
        z[0] = self.DELTA
        mu = np.zeros(d)
        mu[0] = self.MU_VAL

        samplesX = draw_pi(x, mu, self.GAMMA, self.THETA, n, rng)
        surrogate = make_surrogate(self.GAMMA, samplesX=samplesX)

        est = RatioEstimator(surrogate, burnin=0, thinning=1, type='is')
        result = est.log_ratio_estimate(Vector(x), Vector(z))
        truth = analytical_log_ratio(x, z, mu, self.GAMMA, self.THETA)
        return result, truth

    def test_accuracy_d5(self):
        est, truth = self._run(d=5, n=2000)
        assert abs(est - truth) < self.TOL, (
            f"IS: |est {est:.4f} - truth {truth:.4f}| >= {self.TOL}")

    def test_accuracy_d10(self):
        est, truth = self._run(d=10, n=4000)
        assert abs(est - truth) < self.TOL, (
            f"IS (d=10): |est {est:.4f} - truth {truth:.4f}| >= {self.TOL}")


# ---------------------------------------------------------------------------
# Tests: geometric bridge accuracy
# ---------------------------------------------------------------------------

class TestGeometricBridgeCorrectionAccuracy:
    """Geometric bridge estimate against known analytical value."""

    GAMMA = 1.0
    THETA = 1.0
    DELTA = 1.0
    MU_VAL = 3.0
    TOL = 0.15

    def _run(self, d, n, seed=SEED):
        rng = np.random.default_rng(seed)
        x = np.zeros(d)
        z = np.zeros(d)
        z[0] = self.DELTA
        mu = np.zeros(d)
        mu[0] = self.MU_VAL

        samplesX = draw_pi(x, mu, self.GAMMA, self.THETA, n, rng)
        samplesZ = draw_pi(z, mu, self.GAMMA, self.THETA, n, rng)
        surrogate = make_surrogate(
            self.GAMMA, samplesX=samplesX, samplesZ=samplesZ)

        est = RatioEstimator(surrogate, burnin=0,
                             thinning=1, type='bridge')
        proposalTrajectory = np.concatenate(
            [samplesZ, samplesZ[-1:]], axis=0
        )
        result = est.log_ratio_estimate(
            Vector(x), Vector(z), proposalTrajectory=proposalTrajectory
        )
        truth = analytical_log_ratio(x, z, mu, self.GAMMA, self.THETA)
        return result, truth

    def test_accuracy_d5(self):
        est, truth = self._run(d=5, n=1000)
        assert abs(est - truth) < self.TOL, (
            f"GeomBridge: |est {est:.4f} - truth {truth:.4f}| >= {self.TOL}")

    def test_accuracy_d10(self):
        est, truth = self._run(d=10, n=2000)
        assert abs(est - truth) < self.TOL, (
            f"GeomBridge (d=10): |est {est:.4f} - truth {truth:.4f}| >= {self.TOL}")


# ---------------------------------------------------------------------------
# Tests: geometric bridge variance reduction vs IS
# ---------------------------------------------------------------------------

class TestGeometricBridgeVarianceReduction:
    """Geometric bridge should have lower estimator variance than IS."""

    GAMMA = 1.0
    THETA = 1.0
    DELTA = 2.0
    MU_VAL = 3.0

    def _variance_test(self, d, n, nRep, seed=SEED):
        rng = np.random.default_rng(seed)
        x = np.zeros(d)
        z = np.zeros(d)
        z[0] = self.DELTA
        mu = np.zeros(d)
        mu[0] = self.MU_VAL

        stateParam = Vector(x)
        propParam = Vector(z)

        isEsts, bridgeEsts = [], []

        for _ in range(nRep):
            samplesX = draw_pi(x, mu, self.GAMMA, self.THETA, n, rng)
            samplesZ = draw_pi(z, mu, self.GAMMA, self.THETA, n, rng)

            surrogateIS = make_surrogate(self.GAMMA, samplesX=samplesX)
            estIS = RatioEstimator(surrogateIS, burnin=0,
                                   thinning=1, type='is')
            isEsts.append(estIS.log_ratio_estimate(stateParam, propParam))

            surrogateBr = make_surrogate(
                self.GAMMA, samplesX=samplesX, samplesZ=samplesZ)
            estBr = RatioEstimator(surrogateBr, burnin=0,
                                   thinning=1, type='bridge')
            proposalTrajectory = np.concatenate(
                [samplesZ, samplesZ[-1:]], axis=0
            )
            bridgeEsts.append(estBr.log_ratio_estimate(
                stateParam,
                propParam,
                proposalTrajectory=proposalTrajectory,
            ))

        return float(np.std(isEsts)), float(np.std(bridgeEsts))

    def test_lower_variance_d5(self):
        isStd, bridgeStd = self._variance_test(d=5, n=200, nRep=50)
        assert bridgeStd < isStd, (
            f"Bridge std {bridgeStd:.4f} should be < IS std {isStd:.4f} (d=5)")

    def test_lower_variance_d10(self):
        isStd, bridgeStd = self._variance_test(d=10, n=500, nRep=30)
        assert bridgeStd < isStd, (
            f"Bridge std {bridgeStd:.4f} should be < IS std {isStd:.4f} (d=10)")


# ---------------------------------------------------------------------------
# Tests: cumulant accuracy
# ---------------------------------------------------------------------------

class TestCumulantCorrectionAccuracy:
    """Cumulant estimate against known analytical value and IS baseline."""

    GAMMA = 1.0
    THETA = 1.0
    DELTA = 1.0
    MU_VAL = 3.0
    TOL = 0.10

    def _run(self, d, n, seed=SEED):
        rng = np.random.default_rng(seed)
        x = np.zeros(d)
        z = np.zeros(d)
        z[0] = self.DELTA
        mu = np.zeros(d)
        mu[0] = self.MU_VAL

        samplesX = draw_pi(x, mu, self.GAMMA, self.THETA, n, rng)
        surrogate = make_surrogate(self.GAMMA, samplesX=samplesX)

        est = RatioEstimator(surrogate, burnin=0, thinning=1, type='cumulant')
        result = est.log_ratio_estimate(Vector(x), Vector(z))
        truth = analytical_log_ratio(x, z, mu, self.GAMMA, self.THETA)
        return result, truth

    def test_accuracy_d5(self):
        est, truth = self._run(d=5, n=2000)
        assert abs(est - truth) < self.TOL, (
            f"Cumulant: |est {est:.4f} - truth {truth:.4f}| >= {self.TOL}")

    def test_accuracy_d10(self):
        est, truth = self._run(d=10, n=4000)
        assert abs(est - truth) < self.TOL, (
            f"Cumulant (d=10): |est {est:.4f} - truth {truth:.4f}| >= {self.TOL}")

    def test_constant_weights_matches_is(self):
        # With constant weights, cumulant and IS must agree to 1e-12.
        # w_i is constant -> var(w) = 0, so -mean(w) == logsumexp(-w) - log(len(w)).
        d = 3
        x = np.zeros(d)
        z = np.zeros(d)
        z[0] = self.DELTA
        
        # Using mock where we control weights by setting a constant array.
        # w = gamma * ((samples - mid) @ diff)
        # If all samples in samplesX are identical, then all elements of w are identical.
        samplesX = np.ones((10, d)) * 2.0
        surrogate = make_surrogate(self.GAMMA, samplesX=samplesX)
        
        est_is = RatioEstimator(surrogate, burnin=0, thinning=1, type='is')
        est_cum = RatioEstimator(surrogate, burnin=0, thinning=1, type='cumulant')
        
        res_is = est_is.log_ratio_estimate(Vector(x), Vector(z))
        res_cum = est_cum.log_ratio_estimate(Vector(x), Vector(z))
        
        assert abs(res_is - res_cum) < 1e-12

    def test_singleton_is_rejected(self):
        d = 3
        rng = np.random.default_rng(SEED)
        x = np.zeros(d)
        z = np.zeros(d)
        z[0] = self.DELTA
        mu = np.zeros(d)
        mu[0] = self.MU_VAL
        
        samplesX = draw_pi(x, mu, self.GAMMA, self.THETA, 1, rng)
        surrogate = make_surrogate(self.GAMMA, samplesX=samplesX)
        
        est = RatioEstimator(surrogate, burnin=0, thinning=1, type='cumulant')
        with pytest.raises(ValueError, match="at least 2 retained"):
            est.log_ratio_estimate(Vector(x), Vector(z))

    def test_variance_regime_check(self):
        # Regime check, NOT an invariant. Cumulant and IS share their
        # leading-order variance, so which is smaller is regime dependent.
        # We only assert the two are comparable at moderate tau.
        master = np.random.default_rng(SEED)
        R, m, tau = 400, 30, 1.0
        cum, isv = [], []
        for _ in range(R):
            v = master.normal(0.0, tau, size=m)            # symmetric: isolates variance
            traj = list(v.reshape(-1, 1)) + [np.array([0.0])]
            meas = StubMeasure(traj, gamma=1.0)
            x, z = State(np.array([0.5])), State(np.array([-0.5]))
            cum.append(RatioEstimator(meas, 0, 1, 'cumulant').log_ratio_estimate(x, z))
            isv.append(RatioEstimator(meas, 0, 1, 'is').log_ratio_estimate(x, z))
        vc, vi = float(np.var(cum, ddof=1)), float(np.var(isv, ddof=1))
        assert vc <= 1.3 * vi

    def test_cumulant_bias_is_positive_and_cubic(self):
        rng = np.random.default_rng(SEED)
        m = 100_000
        y = rng.gamma(shape=4.0, size=m)
        c = (y - y.mean()) / y.std()          # standardised, right-skewed
        skew = float((c ** 3).mean())
        assert skew > 0.5                      # sanity: the sample really is skewed

        traj = list(c.reshape(-1, 1)) + [np.array([0.0])]
        meas = StubMeasure(traj, gamma=1.0)

        def bias(delta):
            x = State(np.array([0.5 * delta]))
            z = State(np.array([-0.5 * delta]))
            cum = RatioEstimator(meas, 0, 1, 'cumulant').log_ratio_estimate(x, z)
            isv = RatioEstimator(meas, 0, 1, 'is').log_ratio_estimate(x, z)
            return cum - isv

        b = [bias(d) for d in (0.4, 0.2, 0.1)]

        assert all(bi > 0 for bi in b)          # sign matches positive skew
        assert b[0] > b[1] > b[2]               # magnitude shrinks with the step
        r0, r1 = b[0] / b[1], b[1] / b[2]
        assert 6.0 <= r0 <= 9.0                 # ~2**3 = 8, contaminated at the large step
        assert 7.0 <= r1 <= 9.0                 # cleaner cubic regime
        assert r1 >= r0 - 0.5                    # ratio approaches 8 as the step shrinks
