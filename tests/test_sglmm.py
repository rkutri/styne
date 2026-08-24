import numpy as np

from numpy.random import default_rng
from scipy.optimize import approx_fprime

from styne.gp.gaussianprocess import GaussianProcess
from styne.parameter.vector import Vector
from styne.parameter.block import BlockParameter
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.gaussian import Gaussian
from styne.statistics.data import Data
from styne.statistics.likelihood import RegressionLikelihood
from styne.statistics.response import GaussianResponse
from styne.statistics.bayes import UnnormalisedPosterior
from styne.statistics.stationary import MaternCovariance2D
from styne.model.sglmm import SGLMM, SGLMMPredictor
from styne.utility.grid import Grid

def _compute_jacobian(model: SGLMM) -> np.ndarray:
    """Construct the full Jacobian matrix using directional derivatives."""
    latentDim = model._gp.parameter.dimension
    nParam = model.pDim
    columns = []

    if model._features is not None:
        p = model._features.shape[1]
        for i in range(latentDim):
            e_zeta = np.zeros(latentDim)
            e_zeta[i] = 1.0
            e_beta = np.zeros(p)
            param = BlockParameter([Vector(e_zeta), Vector(e_beta)])
            columns.append(model.directional_derivative(param))

        for j in range(p):
            e_zeta = np.zeros(latentDim)
            e_beta = np.zeros(p)
            e_beta[j] = 1.0
            param = BlockParameter([Vector(e_zeta), Vector(e_beta)])
            columns.append(model.directional_derivative(param))
    else:
        for i in range(nParam):
            e = np.zeros(nParam)
            e[i] = 1.0
            columns.append(model.directional_derivative(Vector(e)))

    return np.column_stack(columns)



# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_sglmm_setup(resolution=3, alpha=1.0, ell=0.3, nu=1.5,
                      margVar=1.0, noiseVar=0.05, nObs=15, seed=42):
    rng = default_rng(seed)
    covFcn = MaternCovariance2D(ell, nu, margVar)
    gp = GaussianProcess.dna(covFcn, q=resolution, d=2, alpha=alpha)
    noiseCov = IIDCovarianceMatrix(1, noiseVar)
    noise = Gaussian(noiseCov)
    noiseModel = GaussianResponse(noiseCov)
    obsSites = Grid(rng.uniform(0., alpha, (nObs, 2)))
    model = SGLMM(gp, obsSites)
    prior = gp.measure
    J = _compute_jacobian(model)
    zetaTrue = prior.generate_realisation()
    uTrue = J @ zetaTrue.coordinate
    y = noiseModel.simulate(uTrue, rng=rng).coordinate
    data = Data(1, obsSites.to_array())
    data.measurement = y[:, None]
    likelihood = RegressionLikelihood(data, model, noiseModel)
    posterior = UnnormalisedPosterior(prior, likelihood)
    return dict(
        gp=gp, model=model, noiseModel=noiseModel, prior=prior, J=J,
        zetaTrue=zetaTrue, y=y, data=data, likelihood=likelihood,
        posterior=posterior, obsSites=obsSites, alpha=alpha, noiseVar=noiseVar,
    )


def _make_sglmm_fixed_effects_setup(p=2, resolution=3, alpha=1.0, ell=0.3,
                                    nu=1.5, margVar=1.0, noiseVar=0.05,
                                    nObs=15, seed=42):
    rng = default_rng(seed)
    covFcn = MaternCovariance2D(ell, nu, margVar)
    gp = GaussianProcess.dna(covFcn, q=resolution, d=2, alpha=alpha)
    noiseCov = IIDCovarianceMatrix(1, noiseVar)
    noise = Gaussian(noiseCov)
    noiseModel = GaussianResponse(noiseCov)
    obsSites = Grid(rng.uniform(0., alpha, (nObs, 2)))
    X = rng.standard_normal((nObs, p))
    model = SGLMM(gp, obsSites, features=X)
    prior = gp.measure
    J = _compute_jacobian(model)
    latentDim = gp.parameter.dimension
    zetaTrue = prior.generate_realisation()
    betaTrue = rng.standard_normal(p)
    uTrue = J[:, :latentDim] @ zetaTrue.coordinate + X @ betaTrue
    y = noiseModel.simulate(uTrue, rng=rng).coordinate
    data = Data(1, obsSites.to_array())
    data.measurement = y[:, None]
    likelihood = RegressionLikelihood(data, model, noiseModel)
    paramTrue = BlockParameter([zetaTrue, Vector(betaTrue)])
    return dict(
        gp=gp, model=model, noiseModel=noiseModel, prior=prior, J=J,
        zetaTrue=zetaTrue, betaTrue=betaTrue, paramTrue=paramTrue,
        X=X, y=y, data=data, likelihood=likelihood,
        obsSites=obsSites, alpha=alpha, noiseVar=noiseVar,
        latentDim=latentDim, p=p,
    )


# ---------------------------------------------------------------------------
# SGLMM (no fixed effects)
# ---------------------------------------------------------------------------

class TestSGLMMForwardMap:

    def setup_method(self):
        self.s = _make_sglmm_setup()

    def test_jacobian_times_zeta_matches_model_evaluation(self):
        s = self.s
        evaluation = s['model'](s['zetaTrue'])
        err = np.max(
            np.abs(s['J'] @ s['zetaTrue'].coordinate
                   - evaluation.ravel())
        )
        assert err < 1e-12, f"Forward map error: {err:.2e}"


class TestSGLMMPosteriorGradient:

    def setup_method(self):
        self.s = _make_sglmm_setup()

    def test_gradient_matches_finite_differences(self):
        s = self.s

        def logpost_scalar(x):
            return s['posterior'].evaluate_log(Vector(x))

        grad = s['posterior'].evaluate_log_gradient(s['zetaTrue'])
        fd = approx_fprime(s['zetaTrue'].coordinate.copy(), logpost_scalar, 1e-6)
        err = np.max(np.abs(grad - fd))
        assert err < 1e-4, f"Gradient error: {err:.2e}"


class TestSGLMMPosteriorAccuracy:

    def setup_method(self):
        self.s = _make_sglmm_setup()

    def _build_prediction(self):
        s = self.s
        alpha = s['alpha']
        noiseVar = s['noiseVar']
        J = s['J']
        y = s['y']
        latentMargVar = s['gp'].measure.covariance.marginalVariance

        noisePrec = 1. / noiseVar
        precMatrix = (np.diag(1. / latentMargVar)
                      + noisePrec * (J.T @ J))
        rhs = noisePrec * J.T @ y
        zetaPost = np.linalg.solve(precMatrix, rhs)
        sigmaPost = np.linalg.inv(precMatrix)

        nPred1d = 5
        pCoords = np.linspace(0.05, alpha - 0.05, nPred1d)
        predSites = Grid(np.array([[px, py] for px in pCoords for py in pCoords]))

        predModel = SGLMM(s['gp'], predSites)
        Jp = _compute_jacobian(predModel)

        muDNA = Jp @ zetaPost
        stdDNA = np.sqrt(np.diag(Jp @ sigmaPost @ Jp.T))

        KObs = J @ np.diag(latentMargVar) @ J.T
        KNoisy = KObs + noiseVar * np.eye(len(y))
        alpha_ref = np.linalg.solve(KNoisy, y)
        KPredObs = Jp @ np.diag(latentMargVar) @ J.T
        muKriging = KPredObs @ alpha_ref

        KPred = Jp @ np.diag(latentMargVar) @ Jp.T
        stdPrior = np.sqrt(np.diag(KPred))

        return muDNA, stdDNA, muKriging, stdPrior

    def test_dna_posterior_mean_matches_kriging(self):
        muDNA, _, muKriging, _ = self._build_prediction()
        err = np.max(np.abs(muDNA - muKriging))
        assert err < 0.05, f"Posterior mean vs kriging: {err:.4f}"

    def test_posterior_std_smaller_than_prior_std(self):
        _, stdDNA, _, stdPrior = self._build_prediction()
        assert np.all(stdDNA < stdPrior), (
            "Posterior std not everywhere smaller than prior std"
        )


# ---------------------------------------------------------------------------
# SGLMM with fixed effects
# ---------------------------------------------------------------------------

class TestSGLMMFixedEffectsStructure:

    def setup_method(self):
        self.s = _make_sglmm_fixed_effects_setup(p=2)

    def test_ptype_is_block_parameter(self):
        assert self.s['model'].pType is BlockParameter

    def test_pdim_equals_latent_plus_fixed(self):
        s = self.s
        assert s['model'].pDim == s['latentDim'] + s['p']

    def test_jacobian_shape(self):
        s = self.s
        assert s['J'].shape == (15, s['latentDim'] + s['p'])

    def test_jacobian_fixed_effect_block_equals_design_matrix(self):
        s = self.s
        np.testing.assert_array_equal(s['J'][:, s['latentDim']:], s['X'])

    def test_jacobian_latent_block_nonzero(self):
        s = self.s
        latentBlock = s['J'][:, :s['latentDim']]
        assert not np.all(latentBlock == 0.)


class TestSGLMMFixedEffectsForwardMap:

    def setup_method(self):
        self.s = _make_sglmm_fixed_effects_setup(p=2)

    def test_jacobian_times_full_param_matches_model_evaluation(self):
        s = self.s
        evaluation = s['model'](s['paramTrue'])
        fullCoord = np.hstack([s['zetaTrue'].coordinate, s['betaTrue']])
        err = np.max(
            np.abs(s['J'] @ fullCoord - evaluation.ravel())
        )
        assert err < 1e-12, f"Forward map error: {err:.2e}"


class TestSGLMMFixedEffectsLikelihoodGradient:

    def setup_method(self):
        self.s = _make_sglmm_fixed_effects_setup(p=2)

    def test_gradient_matches_finite_differences(self):
        s = self.s
        latentDim = s['latentDim']

        def loglik_scalar(x):
            param = BlockParameter([Vector(x[:latentDim]), Vector(x[latentDim:])])
            return s['likelihood'].evaluate_log(param)

        grad = s['likelihood'].evaluate_log_gradient(s['paramTrue'])
        x0 = np.hstack([s['zetaTrue'].coordinate, s['betaTrue']])
        fd = approx_fprime(x0, loglik_scalar, 1e-6)
        err = np.max(np.abs(grad - fd))
        assert err < 5e-4, f"Gradient error: {err:.2e}"
