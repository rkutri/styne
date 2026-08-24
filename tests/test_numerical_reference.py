import numpy as np
from scipy.sparse.linalg import spsolve

from styne.gp import GaussianProcess
from styne.gp.dna import DNAFourierExpansion
from styne.mcmc.diagnostics import AcceptanceRateDiagnostics
from styne.mcmc.method.mala import (
    MALAProposal,
    MetropolisAdjustedLangevinAlgorithm,
)
from styne.mcmc.method.pcn import PCNProposal
from styne.mcmc.transition import TransitionData
from styne.model import LinearForwardMap, SGLMM
from styne.parameter import Vector
from styne.statistics import (
    Data,
    DenseCovarianceMatrix,
    DiagonalCovarianceMatrix,
    Gaussian,
    GaussianDensity,
    GaussianResponse,
    IIDCovarianceMatrix,
    MaternCovariance1D,
    MaternCovariance2D,
    PoissonResponse,
    RegressionLikelihood,
    SGLMMLikelihood,
    UnnormalisedPosterior,
)
from styne.utility import Grid, UniformGrid
from styne.utility.finiteelement import (
    apply_dirichlet_1d,
    p1_mass_lumped_1d,
    p1_stiffness_1d,
)
from tests.reference_oracles import (
    compute_log_length_multiplier,
    dna_adjoint_synthesis,
    dna_synthesis_matrix,
)


def test_covariance_reference_values():
    sites = np.array([0.0, 0.2, 0.7])
    covariance = MaternCovariance1D(0.35, 1.5, 1.7)

    expectedCovariance = np.array([
        [1.7, 1.2572044647078546, 0.23754329532693494],
        [1.2572044647078546, 1.7, 0.4974201457260135],
        [0.23754329532693494, 0.4974201457260135, 1.7],
    ])
    expectedSpectrum = np.array([
        1.3740936406713096,
        1.1340630389969144,
        0.20139942094159155,
    ])

    np.testing.assert_allclose(
        covariance.evaluate_covariance(sites, sites), expectedCovariance,
        rtol=0.0, atol=1e-12,
    )
    np.testing.assert_allclose(
        covariance.evaluate_fourier(np.array([0.0, 0.25, 1.0])),
        expectedSpectrum, rtol=0.0, atol=1e-12,
    )


def test_dense_covariance_operator_reference_values():
    covariance = DenseCovarianceMatrix(np.array([[2.0, 0.3], [0.3, 1.1]]))
    covariance.scaling = 1.25
    coordinate = np.array([0.4, -1.2])

    assert np.isclose(covariance.log_determinant(), 1.1929750501163947)
    np.testing.assert_allclose(
        covariance.apply(coordinate), [0.55, -1.5], rtol=0.0, atol=1e-12,
    )
    np.testing.assert_allclose(
        covariance.apply_inverse(coordinate),
        [0.3033175355450236, -0.9554502369668245],
        rtol=0.0, atol=1e-12,
    )
    assert np.isclose(covariance.quadratic_form(coordinate), 2.02)
    assert np.isclose(
        covariance.dual_quadratic_form(coordinate), 1.267867298578199,
    )
    factor = covariance.to_cholesky()
    np.testing.assert_allclose(
        factor @ factor.T,
        covariance.scaling * covariance.to_dense(),
        rtol=0.0,
        atol=1e-12,
    )


def test_diagonal_covariance_cholesky_respects_scaling():
    covariance = DiagonalCovarianceMatrix(np.array([1.5, 0.6]))
    covariance.scaling = 1.25

    factor = covariance.to_cholesky()

    np.testing.assert_allclose(
        factor @ factor.T,
        np.diag(covariance.scaling * covariance.marginalVariance),
        rtol=0.0,
        atol=1e-12,
    )


def test_direct_and_dna_gp_reference_evaluations():
    directCovariance = MaternCovariance1D(0.35, 1.5, 1.7)
    directGrid = UniformGrid(0.0, 1.0, 5)
    direct = GaussianProcess.direct(directGrid, directCovariance)
    directCoefficient = np.array([0.5, -1.0, 0.25, 0.75, -0.4])
    np.testing.assert_allclose(
        direct.evaluate(directCoefficient, directGrid),
        [
            0.6519202405202649,
            -0.5684390912084771,
            -0.35485867317297426,
            0.6203633466460737,
            0.166843691017669,
        ],
        rtol=0.0, atol=1e-12,
    )

    dna = GaussianProcess.dna(
        MaternCovariance1D(0.3, 1.5, 0.8), q=3, d=1,
    )
    dnaSites = UniformGrid(0.0, 1.0, 5)
    dnaCoefficient = np.linspace(-0.75, 0.9, dna.parameterDimension)
    np.testing.assert_allclose(
        dna.evaluate(dnaCoefficient, dnaSites),
        [
            -0.7205954985785141,
            -0.11406982021787518,
            -0.3084444638094802,
            -0.13269575353361024,
            -0.20537926387593686,
        ],
        rtol=0.0, atol=1e-12,
    )


def test_dna_transform_and_adjoint_closed_form_oracle():
    q = (2, 1)
    expansion = DNAFourierExpansion(q, d=2, alpha=(1.0, 1.5))
    coefficient = np.linspace(-0.8, 1.1, expansion.dimension)
    residual = np.linspace(-0.4, 0.7, 12)

    synthesisMatrix = dna_synthesis_matrix(q, d=2)
    np.testing.assert_allclose(
        expansion.evaluate_native(coefficient), synthesisMatrix @ coefficient,
        rtol=0.0, atol=1e-12,
    )
    np.testing.assert_allclose(
        expansion.adjoint_synthesis(residual),
        dna_adjoint_synthesis(q, 2, residual),
        rtol=0.0, atol=1e-12,
    )


def test_log_length_multiplier_closed_form_oracle():
    q = (3, 2)
    alpha = (1.0, 1.4)
    gp = GaussianProcess.dna(
        MaternCovariance2D(0.3, 1.5, 1.0), q, d=2, alpha=alpha
    )
    expected = compute_log_length_multiplier(
        q, alpha, d=2, nu=1.5, lengthScale=0.3,
    )
    np.testing.assert_allclose(
        gp.compute_log_length_multiplier(1.5, 0.3), expected,
        rtol=0.0, atol=1e-12,
    )


def test_density_and_manual_gradient_reference_values():
    covariance = DenseCovarianceMatrix(
        np.array([[1.4, 0.2], [0.2, 0.9]])
    )
    density = GaussianDensity(covariance, Vector(np.array([0.3, -0.4])))
    state = Vector(np.array([1.1, -0.7]))

    assert np.isclose(density.evaluate_log(state), -0.3270491803278689)
    assert np.isclose(
        density.evaluate_log(state, normalised=True), -2.264351676109797,
    )
    np.testing.assert_allclose(
        density.evaluate_log_gradient(state),
        [-0.639344262295082, 0.4754098360655737],
        rtol=0.0, atol=1e-12,
    )
    np.testing.assert_allclose(
        density.evaluate_log_hessian(state),
        [
            [-0.7377049180327869, 0.1639344262295082],
            [0.1639344262295082, -1.1475409836065573],
        ],
        rtol=0.0, atol=1e-12,
    )


def _reduced_quickstart():
    features = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
    data = Data(1, np.array([[0.0], [0.5], [1.0]]))
    data.measurement = np.array([[0.2], [1.1], [1.7]])
    likelihood = RegressionLikelihood(
        data,
        LinearForwardMap(features),
        GaussianResponse(IIDCovarianceMatrix(3, 0.4)),
    )
    prior = Gaussian(IIDCovarianceMatrix(2, 2.0), Vector(np.zeros(2)))
    posterior = UnnormalisedPosterior(prior, likelihood)
    return posterior, Vector(np.array([0.1, 1.4]))


def test_reduced_quickstart_density_and_gradient_output():
    posterior, parameter = _reduced_quickstart()
    assert np.isclose(posterior.evaluate_log(parameter), -0.6675)
    np.testing.assert_allclose(
        posterior.evaluate_log_gradient(parameter), [1.45, 0.175],
        rtol=0.0, atol=1e-12,
    )


def test_deterministic_proposal_reference_calculations():
    covariance = DenseCovarianceMatrix(np.array([[1.4, 0.2], [0.2, 0.9]]))
    density = GaussianDensity(covariance, Vector(np.array([0.3, -0.4])))
    state = Vector(np.array([1.1, -0.7]))
    malaProposal = MALAProposal(2, 0.3, density.evaluate_log_gradient)
    expectedDrift = np.array([1.0712295081967214, -0.6786065573770491])
    np.testing.assert_allclose(
        malaProposal._drift(state), expectedDrift, rtol=0.0, atol=1e-12,
    )

    mala = MetropolisAdjustedLangevinAlgorithm(
        density, 0.3, AcceptanceRateDiagnostics(),
    )
    transition = TransitionData(
        state,
        Vector(np.array([0.95, -0.2])),
        auxiliary={"drift": expectedDrift},
    )
    assert np.isclose(mala._log_mh_ratio(transition), 0.004726111092448165)

    reference = Gaussian(
        DiagonalCovarianceMatrix(np.array([1.5, 0.6])),
        Vector(np.array([0.2, -0.1])),
    )
    pcn = PCNProposal(reference, 0.4)
    pcn.state = Vector(np.array([0.8, -0.5]))
    fixedReferenceInput = Vector(np.array([1.1, 0.3]))
    reference.generate_realisation = lambda rng: fixedReferenceInput
    np.testing.assert_allclose(
        pcn.generate_proposal(None).proposal.coordinate,
        [1.109909083394701, -0.30660605559646714],
        rtol=0.0, atol=1e-12,
    )


def test_reduced_sglmm_example_output():
    covariance = MaternCovariance2D(0.3, 1.5, 1.0)
    gp = GaussianProcess.dna(covariance, q=2, d=2)
    sites = Grid(np.array([[0.1, 0.2], [0.4, 0.7], [0.8, 0.3]]))
    model = SGLMM(gp, sites)
    parameter = Vector(np.linspace(-0.3, 0.5, gp.parameterDimension))
    data = Data(1, sites.to_array())
    data.measurement = np.array([[1.0], [0.0], [3.0]])
    likelihood = SGLMMLikelihood(data, model, PoissonResponse())

    assert np.isclose(likelihood.evaluate_log(parameter), -3.1271698680170905)
    np.testing.assert_allclose(
        model(parameter),
        [-0.13849327906176964, 0.00300770988501932, -0.05670864098190717],
        rtol=0.0, atol=1e-12,
    )
    expectedGradient = np.array([
        0.4442181458436457, 0.6815406654447387, -0.07114562666984903,
        -0.6265769490514368, -0.23073723767951707, 0.14973151484643904,
        0.15581410134667315, -0.02729877246109536, -0.03960179427186451,
        0.3411518992321152, 0.4907368466793489, -0.5460400763591138,
        -0.20041409549677955, 0.14205230354198595, -0.03192326596486208,
        0.08953000955890246, 0.45771104688080994, -0.01603446877984283,
        -0.3114588344835292, -0.06695522619736648, 0.09219472280995618,
        0.07263588105229696, 0.3688401754274009, -0.29498312627273154,
        -0.06733364819637247,
    ])
    np.testing.assert_allclose(
        likelihood.evaluate_log_gradient(parameter), expectedGradient,
        rtol=0.0, atol=1e-12,
    )


def test_reduced_pde_example_output():
    vertices = np.linspace(0.0, 1.0, 7)
    midpoints = 0.5 * (vertices[:-1] + vertices[1:])
    diffusion = np.exp(0.3 * np.sin(2.0 * np.pi * midpoints))
    stiffness = p1_stiffness_1d(vertices, diffusion)
    mass = p1_mass_lumped_1d(vertices)
    rightHandSide = mass.diagonal().copy()
    stiffness, rightHandSide = apply_dirichlet_1d(stiffness, rightHandSide)

    np.testing.assert_allclose(
        spsolve(stiffness, rightHandSide),
        [
            0.0,
            0.06685125864331035,
            0.10381238628383663,
            0.12284653512575483,
            0.11626677517561175,
            0.07112610676838108,
            0.0,
        ],
        rtol=0.0, atol=1e-12,
    )
