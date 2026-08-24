"""
Bayesian Inverse Problem with PDE forward model

Recover a spatially varying diffusion coefficient from noisy point observations
of the elliptic PDE solution. Defines a custom forward model and a DNA Gaussian
process parametrisation and prior.
"""

import numpy as np
from scipy.sparse.linalg import spsolve

import styne
from styne.gp import GaussianProcess
from styne.parameter import Function
from styne.statistics import (
    MaternCovariance1D, Data, GaussianResponse, RegressionLikelihood,
    UnnormalisedPosterior, IIDCovarianceMatrix,
)
from styne.mcmc import PCNFactory
from styne.utility import (
    UniformGrid, PCNTuner, RWTunerConfig, ExplicitFunction,
    linear_interpolation_matrix,
)
from styne.utility.finiteelement import (
    p1_stiffness_1d, p1_mass_lumped_1d, apply_dirichlet_1d,
)

# check for matplotlib
try:
    import matplotlib.pyplot as plt
    hasMatplotlib = True
except ImportError:
    hasMatplotlib = False
    print("matplotlib not installed (pip install styne[plotting]); skipping plot.")

# fix seed
rng = np.random.default_rng(2026)


# definition of a custom forward model. For more details on the
# interface, see src/styne/model/model.py
class EllipticForwardMap(styne.ForwardMap):
    """
    Forward map: parameter to PDE solution at the observation sites.

    The solution is obtained via a P1 finite-element solve of
    -d/dx(exp(theta) dw/dx) = f with homogeneous Dirichlet boundary
    conditions. The log-diffusion theta is a parametrised GP, entering
    the assembly as a per-element coefficient exp(theta) evaluated at
    the element midpoints.

    Everything that does not depend on the parameter (mass matrix,
    right-hand side, observation interpolation matrix) is assembled once
    at construction. Per parameter update, _prepare reassembles and
    constrains the stiffness matrix, so that _evaluate only solves the
    system and interpolates to the observation sites.
    """

    def __init__(self, gp: GaussianProcess, source: ExplicitFunction,
                 feMesh: UniformGrid, obsSites: np.ndarray):
        super().__init__()

        self._gp = gp
        self._vertices = feMesh.axis

        # Evaluate the GP at the element midpoints used by the FE operator.
        h = self._vertices[1] - self._vertices[0]
        midpoints = UniformGrid(
            self._vertices[0] + 0.5 * h,
            self._vertices[-1] - 0.5 * h,
            len(feMesh) - 1,
        )
        self._gpGrid = midpoints

        # interpolation matrix from the FE solution to the observation sites
        self._obsInterp = linear_interpolation_matrix(obsSites, self._vertices)

        # parameter-independent right-hand side, boundary entries zeroed
        # to match the Dirichlet rows of the constrained stiffness matrix
        massMat = p1_mass_lumped_1d(self._vertices)
        rhs = massMat.diagonal() * source.evaluate(feMesh)
        rhs[0] = 0.0
        rhs[-1] = 0.0
        self._rhs = rhs

    @property
    def pType(self):
        """
        parameter type associated with the forward map. Must be derived from
        styne.Parameter.
        """
        return Function

    @property
    def pDim(self):
        """
        parameter dimension, as in: length of the coordinate vector (ndarray)
        """
        return self._gp.parameter.dimension

    def _prepare(self, parameter):
        """
        Set parameter as new state of the forward map and perform the
        parameter-dependent precomputation.
        """
        diffusion = np.exp(
            self._gp.evaluate(parameter.coordinate, self._gpGrid)
        )

        stiffMat = p1_stiffness_1d(self._vertices, diffusion)
        return apply_dirichlet_1d(stiffMat)

    def _evaluate(self, preparedState):
        """
        Evaluate the forward map: solve the precomputed FE system and
        interpolate the solution to the observation sites.
        """
        solution = spsolve(preparedState, self._rhs)
        return self._obsInterp @ solution


# --- SETUP ---

DIM = 1

# observation sites, away from the boundary where the Dirichlet
# conditions pin the solution
nObs = 10
obsSites = np.sort(rng.uniform(0.05, 0.95, nObs))

# GP parametrisation
corrLength = 0.2
smoothness = 2.5
margVar = 1.
gpCov = MaternCovariance1D(corrLength, smoothness, margVar)
gp = GaussianProcess.dna(gpCov, q=30, d=DIM)

# forward map
nFEM = 50
feMesh = UniformGrid(0., 1., nFEM)
source = ExplicitFunction(lambda x: 1.)
fMap = EllipticForwardMap(gp, source, feMesh, obsSites)


# --- SYNTHETIC DATA GENERATION ---

# the ground truth is computed on a finer mesh. Avoid the inverse crime.
truthMesh = UniformGrid(0., 1., 200)
truthVertices = truthMesh.axis


def log_diffusion_true(x):
    return 0.8 * np.sin(2 * np.pi * x) + 0.4 * np.cos(np.pi * x)


# FE setup, diffusion evaluated at the element midpoints
truthMidpoints = 0.5 * (truthVertices[:-1] + truthVertices[1:])
diffusionTrue = np.exp(log_diffusion_true(truthMidpoints))

stiffTrue = p1_stiffness_1d(truthVertices, diffusionTrue)
massTrue = p1_mass_lumped_1d(truthVertices)
rhsTrue = massTrue.diagonal() * source.evaluate(truthMesh)

# apply boundary conditions and solve for the ground truth
stiffTrue, rhsTrue = apply_dirichlet_1d(stiffTrue, rhsTrue)
solutionTrue = spsolve(stiffTrue, rhsTrue)

# generate measurement data at the observation sites
truthInterp = linear_interpolation_matrix(obsSites, truthVertices)
trueNoiseVar = 1e-4
measurement = truthInterp @ solutionTrue \
    + np.sqrt(trueNoiseVar) * rng.standard_normal(nObs)


# --- PROBLEM FORMULATION ---

# prior
prior = gp.measure

# noise model (i.i.d. Gaussian measurement noise, variance assumed known)
inferenceNoiseVar = trueNoiseVar
noiseModel = GaussianResponse(IIDCovarianceMatrix(nObs, inferenceNoiseVar))

# likelihood
data = Data(dimension=nObs, design=obsSites)
data.measurement = measurement
likelihood = RegressionLikelihood(data, fMap, noiseModel)

# posterior
posterior = UnnormalisedPosterior(prior, likelihood)


# --- INFERENCE ---

# choose MCMC method
factory = PCNFactory()
factory.target = posterior
factory.rng = rng

# tune proposal scale
nTuning = 500
initState = prior.mean.clone()
sampler = PCNTuner(factory, initState, RWTunerConfig(nTuning=nTuning)).tune()

# run mcmc
nSteps = 15_000
sampler.run(nSteps, initState, progress=True, description="Sampling pCN")

# discard burn-in
nBurnIn = 5000
trajectory = np.array(sampler.chain.trajectory)[nBurnIn:]

# diagnostics
acceptanceRate = sampler.diagnostics.global_acceptance_rate()
print(f"acceptance rate={acceptanceRate:.3f}")


# --- POSTPROCESSING ---

# reconstruct the diffusion coefficient exp(theta) on a display grid.
plotMesh = UniformGrid(0., 1., 200)

fields = np.empty((len(trajectory), len(plotMesh)))
for i, coefficients in enumerate(trajectory):
    fields[i] = np.exp(gp.evaluate(coefficients, plotMesh))

# posterior mean and pointwise 95% credible band
posteriorMean = fields.mean(axis=0)
diffusionLower, diffusionUpper = np.percentile(fields, [2.5, 97.5], axis=0)

# summary statistics against the ground truth
diffusionTrueNodal = np.exp(log_diffusion_true(plotMesh.axis))
correlation = np.corrcoef(diffusionTrueNodal, posteriorMean)[0, 1]
coverage = np.mean(
    (diffusionTrueNodal >= diffusionLower)
    & (diffusionTrueNodal <= diffusionUpper)
)
print(f"posterior mean vs truth: diffusion correlation = {correlation:.3f}")
print(f"pointwise 95% band covers {100 * coverage:.1f}% of the true field")

if hasMatplotlib:
    fig, (axDiffusion, axSolution) = plt.subplots(
        1, 2, figsize=(10, 4), layout="constrained"
    )

    axDiffusion.plot(
        plotMesh.axis, diffusionTrueNodal, "k-", lw=2, label="true diffusion"
    )
    axDiffusion.plot(
        plotMesh.axis, posteriorMean, "r--", lw=2, label="posterior mean"
    )
    axDiffusion.fill_between(
        plotMesh.axis, diffusionLower, diffusionUpper, color="r", alpha=0.2,
        label="95% credible band",
    )
    axDiffusion.set_xlabel("x")
    axDiffusion.set_ylabel("exp(theta)")
    axDiffusion.legend()

    axSolution.plot(
        truthVertices, solutionTrue, "k-", lw=2, label="true solution"
    )
    axSolution.plot(
        obsSites, measurement, "ko", ms=5, label="noisy observations"
    )
    axSolution.set_xlabel("x")
    axSolution.set_ylabel("solution")
    axSolution.legend()

    plt.savefig("pde_inverse.png", dpi=150)
    plt.close()
    print("saved pde_inverse.png")
