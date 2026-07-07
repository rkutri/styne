"""
PDE-Constrained Bayesian Inverse Problem

Recover a spatially varying diffusion coefficient from noisy point observations
of an elliptic PDE solution, using a custom forward model and a DNA Gaussian
process prior.
"""

import numpy as np
from scipy.sparse.linalg import spsolve

from styne.gp import GaussianProcess
from styne.parameter import Function
from styne.statistics import (
    MaternCovariance1D, Data, GaussianResponse, RegressionLikelihood,
    UnnormalisedPosterior, IIDCovarianceMatrix,
)
from styne.mcmc import PCNFactory
from styne.utility import UniformGrid, PCNTuner, RWTunerConfig
from styne.model.model import Model
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


class EllipticForwardModel(Model):
    """Forward map theta -> solution values at observation nodes.

    P1 finite-element solve of -d/dx(exp(theta) dw/dx) = source with
    homogeneous Dirichlet conditions. The field enters as a per-element
    diffusion coefficient exp(theta).
    """

    def __init__(self, gp, mesh, source, obsIndices):
        super().__init__()
        self._gp = gp
        self._mesh = mesh
        self._meshVertices = mesh.axis
        self._source = source
        self._obsIndices = obsIndices
        self._diffusion = None

        self._gp.sites = mesh

    @property
    def pType(self):
        return Function

    @property
    def pDim(self):
        return self._gp.parameter.dimension

    def _interpolate(self, parameter):
        self._gp.sites = self._mesh
        self._gp.parameter.coordinate = parameter.coordinate
        field = self._gp.at_sites()
        self._diffusion = np.exp(0.5 * (field[:-1] + field[1:]))

    def _evaluate(self):
        stiffness = p1_stiffness_1d(self._meshVertices, self._diffusion)
        mass = p1_mass_lumped_1d(self._meshVertices)
        rhs = mass.diagonal() * self._source
        stiffnessBC, rhsBC = apply_dirichlet_1d(stiffness, rhs)
        solution = spsolve(stiffnessBC, rhsBC)
        self._evaluation = solution[self._obsIndices]


# --- SETUP ---

# PDE mesh and GP prior
spatialDim = 1
resolution = 24

covariance = MaternCovariance1D(0.3, 2.5, 1.)
gp = GaussianProcess.dna(covariance, q=resolution, d=spatialDim)
mesh = gp.engine.nativeGrid
dim = len(mesh)

vertices = mesh.axis
source = np.ones(dim)

# measurement locations
nObs = 20
obsMargin = 1
obsIndices = np.linspace(obsMargin, dim - obsMargin - 1, nObs).astype(int)
noiseStd = 0.015

# synthetic data generation
thetaTrue = 0.8 * np.sin(2 * np.pi * vertices) + 0.4 * np.cos(np.pi * vertices)
diffusionTrue = np.exp(0.5 * (thetaTrue[:-1] + thetaTrue[1:]))
stiffnessTrue = p1_stiffness_1d(vertices, diffusionTrue)
massTrue = p1_mass_lumped_1d(vertices)
stiffnessBC, rhsBC = apply_dirichlet_1d(
    stiffnessTrue, massTrue.diagonal() * source,
)
solutionTrue = spsolve(stiffnessBC, rhsBC)
measurement = solutionTrue[obsIndices] + noiseStd * rng.standard_normal(nObs)


# --- PROBLEM FORMULATION ---

# prior definition
prior = gp.measure

# likelihood definition
model = EllipticForwardModel(gp, mesh, source, obsIndices)
data = Data(dimension=nObs, design=mesh.to_array()[obsIndices])
data.measurement = measurement
noiseModel = GaussianResponse(IIDCovarianceMatrix(nObs, noiseStd**2))
likelihood = RegressionLikelihood(data, model, noiseModel)

# posterior definition
posterior = UnnormalisedPosterior(prior, likelihood)


# --- INFERENCE ---

factory = PCNFactory()
factory.target = posterior
factory.rng = rng

# tune proposal scale
nTuning = 500
initState = prior.generate_realisation(rng=rng)
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

# compute the posterior mean of the diffusion coefficient exp(theta)
fields = np.empty((len(trajectory), dim))
for i, coefficients in enumerate(trajectory):
    gp.sites = mesh
    gp.parameter.coordinate = coefficients
    fields[i] = np.exp(gp.at_sites())

posteriorMean = fields.mean(axis=0)
diffusionTrueNodal = np.exp(thetaTrue)

correlation = np.corrcoef(diffusionTrueNodal, posteriorMean)[0, 1]
print(f"posterior mean vs truth: diffusion correlation = {correlation:.3f}")

if hasMatplotlib:
    diffusionLower, diffusionUpper = np.percentile(
        fields, [2.5, 97.5], axis=0
    )
    fig, (axDiffusion, axSolution) = plt.subplots(
        1, 2, figsize=(10, 4), layout="constrained"
    )
    axDiffusion.plot(
        vertices, diffusionTrueNodal, "k-", lw=2, label="true diffusion"
    )
    axDiffusion.plot(
        vertices, posteriorMean, "r--", lw=2, label="posterior mean"
    )
    axDiffusion.fill_between(
        vertices, diffusionLower, diffusionUpper, color="r", alpha=0.2,
        label="95% credible band",
    )
    axDiffusion.set_xlabel("x")
    axDiffusion.set_ylabel("exp(theta)")
    axDiffusion.legend()

    axSolution.plot(vertices, solutionTrue, "k-", lw=2, label="true solution")
    axSolution.plot(
        vertices[obsIndices], measurement, "ko",
        ms=5, label="noisy observations",
    )
    axSolution.set_xlabel("x")
    axSolution.set_ylabel("solution")
    axSolution.legend()

    plt.savefig("pde_inverse.png", dpi=150)
    plt.close()
    print("saved pde_inverse.png")
