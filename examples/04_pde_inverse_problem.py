"""Elliptic PDE inverse problem. Recover a spatially varying log-diffusion
coefficient in -d/dx(exp(theta) dw/dx) = 1 from noisy point observations of the
solution, using a custom forward model and a DNA Gaussian process prior."""

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

try:
    import matplotlib.pyplot as plt
    hasMatplotlib = True
except ImportError:
    hasMatplotlib = False
    print("matplotlib not installed (pip install styne[plotting]); skipping plot.")


class EllipticForwardModel(Model):
    """Forward map theta -> solution values at observation nodes.

    P1 finite-element solve of -d/dx(exp(theta) dw/dx) = source with
    homogeneous Dirichlet conditions. The field enters as a per-element
    diffusion coefficient exp(theta).
    """

    def __init__(self, gp, mesh, source, obsIndices):
        super().__init__()
        self._gp = gp
        self._meshVertices = mesh.axis
        self._source = source
        self._obsIndices = obsIndices
        self._diffusion = None

    @property
    def pType(self):
        return Function

    @property
    def pDim(self):
        return self._gp.parameter.dimension

    def _interpolate(self, parameter):
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


rng = np.random.default_rng(0)

covariance = MaternCovariance1D(0.4, 2.5, 0.5)
gp = GaussianProcess.dna(covariance, q=24, d=1)
dim = gp.parameter.dimension
mesh = UniformGrid(0.0, 1.0, dim)
gp.sites = mesh
vertices = mesh.axis
source = np.ones(dim)

nObs = 10
obsIndices = np.linspace(6, dim - 7, nObs).astype(int)
noiseStd = 0.035

thetaTrue = 0.8 * np.sin(2 * np.pi * vertices) + 0.4 * np.cos(np.pi * vertices)
diffusionTrue = np.exp(0.5 * (thetaTrue[:-1] + thetaTrue[1:]))
stiffnessTrue = p1_stiffness_1d(vertices, diffusionTrue)
massTrue = p1_mass_lumped_1d(vertices)
sBC, rBC = apply_dirichlet_1d(stiffnessTrue, massTrue.diagonal() * source)
solutionTrue = spsolve(sBC, rBC)
measurement = solutionTrue[obsIndices] + noiseStd * rng.standard_normal(nObs)

model = EllipticForwardModel(gp, mesh, source, obsIndices)
data = Data(nObs, mesh.to_array()[obsIndices])
data.measurement = measurement
likelihood = RegressionLikelihood(
    data, model, GaussianResponse(IIDCovarianceMatrix(nObs, noiseStd**2))
)
posterior = UnnormalisedPosterior(gp.measure, likelihood)

factory = PCNFactory()
factory.target = posterior
factory.rng = rng
sampler = PCNTuner(
    factory, gp.measure.generate_realisation(rng=rng), RWTunerConfig(nTuning=400)
).tune()

print("running pCN (a few seconds) ...")
sampler.run(8000, gp.measure.generate_realisation(rng=rng))
trajectory = np.array(sampler.chain.trajectory)[2000:]

fields = np.empty((len(trajectory), dim))
for i, coefficients in enumerate(trajectory):
    gp.parameter.coordinate = coefficients
    fields[i] = np.exp(gp.at_sites())
posteriorMean = fields.mean(axis=0)
diffusionTrueNodal = np.exp(thetaTrue)

correlation = np.corrcoef(diffusionTrueNodal, posteriorMean)[0, 1]
print(f"posterior mean vs truth: diffusion correlation = {correlation:.3f}")

if hasMatplotlib:
    plt.figure(figsize=(8, 4))
    plt.plot(vertices, diffusionTrueNodal, "k-", lw=2, label="true diffusion")
    plt.plot(vertices, posteriorMean, "r--", lw=2, label="posterior mean")
    plt.plot(vertices[obsIndices], np.exp(thetaTrue[obsIndices]), "ko",
             ms=5, label="observation sites")
    plt.xlabel("x")
    plt.ylabel("exp(theta)")
    plt.legend()
    plt.savefig("pde_inverse.png", dpi=150)
    print("saved pde_inverse.png")
