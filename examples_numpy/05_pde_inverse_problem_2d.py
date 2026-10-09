"""Two-dimensional Bayesian elliptic inverse problem.

The fine posterior uses a DNA log-diffusion field and a Q1 finite-element
solve. A nested coarse mesh supplies the shared MLDA/DART surrogate. Smoke
mode checks the complete interface; longer runs are intended to be archived
by the dissertation reproducibility workflow.
"""

import argparse
import json
import time

import numpy as np
from scipy.sparse.linalg import spsolve

import styne
from styne.gp import GaussianProcess
from styne.mcmc import DARTFactory, MALAFactory, MLDAFactory, PCNFactory
from styne.parameter import Vector
from styne.statistics import (
    Data,
    GaussianResponse,
    IIDCovarianceMatrix,
    MaternCovariance2D,
    RegressionLikelihood,
    UnnormalisedPosterior,
)
from styne.utility import UniformGrid, bilinear_interpolation_matrix
from styne.utility.finiteelement import (
    apply_dirichlet_2d,
    q1_log_diffusion_adjoint_2d,
    q1_mass_lumped_2d,
    q1_stiffness_2d,
)


def boundary_nodes(nx, ny):
    return np.array([
        i * (ny + 1) + j
        for i in range(nx + 1)
        for j in range(ny + 1)
        if i in (0, nx) or j in (0, ny)
    ], dtype=int)


class EllipticForwardMap2D(styne.ForwardMap):
    """Q1 solve for ``-div(exp(u) grad p) = 1`` with point observations."""

    def __init__(self, gp, nElements, observationSites):
        super().__init__()
        self._gp = gp
        self._nx = self._ny = int(nElements)
        self._x = np.linspace(0.0, 1.0, self._nx + 1)
        self._y = np.linspace(0.0, 1.0, self._ny + 1)
        hx = 1.0 / self._nx
        hy = 1.0 / self._ny
        self._centers = UniformGrid(
            (0.5 * hx, 1.0 - 0.5 * hx, self._nx),
            (0.5 * hy, 1.0 - 0.5 * hy, self._ny),
        )
        self._observation = bilinear_interpolation_matrix(
            observationSites, self._x, self._y
        )
        self._boundary = boundary_nodes(self._nx, self._ny)
        self._rhs = q1_mass_lumped_2d(self._x, self._y).diagonal()
        self._rhs[self._boundary] = 0.0
        self.forwardSolves = 0
        self.adjointSolves = 0

    @property
    def pType(self):
        return Vector

    @property
    def pDim(self):
        return self._gp.parameterDimension

    def reset_counters(self):
        self.forwardSolves = 0
        self.adjointSolves = 0

    def _system(self, parameter):
        logDiffusion = self._gp.evaluate(
            parameter.coordinate, self._centers
        )
        diffusion = np.exp(logDiffusion)
        matrix = q1_stiffness_2d(self._x, self._y, diffusion)
        matrix = apply_dirichlet_2d(
            matrix, nx=self._nx, ny=self._ny
        )
        return matrix, diffusion

    def _prepare(self, parameter):
        return self._system(parameter)[0]

    def _evaluate(self, preparedState):
        self.forwardSolves += 1
        return self._observation @ spsolve(preparedState, self._rhs)

    def directional_derivative(self, parameter, direction):
        matrix, diffusion = self._system(parameter)
        primal = spsolve(matrix, self._rhs)
        deltaLog = self._gp.evaluate(
            direction.coordinate, self._centers
        )
        derivative = q1_stiffness_2d(
            self._x, self._y, diffusion * deltaLog
        ).tolil()
        derivative[self._boundary, :] = 0.0
        tangent = spsolve(matrix, -(derivative.tocsc() @ primal))
        self.forwardSolves += 1
        return self._observation @ tangent

    def adjoint_derivative(self, parameter, cotangent):
        matrix, diffusion = self._system(parameter)
        primal = spsolve(matrix, self._rhs)
        adjoint = spsolve(matrix.T, self._observation.T @ cotangent)
        adjoint[self._boundary] = 0.0
        elementGradient = q1_log_diffusion_adjoint_2d(
            self._x, self._y, diffusion, primal, adjoint
        )
        self.forwardSolves += 1
        self.adjointSolves += 1
        return self._gp.adjoint_derivative(
            parameter.coordinate, elementGradient, self._centers
        )


def solve_truth(nElements, observationSites):
    x = np.linspace(0.0, 1.0, nElements + 1)
    y = np.linspace(0.0, 1.0, nElements + 1)
    hx = 1.0 / nElements
    centers = UniformGrid(
        (0.5 * hx, 1.0 - 0.5 * hx, nElements),
        (0.5 * hx, 1.0 - 0.5 * hx, nElements),
    ).to_array()
    logDiffusion = (
        0.7 * np.sin(2.0 * np.pi * centers[:, 0])
        * np.sin(np.pi * centers[:, 1])
        + 0.35 * np.cos(np.pi * centers[:, 1])
    )
    matrix = q1_stiffness_2d(x, y, np.exp(logDiffusion))
    rhs = q1_mass_lumped_2d(x, y).diagonal()
    matrix, rhs = apply_dirichlet_2d(
        matrix, rhs, nx=nElements, ny=nElements
    )
    solution = spsolve(matrix, rhs)
    interpolation = bilinear_interpolation_matrix(observationSites, x, y)
    return interpolation @ solution


def integrated_autocorrelation(values):
    centered = np.asarray(values) - np.mean(values)
    variance = centered @ centered
    if variance == 0.0:
        return np.inf
    total = 1.0
    for lag in range(1, len(centered)):
        correlation = centered[:-lag] @ centered[lag:] / variance
        if correlation <= 0.0:
            break
        total += 2.0 * correlation
    return max(total, 1.0)


def make_sampler(name, target, surrogate, rng, rootSteps, gamma):
    if name == "pcn":
        factory = PCNFactory()
        factory.beta = 0.12
    elif name == "mala":
        factory = MALAFactory()
        factory.stepSize = 0.025
        factory.gradient = target.evaluate_log_gradient
    elif name == "mlda":
        factory = MLDAFactory(root="pcn")
        factory.surrogate = [surrogate]
        factory.nChain = [rootSteps]
        factory.root.beta = 0.12
    elif name == "dart":
        factory = DARTFactory(root="pcn")
        factory.surrogate = [surrogate]
        factory.regularisation = [gamma]
        factory.tempering = [0.5]
        factory.nChain = [rootSteps]
        factory.burnin = 1
        factory.root.beta = 0.12
    else:
        raise ValueError(f"unknown method: {name}")
    factory.target = target
    factory.rng = rng
    return factory.create()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--method", choices=("pcn", "mala", "mlda", "dart", "all"),
        default="all",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--gamma", type=float, default=4.0)
    args = parser.parse_args()

    rng = np.random.default_rng(2026)
    nSide = 3 if args.smoke else 5
    axis = np.linspace(0.15, 0.85, nSide)
    obsX, obsY = np.meshgrid(axis, axis, indexing="ij")
    observationSites = np.column_stack((obsX.ravel(), obsY.ravel()))

    covariance = MaternCovariance2D(0.22, 1.5, 0.7)
    gp = GaussianProcess.dna(covariance, q=3 if args.smoke else 7, d=2)
    fineMap = EllipticForwardMap2D(
        gp, 8 if args.smoke else 24, observationSites
    )
    coarseMap = EllipticForwardMap2D(
        gp, 4 if args.smoke else 12, observationSites
    )

    truth = solve_truth(14 if args.smoke else 48, observationSites)
    noiseVariance = 2.5e-5
    measurement = truth + np.sqrt(noiseVariance) * rng.standard_normal(
        len(observationSites)
    )
    data = Data(dimension=len(observationSites), design=observationSites)
    data.measurement = measurement
    response = GaussianResponse(
        IIDCovarianceMatrix(len(observationSites), noiseVariance)
    )
    fineLikelihood = RegressionLikelihood(data, fineMap, response)
    coarseLikelihood = RegressionLikelihood(data, coarseMap, response)
    target = UnnormalisedPosterior(gp.measure, fineLikelihood)
    surrogate = UnnormalisedPosterior(gp.measure, coarseLikelihood)

    methods = ("pcn", "mala", "mlda", "dart") \
        if args.method == "all" else (args.method,)
    nSteps = args.steps or (12 if args.smoke else 2000)
    burnin = 2 if args.smoke else nSteps // 4
    rootSteps = 4 if args.smoke else 12
    initialState = gp.measure.mean
    summaries = {}

    for index, method in enumerate(methods):
        methodRng = np.random.default_rng(2026 + index)
        fineMap.reset_counters()
        coarseMap.reset_counters()
        sampler = make_sampler(
            method, target, surrogate, methodRng, rootSteps, args.gamma
        )
        started = time.perf_counter()
        sampler.run(
            nSteps, initialState, progress=not args.smoke,
            description=f"Sampling {method.upper()}",
        )
        elapsed = time.perf_counter() - started
        trajectory = np.asarray(sampler.chain.trajectory)[burnin:]
        qoi = trajectory[:, 0]
        iat = integrated_autocorrelation(qoi)
        summaries[method] = {
            "acceptance": sampler.diagnostics.global_acceptance_rate(),
            "adjoint_solves": fineMap.adjointSolves,
            "coarse_forward_solves": coarseMap.forwardSolves,
            "fine_forward_solves": fineMap.forwardSolves,
            "qoi_ess": len(qoi) / iat,
            "seconds": elapsed,
            "steps": nSteps,
        }

    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
