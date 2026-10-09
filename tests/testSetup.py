import numpy as np

from enum import Enum
from sys import exit


from scipy.stats import multivariate_normal
from scipy.integrate import solve_ivp

from styne.parameter.vector import Vector
from styne.model.forwardmap import ForwardMap
from styne.statistics.interface import DensityInterface
from styne.statistics.data import Data


class EvaluationStatus(Enum):
    NONE = 0
    SUCCESS = 1
    FAILURE = 2


def covariance_matrix(dim, kappa):

    eigs = np.geomspace(1., kappa, num=dim)
    return np.diag(eigs)


class RosenbrockTargetDensity(DensityInterface):
    """
    Multidimensional Rosenbrock ("banana") target density.
    The parameter 'cond' acts as an inverse variance scaling:
    larger values make the valley narrower and thus harder.
    """

    def __init__(self, mode: Vector, cond: float):
        self._mode = mode
        self._cond = float(cond)
        self._dim = len(mode.coordinate)

        self._scale = np.sqrt(cond)

    @property
    def domainType(self):
        return type(self._mode)

    @property
    def domainDimension(self):
        return self._mode.dimension

    def evaluate_log(self, parameter):
        x = np.atleast_1d(parameter.coordinate)
        if len(x) != self._dim:
            raise ValueError(f"Expected dim={self._dim}, got {len(x)}")

        x0 = x[:-1]
        x1 = x[1:]

        # standard Rosenbrock potential
        V = 100.0 * (x1 - x0**2)**2 + (x0 - 1.0)**2

        return -0.5 * self._scale**2 * np.sum(V)

    def evaluate_on_mesh(self, mesh):
        """
        Evaluate exp(log π(x)) on a given mesh (nPoints x dim).
        """
        mesh = np.atleast_2d(mesh)
        vals = np.empty(mesh.shape[0])
        for i, point in enumerate(mesh):
            vals[i] = np.exp(self.evaluate_log(Vector(point)))
        return vals


class GaussianTargetDensity(DensityInterface):

    def __init__(self, mean: Vector, cov):

        meanCoeff = np.atleast_1d(mean.coordinate)
        cov = np.atleast_2d(cov)

        self._mean = mean
        self._cov = cov
        self._dist = multivariate_normal(mean=meanCoeff, cov=cov)

    @property
    def domainType(self):
        return type(self._mean)

    @property
    def domainDimension(self):
        return self._mean.dimension

    def evaluate_log(self, parameter):

        paramCoeff = np.atleast_1d(parameter.coordinate)
        return self._dist.logpdf(paramCoeff)

    def evaluate_on_mesh(self, mesh):
        return self._dist.pdf(mesh)


class LotkaVolterraParameter(Vector):

    @classmethod
    def from_coefficient(cls, coefficient):
        return cls(coefficient)

    @classmethod
    def from_interpolation(cls, value):
        return cls(np.log(value))

    def evaluate(self):
        return np.exp(self.coordinate)


class LotkaVolterraSolver(ForwardMap):

    def __init__(self, design, config):

        super().__init__()

        self.x_ = design
        self.tBoundary_ = (0., config['T'])
        self._fixedParam = [config['alpha'], config['gamma']]
        self._dataShape = (config['nData'], config['dataDim'])
        self._solverMethod = config['solver']
        self._solverRTol = config['rtol']

        self._status = EvaluationStatus.NONE

    @property
    def pType(self):
        return LotkaVolterraParameter

    @property
    def pDim(self):
        return self._dataShape[1]

    @property
    def status(self):
        return self._status

    @property
    def dataShape(self):
        return self._dataShape

    @property
    def nData(self):
        return self._dataShape[0]

    @property
    def dataDim(self):
        return self.pDim

    def _flow(self, t, x, alpha, beta, gamma, delta):

        return [alpha * x[0] - beta * x[0] * x[1],
                delta * x[0] * x[1] - gamma * x[1]]

    def _prepare(self, parameter):
        paramEval = parameter.evaluate()
        return paramEval[0], paramEval[1]

    def _evaluate(self, preparedState):

        self._status = EvaluationStatus.SUCCESS

        alpha = self._fixedParam[0]
        beta = preparedState[0]
        gamma = self._fixedParam[1]
        delta = preparedState[1]

        evaluation = np.zeros(self._dataShape)

        def odeFlow(t, x): return self._flow(t, x, alpha, beta, gamma, delta)

        for n in range(self._dataShape[0]):

            odeResult = solve_ivp(
                odeFlow, self.tBoundary_, self.x_[n, :],
                method=self._solverMethod, rtol=self._solverRTol)

            if odeResult.status != 0:

                print("forward map evaluation failed. Reason: \n"
                      + odeResult.message)

                self._status = EvaluationStatus.FAILURE
                raise RuntimeError(
                    "ODE solver failed: " + odeResult.message)

            evaluation[n, :] = odeResult.y[:, -1]

        return evaluation

    def full_solution(self, parameter, y0):

        paramEval = parameter.evaluate()

        beta = paramEval[0]
        delta = paramEval[1]

        alpha = self._fixedParam[0]
        gamma = self._fixedParam[1]

        def odeFlow(t, x): return self._flow(t, x, alpha, beta, gamma, delta)
        odeResult = solve_ivp(odeFlow, self.tBoundary_, y0, method='LSODA')

        if odeResult.status != 0:

            print("forward map evaluation failed. aborting program.")
            print(odeResult.message)
            exit()

        return (odeResult.t, odeResult.y)


def generate_synthetic_data(parameter, solver, noiseVar, rng=None):

    sig = np.sqrt(noiseVar)

    evaluation = solver(parameter)

    if rng is None:
        rng = np.random.default_rng()

    measurement = evaluation + sig * rng.standard_normal(solver.dataShape)

    data = Data(solver.dataDim, solver.x_)
    data.measurement = measurement

    return data
