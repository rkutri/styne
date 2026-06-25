import numpy as np
import pytest

from styne.parameter.vector import Vector
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.gaussian import Gaussian, GaussianDensity
from styne.statistics.radonnikodym import RadonNikodym
from styne.mcmc.method.mlda import MLDAFactory
from styne.utility.densityarithmetic import LogScalingWrapper
from tests.testSetup import GaussianTargetDensity


def _make_rn_surrogate(
    priorVariance: float, likelihoodVariance: float, dimension: int
) -> RadonNikodym:
    priorCovariance = IIDCovarianceMatrix(dimension, priorVariance)
    priorMeasure = Gaussian(priorCovariance)
    priorMeasure.mean = Vector(np.zeros(dimension))
    likelihoodCovariance = IIDCovarianceMatrix(dimension, likelihoodVariance)
    likelihoodDensity = GaussianDensity(
        likelihoodCovariance, Vector(np.zeros(dimension))
    )
    return RadonNikodym(priorMeasure, likelihoodDensity)


def test_mlda_tempering_validation():
    dimension = 2
    targetDensity = GaussianTargetDensity(
        Vector(np.zeros(dimension)), np.eye(dimension)
    )
    rnSurrogate = _make_rn_surrogate(
        priorVariance=1.0, likelihoodVariance=1.0, dimension=dimension
    )
    nonRnSurrogate = GaussianTargetDensity(
        Vector(np.zeros(dimension)), np.eye(dimension)
    )

    factory = MLDAFactory()
    factory.target = targetDensity
    factory.surrogate = [rnSurrogate]
    factory.nChain = [5]
    factory.tempering = None
    factory._validate()

    factory.tempering = 0.5
    with pytest.raises(TypeError, match="tempering must be a list of floats"):
        factory._validate()

    factory.tempering = [True]
    with pytest.raises(TypeError, match="tempering must be a list of floats"):
        factory._validate()

    factory.tempering = ["0.5"]
    with pytest.raises(TypeError, match="tempering must be a list of floats"):
        factory._validate()

    factory.tempering = [0.5, 0.8]
    with pytest.raises(ValueError, match="Inconsistent parameter lengths"):
        factory._validate()

    factory.tempering = [0.0]
    with pytest.raises(ValueError, match="All tempering parameters must be in"):
        factory._validate()

    factory.tempering = [1.2]
    with pytest.raises(ValueError, match="All tempering parameters must be in"):
        factory._validate()

    factory.surrogate = [nonRnSurrogate]
    factory.tempering = [0.8]
    with pytest.raises(TypeError, match="Tempering < 1.0 is only supported"):
        factory._validate()

    factory.tempering = [1.0]
    factory._validate()


def test_mlda_tempering_sampler_creation():
    dimension = 2
    targetDensity = GaussianTargetDensity(
        Vector(np.zeros(dimension)), np.eye(dimension)
    )
    rnSurrogate0 = _make_rn_surrogate(
        priorVariance=2.0, likelihoodVariance=0.5, dimension=dimension
    )
    rnSurrogate1 = _make_rn_surrogate(
        priorVariance=1.0, likelihoodVariance=1.0, dimension=dimension
    )

    factory = MLDAFactory()
    factory.target = targetDensity
    factory.surrogate = [rnSurrogate0, rnSurrogate1]
    factory.nChain = [5, 5]
    factory.tempering = [0.7, 1.0]
    factory.root.proposalCovariance = IIDCovarianceMatrix(dimension, 1.0)

    mcmc = factory.create()

    lvl1Chain = mcmc._proposalMethod._surrogateMeasure.mcmc
    rootChain = lvl1Chain._proposalMethod._surrogateMeasure.mcmc
    rootTarget = rootChain.target

    assert isinstance(rootTarget, RadonNikodym)
    assert rootTarget.reference is rnSurrogate0.reference

    assert isinstance(rootTarget.derivative, LogScalingWrapper)
    assert rootTarget.derivative.scaling == pytest.approx(0.7)
    assert rootTarget.derivative.baseDensity is rnSurrogate0.derivative

    coordinate = Vector(np.array([0.5, -0.5]))
    expectedLogDensity = (
        0.7 * rnSurrogate0.derivative.evaluate_log(coordinate)
        + rnSurrogate0.reference.density.evaluate_log(coordinate)
    )
    assert rootTarget.evaluate_log(coordinate) == pytest.approx(
        expectedLogDensity
    )
