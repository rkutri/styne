import pytest
import numpy as np
from styne.mcmc.diagnostics import (
    AcceptanceRateDiagnostics, PersistentAcceptanceRateDiagnostics
)
from styne.mcmc.method.dart import DARTFactory
from styne.mcmc.method.gibbs import GibbsBuilder, BlockGibbs, GibbsSampler
from styne.mcmc.sampler import MCMCSampler
from styne.parameter.parameter import Parameter
from styne.parameter.vector import Vector
from styne.parameter.block import BlockParameter
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.gaussian import Gaussian
from styne.utility.progress import make_reporter, TqdmProgress
from styne.mcmc.method.mrw import MRWFactory
from styne.mcmc.surrogate import SurrogateTransitionMeasure
from styne.mcmc.localised import LocalisedSurrogateTransitionMeasure
from styne.mcmc.transition import TransitionData

from styne.statistics.radonnikodym import RadonNikodym
class DummyParameter(Parameter):
    def __init__(self, coord):
        self._coord = coord

    @property
    def coordinate(self):
        return self._coord

    @property
    def dimension(self):
        return len(self._coord)

    def with_coordinate(self, coordinate):
        return DummyParameter(coordinate)


class DummySampler(MCMCSampler):
    def __init__(self):
        super().__init__()
        class DummyChain:
            def append(self, x, a=None): pass
            def clear(self): pass
        self._chain = DummyChain()

    @property
    def chain(self):
        return self._chain

    def _iterate(self):
        return DummyParameter(self._lastState.coordinate + 1.0)


def test_progress_regression():
    reporter = make_reporter(True, total=3)
    assert isinstance(reporter, TqdmProgress)
    
    sampler = DummySampler()
    initialState = DummyParameter(np.array([0.0]))
    sampler.run(3, initialState, progress=True)
    assert sampler.lastState.coordinate[0] == 3.0


def test_stream_seam():
    sampler = DummySampler()
    initialState = DummyParameter(np.array([0.0]))
    
    states = list(sampler.stream_run(5, initialState))
    assert len(states) == 5
    assert sampler.lastState.coordinate[0] == 5.0
    assert states[-1].coordinate[0] == 5.0
    
    sampler.run(5, initialState)
    assert sampler.lastState.coordinate[0] == 5.0


def test_gibbs_seam():
    class DummyModel:
        nBlocks = 2
        def conditional(self, idx, state):
            class Conditional:
                def sample(self, randomState):
                    return (
                        DummyParameter(state.block(idx).coordinate + 1.0),
                        randomState,
                    )
            return Conditional()

    builder = GibbsBuilder()
    builder.model = DummyModel()
    sampler = builder.build()
    
    initialState = BlockParameter([
        DummyParameter(np.array([0.0])), DummyParameter(np.array([1.0]))
    ])
    sampler.storeChain = False
    
    states = list(sampler.stream_run(4, initialState))
    assert len(states) == 4
    
    assert 'run' not in GibbsSampler.__dict__


def test_diagnostics_setter():
    sampler = DummySampler()
    diagnostics = AcceptanceRateDiagnostics()
    sampler.diagnostics = diagnostics
    assert sampler.diagnostics is diagnostics


def test_subsampler_reset():
    from tests.testSetup import GaussianTargetDensity
    targetDensity = GaussianTargetDensity(Vector(np.zeros(2)), np.eye(2))
    surrogateDensity1 = GaussianTargetDensity(Vector(np.zeros(2)), np.eye(2))
    surrogateDensity2 = GaussianTargetDensity(Vector(np.zeros(2)), np.eye(2))

    factory = DARTFactory(root="mrw")
    factory.target = targetDensity
    factory.surrogate = [surrogateDensity1, surrogateDensity2]
    factory.nChain = [2, 2]
    factory.regularisation = [0.1, 0.1]
    factory.subDiagnostics = PersistentAcceptanceRateDiagnostics
    factory.root.proposalCovariance = IIDCovarianceMatrix(2, 1.0)
    
    mainChain = factory.create()
    assert len(mainChain.subsamplers) == 2
    
    initialState = Vector(np.zeros(2))
    mainChain.run(5, initialState)
    
    for subsampler in mainChain.subsamplers:
        assert len(subsampler.diagnostics._recent) > 0 or subsampler.diagnostics._total > 0
        
    mainChain.clear()
    
    for subsampler in mainChain.subsamplers:
        assert len(subsampler.diagnostics._recent) == 0
        assert subsampler.diagnostics._total == 0


def test_mutable_default_fix():
    from styne.mcmc.method.mrw import MRWFactory
    from tests.testSetup import GaussianTargetDensity
    factory = MRWFactory()
    factory.target = GaussianTargetDensity(Vector(np.zeros(1)), np.eye(1))
    factory.proposalCovariance = IIDCovarianceMatrix(1, 1.0)
    sampler = factory.create()
    
    measure1 = SurrogateTransitionMeasure(sampler, 1)
    measure2 = SurrogateTransitionMeasure(sampler, 1)
    assert measure1.initialMeasure is not measure2.initialMeasure
    
    from styne.mcmc.localised import LocalisedSurrogateDensity
    from styne.statistics.gaussian import GaussianDensity
    localisedDensity = LocalisedSurrogateDensity(
        1.0, 1.0, GaussianDensity(IIDCovarianceMatrix(1, 1.0), Vector(np.zeros(1)))
    )
    factory.target = localisedDensity
    localisedSampler = factory.create()
    
    localisedMeasure1 = LocalisedSurrogateTransitionMeasure(localisedSampler, 1)
    localisedMeasure2 = LocalisedSurrogateTransitionMeasure(localisedSampler, 1)
    assert localisedMeasure1.initialMeasure is not localisedMeasure2.initialMeasure


def test_acceptance_rate_equivalence():
    diagnostics = AcceptanceRateDiagnostics(window=3)
    decisions = [1, 0, 1, 1, 0]
    for decision in decisions:
        outcome = TransitionData.ACCEPTED if decision == 1 else TransitionData.REJECTED
        diagnostics.process(TransitionData(None, None, outcome))
        
    assert diagnostics.global_acceptance_rate() == 3 / 5
    assert diagnostics.rolling_acceptance_rate() == 2 / 3
    assert len(diagnostics._recent) <= 3


def test_manuscript_pattern_smoke_test():
    from styne.statistics.conditional import MetropolisWithinGibbsConditional
    from styne.statistics.bayes import HierarchicalBayes
    
    dimension = 1
    targetMeasure = Gaussian(IIDCovarianceMatrix(dimension, 1.0), Vector(np.zeros(dimension)))
    targetDensity = RadonNikodym(
        targetMeasure,
        targetMeasure.density
    )
    surrogateMeasure = Gaussian(IIDCovarianceMatrix(dimension, 2.0), Vector(np.zeros(dimension)))
    surrogateDensity = RadonNikodym(
        surrogateMeasure,
        surrogateMeasure.density
    )
    
    latentFactory = DARTFactory(root="mrw")
    latentFactory.target = targetDensity
    latentFactory.surrogate = [surrogateDensity]
    latentFactory.nChain = [2]
    latentFactory.regularisation = [0.1]
    latentFactory.subDiagnostics = PersistentAcceptanceRateDiagnostics
    latentFactory.root.proposalCovariance = IIDCovarianceMatrix(dimension, 1.0)
    latentMCMC = latentFactory.create()
    
    latentTransition = MetropolisWithinGibbsConditional(
        latentMCMC, blockIdx=0, nSteps=1
    )
    
    hyperFactory = MRWFactory()
    hyperMeasure = Gaussian(IIDCovarianceMatrix(dimension, 1.0), Vector(np.zeros(dimension)))
    hyperFactory.target = RadonNikodym(
        hyperMeasure,
        hyperMeasure.density
    )
    hyperFactory.proposalCovariance = IIDCovarianceMatrix(dimension, 1.0)
    hyperMCMC = hyperFactory.create()
    hyperTransition = MetropolisWithinGibbsConditional(
        hyperMCMC, blockIdx=1, nSteps=1
    )
    
    hierarchicalBayes = HierarchicalBayes(
        [latentTransition, hyperTransition], root=targetDensity.reference
    )
    
    builder = GibbsBuilder()
    builder.model = hierarchicalBayes
    sampler = builder.build()
    
    initialState = BlockParameter([
        Vector(np.zeros(dimension)), Vector(np.zeros(dimension))
    ])
    
    for n, state in enumerate(sampler.stream_run(5, initialState, progress=True)):
        pass
        
    acceptanceRate = latentMCMC.subsamplers[0].diagnostics.global_acceptance_rate()
    assert 0.0 <= acceptanceRate <= 1.0


def test_default_path_regression():
    from styne.mcmc.diagnostics import AcceptanceRateDiagnostics
    from tests.testSetup import GaussianTargetDensity
    targetDensity = GaussianTargetDensity(Vector(np.zeros(2)), np.eye(2))
    surrogateDensity = GaussianTargetDensity(Vector(np.zeros(2)), np.eye(2))

    factory = DARTFactory(root="mrw")
    factory.target = targetDensity
    factory.surrogate = [surrogateDensity]
    factory.nChain = [2]
    factory.regularisation = [0.1]
    factory.root.proposalCovariance = IIDCovarianceMatrix(2, 1.0)
    
    mainChain = factory.create()
    initialState = Vector(np.zeros(2))
    
    mainChain.clear()
    mainChain.run(2, initialState)

    factory.subDiagnostics = AcceptanceRateDiagnostics
    mainChain2 = factory.create()
    mainChain2.clear()
    mainChain2.run(2, initialState)


def test_reset_universal():
    from styne.mcmc.diagnostics import DummyDiagnostics, AcceptanceRateDiagnostics, FullDiagnostics
    DummyDiagnostics().reset()
    AcceptanceRateDiagnostics().reset()
    FullDiagnostics().reset()
