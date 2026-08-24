import copy

from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.chain import Chain
from styne.parameter.block import BlockParameter
from styne.parameter.parameter import Parameter
from styne.statistics.measure import ConditionalMeasure


class MetropolisWithinGibbsConditional(ConditionalMeasure):
    """
    ConditionalMeasure that draws via a Metropolis-Hastings Markov chain.

    Each call to :meth:`condition` returns an isolated sampler whose target
    has been conditioned on the supplied joint state. ``sample`` then runs
    that sampler from the selected block and returns its propagated random
    state. No Gibbs sweep mutates the model's template conditional.
    """

    def __init__(
        self, sampler: MetropolisHastings, blockIdx: int, nSteps: int = 1
    ):
        self._sampler = sampler
        self._sampler.storeChain = False
        self._blockIdx = blockIdx
        self._nSteps = nSteps
        self._initialized = False
        self._currentBlock: Parameter = None

    @property
    def density(self):
        return self._sampler.target

    @property
    def blockDimension(self) -> int:
        return self._sampler.target.domainDimension

    def condition_on(self, state: BlockParameter) -> None:
        """
        Prepare the target density given the full joint MCMC state.
        Delegates to the sampler's target `condition_on` method.
        """
        target = self._sampler.target
        if not hasattr(target, "condition_on"):
            raise TypeError(
                f"Sampler target {type(target).__name__} is missing "
                "'condition_on'. Every block-conditional target must "
                "implement the conditioning protocol."
            )
        target.condition_on(state)
        self._currentBlock = state.block(self._blockIdx)

    def condition(self, state: BlockParameter):
        """Return a conditional with an isolated conditioned sampler."""
        target = self._sampler.target
        if not callable(getattr(target, "condition", None)):
            raise TypeError(
                f"Sampler target {type(target).__name__} is missing "
                "'condition(state)'."
            )

        conditioned = copy.copy(self)
        conditioned._sampler = copy.copy(self._sampler)
        conditioned._sampler._tgtDensity = target.condition(state)
        conditioned._sampler._chain = Chain()
        conditioned._sampler._diagnostics = copy.deepcopy(
            self._sampler.diagnostics
        )
        conditioned._sampler._lastState = None
        conditioned._sampler._runnerState = None
        conditioned._sampler._iteration = 0
        conditioned._currentBlock = state.block(self._blockIdx)
        conditioned._initialized = False
        return conditioned

    def sample(self, randomState):
        """Run the isolated conditional sampler with explicit RNG flow."""
        if self._currentBlock is None:
            raise RuntimeError("Condition the measure before sampling it.")
        self._sampler._rng = randomState
        self._sampler.run(self._nSteps, self._currentBlock)
        return self._sampler.lastState, self._sampler._rng

    def draw(self, _rng) -> Parameter:
        """Compatibility adapter returning only the sampled block."""
        return self.sample(_rng)[0]

    def clear(self):
        """Reset the conditional and the underlying sampler."""
        self._initialized = False
        self._sampler.clear()
