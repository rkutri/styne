from styne.mcmc.metropolishastings import MetropolisHastings
from styne.parameter.block import BlockParameter
from styne.parameter.parameter import Parameter
from styne.statistics.measure import ConditionalMeasure


class MetropolisWithinGibbsConditional(ConditionalMeasure):
    """
    ConditionalMeasure that draws via a Metropolis-Hastings Markov chain.

    At each Gibbs sweep, `condition_on` prepares the target distribution
    for the block context. `draw` warm-starts the Markov chain from the 
    current state, runs `nSteps` transitions, and returns the last state.

    This implementation uses `continue_run` to maintain chain continuity
    across Gibbs sweeps, preserving cumulative diagnostics and supporting
    adaptive samplers.
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
                f"Sampler target {type(target).__name__} is missing 'condition_on'. "
                "Every block-conditional target must implement the conditioning protocol."
            )
        target.condition_on(state)
        self._currentBlock = state.block(self._blockIdx)

    def draw(self, _rng) -> Parameter:
        if _rng is not None:
            self._sampler._rng = _rng
        if not self._initialized:
            # First sweep: start fresh from the initial coordinate
            self._sampler.run(
                self._nSteps, self._currentBlock.clone()
            )
            self._initialized = True
        else:
            # Subsequent sweeps: continue the existing chain.
            # This preserves cumulative diagnostics (e.g. acceptance rate).
            self._sampler.continue_run(self._nSteps)

        return self._sampler.lastState

    def clear(self):
        """Reset the conditional and the underlying sampler."""
        self._initialized = False
        self._sampler.clear()
