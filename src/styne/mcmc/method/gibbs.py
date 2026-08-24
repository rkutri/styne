from abc import abstractmethod
from typing import Optional
from numpy.random import Generator

from styne.mcmc.sampler import MCMCSampler
from styne.mcmc.chain import GibbsChain
from styne.parameter.parameter import Parameter
from styne.parameter.block import BlockParameter


class GibbsSampler(MCMCSampler):
    """
    Template for Gibbs-type samplers over a block-structured state space.

    Implements the sweep loop, for each block in sequence, draw a new value
    from the block's conditional measure (using the most recently updated
    values of all other blocks), then record the full compound state.

    Subclasses implement `_sample_block` to define how each block is drawn.

    Parameters
    ----------
    nBlocks : int
        Number of blocks in the compound state.
    rng : Generator, optional
    """
    name = "Gibbs"

    def __init__(self, nBlocks: int, rng: Optional[Generator] = None):
        super().__init__(rng=rng)
        self._nBlocks = nBlocks
        self._chain = GibbsChain(nBlocks)

    @property
    def chain(self) -> GibbsChain:
        return self._chain

    def _validate_initial(self, initialState):
        if not isinstance(initialState, BlockParameter):
            raise ValueError(
                f"GibbsSampler requires a BlockParameter initial state. Got {type(initialState)}."
            )

    def _append_initial(self, initialState):
        if self._storeChain:
            self._chain.append(
                [initialState.block(i).coordinate for i in range(self._nBlocks)]
            )


    def _iterate(self) -> BlockParameter:
        state = self._lastState.clone()
        for i in range(self._nBlocks):
            newBlock = self._sample_block(i, state)
            state.block(i).coordinate = newBlock.coordinate
        if self._storeChain:
            self._chain.append(
                [state.block(i).coordinate for i in range(self._nBlocks)]
            )
        return state

    def clear(self):
        super().clear()
        self._chain.clear()

    @abstractmethod
    def _sample_block(self, idx: int, state: BlockParameter) -> Parameter:
        """Draw the idx-th block conditional on the current compound state."""
        ...


class BlockGibbs(GibbsSampler):
    """
    Gibbs sampler for a HierarchicalBayes model.

    Cycles through each block, conditioning on the current values of all
    other blocks, and delegates the draw to the conditional measure returned
    by the model. Agnostic about how `generate_realisation` is implemented,
    exact draws and Metropolis-within-Gibbs steps are both transparent here.

    Parameters
    ----------
    model : HierarchicalBayes
        Duck-typed, not enforced by `isinstance`. Only `nBlocks` and
        `conditional(idx, state)` are actually used.
    rng : Generator, optional
    """
    name = "BlockGibbs"

    def __init__(self, model, rng: Optional[Generator] = None):
        super().__init__(model.nBlocks, rng=rng)
        self._model = model

    def _sample_block(self, idx: int, state: BlockParameter) -> Parameter:
        return self._model.conditional(idx, state).generate_realisation(rng=self._rng)


class GibbsBuilder:
    """Builder for BlockGibbs samplers."""

    def __init__(self):
        self._model = None
        self._rng = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        self._model = m

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, r):
        self._rng = r

    def build(self) -> BlockGibbs:
        """
        Construct the `BlockGibbs` sampler from the configured model and rng.

        Returns
        -------
        BlockGibbs

        Raises
        ------
        ValueError
            If `model` hasn't been set.
        """
        if self._model is None:
            raise ValueError("Bayesian model not set.")
        return BlockGibbs(self._model, rng=self._rng)
