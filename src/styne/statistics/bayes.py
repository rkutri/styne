from typing import List, Optional, Sequence

from styne.statistics.interface import BayesianModelInterface, DensityInterface
from styne.statistics.likelihood import LikelihoodInterface
from styne.statistics.measure import (
    AbsolutelyContinuousProbabilityMeasure,
    ConditionalMeasure,
)
from styne.statistics.radonnikodym import RadonNikodym


class UnnormalisedPosterior(RadonNikodym):
    """Unnormalised posterior density

    Parameters
    ----------
    prior : AbsolutelyContinuousProbabilityMeasure
        Prior measure (must be absolutely continuous).
    likelihood : LikelihoodInterface
        Likelihood function.
    """

    def __init__(self,
                 prior: AbsolutelyContinuousProbabilityMeasure,
                 likelihood: LikelihoodInterface):

        if not isinstance(prior, AbsolutelyContinuousProbabilityMeasure):
            raise TypeError(
                "Prior must be an AbsolutelyContinuousProbabilityMeasure. "
                f"Got {type(prior)} instead."
            )

        if not isinstance(likelihood, LikelihoodInterface):
            raise TypeError(
                "Posterior derivative must implement LikelihoodInterface. "
                f"Got {type(likelihood)} instead."
            )

        super().__init__(prior, likelihood)


class HierarchicalBayes(DensityInterface):
    """
    Composite hierarchical Bayesian model.

    Holds one conditional joint-density factor per non-root block and an
    unconditional root factor. Gibbs updates are configured separately, one
    full conditional or invariant transition per block.

    Notes
    -----
    `evaluate_log` sums only the generative factors. Update densities are not
    included, preventing double-counting when full conditionals share factors.

    Parameters
    ----------
    conditionals : List[ConditionalMeasure]
        One conditional per non-root block, in block order.
    root : AbsolutelyContinuousProbabilityMeasure
        Unconditional measure for the root block.
    updates : sequence of ConditionalMeasure, optional
        One invariant update for every block, including the root, in block
        order. Required for `BlockGibbs`, but not for density evaluation.
    blockNames : sequence of str, optional
        Names in the same order as the non-root blocks followed by the root.
    """

    def __init__(
        self,
        conditionals: List[ConditionalMeasure],
        root: AbsolutelyContinuousProbabilityMeasure,
        updates: Optional[Sequence[ConditionalMeasure]] = None,
        blockNames: Optional[Sequence[str]] = None,
    ):
        self._factors = tuple(conditionals)
        self._root = root
        self._updates = (
            tuple(updates) if updates is not None
            else (None,) * (len(self._factors) + 1)
        )
        if len(self._updates) != self.nBlocks:
            raise ValueError(
                f"Expected {self.nBlocks} block updates, got "
                f"{len(self._updates)}."
            )
        if blockNames is not None:
            if len(blockNames) != self.nBlocks:
                raise ValueError(
                    f"Expected {self.nBlocks} block names, got "
                    f"{len(blockNames)}."
                )
            if len(set(blockNames)) != len(blockNames):
                raise ValueError("Block names must be unique.")
        self._blockNames = (
            tuple(blockNames) if blockNames is not None else None
        )

    @property
    def domainType(self):
        from styne.parameter.block import BlockParameter
        return BlockParameter

    @property
    def domainDimension(self) -> int:
        return (sum(c.blockDimension for c in self._factors)
                + self._root.density.domainDimension)

    @property
    def nBlocks(self) -> int:
        return len(self._factors) + 1

    @property
    def blockDimensions(self) -> tuple[int, ...]:
        return tuple(
            factor.blockDimension for factor in self._factors
        ) + (self._root.density.domainDimension,)

    @property
    def blockNames(self) -> Optional[tuple[str, ...]]:
        return self._blockNames

    @property
    def hasCompleteUpdates(self) -> bool:
        return all(update is not None for update in self._updates)

    @property
    def root(self) -> AbsolutelyContinuousProbabilityMeasure:
        return self._root

    def conditional(self, idx: int, state) -> ConditionalMeasure:
        """Return the independently conditioned update for block ``idx``."""
        if not 0 <= idx < self.nBlocks:
            raise IndexError(f"Block index {idx} is out of range.")
        update = self._updates[idx]
        if update is None:
            raise RuntimeError(
                f"No invariant update is configured for block {idx}."
            )
        return update.condition(state)

    def validate_state(self, state) -> None:
        """Validate block count, dimensions, and configured name ordering."""
        from styne.parameter.block import BlockParameter

        if not isinstance(state, BlockParameter):
            raise TypeError("Hierarchical state must be a BlockParameter.")
        if state.nBlocks != self.nBlocks:
            raise ValueError(
                f"Expected {self.nBlocks} state blocks, got {state.nBlocks}."
            )
        dimensions = tuple(
            state.block(index).dimension for index in range(state.nBlocks)
        )
        if dimensions != self.blockDimensions:
            raise ValueError(
                f"Expected block dimensions {self.blockDimensions}, got "
                f"{dimensions}."
            )
        if self._blockNames is not None:
            expectedNames = {
                name: index for index, name in enumerate(self._blockNames)
            }
            if state.names != expectedNames:
                raise ValueError(
                    f"Expected block names {expectedNames}, got {state.names}."
                )

    def evaluate_log(self, state) -> float:
        """
        Unnormalised full joint log-density.

        Sums each non-root factor's unnormalised log-density at its block and
        the root log-density. Sampling updates do not contribute here.
        """
        self.validate_state(state)
        rootBlock = state.block(self.nBlocks - 1)
        logp = self._root.density.evaluate_log(rootBlock)
        for i, factor in enumerate(self._factors):
            conditioned = factor.condition(state)
            logp += conditioned.density.evaluate_log(state.block(i))
        return logp


class HierarchicalBayesModelBuilder:
    """
    Fluent builder for `HierarchicalBayes` models.

    Use `set_root` and `add_conditional` to define the joint factorisation.
    Add one invariant update per block with `add_update` when the model will
    be sampled by `BlockGibbs`, then call `build`.
    """

    def __init__(self):
        self._factors = []
        self._updates = []
        self._blockNames = []
        self._root = None
        self._rootName = None

    def set_root(
        self, root: AbsolutelyContinuousProbabilityMeasure, name: str = None
    ) -> "HierarchicalBayesModelBuilder":
        self._root = root
        self._rootName = name
        return self

    def add_conditional(
        self, conditional: ConditionalMeasure, name: str = None
    ) -> "HierarchicalBayesModelBuilder":
        self._factors.append(conditional)
        self._blockNames.append(name)
        return self

    def add_update(
        self, update: ConditionalMeasure
    ) -> "HierarchicalBayesModelBuilder":
        """Append one invariant block update in block order."""
        self._updates.append(update)
        return self

    def build(self) -> HierarchicalBayes:
        """
        Construct the `HierarchicalBayes` model from the configured root and
        conditionals.

        Returns
        -------
        HierarchicalBayes
        """
        if self._root is None:
            raise ValueError("Root measure not set.")
        names = [*self._blockNames, self._rootName]
        blockNames = None if all(name is None for name in names) else names
        if blockNames is not None and any(name is None for name in names):
            raise ValueError("Either name every block or leave all unnamed.")
        return HierarchicalBayes(
            list(self._factors),
            self._root,
            updates=list(self._updates) if self._updates else None,
            blockNames=blockNames,
        )
