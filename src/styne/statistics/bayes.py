from typing import List

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

    Holds one ConditionalMeasure per non-root block and an unconditional
    root measure. Implements DensityInterface over the full joint
    BlockParameter and provides access to configured block conditionals for
    Gibbs/MwG sampling.

    Notes
    -----
    `evaluate_log` sums the root log-density and each conditional's
    log-density; the result is unnormalised. Correctness holds only when
    conditionals contribute distinct, non-overlapping factors, the
    standard 2-block case.

    Parameters
    ----------
    conditionals : List[ConditionalMeasure]
        One conditional per non-root block, in block order.
    root : AbsolutelyContinuousProbabilityMeasure
        Unconditional measure for the root block.
    """

    def __init__(
        self,
        conditionals: List[ConditionalMeasure],
        root: AbsolutelyContinuousProbabilityMeasure,
    ):
        self._conditionals = conditionals
        self._root = root

    @property
    def domainType(self):
        from styne.parameter.block import BlockParameter
        return BlockParameter

    @property
    def domainDimension(self) -> int:
        return (sum(c.blockDimension for c in self._conditionals)
                + self._root.density.domainDimension)

    @property
    def nBlocks(self) -> int:
        return len(self._conditionals)

    @property
    def root(self) -> AbsolutelyContinuousProbabilityMeasure:
        return self._root

    def conditional(self, idx: int, state) -> ConditionalMeasure:
        """Condition the idx-th block measure on `state` and return it."""
        return self._conditionals[idx].condition(state)

    def evaluate_log(self, state) -> float:
        """
        Unnormalised full joint log-density.

        Sums each conditional's unnormalised log-density at its block and
        the root log-density. Correct when each conditional contributes a
        distinct non-overlapping factor — the standard 2-block case.
        """
        rootBlock = state.block(state.nBlocks - 1)
        logp = self._root.density.evaluate_log(rootBlock)
        for i, cond in enumerate(self._conditionals):
            conditioned = cond.condition(state)
            logp += conditioned.density.evaluate_log(state.block(i))
        return logp


class HierarchicalBayesModelBuilder:
    """
    Fluent builder for `HierarchicalBayes` models.

    Chain `set_root` and `add_conditional` calls, then `build` to construct
    the model.
    """

    def __init__(self):
        self._conditionals = []
        self._root = None

    def set_root(
        self, root: AbsolutelyContinuousProbabilityMeasure
    ) -> "HierarchicalBayesModelBuilder":
        self._root = root
        return self

    def add_conditional(
        self, conditional: ConditionalMeasure
    ) -> "HierarchicalBayesModelBuilder":
        self._conditionals.append(conditional)
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
        return HierarchicalBayes(list(self._conditionals), self._root)
