from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Sequence, TYPE_CHECKING

from numpy.random import Generator, SeedSequence

from styne.parameter.parameter import Parameter

if TYPE_CHECKING:
    from styne.statistics.interface import DensityInterface

SeedType = Union[int, Sequence[int], SeedSequence, None]


class ProbabilityMeasure(ABC):
    """
    Abstract base class for probability measures that can be sampled.

    Subclasses implement `draw`; the public interface is
    `generate_realisation`, which handles RNG construction.
    """

    @abstractmethod
    def draw(self, rng: Generator) -> Parameter:
        """
        Draw a sample using the provided random number generator.

        Parameters
        ----------
        rng : Generator
            NumPy random generator instance.

        Returns
        -------
        Parameter
            A sample from this measure.
        """
        ...

    def generate_realisation(
        self, *, rng: Optional[Generator] = None, seed: SeedType = None
    ) -> Parameter:
        """
        Draw a sample, optionally constructing an RNG from a seed.

        Exactly one of `rng` or `seed` may be provided. If neither is
        given, a fresh default RNG is created for this single draw.
        For reproducible sequences of draws, pass a persistent `rng`.

        Parameters
        ----------
        rng : Generator, optional
            Existing random generator to use.
        seed : SeedType, optional
            Seed for a new generator (cannot be combined with `rng`).

        Returns
        -------
        Parameter
            A sample from this measure.
        """
        if rng is not None and seed is not None:
            raise ValueError("Pass either rng or seed, not both.")

        if rng is None:
            rng = np.random.default_rng(seed)

        return self.draw(rng)


class AbsolutelyContinuousProbabilityMeasure(ProbabilityMeasure):
    """
    A probability measure that possesses a density with respect to
    Lebesgue measure.
    """

    @property
    @abstractmethod
    def density(self) -> DensityInterface:
        ...


class ConditionalMeasure(AbsolutelyContinuousProbabilityMeasure):
    """
    A probability measure for one block of a hierarchical model, conditioned
    on the remaining blocks. Call `condition_on` to fix the conditioning
    context before drawing or evaluating.
    """

    @property
    @abstractmethod
    def blockDimension(self) -> int:
        """Dimension of the block this conditional is defined over."""
        ...

    @abstractmethod
    def condition_on(self, state) -> None:
        """
        Set the conditioning context to `state`.

        `state` is the full joint BlockParameter. After this call, `density`
        reflects the conditional distribution of this block given all others
        fixed to their values in `state`. Which block this conditional covers
        is determined by the concrete implementation.
        """
        ...
