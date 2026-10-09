from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Optional, Union, Sequence, TYPE_CHECKING

from numpy.random import Generator, SeedSequence

from styne.backend import get_backend
from styne.parameter.parameter import Parameter

if TYPE_CHECKING:
    from styne.statistics.interface import DensityInterface

SeedType = Union[int, Sequence[int], SeedSequence, None]


class ProbabilityMeasure(ABC):
    """
    Abstract base class for probability measures that can be sampled.

    New implementations override ``sample(randomState)`` and return both a
    sample and the next random state. ``draw`` and ``generate_realisation``
    remain NumPy-compatible convenience adapters for existing callers.
    """

    def sample(self, randomState) -> tuple[Parameter, object]:
        """Return a sample and the explicitly propagated random state."""
        if type(self).draw is ProbabilityMeasure.draw:
            raise NotImplementedError(
                f"{type(self).__name__} must implement sample or draw."
            )
        return self.draw(randomState), randomState

    def draw(self, rng: Generator) -> Parameter:
        """Compatibility adapter returning only the sampled parameter."""
        return self.sample(rng)[0]

    def generate_realisation(
        self, *, randomState=None, rng: Optional[Generator] = None,
        seed: SeedType = None, backend: str = "numpy",
    ) -> Parameter:
        """
        Draw a sample, optionally constructing an RNG from a seed.

        Exactly one of ``randomState``, ``rng``, or ``seed`` may be provided.
        ``randomState`` is the backend-native state for explicit propagation;
        use ``sample`` when the next state is needed. ``rng`` is retained as
        the NumPy compatibility spelling. If no state is supplied, ``backend``
        constructs a fresh backend-specific state from ``seed``.

        Parameters
        ----------
        randomState : object, optional
            Backend-native random state.
        rng : Generator, optional
            Existing random generator to use.
        seed : SeedType, optional
            Seed for a new generator (cannot be combined with `rng`).

        Returns
        -------
        Parameter
            A sample from this measure.
        """
        supplied = sum(value is not None for value in (randomState, rng, seed))
        if supplied > 1:
            raise ValueError("Pass only one of randomState, rng, or seed.")

        if randomState is None:
            if rng is not None:
                randomState = rng
            else:
                randomState = get_backend(backend).random_state(seed)

        return self.sample(randomState)[0]


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
    on the remaining blocks. ``condition`` returns an independent conditioned
    measure; ``condition_on`` remains the legacy in-place hook.
    """

    @property
    @abstractmethod
    def blockDimension(self) -> int:
        """Dimension of the block this conditional is defined over."""
        ...

    @abstractmethod
    def condition_on(self, state) -> None:
        """
        Set the conditioning context to `state` in place.

        `state` is the full joint BlockParameter. After this call, `density`
        reflects the conditional distribution of this block given all others
        fixed to their values in `state`. Which block this conditional covers
        is determined by the concrete implementation.
        """
        ...

    def condition(self, state) -> "ConditionalMeasure":
        """Return an independent measure conditioned on ``state``."""
        conditioned = copy.deepcopy(self)
        conditioned.condition_on(state)
        return conditioned
