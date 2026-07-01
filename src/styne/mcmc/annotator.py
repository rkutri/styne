"""Per-step chain annotation interface.

An 'Annotator' is attached to a 'MetropolisHastings' instance to record a
scalar value alongside every trajectory entry. Annotations are stored in
'Chain.annotations' in strict 1-to-1 correspondence with 'Chain.trajectory'.

Intended uses include tracking surrogate log-potential values for the
constant-bridge ratio estimator, energy monitoring, and adaptive variants
that observe per-step statistics.
"""

from abc import ABC, abstractmethod

from styne.parameter.parameter import Parameter
from styne.statistics.interface import DensityInterface


class Annotator(ABC):
    """Interface for per-step chain annotation.

    Subclasses implement 'annotate', which maps a chain state to a scalar
    that is stored alongside the trajectory entry for that step.
    """

    @abstractmethod
    def annotate(self, state: Parameter) -> float:
        """Return the scalar annotation for 'state'.

        Parameters
        ----------
        state : Parameter
            The state accepted or rejected at the current step.

        Returns
        -------
        float
            Scalar annotation to be stored in 'Chain.annotations'.
        """
        ...
