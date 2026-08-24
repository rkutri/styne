"""Immutable numerical records exchanged by MCMC transitions."""
from dataclasses import dataclass
from typing import Any

from styne.parameter.parameter import Parameter

MISSING = object()


@dataclass(frozen=True)
class EvaluatedState:
    """A parameter and its backend-native log-density value.

    ``logDensity`` is ``None`` for a proposal that has not yet been evaluated.
    This permits proposal construction to remain independent of the target
    density while making evaluated values explicit once available.
    """

    parameter: Parameter
    logDensity: Any = None


@dataclass(frozen=True, init=False)
class TransitionData:
    """Immutable current/proposed states and transition outcome.

    ``current`` and ``proposed`` are :class:`EvaluatedState` instances.
    ``outcome`` and ``logAcceptanceProbability`` intentionally retain their
    originating backend arrays; transition implementations must not coerce
    them to Python scalars.

    The ``state`` and ``proposal`` constructor arguments and properties are
    retained temporarily for the pre-0.3 MCMC methods. They expose only the
    wrapped parameters and will be removed as those methods adopt
    ``current`` and ``proposed`` directly.
    """

    REJECTED = False
    ACCEPTED = True

    current: EvaluatedState
    proposed: EvaluatedState
    outcome: Any
    logAcceptanceProbability: Any
    auxiliary: Any

    def __init__(
            self, state=MISSING, proposal=MISSING, outcome=None,
            auxiliary=None, *, logAcceptanceProbability=None,
            current=MISSING, proposed=MISSING):
        if current is not MISSING:
            if state is not MISSING:
                raise TypeError("Specify current or state, not both.")
            state = current
        if proposed is not MISSING:
            if proposal is not MISSING:
                raise TypeError("Specify proposed or proposal, not both.")
            proposal = proposed
        if state is MISSING or proposal is MISSING:
            raise TypeError(
                "TransitionData requires current and proposed states."
            )

        object.__setattr__(self, "current", self._as_evaluated(state))
        object.__setattr__(self, "proposed", self._as_evaluated(proposal))
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(
            self, "logAcceptanceProbability", logAcceptanceProbability
        )
        object.__setattr__(self, "auxiliary", auxiliary)

    @staticmethod
    def _as_evaluated(value):
        if isinstance(value, EvaluatedState):
            return value
        if not isinstance(value, Parameter) and value is not None:
            raise TypeError("Transition states must be Parameter instances.")
        return EvaluatedState(value)

    @property
    def state(self):
        """Legacy parameter view of :attr:`current`."""
        return self.current.parameter

    @property
    def proposal(self):
        """Legacy parameter view of :attr:`proposed`."""
        return self.proposed.parameter
