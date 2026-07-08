from abc import ABC, abstractmethod

import numpy as np

from styne.mcmc.transition import TransitionData
from styne.statistics.welford import WelfordAccumulator


from collections import deque

class ChainDiagnostics(ABC):
    """
    Interface for accumulating statistics over an MCMC chain's transitions.

    Notes
    -----
    Subclasses implement `process`, `summary`, and `clear`. `reset` is
    concrete here and delegates to `clear` by default.
    """

    @abstractmethod
    def process(self, transitionData):
        """
        Update accumulated statistics with one transition.

        Parameters
        ----------
        transitionData : TransitionData
        """
        ...

    def reset(self):
        """Hard reset of accumulated state. Default delegates to clear()."""
        self.clear()

    @abstractmethod
    def summary(self) -> dict:
        """
        Current summary statistics.

        Returns
        -------
        dict
        """
        ...

    @abstractmethod
    def clear(self):
        """
        Discard all accumulated statistics.
        """
        ...


class DummyDiagnostics(ChainDiagnostics):
    def process(self, transitionData):
        return

    def summary(self) -> dict:
        return {}

    def clear(self):
        return


class AcceptanceRateDiagnostics(ChainDiagnostics):
    def __init__(self, window: int = 200):
        self._window = window
        self._recent = deque(maxlen=window)
        self._total = 0
        self._accepted = 0

    def rolling_acceptance_rate(self):
        if not self._recent:
            return 0.0
        return np.sum(self._recent) / len(self._recent)

    def global_acceptance_rate(self):
        return self._accepted / self._total if self._total else 0.0

    def process(self, transitionData):
        if transitionData.outcome == TransitionData.ACCEPTED:
            self._recent.append(1)
            self._accepted += 1
        elif transitionData.outcome == TransitionData.REJECTED:
            self._recent.append(0)
        else:
            raise ValueError("Invalid acceptance decision.")
        self._total += 1

    def summary(self) -> dict:
        return {"acc": self.rolling_acceptance_rate()}

    def clear(self):
        self._recent.clear()
        self._total = 0
        self._accepted = 0


class PersistentAcceptanceRateDiagnostics(AcceptanceRateDiagnostics):
    """
    AcceptanceRateDiagnostics that ignores clear() calls.

    Standard MCMC samplers call clear() at the start of each run(), which
    resets cumulative diagnostics. In multilevel methods like DART or MLDA,
    the subchain (coarse sampler) is run once for every proposal in the
    fine chain. Using a standard diagnostics object on the coarse sampler
    would therefore only capture the acceptance rate of the very last
    subchain run.

    This class prevents this by ignoring the clear() signal, allowing
    acceptance decisions to accumulate across many independent sampler
    runs. Use reset() to explicitly clear the history.
    """

    def clear(self):
        """Ignore the clear signal to preserve cumulative history."""
        pass

    def reset(self):
        """Explicitly clear all accumulated decisions."""
        super().clear()


class FullDiagnostics(ChainDiagnostics):

    def __init__(self):
        self._diagnostics = AcceptanceRateDiagnostics()
        self._accumulator = WelfordAccumulator()

    def global_acceptance_rate(self):
        return self._diagnostics.global_acceptance_rate()

    def marginal_variance(self):
        return self._accumulator.marginal_variance()

    def mean(self):
        return self._accumulator.mean()

    def process(self, transitionData):
        self._diagnostics.process(transitionData)
        self._accumulator.update(transitionData.state.coordinate)

    def summary(self) -> dict:
        data = self._diagnostics.summary()
        try:
            data["cond"] = self._accumulator.condition_number()
        except RuntimeError:
            data["cond"] = float('nan')
        return data

    def clear(self):
        self._diagnostics.clear()
        self._accumulator.reset()
