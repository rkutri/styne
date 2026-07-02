from abc import ABC, abstractmethod

from numpy import isnan, logaddexp


class AcceptanceProbability(ABC):
    """
    Interface for mapping a log Metropolis-Hastings ratio to a log
    acceptance probability.

    Notes
    -----
    Subclasses implement `log_probability`, already fully documented there,
    no separate note needed here.
    """

    @abstractmethod
    def log_probability(self, logMHRatio: float) -> float:
        """
        Map a log MH ratio to a log acceptance probability in (-inf, 0].

        Parameters
        ----------
        logMHRatio : float
            Log of the Metropolis-Hastings ratio.

        Returns
        -------
        float
            Log acceptance probability. Always in (-inf, 0].
        """
        pass


class StandardAcceptance(AcceptanceProbability):
    """
    Standard Metropolis-Hastings acceptance, $\log \alpha(r) = \min(0, r)$.
    """

    def log_probability(self, logMHRatio: float) -> float:
        """
        Log acceptance probability for a given log MH ratio.

        Parameters
        ----------
        logMHRatio : float

        Returns
        -------
        float
            $\min(0, \text{logMHRatio})$.
        """
        if isnan(logMHRatio):
            return float('-inf')
        return min(0., float(logMHRatio))


class BarkerAcceptance(AcceptanceProbability):
    """
    Barker (1965) acceptance, $\log \alpha_B(r) = r - \text{logaddexp}(0, r)$.

    Satisfies detailed balance via $\alpha_B(r) / \alpha_B(-r) = \exp(r)$.
    Numerically stable for all $r$ via `numpy.logaddexp`.
    """

    def log_probability(self, logMHRatio: float) -> float:
        """
        Log acceptance probability for a given log MH ratio.

        Parameters
        ----------
        logMHRatio : float

        Returns
        -------
        float
        """
        if isnan(logMHRatio):
            return float('-inf')

        lr = float(logMHRatio)

        if lr == float('inf'):
            return 0.

        return lr - float(logaddexp(0., lr))
