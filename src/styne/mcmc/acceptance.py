from abc import ABC, abstractmethod

from numpy import isnan, logaddexp


class AcceptanceProbability(ABC):

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
    Standard Metropolis-Hastings acceptance: log alpha(logRatio) = min(0, logRatio).
    """

    def log_probability(self, logMHRatio: float) -> float:
        if isnan(logMHRatio):
            return float('-inf')
        return min(0., float(logMHRatio))


class BarkerAcceptance(AcceptanceProbability):
    """
    Barker (1965) acceptance: log alpha_B(logRatio) = logRatio - logaddexp(0, logRatio).

    Satisfies detailed balance via alpha_B(logRatio) / alpha_B(-logRatio) = exp(logRatio).
    Numerically stable for all logRatio via numpy.logaddexp.
    """

    def log_probability(self, logMHRatio: float) -> float:

        if isnan(logMHRatio):
            return float('-inf')

        lr = float(logMHRatio)

        if lr == float('inf'):
            return 0.

        return lr - float(logaddexp(0., lr))
