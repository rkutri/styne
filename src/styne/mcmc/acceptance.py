from abc import ABC, abstractmethod

from styne.backend import BackendInferenceError, get_backend, infer_backend


def acceptance_array(logMHRatio):
    """Return a log ratio as a native array with its inferred backend."""
    try:
        backend = infer_backend(logMHRatio)
    except BackendInferenceError:
        backend = get_backend("numpy")
        logMHRatio = backend.asarray(logMHRatio)
    return backend, logMHRatio


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
    def log_probability(self, logMHRatio):
        """
        Map a log MH ratio to a log acceptance probability in (-inf, 0].

        Parameters
        ----------
        logMHRatio : array-like
            Backend-native log of the Metropolis-Hastings ratio.

        Returns
        -------
        array
            Backend-native log acceptance probability in (-inf, 0].
        """
        pass


class StandardAcceptance(AcceptanceProbability):
    r"""
    Standard Metropolis-Hastings acceptance, $\log \alpha(r) = \min(0, r)$.
    """

    def log_probability(self, logMHRatio):
        r"""
        Log acceptance probability for a given log MH ratio.

        Parameters
        ----------
        logMHRatio : array-like

        Returns
        -------
        array
            $\min(0, \text{logMHRatio})$.
        """
        backend, logMHRatio = acceptance_array(logMHRatio)
        metadata = backend.metadata(logMHRatio)
        zero = backend.asarray(
            0., dtype=metadata.dtype, device=metadata.device
        )
        negativeInfinity = backend.asarray(
            float('-inf'), dtype=metadata.dtype, device=metadata.device
        )
        probability = backend.namespace.minimum(
            zero, logMHRatio
        )
        return backend.namespace.where(
            logMHRatio == logMHRatio, probability, negativeInfinity
        )


class BarkerAcceptance(AcceptanceProbability):
    r"""
    Barker (1965) acceptance, $\log \alpha_B(r) = r - \text{logaddexp}(0, r)$.

    Satisfies detailed balance via $\alpha_B(r) / \alpha_B(-r) = \exp(r)$.
    Numerically stable for all $r$ via backend-native 'logaddexp'.
    """

    def log_probability(self, logMHRatio):
        """
        Log acceptance probability for a given log MH ratio.

        Parameters
        ----------
        logMHRatio : array-like

        Returns
        -------
        array
        """
        backend, logMHRatio = acceptance_array(logMHRatio)
        metadata = backend.metadata(logMHRatio)
        zero = backend.asarray(
            0., dtype=metadata.dtype, device=metadata.device
        )
        negativeInfinity = backend.asarray(
            float('-inf'), dtype=metadata.dtype, device=metadata.device
        )
        probability = -backend.namespace.logaddexp(
            zero, -logMHRatio
        )
        return backend.namespace.where(
            logMHRatio == logMHRatio, probability, negativeInfinity
        )
