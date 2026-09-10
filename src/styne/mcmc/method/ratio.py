import copy
import numpy as np

from styne.backend import infer_backend
from styne.parameter.parameter import Parameter


def log_dot_product_weights(gamma, samples, x, z, spectralWeights=None):
    """Raw IS log-weights for the regularisation ratio."""
    backend = infer_backend(samples)
    mid = 0.5 * (x + z)
    diff = x - z
    if spectralWeights is not None:
        diff = spectralWeights**2 * diff
    return gamma * backend.namespace.sum((samples - mid) * diff, axis=-1)


class RatioEstimator:
    """
    Normalising constant ratio estimator for DART.

    Estimates log(N_z / N_x) from retained states of localised surrogate
    trajectories. Importance sampling and geometric bridge estimation require
    at least one state. The second-order cumulant approximation requires two
    states because it estimates a sample variance.

    Parameters
    ----------
    surrogateMeasure : LocalisedSurrogateTransitionMeasure
        Surrogate measure whose trajectories are used in estimation.
    burnin : int
        Number of leading non-terminal trajectory states to discard.
    thinning : int
        Keep every thinning-th state after burnin.
    type : str
        One of 'is', 'bridge', or 'cumulant'. Defaults to 'cumulant'.
    """

    def __init__(
        self,
        surrogateMeasure,
        burnin: int,
        thinning: int,
        type: str = 'cumulant'
    ):
        if type not in ('is', 'bridge', 'cumulant'):
            raise ValueError(
                f"type must be 'is', 'bridge', or 'cumulant', got '{type}'."
            )
        if not isinstance(burnin, int) or burnin < 0:
            raise ValueError(
                f"burnin must be a non-negative integer. Got {burnin}."
            )
        if not isinstance(thinning, int) or thinning < 1:
            raise ValueError(
                f"thinning must be a positive integer. Got {thinning}."
            )

        self._surrogateMeasure = surrogateMeasure
        self._burnin = burnin
        self._thinning = thinning
        self._type = type
        self._validate_configured_trajectory()

    @staticmethod
    def minimum_samples(estimatorType):
        """Return the retained-state requirement for an estimator."""
        return 2 if estimatorType == 'cumulant' else 1

    @staticmethod
    def retained_sample_count(availableStates, burnin, thinning):
        """Count states retained after burn-in and thinning."""
        remaining = max(0, availableStates - burnin)
        return (remaining + thinning - 1) // thinning

    @property
    def estimatorType(self):
        return self._type

    @property
    def minimumSamples(self):
        return self.minimum_samples(self._type)

    @property
    def requires_proposal_trajectory(self):
        return self._type == 'bridge'

    def _validate_configured_trajectory(self):
        subchainLength = getattr(
            self._surrogateMeasure, "subchainLength", None
        )
        if not isinstance(subchainLength, int) or subchainLength == 0:
            return
        retained = self.retained_sample_count(
            subchainLength, self._burnin, self._thinning
        )
        if retained < self.minimumSamples:
            raise ValueError(
                f"{self._type} ratio estimation requires at least "
                f"{self.minimumSamples} retained state(s); configuration "
                f"retains {retained}."
            )

    def with_surrogate_measure(self, surrogateMeasure):
        """Return the estimator bound to ``surrogateMeasure``."""
        result = copy.copy(self)
        result._surrogateMeasure = surrogateMeasure
        result._validate_configured_trajectory()
        return result

    def _as_trajectory(self, trajectory, backend, metadata, dimension):
        if isinstance(trajectory, list):
            if not trajectory:
                return backend.zeros(
                    (0, dimension),
                    dtype=metadata.dtype,
                    device=metadata.device,
                )
            return backend.asarray(
                np.stack(trajectory),
                dtype=metadata.dtype,
                device=metadata.device,
            )
        return trajectory

    def _retained_samples(
            self, trajectory, backend, metadata, dimension, label):
        trajectory = self._as_trajectory(
            trajectory, backend, metadata, dimension
        )
        samples = trajectory[:-1][self._burnin::self._thinning]
        if samples.shape[0] < self.minimumSamples:
            raise ValueError(
                f"{self._type} ratio estimation requires at least "
                f"{self.minimumSamples} retained state(s) in {label}; got "
                f"{samples.shape[0]}."
            )
        return samples

    @staticmethod
    def _concrete_truth(predicate):
        try:
            return bool(np.asarray(predicate))
        except Exception:
            return None

    def log_ratio_estimate(
        self, state: Parameter, proposal: Parameter, trajectory=None,
        proposalTrajectory=None,
    ):
        """Estimate log(N_z / N_x) from explicit retained trajectories."""
        backend = infer_backend(state.coordinate)
        metadata = backend.metadata(state.coordinate)
        zero = backend.asarray(
            0., dtype=metadata.dtype, device=metadata.device
        )
        coincident = backend.namespace.all(
            state.coordinate == proposal.coordinate
        )
        if self._concrete_truth(coincident):
            return zero

        trajectory = (
            self._surrogateMeasure.chain.trajectory
            if trajectory is None else trajectory
        )
        samples = self._retained_samples(
            trajectory,
            backend,
            metadata,
            state.coordinate.shape[-1],
            "the state trajectory",
        )

        spectralWeights = self._surrogateMeasure.density.spectralWeights
        weights = log_dot_product_weights(
            self._surrogateMeasure.regularisation,
            samples,
            state.coordinate,
            proposal.coordinate,
            spectralWeights,
        )
        nSamples = weights.shape[0]
        logSamples = backend.namespace.log(backend.asarray(
            nSamples, dtype=metadata.dtype, device=metadata.device
        ))

        if self._type == 'is':
            estimate = backend.namespace.logsumexp(-weights) - logSamples
            return backend.namespace.where(coincident, zero, estimate)

        if self._type == 'cumulant':
            mean = backend.namespace.mean(weights)
            variance = backend.namespace.sum(
                (weights - mean) ** 2
            ) / (nSamples - 1)
            estimate = -mean + 0.5 * variance
            return backend.namespace.where(coincident, zero, estimate)

        if proposalTrajectory is None:
            raise ValueError(
                "bridge ratio estimation requires an explicit proposal "
                "trajectory."
            )
        proposalSamples = self._retained_samples(
            proposalTrajectory,
            backend,
            metadata,
            proposal.coordinate.shape[-1],
            "the proposal trajectory",
        )
        proposalWeights = log_dot_product_weights(
            self._surrogateMeasure.regularisation,
            proposalSamples,
            state.coordinate,
            proposal.coordinate,
            spectralWeights,
        )
        logStateEstimate = (
            backend.namespace.logsumexp(-0.5 * weights) - logSamples
        )
        logProposalSamples = backend.namespace.log(backend.asarray(
            proposalWeights.shape[0],
            dtype=metadata.dtype,
            device=metadata.device,
        ))
        logProposalEstimate = (
            backend.namespace.logsumexp(0.5 * proposalWeights)
            - logProposalSamples
        )
        estimate = logStateEstimate - logProposalEstimate
        return backend.namespace.where(coincident, zero, estimate)
