import numpy as np

from styne.backend import infer_backend
from styne.mcmc.localised import apply_metric
from styne.parameter.parameter import Parameter


def log_dot_product_weights(penaltyMetric, samples, x, z, spectralWeights=None):
    """Raw IS log-weights (v - (x + z)/2)^T B (x - z) of the ratio.

    They are the penalty differences between the localisations at z and x,
    where a number gamma as the penalty metric B stands for gamma I.
    """
    backend = infer_backend(samples)
    mid = 0.5 * (x + z)
    difference = x - z
    if spectralWeights is not None:
        difference = spectralWeights ** 2 * difference
    return backend.namespace.sum(
        (samples - mid) * apply_metric(penaltyMetric, difference), axis=-1
    )


class NormalisingRatioEstimator:
    """
    Normalising-ratio estimator for DART.

    Estimates log(N_z / N_x) from retained states of the root chains.
    Importance sampling and geometric bridge estimation require at least one
    state. The Gaussian estimator, a second-order cumulant approximation,
    requires two states because it estimates a sample variance.

    Parameters
    ----------
    burnin : int
        Number of leading non-terminal trajectory states to discard.
    thinning : int
        Keep every thinning-th state after burnin.
    type : str
        One of 'importance', 'bridge', or 'gaussian'. Defaults to 'gaussian'.
    """

    def __init__(
        self,
        burnin: int = 0,
        thinning: int = 1,
        type: str = 'gaussian'
    ):
        type = {'is': 'importance', 'cumulant': 'gaussian'}.get(type, type)
        if type not in ('importance', 'bridge', 'gaussian'):
            raise ValueError(
                "type must be 'importance', 'bridge', or 'gaussian', "
                f"got '{type}'."
            )
        if not isinstance(burnin, int) or burnin < 0:
            raise ValueError(
                f"burnin must be a non-negative integer. Got {burnin}."
            )
        if not isinstance(thinning, int) or thinning < 1:
            raise ValueError(
                f"thinning must be a positive integer. Got {thinning}."
            )

        self._burnin = burnin
        self._thinning = thinning
        self._type = type

    @staticmethod
    def minimum_samples(estimatorType):
        """Return the retained-state requirement for an estimator."""
        return 2 if estimatorType in ('gaussian', 'cumulant') else 1

    @staticmethod
    def retained_sample_count(availableStates, burnin, thinning):
        """Count states retained after burn-in and thinning."""
        remaining = max(0, availableStates - burnin)
        return (remaining + thinning - 1) // thinning

    @staticmethod
    def default_burnin(rootLength):
        """States discarded from a root run of ``rootLength`` by default.

        A quarter of the non-terminal states, rounded down, so a run of 20
        discards five and a run of 10 discards two.
        """
        return rootLength // 4

    @property
    def estimatorType(self):
        return self._type

    @property
    def burnin(self) -> int:
        """Leading non-terminal root states discarded before estimation."""
        return self._burnin

    @property
    def thinning(self) -> int:
        return self._thinning

    @property
    def minimumSamples(self):
        return self.minimum_samples(self._type)

    @property
    def requires_proposal_trajectory(self):
        return self._type == 'bridge'

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
        self, state: Parameter, proposal: Parameter, trajectory,
        proposalTrajectory=None, *, penaltyMetric, returnDiagnostics=False,
    ):
        """Estimate log(N_z / N_x) from explicit retained trajectories."""
        backend = infer_backend(state.coordinate)
        metadata = backend.metadata(state.coordinate)
        zero = backend.asarray(
            0., dtype=metadata.dtype, device=metadata.device
        )
        # Retained states run along the first axis and chains along the
        # batch axes, so every reduction below keeps the chains apart.
        coincident = backend.namespace.all(
            state.coordinate == proposal.coordinate, axis=-1
        )
        if self._concrete_truth(backend.namespace.all(coincident)):
            estimate = backend.namespace.where(coincident, zero, zero)
            diagnostics = {"logWeightMean": estimate,
                           "logWeightVariance": estimate}
            return (estimate, diagnostics) if returnDiagnostics else estimate

        samples = self._retained_samples(
            trajectory,
            backend,
            metadata,
            state.coordinate.shape[-1],
            "the state trajectory",
        )

        weights = log_dot_product_weights(
            penaltyMetric, samples, state.coordinate, proposal.coordinate
        )
        nSamples = weights.shape[0]
        logSamples = backend.namespace.log(backend.asarray(
            nSamples, dtype=metadata.dtype, device=metadata.device
        ))
        weightMean = backend.namespace.mean(weights, axis=0)
        weightVariance = backend.namespace.sum(
            (weights - weightMean) ** 2, axis=0
        ) / (nSamples - 1) if nSamples > 1 else zero

        def result(estimate):
            estimate = backend.namespace.where(coincident, zero, estimate)
            if not returnDiagnostics:
                return estimate
            diagnostics = {
                "logWeightMean": backend.namespace.where(
                    coincident, zero, weightMean),
                "logWeightVariance": backend.namespace.where(
                    coincident, zero, weightVariance),
            }
            return estimate, diagnostics

        if self._type == 'importance':
            estimate = backend.namespace.logsumexp(-weights, axis=0) - logSamples
            return result(estimate)

        if self._type == 'gaussian':
            estimate = -weightMean + 0.5 * weightVariance
            return result(estimate)

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
            penaltyMetric, proposalSamples, state.coordinate, proposal.coordinate
        )
        logStateEstimate = (
            backend.namespace.logsumexp(-0.5 * weights, axis=0) - logSamples
        )
        logProposalSamples = backend.namespace.log(backend.asarray(
            proposalWeights.shape[0],
            dtype=metadata.dtype,
            device=metadata.device,
        ))
        logProposalEstimate = (
            backend.namespace.logsumexp(0.5 * proposalWeights, axis=0)
            - logProposalSamples
        )
        estimate = logStateEstimate - logProposalEstimate
        return result(estimate)


class RatioEstimator(NormalisingRatioEstimator):
    """Legacy constructor retaining only a metric snapshot and estimator settings.

    Trajectories are explicit arguments; an estimate never reads a sampler's
    stored history. DART supplies its current metric on every call.
    """

    def __init__(self, surrogateMeasure, burnin, thinning, type='gaussian'):
        super().__init__(burnin, thinning, type)
        from styne.statistics.covariance import CovarianceMatrix
        self._metric = getattr(surrogateMeasure, 'penaltyMetric', None)
        if not isinstance(self._metric, (int, float, np.number, CovarianceMatrix)):
            self._metric = surrogateMeasure.regularisation
            weights = getattr(surrogateMeasure.density, 'spectralWeights', None)
            if weights is not None:
                from styne.statistics.covariance import DiagonalCovarianceMatrix
                self._metric = DiagonalCovarianceMatrix(self._metric * weights ** 2)
        length = getattr(surrogateMeasure, 'subchainLength', None)
        if isinstance(length, int) and length > 0:
            retained = self.retained_sample_count(length, burnin, thinning)
            if retained < self.minimumSamples:
                raise ValueError(f"{self.estimatorType} requires at least "
                                 f"{self.minimumSamples} retained state(s); "
                                 f"configuration retains {retained}.")

    def log_ratio_estimate(self, state, proposal, trajectory, proposalTrajectory=None,
                           *, penaltyMetric=None, returnDiagnostics=False):
        return super().log_ratio_estimate(
            state, proposal, trajectory, proposalTrajectory,
            penaltyMetric=self._metric if penaltyMetric is None else penaltyMetric,
            returnDiagnostics=returnDiagnostics,
        )

    def with_surrogate_measure(self, surrogateMeasure):
        return type(self)(
            surrogateMeasure, self.burnin, self.thinning, self.estimatorType
        )
