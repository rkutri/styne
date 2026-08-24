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

    Estimates log(N_z / N_x) from the samples of a localised surrogate chain using
    either one-sided importance sampling ('is'), a geometric bridge ('bridge'),
    or a second-order cumulant approximation ('cumulant').

    Parameters
    ----------
    surrogateMeasure : LocalisedSurrogateTransitionMeasure
        Surrogate measure whose chain trajectories are used in estimation.
    burnin : int
        Number of leading trajectory samples to discard.
    thinning : int
        Keep every thinning-th sample after burnin.
    type : str
        One of 'is', 'bridge', or 'cumulant'. Defaults to 'cumulant'.
        'cumulant' is a one-sided second-order CGF approximation;
        same cost as 'is', estimates the log-ratio directly, robust under large
        weight spread, but carries a bias that does not vanish with sample size.
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

        self._surrogateMeasure = surrogateMeasure
        self._burnin = burnin
        self._thinning = thinning
        self._type = type

    @property
    def requires_proposal_trajectory(self):
        return self._type == 'bridge'

    def log_ratio_estimate(
        self, state: Parameter, proposal: Parameter, trajectory=None,
        proposalTrajectory=None,
    ):
        """
        Estimate log(N_z / N_x) from the surrogate chain trajectory.

        Parameters
        ----------
        state : Parameter
            Current fine-level state x.
        proposal : Parameter
            Proposed fine-level state z.

        Returns
        -------
        Backend-native scalar estimate of log(N_z / N_x).
        """
        trajectory = (
            self._surrogateMeasure.chain.trajectory
            if trajectory is None else trajectory
        )
        # Exclude the accepted proposal ψ_n from the ratio estimate. Its weight
        # is deterministic given z = ψ_n, introducing a conditional bias that
        # correlates with the proposal distance ‖z - x‖.
        backend = infer_backend(state.coordinate)
        metadata = backend.metadata(state.coordinate)
        if isinstance(trajectory, list):
            trajectory = backend.asarray(
                np.stack(trajectory),
                dtype=metadata.dtype, device=metadata.device,
            )
        trimmed = trajectory[:-1]
        samples = trimmed[self._burnin::self._thinning]

        if samples.shape[0] == 0:
            return backend.asarray(0., dtype=metadata.dtype, device=metadata.device)

        sw = self._surrogateMeasure.density.spectralWeights
        w = log_dot_product_weights(
            self._surrogateMeasure.regularisation, samples,
            state.coordinate, proposal.coordinate, sw
        )
        nSamples = w.shape[0]
        logSamples = backend.namespace.log(
            backend.asarray(nSamples, dtype=metadata.dtype, device=metadata.device)
        )

        if self._type == 'is':
            return backend.namespace.logsumexp(-w) - logSamples

        if self._type == 'cumulant':
            mean = backend.namespace.mean(w)
            if nSamples == 1:
                return -mean
            variance = backend.namespace.sum((w - mean) ** 2) / (nSamples - 1)
            return -mean + 0.5 * variance

        # --- Geometric bridge ---

        # The auxiliary Π_z chain reuses the surrogate measure's configured
        # subchain length. To use a different length for the bridge, configure
        # the measure accordingly before constructing the estimator.
        logEstX = backend.namespace.logsumexp(-0.5 * w) - logSamples

        if proposalTrajectory is None:
            xLocation = self._surrogateMeasure.location
            self._surrogateMeasure.location = proposal
            self._surrogateMeasure.generate_realisation()
            trajectoryZ = self._surrogateMeasure.chain.trajectory
        else:
            trajectoryZ = proposalTrajectory
        # Same trimming as for the Π_x trajectory: exclude the terminal state.
        if isinstance(trajectoryZ, list):
            trajectoryZ = backend.asarray(
                np.stack(trajectoryZ),
                dtype=metadata.dtype, device=metadata.device,
            )
        samplesZ = trajectoryZ[:-1][self._burnin::self._thinning]

        if samplesZ.shape[0] == 0:
            # Safe to restore location without explicitly saving/restoring chain state.
            # The surrogate measure's generate_realisation() internally calls run()
            # with a fresh initial state derived from the new location, ensuring safe
            # re-initialisation.
            if proposalTrajectory is None:
                self._surrogateMeasure.location = xLocation
            return logEstX

        wZ = log_dot_product_weights(
            self._surrogateMeasure.regularisation, samplesZ,
            state.coordinate, proposal.coordinate, sw
        )
        logEstZ = backend.namespace.logsumexp(0.5 * wZ) - backend.namespace.log(
            backend.asarray(
                wZ.shape[0], dtype=metadata.dtype, device=metadata.device,
            )
        )

        # Safe to restore location without explicitly saving/restoring chain state.
        # The surrogate measure's generate_realisation() internally calls run()
        # with a fresh initial state derived from the new location, ensuring safe
        # re-initialisation.
        if proposalTrajectory is None:
            self._surrogateMeasure.location = xLocation

        return logEstX - logEstZ
