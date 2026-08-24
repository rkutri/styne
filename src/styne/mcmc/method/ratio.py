from numpy import ndarray, asarray, log
from typing import Optional, List
from scipy.special import logsumexp
from styne.parameter.parameter import Parameter


def log_dot_product_weights(
    gamma: float, samples: ndarray, x: ndarray, z: ndarray,
    spectralWeights: ndarray = None
) -> ndarray:
    """Raw IS log-weights for the regularisation ratio."""
    mid = 0.5 * (x + z)
    diff = x - z
    if spectralWeights is not None:
        diff = spectralWeights**2 * diff
    return gamma * ((samples - mid) @ diff)


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

    def log_ratio_estimate(
        self, state: Parameter, proposal: Parameter, trajectory=None
    ) -> float:
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
        float
            Estimated log(N_z / N_x).
        """
        traj = (
            self._surrogateMeasure.chain.trajectory
            if trajectory is None else trajectory
        )
        # Exclude the accepted proposal ψ_n from the ratio estimate. Its weight
        # is deterministic given z = ψ_n, introducing a conditional bias that
        # correlates with the proposal distance ‖z - x‖.
        trimmed = traj[:-1] if len(traj) > 1 else traj
        sub = trimmed[self._burnin::self._thinning]

        if not sub:
            return 0.

        sw = self._surrogateMeasure.density.spectralWeights
        samplesX = asarray(sub)
        w = log_dot_product_weights(
            self._surrogateMeasure.regularisation, samplesX,
            state.coordinate, proposal.coordinate, sw
        )

        if self._type == 'is':
            return float(logsumexp(-w) - log(len(w)))

        if self._type == 'cumulant':
            if len(w) == 1:
                return float(-w.mean())
            return float(-w.mean() + 0.5 * w.var(ddof=1))

        # --- Geometric bridge ---

        # The auxiliary Π_z chain reuses the surrogate measure's configured
        # subchain length. To use a different length for the bridge, configure
        # the measure accordingly before constructing the estimator.
        logEstX = float(logsumexp(-0.5 * w) - log(len(w)))

        xLocation = self._surrogateMeasure.location
        self._surrogateMeasure.location = proposal
        self._surrogateMeasure.generate_realisation()

        trajZ = self._surrogateMeasure.chain.trajectory
        # Same trimming as for the Π_x trajectory: exclude the terminal state.
        trimmedZ = trajZ[:-1] if len(trajZ) > 1 else trajZ
        subZ = trimmedZ[self._burnin::self._thinning]

        if not subZ:
            # Safe to restore location without explicitly saving/restoring chain state.
            # The surrogate measure's generate_realisation() internally calls run()
            # with a fresh initial state derived from the new location, ensuring safe
            # re-initialisation.
            self._surrogateMeasure.location = xLocation
            return logEstX

        samplesZ = asarray(subZ)
        wZ = log_dot_product_weights(
            self._surrogateMeasure.regularisation, samplesZ,
            state.coordinate, proposal.coordinate, sw
        )
        logEstZ = float(logsumexp(0.5 * wZ) - log(len(wZ)))

        # Safe to restore location without explicitly saving/restoring chain state.
        # The surrogate measure's generate_realisation() internally calls run()
        # with a fresh initial state derived from the new location, ensuring safe
        # re-initialisation.
        self._surrogateMeasure.location = xLocation

        return logEstX - logEstZ
