import numpy as np
from typing import Optional

from styne.statistics.measure import ConditionalMeasure
from styne.statistics.interface import DensityInterface, LikelihoodInterface
from styne.parameter.parameter import Parameter
from styne.parameter.block import BlockParameter
from styne.model.sglmm import SGLMM
from styne.gp.gaussianprocess import GaussianProcess

class SGLMMHyperConditionalDensity(DensityInterface):
    """
    Log-density target for Metropolis Random Walk on log-hyperparameters.

    Parameters
    ----------
    pcPrior : DensityInterface
        Prior on the (untransformed) hyperparameters.
    gp : GaussianProcess
        GP whose covariance function is rebuilt at each evaluation.
    model : SGLMM
        Forward model evaluated at the current latent state.
    likelihood : LikelihoodInterface
        Likelihood the data are evaluated under.
    latentState : Parameter
        Latent field value the forward map is prepared at.
    """
    def __init__(
            self,
            pcPrior: DensityInterface,
            gp: GaussianProcess,
            model: SGLMM,
            likelihood: LikelihoodInterface,
            latentState: Parameter
    ):
        self._pcPrior = pcPrior
        self._gp = gp
        self._model = model
        self._likelihood = likelihood
        self._latentState = latentState
        self._cachedProposalLog = None
        self._cachedProposalParams = None

    @property
    def domainType(self):
        from styne.parameter.vector import Vector
        return Vector

    @property
    def domainDimension(self) -> int:
        return 2

    def condition_on(self, state: BlockParameter) -> None:
        """
        Synchronize the GP with current hyperparameters from the joint state.

        This resolves the stale state issue where an MH rejection would otherwise 
        leave the shared GP object in the 'proposed' state. It also updates the 
        latent state reference to match the current Gibbs block.
        """
        self._latentState = state.block(0)
        hyperparameters = np.exp(np.asarray(state.block(1).coordinate).ravel())
        lengthScale, sigma = hyperparameters[0], hyperparameters[1]

        try:
            covarianceType = type(self._gp.covarianceFunction)
            smoothness = self._gp.covarianceFunction._smoothness
            self._gp.covarianceFunction = covarianceType(
                lengthScale, smoothness, sigma**2
            )
            self._gp.measure.covariance.scaling = 1.0
            # Defer evaluation to evaluate_log to avoid double-pass
        except np.linalg.LinAlgError:
            pass
            
        self._cachedProposalParams = None
        self._cachedProposalLog = None

    def evaluate_log(self, parameter: Parameter, normalised: bool = False) -> float:
        """
        Log-posterior density on log-hyperparameters `(log rho, log sigma)`,
        including the log-Jacobian of the log transform. Unnormalised overall,
        sums a properly normalised prior term (`pcPrior.evaluate_log`) with an
        unnormalised likelihood term (`response.log_likelihood`), so the total
        carries whatever constant the likelihood side drops. Fine for a
        Metropolis Random Walk target, not a properly normalised density.

        Parameters
        ----------
        parameter : Parameter
            Log-hyperparameters to evaluate at.
        normalised : bool, default False
            Unused, accepted for interface compatibility.

        Returns
        -------
        float
            `-inf` if the prior is zero, hyperparameters are extreme, or the
            covariance factorisation fails.
        """
        logHyperparameters = np.asarray(parameter.coordinate).ravel()
        hyperparameters = np.exp(logHyperparameters)
        lengthScale, sigma = hyperparameters[0], hyperparameters[1]

        from styne.parameter.vector import Vector
        hyperparameterVector = Vector(hyperparameters)
        logPrior = self._pcPrior.evaluate_log(hyperparameterVector)
        if logPrior == -np.inf or not np.all(np.abs(logHyperparameters) < 150.0):
            return -np.inf

        logJacobian = np.sum(logHyperparameters)
        try:
            covarianceType = type(self._gp.covarianceFunction)
            smoothness = self._gp.covarianceFunction._smoothness
            newCovariance = covarianceType(lengthScale, smoothness, sigma**2)
            self._gp.covarianceFunction = newCovariance
            self._gp.measure.covariance.scaling = 1.0

            linearPredictor = self._model(self._latentState)
        except np.linalg.LinAlgError:
            return -np.inf
            
        logLikelihood = self._likelihood.response.log_likelihood(
            self._likelihood.data.measurement, linearPredictor
        )
        
        logPriorContribution = 0.0

        totalLogProbability = (logPrior + logPriorContribution + 
                               logLikelihood + logJacobian)
        
        self._cachedProposalLog = totalLogProbability
        self._cachedProposalParams = (lengthScale, sigma)

        return float(totalLogProbability)

    def evaluate_log_gradient(self, parameter: Parameter) -> np.ndarray:
        logHyperparameters = np.asarray(parameter.coordinate).ravel()
        hyperparameters = np.exp(logHyperparameters)
        lengthScale, sigma = hyperparameters[0], hyperparameters[1]

        try:
            if self._cachedProposalParams != (lengthScale, sigma):
                covarianceType = type(self._gp.covarianceFunction)
                smoothness = self._gp.covarianceFunction._smoothness
                self._gp.covarianceFunction = covarianceType(
                    lengthScale, smoothness, sigma**2
                )
                self._gp.measure.covariance.scaling = 1.0

            linearPredictor = self._model(self._latentState)
        except np.linalg.LinAlgError:
            raise RuntimeError("Failed model evaluation due to singular covariance.")
            
        linearPredictorScore = self._likelihood.response.score(
            self._likelihood.data.measurement, linearPredictor
        )

        from styne.parameter.vector import Vector
        priorGradient = self._pcPrior.evaluate_log_gradient(
            Vector([lengthScale, sigma])
        )

        if self._gp.hasHyperGradient:
            hyperparameterGradients = self._gp.evaluate_hyper_gradient(
                self._latentState, linearPredictorScore
            )
            gradLogLengthScale = hyperparameterGradients.get('log_rho', 0.0) \
                + priorGradient[0] * lengthScale + 1.0
            gradLogSigma = hyperparameterGradients.get('log_sigma', 0.0) \
                + priorGradient[1] * sigma + 1.0
            return np.array([gradLogLengthScale, gradLogSigma])

        if not self._gp.hasLogLengthMultiplier:
            raise NotImplementedError(
                "Hyperparameter gradients not implemented for "
                f"{type(self._gp.expansion).__name__}."
            )

        lengthMultiplier = self._gp.compute_log_length_multiplier(
            smoothness, lengthScale
        )

        # latentCoordinate must be the coordinate vector
        latentCoordinate = self._latentState.coordinate \
            if isinstance(self._latentState, Parameter) else self._latentState

        gradLogLengthScale = np.dot(linearPredictorScore, lengthMultiplier * latentCoordinate) \
            + priorGradient[0] * lengthScale + 1.0
        gradLogSigma = np.dot(linearPredictorScore, latentCoordinate) \
            + priorGradient[1] * sigma + 1.0

        return np.array([gradLogLengthScale, gradLogSigma])


class SGLMMHyperConditional(ConditionalMeasure, DensityInterface):
    """
    Hyperparameter conditional block for SGLMM Gibbs sampling.

    Constructs a fresh `SGLMMHyperConditionalDensity` on every
    `condition_on` call, rebuilding the GP's covariance function at the
    proposed hyperparameters.

    Cannot be drawn from directly, sampling happens through `density` inside
    a Metropolis sub-chain elsewhere. `draw` unconditionally raises
    `NotImplementedError`. See flag 3 above.

    Parameters
    ----------
    pcPrior : DensityInterface
        Prior on the (untransformed) hyperparameters.
    gp : GaussianProcess
        GP whose covariance function is rebuilt at each `condition_on` call.
    model : SGLMM
        Forward model evaluated at the current latent state.
    likelihood : LikelihoodInterface
        Likelihood the data are evaluated under.
    hyperIdx : int, default 1
        Index of the hyperparameter block within the joint state.
    """

    def __init__(
            self,
            pcPrior: DensityInterface,
            gp: GaussianProcess,
            model: SGLMM,
            likelihood: LikelihoodInterface,
            hyperIdx: int = 1
    ):
        self._pcPrior = pcPrior
        self._gp = gp
        self._model = model
        self._likelihood = likelihood
        self._density = None
        self._hyperIdx = hyperIdx

    @property
    def blockDimension(self) -> int:
        return 2

    @property
    def density(self) -> DensityInterface:
        if self._density is None:
            raise RuntimeError("Must condition_on before accessing density.")
        return self._density

    @property
    def domainType(self):
        return self.density.domainType

    @property
    def domainDimension(self) -> int:
        return self.density.domainDimension

    def evaluate_log(self, parameter: Parameter) -> float:
        return self.density.evaluate_log(parameter)

    def condition_on(self, state: Parameter):
        """
        Rebuild the GP covariance at the proposed hyperparameters and construct
        a fresh `SGLMMHyperConditionalDensity` for `density`.

        Parameters
        ----------
        state : Parameter
            Full joint state. The latent block is `state.block(0)` unless
            `hyperIdx` is 1, in which case it's `state.block(1)`.
        """
        latentState = state.block(0) if self._hyperIdx == 1 else state.block(1)
        hyperState = state.block(self._hyperIdx)

        phi = hyperState.coordinate
        zeta = np.exp(phi)
        rho, sigma = zeta[0], zeta[1]

        covType = type(self._gp.covarianceFunction)
        smoothness = self._gp.covarianceFunction._smoothness
        self._gp.covarianceFunction = covType(rho, smoothness, sigma**2)
        self._gp.measure.covariance.scaling = 1.0

        self._density = SGLMMHyperConditionalDensity(
            self._pcPrior,
            self._gp,
            self._model,
            self._likelihood,
            latentState
        )

    def evaluate_log_gradient(self, parameter: Parameter) -> np.ndarray:
        return self.density.evaluate_log_gradient(parameter)

    def draw(self, rng):
        raise NotImplementedError("SGLMMHyperConditional cannot be drawn from.")

class SGLMMLatentConditional(ConditionalMeasure, DensityInterface):
    """
    Blocks latent evaluation until GP dependencies are synced to
    hyperparameters.

    Proxies unset attributes to `target` via `__getattr__`, so attributes
    like `derivative` and `reference` on a `RadonNikodym` target are
    accessible directly on this wrapper.

    Cannot be drawn from directly. `draw` unconditionally raises
    `NotImplementedError`. See flag 3 above.

    Parameters
    ----------
    target : DensityInterface
        The wrapped density, evaluated once GP dependencies are synced.
    gp : GaussianProcess
        GP kept in sync with the current hyperparameter block.
    coarseGP : GaussianProcess, optional
        Coarse-resolution companion GP, kept in sync alongside `gp` if given.
    partition : DNACoarseFinePartition, optional
        Coarse/fine partition used to extract the fine-block prior variance.
    finePrior : object, optional
        Prior on the fine block, updated in place when `partition` is given.
        Untyped in source, inferred from usage.
    hyperIdx : int, default 1
        Index of the hyperparameter block within the joint state.
    localisedDensity : object, optional
        Target with a `sync_weights` method, called with the coarse
        expansion's spectral weights when present. Untyped in source, inferred
        from usage.
    """

    def __init__(
            self,
            target: DensityInterface,
            gp: GaussianProcess,
            coarseGP: Optional[GaussianProcess] = None,
            partition=None,
            finePrior=None,
            hyperIdx: int = 1,
            localisedDensity=None
    ):
        self._target = target
        self._gp = gp
        self._coarseGP = coarseGP
        self._partition = partition
        self._finePrior = finePrior
        self._hyperIdx = hyperIdx
        self._localisedDensity = localisedDensity
        self._cachedHyper = None

    @property
    def blockDimension(self) -> int:
        return self._target.domainDimension

    @property
    def density(self) -> DensityInterface:
        return self._target

    @property
    def domainType(self):
        return self._target.domainType

    @property
    def domainDimension(self) -> int:
        return self._target.domainDimension

    def __getattr__(self, name):
        # Pass through attributes like 'derivative' and 'reference' for RadonNikodym targets
        if hasattr(self._target, name):
            return getattr(self._target, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def evaluate_log(self, parameter: Parameter) -> float:
        return self._target.evaluate_log(parameter)

    def evaluate_log_gradient(self, parameter: Parameter) -> np.ndarray:
        return self._target.evaluate_log_gradient(parameter)

    def condition_on(self, state: BlockParameter) -> None:
        """
        Sync GP covariance functions to the current hyperparameter block, then
        forward conditioning to `target`.

        If the hyperparameter block is unchanged since the last call, the
        rebuild is skipped. Otherwise, rebuilds `gp`'s covariance function at
        the new hyperparameters, and `coarseGP`'s too if set. If `partition` and
        `finePrior` are both set, also extracts the fine-block marginal variance
        from `gp`'s covariance and rebuilds `finePrior` from it. If
        `localisedDensity` is set, syncs its weights from `coarseGP`'s engine.

        Forwarding to `target` happens on every call regardless of whether the
        rebuild ran, this updates the likelihood and prior evaluation points for
        the current hyperparameters, and is required even when the covariance
        itself hasn't changed.

        Parameters
        ----------
        state : BlockParameter
            Full joint state. The hyperparameter block is
            `state.block(self.hyperIdx)`, index 1 by default.
        """
        # Finding 3: jointState accesses the hyper block via state.block(self._hyperIdx)
        # where _hyperIdx defaults to 1.
        hyperState = state.block(self._hyperIdx)

        if self._cachedHyper is not None and self._cachedHyper == hyperState:
            # Hyperparameters are unchanged, so we can skip the rebuild.
            pass
        else:
            # Finding 1: The "rebuild" is constituted by creating a new covariance object
            # and assigning it to self._gp.covarianceFunction (and potentially _coarseGP and _finePrior).
            # This invalidates the previously cached covariance, forcing assembly and
            # factorisation upon the next evaluation.
            phi = hyperState.coordinate
            zeta = np.exp(phi)
            rho, sigma = zeta[0], zeta[1]

            covType = type(self._gp.covarianceFunction)
            smoothness = self._gp.covarianceFunction._smoothness
            newCov = covType(rho, smoothness, sigma**2)
            
            self._gp.covarianceFunction = newCov
            if self._coarseGP is not None:
                self._coarseGP.covarianceFunction = covType(
                    rho, smoothness, sigma**2
                )
                if self._localisedDensity is not None:
                    self._localisedDensity.sync_weights(
                        self._coarseGP.expansion.spectralWeights
                    )

            if self._finePrior is not None and self._partition is not None:
                cov = self._gp.measure.covariance
                compVar = self._partition.rule.extract(
                    1, cov.marginalVariance
                )
                self._finePrior.covariance = (
                    self._finePrior.covariance.__class__(compVar)
                )

            self._cachedHyper = hyperState.clone()

        # Finding 2: The operations that must still run even when theta is unchanged
        # are updating the likelihood evaluation point and the prior evaluation point
        # for the current zeta, which are handled by target.condition_on(state).
        if hasattr(self._target, 'condition_on'):
            self._target.condition_on(state)
        elif hasattr(self._target, 'derivative') and hasattr(self._target.derivative, 'condition_on'):
            self._target.derivative.condition_on(state)

    def draw(self, rng):
        raise NotImplementedError("SGLMMLatentConditional cannot be drawn from.")
