import copy
import numpy as np
from typing import Optional

from styne.backend import infer_backend
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

    def _model_at(self, lengthScale, sigma):
        """Construct an isolated model for one hyperparameter evaluation."""
        model = copy.deepcopy(self._model)
        covarianceType = type(model._gp.covarianceFunction)
        smoothness = model._gp.covarianceFunction._smoothness
        model._gp = model._gp.with_covariance_function(
            covarianceType(lengthScale, smoothness, sigma**2)
        )
        return model
            
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
        logHyperparameters = parameter.coordinate.reshape((-1,))
        backend = infer_backend(logHyperparameters)
        hyperparameters = backend.namespace.exp(logHyperparameters)
        lengthScale, sigma = hyperparameters[0], hyperparameters[1]

        from styne.parameter.vector import Vector
        hyperparameterVector = Vector(hyperparameters)
        logPrior = self._pcPrior.evaluate_log(hyperparameterVector)
        logJacobian = backend.namespace.sum(logHyperparameters)
        linearPredictor = self._model_at(lengthScale, sigma)(self._latentState)
            
        logLikelihood = self._likelihood.response.log_likelihood(
            self._likelihood.data.measurement, linearPredictor
        )
        
        return logPrior + logLikelihood + logJacobian

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

        self._density = SGLMMHyperConditionalDensity(
            self._pcPrior,
            self._gp,
            self._model,
            self._likelihood,
            latentState
        )

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

        Rebuilds `gp`'s covariance function at the supplied hyperparameters,
        and `coarseGP`'s too if set. If `partition` and `finePrior` are both
        set, also extracts the fine-block marginal variance from `gp`'s
        covariance and rebuilds `finePrior` from it. If `localisedDensity` is
        set, syncs its weights from `coarseGP`'s expansion.

        Parameters
        ----------
        state : BlockParameter
            Full joint state. The hyperparameter block is
            `state.block(self.hyperIdx)`, index 1 by default.
        """
        hyperState = state.block(self._hyperIdx)
        phi = hyperState.coordinate
        zeta = np.exp(phi)
        rho, sigma = zeta[0], zeta[1]

        covType = type(self._gp.covarianceFunction)
        smoothness = self._gp.covarianceFunction._smoothness
        self._gp = self._gp.with_covariance_function(
            covType(rho, smoothness, sigma**2)
        )
        if self._coarseGP is not None:
            self._coarseGP = self._coarseGP.with_covariance_function(
                covType(rho, smoothness, sigma**2)
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
            self._finePrior = self._finePrior.with_covariance(
                self._finePrior.covariance.__class__(compVar)
            )

        if hasattr(self._target, 'condition_on'):
            self._target.condition_on(state)
        elif hasattr(self._target, 'derivative') and hasattr(self._target.derivative, 'condition_on'):
            self._target.derivative.condition_on(state)

    def draw(self, rng):
        raise NotImplementedError("SGLMMLatentConditional cannot be drawn from.")
