import copy
import numpy as np
from typing import Optional

from styne.backend import infer_backend
from styne.statistics.measure import ConditionalMeasure
from styne.statistics.interface import DensityInterface, LikelihoodInterface
from styne.statistics.likelihood import RegressionLikelihood
from styne.statistics.radonnikodym import RadonNikodym
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

    def condition(self, state: BlockParameter):
        """Return a density using the latent block from ``state``."""
        return type(self)(
            self._pcPrior,
            self._gp,
            self._model,
            self._likelihood,
            state.block(0),
        )

    def _model_at(self, lengthScale, sigma):
        """Construct an isolated model for one hyperparameter evaluation."""
        covarianceType = type(self._gp.covarianceFunction)
        smoothness = self._gp.covarianceFunction._smoothness
        gp = self._gp.with_covariance_function(
            covarianceType(lengthScale, smoothness, sigma**2)
        )
        if isinstance(self._model, SGLMM):
            return self._model.with_gp(gp)
        model = copy.deepcopy(self._model)
        model._gp = gp
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
        try:
            linearPredictor = self._model_at(
                lengthScale, sigma
            )(self._latentState)
        except np.linalg.LinAlgError:
            return -np.inf
            
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
        Prior on the fine block, reconstructed when `partition` is given.
        Untyped in source, inferred from usage.
    hyperIdx : int, default 1
        Index of the hyperparameter block within the joint state.
    localisedDensity : object, optional
        Localised target reconstructed from the conditioned coarse GP and its
        spectral weights. Untyped in source, inferred from usage.
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
        target = self.__dict__.get("_target")
        if target is not None and hasattr(target, name):
            return getattr(target, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def evaluate_log(self, parameter: Parameter) -> float:
        return self._target.evaluate_log(parameter)

    def evaluate_log_gradient(self, parameter: Parameter) -> np.ndarray:
        return self._target.evaluate_log_gradient(parameter)

    @property
    def gp(self):
        return self._gp

    @property
    def coarseGP(self):
        return self._coarseGP

    @property
    def partition(self):
        return self._partition

    @property
    def finePrior(self):
        return self._finePrior

    @property
    def localisedDensity(self):
        return self._localisedDensity

    @staticmethod
    def _with_gp(density, currentGP, replacementGP):
        if isinstance(density, RadonNikodym):
            reference = density.reference
            if reference is currentGP.measure:
                reference = replacementGP.measure
            derivative = SGLMMLatentConditional._with_gp(
                density.derivative, currentGP, replacementGP
            )
            return density.with_reference(reference).with_derivative(derivative)

        if isinstance(density, RegressionLikelihood):
            model = density.model
            if isinstance(model, SGLMM) and model._gp is currentGP:
                return density.with_model(model.with_gp(replacementGP))

        return copy.copy(density)

    @staticmethod
    def _condition_gp(gp, hyperState):
        coordinate = hyperState.coordinate
        backend = infer_backend(coordinate)
        rho, sigma = backend.namespace.exp(coordinate)
        covarianceType = type(gp.covarianceFunction)
        smoothness = gp.covarianceFunction._smoothness
        return gp.with_covariance_function(
            covarianceType(rho, smoothness, sigma**2)
        )

    def condition(self, state: BlockParameter):
        """Return a consistently reconstructed conditional composition."""
        hyperState = state.block(self._hyperIdx)
        gp = self._condition_gp(self._gp, hyperState)
        target = self._with_gp(self._target, self._gp, gp)

        coarseGP = None
        if self._coarseGP is not None:
            coarseGP = self._condition_gp(self._coarseGP, hyperState)

        partition = self._partition
        if partition is not None:
            if callable(getattr(partition, "with_process", None)):
                partition = partition.with_process(gp, state.block(0))
            else:
                partition = copy.copy(partition)
                partition.parameter = state.block(0)

        finePrior = self._finePrior
        if finePrior is not None and partition is not None:
            componentVariance = partition.rule.extract(
                1, gp.measure.covariance.marginalVariance
            )
            finePrior = finePrior.with_covariance(
                finePrior.covariance.__class__(componentVariance)
            )

        localisedDensity = self._localisedDensity
        if localisedDensity is not None and coarseGP is not None:
            surrogate = self._with_gp(
                localisedDensity.surrogateDensity,
                self._coarseGP,
                coarseGP,
            )
            localisedDensity = localisedDensity.with_surrogate_density(
                surrogate, coarseGP.expansion.spectralWeights
            )

        return type(self)(
            target,
            gp,
            coarseGP=coarseGP,
            partition=partition,
            finePrior=finePrior,
            hyperIdx=self._hyperIdx,
            localisedDensity=localisedDensity,
        )

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
        conditioned = self.condition(state)
        self._target = conditioned._target
        self._gp = conditioned._gp
        self._coarseGP = conditioned._coarseGP
        self._partition = conditioned._partition
        self._finePrior = conditioned._finePrior
        self._localisedDensity = conditioned._localisedDensity

    def draw(self, rng):
        raise NotImplementedError("SGLMMLatentConditional cannot be drawn from.")
