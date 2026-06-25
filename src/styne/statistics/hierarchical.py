import numpy as np
import copy
from typing import Optional

from styne.statistics.measure import ConditionalMeasure
from styne.statistics.interface import DensityInterface, LikelihoodInterface
from styne.parameter.parameter import Parameter
from styne.parameter.block import BlockParameter
from styne.parameter.function import Function
from styne.model.sglmm import SGLMM
from styne.gp.gaussianprocess import GaussianProcess
from styne.statistics.stationary import MaternCovariance1D
from styne.statistics.covariance import DenseCovarianceMatrix, IIDCovarianceMatrix
from styne.gp.direct import DirectGPEngine
from styne.gp.dna import DNAFourierEngine

class SGLMMHyperConditionalDensity(DensityInterface):
    """
    Log-density target for Metropolis Random Walk on log-hyperparameters.
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

            self._model.reset()
            self._model.interpolate(self._latentState)
            self._model.evaluate()
        except np.linalg.LinAlgError:
            return -np.inf
            
        linearPredictor = self._model.evaluation

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

                self._model.reset()
                self._model.interpolate(self._latentState)
                self._model.evaluate()
        except np.linalg.LinAlgError:
            raise RuntimeError("Failed model evaluation due to singular covariance.")
            
        linearPredictorScore = self._likelihood.response.score(
            self._likelihood.data.measurement, self._model
        )

        from styne.parameter.vector import Vector
        priorGradient = self._pcPrior.evaluate_log_gradient(
            Vector([lengthScale, sigma])
        )

        if hasattr(self._gp.engine, 'evaluate_hyper_gradient'):
            # Engine expects the Expansion representation
            latentExpansion = self._latentState.function \
                if isinstance(self._latentState, Function) else self._latentState
            hyperparameterGradients = self._gp.engine.evaluate_hyper_gradient(
                latentExpansion, linearPredictorScore, self._gp.covarianceFunction
            )
            gradLogLengthScale = hyperparameterGradients.get('log_rho', 0.0) \
                + priorGradient[0] * lengthScale + 1.0
            gradLogSigma = hyperparameterGradients.get('log_sigma', 0.0) \
                + priorGradient[1] * sigma + 1.0
            return np.array([gradLogLengthScale, gradLogSigma])

        if not hasattr(self._gp.engine, "compute_log_length_multiplier"):
            raise NotImplementedError(
                "Hyperparameter gradients not implemented for "
                f"{type(self._gp.engine).__name__}."
            )

        lengthMultiplier = self._gp.engine.compute_log_length_multiplier(
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
        latentState = state.block(0) if self._hyperIdx == 1 else state.block(1)
        hyperState = state.block(self._hyperIdx)

        phi = hyperState.coordinate
        zeta = np.exp(phi)
        rho, sigma = zeta[0], zeta[1]

        covType = type(self._gp.covarianceFunction)
        smoothness = self._gp.covarianceFunction._smoothness
        self._gp.covarianceFunction = covType(rho, smoothness, sigma**2)
        self._gp.measure.covariance.scaling = 1.0

        self._model.reset()
        self._model.interpolate(latentState)
        self._model.evaluate()

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
    Blocks latent evaluation until GP dependencies are synced to hyperparameters.
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
                        self._coarseGP.engine.spectralWeights
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
