import copy

import numpy as np

from styne.statistics.measure import ProbabilityMeasure
from typing import Optional

from styne.backend import infer_backend
from styne.statistics.covariance import (
    CovarianceMatrix,
    DiagonalCovarianceMatrix,
    IIDCovarianceMatrix,
    PrecisionCovarianceMatrix,
)
from styne.statistics.interface import DensityInterface, RadonNikodymInterface
from styne.statistics.gaussian import Gaussian
from styne.statistics.dirac import DiracMeasure
from styne.statistics.radonnikodym import RadonNikodym
from styne.parameter.parameter import Parameter
from styne.parameter.vector import Vector
from styne.mcmc.sampler import MCMCSampler
from styne.mcmc.surrogate import SurrogateTransitionMeasure
from styne.utility.densityarithmetic import LogScalingWrapper


# A penalty metric is a covariance matrix or one of these numbers. Testing the
# concrete number types, not the abstract covariance class, keeps the test as
# cheap as an identity check on every root step.
numberTypes = (int, float, np.number)


def apply_metric(penaltyMetric, vector):
    """B v for a penalty metric B, where a number gamma stands for gamma I."""
    if isinstance(penaltyMetric, numberTypes):
        return penaltyMetric * vector
    return penaltyMetric.apply(vector)


class LocalisedSurrogateDensity(RadonNikodym):
    """
    Tempered surrogate density localised at a point x in a penalty metric B.

    For a Radon-Nikodym surrogate with derivative L, the log density
    relative to the surrogate's reference is

        tau * log L(v) - (v - x)^T B (v - x)/2,

    and for any other surrogate the whole surrogate density takes the place
    of L, relative to the Lebesgue measure. B is a covariance matrix, or a
    positive number gamma standing for gamma I, so that the isotropic
    penalty needs no matrix. No Gaussian measure represents the penalty, so
    B may be positive semidefinite. The origin serves as x until a location
    is set.

    A Gaussian approximation N(m, C) moves the penalty into the reference
    N(mu_x, S^-1), with S = C^-1 + B and S mu_x = C^-1 m + B x. The
    derivative is then the tempered surrogate relative to N(m, C), which
    does not depend on x, and the log density changes by a constant that
    depends on x only.

    For a surrogate relative to the Lebesgue measure and positive-definite
    B, the penalty reference takes the penalty's own Gaussian N(x, B^-1) as
    reference, the limit of an approximation whose precision C^-1 vanishes.
    The derivative is then the tempered surrogate itself, so a pCN root is
    preconditioned against the penalty and corrected by the tempered
    surrogate alone. This reference needs the location to be set.

    Parameters
    ----------
    penaltyMetric : CovarianceMatrix or float
        Metric B of the penalty on the surrogate's backend, or a positive
        number gamma for gamma I.
    tempering : float
        Tempering tau in (0, 1].
    surrogateDensity : DensityInterface
        The surrogate density.
    approximation : Gaussian, optional
        Gaussian N(m, C) on the surrogate's backend, typically an
        approximation of the tempered surrogate posterior.
    penaltyReference : bool, optional
        Whether N(x, B^-1) is the reference of a surrogate relative to the
        Lebesgue measure. False for penaltyMetric; legacy regularisation
        selects this reference when no approximation or reference is given.
    """

    class Derivative(DensityInterface):
        """Tempered surrogate derivative plus the localisation penalty."""

        def __init__(self, localised):
            self._localised = localised

        @property
        def domainType(self):
            return self._localised._tempered.domainType

        @property
        def domainDimension(self):
            return self._localised._tempered.domainDimension

        def evaluate_log(self, parameter: Parameter):
            return (
                self._localised.log_localisation(parameter)
                + self._localised._tempered.derivative.evaluate_log(parameter)
            )

        def evaluate_log_gradient(self, parameter: Parameter):
            localised = self._localised
            tempered = localised._tempered.derivative
            return tempered.evaluate_log_gradient(parameter) - apply_metric(
                localised._penaltyMetric, localised._offset(parameter)
            )

    def __init__(
        self, regularisation=None, tempering=1.0, surrogateDensity=None,
        spectralWeights=None, temperFullDensity=False, *, penaltyMetric=None,
        approximation=None, penaltyReference=None,
    ):
        if regularisation is not None and penaltyMetric is not None:
            raise ValueError("Specify regularisation or penaltyMetric, not both.")
        self._regularisation = regularisation
        self._spectralWeights = spectralWeights
        self._temperFullDensity = temperFullDensity
        if penaltyMetric is None:
            penaltyMetric = regularisation
            if spectralWeights is not None:
                penaltyMetric = DiagonalCovarianceMatrix(
                    regularisation * spectralWeights ** 2
                )
        elif spectralWeights is not None:
            raise ValueError("spectralWeights requires legacy regularisation.")
        if penaltyReference is None:
            penaltyReference = (regularisation is not None
                                and approximation is None and not isinstance(
                                    surrogateDensity, RadonNikodymInterface
                                ))
        if not isinstance(penaltyMetric, CovarianceMatrix) and not (
                isinstance(penaltyMetric, numberTypes) and penaltyMetric > 0):
            raise ValueError(
                "penaltyMetric must be a covariance matrix or a positive number."
            )
        if tempering <= 0. or 1. < tempering:
            raise ValueError("Tempering must be in (0, 1].")
        if approximation is not None and not isinstance(approximation, Gaussian):
            raise TypeError("approximation must be a Gaussian.")
        if penaltyReference and (
                approximation is not None
                or isinstance(surrogateDensity, RadonNikodymInterface)):
            raise ValueError(
                "A penalty reference serves a surrogate relative to the "
                "Lebesgue measure, without an approximation."
            )

        self._surrogateDensity = surrogateDensity
        self._tempering = tempering
        self._penaltyMetric = penaltyMetric
        self._approximation = approximation
        self._penaltyReference = penaltyReference
        self._location = None
        self._tempered = self._temper(surrogateDensity)

        if penaltyReference:
            # The inverse metric is formed once. A number forms it on NumPy
            # here, and a location on another backend forms its own.
            self._penaltyCovariance = self._metric_inverse(
                penaltyMetric, surrogateDensity
            )
            reference = self._penalty_reference()
            derivative = self._tempered.derivative
        elif approximation is None:
            reference = self._tempered.reference
            derivative = LocalisedSurrogateDensity.Derivative(self)
        else:
            # The penalty joins N(m, C) in the reference, which leaves the
            # tempered surrogate relative to N(m, C) as the derivative.
            covariance = approximation.covariance
            self._shift = covariance.apply_inverse(approximation.mean.coordinate)
            self._rootCovariance = self._root_covariance(covariance)
            reference = self._root_reference()
            derivative = self._tempered.derivative
        super().__init__(reference, derivative)
        if regularisation is not None:
            mean = getattr(getattr(surrogateDensity, 'reference', None), 'mean', None)
            if mean is None:
                self.location = Vector(np.zeros(surrogateDensity.domainDimension))
            else:
                backend = infer_backend(mean.coordinate)
                metadata = backend.metadata(mean.coordinate)
                self.location = mean.with_coordinate(backend.zeros(
                    mean.coordinate.shape, dtype=metadata.dtype, device=metadata.device
                ))

    def _temper(self, surrogateDensity):
        """The tempered surrogate, relative to the approximation if given."""
        # A Radon-Nikodym surrogate keeps its reference, typically the
        # prior, which preconditioned root chains use as theirs.
        if self._temperFullDensity and isinstance(
                surrogateDensity, RadonNikodymInterface):
            reference = surrogateDensity.reference
            reference = Gaussian(reference.covariance.with_scaling(
                reference.covariance.scaling / self._tempering
            ), reference.mean)
            tempered = RadonNikodym(reference, LogScalingWrapper(
                surrogateDensity.derivative, self._tempering
            ))
        elif isinstance(surrogateDensity, RadonNikodymInterface):
            tempered = RadonNikodym(
                surrogateDensity.reference,
                LogScalingWrapper(surrogateDensity.derivative, self._tempering),
            )
        else:
            tempered = RadonNikodym(
                None, LogScalingWrapper(surrogateDensity, self._tempering)
            )
        if self._approximation is None:
            return tempered
        return tempered.rebased(self._approximation)

    def _root_covariance(self, covariance):
        """The covariance S^-1, diagonal whenever C and B are."""
        backend = infer_backend(self._shift)
        metadata = backend.metadata(self._shift)
        dimension = self._shift.shape[-1]
        metric = self._penaltyMetric
        # A number stands for a multiple of the identity, which is diagonal.
        diagonalMetric = isinstance(
            metric, (*numberTypes, DiagonalCovarianceMatrix)
        )
        if isinstance(covariance, DiagonalCovarianceMatrix) and diagonalMetric:
            # A diagonal operator applied to ones returns its diagonal.
            ones = backend.ones(
                dimension, dtype=metadata.dtype, device=metadata.device
            )
            return DiagonalCovarianceMatrix(1.0 / (
                covariance.apply_inverse(ones) + apply_metric(metric, ones)
            ))
        identity = backend.eye(
            dimension, dtype=metadata.dtype, device=metadata.device
        )
        return PrecisionCovarianceMatrix(
            covariance.apply_inverse(identity) + apply_metric(metric, identity)
        )

    def _root_reference(self):
        """The Gaussian N(mu_x, S^-1) at the current location x."""
        shift = self._shift
        if self._location is not None:
            shift = shift + apply_metric(
                self._penaltyMetric, self._location.coordinate
            )
        mean = self._rootCovariance.apply(shift)
        return Gaussian(
            self._rootCovariance,
            self._approximation.mean.with_coordinate(mean),
        )

    @staticmethod
    def _metric_inverse(metric, surrogateDensity):
        """The covariance B^-1 on the surrogate's domain, diagonal whenever B is."""
        if isinstance(metric, numberTypes):
            return IIDCovarianceMatrix(
                surrogateDensity.domainDimension, 1.0 / metric
            )
        if isinstance(metric, DiagonalCovarianceMatrix):
            return DiagonalCovarianceMatrix(
                1.0 / (metric.scaling * metric.marginalVariance)
            )
        return PrecisionCovarianceMatrix(metric)

    def _penalty_reference(self):
        """The Gaussian N(x, B^-1) at the current location x, mean unset before."""
        covariance = self._penaltyCovariance
        if self._location is not None and isinstance(
                self._penaltyMetric, numberTypes):
            coordinate = self._location.coordinate
            backend = infer_backend(coordinate)
            if backend.name != "numpy":
                metadata = backend.metadata(coordinate)
                covariance = IIDCovarianceMatrix(
                    coordinate.shape[-1],
                    backend.asarray(
                        1.0 / self._penaltyMetric,
                        dtype=metadata.dtype, device=metadata.device,
                    ),
                )
        return Gaussian(covariance, self._location)

    def evaluate_log_surrogate(self, y: Parameter):
        """Log density of the tempered surrogate, without the localisation.

        In the DART acceptance ratio, the localisation terms cancel by
        symmetry. Only this component contributes to the outer MH ratio.
        """
        return self._tempered.evaluate_log(y)

    @property
    def surrogateReference(self):
        """Measure the tempered surrogate refers to, None for Lebesgue.

        This is the approximation if one is given, and the surrogate's
        reference otherwise.
        """
        return self._tempered.reference

    def log_surrogate_from_stored(self, logDensity, parameter: Parameter):
        """Tempered surrogate relative to `surrogateReference` at ``parameter``.

        ``logDensity`` is the derivative a root chain stored there. It
        contains the localisation unless the reference has absorbed it.
        """
        if self._approximation is not None or self._penaltyReference:
            return logDensity
        return logDensity - self.log_localisation(parameter)

    def _offset(self, parameter: Parameter):
        if self._location is None:
            return parameter.coordinate
        return parameter.coordinate - self._location.coordinate

    def log_localisation(self, parameter: Parameter):
        """The penalty -(v - x)^T B (v - x)/2 at ``parameter``."""
        offset = self._offset(parameter)
        if isinstance(self._penaltyMetric, numberTypes):
            namespace = infer_backend(offset).namespace
            return -0.5 * self._penaltyMetric * namespace.sum(offset * offset, axis=-1)
        return -0.5 * self._penaltyMetric.quadratic_form(offset)

    @property
    def location(self) -> Optional[Parameter]:
        return self._location

    @location.setter
    def location(self, location: Parameter):
        self._location = location
        if self._penaltyReference:
            self._reference = self._penalty_reference()
        elif self._approximation is not None:
            self._reference = self._root_reference()

    def with_location(self, location: Parameter):
        """Return this density localised at ``location``."""
        result = copy.copy(self)
        result.location = location
        if self._approximation is None and not self._penaltyReference:
            result._derivative = LocalisedSurrogateDensity.Derivative(result)
        return result

    def with_surrogate_density(self, surrogateDensity, spectralWeights=None, *,
                               penaltyMetric=None):
        """Rebuild the localised density around a replacement surrogate.

        A given ``penaltyMetric`` replaces the metric. A penalty reference is
        kept, an approximation is not.
        """
        if self._regularisation is not None and penaltyMetric is None:
            settings = {'regularisation': self._regularisation,
                        'spectralWeights': spectralWeights}
        else:
            if spectralWeights is not None:
                raise ValueError("spectralWeights requires legacy regularisation.")
            settings = {'penaltyMetric': self._penaltyMetric
                        if penaltyMetric is None else penaltyMetric}
        result = type(self)(
            tempering=self._tempering,
            surrogateDensity=surrogateDensity,
            temperFullDensity=self._temperFullDensity,
            penaltyReference=self._penaltyReference,
            **settings,
        )
        if self._location is not None:
            result.location = self._location
        return result

    def condition(self, state: Parameter):
        """Return this density around its surrogate given ``state``.

        The penalty and a Gaussian root reference do not involve the
        surrogate, so they are kept rather than rebuilt.
        """
        condition = getattr(self._surrogateDensity, "condition", None)
        surrogateDensity = (self._surrogateDensity if condition is None
                            else condition(state))
        if surrogateDensity is self._surrogateDensity:
            return self
        result = copy.copy(self)
        result._surrogateDensity = surrogateDensity
        result._tempered = self._temper(surrogateDensity)
        if self._approximation is None and not self._penaltyReference:
            result._reference = result._tempered.reference
            result._derivative = LocalisedSurrogateDensity.Derivative(result)
        else:
            result._derivative = result._tempered.derivative
        return result

    @property
    def surrogateDensity(self):
        return self._surrogateDensity

    @property
    def tempering(self) -> float:
        return self._tempering

    @property
    def regularisation(self):
        if self._regularisation is not None:
            return self._regularisation
        if isinstance(self._penaltyMetric, numberTypes):
            return self._penaltyMetric
        raise ValueError("A general penaltyMetric has no scalar regularisation.")

    @property
    def spectralWeights(self):
        return self._spectralWeights

    def sync_weights(self, weights):
        if weights is None:
            return
        if self._regularisation is None:
            raise ValueError("spectralWeights requires legacy regularisation.")
        replacement = type(self)(
            self._regularisation, self._tempering, self._surrogateDensity,
            spectralWeights=weights, temperFullDensity=self._temperFullDensity,
            approximation=self._approximation, penaltyReference=self._penaltyReference,
        )
        replacement.location = self._location
        self.__dict__.update(replacement.__dict__)

    @property
    def penaltyMetric(self):
        """Metric B of the penalty, a covariance matrix or a number gamma."""
        return self._penaltyMetric

    @property
    def approximation(self) -> Optional[Gaussian]:
        return self._approximation

    @property
    def penaltyReference(self) -> bool:
        """Whether the penalty's Gaussian N(x, B^-1) is the reference."""
        return self._penaltyReference


class LocalisedSurrogateTransitionMeasure(SurrogateTransitionMeasure):
    """
    Surrogate transition measure whose target is a localised density.

    On each draw, the localisation centre and the Dirac initial measure
    are both moved to the current fine-level state before running the
    surrogate chain.

    Parameters
    ----------
    surrogateChain : MCMCSampler
        Must target a 'LocalisedSurrogateDensity'.
    nChain : int
        Length of each surrogate chain run.
    """

    def __init__(self,
        surrogateChain: MCMCSampler,
        nChain: int,
        initialMeasure: Optional[ProbabilityMeasure] = None
    ):

        if not isinstance(surrogateChain.target, LocalisedSurrogateDensity):
            raise TypeError("surrogateChain target must be a "
                            "LocalisedSurrogateDensity instance.")
        if nChain == 0 and initialMeasure is not None and not isinstance(
                initialMeasure, (DiracMeasure, Gaussian)):
            raise TypeError(
                "Zero-subchain DART requires a Dirac or Gaussian initial "
                "measure."
            )

        super().__init__(surrogateChain, nChain, initialMeasure)

    @property
    def location(self) -> Parameter:
        return self._mcmc.target.location

    @location.setter
    def location(self, location: Parameter):
        self._initialMeasure = self._localise_initial_measure(location)
        self._mcmc.target.location = location

    def _localise_initial_measure(self, location: Parameter):
        if hasattr(self._initialMeasure, "with_location"):
            return self._initialMeasure.with_location(location)
        if hasattr(self._initialMeasure, "with_mean"):
            return self._initialMeasure.with_mean(location)

        initialMeasure = copy.copy(self._initialMeasure)
        initialMeasure.location = location
        return initialMeasure

    def with_density(self, density: LocalisedSurrogateDensity):
        """Return an isolated transition measure targeting ``density``."""
        if not isinstance(density, LocalisedSurrogateDensity):
            raise TypeError("density must be a LocalisedSurrogateDensity.")
        result = copy.copy(self)
        result._mcmc = self._mcmc.with_target(density)
        result._initialMeasure = copy.copy(self._initialMeasure)
        return result

    def transition_trajectory(self, initialState: Parameter, randomState):
        """Run an isolated surrogate trajectory localised at ``initialState``."""
        measure = self.with_density(self.density.with_location(initialState))
        measure._initialMeasure = self._localise_initial_measure(initialState)
        trajectoryStart, randomState = measure._initialMeasure.sample(randomState)
        return super(
            LocalisedSurrogateTransitionMeasure, measure
        ).transition_trajectory(trajectoryStart, randomState)

    @property
    def penaltyMetric(self):
        return self.density.penaltyMetric

    @property
    def regularisation(self) -> float:
        return self.density.regularisation

    @property
    def tempering(self) -> float:
        return self.density.tempering
