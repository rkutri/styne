from styne.statistics.interface import DensityInterface
from styne.statistics.measure import AbsolutelyContinuousProbabilityMeasure
from styne.parameter.parameter import Parameter


class RadonNikodym(DensityInterface):
    """
    Density defined as a Radon-Nikodym derivative with respect to a
    reference measure.

    The log-density is

        log p(x) = f(x) + log \varphi(x)

    where \varphi is the reference measure's density
    and f is the derivative.

    Parameters
    ----------
    reference : AbsolutelyContinuousProbabilityMeasure
        Reference measure.
    derivative : DensityInterface
        The Radon-Nikodym derivative.
    """

    def __init__(self,
                 reference: AbsolutelyContinuousProbabilityMeasure,
                 derivative: DensityInterface
                 ):

        if not isinstance(reference, AbsolutelyContinuousProbabilityMeasure):
            raise TypeError(
                "reference must be an AbsolutelyContinuousProbabilityMeasure."
            )
        if not isinstance(derivative, DensityInterface):
            raise TypeError("derivative must be a DensityInterface.")

        self._reference = reference
        self._derivative = derivative

    def with_reference(self, reference):
        return type(self)(reference, self._derivative)

    def with_derivative(self, derivative):
        return type(self)(self._reference, derivative)

    @property
    def domainType(self):
        return self._derivative.domainType

    @property
    def domainDimension(self):
        return self._derivative.domainDimension

    @property
    def derivative(self):
        return self._derivative

    @property
    def reference(self):
        return self._reference

    def evaluate_log(self, parameter: Parameter):
        return (self._derivative.evaluate_log(parameter)
                + self._reference.density.evaluate_log(parameter))

    def evaluate_log_gradient(self, parameter: Parameter):
        """Sum of reference-density gradient and derivative gradient.

        Both components must expose an ``evaluate_log_gradient`` method.
        Raises RuntimeError if either is non-differentiable.
        """
        ref_density = self._reference.density

        if not callable(getattr(ref_density, "evaluate_log_gradient", None)):
            raise RuntimeError(
                f"{type(ref_density).__name__} does not support "
                "evaluate_log_gradient."
            )
        if not callable(
                getattr(self._derivative, "evaluate_log_gradient", None)):
            raise RuntimeError(
                f"{type(self._derivative).__name__} does not support "
                "evaluate_log_gradient."
            )

        return (ref_density.evaluate_log_gradient(parameter)
                + self._derivative.evaluate_log_gradient(parameter))

    def condition_on(self, state: Parameter) -> None:
        """
        Sync reference measure and RN-derivative with updated context state.
        """
        if hasattr(self._reference, 'condition_on'):
            self._reference.condition_on(state)
        elif hasattr(self._reference, 'density') and \
                hasattr(self._reference.density, 'condition_on'):
            self._reference.density.condition_on(state)

        if hasattr(self._derivative, 'condition_on'):
            self._derivative.condition_on(state)

    def condition(self, state: Parameter):
        """Return an independently conditioned composition."""
        import copy
        conditioned = copy.deepcopy(self)
        conditioned.condition_on(state)
        return conditioned
