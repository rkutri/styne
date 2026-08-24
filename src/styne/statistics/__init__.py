"""Curated public surface of styne.statistics."""
import importlib

_SYMBOL_TO_MODULE = {
    "DensityInterface": "styne.statistics.interface",
    "LikelihoodInterface": "styne.statistics.interface",
    "CovarianceFunctionInterface": "styne.statistics.interface",
    "CovarianceOperatorInterface": "styne.statistics.interface",
    "BayesianModelInterface": "styne.statistics.interface",
    "ProbabilityMeasure": "styne.statistics.measure",
    "AbsolutelyContinuousProbabilityMeasure": "styne.statistics.measure",
    "ConditionalMeasure": "styne.statistics.measure",
    "RadonNikodym": "styne.statistics.radonnikodym",
    "GaussianDensity": "styne.statistics.gaussian",
    "Gaussian": "styne.statistics.gaussian",
    "DiagonalCovarianceMatrix": "styne.statistics.covariance",
    "IIDCovarianceMatrix": "styne.statistics.covariance",
    "DenseCovarianceMatrix": "styne.statistics.covariance",
    "StationaryCovariance": "styne.statistics.stationary",
    "ExponentialCovariance1D": "styne.statistics.stationary",
    "MaternCovariance1D": "styne.statistics.stationary",
    "MaternCovariance2D": "styne.statistics.stationary",
    "Data": "styne.statistics.data",
    "DiracMeasure": "styne.statistics.dirac",
    "Binomial": "styne.statistics.binomial",
    "Poisson": "styne.statistics.poisson",
    "GaussianResponse": "styne.statistics.response",
    "PoissonResponse": "styne.statistics.response",
    "BinomialResponse": "styne.statistics.response",
    "RegressionLikelihood": "styne.statistics.likelihood",
    "SGLMMLikelihood": "styne.statistics.likelihood",
    "MetropolisWithinGibbsConditional": "styne.statistics.conditional",
    "UnnormalisedPosterior": "styne.statistics.bayes",
    "HierarchicalBayes": "styne.statistics.bayes",
    "HierarchicalBayesModelBuilder": "styne.statistics.bayes",
    "SGLMMHyperConditionalDensity": "styne.statistics.hierarchical",
    "SGLMMHyperConditional": "styne.statistics.hierarchical",
    "SGLMMLatentConditional": "styne.statistics.hierarchical",
    "MaternRangePCPrior": "styne.statistics.pc",
    "MaternSigmaPCPrior": "styne.statistics.pc",
    "JointMaternPCPrior": "styne.statistics.pc",
    "WelfordAccumulator": "styne.statistics.welford",
}

__all__ = list(_SYMBOL_TO_MODULE.keys())


def __getattr__(name):
    module_path = _SYMBOL_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path)
    return getattr(module, name)


def __dir__():
    return sorted(__all__)
