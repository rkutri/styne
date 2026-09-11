import importlib

import pytest


EXPECTED_EXPORTS = {
    "styne": {
        "enable_logging", "__version__", "ForwardMap",
        "DifferentiableForwardMap", "Parameter", "MetropolisHastings",
    },
    "styne.backend": {
        "ArrayNamespace", "Backend", "BackendCapabilities",
        "BackendCapabilityError", "BackendMetadata", "BackendInferenceError",
        "BackendNotFoundError", "BackendRegistry", "BackendRegistryError",
        "BackendUnavailableError", "MixedBackendError", "get_backend",
        "infer_backend", "register_backend",
    },
    "styne.parameter": {
        "Parameter", "Scalar", "Vector", "Function", "BlockParameter",
    },
    "styne.model": {
        "ForwardMap", "DifferentiableForwardMap", "LinearForwardMap",
        "SGLMM", "Trend", "ConstantTrend", "LinearTrend",
    },
    "styne.model.representation": {
        "Expansion", "LinearExpansion", "BoundExpansion",
        "BoundLinearExpansion", "GridFunction", "BSpline1D", "BSpline2D",
    },
    "styne.gp": {
        "GaussianProcess", "GPSampler", "DirectExpansion", "BC",
        "BoundaryCondition", "DNAFourierComponentExpansion",
        "DNAFourierExpansion", "DNACoarseFineSplit",
        "DNACoarseFinePartition", "CirculantEmbeddingEngine1D",
        "CirculantEmbeddingEngine2D",
        "ApproximateCirculantEmbeddingEngine1D",
        "ApproximateCirculantEmbeddingEngine2D",
    },
    "styne.statistics": {
        "DensityInterface", "RadonNikodymInterface", "LikelihoodInterface",
        "CovarianceFunctionInterface", "CovarianceOperatorInterface",
        "BayesianModelInterface", "ProbabilityMeasure",
        "AbsolutelyContinuousProbabilityMeasure", "ConditionalMeasure",
        "RadonNikodym", "GaussianDensity", "Gaussian",
        "DiagonalCovarianceMatrix", "IIDCovarianceMatrix",
        "DenseCovarianceMatrix", "StationaryCovariance",
        "ExponentialCovariance1D", "MaternCovariance1D",
        "MaternCovariance2D", "Data", "DiracMeasure", "Binomial", "Poisson",
        "GaussianResponse", "PoissonResponse", "BinomialResponse",
        "RegressionLikelihood", "MetropolisWithinGibbsConditional",
        "UnnormalisedPosterior", "HierarchicalBayes",
        "HierarchicalBayesModelBuilder", "SGLMMHyperConditionalDensity",
        "SGLMMHyperConditional", "SGLMMLatentConditional",
        "MaternRangePCPrior", "MaternSigmaPCPrior", "JointMaternPCPrior",
        "WelfordAccumulator",
    },
    "styne.mcmc": {
        "MetropolisHastings", "MCMCSampler", "ProposalMethod", "MHFactory",
        "Annotator", "AcceptanceProbability", "ChainDiagnostics", "Chain",
        "GibbsChain", "EvaluatedState", "TransitionData", "DART",
        "DARTFactory", "DirectDART", "MetropolisAdjustedLangevinAlgorithm",
        "MALAFactory", "PreconditionedMALA", "PMALAFactory",
        "PreconditionedCrankNicolson", "PCNFactory",
        "MetropolisedRandomWalk", "MRWFactory", "RobbinsMonroMRW",
        "RobbinsMonroMRWFactory", "MultilevelDelayedAcceptanceMCMC",
        "MLDAFactory", "BlockGibbs", "GibbsBuilder", "RatioEstimator",
        "BlockProposal", "StandardAcceptance", "BarkerAcceptance",
    },
    "styne.utility": {
        "Grid", "UniformGrid", "Interpolation1D", "GridInterpolation2D",
        "linear_interpolation_matrix", "bilinear_interpolation_matrix",
        "ExplicitFunction", "estimate_autocorrelation_function_1d",
        "sokal_heuristic", "integrated_autocorrelation_1d",
        "integrated_autocorrelation", "multichain_ess_per_iter",
        "RWTunerConfig", "LangevinTunerConfig", "MRWTuner", "MALATuner",
        "PCNTuner", "PMALATuner", "infer_init", "LogScalingWrapper",
        "ProductWrapper", "NotPositiveDefinite", "bisection", "NullProgress",
        "TqdmProgress", "make_reporter",
    },
}


@pytest.mark.parametrize("moduleName, expected", EXPECTED_EXPORTS.items())
def test_public_exports_are_frozen_and_resolvable(moduleName, expected):
    module = importlib.import_module(moduleName)

    assert set(module.__all__) == expected
    for name in expected:
        assert getattr(module, name) is not None


@pytest.mark.parametrize(
    "moduleName, removedName",
    [
        ("styne", "DifferentiableModel"),
        ("styne.model", "DifferentiableModel"),
        ("styne.statistics", "SGLMMLikelihood"),
    ],
)
def test_removed_compatibility_names_are_absent(moduleName, removedName):
    module = importlib.import_module(moduleName)

    assert removedName not in module.__all__
    assert not hasattr(module, removedName)


@pytest.mark.parametrize(
    "moduleName",
    ["styne.gp.spde", "styne.model.conditionals", "styne.utility.map"],
)
def test_removed_experimental_modules_are_absent(moduleName):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(moduleName)
