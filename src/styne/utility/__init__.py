"""Curated public surface of styne.utility."""
import importlib

_SYMBOL_TO_MODULE = {
    "Grid": "styne.utility.grid",
    "UniformGrid": "styne.utility.grid",
    "Interpolation1D": "styne.utility.interpolation",
    "GridInterpolation2D": "styne.utility.interpolation",
    "linear_interpolation_matrix": "styne.utility.interpolation",
    "bilinear_interpolation_matrix": "styne.utility.interpolation",
    "estimate_autocorrelation_function_1d": "styne.utility.postprocessing",
    "sokal_heuristic": "styne.utility.postprocessing",
    "integrated_autocorrelation_1d": "styne.utility.postprocessing",
    "integrated_autocorrelation": "styne.utility.postprocessing",
    "multichain_ess_per_iter": "styne.utility.postprocessing",
    "RWTunerConfig": "styne.utility.tuning",
    "LangevinTunerConfig": "styne.utility.tuning",
    "MRWTuner": "styne.utility.tuning",
    "MALATuner": "styne.utility.tuning",
    "PCNTuner": "styne.utility.tuning",
    "PMALATuner": "styne.utility.tuning",
    "infer_init": "styne.utility.tuning",
    "EvaluationCache": "styne.utility.memoisation",
    "LogScalingWrapper": "styne.utility.densityarithmetic",
    "ProductWrapper": "styne.utility.densityarithmetic",
    "NotPositiveDefinite": "styne.utility.exceptions",
    "bisection": "styne.utility.bisection",
    "NullProgress": "styne.utility.progress",
    "TqdmProgress": "styne.utility.progress",
    "make_reporter": "styne.utility.progress",
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
