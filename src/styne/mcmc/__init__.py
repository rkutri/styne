"""Curated public surface of styne.mcmc. mcmc.method is an implementation detail, not a documented import path."""
import importlib

_SYMBOL_TO_MODULE = {
    "MetropolisHastings": "styne.mcmc.metropolishastings",
    "MCMCSampler": "styne.mcmc.sampler",
    "ProposalMethod": "styne.mcmc.proposal",
    "MHFactory": "styne.mcmc.factory",
    "Annotator": "styne.mcmc.annotator",
    "AcceptanceProbability": "styne.mcmc.acceptance",
    "ChainDiagnostics": "styne.mcmc.diagnostics",
    "Chain": "styne.mcmc.chain",
    "GibbsChain": "styne.mcmc.chain",
    "DART": "styne.mcmc.method.dart",
    "DARTFactory": "styne.mcmc.method.dart",
    "DirectDART": "styne.mcmc.method.dartdirect",
    "MetropolisAdjustedLangevinAlgorithm": "styne.mcmc.method.mala",
    "MALAFactory": "styne.mcmc.method.mala",
    "PreconditionedMALA": "styne.mcmc.method.pmala",
    "PMALAFactory": "styne.mcmc.method.pmala",
    "PreconditionedCrankNicolson": "styne.mcmc.method.pcn",
    "PCNFactory": "styne.mcmc.method.pcn",
    "MetropolisedRandomWalk": "styne.mcmc.method.mrw",
    "MRWFactory": "styne.mcmc.method.mrw",
    "RobbinsMonroMRW": "styne.mcmc.method.mrw",
    "RobbinsMonroMRWFactory": "styne.mcmc.method.mrw",
    "MultilevelDelayedAcceptanceMCMC": "styne.mcmc.method.mlda",
    "MLDAFactory": "styne.mcmc.method.mlda",
    "BlockGibbs": "styne.mcmc.method.gibbs",
    "GibbsBuilder": "styne.mcmc.method.gibbs",
    "RatioEstimator": "styne.mcmc.method.ratio",
    "BlockProposal": "styne.mcmc.proposal",
    "StandardAcceptance": "styne.mcmc.acceptance",
    "BarkerAcceptance": "styne.mcmc.acceptance",
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
