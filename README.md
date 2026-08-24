<p align="center">
  <img src="docs/assets/styne-logo-wide.svg" alt="styne logo" width="420">
</p>

[![DOI](https://zenodo.org/badge/1278349844.svg)](https://zenodo.org/badge/latestdoi/1278349844) [![DART](https://img.shields.io/badge/DART-arXiv%3A2606.27564-b31b1b)](https://arxiv.org/abs/2606.27564) [![DNA](https://img.shields.io/badge/DNA-10.1137%2F24M1715854-blue)](https://doi.org/10.1137/24M1715854)

*Pre-1.0 (`v0.2.1`)*

A Python library for Bayesian inference designed for high-dimensional problems
and computationally expensive forward models.

The library separates forward models from inference algorithms through a small
set of interfaces. Existing simulators, PDE solvers and other
application-specific models can therefore be combined with any compatible
sampler, while new inference algorithms can be developed independently of the
underlying model implementation.

This architecture naturally supports Gaussian-process priors, including the DNA
parametrisation, latent Gaussian models such as spatial GLMMs,
surrogate-assisted algorithms such as DART, and hierarchical Bayesian models.
These are not separate frameworks but compositions of the same underlying
interfaces and components.

## Install

Requires Python 3.10+.

```bash
pip install styne
```

The example scripts save plots and require the plotting extra:

```bash
pip install styne[plotting]
```

## Quickstart

The runnable version is `examples/01_quickstart.py`, showing a minimal Bayesian
linear regression example end to end. The essential wiring is summarised below.

```python
# prior definition
priorCov = IIDCovarianceMatrix(2, 6.0)
prior = Gaussian(priorCov, mean=Vector(np.zeros(2)))

# likelihood definition
noiseModel = GaussianResponse(IIDCovarianceMatrix(nObs, noiseVar))
forwardModel = LinearForwardMap(features)
likelihood = RegressionLikelihood(data, forwardModel, noiseModel)

# posterior definition
posterior = UnnormalisedPosterior(prior, likelihood)

# MCMC setup
factory = MRWFactory()
factory.target = posterior
factory.proposalCovariance = DiagonalCovarianceMatrix([0.02, 0.08])

sampler = factory.create()

# run MCMC
nSteps = 20000
initState = Vector(np.zeros(2))
sampler.run(nSteps, initState)
```

The example is intentionally assembled from interchangeable components. Most
objects shown here can be replaced independently, either by alternative library
implementations or by user-defined ones implementing the corresponding
interface, without changing the surrounding code. The principal extension points
are custom forward maps (`ForwardMap`), likelihoods (`LikelihoodInterface`) and
samplers (`MCMCSampler`).

## Usage

Start with the four runnable examples in `examples/`.

* `01_quickstart.py`: Bayesian linear regression, the shortest complete wiring
  from prior to posterior.
* `02_gp.py`: interchangeable Gaussian-process representations.
* `03_sglmm.py`: a spatial GLMM with Poisson observations.
* `04_pde_inverse_problem.py`: a custom PDE inverse problem.

## Components

**Sampling algorithms.** Random-walk Metropolis, MALA, pCN, pMALA, multilevel
delayed acceptance (MLDA), and DART, a surrogate-assisted delayed-acceptance
algorithm developed alongside the library.

**Gaussian-process simulation and priors.** Dense Cholesky, B-spline and DNA
parametrisations provide interchangeable ways to sample Gaussian-process
functions, evaluate them on spatial grids and use the corresponding Gaussian
measures as priors in Bayesian inverse problems. DNA is particularly useful when
efficient simulation of high-dimensional Gaussian fields is itself part of the
workflow.

**Models.** Ready-to-use implementations of Bayesian linear regression and
spatial GLMMs, together with the `ForwardMap` interface for wrapping arbitrary
application-specific forward models, including expensive PDE solvers and other
simulators.

**Infrastructure.** Common factory interfaces, automatic proposal tuning, and
composable abstractions for hierarchical and multilevel Bayesian models.

## Design

```text
Parameter ..> Model ---+
                       |
ResponseFamily --------+--> Likelihood ---+
                       |                  |
Data ------------------+                  |
                                          |
GaussianProcess --> Prior ----------------+
                                          |
                                          +--> Posterior --> Sampler
```

ForwardMap, ResponseFamily and Data compose into the Likelihood. GaussianProcess
builds the Prior. Prior and Likelihood compose into the Posterior, and the
Sampler targets it.

This diagram shows the simplest configuration: a single Gaussian-process prior
and a single sampler. The same composition naturally extends to directly sampled
priors, hierarchical models with hyperpriors, multilevel methods, and the other
sampling algorithms provided by the library.

The architecture separates models, likelihoods, priors and samplers along clear
constructor boundaries. Each component depends only on the interfaces of its
immediate neighbours, allowing individual parts of a Bayesian model to evolve independently.

This supports two complementary workflows. Existing simulators and forward
models can be wrapped in a `ForwardMap` subclass and immediately used with every
compatible sampler. Conversely, new inference algorithms can be developed
against the density interfaces without knowledge of, or dependence on, individual models.

In many uncertainty-quantification problems the forward model dominates the
computational cost. The `ForwardMap` interface therefore acts as the communication
boundary between the parameter space and the expensive computation producing the
model prediction. Surrogate-assisted methods such as DART are designed around
this boundary.

## Mathematical correspondence

This section describes the particular factorisation of a Bayesian sampling
problem represented by the `styne` interfaces.

| Mathematical object                 | Role in the formulation                                                              | `styne` object                                                   | Examples                                                                                                                                             |
| ----------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Parameter                           | Coordinate representation of the unknown                                             | `Parameter`, `Vector`, `Function`, `BlockParameter`              | regression coefficients; latent GP coefficients; PDE coefficient representation                                                                      |
| Prior / reference measure           | Distribution before conditioning on data, or reference measure for a target          | `ProbabilityMeasure`, `Gaussian`, `gp.measure`                   | IID Gaussian prior in the quickstart; DNA Gaussian-process prior in the PDE example; latent-field prior in hierarchical SGLMMs                       |
| Forward-map evaluation / predictor  | Deterministic quantity computed from the parameter and passed to the response family | `ForwardMap`                                                     | `LinearForwardMap`; `SGLMM`; custom `EllipticForwardMap`                                                                                                  |
| Response family / measurement model | Conditional law of observations given the model evaluation                           | `ResponseFamily`                                                 | `GaussianResponse`; `PoissonResponse`; `BinomialResponse`                                                                                            |
| Likelihood                          | Data-dependent log-density contribution                                              | `LikelihoodInterface`, `RegressionLikelihood`, `SGLMMLikelihood` | Gaussian regression likelihood; Poisson SGLMM likelihood                                                                                             |
| Target                              | Measure or unnormalised density sampled by an algorithm                              | `DensityInterface`, `RadonNikodym`, `UnnormalisedPosterior`      | `UnnormalisedPosterior(prior, likelihood)`; `RadonNikodym(prior, likelihood)`; direct targets such as `GaussianDensity` and `GaussianMixtureDensity` |
| Markov transition / sampler         | Transition mechanism targeting the chosen measure or density                         | `MCMCSampler`, `GibbsSampler`, `MetropolisHastings`                     | MRW in the quickstart; MALA for the SGLMM; pCN for the PDE inverse problem; DART and Gibbs samplers in the manuscript examples                       |

The main compression is the `ForwardMap` interface. Mathematically, one may separate
a forward map $\mathcal{G}$, an observation functional $F$, and a response
model. In `styne`, the model returns the deterministic quantity passed to the
response family, corresponding at the software boundary to
$F(\mathcal{G}(\theta))$. Its internals may contain interpolation, a PDE
solve, a simulator or any other application-specific computation. The model owns
the deterministic computation, while the response family owns the observation
law.

The Gaussian-process layer follows a similar convention. A `GaussianProcess`
combines a covariance function with a parametrisation engine. The engine maps
white-noise coordinates to coefficients and chooses an `Expansion`, while
`gp.measure` supplies the corresponding Gaussian reference measure. Dense
Cholesky, B-spline and DNA are therefore different parametrisations of the
prior, not different model classes.

### Functions and expansions

A `Function` is a parameter that can also be evaluated on a grid. It binds a
coordinate vector $\theta$ to an `Expansion` $E$, with

$$u_\theta(x) = E(\theta, x).$$

The coordinate belongs to the `Function`; the `Expansion` contains only the
static evaluation strategy and representation data, such as a Fourier basis,
B-spline knots or explicit sites. Expansions are therefore reusable and do not
change when a function is evaluated. Coordinates consistently use a trailing
feature axis, `(..., dimension)`, and evaluation preserves every leading batch
dimension. The same convention applies to GP Jacobian and adjoint operations,
matching JAX and PyTorch batching without transposes at the public boundary.

```python
field = Function(coordinate, expansion)
values = field.evaluate(grid)

updated = field.with_coordinate(new_coordinate)
assert updated.expansion is field.expansion
```

The refactored API replaces the previous mutable-realisation interface:

| Previous API | Current API |
| ------------ | ----------- |
| `function.function` | `function.expansion` |
| `function.function.evaluate(grid)` | `function.evaluate(grid)` |
| assign `expansion.coefficient` or call `project(...)` | `expansion.evaluate(coefficient, grid)` |
| `engine.build_realisation()` | `engine.build_expansion()` |
| representation-specific `*Realisation` classes | stateless `*Expansion` classes |

## Citation

If you use `styne` in academic work, please cite the archived version on Zenodo:
[![DOI](https://zenodo.org/badge/1278349844.svg)](https://zenodo.org/badge/latestdoi/1278349844) .
The DOI resolves to the latest archived release. For exact reproducibility,
please cite the DOI of the specific release version used.

## Licence

MIT. See [LICENSE](LICENSE).
