# Extending MCMC transitions

A proposal owns how a candidate is drawn and how its forward and reverse
densities differ. `MetropolisHastings` owns the acceptance rule. Keeping these
responsibilities separate lets a new proposal use the same acceptance,
batching, diagnostics and random-state handling as the existing methods.

Start with `MRWProposal` and `MetropolisedRandomWalk` in
[`mrw.py`](../src/styne/mcmc/method/mrw.py), then read
[`ProposalMethod`](../src/styne/mcmc/proposal.py),
[`MetropolisHastings`](../src/styne/mcmc/metropolishastings.py) and
[`MCMCSampler`](../src/styne/mcmc/sampler.py).

## Proposal contract

`propose(state, rng)` returns a `TransitionData` and the next random state.
Use `ProposalMethod.record(current, proposed, logCorrection)` to construct
the record. The correction is `log q_y(x) - log q_x(y)`, with both proposal
densities taken relative to `proposal.reference`. `None` denotes Lebesgue
measure. A symmetric random walk has zero correction relative to Lebesgue;
pCN has zero correction relative to the Gaussian measure it preserves.

The sampler combines that correction with the target ratio relative to the
same measure. Radon–Nikodym targets store their derivative, avoiding reference
evaluations when the proposal preserves that reference. Langevin, MLDA and
DART proposals supply their own corrections through this same contract.

`condition(state)` is called once before drawing a proposal. A conditioned
proposal may depend on held coordinates, which are identical in the forward
and reverse moves. It is not a licence to omit the correction for arbitrary
state-dependent parameters. `for_target(target, previous)` reconstructs
settings taken from the old target; explicitly supplied settings stay fixed.

`with_target` and `with_proposal` create fresh outer runner history and
diagnostics. They can share unchanged nested proposal components. Construct
separate nested samplers when their stateful diagnostics must be independent.

Block proposals propagate the random state from one block to the next. Their
corrections are converted to Lebesgue before summation, since the blocks may
use different reference measures. Partition rules preserve backend arrays and
batch axes when extracting and merging coordinates.

## Warm-up and production

`RobbinsMonroAdaptation` in `styne.utility` wraps an existing sampler. Its
numerical state contains the evaluated chain state, log scale and step count.
It delegates each transition to the ordinary sampler, then adjusts the scale.
This avoids a second implementation of acceptance inside an adaptive sampler.

The supported scales are an IID random-walk variance, including its covariance
scaling, and a pCN step size. A batch shares one scale, updated from its mean
acceptance; individual chains retain independent acceptance decisions.
Unsupported proposals raise an error rather than silently doing no adaptation.

```python
from styne.utility import RobbinsMonroAdaptation

warmup = RobbinsMonroAdaptation(sampler, targetAcceptance=0.3)
state = warmup.initial_state(initialParameter)
state, rng = warmup.run(1000, state, rng)
production = warmup.freeze(state, rng)
production.run(5000, state.parameter)
```

Pass the returned random state into `freeze` when continuing through `run`.
For explicit transitions, use `production.step(state.chainState, rng)`.
Warm-up does not populate the production chain. Freezing fixes the proposal;
it does not establish convergence or remove the need to assess mixing.
The complete example is [`examples/adaptation.py`](../examples/adaptation.py).

For a new adaptation method, keep its numerical history in its warm-up state
and return a fixed sampler for production. `Chain` and `Annotator` remain
runner storage and observation interfaces. They are not implicit history
inputs to a pure `step` or to a compiled trajectory.

## Compiled trajectories

`transformed_trajectory(nSteps, initialParameter, rng, thinning=1, readout=None)`
returns a `Trajectory` with `final`, `coordinates`, `readouts`, `sums` and `rng`.
Coordinates are retained every `thinning` transitions; thinning must divide
`nSteps`. An optional array-valued readout runs after every transition, and
the accepted-transition sums count every transition, separately per chain.
The stateful runner, stored chain and diagnostics are untouched.

For several chunks, create `run = sampler.compiled_trajectory(nSteps, ...)`
once and call it with each preceding result's `final` and `rng`. Compatible
input shapes and dtypes reuse compilation. Keep the sampler's configuration
fixed for the function's lifetime. The
[`trajectory benchmark`](../benchmarks/trajectory.py) compares this with
compiling a fresh function per chunk.

JAX supports this compiled path. NumPy and PyTorch use the eager runner;
explicit PyTorch generator state remains unsupported in transformed loops.

This is not a general speedup for eager execution. Constructing warm-up
proposals and converting between reference measures add work. In particular,
an RN target used with a Lebesgue proposal evaluates the reference at both
ends of a transition. The tradeoff buys one acceptance implementation and
explicit measure conversions; benchmark inexpensive targets before choosing
an execution path.

## Migration within 0.3.0

- Custom proposals must return `logCorrection`; sampler subclasses no longer
  implement method-specific `_log_mh_ratio` calculations.
- Replace `RobbinsMonroMRW` and `RobbinsMonroMRWFactory` with an ordinary
  `MetropolisedRandomWalk` or `MRWFactory` sampler and `RobbinsMonroAdaptation`.
  The former `adaptOffset` and `adaptDecay` settings become `offset` and `decay`.
- Replace three-value unpacking of `transformed_trajectory` with named access
  to `result.final`, `result.coordinates` and `result.rng`.
- `SurrogateTransitionMeasure.transition_trajectory` now returns the first and
  last evaluated states, coordinates and random state. An RN state's cached
  `logDensity` is its log derivative; use the target's `evaluate_log` when the
  full density is needed.
- Existing MLDA/DART factories retain `nChain`, `regularisation`, root tuning
  and partition configuration. Nested adaptation, mixture selection and a new
  Gibbs composition are outside this interface update. Their own acceptance
  statistics and scale-selection policies need a separate design decision.
