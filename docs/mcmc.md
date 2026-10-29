# MCMC proposals and adaptation

A Metropolis-Hastings transition has two parts: a proposal draws a candidate,
and an acceptance rule determines whether to retain it. `ProposalMethod`
defines the proposal; `MetropolisHastings` combines it with the target and
acceptance rule. MRW, pCN, MALA, MLDA and DART use this common interface.

## Proposals

`propose(parameter, rng)` returns a `TransitionData` and the next random state.
The transition contains the current and proposed parameters and a correction
`log q_y(x) - log q_x(y)`. Both densities are taken relative to
`proposal.reference`; `None` denotes Lebesgue measure. A symmetric random walk
has zero correction relative to Lebesgue measure. For pCN, the correction is
zero relative to its Gaussian reference measure.

MH adds the target ratio relative to the same measure. An `EvaluatedState`
caches the parameter and its log density. For a Radon-Nikodym target, the cached
value is the log derivative relative to the target's reference measure.

Use `ProposalMethod.record(current, proposed, logCorrection)` to construct a
transition. The [MRW implementation](../src/styne/mcmc/method/mrw.py) gives a
minimal example. `condition(parameter)` prepares a proposal before each draw;
state-dependent proposal densities still require their full correction.
`for_target(target, previous)` updates settings inherited from a replaced target.

`sampler.with_target(...)` and `sampler.with_proposal(...)` return a sampler
with fresh history. Unchanged nested proposal components may be shared.

## Adaptation

Adaptation adjusts proposal settings from completed transitions. It belongs
in MH because MH knows the acceptance decision and the retained state, including
repetitions after rejection. An `Adaptation` strategy specifies the adjustment
rule. MH prepares the proposal, completes the transition with those settings,
then updates the adaptation state for the next step.

The sampler and proposal keep their classes: an adaptive MRW is still a
`MetropolisedRandomWalk` with an `MRWProposal`. The default `NoAdaptation`
leaves the proposal unchanged.

### Writing an adaptation

Subclass [`Adaptation`](../src/styne/mcmc/adaptation.py) and implement:

| Method | Role |
| --- | --- |
| `initial_state(parameter, proposalMethod)` | Validate and initialise numerical state. |
| `prepare_proposal(proposalMethod, state)` | Configure the next transition's proposal. |
| `update_state(state, nextState, transitionData)` | Return the updated numerical state. |

Keep fixed configuration on the strategy and evolving quantities in a small
immutable container, such as `RobbinsMonroState`. MH carries this container in
`EvaluatedState.adaptationState`. `nextState` is the retained chain state;
`transitionData` supplies both endpoints, the outcome and log acceptance
probability. The strategy defines any aggregation across batched chains.

For JAX compilation, use backend arrays and preserve the state's structure,
shapes and dtypes. Return new values without mutating the strategy, proposal
template or input state. Avoid Python branching and scalar conversions on
changing arrays. Proposal constructors must preserve these arrays too.

This interface updates proposal settings within the existing MH transition.
Algorithms that change its structure require their own sampler. In particular,
adapting a nested inner chain needs a separate correctness argument.

### Running an adaptive sampler

Assign the strategy to `sampler.adaptation` or `factory.adaptation`.
`RobbinsMonroAdaptation` in `styne.utility` adjusts IID MRW variance or pCN beta
from the mean acceptance across a batch. `stopAfter=1000` fixes the scale after
1000 transitions; omitting it permits continued adaptation. Other strategies
may update periodically or conditionally. Convergence requires conditions on
the chosen rule; stabilising settings alone do not establish it.

`run` starts a fresh chain; `continue_run` preserves its numerical state and
RNG. Warm-up remains in the stored chain and must be discarded when analysing
samples. `freeze_adaptation()` returns an optional fixed-proposal copy with
fresh history. For explicit execution, pass its final state and RNG as
`freeze_adaptation(state, rng)`.

`currentState` exposes the eager runner's full state; `lastState` is its
parameter. `sampler.proposal` gives the effective proposal after an eager run;
`proposal_at(state)` gives it for an explicit state. Treat adapted proposals as
snapshots. Replacing `sampler.adaptation` or assigning `proposalCovariance` on
MRW clears eager history. Reinitialise numerical state and rebuild compiled
functions after changing static configuration.

## Compiled execution

JAX can compile a trajectory once and reuse it across chunks:

```python
execute = sampler.compiled_trajectory(1000)
first = execute(initialParameter, rng)
second = execute(first.finalState, first.rng)
```

`finalState` preserves cached evaluations and adaptation state. `final` contains
only the parameter; passing it back starts fresh adaptation. Keep the sampler's
configuration fixed while using `execute`. This path leaves the eager runner,
stored chain and diagnostics untouched.

The result also contains `coordinates`, `readouts`, `sums` and `rng`. Optional
`thinning` retains every specified number of transitions and must divide the
trajectory length. A `readout` maps each retained chain state to an array on
every transition; `sums` counts acceptances per chain. NumPy and PyTorch use
the eager runner. See the [trajectory benchmark](../benchmarks/trajectory.py)
for compilation reuse.

## Migration within 0.3.0

- Custom proposals supply `logCorrection` through `TransitionData`.
- Replace the Robbins-Monro wrapper or adaptive MRW subclass with an ordinary
  sampler and its `adaptation` member. Factories also accept this member.
  `adaptOffset` and `adaptDecay` become `offset` and `decay`.
- Use `sampler.freeze_adaptation(state, rng)` in place of the wrapper's `freeze`.
- Access trajectory results by name. `finalState` is the new continuation
  state; positional unpacking must account for the added field.
- `SurrogateTransitionMeasure.transition_trajectory` returns the first and last
  evaluated states, coordinates and RNG. For an RN target, use its
  `evaluate_log` when the full density is needed rather than the cached derivative.

## Example: adapting an MRW covariance

Suppose a diagonal MRW proposal should follow the chain's marginal variances.
After every ten transitions, set each proposal variance to its empirical
estimate plus a positive `nugget`. Between updates, retain the current settings.
The nugget keeps every proposal variance positive, even after repeated rejections.

The [complete example](../examples/adaptation.py) implements this rule as a
`CovarianceAdaptation` strategy:

```python
from examples.adaptation import CovarianceAdaptation

sampler = MetropolisedRandomWalk(
    target, initialCovariance, diagnostics,
    adaptation=CovarianceAdaptation(interval=10, nugget=0.01),
)
sampler.run(1000, initialParameter)
sampler.continue_run(5000)
```

Internally, `WelfordAccumulator` accumulates marginal moments from every retained
state, including repetitions after rejection. Its pure record operations allow
these moments to pass through JAX loops; its ordinary eager interface uses the
same recurrence. The strategy prepares an `MRWProposal` with the existing
`DiagonalCovarianceMatrix`, which uses componentwise square roots. Storage and
updates are linear in dimension; no dense factorisation is needed.

The example handles one chain and learns coordinate scales, without estimating
correlations. It continues adapting and illustrates the interface rather than
establishing convergence of this rule.
