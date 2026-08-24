# Contributing to styne

Thanks for considering a contribution. This file covers what to expect
before you write any code.

## Maintenance

The styne library is currently managed by a single maintainer. To keep the
project sustainable, review times may occasionally be slow and the library
must adhere to a deliberately narrow design and strict style guidelines.

Pull requests that fall outside this specific scope or style will typically be
closed rather than put through a lengthy revision cycle. This is entirely a
practical decision to manage workload rather than a critique of your code.

If you have developed a valuable feature that does not quite fit the core
project, you are strongly encouraged to fork the repository and build it
without these constraints. An independent fork that serves new use cases is a
highly successful outcome for the community. Leaving a star and sharing your
version with others is always appreciated.

## Requirements for acceptance

- **Tests.** All new features require unit tests. Changes affecting component
  interactions also require integration tests. Pull requests must include
  appropriate tests to qualify for review.
- **Style.** The codebase relies on strict uniformity. Every file must adhere
  to the formatting conventions detailed below to ensure long-term
  maintainability.
- **Design.** Components must remain as independent as possible. Submissions
  that tightly couple components to force a local solution will be rejected
  in favour of a decoupled approach.
- **Code Quality.** Submissions must prioritise clean architecture. Temporary
  workarounds, duplicated logic or unrefined hacks will not be merged even if the
  test suite passes.

## Explicitly welcomed contributions

- **New samplers, forward maps, or GP representations** that fit the existing
  interfaces (`MetropolisHastings`, `ForwardMap`, `GaussianProcess`, etc.).
  Please open an issue before writing code if you are unsure about structural
  compatibility.
- **Interface extensions.** If existing interfaces do not support specific
  surrogate classes, your model or likelihood, open an issue detailing the
  structural gap. Modifying interfaces to robustly accommodate new computational
  tools is a legitimate expansion of the library.
- **Performance improvements.** Optimisations require
  quantitative evidence. Please provide a benchmark demonstrating the
  improvement, ideally alongside a profile showing the previous computational
  bottleneck.
- **Architectural proposals.** Discussions regarding library architecture
  and design patterns are highly encouraged. If you identify a structural
  limitation or wish to propose a new design paradigm, please open an issue.
  Exploring the mathematical and software concepts together first guarantees
  that any subsequent coding effort is well directed.

## Code style

These conventions are strictly enforced to maintain codebase uniformity. Pull
requests failing to adhere to these rules will not be merged.

- **Naming conventions.** Use `camelCase` for variables and properties,
  `snake_case` for methods and functions, `PascalCase` for class names.
  Module filenames are lower-case with no separators at all.
- **Encapsulation.** Reserve a leading underscore for non-public members of a
  class: state and behaviour owned by an object are the level at which the
  convention communicates a meaningful boundary. Do not prefix module-level
  classes or free functions with an underscore merely to imply module privacy.
  A class used only as a component of another class may be nested when that
  ownership is intrinsic; otherwise give it an ordinary `PascalCase` name in
  the module. Give free functions ordinary `snake_case` names. Control the
  supported package API through curated exports and documentation, not through
  underscore-prefixed module declarations. Python protocol hooks such as
  `__getattr__` and `__dir__` are required exceptions.
- **Line width.** Maximum 90 characters.
- **Docstrings.** Please provide minimal and signal-dense docstrings for the
  core entry-points of the new features.
- **Comments.** Comments must explain the underlying reasoning rather than
  the mechanics. Delete any comment that simply describes the execution of the
  next line or a fixed parameter value. The code must be self-explanatory through
  structure and naming.

## Workflow

1. Fork the repository and clone your fork.

   ```bash
   git clone https://github.com/<you>/styne.git
   cd styne
   ```

2. Set up the development environment.

   ```bash
   uv sync --extra plotting --group dev
   ```

3. Branch from `main`.

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. Write your code in strict accordance with the style guidelines and include
   appropriate tests.

5. Run the specific tests for your changes followed by the full test suite.

   ```bash
   uv run --extra plotting pytest tests/
   ```

6. Open a pull request against `main`. Describe the changes, the reasoning and
   any interfaces the new code extends or relies upon.

## Reporting bugs

Open an issue containing a minimal working example that reproduces the problem.
Include the expected versus actual behaviour and specify your Python and styne
versions. Bug reports require a reproducing example to be actionable and will
remain on hold until one is provided. Where applicable, including a failing
regression test alongside your report is highly encouraged.

## Code of conduct

Contributors must adhere to standard open-source norms. Please remain
respectful and keep all discussions strictly focused on the code and the
underlying concepts.
