# Contributing Guidelines

Thank you for your interest in contributing to this project! We welcome contributions from the community to help improve this toolkit. 

To ensure a smooth collaboration, please follow the guidelines outlined below. These guidelines are designed to align with the Journal of Open Source Software (JOSS) community standards.

## How to Report Bugs or Seek Support

If you encounter any bugs, unexpected behavior, or need help using the toolkit, please open an issue in the GitHub repository:

1. Go to the **Issues** tab on GitHub.
2. Click **New Issue**.
3. Use a clear and descriptive title.
4. Provide a detailed description of the problem, including:
   - Steps to reproduce the issue.
   - A minimal working example (MWE) demonstrating the bug.
   - The expected vs. actual behavior.
   - Any relevant traceback or log output.
   - Your system configuration (OS, Python version, dependency versions).

For general support questions or discussion, you can also open an issue or use the repository's Discussion section (if enabled).

## How to Submit Pull Requests

We welcome improvements to code, documentation, and tests. Please use the following workflow to submit your changes:

1. **Fork the Repository**: Create a personal copy of the repository on GitHub.
2. **Clone the Fork**: Clone your fork to your local machine:
   ```bash
   git clone https://github.com/rkutri/nelo.git
   cd nelo
   ```
3. **Set Up the Environment**: We use `uv` for dependency management. You can sync the development environment with:
   ```bash
   uv sync --extra plotting
   ```
4. **Create a Branch**: Create a descriptive feature branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. **Make Your Changes**: Write clean, readable code and document any new functions or classes. Ensure your code follows the project's design and style.
6. **Write and Run Tests**: All contributions must be accompanied by relevant tests. Before submitting a pull request, the test suite must pass using `pytest`. Run the test suite with:
   ```bash
   pytest tests/
   ```
   **Important**: Any pull request with failing tests or decreased test coverage will not be accepted.
7. **Commit Your Changes**: Use clear, concise commit messages.
8. **Push and Open a Pull Request**: Push your branch to your fork and submit a Pull Request (PR) to the `main` or development branch of the parent repository. Please provide a clear description of the changes and reference any related issues in the PR description.

## Code of Conduct

By participating in this project, you agree to abide by standard open-source community norms, maintaining a respectful, welcoming, and collaborative environment for everyone.
