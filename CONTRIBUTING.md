# Contributing to CTINexus

Thank you for your interest in contributing to CTINexus! We welcome contributions in various forms, including bug reports, feature requests, documentation improvements, and code contributions.

## How to Contribute

### 🐛 Reporting Bugs

If you find a bug, please open an issue on our GitHub repository. Provide as much information as possible, including:

- A clear and descriptive title.
- Steps to reproduce the bug.
- Expected and actual behavior.
- Screenshots, logs, or code snippets, if applicable.
- Environment details (OS, Python version, API provider).

### 💡 Suggesting Features

If you have an idea for a new feature or an improvement, please open an issue with the following details:

- A clear and descriptive title.
- A detailed description of the feature.
- Any relevant use cases or examples.

### 📖 Improving Documentation

Good documentation is key to a successful project. If you find areas in our documentation that need improvement, feel free to submit a pull request. Here are some ways you can help:

- Fix typos or grammatical errors.
- Clarify confusing sections.
- Add missing information.
- Update CLI documentation or add examples.

### Contributing Code

1. **Fork the Repository:** Fork the [repository](https://github.com/peng-gao-lab/CTINexus) to your own GitHub account.

1. **Clone the Fork:** Clone your fork to your local machine:

   ```bash
   git clone https://github.com/YOUR-USERNAME/CTINexus
   cd CTINexus
   ```

1. **Create a virtual environment and install dependencies:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # .venv\Scripts\activate    # On Windows

   pip install -e ".[dev]"
   ```

1. **Set up pre-commit hooks:**

   ```bash
   pre-commit install
   ```

   This installs git hooks that automatically run code quality checks before each commit.

1. **Configure environment variables:**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys (OpenAI, Gemini, AWS)
   # OR set up Ollama for local models
   ```

1. **Create a Branch:** Create a new branch for your work:

   ```bash
   git checkout -b feature-name
   ```

1. **Make Changes:** Make your changes in your branch.

1. **Run tests:** Run the test suite to ensure your changes don't break existing functionality:

   ```bash
   # Run all tests
   uv run pytest tests/ -v

   # Run with coverage report
   uv run pytest tests/ --cov=ctinexus --cov-report=term-missing
   ```

   All tests must pass before submitting a pull request. If you've added new features, include tests for them.

1. **Format and Lint your code:** Before committing, ensure your code follows our style guidelines:

   ```bash
   pre-commit run --all-files
   ```

1. **Test your changes manually:** Ensure your changes work correctly in practice:

   ```bash
   ctinexus
   ```

1. **Commit Changes:** Commit your changes with a descriptive commit message. Use a category to indicate the type of change. Common categories include:

   - `feat`: New feature
   - `fix`: Bug fix
   - `docs`: Documentation changes
   - `refactor`: Code refactoring
   - `test`: Adding or updating tests
   - `chore`: Maintenance tasks

   Example:

   ```bash
   git commit -m "feat: add support for new AI provider"
   ```

   If pre-commit hooks are installed, they will automatically run when you commit. If any checks fail, fix the issues and commit again.

1. **Push to Fork:** Push your changes to your forked repository:

   ```bash
   git push origin feature-name
   ```

1. **Open a Pull Request:** Open a pull request from your fork to the main repository. Include a detailed description of your changes and any related issues.

## Automated Checks

When you open a pull request, GitHub Actions will automatically run code quality checks. These checks must pass before your PR can be merged. If they fail:

1. Review the error messages in the GitHub Actions log
1. Run the checks locally: `ruff check . --fix && ruff format .`
1. Commit and push the fixes
1. The checks will run again automatically

## Code Quality & Style

Please follow the code style used in the project. We use [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code.

## Review Process

All pull requests are reviewed by our maintainers. We strive to provide feedback promptly, typically within a few days. Thank you for helping to improve CTINexus.
