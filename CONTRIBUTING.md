# Contributing to CTINexus

Thank you for your interest in contributing to CTINexus! We welcome contributions in various forms, including bug reports, feature requests, documentation improvements, and code contributions.

## How to Contribute

### 🐛  Reporting Bugs

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

2. **Clone the Fork:** Clone your fork to your local machine:
   ```bash
   git clone https://github.com/YOUR-USERNAME/CTINexus
   cd CTINexus
   ```

3. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # .venv\Scripts\activate    # On Windows
   
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (OpenAI, Gemini, AWS)
   # OR set up Ollama for local models
   ```

5. **Create a Branch:** Create a new branch for your work:
   ```bash
   git checkout -b feature-name
   ```

6. **Make Changes:** Make your changes in your branch.

7. **Test your changes:** Ensure your changes work correctly:
   ```bash
   python app/app.py
   ```

8. **Commit Changes:** Commit your changes with a descriptive commit message. Use a category to indicate the type of change. Common categories include:

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


9. **Push to Fork:** Push your changes to your forked repository:
   ```bash
   git push origin feature-name
   ```

10. **Open a Pull Request:** Open a pull request from your fork to the main repository. Include a detailed description of your changes and any related issues.

## Code Style

Please follow the code style used in the project. We use [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code.

## Review Process

All pull requests are reviewed by our maintainers. We strive to provide feedback promptly, typically within a few days. Thank you for helping to improve CTINexus.
