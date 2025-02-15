# Contributing to spacelab

Thank you for your interest in contributing to this project! We welcome contributions from the open-source community, whether you're fixing bugs, adding features, improving documentation, or helping with discussions.

## How to Contribute

### 1. Report Issues
If you find a bug, have a feature request, or need clarification, please open an issue on GitHub:
- [Search existing issues](https://github.com/junkwax/spacelab/issues) before submitting a new one.
- If your issue is new, create one with a **clear title** and **detailed description**.

### 2. Fork the Repository
To make changes, first fork the repository:
```bash
git clone https://github.com/junkwax/spacelab.git
cd spacelab

Create a new branch for your changes:

git checkout -b feature-name

3. Make Your Changes
	•	Follow the PEP 8 style guide for Python code.
	•	Write clear and concise commit messages.
	•	Ensure any new code includes comments and docstrings.

4. Run Tests

Before submitting a pull request, run the test suite to ensure everything works correctly:

pytest tests/

If you’re adding a new feature, consider writing new tests.

5. Submit a Pull Request

Once your changes are ready:

git push origin feature-name

Then, open a pull request (PR) on GitHub:
	•	Provide a clear title and description of your changes.
	•	Reference relevant issues (if applicable).
	•	Ensure all checks pass before requesting a review.

6. Code Review and Discussion

A project maintainer will review your PR, provide feedback, and request changes if needed. Be open to discussions and improvements!

Development Guidelines

Code Style
	•	Python: Follow PEP 8 and use black for formatting.

Directory Structure
	•	src/: Main source code.
	•	configs/: Configuration files.
	•	tests/: Unit tests.
	•	examples/: Example input/output files.

Commit Message Format

Use concise and informative commit messages:

feat: Add new black hole simulation model
fix: Resolve bug in dark matter Lagrangian calculations
docs: Update README with new installation steps

Getting Help

If you have questions, feel free to:
	•	Ask in GitHub Discussions.
	•	Open an issue if your question relates to a bug or feature request.

Code of Conduct

Please be respectful and professional when interacting with the community. We follow the Contributor Covenant Code of Conduct.

License

By contributing, you agree that your code will be licensed under the MIT License.

Thank you for contributing!