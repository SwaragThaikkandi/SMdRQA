# Coding Standards for SMdRQA

## Introduction

These coding standards are intended to ensure consistency, readability, and maintainability of the codebase for the SMdRQA project. All contributors are expected to adhere to these standards when making contributions to the project.

## Python Coding Standards

### Naming Conventions

- Use descriptive and meaningful names for variables, functions, classes, and modules.
- Follow the snake_case naming convention for variables and functions (e.g., `my_variable`, `calculate_score()`).
- Use PascalCase for class names (e.g., `MyClass`) and module names (e.g., `MyModule.py`).

### Code Formatting

- Use consistent indentation with four spaces for each level of indentation.
- Limit lines to a maximum of 79 characters to ensure readability.
- Use blank lines to separate logical sections of code for clarity.

### Comments and Documentation

- Include descriptive comments to explain complex code logic, algorithms, or tricky parts of the code.
- Use docstrings to provide documentation for functions, classes, and modules following the NumPy or Google style docstring conventions.

### Error Handling

- Use try-except blocks for handling exceptions and errors gracefully.
- Avoid using broad except clauses and handle specific exceptions whenever possible.

### Testing and Quality

- Write unit tests for all new features and bug fixes.
- Ensure all tests pass before submitting code for review.
- Use static code analysis tools such as pylint or flake8 to check for code quality and style compliance.

## Version Control Guidelines

- Follow a branching strategy such as Git Flow for managing code changes.
- Use meaningful commit messages that describe the purpose of each commit concisely.
- Keep commits focused on a single task or feature.

## Conclusion

By following these coding standards, we can maintain a high-quality codebase that is easy to understand, modify, and maintain. Thank you for your contributions to the SMdRQA project!

For more information on contributing to SMdRQA, please refer to the [Contribution Guidelines](https://github.com/SwaragThaikkandi/SMdRQA/blob/main/CONTRIBUTING.md).
