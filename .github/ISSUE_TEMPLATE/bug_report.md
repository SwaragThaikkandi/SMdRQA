---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

# Bug Report Template for SMdRQA

Start with a concise and informative title that summarizes the bug.
For example: "Error in SMdRQA module when processing certain inputs"

## Description

Provide a detailed description of the bug, including:
- Steps to reproduce the bug (if possible).
- Expected behavior vs. actual behavior.
- Any error messages or logs encountered.
- Use screenshots or screen recordings (if applicable) to visually illustrate the issue.

## Version

Specify the version of the SMdRQA package where the bug was encountered.
If applicable, mention the version of Python and any other relevant dependencies.

## Environment

Briefly describe your development environment, including:
- Operating system
- Python version
- Any other relevant dependencies or configurations

## Additional Information

Include any other relevant details that might help diagnose the bug.
This could involve code snippets, configuration files, or specific use cases where the bug occurs.

Here's an example of a bug report using this template:

---

**Title:** Error in SMdRQA module when processing certain inputs

**Description:**

When using the SMdRQA module to analyze specific input data, the program crashes with an error message.

**Steps to Reproduce:**

1. Import the SMdRQA module into a Python script.
2. Provide input data that triggers the error.
3. Execute the relevant function/method.

**Expected Behavior:**

The SMdRQA module should process the input data without errors and produce the expected output.

**Actual Behavior:**

The program crashes with an error message related to input validation.

**Version:**

- SMdRQA Package: v1.0.0
- Python: 3.9.6

**Environment:**

- Operating System: Windows 10
- Python Environment: Anaconda

**Additional Information:**

The error occurs consistently with specific types of input data. Included below is a code snippet demonstrating the issue:

```python
import SMdRQA

data = [...]  # Insert problematic input data here
result = SMdRQA.analyze(data)  # This line triggers the error
```
