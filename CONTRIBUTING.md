# Contributing to ClaudeGPT

Thank you for your interest in contributing to ClaudeGPT! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:

1. A clear, descriptive title
2. A detailed description of the issue
3. Steps to reproduce the bug
4. Expected behavior
5. Actual behavior
6. Screenshots (if applicable)
7. Environment information (OS, Python version, package versions)

### Suggesting Enhancements

We welcome suggestions for enhancements! Please create an issue with:

1. A clear, descriptive title
2. A detailed description of the proposed enhancement
3. Any relevant examples or mockups
4. An explanation of why this enhancement would be useful

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Add or update tests as necessary
5. Ensure all tests pass
6. Submit a pull request with a clear description of the changes

## Development Setup

1. Clone the repository
   ```
   git clone https://github.com/yourusername/claudegpt.git
   cd claudegpt
   ```

2. Create a virtual environment
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys (see `.env.example`)

## Coding Standards

- Follow PEP 8 style guidelines
- Write docstrings for all functions, classes, and modules
- Include type hints where appropriate
- Write tests for new functionality

## Testing

Run tests with:
```
python -m unittest discover
```

## Documentation

- Update documentation when changing functionality
- Use clear, concise language
- Include examples where appropriate

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT license.