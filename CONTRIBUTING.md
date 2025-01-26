# Contributing to Leadership Emergence ABM

Thank you for your interest in contributing to our project! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/abm-lead-emergence.git
   cd abm-lead-emergence
   ```

2. **Set Up Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Project Structure

Please review [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for a detailed explanation of the codebase organization.

## Adding New Features

### 1. New Models
- Add to `src/models/perspectives/`
- Follow existing model patterns
- Include comprehensive docstrings
- Add corresponding tests
- Update `MODELS.md`

### 2. New Contexts
- Add to `src/simulation/contexts/`
- Inherit from base Context class
- Include parameter validation
- Add test cases
- Update documentation

### 3. New Analysis Tools
- Add to `scripts/`
- Follow existing script patterns
- Include usage examples
- Update `PIPELINE.md`

## Code Style

1. **Python Style**
   - Follow PEP 8 guidelines
   - Use type hints
   - Write descriptive docstrings
   - Keep functions focused and small

2. **Documentation**
   - Update relevant documentation files
   - Include inline comments for complex logic
   - Add usage examples
   - Keep README.md current

3. **Testing**
   - Write unit tests for new features
   - Ensure all tests pass
   - Maintain test coverage
   - Document test cases

## Pull Request Process

1. **Before Submitting**
   - Update documentation
   - Add/update tests
   - Run test suite
   - Check code style

2. **Pull Request**
   - Create PR against main branch
   - Describe changes clearly
   - Reference related issues
   - List any breaking changes

3. **Review Process**
   - Address review comments
   - Keep PR focused
   - Update based on feedback
   - Ensure CI passes

## Development Workflow

1. **Issue First**
   - Create an issue for new features
   - Discuss implementation approach
   - Get feedback early

2. **Development**
   - Write tests first (TDD)
   - Implement feature
   - Document changes
   - Update relevant files

3. **Testing**
   - Run `pytest` locally
   - Check coverage
   - Validate functionality
   - Test edge cases

## Best Practices

### 1. Code Quality
- Write clear, readable code
- Follow project conventions
- Keep changes focused
- Document complex logic

### 2. Testing
- Write comprehensive tests
- Test edge cases
- Maintain coverage
- Document test scenarios

### 3. Documentation
- Keep docs up to date
- Include examples
- Explain complex features
- Update changelogs

## Questions or Problems?

- Open an issue for discussion
- Ask for clarification
- Propose improvements
- Report bugs

## License

By contributing, you agree that your contributions will be licensed under the MIT License. 