# Testing Documentation

This document describes the testing setup, procedures, and guidelines for the leadership emergence simulation project.

## Test Structure

The project uses several types of tests:

1. **Unit Tests**: Test individual components in isolation
   - Model behavior
   - Feature extraction
   - Data processing

2. **Integration Tests**: Test component interactions
   - End-to-end workflows
   - Error handling
   - Interruption recovery
   - Parameter space coverage
   - Reproducibility

3. **Performance Tests**: Test system scalability
   - Large simulations
   - Parallel processing
   - Memory usage

## Running Tests

### Basic Test Run
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_integration.py

# Run tests matching a pattern
pytest -k "test_end_to_end"
```

### Test Coverage
Coverage reports are generated in HTML format in the `htmlcov` directory. Open `htmlcov/index.html` to view:
- Line coverage
- Branch coverage
- Uncovered lines
- Coverage trends

### Continuous Integration
Tests run automatically on GitHub Actions:
- On every push to main
- On pull requests
- Tests run on Python 3.8, 3.9, and 3.10
- Coverage reports uploaded to Codecov

## Test Categories

### Unit Tests
- `test_base_model.py`: Tests base model functionality
- `test_schema_model.py`: Tests schema-based model implementation
- `test_feature_extraction.py`: Tests feature extraction methods

### Integration Tests
- `test_integration.py`: Tests complete system workflows
  - End-to-end simulation and analysis
  - Error handling
  - Interruption recovery
  - Parameter space coverage
  - Result reproducibility

### Performance Tests (TODO)
- Large-scale simulations
- Memory profiling
- Execution time benchmarks

## Writing Tests

### Guidelines
1. Each test should have a clear purpose
2. Use descriptive test names
3. Include docstrings explaining test scenarios
4. Use appropriate fixtures for setup/teardown
5. Test both success and failure cases
6. Test edge cases and boundary conditions

### Example Test Structure
```python
def test_something():
    """
    Test description explaining:
    - What is being tested
    - Expected behavior
    - Any special conditions
    """
    # Setup
    ...
    
    # Execute
    ...
    
    # Assert
    ...
```

### Using Fixtures
```python
@pytest.fixture
def test_data():
    """Create test data."""
    return ...

def test_with_fixture(test_data):
    """Use fixture in test."""
    assert ...
```

## Test Coverage Goals

- Line coverage: >90%
- Branch coverage: >85%
- Critical components: 100%

### Critical Components
- Model state management
- Parameter validation
- Data persistence
- Feature extraction
- Error handling

## Adding New Tests

When adding new features:
1. Write tests before implementation (TDD)
2. Cover both normal and error cases
3. Include performance considerations
4. Update documentation

## Debugging Failed Tests

1. Use `-v` flag for verbose output
2. Use `-s` flag to see print statements
3. Use `pytest.set_trace()` for debugging
4. Check coverage reports for gaps

## Performance Testing

### Tools
- cProfile for profiling
- memory_profiler for memory usage
- pytest-benchmark for benchmarking

### Running Performance Tests
```bash
# Run with profiling
python -m cProfile -o profile.stats tests/test_performance.py

# Run with memory profiling
python -m memory_profiler tests/test_performance.py

# Run benchmarks
pytest --benchmark-only
```

## Future Improvements

1. Add property-based testing
2. Expand performance test suite
3. Add mutation testing
4. Improve test documentation
5. Add visual test result reporting 