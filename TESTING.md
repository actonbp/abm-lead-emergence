# Testing Documentation

## Overview

This document outlines the testing strategy for the leadership emergence simulation project, focusing on the base model, contexts, and parameter sweeps.

## Test Structure

### 1. Model Tests
- Base leadership model behavior
- Context modifications
- Parameter validation
- State management

### 2. Integration Tests
- End-to-end simulations
- Parameter sweeps
- Context integration
- Result collection

### 3. Analysis Tests
- Metrics calculation
- Pattern detection
- Result validation

## Running Tests

### Basic Commands
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific components
pytest tests/test_base_model.py
pytest tests/test_contexts.py
pytest tests/test_parameter_sweep.py
```

## Test Categories

### 1. Base Model Tests (`tests/test_base_model.py`)
```python
def test_claim_probability():
    """Test claim probability calculation."""
    model = BaseLeadershipModel(n_agents=4)
    prob = model.calculate_claim_probability(agent_id=0)
    assert 0 <= prob <= 1

def test_grant_probability():
    """Test grant probability calculation."""
    model = BaseLeadershipModel(n_agents=4)
    prob = model.calculate_grant_probability(granter_id=1, claimer_id=0)
    assert 0 <= prob <= 1

def test_score_updates():
    """Test leadership score updates."""
    model = BaseLeadershipModel(n_agents=4)
    initial_score = model.agents[0].lead_score
    model.update_scores(claimer_id=0, granter_id=1, success=True)
    assert model.agents[0].lead_score > initial_score
```

### 2. Context Tests (`tests/test_contexts.py`)
```python
def test_crisis_context():
    """Test crisis context modifications."""
    context = CrisisContext(intensity=0.7)
    base_prob = 0.5
    modified_prob = context.modify_claim_probability(base_prob, agent_id=0)
    assert modified_prob > base_prob

def test_context_integration():
    """Test context integration with model."""
    model = BaseLeadershipModel(n_agents=4)
    context = CrisisContext(intensity=0.7)
    model.set_context(context)
    # Run simulation and check effects
```

### 3. Parameter Sweep Tests (`tests/test_parameter_sweep.py`)
```python
def test_parameter_combinations():
    """Test parameter combination generation."""
    param_grid = {
        "n_agents": [4, 6],
        "claim_multiplier": [0.5, 0.7]
    }
    combinations = generate_param_combinations(param_grid)
    assert len(combinations) == 4

def test_metrics_calculation():
    """Test leadership emergence metrics."""
    history = run_simulation(steps=100)
    metrics = calculate_metrics(history)
    assert "time_to_first_leader" in metrics
    assert "num_leaders" in metrics
```

## Test Fixtures

### 1. Model Fixtures
```python
@pytest.fixture
def base_model():
    """Create base model for testing."""
    return BaseLeadershipModel(
        n_agents=4,
        claim_multiplier=0.7,
        grant_multiplier=0.6
    )

@pytest.fixture
def crisis_context():
    """Create crisis context for testing."""
    return CrisisContext(
        intensity=0.7,
        claim_boost=1.5,
        grant_boost=1.3
    )
```

### 2. Simulation Fixtures
```python
@pytest.fixture
def simulation_history():
    """Generate simulation history for testing."""
    model = base_model()
    return model.run(steps=100)

@pytest.fixture
def parameter_sweep_results():
    """Generate parameter sweep results for testing."""
    return run_parameter_sweep(
        n_steps=50,
        n_replications=3
    )
```

## Test Coverage Goals

### Critical Components (100% Coverage)
1. Base Model
   - Claim/grant calculations
   - Score updates
   - State management

2. Contexts
   - Probability modifications
   - State update modifications
   - Parameter validation

3. Parameter Sweep
   - Parameter combination generation
   - Result collection
   - Metrics calculation

### General Coverage Goals
- Line coverage: >90%
- Branch coverage: >85%
- Documentation coverage: 100%

## Writing New Tests

### Guidelines
1. Test core functionality first
2. Include edge cases
3. Validate parameter ranges
4. Check state consistency
5. Verify metric calculations

### Example Test Structure
```python
def test_feature():
    """
    Test description:
    1. What is being tested
    2. Expected behavior
    3. Edge cases considered
    """
    # Setup
    model = setup_test_model()
    
    # Execute
    result = model.some_feature()
    
    # Assert
    assert_expected_behavior(result)
```

## Debugging Tests

### Common Issues
1. Random seed inconsistency
2. Parameter validation errors
3. State tracking issues
4. Metric calculation errors

### Debugging Tools
```bash
# Verbose output
pytest -v

# Print statements
pytest -s

# Debug on error
pytest --pdb

# Show coverage
pytest --cov=src --cov-report=term-missing
```

## Future Improvements

### 1. Test Extensions
- Property-based testing
- Mutation testing
- Performance benchmarks
- Visual test reports

### 2. Coverage Improvements
- Additional edge cases
- Error scenarios
- Parameter combinations
- Context interactions

### 3. Documentation
- Test case descriptions
- Coverage reports
- Debugging guides
- Best practices 