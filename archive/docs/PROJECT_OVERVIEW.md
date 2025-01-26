# Leadership Emergence Model Analysis Framework

## Project Overview

This framework provides tools for simulating and analyzing leadership emergence patterns through agent-based modeling. It focuses on the claim-grant dynamics of leadership emergence while supporting multiple theoretical perspectives and contextual variations.

## System Architecture

### 1. Model Layer (`src/models/`)
- **Base Model**: Core claim-grant mechanics
- **Perspectives**: Different theoretical implementations
  - Social Interactionist
  - Identity-based (planned)
  - Cognitive (planned)
- **Contexts**: Situational modifiers
  - Base Context
  - Crisis Context
  - Others planned

### 2. Simulation Engine (`src/simulation/`)
- **Runner**: Manages simulation execution
  - Parameter sweep capabilities
  - Context integration
  - Result collection
- **State Management**: Tracks agent states and interactions
- **History**: Records simulation trajectories

### 3. Analysis Tools (`scripts/`)
- **Parameter Sweep**: Systematic exploration
- **Metrics Calculation**: Leadership emergence measures
- **Pattern Detection**: Identify emergence patterns
- **Visualization**: Result plotting and animation

## Key Features

### 1. Base Leadership Model
- Claim-grant interaction mechanism
- Leadership score tracking
- Flexible parameter configuration

### 2. Context System
- Modular context implementation
- Dynamic behavior modification
- Environmental effects simulation

### 3. Analysis Pipeline
- Parameter space exploration
- Multiple replication support
- Statistical analysis tools
- Result visualization

## Directory Structure

```
abm-lead-emergence/
├── src/
│   ├── models/
│   │   ├── base_model.py
│   │   └── perspectives/
│   │       └── social_interactionist.py
│   └── simulation/
│       ├── runner.py
│       └── contexts/
│           ├── base.py
│           └── crisis.py
├── scripts/
│   ├── test_base_model.py
│   └── parameter_sweep.py
├── outputs/
│   └── parameter_sweep/
│       ├── results/
│       └── figures/
└── docs/
    ├── PROJECT_OVERVIEW.md
    ├── MODELS.md
    ├── PIPELINE.md
    └── TESTING.md
```

## Workflow

### 1. Model Development
1. Implement theoretical perspective
2. Add context variations
3. Configure parameters
4. Validate behavior

### 2. Simulation
1. Define parameter space
2. Set up contexts
3. Run parameter sweep
4. Collect results

### 3. Analysis
1. Calculate metrics
2. Detect patterns
3. Visualize results
4. Generate insights

## Configuration

### Parameter Configuration
```python
{
    "model_params": {
        "n_agents": 4,
        "claim_multiplier": 0.7,
        "grant_multiplier": 0.6,
        "success_boost": 5.0,
        "failure_penalty": 3.0,
        "grant_penalty": 2.0
    },
    "simulation_params": {
        "n_steps": 100,
        "n_replications": 5
    }
}
```

### Context Configuration
```python
{
    "context_type": "crisis",
    "params": {
        "intensity": 0.7,
        "claim_boost": 1.5,
        "grant_boost": 1.3
    }
}
```

## Extending the Framework

### 1. Adding New Models
1. Inherit from BaseLeadershipModel
2. Implement theoretical mechanisms
3. Add parameter configuration
4. Create test cases

### 2. Adding New Contexts
1. Inherit from Context base class
2. Implement modification methods
3. Configure parameters
4. Test with base model

### 3. Adding Analysis Tools
1. Create analysis script
2. Define metrics
3. Implement visualization
4. Document usage

## Current Status

### Completed Features
- Base leadership model
- Social interactionist perspective
- Crisis context
- Parameter sweep framework
- Basic analysis tools

### In Progress
- Identity-based perspective
- Advanced metrics
- Pattern detection
- Visualization tools

### Planned Features
- Cognitive perspective
- Network effects
- Dynamic contexts
- Interactive visualization

## Best Practices

### 1. Code Organization
- Clear module structure
- Consistent naming
- Comprehensive documentation
- Type hints

### 2. Testing
- Unit tests for components
- Integration tests
- Parameter validation
- Context verification

### 3. Documentation
- Code comments
- API documentation
- Usage examples
- Result interpretation

## Contributing

### Process
1. Fork repository
2. Create feature branch
3. Implement changes
4. Add tests
5. Submit pull request

### Guidelines
- Follow code style
- Add documentation
- Include tests
- Update relevant docs 