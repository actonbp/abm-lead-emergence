# Project Structure Documentation

## Overview

This document provides a detailed explanation of the project's directory structure and organization. The project is organized to separate core model logic, simulation framework, analysis tools, and documentation.

## Directory Structure

```
abm-lead-emergence/
├── src/                         # Source code
│   ├── models/                  # Model implementations
│   │   ├── base_model.py       # Core leadership model
│   │   ├── perspectives/       # Theoretical perspectives
│   │   └── parameters.py       # Parameter definitions
│   │
│   ├── simulation/             # Simulation framework
│   │   ├── runner.py          # Simulation execution
│   │   └── contexts/          # Context implementations
│   │
│   ├── analysis/              # Analysis tools
│   ├── utils/                 # Utility functions
│   └── visualization/         # Visualization tools
│
├── scripts/                    # Analysis scripts
│   ├── parameter_sweep.py     # Parameter exploration
│   ├── test_base_model.py     # Model testing
│   └── analyze_results.py     # Results analysis
│
├── tests/                     # Test suite
├── docs/                      # Documentation
└── outputs/                   # Simulation outputs
```

## Key Components

### 1. Source Code (`src/`)

#### Models (`models/`)
- `base_model.py`: Core implementation of the leadership emergence model
  - Claim-grant mechanics
  - Agent state management
  - Interaction logic
- `perspectives/`: Different theoretical implementations
  - Social interactionist perspective
  - Future: Identity-based, cognitive perspectives
- `parameters.py`: Parameter definitions and validation

#### Simulation (`simulation/`)
- `runner.py`: Manages simulation execution
  - Parameter sweep capabilities
  - Result collection
  - State tracking
- `contexts/`: Context implementations
  - Base context interface
  - Crisis context
  - Future contexts

#### Support Modules
- `analysis/`: Tools for analyzing simulation results
- `utils/`: Common utility functions
- `visualization/`: Data visualization tools

### 2. Scripts (`scripts/`)

#### Analysis Scripts
- `parameter_sweep.py`: Main script for parameter exploration
  - Parameter space definition
  - Multiple replications
  - Results collection
- `analyze_results.py`: Analysis of simulation results
  - Metric calculation
  - Pattern detection
  - Statistical analysis

#### Testing Scripts
- `test_base_model.py`: Basic model validation
  - Core functionality tests
  - Parameter validation
  - State consistency checks

### 3. Tests (`tests/`)
- Unit tests for components
- Integration tests
- Parameter validation tests
- Context verification tests

### 4. Documentation (`docs/`)
- `PROJECT_OVERVIEW.md`: High-level project description
- `MODELS.md`: Model documentation
- `PIPELINE.md`: Analysis pipeline details
- `TESTING.md`: Testing procedures
- `PROJECT_STRUCTURE.md`: This file

### 5. Outputs (`outputs/`)
- `parameter_sweep/`: Parameter sweep results
  - Raw simulation data
  - Statistical summaries
  - Visualization outputs

## Archive

The project maintains an archive of previous implementations and experimental features:

```
archive/
├── models/                     # Previous model implementations
│   ├── schema_model.py        # Schema-based approach
│   ├── network_model.py       # Network effects
│   └── schema_network_model.py # Combined approach
│
└── scripts/                    # Previous analysis scripts
    ├── run_feature_analysis.py # Feature extraction
    └── run_parameter_exploration.py # Early parameter exploration
```

These files are maintained for reference but are not part of the active codebase.

## Best Practices

### 1. Code Organization
- Keep related code together in modules
- Use clear, descriptive file names
- Maintain separation of concerns
- Document module purposes

### 2. File Placement
- New models go in `src/models/perspectives/`
- New contexts go in `src/simulation/contexts/`
- Analysis scripts go in `scripts/`
- Tests mirror source structure in `tests/`

### 3. Documentation
- Update relevant docs when adding features
- Keep README.md current
- Document all public interfaces
- Include usage examples

## Contributing

When adding new features:

1. **New Models**
   - Add to `src/models/perspectives/`
   - Update `MODELS.md`
   - Add corresponding tests

2. **New Contexts**
   - Add to `src/simulation/contexts/`
   - Update `MODELS.md`
   - Add test cases

3. **New Analysis**
   - Add scripts to `scripts/`
   - Update `PIPELINE.md`
   - Include documentation

## Future Organization

As the project grows, consider:

1. **Module Structure**
   - Separate packages for major components
   - Clear dependency management
   - Version control for models

2. **Documentation**
   - API documentation
   - Interactive examples
   - Contribution guidelines

3. **Testing**
   - Automated test suite
   - Performance benchmarks
   - Coverage reports 