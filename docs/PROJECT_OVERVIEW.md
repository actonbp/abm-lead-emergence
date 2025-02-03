# Leadership Emergence Project Overview

## Background & Motivation

This project explores how leadership naturally emerges in groups through a computational modeling approach. The core premise is that leadership emerges through micro-level interactions where individuals:
1. Make leadership claims (attempting to influence others)
2. Grant or reject those claims (accepting/rejecting influence)

Rather than assuming leadership is a fixed trait or formally assigned role, we model it as an emergent phenomenon arising from these interactions. This aligns with modern organizational theory that views leadership as a dynamic, socially constructed process.

## Model Architecture

### Foundation: Base Model
The base model implements core claim-grant mechanics but does not produce emergence alone:
- Schema matching between characteristics and ILTs
- Basic perception tracking
- Simple interaction rules
- Provides foundation for testing emergence mechanisms

### Emergence Mechanisms
Each theoretical perspective adds a distinct mechanism on top of the base model:

1. **Social Interactionist**
   - Mechanism: Identity Development
   - Process: Repeated interactions → Role Identities → Stable Structure
   - Key Feature: Two-stage transition from schemas to identities
   - Emergence through: Role crystallization and mutual reinforcement

2. **Cognitive**
   - Mechanism: Social Learning
   - Process: Observation → ILT Adaptation → Prototype Convergence
   - Key Feature: Memory of successful interactions
   - Emergence through: Collective learning about effective leadership

3. **Identity** (In Progress)
   - Mechanism: Group Prototypes
   - Process: Group Identity → Shared Prototypes → Collective Leadership
   - Key Feature: Group-level processes
   - Emergence through: Social identity and prototype alignment

### Control: Null Model
- Random claim/grant decisions
- No underlying mechanism
- Baseline for comparing emergence patterns

### Contextual Differentiation (Planned)
Testing how different contexts favor different emergence mechanisms:
1. Task Types
   - Crisis vs. Routine
   - Creative vs. Structured
   - Complex vs. Simple

2. Group Characteristics
   - Size and composition
   - Time horizon
   - Prior relationships

3. Environmental Factors
   - Resource availability
   - External pressure
   - Uncertainty levels

## Current Implementation

### Core Components
1. Base Model (`src/models/base_model.py`)
   - Core interaction mechanics
   - No emergence mechanism
   - Foundation for perspectives

2. Emergence Perspectives (`src/models/perspectives/`)
   - `interactionist.py`: Identity development
   - `cognitive.py`: Learning/adaptation
   - `identity.py`: Group prototypes (in progress)

3. Control (`src/models/null_model.py`)
   - Random baseline
   - No mechanism

### Analysis Pipeline
1. Parameter Optimization
   - Optimize each mechanism separately
   - Context-specific optimization
   - Performance validation

2. Mechanism Comparison
   - Compare emergence patterns
   - Analyze effectiveness
   - Context sensitivity

3. Visualization
   - Emergence trajectories
   - Mechanism differences
   - Contextual effects

## Research Goals

1. **Theoretical Differentiation**
   - Show how different mechanisms produce emergence
   - Identify unique patterns for each perspective
   - Understand mechanism interactions

2. **Contextual Effectiveness**
   - Map mechanisms to contexts
   - Identify when each works best
   - Guide practical applications

3. **Integration**
   - Combine compatible mechanisms
   - Understand interaction effects
   - Build comprehensive theory

## Current Status

1. **Completed**
   - Base model implementation
   - Two emergence mechanisms:
     - Social Interactionist
     - Cognitive
   - Parameter optimization
   - Basic comparison

2. **In Progress**
   - Identity mechanism
   - Context implementation
   - Enhanced visualization

3. **Planned**
   - Context-mechanism mapping
   - Additional contexts
   - Mechanism combinations

## Development Process

1. **Mechanism Development**
   ```
   Base Model -> Add Mechanism -> Validate -> Compare
   ```

2. **Context Integration**
   ```
   Define Context -> Test Mechanisms -> Map Effectiveness
   ```

3. **Analysis Pipeline**
   ```
   Optimize -> Compare -> Map to Context
   ```

## Key Files

1. **Core Implementation**:
   - `src/models/base_model.py`: Foundation
   - `src/models/perspectives/*.py`: Emergence mechanisms
   - `src/models/null_model.py`: Control baseline

2. **Analysis Tools**:
   - `scripts/run_parameter_sweep.py`: Mechanism optimization
   - `scripts/compare_perspectives.py`: Pattern comparison
   - `scripts/visualize/*.py`: Mechanism analysis

3. **Configuration**:
   - `config/parameters.yaml`: Base parameters
   - `config/parameter_sweep.yaml`: Optimization settings

## Future Directions

1. **Technical**
   - Complete identity mechanism
   - Implement contexts
   - Enhanced visualization

2. **Theoretical**
   - Context-mechanism mapping
   - Mechanism combinations
   - Comprehensive framework

3. **Validation**
   - Context predictions
   - Mechanism patterns
   - Combined effects

## Key Files for Understanding

1. **Core Model Understanding**:
   - `src/models/base_model.py`: Basic leadership emergence mechanics
   - `src/models/perspectives/*.py`: Theoretical perspectives
   - `src/models/null_model.py`: Control condition

2. **Analysis Pipeline**:
   - `scripts/run_parameter_sweep.py`: Parameter optimization
   - `scripts/compare_perspectives.py`: Model comparison
   - `scripts/visualize_*.py`: Result visualization

3. **Documentation**:
   - `README.md`: Project overview and setup
   - `docs/MODELS.md`: Detailed model descriptions
   - `docs/ROADMAP.md`: Development plans

## Development Approach

1. **Model Development**
   ```
   Base Model -> Perspectives -> Validation -> Integration
   ```

2. **For Each Model**:
   ```
   Implement -> Optimize -> Validate -> Compare
   ```

3. **Analysis Pipeline**:
   ```
   Parameter Sweep -> Model Comparison -> Visualization
   ```

## Recommendations for New Contributors

1. Start with `README.md` for project overview
2. Review `src/models/base_model.py` to understand core mechanics
3. Look at `scripts/run_parameter_sweep.py` to see how models are optimized
4. Check `outputs/sweeps/` for current findings
5. Read `docs/MODELS.md` for theoretical background

## Future Directions

1. **Technical**
   - Identity model implementation
   - Enhanced visualization tools
   - Additional metrics

2. **Theoretical**
   - Additional perspectives
   - Context effects
   - Multi-team dynamics

3. **Validation**
   - Empirical comparisons
   - Pattern validation
   - Predictive testing 