## Model Overview

### Base Model (v1.0)
The base model implements core claim-grant mechanics to serve as a foundation for all perspective models.

### Key Features
- Pure matching between characteristics and Implicit Leadership Theories (ILTs)
- Identity updates that don't affect decisions
- Continuous schema space with normal distributions

### Implemented Perspectives

1. **Social Interactionist Model**
   - **Status**: ✅ Implemented
   - **Mechanism**: Identity Development
   - **Process**: Repeated interactions → Role Identities → Stable Structure
   - **Key Feature**: Two-stage transition from schemas to identities
   - **Parameters**:
     - `dyadic_interactions_before_switch`: When to transition (5-20 steps)
     - `identity_update_rate`: Identity change speed (0.1-0.8)
     - `perception_update_rate`: Perception change speed (0.1-0.8)

2. **Cognitive Model**
   - **Status**: ✅ Implemented
   - **Mechanism**: Social Learning
   - **Process**: Observation → ILT Adaptation → Prototype Convergence
   - **Key Feature**: Schema adaptation through observation
   - **Parameters**:
     - `ilt_learning_rate`: Schema adaptation speed (0.1-0.5)
     - `dyadic_interactions_before_switch`: When to start learning (5-50 steps)

3. **Social Identity Model**
   - **Status**: 📝 Planned
   - **Mechanism**: Group Prototypes
   - **Process**: Group Identity → Shared Prototypes → Collective Leadership
   - **Key Feature**: Group-level processes
   - **Parameters**: TBD

### Control Model
- **Status**: ✅ Implemented
- Random claim/grant decisions
- No underlying mechanism
- Baseline for comparing emergence patterns

### Parameter Standardization
All perspective models inherit from the base model and share these core parameters:

1. **Simulation Control**
   - `n_steps`: 10-500 steps
   - `n_agents`: 5-10 agents
   - `schema_dimensions`: 2-3 dimensions

2. **Matching Parameters**
   - `match_threshold`: 0.4-0.7
   - `match_algorithm`: "average" or "minimum"
   - `dimension_weights`: "uniform", "primary", or "sequential"

3. **Update Parameters**
   - `success_boost`: 3.0-30.0
   - `failure_penalty`: 2.0-25.0
   - `identity_inertia`: 0.1-0.5

4. **Base Probabilities**
   - `base_claim_probability`: 0.3-0.8

### Model Comparison Framework
Each perspective model is evaluated on:
1. Leadership emergence speed
2. Hierarchy stability
3. Role differentiation
4. Perception consensus

### Implementation Guidelines
1. **Inheritance**
   - All perspectives inherit from `BaseLeadershipModel`
   - Must implement `step()` and `process_leadership_claim()`
   - Should extend `ModelParameters` for new parameters

2. **Two-Stage Process**
   - Initial stage: Pure schema matching (base model behavior)
   - Perspective stage: Unique emergence mechanism

3. **Parameter Validation**
   - Must validate all parameters in `__post_init__`
   - Should enforce consistent ranges across models
   - Must handle parameter relationships

4. **State Tracking**
   - Track agent states consistently
   - Maintain interaction history
   - Calculate standard metrics

### Future Directions
1. **Model Integration**
   - Combine compatible mechanisms
   - Study interaction effects
   - Build comprehensive theory

2. **Context Effects**
   - Crisis vs. routine tasks
   - Short vs. long-term groups
   - Task complexity effects

3. **Validation**
   - Empirical pattern matching
   - Parameter sensitivity analysis
   - Cross-perspective comparisons 