# Leadership Emergence Models

## Base Leadership Model

### Overview
The base leadership model implements a claim-grant mechanism for leadership emergence, where agents can claim leadership and others can grant or deny those claims. This serves as the foundation for exploring different theoretical perspectives on leadership emergence.

### Core Components

1. **Agent State**
```python
class Agent:
    def __init__(self):
        self.lead_score = 50.0  # Leadership score (0-100)
        self.last_interaction = None  # Track recent interactions
```

2. **Interaction Dynamics**
- Claim Probability = base_prob * claim_multiplier
- Grant Probability = base_prob * grant_multiplier
- Score Updates:
  - Successful Claim: +success_boost
  - Failed Claim: -failure_penalty
  - Granting: -grant_penalty

### Parameters

```python
class ModelParameters:
    def __init__(self):
        # Core parameters
        self.n_agents = 4
        self.claim_multiplier = 0.7
        self.grant_multiplier = 0.6
        
        # Score adjustments
        self.success_boost = 5.0
        self.failure_penalty = 3.0
        self.grant_penalty = 2.0
```

### Key Findings

1. **Fast Leadership Emergence**
   - Small groups (4 agents)
   - High claim probability
   - Moderate grant probability
   - Strong success rewards

2. **Multiple Leaders**
   - Larger groups
   - Balanced claim/grant rates
   - Moderate penalties

## Context System

### Overview
The context system allows for modification of base model behavior to simulate different environmental conditions or situational factors.

### Base Context
```python
class Context:
    def modify_claim_probability(self, base_prob, agent):
        return base_prob
        
    def modify_grant_probability(self, base_prob, agent):
        return base_prob
        
    def modify_state_update(self, base_update, agent):
        return base_update
```

### Crisis Context Example
```python
class CrisisContext(Context):
    def __init__(self, intensity=0.7):
        self.intensity = intensity
        self.claim_boost = 1.5
        self.grant_boost = 1.3
        
    def modify_claim_probability(self, base_prob, agent):
        return base_prob * self.claim_boost * self.intensity
```

## Theoretical Perspectives

### 1. Social Interactionist Perspective

#### Key Features
- Focus on dyadic interactions
- Leadership through social recognition
- Emphasis on claim-grant sequences

#### Implementation
```python
class SocialInteractionistModel(BaseLeadershipModel):
    def calculate_claim_probability(self, agent, target):
        base_prob = super().calculate_claim_probability(agent, target)
        return base_prob * self.get_relationship_factor(agent, target)
```

### 2. Identity-Based Perspective (Planned)

#### Key Features
- Leadership identity construction
- Group prototypes
- Collective influence processes

#### Implementation Notes
- Extend base model with identity scores
- Add group prototype matching
- Implement collective influence mechanisms

### 3. Cognitive Perspective (Planned)

#### Key Features
- Learning from interactions
- Competence evaluation
- Dynamic schema updates

#### Implementation Notes
- Add learning mechanisms
- Track interaction outcomes
- Update agent beliefs

## Model Extensions

### 1. Network Effects
- Implement network structure
- Local vs global interactions
- Influence propagation

### 2. Dynamic Contexts
- Context switching
- Adaptive behaviors
- Environmental feedback

### 3. Multi-level Analysis
- Individual dynamics
- Group patterns
- System-level emergence

## Validation Approaches

### 1. Pattern Matching
- Compare with empirical findings
- Leadership structure emergence
- Temporal dynamics

### 2. Parameter Sensitivity
- Systematic parameter sweeps
- Interaction effects
- Robustness checks

### 3. Theory Testing
- Hypothesis formulation
- Controlled experiments
- Statistical analysis

## Future Developments

### 1. Model Enhancements
- Additional theoretical perspectives
- More complex contexts
- Advanced interaction rules

### 2. Analysis Tools
- Network analysis
- Pattern detection
- Visualization tools

### 3. Integration Features
- Multiple perspectives
- Context combinations
- Hybrid models

## Adding New Models

To add a new theoretical perspective:

1. **Create Model File**
```python
# src/models/perspectives/new_perspective.py
from ..base_model import BaseLeadershipModel

class NewPerspectiveModel(BaseLeadershipModel):
    def __init__(self, **params):
        super().__init__(**params)
        # Add perspective-specific initialization
    
    def modify_claim_probability(self, base_prob, agent):
        # Implement perspective-specific modifications
        return modified_prob
```

2. **Document Model**
   - Add section to this file
   - Describe theoretical background
   - Explain parameters
   - Detail modifications to base dynamics

3. **Add Tests**
   - Create test file
   - Test perspective-specific behavior
   - Verify parameter effects

4. **Update Configuration**
   - Add perspective to registry
   - Create default parameters
   - Update simulation runner

## Future Perspectives

### Planned Additions
1. **Cognitive Perspective**
   - Mental model influence
   - Learning and adaptation
   - Individual differences

2. **Identity Perspective**
   - Group identity effects
   - Prototype matching
   - Identity negotiation

### Contributing
To propose a new perspective:
1. Open an issue describing the perspective
2. Provide theoretical background
3. Outline key parameters
4. Suggest implementation approach 

# Model Documentation

## Base Model (v1.0)
The base model implements pure ILT-based claim-grant mechanics to serve as a control/baseline. 

### Key Features
- Pure matching between characteristics and Implicit Leadership Theories (ILTs)
- Identity updates that don't affect decisions
- Continuous schema space with normal distributions

### Implementation
- Location: `src/models/base_model.py`
- Results: `results/base_model_results.md`
- Visualizations: `outputs/parameter_sweep/`

### Key Findings
1. Identity Development
   - Leader identities show significant variance (2.409)
   - Follower identities show moderate variance (0.402)
   - Clear differentiation between leader and follower roles emerges

2. Leadership Structure
   - Weak but stable hierarchy (strength = 0.012)
   - Consistent rankings but fluctuating perceptions
   - High agreement between agents on leadership perceptions

3. Optimal Parameters
   - Group Size: 5 agents
   - Schema: 2D continuous space
   - Distribution Std: 0.169
   - Match Threshold: 0.570
   - Success Boost: 6.992
   - Failure Penalty: 2.299

### Limitations
- Hierarchy strength remains relatively weak
- Leadership perceptions show high fluctuation
- Identity development alone may be insufficient for strong leadership emergence

## Theoretical Perspectives (In Development)

### 1. Social Interactionist Model
Status: 🔄 In Progress
- Identity feedback into decisions
- Dynamic leader/follower role development
- Location: `src/models/social_interactionist.py`

### 2. Social Cognitive Model
Status: 📅 Planned
- Schema adaptation from observation
- Group-level learning about leadership
- Location: `src/models/social_cognitive.py`

### 3. Social Identity Model
Status: 📅 Planned
- Collective leadership prototype
- Group identity influence
- Location: `src/models/social_identity.py`

## Parameter Reference
See [`docs/parameter_reference.md`](docs/parameter_reference.md) for comprehensive documentation of all model parameters.

## Validation
See [`validation/criteria.py`](validation/criteria.py) for model validation criteria and tests. 