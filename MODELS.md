# Leadership Emergence Models

This document describes the theoretical background and implementation details of each leadership emergence model in the framework.

## Schema Model

### Theoretical Background
The Schema Model is based on Identity Control Theory and Leadership Categorization Theory. It posits that leadership emergence occurs through the interaction of:
1. Individual leadership schemas (mental models of leadership)
2. Identity claims and grants in social interaction
3. Dynamic adjustment of self-views and other-views

### Model Dynamics
- **Leadership Identity (LI)**: Represents an agent's self-view as a leader
- **Followership Identity (FI)**: Represents an agent's self-view as a follower
- **Schema Activation**: Determines how strongly leadership/followership schemas influence behavior
- **Identity Claims**: Actions that assert leadership
- **Identity Grants**: Recognition of leadership by others

### Parameters
- `n_agents`: Number of agents in the simulation
- `initial_li_equal`: Whether agents start with equal leadership identities
- `li_change_rate`: Rate of leadership identity change
- `schema_weight`: Influence of schemas on behavior
- `interaction_radius`: How many agents each agent interacts with
- `claim_threshold`: Threshold for making leadership claims
- `grant_threshold`: Threshold for granting leadership recognition

### Implementation Details
The model updates in discrete time steps:
1. Agents assess their current identities
2. Schema activation is calculated
3. Agents make leadership claims based on their identities
4. Other agents respond with identity grants
5. Identities are updated based on the interaction outcomes

### Emergent Patterns
The model can produce several leadership emergence patterns:
1. **Centralized Leadership**: One dominant leader emerges
2. **Shared Leadership**: Multiple agents maintain high leadership identities
3. **Rotating Leadership**: Leadership shifts among agents over time
4. **Failed Leadership**: No clear leadership structure emerges

## Base Leadership Model

### Theoretical Background
The Base Leadership Model implements fundamental leadership emergence dynamics without specific theoretical commitments. It serves as:
1. A baseline for comparing more complex models
2. A template for implementing new models
3. A simple test case for the framework

### Model Dynamics
- Basic identity updates based on interactions
- Simple claim and grant mechanisms
- Linear identity change functions

### Parameters
- `n_agents`: Number of agents
- `initial_li_equal`: Whether agents start with equal leadership identities
- `li_change_rate`: Rate of leadership identity change

### Implementation Details
Simplified version of the schema model:
1. Agents interact randomly
2. Identity updates are linear
3. No schema influences
4. Basic claim/grant mechanics

## Adding New Models

To add a new model:

1. **Create Model File**
   ```python
   # src/models/new_model.py
   from .base_model import BaseLeadershipModel
   
   class NewModel(BaseLeadershipModel):
       def __init__(self, **params):
           super().__init__(**params)
           # Add model-specific initialization
   
       def step(self):
           # Implement model dynamics
           return self.get_state()
   ```

2. **Document Model**
   - Add section to this file
   - Describe theoretical background
   - Explain parameters
   - Detail implementation

3. **Add Tests**
   - Create test file
   - Test model dynamics
   - Verify parameter effects

4. **Update Configuration**
   - Add model to registry
   - Create default parameters
   - Update analysis pipeline

## Model Comparison

### Key Differences
| Feature           | Base Model | Schema Model |
|------------------|------------|--------------|
| Complexity       | Low        | Medium       |
| Schema Influence | No         | Yes          |
| Identity Updates | Linear     | Non-linear   |
| Interactions     | Random     | Structured   |

### When to Use Each Model
- **Base Model**:
  - Quick prototyping
  - Testing framework features
  - Simple scenarios
  
- **Schema Model**:
  - Realistic leadership emergence
  - Testing theoretical predictions
  - Complex group dynamics

## Future Models

### Planned Additions
1. **Network Model**
   - Leadership emergence in networks
   - Influence of network structure
   - Dynamic network adaptation

2. **Cultural Model**
   - Cultural influences on leadership
   - Cross-cultural differences
   - Cultural adaptation

3. **Cognitive Model**
   - Detailed mental models
   - Learning and adaptation
   - Individual differences

### Contributing
To propose a new model:
1. Open an issue describing the model
2. Provide theoretical background
3. Outline key parameters
4. Suggest implementation approach 