# Suggested Model Adjustments

## Current State
The models have good foundational differences but could be made more distinct to better test our context predictions.

## Recommended Adjustments

### 1. Social Interactionist Model

**Current Strengths:**
- Two-stage transition
- Strong identity inertia
- Identity-based decisions

**Suggested Enhancements:**
```python
# Make role stability stronger
class InteractionistAgent:
    def update_identity(self):
        # Increase identity inertia over time
        self.identity_inertia *= (1 + self.time_in_role * 0.1)
        
    def calculate_cost(self):
        # More efficient when roles are stable
        if self.time_in_role > threshold:
            return base_cost * 0.5
```

### 2. Cognitive Model

**Current Strengths:**
- ILT updating
- Learning from observation
- Pattern recognition

**Suggested Enhancements:**
```python
class CognitiveAgent:
    def update_ilt(self):
        # Add exploration bonus
        exploration_value = self.calculate_novelty()
        self.learning_rate *= (1 + exploration_value)
        
    def decide_claim(self):
        # More willing to try new approaches
        if uncertainty_high:
            self.claim_threshold *= 0.8
```

### 3. Identity Model

**Current Strengths:**
- Group prototype
- Collective influence
- Group alignment

**Suggested Enhancements:**
```python
class IdentityModel:
    def update_group(self):
        # Stronger group alignment pressure
        alignment_force = self.calculate_group_cohesion()
        self.prototype_influence *= (1 + alignment_force)
        
    def handle_threat(self):
        # Quick consensus under threat
        if threat_present:
            self.prototype_learning_rate *= 1.5
```

## Expected Impact on Contexts

### 1. High Uncertainty
```
Cognitive:
- More exploration
- Faster learning
- Better adaptation

Interactionist:
- Even more rigid
- Clearer failure

Identity:
- Stronger consensus issues
- Clearer group-think
```

### 2. Resource Scarcity
```
Interactionist:
- More efficient stable roles
- Clearer benefits

Cognitive:
- More costly exploration
- Clearer resource drain

Identity:
- Better group conservation
- Clearer norms
```

### 3. Threat Context
```
Identity:
- Faster group alignment
- Stronger unity

Cognitive:
- Even slower decisions
- More individual focus

Interactionist:
- Clearer role transition cost
- Better late-stage performance
```

## Implementation Notes

1. **Keep Core Mechanisms**
   - Don't change fundamental approach
   - Just amplify key differences
   - Make trade-offs clearer

2. **Context Sensitivity**
   - Add explicit context awareness
   - Strengthen natural tendencies
   - Make failures more visible

3. **Measurement Focus**
   - Track mechanism-specific metrics
   - Measure trade-offs clearly
   - Document adaptation patterns 