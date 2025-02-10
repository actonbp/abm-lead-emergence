# Theoretical Perspectives Across Contexts

## How Agents Adapt to Leadership

### Base Adaptation Mechanics
When a leader emerges (high leader identity), agents modify their behavior:

1. **Followers**
   - Increase grant probability for leader's claims
   - Weight leader's information more heavily
   - Reduce their own claiming frequency
   - Conserve resources by following

2. **Leaders**
   - Share information more confidently
   - Operate more efficiently (lower costs)
   - Take more initiative in exploration
   - Guide group direction

### Social Learning Component
Agents learn over time that:
- Following effective leaders saves resources
- Leadership helps in specific contexts
- When to lead vs. when to follow
- How to adapt to different contexts

## Perspective-Context Alignment

### 1. Social Interactionist Perspective

**Strong at Explaining:**
- Resource Scarcity Context
  * Identity-based decisions reduce wasteful claims
  * Clear role differentiation saves resources
  * Stable identities prevent costly conflicts
  * Strong theoretical fit with efficiency focus

- Intergroup Threat Context
  * Quick identity formation under pressure
  * Clear leader-follower roles emerge
  * Group cohesion through identity alignment
  * Matches real group behavior under threat

**Weak at Explaining:**
- High Uncertainty Context
  * Too rigid once identities form
  * May lock in suboptimal leadership
  * Struggles with need for flexibility
  * Less adaptive to changing conditions

### 2. Cognitive Perspective

**Strong at Explaining:**
- High Uncertainty Context
  * Continuous learning and adaptation
  * Flexible leadership based on performance
  * Good at filtering signal from noise
  * Matches need for dynamic adjustment

- Intergroup Threat Context
  * Quick learning under pressure
  * Effective pattern recognition
  * Rapid adaptation to challenges
  * Good at crisis response

**Weak at Explaining:**
- Resource Scarcity Context
  * Learning process is resource-intensive
  * Constant adaptation costs energy
  * May be too exploratory
  * Less efficient than stable roles

### 3. Identity Perspective

**Strong at Explaining:**
- Intergroup Threat Context
  * Strong group identity formation
  * Clear in-group/out-group dynamics
  * Unified response to threats
  * Matches group psychology

- Resource Scarcity Context
  * Shared identity reduces conflicts
  * Efficient coordination through norms
  * Clear group boundaries
  * Natural resource management

**Weak at Explaining:**
- High Uncertainty Context
  * Too focused on group consensus
  * May resist necessary changes
  * Less individual adaptation
  * Could maintain incorrect beliefs

## Context-Specific Predictions

### 1. High Uncertainty Context

**Best Explained By: Cognitive Perspective**
- Continuous learning
- Flexible leadership
- Pattern recognition
- Adaptive behavior

**Why Others Struggle:**
- Interactionist: Too rigid
- Identity: Too consensus-focused

**Agent Adaptation:**
```
Under High Uncertainty:
- Increase learning rate
- More weight on recent performance
- Less stable leadership
- More exploration
```

### 2. Resource Scarcity Context

**Best Explained By: Social Interactionist**
- Clear role differentiation
- Stable relationships
- Efficient coordination
- Resource conservation

**Why Others Struggle:**
- Cognitive: Too exploratory
- Identity: Too consensus-focused

**Agent Adaptation:**
```
Under Resource Scarcity:
- Higher threshold for claiming
- Strong leader efficiency bonus
- More selective granting
- Focus on conservation
```

### 3. Intergroup Threat Context

**Best Explained By: Identity Perspective**
- Strong group cohesion
- Quick consensus
- Clear leadership
- Unified action

**Why Others Struggle:**
- Cognitive: Too slow
- Interactionist: Too individual

**Agent Adaptation:**
```
Under Threat:
- Lower leadership threshold
- Faster decision-making
- Higher group alignment
- Speed over accuracy
```

## Implementation Implications

### 1. Agent Learning Rules
```
def adapt_to_context(agent, context):
    if context == "high_uncertainty":
        agent.learning_rate *= 1.5
        agent.exploration_rate *= 1.2
    elif context == "resource_scarcity":
        agent.claim_threshold *= 1.3
        agent.efficiency_bonus *= 1.5
    elif context == "threat":
        agent.decision_speed *= 1.5
        agent.group_alignment *= 1.3
```

### 2. Leadership Effects
```
def modify_leadership_impact(leader, context):
    if context == "high_uncertainty":
        leader.influence_weight = moderate
        leader.stability = low
    elif context == "resource_scarcity":
        leader.influence_weight = high
        leader.efficiency_bonus = very_high
    elif context == "threat":
        leader.influence_weight = very_high
        leader.decision_speed = very_high
```

## Research Implications

1. **Theory Development**
   - Different mechanisms for different contexts
   - Need for integrated perspective
   - Context-specific predictions
   - Boundary conditions

2. **Empirical Testing**
   - Context as moderator
   - Perspective effectiveness
   - Leadership emergence patterns
   - Group adaptation

3. **Practical Applications**
   - Context-appropriate leadership
   - Group structure design
   - Training and development
   - Organizational design 