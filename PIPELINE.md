# Leadership Emergence ABM Pipeline

## Core Model Architecture: Claims and Grants

The model centers on the fundamental social process of leadership claims and grants, with different theoretical perspectives implemented through parameter configurations rather than separate model implementations.

### Base Mechanism
```python
class LeadershipEmergenceModel:
    """Core model centered on claim/grant dynamics."""
    
    def step(self):
        # Select interaction pair
        agent1, agent2 = self.select_interaction()
        
        # Core claim/grant mechanism
        claim = self.evaluate_claim(agent1, agent2)
        grant = self.evaluate_grant(agent2, agent1, claim)
        
        # Update identities and schemas based on interaction
        self.update_after_interaction(agent1, agent2, claim, grant)
```

## Parameter Space

### Core Parameters
```yaml
# Fundamental claim/grant parameters
interaction:
  claim_decision_rule: str    # How agents decide to claim
  grant_decision_rule: str    # How agents decide to grant
  threshold: float           # Base threshold for decisions

# Identity parameters
identity:
  leader_identity:
    initial_value: float     # Starting leadership identity
    update_rule: str        # How identity changes
  follower_identity:
    initial_value: float
    update_rule: str

# Schema parameters
schemas:
  type: str                 # ILT, group prototype, etc.
  update_mechanism: str     # How schemas evolve
  social_identity_influence: float  # Group-level effects
```

## Theoretical Perspectives Through Parameters

### 1. Social-Interactionist (SIP)
Uses stable ILTs and hierarchical/shared structures:
```yaml
sip_config:
  # Core mechanism
  interaction:
    claim_decision_rule: "ilt_threshold"
    grant_decision_rule: "single_leader"  # or "multiple_leaders"
  
  # Supporting parameters
  schemas:
    type: "ILT"
    update_mechanism: "static"
    social_identity_influence: 0.0  # Hierarchical
    # or 0.3 for shared leadership
```

### 2. Social-Cognitive (SCP)
Emphasizes dynamic identities and learning:
```yaml
scp_config:
  # Core mechanism
  interaction:
    claim_decision_rule: "identity_based"
    grant_decision_rule: "competence_based"
  
  # Supporting parameters
  identity:
    leader_identity:
      update_rule: "dynamic_feedback"
    follower_identity:
      update_rule: "dynamic_feedback"
  schemas:
    update_mechanism: "observational"
```

### 3. Social-Identity (SI)
Focuses on group prototypes and collective influence:
```yaml
si_config:
  # Core mechanism
  interaction:
    claim_decision_rule: "prototype_match"
    grant_decision_rule: "group_aligned"
  
  # Supporting parameters
  schemas:
    type: "group_proto"
    social_identity_influence: 0.7
    update_mechanism: "collective"
```

## Parameter Effects on Core Mechanisms

### Claim Evaluation
```python
def evaluate_claim(self, agent, target):
    """Evaluate claim probability using configured parameters."""
    base_probability = agent.leader_identity
    
    if self.config.schemas.type == "ILT":
        # SIP: ILT-based claims
        return base_probability * self.ilt_match_score(agent, target)
    elif self.config.schemas.type == "group_proto":
        # SI: Prototype-based claims
        return base_probability * self.prototype_match_score(agent)
    else:
        # SCP: Identity-based claims
        return base_probability * self.competence_score(agent)
```

### Grant Evaluation
```python
def evaluate_grant(self, agent, claimant, claim):
    """Evaluate grant probability using configured parameters."""
    base_probability = agent.follower_identity
    
    if self.config.schemas.update_mechanism == "observational":
        # SCP: Include observed performance
        return base_probability * self.observed_competence(claimant)
    elif self.config.schemas.social_identity_influence > 0:
        # SI: Include group influence
        return base_probability * self.group_alignment(claimant)
    else:
        # SIP: ILT-based granting
        return base_probability * self.ilt_match_score(agent, claimant)
```

## Theory Integration

### Hybrid Configurations
```yaml
hybrid_config:
  # Core mechanism remains unchanged
  interaction:
    claim_decision_rule: "weighted_combination"
    grant_decision_rule: "multi_factor"
  
  # Blend theoretical parameters
  schemas:
    type: "hybrid"
    components:
      ilt: 
        weight: 0.6  # SIP influence
      group_proto:
        weight: 0.4  # SI influence
    update_mechanism: "dynamic"  # SCP influence
```

## Best Practices

### 1. Parameter Selection
- Start with claim/grant rules
- Add minimal theoretical adjustments
- Document parameter interactions

### 2. Theory Testing
- Compare theories using same base mechanism
- Vary only necessary parameters
- Track emergence patterns

### 3. Documentation
```yaml
parameter_doc:
  name: "grant_decision_rule"
  core_mechanism: "Determines how agents evaluate leadership grants"
  theoretical_variants:
    sip: "single_leader - Hierarchical structure"
    scp: "competence_based - Learning from performance"
    si: "group_aligned - Prototype matching"
```

[Previous sections about pipeline structure remain unchanged...]
 