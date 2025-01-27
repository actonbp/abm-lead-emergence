# Base Model Results (v1.0)

## Model Description

The base model implements a simple agent-based model of leadership emergence with the following key features:

1. **Agent Characteristics**: 
   - Each agent has a characteristic vector and an ILT vector in a continuous schema space
   - Initial characteristics and ILTs are drawn from a normal distribution
   - Leader and follower identities are initialized with some variation around 50

2. **Decision Making**:
   - Claims and grants are based solely on ILT matching
   - No influence from identities on decisions
   - Match threshold determines minimum similarity required for positive interactions

3. **Updates**:
   - Identities update after successful/failed interactions
   - Both leader and follower identities receive full success boosts
   - Failed claims result in penalties to both identities
   - Observers learn from successful claims with reduced effect

## Key Findings

### Identity Development
- Leader identities show significant variance (2.409)
- Follower identities show moderate variance (0.402)
- Clear differentiation between leader and follower roles emerges

### Leadership Structure
- Weak but stable hierarchy (strength = 0.012)
- Consistent rankings but fluctuating perceptions
- High agreement between agents on leadership perceptions

### Best Parameters
- Group Size: 5 agents
- Schema: 2D continuous space
- Distributions: Normal for both characteristics and ILTs
- Standard Deviation: 0.169
- Match Threshold: 0.570
- Success Boost: 6.992
- Failure Penalty: 2.299

## Visualizations

### Identity Evolution
![Identity Evolution](../outputs/parameter_sweep/identity_evolution.png)

This plot shows:
- Individual identity trajectories over time
- Final distributions of leader and follower identities

### Leadership Structure
![Leadership Structure](../outputs/parameter_sweep/leadership_structure.png)

This visualization includes:
- Leadership perception network
- Average leadership received by each agent
- Perception agreement over time

### Parameter Sensitivity
![Parameter Sensitivity](../outputs/parameter_sweep/parameter_sensitivity.png)

Analysis of how key parameters affect:
- Identity differentiation
- Hierarchy strength

## Conclusions

1. The base model successfully demonstrates:
   - Development of differentiated leader and follower identities
   - Formation of stable but weak leadership hierarchies
   - Consistent perception agreement between agents

2. Limitations:
   - Hierarchy strength remains relatively weak
   - Leadership perceptions show high fluctuation
   - Identity development alone may be insufficient for strong leadership emergence

3. Next Steps:
   - Explore perspective models to enhance leadership emergence
   - Test different mechanisms for perception updates
   - Investigate ways to strengthen hierarchy formation 