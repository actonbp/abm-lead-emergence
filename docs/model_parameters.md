# Base Leadership Model Parameters and Assumptions

## Core Parameters

### Group Structure
- `n_agents`: Number of agents in the group (2-100, default: 6)
- `initial_li_equal`: Whether agents start with equal leadership identities (boolean, default: True)
- `initial_identity`: Initial identity value if equal (0-100, default: 50.0)

### Identity Dynamics
- `li_change_rate`: Rate of leadership identity change (0.0-5.0, default: 2.0)
- `perception_change_success`: Perception increase after successful claim (0.0-5.0, default: 2.0)
- `perception_change_reject`: Perception decrease after rejected claim (0.0-5.0, default: 3.0)
- `perception_change_noclaim`: Perception decrease when no claim made (0.0-5.0, default: 1.0)

### Interaction Parameters
- `claim_multiplier`: Multiplier for claim probability (0.0-1.0, default: 0.7)
- `grant_multiplier`: Multiplier for grant probability (0.0-1.0, default: 0.6)
- `claim_threshold`: Minimum score for leadership claims (0.0-1.0, default: 0.5)
- `penalize_no_claim`: Whether to penalize agents for not claiming (boolean, default: False)

### Schema-Identity Balance
- `schema_weight`: Weight given to schema matching (0.0-1.0)
- `weight_transition_start`: When to start transitioning from schema to identity (0.0-1.0, default: 0.2)
- `weight_transition_end`: When to end transition (0.0-1.0, default: 0.8)
- `weight_function`: Type of transition ('linear', 'sigmoid', 'quadratic', 'sqrt', default: 'linear')

## Currently Implemented Features

### ILT Matching Methods
- [x] Euclidean Distance (default)
- [x] Gaussian Similarity
- [x] Sigmoid Function
- [x] Threshold-based

### Interaction Selection
- [x] Random pair selection (default)
- [ ] Network-based selection (planned)
- [ ] Strategic selection (planned)

### Memory and History
- [x] Track interaction history
- [x] Track perception changes
- [x] Track identity changes
- [x] Network formation
- [ ] Memory decay (planned)

## Model Assumptions (Fixed)

### Agent Assumptions
1. Agents have both leader and follower identities (0-100 scale)
2. Agents have implicit leadership theories (ILTs)
3. Agents can accurately perceive others' characteristics
4. Agents make independent decisions

### Interaction Assumptions
1. Binary interactions (two agents at a time)
2. Sequential claim-grant process
3. Symmetric perception updates
4. No direct communication of intentions

### Learning Assumptions
1. Continuous identity updates
2. Local learning (from direct interactions)
3. No strategic planning
4. No memory decay (currently)

## Measurement Options

### Currently Implemented Metrics
- [x] Kendall's W (agreement on rankings)
- [x] Krippendorff's alpha (inter-rater reliability)
- [x] Normalized entropy (leadership distribution)
- [x] Top leader agreement
- [x] Network centralization
- [x] Network density

### Planned/Possible Metrics
- [ ] Role stability
- [ ] Hierarchy formation rate
- [ ] Leadership effectiveness
- [ ] Group performance

## Output Options

### Currently Available
1. Leadership identities over time
2. Follower identities over time
3. Perception matrices
4. Interaction networks
5. Emergence metrics
6. Claim/grant patterns

### Visualization Options
1. Identity trajectories
2. Network evolution
3. Emergence patterns
4. Distribution plots

## Potential Extensions

### Ready for Implementation
1. Memory decay
2. Network-based interactions
3. Strategic behavior
4. Multiple leadership dimensions

### Requires Significant Changes
1. Group-level processes
2. Environmental influences
3. Organizational constraints
4. Complex communication patterns 