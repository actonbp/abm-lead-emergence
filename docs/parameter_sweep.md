# Parameter Sweep and Model Validation

This document describes our approach to finding optimal parameters for the leadership emergence model while ensuring both theoretical validity and realistic behavior.

## Validation Framework

Our validation framework combines two key aspects:
1. Stylized facts from leadership emergence literature
2. Realistic variation checks based on empirical observations

### Stylized Facts
- Leadership emerges naturally over time (20-80 time steps)
- Clear role differentiation develops
- Stable but not static leadership identities
- Consensus around leadership structure

### Realism Criteria

#### Role Distribution
- High Leaders (70-85% identity)
  - ~30% of team
  - Clear leadership without being perfect
- Mid-Range (40-60% identity)
  - ~30% of team
  - Active participants with moderate leadership
- Low-Range (15-35% identity)
  - ~40% of team
  - Engaged followers

#### Stability and Variation
- Minimum stability score: 0.7
- Maximum rank change rate: 0.5
- Identity variation: 5-25% standard deviation
- Maintains relative positions while allowing small fluctuations

#### Between-Team Variation
- Different teams show different patterns
- Some teams develop single clear leaders
- Others show co-leadership
- Consistent with empirical observations

## Parameter Sweep Approach

### Search Space
- Discrete Parameters:
  - Number of agents: 3-10
  - Schema dimensions: 2-3
  - Match algorithm: average, minimum
  - Dimension weights: uniform
  
- Continuous Parameters:
  - Match threshold: 0.3-0.6
  - Success boost: 6.0-10.0
  - Failure penalty: 1.0-2.0
  - Identity inertia: 0.1-0.3
  - Base claim probability: 0.3-0.7

### Optimization Method
- Bayesian optimization using Gaussian Process
- 50 initial exploration points
- 100 optimization iterations
- 5 replications per parameter set
- Total simulations: 750

### Scoring Weights
- Role Distribution: 40%
- Stability: 20%
- Emergence Timing: 20%
- Between-team Variation: 20%

## Validation Process

1. Parameter Search
   - Bayesian optimization to find promising parameters
   - Comprehensive scoring based on all criteria
   - Multiple replications to ensure robustness

2. Validation Testing
   - 10 validation replications with best parameters
   - Checks for consistent performance
   - Verifies both stylized facts and realism criteria

3. Visualization
   - Individual team trajectories
   - Aggregate patterns
   - Distribution of outcomes
   - Parameter sensitivity analysis

## Results

The parameter sweep identified configurations that produce realistic leadership emergence patterns:
- Natural emergence timing (40-60 steps)
- Clear but not extreme role differentiation
- Stable hierarchies with appropriate variation
- Consistent with empirical observations of team dynamics

See `outputs/parameter_sweep/` for detailed results and visualizations. 