# Base Model Parameter Reference

## Simulation Control Parameters

### Time and Iterations
- `n_steps`: Number of time steps to simulate
  - Type: Integer
  - Range: 10-500
  - Default: 100
  - Description: Total number of discrete time steps in the simulation. Most leadership patterns emerge within 200-500 steps in small groups.

### Interaction Selection
- `interaction_selection`: Method for selecting which agents interact
  - Type: Categorical
  - Options: ["random", "sequential"]
  - Default: "random"
  - Description: How to choose which agents participate in each interaction. Future extensions may include network-based selection.

## Core Structure Parameters

### Population Size
- `n_agents`: Total number of agents in the simulation
  - Type: Integer
  - Range: 2-10
  - Default: 4
  - Description: The total population size from which interaction groups are drawn. Focused on small groups to study emergent leadership dynamics.

## Schema/Characteristic Structure

### Dimensionality
- `schema_dimensions`: Number of dimensions for characteristics and ILT schemas
  - Type: Integer
  - Range: 1-3
  - Default: 1
  - Description: Higher dimensions allow for more complex leadership prototypes. Note: When schema_dimensions=1, dimension_weights and per_dimension thresholds are ignored.

### Schema Type
- `schema_type`: Type of values used in schemas
  - Type: Categorical
  - Options: ["continuous", "binary"]
  - Default: "continuous"
  - Description: Determines whether characteristics/ILT use continuous or binary values. Note: schema_correlation is only applied for continuous schemas.

### Schema Correlation
- `schema_correlation`: Correlation between schema dimensions
  - Type: Float
  - Range: 0.0-1.0
  - Default: 0.0
  - Description: Controls how related different dimensions are. Only applies to continuous schemas with schema_dimensions > 1.

## Matching Parameters

### Algorithm
- `match_algorithm`: How to combine dimension matches
  - Type: Categorical
  - Options: ["average", "minimum", "weighted"]
  - Default: "average"
  - Description: Method for calculating overall match between characteristics and ILT. Note: "weighted" requires schema_dimensions > 1.

### Dimension Weights
- `dimension_weights`: How to weight different dimensions
  - Type: Categorical
  - Options: ["uniform", "primary", "sequential"]
  - Default: "uniform"
  - Description: Determines relative importance of dimensions. Only applies when schema_dimensions > 1.

### Threshold Type
- `match_threshold_type`: How to apply thresholds
  - Type: Categorical
  - Options: ["single", "per_dimension"]
  - Default: "single"
  - Description: Whether to use one threshold or separate thresholds per dimension. "per_dimension" requires schema_dimensions > 1.

### Match Threshold
- `match_threshold`: Threshold for good ILT match
  - Type: Float
  - Range: 0.0-1.0
  - Default: 0.6
  - Description: Minimum value for considering a match successful. Used for both claiming and granting decisions.

## Interaction Parameters

### Agents Per Interaction
- `interaction_size`: Number of agents in each interaction event
  - Type: Integer
  - Range: 2-2
  - Default: 2
  - Description: Currently fixed at 2 for dyadic interactions only.

### Interaction Rules
- `allow_mutual_claims`: Whether both agents can claim leadership
  - Type: Boolean
  - Default: False
  - Description: If True, both agents in an interaction can attempt to claim leadership.

- `grant_first`: Whether granting happens before claiming
  - Type: Boolean
  - Default: False
  - Description: Controls the order of claim/grant decisions in each interaction.

- `allow_self_loops`: Whether agents can interact with themselves
  - Type: Boolean
  - Default: False
  - Description: Controls self-interaction possibility. Generally kept false for leadership emergence studies.

- `simultaneous_roles`: Whether agents can be leader and follower simultaneously
  - Type: Boolean
  - Default: True
  - Description: Allows for flexible role adoption across different relationships.

## Distribution Parameters

### Initial Distributions
- `characteristic_distribution`: Distribution type for characteristics
  - Type: Categorical
  - Options: ["uniform", "normal", "fixed", "power_law"]
  - Default: "uniform"
  - Description: How initial characteristics are distributed. All values are truncated to [0,1].

- `ilt_distribution`: Distribution type for ILT schemas
  - Type: Categorical
  - Options: ["uniform", "normal", "fixed", "power_law"]
  - Default: "uniform"
  - Description: How initial ILT schemas are distributed. All values are truncated to [0,1].

- `leader_identity_distribution`: Distribution type for initial leader identities
  - Type: Categorical
  - Options: ["fixed", "uniform"]
  - Default: "fixed"
  - Description: How initial leader identities are distributed. Values are scaled to [0,100].

- `follower_identity_distribution`: Distribution type for initial follower identities
  - Type: Categorical
  - Options: ["fixed", "uniform"]
  - Default: "fixed"
  - Description: How initial follower identities are distributed. Values are scaled to [0,100].

### Distribution Parameters
- `distribution_mean`: Mean for normal distributions
  - Type: Float
  - Range: 0.0-1.0
  - Default: 0.5
  - Description: Center point for normal distributions. Applied to both characteristics and ILT schemas.

- `distribution_std`: Standard deviation for normal distributions
  - Type: Float
  - Range: 0.0-0.5
  - Default: 0.2
  - Description: Spread for normal distributions. Limited to 0.5 to maintain meaningful [0,1] distributions.

- `power_law_alpha`: Alpha parameter for power law distributions
  - Type: Float
  - Range: > 1.0
  - Default: 2.0
  - Description: Shape parameter for power law distributions. Values truncated to [0,1].

## Update Parameters

### Identity Updates
- `success_boost`: Identity boost from successful interaction
  - Type: Float
  - Range: 0.0-10.0
  - Default: 5.0
  - Description: How much identities increase after successful leadership claims. Final values clamped to [0,100].

- `failure_penalty`: Identity penalty from failed interaction
  - Type: Float
  - Range: 0.0-10.0
  - Default: 3.0
  - Description: How much identities decrease after failed leadership claims. Final values clamped to [0,100].

- `identity_inertia`: How much identities resist change
  - Type: Float
  - Range: 0.0-1.0
  - Default: 0.1
  - Description: Higher values mean identities change more slowly. Helps prevent rapid oscillations.

## Parameter Relationships and Constraints

### Dimensional Dependencies
1. When `schema_dimensions = 1`:
   - `dimension_weights` is ignored (defaults to "uniform")
   - `match_threshold_type` must be "single"
   - `schema_correlation` has no effect
   - `match_algorithm` cannot be "weighted"

2. When `schema_type = "binary"`:
   - `schema_correlation` has no effect
   - `distribution_mean` and `distribution_std` are ignored
   - Values are strictly 0 or 1

### Interaction Logic
1. When `interaction_size = 2`:
   - Triadic interactions are disabled
   - `interaction_order` fully determines claim/grant sequence

2. When `allow_self_loops = False`:
   - Must have `n_agents > interaction_size`
   - Ensures meaningful interactions between distinct agents

3. When `simultaneous_roles = False`:
   - An agent cannot accumulate both leader and follower identities
   - May affect identity update dynamics

### Distribution Constraints
1. For all distributions:
   - Characteristic and ILT schema values are clamped to [0,1]
   - Identity values are scaled to [0,100]
   - Must maintain meaningful differentiation between agents

2. When using normal distribution:
   - `distribution_mean` should be sufficiently far from 0 and 1
   - `distribution_std` limited to prevent excessive clustering at bounds

3. When using power law:
   - Values are rescaled to [0,1] after generation
   - Higher `power_law_alpha` leads to more extreme differentiation

### Update Dynamics
1. Identity Updates:
   - `success_boost` and `failure_penalty` should be balanced
   - Large updates with low `identity_inertia` can cause instability
   - Consider total magnitude: `(success_boost + failure_penalty) * (1 - identity_inertia)`

2. Matching and Claims:
   - Match score is combined with current identity scores
   - Higher thresholds with low identities may prevent emergence
   - Balance needed between selectivity and activity

### Recommended Parameter Sets
1. Fast Emergence:
   - Lower `match_threshold` (≈0.4)
   - Higher `success_boost` (≈5.0)
   - Lower `identity_inertia` (≈0.1)
   - `n_agents` = 4

2. Stable Hierarchy:
   - Higher `match_threshold` (≈0.6)
   - Balanced `success_boost`/`failure_penalty` (≈3.0/3.0)
   - Higher `identity_inertia` (≈0.3)
   - `n_agents` = 6

3. Complex Prototypes:
   - `schema_dimensions = 2` or `3`
   - `schema_type = "continuous"`
   - `match_algorithm = "weighted"`
   - Moderate correlation and thresholds 