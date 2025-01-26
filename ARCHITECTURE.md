# Technical Architecture

This document outlines the technical implementation of our agent-based modeling (ABM) framework for studying leadership emergence across different theoretical perspectives.

## Core Components

### 1. Base Model (`src/models/base_model.py`)
```python
class BaseLeadershipModel:
    def __init__(self, n_agents: int, agent_class: Type[Agent] = BaseAgent):
        # Allow perspectives to use their own agent types
        self.agents = [agent_class() for _ in range(n_agents)]
        self.environment = Environment()
        self.history = []  # Track time series data
    
    def step(self) -> dict:
        """Single simulation step with basic claim-grant logic."""
        interactions = self._generate_interactions()
        self._process_claims_and_grants(interactions)
        state = self._get_state()
        self.history.append(state)  # Store for time-based metrics
        return state
    
    def _process_claims_and_grants(self, interactions):
        """Base claim-grant logic - minimal implementation.
        Perspectives override this with their specific logic."""
        for interaction in interactions:
            # Basic probability-based claims and grants
            # No specialized identity/cognitive logic here
            pass
```

### 2. Perspectives (`src/models/perspectives/`)
```python
# Base agent types for each perspective
class InteractionistAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.interaction_history = []

class CognitiveAgent(BaseAgent):
    def __init__(self, ilt_weight: float):
        super().__init__()
        self.ilt_weight = ilt_weight
        self.leadership_schema = {}

class IdentityAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.identity_strength = 0.5
        self.group_prototypicality = 0.0

# Perspective-specific models
class InteractionistModel(BaseLeadershipModel):
    def __init__(self, n_agents: int):
        super().__init__(n_agents, agent_class=InteractionistAgent)
    
    def _process_claims_and_grants(self, interactions):
        """Override with interactionist-specific logic"""
        for interaction in interactions:
            # Consider past interactions
            # Update interaction history
            pass

class CognitiveModel(BaseLeadershipModel):
    def __init__(self, n_agents: int, ilt_weight: float):
        self.ilt_weight = ilt_weight
        super().__init__(n_agents, agent_class=lambda: CognitiveAgent(ilt_weight))
    
    def _process_claims_and_grants(self, interactions):
        """Override with cognitive-specific logic"""
        for interaction in interactions:
            # Apply ILT matching
            # Update leadership schemas
            pass
```

### 3. Contexts (`src/simulation/contexts/`)
```python
# base_context.py
class Context:
    def modify_claim_probability(self, base_prob: float) -> float:
        return base_prob

# crisis.py
class CrisisContext(Context):
    def __init__(self, intensity: float):
        self.intensity = intensity
```

## Time Series Handling

### 1. State Tracking
```python
class BaseLeadershipModel:
    def _get_state(self) -> dict:
        """Capture current simulation state."""
        return {
            'step': self.current_step,
            'agents': [{
                'id': agent.id,
                'lead_score': agent.lead_score,
                'last_interaction': agent.last_interaction,
                'is_leader': agent.lead_score > self.LEADER_THRESHOLD
            } for agent in self.agents],
            'metrics': self._calculate_current_metrics()
        }
    
    def _calculate_current_metrics(self) -> dict:
        """Compute key metrics for current state."""
        return {
            'num_leaders': sum(1 for a in self.agents 
                             if a.lead_score > self.LEADER_THRESHOLD),
            'leadership_concentration': self._gini_coefficient(
                [a.lead_score for a in self.agents]
            ),
            # Add other relevant metrics
        }
```

## Parameter Management and Analysis

### 1. Parameter Space Definition
```python
# parameters.py
PARAMETER_SPACE = {
    'interactionist': {
        'n_agents': [4, 6, 8],
        'claim_multiplier': [0.5, 0.7, 0.9],
        'grant_multiplier': [0.4, 0.6, 0.8]
    },
    'cognitive': {
        'n_agents': [4, 6, 8],
        'ilt_weight': [0.3, 0.5, 0.7],
        'schema_update_rate': [0.1, 0.3, 0.5]
    },
    'identity': {
        'n_agents': [4, 6, 8],
        'identity_salience': [0.3, 0.6, 0.9],
        'prototypicality_weight': [0.4, 0.7, 1.0]
    }
}
```

### 2. Advanced Model Selection
```python
def select_best_models(results_df: pd.DataFrame, method='weighted_distance') -> dict:
    """Select best parameter sets per perspective.
    
    Methods:
    - weighted_distance: Simple weighted distance to stylized facts
    - bayesian_opt: Bayesian optimization for large parameter spaces
    - random_forest: Feature importance analysis
    """
    if method == 'weighted_distance':
        return _select_by_weighted_distance(results_df)
    elif method == 'bayesian_opt':
        return _select_by_bayesian_opt(results_df)
    elif method == 'random_forest':
        return _select_by_random_forest(results_df)
```

### 3. Parallel Processing
```python
def run_sweep(
    perspectives: List[str],
    param_space: dict,
    n_replications: int = 5,
    n_jobs: int = -1  # Use all available cores
) -> pd.DataFrame:
    """Run parameter sweep in parallel."""
    from joblib import Parallel, delayed
    
    param_combinations = list(_generate_all_combinations(
        perspectives, param_space
    ))
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_single_simulation)(
            perspective, params, rep
        )
        for perspective, params in param_combinations
        for rep in range(n_replications)
    )
    
    return pd.DataFrame(results)
```

## Output Structure
```
outputs/
├── parameter_sweep/
│   ├── results_{timestamp}.json    # Raw simulation data
│   └── summary_{timestamp}.csv     # Statistical summary
├── analysis/
│   ├── stylized_facts/            # Comparison to stylized facts
│   │   ├── distances.csv          # Distance metrics
│   │   └── rankings.csv           # Parameter set rankings
│   └── figures/                   # Generated plots
├── best_models/                   # Best parameters per perspective
└── context_analysis/             # Context testing results
```

## Implementation Notes

1. **Modularity**
   - Base model provides minimal claim-grant mechanics
   - Each perspective has its own agent class and model logic
   - Contexts modify behavior without changing core logic
   - Analysis pipeline supports different selection methods

2. **Time Series Management**
   - Full state history tracked for each simulation
   - Efficient storage of relevant metrics only
   - Support for time-based analysis (e.g., time to first leader)

3. **Performance**
   - Parameter sweeps can run in parallel
   - Optional Bayesian optimization for large parameter spaces
   - Results caching for expensive computations

4. **Future Extensions**
   - Support for perspective combinations (e.g., Interactionist + Cognitive)
   - Additional metrics and analysis methods
   - Enhanced visualization tools

See [README.md](README.md) for quick start guide and basic usage examples 