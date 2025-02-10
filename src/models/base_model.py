"""
Base model class for leadership emergence simulations.
Uses ILT-based claim-grant mechanics with dual identity updates and perception tracking.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


@dataclass
class ModelParameters:
    """Parameters for leadership emergence model.
    
    Parameters
    ----------
    # Simulation Control
    n_steps : int, default=100
        Number of time steps to simulate (10 to 500)
    interaction_selection : str, default="random"
        Method for selecting interaction pairs ("random" or "sequential")
    n_replications : int, default=5
        Number of replications for parameter sweeps (1 to 50)
    random_seed : int, default=None
        Random seed for reproducibility
    
    # Core Structure  
    n_agents : int, default=4
        Number of agents in simulation (3 to 10)
    interaction_size : int, default=2
        Number of agents per interaction (2 to n_agents)
    
    # Schema/Characteristic Structure
    schema_dimensions : int, default=1
        Number of dimensions for characteristics/ILT (1 to 3)
    schema_type : str, default="continuous"
        Type of schema values ("binary" or "continuous")
    schema_correlation : float, default=0.0
        Correlation between dimensions (0 to 1, ignored for binary)
    
    # Matching Parameters
    match_algorithm : str, default="average"
        How to combine dimension matches ("average", "minimum", or "weighted")
    dimension_weights : str, default="uniform"
        How to weight dimensions ("uniform", "primary", or "sequential")
    match_threshold : float, default=0.5
        Threshold for successful match (0.3 to 0.8)
    
    # Interaction Rules
    grant_first : bool, default=False
        Whether granting happens before claiming
    allow_mutual_claims : bool, default=False
        Whether both agents can claim in same interaction
    allow_self_loops : bool, default=False
        Whether agents can interact with themselves
    simultaneous_roles : bool, default=False
        Whether agents can be leader and follower simultaneously
    
    # Distribution Parameters
    characteristic_distribution : str, default="uniform"
        Distribution for characteristics ("uniform" or "normal")
    ilt_distribution : str, default="uniform"
        Distribution for ILT schema (same options as characteristic_distribution)
    leader_identity_distribution : str, default="uniform"
        Distribution for initial leader identity (same options)
    follower_identity_distribution : str, default="uniform"
        Distribution for initial follower identity (same options)
    distribution_mean : float, default=50.0
        Mean for distributions (0 to 100)
    distribution_std : float, default=15.0
        Standard deviation for distributions (0 to 100)
    
    # Update Parameters
    success_boost : float, default=5.0
        Increase in identity after successful claim (2 to 10)
    failure_penalty : float, default=3.0
        Decrease in identity after failed claim (1 to 5)
    identity_inertia : float, default=0.2
        Resistance to identity changes (0.1 to 0.5)
    
    # Base Probabilities
    base_claim_probability : float, default=0.7
        Base probability of making a claim (0.3 to 0.8)
    """
    
    # Simulation Control
    n_steps: int = 100
    interaction_selection: str = "random"
    n_replications: int = 5
    random_seed: Optional[int] = None
    
    # Core Structure
    n_agents: int = 4
    interaction_size: int = 2
    
    # Schema/Characteristic Structure
    schema_dimensions: int = 1
    schema_type: str = "continuous"
    schema_correlation: float = 0.0
    
    # Matching Parameters
    match_algorithm: str = "average"
    dimension_weights: str = "uniform"
    match_threshold: float = 0.5
    
    # Interaction Rules
    grant_first: bool = False
    allow_mutual_claims: bool = False
    allow_self_loops: bool = False
    simultaneous_roles: bool = False
    
    # Distribution Parameters
    characteristic_distribution: str = "uniform"
    ilt_distribution: str = "uniform"
    leader_identity_distribution: str = "uniform"
    follower_identity_distribution: str = "uniform"
    distribution_mean: float = 50.0
    distribution_std: float = 15.0
    
    # Update Parameters
    success_boost: float = 5.0
    failure_penalty: float = 3.0
    identity_inertia: float = 0.2
    
    # Base Probabilities
    base_claim_probability: float = 0.7
    
    def __post_init__(self):
        """Convert parameters to appropriate types after initialization."""
        self.n_steps = int(self.n_steps)
        self.n_replications = int(self.n_replications)
        self.n_agents = int(self.n_agents)
        self.interaction_size = int(self.interaction_size)
        self.schema_dimensions = int(self.schema_dimensions)
        
        if self.random_seed is not None:
            self.random_seed = int(self.random_seed)
        
        self.schema_correlation = float(self.schema_correlation)
        self.match_threshold = float(self.match_threshold)
        self.distribution_mean = float(self.distribution_mean)
        self.distribution_std = float(self.distribution_std)
        self.success_boost = float(self.success_boost)
        self.failure_penalty = float(self.failure_penalty)
        self.identity_inertia = float(self.identity_inertia)
        self.base_claim_probability = float(self.base_claim_probability)
        
        self.grant_first = bool(self.grant_first)
        self.allow_mutual_claims = bool(self.allow_mutual_claims)
        self.allow_self_loops = bool(self.allow_self_loops)
        self.simultaneous_roles = bool(self.simultaneous_roles)
        
        self.validate_parameter_relationships()
    
    def validate_parameter_relationships(self):
        """Validate relationships between parameters."""
        # Simulation Control
        if not 10 <= self.n_steps <= 500:
            raise ValueError("n_steps must be between 10 and 500")
            
        if self.interaction_selection not in ["random", "sequential"]:
            raise ValueError(f"Unknown interaction selection method: {self.interaction_selection}")
            
        if not 1 <= self.n_replications <= 50:
            raise ValueError("n_replications must be between 1 and 50")
        
        # Core Structure
        if not 5 <= self.n_agents <= 10:
            raise ValueError("n_agents must be between 5 and 10")
            
        if not 2 <= self.interaction_size <= self.n_agents:
            raise ValueError("interaction_size must be between 2 and n_agents")
        
        # Schema Structure
        if not 2 <= self.schema_dimensions <= 3:
            raise ValueError("schema_dimensions must be between 2 and 3")
            
        if self.schema_type not in ["binary", "continuous"]:
            raise ValueError(f"Unknown schema type: {self.schema_type}")
            
        if not 0 <= self.schema_correlation <= 1:
            raise ValueError("schema_correlation must be between 0 and 1")
        
        # Matching Parameters
        if self.match_algorithm not in ["average", "minimum"]:
            raise ValueError(f"Unknown match algorithm: {self.match_algorithm}")
            
        if self.dimension_weights not in ["uniform", "primary", "sequential"]:
            raise ValueError(f"Unknown dimension weights: {self.dimension_weights}")
            
        if not 0.4 <= self.match_threshold <= 0.7:
            raise ValueError("match_threshold must be between 0.4 and 0.7")
        
        # Distribution Parameters
        valid_distributions = ["uniform", "normal"]
        for dist_name, dist in [
            ("characteristic_distribution", self.characteristic_distribution),
            ("ilt_distribution", self.ilt_distribution),
            ("leader_identity_distribution", self.leader_identity_distribution),
            ("follower_identity_distribution", self.follower_identity_distribution)
        ]:
            if dist not in valid_distributions:
                raise ValueError(f"Unknown distribution type for {dist_name}: {dist}")
        
        if not 0 <= self.distribution_mean <= 100:
            raise ValueError("distribution_mean must be between 0 and 100")
            
        if not 0 <= self.distribution_std <= 100:
            raise ValueError("distribution_std must be between 0 and 100")
        
        # Update Parameters
        if not 3.0 <= self.success_boost <= 30.0:
            raise ValueError("success_boost must be between 3.0 and 30.0")
            
        if not 2.0 <= self.failure_penalty <= 25.0:
            raise ValueError("failure_penalty must be between 2.0 and 25.0")
            
        if not 0.1 <= self.identity_inertia <= 0.5:
            raise ValueError("identity_inertia must be between 0.1 and 0.5")
        
        # Base Probabilities
        if not 0.3 <= self.base_claim_probability <= 0.8:
            raise ValueError("base_claim_probability must be between 0.3 and 0.8")
        
        # Schema Dimension Relationships - Auto-adjust instead of raising errors
        if self.schema_dimensions == 1:
            self.dimension_weights = "uniform"  # Force uniform weights for 1D
            if self.match_algorithm == "weighted":
                self.match_algorithm = "average"  # Fall back to average if weighted not possible
        
        # Binary Schema Relationships
        if self.schema_type == "binary":
            self.schema_correlation = 0  # Force correlation to 0 for binary
            self.distribution_std = 0  # Force std to 0 for binary
            if self.match_algorithm == "weighted":
                self.match_algorithm = "average"  # Fall back to average for binary
        
        # Interaction Rule Relationships
        if self.allow_mutual_claims and self.grant_first and not self.simultaneous_roles:
            self.simultaneous_roles = True  # Force simultaneous roles when needed
            
        if self.interaction_size > 2 and not self.allow_mutual_claims:
            self.allow_mutual_claims = True  # Force mutual claims for larger interactions


class Agent:
    """Agent in the leadership emergence model."""
    
    def __init__(self, id: int, rng: np.random.Generator, params: ModelParameters):
        """Initialize agent with characteristics and ILT schema."""
        self.id = id
        self.rng = rng
        self.params = params
        
        # Generate ILT in range 42-90 for each dimension (matching interactionist)
        self.ilt_schema = np.array([
            rng.uniform(42, 90) for _ in range(params.schema_dimensions)
        ])
        
        # Generate characteristics in range 10-90 for each dimension (matching interactionist)
        self.characteristic = np.array([
            rng.uniform(10, 90) for _ in range(params.schema_dimensions)
        ])
        
        # Calculate initial leader identity based on ILT-characteristic match (matching interactionist)
        ilt_char_diff = np.mean(np.abs(self.ilt_schema - self.characteristic))
        base_leader_identity = rng.uniform(60, 80)  # Random base in 60-80 range
        self.leader_identity = np.clip(base_leader_identity - ilt_char_diff, 0, 100)
        
        # Start follower identity at 50 (matching interactionist)
        self.follower_identity = 50.0
        
        # Initialize all perceptions at exactly 50 (neutral)
        self.leadership_perceptions = {}
        for i in range(params.n_agents):
            if i != id:  # Don't perceive self
                self.leadership_perceptions[i] = 50.0
        
        # Last interaction state
        self.last_interaction = {
            'match_score': 0,
            'claimed': False,
            'granted': False
        }
        
        # History for analysis
        self.history = {
            'claims': [],
            'grants': [],
            'perceptions': {}
        }
    
    def get_perception(self, other_id: int) -> float:
        """Get current leadership perception of another agent."""
        return self.leadership_perceptions.get(other_id, 50.0)  # Default to neutral
    
    def update_perception(self, other_id: int, change: float):
        """Update leadership perception of another agent."""
        if other_id not in self.leadership_perceptions:
            self.leadership_perceptions[other_id] = 50.0
        
        # Scale change based on current perception to avoid ceiling/floor effects (matching interactionist)
        current = self.leadership_perceptions[other_id]
        if change > 0:
            # Diminishing returns as perception gets higher
            scale = (100 - current) / 50
        else:
            # Diminishing returns as perception gets lower
            scale = current / 50
        
        adjusted_change = change * scale
        new_perception = np.clip(current + adjusted_change, 0, 100)
        self.leadership_perceptions[other_id] = new_perception
        
        # Track history
        if other_id not in self.history['perceptions']:
            self.history['perceptions'][other_id] = []
        self.history['perceptions'][other_id].append(new_perception)
    
    def calculate_ilt_match(self, other_characteristics: np.ndarray) -> float:
        """Calculate match between other's characteristics and own ILT."""
        if self.params.match_algorithm == "average":
            # Normalize differences to [0,1] range
            matches = [1 - abs(c - i)/100.0 for c, i in zip(other_characteristics, self.ilt_schema)]
            return np.mean(matches)
        elif self.params.match_algorithm == "minimum":
            # Normalize differences to [0,1] range
            matches = [1 - abs(c - i)/100.0 for c, i in zip(other_characteristics, self.ilt_schema)]
            return np.min(matches)
        else:  # uniform weights
            # Normalize differences to [0,1] range
            matches = [1 - abs(c - i)/100.0 for c, i in zip(other_characteristics, self.ilt_schema)]
            return np.mean(matches)
    
    def decide_claim(self, match_score: float) -> bool:
        """Decide whether to make leadership claim."""
        # Use only schema matching (matching interactionist Stage 1)
        claim_probability = match_score * self.params.base_claim_probability
        
        # Add small noise (matching interactionist)
        noise = self.rng.normal(0, 0.05)
        claim_probability = np.clip(claim_probability + noise, 0, 1)
        
        # Store last interaction state
        self.last_interaction['match_score'] = match_score
        self.last_interaction['claimed'] = self.rng.random() < claim_probability
        
        return self.last_interaction['claimed']
    
    def decide_grant(self, match_score: float) -> bool:
        """Decide whether to grant leadership claim."""
        # Use only schema matching (matching interactionist Stage 1)
        grant_probability = match_score
        
        # Only grant if above threshold
        if grant_probability <= self.params.match_threshold:
            grant_probability = 0
        
        # Add small noise (matching interactionist)
        noise = self.rng.normal(0, 0.05)
        grant_probability = np.clip(grant_probability + noise, 0, 1)
        
        # Store last interaction state
        self.last_interaction['granted'] = self.rng.random() < grant_probability
        
        return self.last_interaction['granted']
    
    def get_state(self) -> Dict:
        """Get current agent state."""
        return {
            'id': self.id,
            'characteristic': self.characteristic.tolist(),
            'ilt_schema': self.ilt_schema.tolist(),
            'leader_identity': self.leader_identity,
            'follower_identity': self.follower_identity,
            'leadership_perceptions': self.leadership_perceptions.copy(),
            'last_interaction': self.last_interaction.copy()
        }


class BaseLeadershipModel:
    """Base model for leadership emergence simulation."""
    
    def __init__(self, params: ModelParameters):
        """Initialize model with parameters."""
        self.params = params
        self.rng = np.random.default_rng(params.random_seed)
        
        # Initialize agents
        self.agents = [
            Agent(i, self.rng, self.params)
            for i in range(self.params.n_agents)
        ]
        
        # Track model state
        self.time = 0
        self.history = []
        
        # Initialize task
        self.task = None
    
    def set_task(self, task):
        """Set the task for this model."""
        self.task = task
    
    def select_interaction_pair(self) -> Tuple[Agent, Agent]:
        """Select two agents for interaction."""
        if self.params.interaction_selection == "random":
            return tuple(self.rng.choice(self.agents, size=2, replace=False))
        else:  # sequential
            idx1 = self.time % self.params.n_agents
            idx2 = (idx1 + 1) % self.params.n_agents
            return self.agents[idx1], self.agents[idx2]
    
    def step(self):
        """Execute one step of the model."""
        # Select interaction pair
        agent1, agent2 = self.select_interaction_pair()
        
        # Calculate match scores
        match_score_1 = agent1.calculate_ilt_match(agent2.characteristic)
        match_score_2 = agent2.calculate_ilt_match(agent1.characteristic)
        
        # Track interactions
        recent_interactions = []
        
        # Make leadership claims through task if available
        if self.task is not None:
            # Agent 1 claims
            if match_score_1 > 0.3 and self.rng.random() < self.params.base_claim_probability:
                result = self.task.share_info(
                    from_agent=agent1.id,
                    to_agent=agent2.id,
                    granted=match_score_1 > 0.3
                )
                recent_interactions.append({
                    'claimer': agent1.id,
                    'target': agent2.id,
                    'success': result['success'],
                    'quality': result.get('quality', 0.0),
                    'cost': result.get('cost', 0.0),
                    'time': result.get('time', 0.0),
                    'shared_info': result.get('shared_info', None),
                    'moves_closer': result.get('moves_closer', False)
                })
            
            # Agent 2 claims if allowed
            if match_score_2 > 0.3 and self.rng.random() < self.params.base_claim_probability:
                if not recent_interactions or self.params.allow_mutual_claims:
                    result = self.task.share_info(
                        from_agent=agent2.id,
                        to_agent=agent1.id,
                        granted=match_score_2 > 0.3
                    )
                    recent_interactions.append({
                        'claimer': agent2.id,
                        'target': agent1.id,
                        'success': result['success'],
                        'quality': result.get('quality', 0.0),
                        'cost': result.get('cost', 0.0),
                        'time': result.get('time', 0.0),
                        'shared_info': result.get('shared_info', None),
                        'moves_closer': result.get('moves_closer', False)
                    })
        
        # Increment time
        self.time += 1
        
        # Return current state
        state = self.get_state()
        state['recent_interactions'] = recent_interactions
        return state
    
    def get_state(self) -> Dict:
        """Get current model state."""
        state = {
            'time': self.time,
            'agents': [agent.get_state() for agent in self.agents]
        }
        if self.task is not None:
            state['task'] = self.task.evaluate_current_solution()
        return state
    
    def get_metrics(self) -> Dict:
        """Get model metrics."""
        # Calculate metrics based only on schema matching success
        n = len(self.agents)
        match_matrix = np.zeros((n, n))
        for i, agent1 in enumerate(self.agents):
            for j, agent2 in enumerate(self.agents):
                if i != j:
                    match_matrix[i,j] = agent1.calculate_ilt_match(agent2.characteristic)
        
        # Calculate convergence metrics
        hierarchy_strength = np.std(np.mean(match_matrix, axis=0))
        perception_agreement = np.mean([np.corrcoef(match_matrix[i], match_matrix[j])[0,1] 
                                     for i in range(n) for j in range(i+1,n)])
        
        metrics = {
            'hierarchy_strength': hierarchy_strength,
            'perception_agreement': perception_agreement,
            'convergence_score': perception_agreement  # Use agreement as convergence score
        }
        
        # Add task metrics if available
        if self.task is not None:
            metrics['task_performance'] = self.task.evaluate_current_solution()
        
        return metrics 