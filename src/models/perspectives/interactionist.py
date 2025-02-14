"""
Social interactionist model that transitions from schema to identity-based decisions.
"""

from typing import Dict, Any, Literal
import numpy as np
from dataclasses import dataclass, field

from ..base_model import BaseLeadershipModel, Agent, ModelParameters
from pydantic import Field, BaseModel


@dataclass
class InteractionistParameters(ModelParameters):
    """Parameters for the Social Interactionist model."""
    # Identity transition parameters
    identity_growth_rate: float = 0.15  # Base rate for identity growth (0.1 to 0.3)
    max_identity_weight: float = 0.9  # Maximum weight for identity-based decisions (0.7 to 0.95)
    identity_growth_exponent: float = 2.0  # Exponential growth factor (1.5 to 3.0)
    
    # Identity update parameters
    identity_update_rate: float = 0.3  # How quickly identities change (0.2 to 0.8)
    perception_update_rate: float = 0.3  # How quickly perceptions change (0.2 to 0.8)
    
    # Penalties and boosts (aligned with base model ranges)
    success_boost: float = 5.0  # Boost to identity on successful claim (3.0 to 30.0)
    failure_penalty: float = 3.0  # Penalty for failed claim (2.0 to 25.0)
    identity_inertia: float = 0.2  # Resistance to identity changes (0.1 to 0.5)
    passivity_penalty: float = 1.0  # Penalty for not claiming when should (0.5 to 5.0)
    
    def __post_init__(self):
        """Convert parameters to appropriate types after initialization."""
        super().__post_init__()
        
        # Convert parameters
        self.identity_growth_rate = float(self.identity_growth_rate)
        self.max_identity_weight = float(self.max_identity_weight)
        self.identity_growth_exponent = float(self.identity_growth_exponent)
        self.identity_update_rate = float(self.identity_update_rate)
        self.perception_update_rate = float(self.perception_update_rate)
        self.success_boost = float(self.success_boost)
        self.failure_penalty = float(self.failure_penalty)
        self.identity_inertia = float(self.identity_inertia)
        self.passivity_penalty = float(self.passivity_penalty)
        
        self.validate_parameters()
    
    def validate_parameters(self):
        """Validate model parameters."""
        if not 0.1 <= self.identity_growth_rate <= 0.3:
            raise ValueError("identity_growth_rate must be between 0.1 and 0.3")
            
        if not 0.7 <= self.max_identity_weight <= 0.95:
            raise ValueError("max_identity_weight must be between 0.7 and 0.95")
            
        if not 1.5 <= self.identity_growth_exponent <= 3.0:
            raise ValueError("identity_growth_exponent must be between 1.5 and 3.0")
            
        if not 0.2 <= self.identity_update_rate <= 0.8:
            raise ValueError("identity_update_rate must be between 0.2 and 0.8")
            
        if not 0.2 <= self.perception_update_rate <= 0.8:
            raise ValueError("perception_update_rate must be between 0.2 and 0.8")
            
        if not 3.0 <= self.success_boost <= 30.0:
            raise ValueError("success_boost must be between 3.0 and 30.0")
            
        if not 2.0 <= self.failure_penalty <= 25.0:
            raise ValueError("failure_penalty must be between 2.0 and 25.0")
            
        if not 0.1 <= self.identity_inertia <= 0.5:
            raise ValueError("identity_inertia must be between 0.1 and 0.5")
            
        if not 0.5 <= self.passivity_penalty <= 5.0:
            raise ValueError("passivity_penalty must be between 0.5 and 5.0")


class InteractionistAgent(Agent):
    """Agent that transitions from schema to identity-based decisions."""
    
    def __init__(self, id, rng, params):
        """Initialize agent with characteristics and ILT schema."""
        # Initialize base agent but we'll override the characteristics and ILT
        super().__init__(id, rng, params)
        
        # Generate ILT in range 42-90 for each dimension
        self.ilt_schema = np.array([
            rng.uniform(42, 90) for _ in range(params.schema_dimensions)
        ])
        
        # Generate characteristics in range 10-90 for each dimension
        self.characteristic = np.array([
            rng.uniform(10, 90) for _ in range(params.schema_dimensions)
        ])
        
        # Calculate initial leader identity based on ILT-characteristic match
        ilt_char_diff = np.mean(np.abs(self.ilt_schema - self.characteristic))
        base_leader_identity = rng.uniform(60, 80)  # Random base in 60-80 range
        self.leader_identity = np.clip(base_leader_identity - ilt_char_diff, 0, 100)
        
        # Start follower identity at 50
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
            'granted': False,
            'step': 0
        }
    
    def calculate_identity_weight(self, step: int) -> float:
        """Calculate identity influence weight using exponential growth."""
        # Normalize step to [0,1] range for growth calculation
        normalized_step = min(1.0, step * self.params.identity_growth_rate)
        
        # Apply exponential growth
        weight = normalized_step ** (1.0 / self.params.identity_growth_exponent)
        
        # Scale to max weight and clip
        weight = weight * self.params.max_identity_weight
        return min(self.params.max_identity_weight, weight)
    
    def decide_claim(self, match_score: float, step: int) -> bool:
        """Decide whether to make leadership claim based on schema match and identity."""
        # Calculate identity-based probability
        identity_prob = self.leader_identity / 100.0
        
        # Calculate identity weight using exponential growth
        identity_weight = self.calculate_identity_weight(step)
        schema_weight = 1.0 - identity_weight
        
        # Blend schema and identity from the start, but be more aggressive early
        claim_probability = (
            schema_weight * match_score * self.params.base_claim_probability * 1.5 +  # More aggressive early claims
            identity_weight * identity_prob
        )
        
        # Add small noise
        noise = self.rng.normal(0, 0.05)
        claim_probability = np.clip(claim_probability + noise, 0, 1)
        
        # Store last interaction state
        self.last_interaction['match_score'] = match_score
        self.last_interaction['claimed'] = self.rng.random() < claim_probability
        self.last_interaction['step'] = step
        
        return self.last_interaction['claimed']
    
    def decide_grant(self, match_score: float, step: int) -> bool:
        """Decide whether to grant leadership claim based on schema match and identity."""
        # Calculate identity-based probability
        identity_prob = self.follower_identity / 100.0
        
        # Calculate identity weight using exponential growth
        identity_weight = self.calculate_identity_weight(step)
        schema_weight = 1.0 - identity_weight
        
        # Blend schema and identity from the start, with lower threshold early
        grant_probability = (
            schema_weight * match_score * 1.2 +  # More likely to grant early
            identity_weight * identity_prob
        )
        
        # Lower threshold early to encourage interaction
        effective_threshold = self.params.match_threshold * (0.8 + 0.2 * identity_weight)
        
        # Only grant if above threshold
        if grant_probability <= effective_threshold:
            grant_probability = 0
            
        # Add small noise
        noise = self.rng.normal(0, 0.05)
        grant_probability = np.clip(grant_probability + noise, 0, 1)
        
        # Store last interaction state
        self.last_interaction['granted'] = self.rng.random() < grant_probability
        self.last_interaction['step'] = step
        
        return self.last_interaction['granted']

    def update_perception(self, other_id: int, change: float):
        """Update leadership perception of another agent."""
        if str(other_id) not in self.leadership_perceptions:
            self.leadership_perceptions[str(other_id)] = 50.0
        
        # Scale change based on current perception to avoid ceiling/floor effects
        current = self.leadership_perceptions[str(other_id)]
        if change > 0:
            # Diminishing returns as perception gets higher
            scale = (100 - current) / 50
        else:
            # Diminishing returns as perception gets lower
            scale = current / 50
        
        # Make early changes more impactful based on identity weight
        identity_weight = self.calculate_identity_weight(self.last_interaction.get('step', 0))
        if identity_weight < 0.3:  # Early in the process
            change *= 1.5
        
        adjusted_change = change * scale
        new_perception = np.clip(current + adjusted_change, 0, 100)
        self.leadership_perceptions[str(other_id)] = new_perception


class InteractionistModel(BaseLeadershipModel):
    """Social interactionist model with schema-to-identity transition."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize with social interactionist parameters."""
        if isinstance(params, dict):
            self.params = InteractionistParameters(**params)
        else:
            self.params = params if params else InteractionistParameters()
        
        # Initialize random number generator
        self.rng = np.random.default_rng(self.params.random_seed)
        self.time = 0
        
        # Initialize agents
        self.agents = [
            InteractionistAgent(i, self.rng, self.params)
            for i in range(self.params.n_agents)
        ]
        
        # Track model state
        self.history = []
    
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
            if agent1.decide_claim(match_score_1, self.time):
                result = self.task.share_info(
                    from_agent=agent1.id,
                    to_agent=agent2.id,
                    granted=agent2.decide_grant(match_score_1, self.time)
                )
                if result['success']:
                    # Update identities based on successful claim
                    agent1.leader_identity = min(100, agent1.leader_identity + self.params.success_boost)
                    agent2.follower_identity = min(100, agent2.follower_identity + self.params.success_boost)
                    # Update perceptions
                    agent2.update_perception(agent1.id, self.params.success_boost)
                else:
                    # Penalty for rejected claim
                    agent1.leader_identity = max(0, agent1.leader_identity - self.params.failure_penalty)
                    agent2.update_perception(agent1.id, -self.params.failure_penalty)
                
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
            if agent2.decide_claim(match_score_2, self.time):
                if not recent_interactions or self.params.allow_mutual_claims:
                    result = self.task.share_info(
                        from_agent=agent2.id,
                        to_agent=agent1.id,
                        granted=agent1.decide_grant(match_score_2, self.time)
                    )
                    if result['success']:
                        # Update identities based on successful claim
                        agent2.leader_identity = min(100, agent2.leader_identity + self.params.success_boost)
                        agent1.follower_identity = min(100, agent1.follower_identity + self.params.success_boost)
                        # Update perceptions
                        agent1.update_perception(agent2.id, self.params.success_boost)
                    else:
                        # Penalty for rejected claim
                        agent2.leader_identity = max(0, agent2.leader_identity - self.params.failure_penalty)
                        agent1.update_perception(agent2.id, -self.params.failure_penalty)
                    
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
        
        self.time += 1
        state = self.get_state()
        state['recent_interactions'] = recent_interactions
        return state
    
    def get_state(self):
        """Get current model state."""
        state = super().get_state()
        identity_weight = self.agents[0].calculate_identity_weight(self.time)  # Use first agent's calculation
        state['identity_weight'] = identity_weight
        return state
    
    def get_metrics(self) -> Dict:
        """Get model metrics."""
        metrics = super().get_metrics()
        identity_weight = self.agents[0].calculate_identity_weight(self.time)  # Use first agent's calculation
        metrics['identity_weight'] = identity_weight
        return metrics
