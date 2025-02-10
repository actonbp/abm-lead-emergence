"""
Cognitive Model of Leadership Emergence

This model implements a cognitive learning perspective where agents update their
Implicit Leadership Theories (ILTs) by observing successful leadership interactions.
The influence of learning grows continuously from the start.

Future Research Directions:
- Differential characteristic adaptation: Some characteristics might be more salient 
  or easier to observe/copy than others
- Context-dependent schema matching: Different situations might make different 
  characteristics more important
- Schema change resistance: Some dimensions of ILTs might be more resistant to 
  change than others
- Individual differences: Agents might have different learning rates for different
  characteristics based on their own traits
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

from ..base_model import BaseLeadershipModel, ModelParameters, Agent


@dataclass
class CognitiveParameters(ModelParameters):
    """Parameters for the Cognitive model."""
    
    # Learning parameters
    learning_growth_rate: float = 0.02  # How quickly learning influence grows (0.01 to 0.05)
    max_learning_weight: float = 0.8  # Maximum weight for learned patterns (0.6 to 0.9)
    base_learning_rate: float = 0.3  # Base rate for ILT updates (0.1 to 0.5)
    
    def __post_init__(self):
        """Convert parameters to appropriate types after initialization."""
        super().__post_init__()
        self.learning_growth_rate = float(self.learning_growth_rate)
        self.max_learning_weight = float(self.max_learning_weight)
        self.base_learning_rate = float(self.base_learning_rate)
        self.validate_parameters()
    
    def validate_parameters(self):
        """Validate model parameters."""
        if not 0.01 <= self.learning_growth_rate <= 0.05:
            raise ValueError("learning_growth_rate must be between 0.01 and 0.05")
        if not 0.6 <= self.max_learning_weight <= 0.9:
            raise ValueError("max_learning_weight must be between 0.6 and 0.9")
        if not 0.1 <= self.base_learning_rate <= 0.5:
            raise ValueError("base_learning_rate must be between 0.1 and 0.5")


class CognitiveAgent(Agent):
    """Agent that learns and adapts ILTs through observation."""
    
    def __init__(self, id: int, rng: np.random.Generator, params: ModelParameters):
        """Initialize cognitive agent."""
        # Initialize base agent but we'll override the characteristics and ILT
        super().__init__(id, rng, params)
        
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
    
    def update_ilt_from_observation(self, leader_characteristic: np.ndarray, step: int):
        """Update ILT schema based on observing a successful leader.
        
        Learning influence grows over time, starting small and increasing.
        """
        # Calculate learning weight that grows over time
        learning_weight = min(
            self.params.max_learning_weight,
            step * self.params.learning_growth_rate
        )
        
        # Adjust learning rate based on current influence
        effective_learning_rate = self.params.base_learning_rate * learning_weight
        
        # Update ILT through weighted interpolation
        self.ilt_schema = self.ilt_schema + effective_learning_rate * (leader_characteristic - self.ilt_schema)
        
        # Ensure schema stays within valid range
        self.ilt_schema = np.clip(self.ilt_schema, 0.0, 100.0)


class CognitiveModel(BaseLeadershipModel):
    """Cognitive model where agents learn and adjust their ILTs through observation.
    
    Learning influence grows continuously from the start, rather than having a
    discrete transition point.
    """
    
    def __init__(self, params: CognitiveParameters):
        """Initialize the cognitive model."""
        super().__init__(params)
        self.params = params
        self.current_step = 0
        
        # Initialize random number generator
        if self.params.random_seed is not None:
            self.random = np.random.default_rng(self.params.random_seed)
        else:
            self.random = np.random.default_rng()
            
        # Initialize agents
        self.agents = []
        for i in range(self.params.n_agents):
            agent = CognitiveAgent(id=i, rng=self.random, params=self.params)
            self.agents.append(agent)
    
    def step(self):
        """Execute one step of the model."""
        # Select interaction pair
        agent_i, agent_j = self.select_interaction_pair()
        
        # Calculate match scores using agent's own methods
        match_score_i = agent_j.calculate_ilt_match(agent_i.characteristic)  # How well i matches j's schema
        match_score_j = agent_i.calculate_ilt_match(agent_j.characteristic)  # How well j matches i's schema
        
        # Track claims and grants
        claims_made = []
        grants_made = []
        
        # Determine if agents will claim leadership based on match score
        claim_i = match_score_i > self.params.match_threshold and self.random.random() < self.params.base_claim_probability
        claim_j = match_score_j > self.params.match_threshold and self.random.random() < self.params.base_claim_probability
        
        if claim_i:
            # Use probabilistic granting
            grant_probability = match_score_i
            if grant_probability <= self.params.match_threshold:
                grant_probability = 0
            noise = self.random.normal(0, 0.05)
            grant_probability = np.clip(grant_probability + noise, 0, 1)
            granted = self.random.random() < grant_probability
            
            if granted:
                self.process_leadership_claim(agent_i, agent_j, True)
                claims_made.append((agent_i.id, agent_j.id))
                grants_made.append((agent_j.id, agent_i.id))
            else:
                self.process_leadership_claim(agent_i, agent_j, False)
                claims_made.append((agent_i.id, agent_j.id))
        
        if claim_j and (not claim_i or self.params.allow_mutual_claims):
            # Use probabilistic granting
            grant_probability = match_score_j
            if grant_probability <= self.params.match_threshold:
                grant_probability = 0
            noise = self.random.normal(0, 0.05)
            grant_probability = np.clip(grant_probability + noise, 0, 1)
            granted = self.random.random() < grant_probability
            
            if granted:
                self.process_leadership_claim(agent_j, agent_i, True)
                claims_made.append((agent_j.id, agent_i.id))
                grants_made.append((agent_i.id, agent_j.id))
            else:
                self.process_leadership_claim(agent_j, agent_i, False)
                claims_made.append((agent_j.id, agent_i.id))
        
        # Update step counter
        self.current_step += 1
        
        return {
            'step': self.current_step,
            'interacting_agents': (agent_i.id, agent_j.id),
            'claims_made': claims_made,
            'grants_made': grants_made,
            'agents': [agent.get_state() for agent in self.agents]
        }
    
    def process_leadership_claim(self, claimer: Agent, target: Agent, success: bool):
        """Process a leadership claim between two agents."""
        if success:
            # Target updates their ILT based on successful leadership
            target.update_ilt_from_observation(claimer.characteristic, self.current_step)
            
            # Other agents learn from observing successful leadership
            for observer in self.agents:
                if observer.id != claimer.id and observer.id != target.id:
                    observer.update_ilt_from_observation(claimer.characteristic, self.current_step)
                    observer.update_perception(str(claimer.id), self.params.success_boost)
            
            # Update target's perception
            target.update_perception(str(claimer.id), self.params.success_boost)
        else:
            # Update target's perception for failed claim
            target.update_perception(str(claimer.id), -self.params.failure_penalty)
            
            # Update observers' perceptions
            for observer in self.agents:
                if observer.id != claimer.id and observer.id != target.id:
                    observer.update_perception(str(claimer.id), -self.params.failure_penalty)
    
    def get_state(self):
        """Get current model state."""
        state = super().get_state()
        learning_weight = min(
            self.params.max_learning_weight,
            self.current_step * self.params.learning_growth_rate
        )
        state['learning_weight'] = learning_weight
        return state
    
    def get_metrics(self) -> Dict:
        """Get model metrics."""
        metrics = super().get_metrics()
        learning_weight = min(
            self.params.max_learning_weight,
            self.current_step * self.params.learning_growth_rate
        )
        metrics['learning_weight'] = learning_weight
        return metrics
