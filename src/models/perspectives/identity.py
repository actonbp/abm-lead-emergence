"""
Social Identity Model of Leadership Emergence

This model implements a social identity perspective where leadership emergence is driven
by the development of a shared group prototype. The influence of the collective prototype
grows continuously from the start.
"""

from typing import Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass

from ..base_model import BaseLeadershipModel, Agent, ModelParameters


@dataclass
class IdentityParameters(ModelParameters):
    """Parameters for the Social Identity model."""
    # Prototype transition parameters
    prototype_growth_rate: float = 0.02  # How quickly prototype influence grows (0.01 to 0.05)
    max_prototype_weight: float = 0.8  # Maximum weight for group prototype (0.6 to 0.9)
    base_prototype_learning: float = 0.2  # Base rate for prototype updates (0.1 to 0.5)
    
    def __post_init__(self):
        """Convert parameters to appropriate types after initialization."""
        super().__post_init__()
        
        # Convert parameters
        self.prototype_growth_rate = float(self.prototype_growth_rate)
        self.max_prototype_weight = float(self.max_prototype_weight)
        self.base_prototype_learning = float(self.base_prototype_learning)
        
        self.validate_parameters()
    
    def validate_parameters(self):
        """Validate model parameters."""
        if not 0.01 <= self.prototype_growth_rate <= 0.05:
            raise ValueError("prototype_growth_rate must be between 0.01 and 0.05")
            
        if not 0.6 <= self.max_prototype_weight <= 0.9:
            raise ValueError("max_prototype_weight must be between 0.6 and 0.9")
            
        if not 0.1 <= self.base_prototype_learning <= 0.5:
            raise ValueError("base_prototype_learning must be between 0.1 and 0.5")


class IdentityAgent(Agent):
    """Agent that uses group prototype for leadership decisions."""
    
    def __init__(self, id: int, rng: np.random.Generator, params: ModelParameters):
        """Initialize identity agent."""
        super().__init__(id, rng, params)
        
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


class IdentityModel(BaseLeadershipModel):
    """Social Identity model with group prototype development.
    
    The influence of the collective prototype grows continuously from the start,
    gradually shifting from individual schemas to group-level understanding.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize with identity parameters."""
        if isinstance(params, dict):
            self.params = IdentityParameters(**params)
        else:
            self.params = params if params else IdentityParameters()
        
        # Initialize random number generator
        self.rng = np.random.default_rng(self.params.random_seed)
        self.time = 0
        
        # Initialize agents
        self.agents = [
            IdentityAgent(i, self.rng, self.params)
            for i in range(self.params.n_agents)
        ]
        
        # Initialize group prototype for each dimension
        # Start at neutral position (50) for each dimension
        self.group_prototype = np.full(self.params.schema_dimensions, 50.0)
        
        # Track model state
        self.history = []
    
    def update_group_prototype(self, successful_leader_characteristic: np.ndarray):
        """Update group prototype based on successful leadership.
        
        The prototype moves toward the characteristics of successful leaders,
        with influence growing over time.
        """
        # Calculate prototype influence that grows over time
        prototype_weight = min(
            self.params.max_prototype_weight,
            self.time * self.params.prototype_growth_rate
        )
        
        # Adjust learning rate based on current influence
        effective_learning_rate = self.params.base_prototype_learning * prototype_weight
        
        # Calculate how much to move prototype toward successful leader's characteristics
        movement = effective_learning_rate * (successful_leader_characteristic - self.group_prototype)
        self.group_prototype += movement
        
        # Ensure prototype stays within valid range
        self.group_prototype = np.clip(self.group_prototype, 0, 100)
    
    def calculate_prototype_match(self, characteristic: np.ndarray) -> float:
        """Calculate how well characteristics match the group prototype."""
        if self.params.match_algorithm == "average":
            # Normalize differences to [0,1] range
            matches = [1 - abs(c - p)/100.0 for c, p in zip(characteristic, self.group_prototype)]
            return np.mean(matches)
        elif self.params.match_algorithm == "minimum":
            # Normalize differences to [0,1] range
            matches = [1 - abs(c - p)/100.0 for c, p in zip(characteristic, self.group_prototype)]
            return np.min(matches)
        else:  # weighted
            # Normalize differences to [0,1] range
            matches = [1 - abs(c - p)/100.0 for c, p in zip(characteristic, self.group_prototype)]
            return np.mean(matches)
    
    def step(self):
        """Execute one step of the model."""
        # Select interaction pair
        agent1, agent2 = self.select_interaction_pair()
        
        # Calculate match scores
        match_score_1 = agent1.calculate_ilt_match(agent2.characteristic)
        match_score_2 = agent2.calculate_ilt_match(agent1.characteristic)
        
        # Calculate prototype match scores
        prototype_match_1 = self.calculate_prototype_match(agent1.characteristic)
        prototype_match_2 = self.calculate_prototype_match(agent2.characteristic)
        
        # Calculate prototype influence that grows over time
        prototype_weight = min(
            self.params.max_prototype_weight,
            self.time * self.params.prototype_growth_rate
        )
        schema_weight = 1.0 - prototype_weight
        
        # Track interactions
        recent_interactions = []
        
        # Blend individual schema and group prototype from the start
        # Agent 1 claims
        blended_match_1 = (
            schema_weight * match_score_1 +
            prototype_weight * prototype_match_1
        )
        if blended_match_1 > self.params.match_threshold and self.rng.random() < self.params.base_claim_probability:
            granted = blended_match_1 > self.params.match_threshold
            if granted:
                # Update prototype based on successful leadership
                self.update_group_prototype(agent1.characteristic)
                # Update perceptions
                agent2.update_perception(agent1.id, self.params.success_boost)
                # Other agents also update perceptions based on prototype match
                for observer in self.agents:
                    if observer.id not in [agent1.id, agent2.id]:
                        observer.update_perception(agent1.id, self.params.success_boost)
            else:
                agent2.update_perception(agent1.id, -self.params.failure_penalty)
                # Other agents also update perceptions
                for observer in self.agents:
                    if observer.id not in [agent1.id, agent2.id]:
                        observer.update_perception(agent1.id, -self.params.failure_penalty)
            
            recent_interactions.append({
                'claimer': agent1.id,
                'target': agent2.id,
                'success': granted,
                'match_score': match_score_1,
                'prototype_match': prototype_match_1,
                'blended_match': blended_match_1
            })
        
        # Agent 2 claims if allowed
        blended_match_2 = (
            schema_weight * match_score_2 +
            prototype_weight * prototype_match_2
        )
        if blended_match_2 > self.params.match_threshold and self.rng.random() < self.params.base_claim_probability:
            if not recent_interactions or self.params.allow_mutual_claims:
                granted = blended_match_2 > self.params.match_threshold
                if granted:
                    # Update prototype based on successful leadership
                    self.update_group_prototype(agent2.characteristic)
                    # Update perceptions
                    agent1.update_perception(agent2.id, self.params.success_boost)
                    # Other agents also update perceptions based on prototype match
                    for observer in self.agents:
                        if observer.id not in [agent1.id, agent2.id]:
                            observer.update_perception(agent2.id, self.params.success_boost)
                else:
                    agent1.update_perception(agent2.id, -self.params.failure_penalty)
                    # Other agents also update perceptions
                    for observer in self.agents:
                        if observer.id not in [agent1.id, agent2.id]:
                            observer.update_perception(agent2.id, -self.params.failure_penalty)
                
                recent_interactions.append({
                    'claimer': agent2.id,
                    'target': agent1.id,
                    'success': granted,
                    'match_score': match_score_2,
                    'prototype_match': prototype_match_2,
                    'blended_match': blended_match_2
                })
        
        self.time += 1
        state = self.get_state()
        state['recent_interactions'] = recent_interactions
        state['group_prototype'] = self.group_prototype.tolist()
        return state
    
    def get_state(self) -> Dict:
        """Get current model state."""
        state = super().get_state()
        state['group_prototype'] = self.group_prototype.tolist()
        prototype_weight = min(
            self.params.max_prototype_weight,
            self.time * self.params.prototype_growth_rate
        )
        state['prototype_weight'] = prototype_weight
        return state
    
    def get_metrics(self) -> Dict:
        """Get model metrics."""
        metrics = super().get_metrics()
        prototype_weight = min(
            self.params.max_prototype_weight,
            self.time * self.params.prototype_growth_rate
        )
        metrics['prototype_weight'] = prototype_weight
        metrics['prototype'] = self.group_prototype.tolist()
        return metrics
