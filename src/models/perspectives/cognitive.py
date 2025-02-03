"""
Cognitive Model of Leadership Emergence

This model implements a cognitive learning perspective where agents update their
Implicit Leadership Theories (ILTs) by observing successful leadership interactions.
When agents observe successful leadership claims, they adjust their ILTs to be more
similar to the characteristics of successful leaders.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

from ..base_model import BaseLeadershipModel, ModelParameters, Agent


@dataclass
class CognitiveParameters(ModelParameters):
    """Parameters for the Cognitive model."""
    
    # Learning parameters
    ilt_learning_rate: float = 0.2  # How quickly ILTs adjust (0.1 to 0.5)
    observation_weight: float = 1.0  # Weight given to observed interactions (0.5 to 2.0)
    
    # Memory parameters
    memory_decay: float = 0.1  # How quickly old observations fade (0.0 to 0.3)
    max_memory: int = 10  # Maximum number of remembered interactions (5 to 20)
    
    # Success parameters
    success_boost: float = 10.0  # How much to boost perceptions for successful claims
    
    def __post_init__(self):
        """Convert parameters to appropriate types after initialization."""
        super().__post_init__()
        
        # Convert parameters
        self.ilt_learning_rate = float(self.ilt_learning_rate)
        self.observation_weight = float(self.observation_weight)
        self.memory_decay = float(self.memory_decay)
        self.max_memory = int(self.max_memory)
        self.success_boost = float(self.success_boost)
        
        self.validate_parameters()
    
    def validate_parameters(self):
        """Validate model parameters."""
        if not 0.1 <= self.ilt_learning_rate <= 0.5:
            raise ValueError("ilt_learning_rate must be between 0.1 and 0.5")
        
        if not 0.5 <= self.observation_weight <= 2.0:
            raise ValueError("observation_weight must be between 0.5 and 2.0")
            
        if not 0.0 <= self.memory_decay <= 0.3:
            raise ValueError("memory_decay must be between 0.0 and 0.3")
            
        if not 5 <= self.max_memory <= 20:
            raise ValueError("max_memory must be between 5 and 20")


class CognitiveAgent(Agent):
    """An agent with cognitive leadership perceptions."""
    
    def __init__(self, id: int, rng: np.random.Generator, params: CognitiveParameters):
        """Initialize a cognitive agent."""
        super().__init__(id, rng, params)
        self.params = params
        self.successful_leaders = {}  # Maps leader_id -> list of characteristics
        self.leadership_claims = 0
        self.leadership_grants = 0
        self.perceptions = {}  # Maps agent_id -> perception score
        
        # Initialize ILT schema and characteristic based on parameters
        if params.schema_type == "binary":
            self.ilt_schema = rng.choice([0, 100], size=params.schema_dimensions)
            self.characteristic = rng.choice([0, 100], size=params.schema_dimensions)
        else:
            if params.characteristic_distribution == "uniform":
                self.characteristic = rng.uniform(0, 100, params.schema_dimensions)
            else:  # normal
                self.characteristic = rng.normal(params.distribution_mean, params.distribution_std, params.schema_dimensions)
                self.characteristic = np.clip(self.characteristic, 0, 100)
                
            if params.ilt_distribution == "uniform":
                self.ilt_schema = rng.uniform(0, 100, params.schema_dimensions)
            else:  # normal
                self.ilt_schema = rng.normal(params.distribution_mean, params.distribution_std, params.schema_dimensions)
                self.ilt_schema = np.clip(self.ilt_schema, 0, 100)
        
    def update_ilt_from_observation(self, leader_id: int, leader_characteristic: np.ndarray):
        """Update ILT schema based on observing a successful leader."""
        # Add to successful leaders if not already present
        if leader_id not in self.successful_leaders:
            self.successful_leaders[leader_id] = []
        
        # Add this observation, weighted by recency
        self.successful_leaders[leader_id].append(leader_characteristic.copy())
        
        # Calculate weighted average of successful leader characteristics
        # More recent observations get higher weights
        all_characteristics = []
        all_weights = []
        
        for leader_id, observations in self.successful_leaders.items():
            n_obs = len(observations)
            for i, characteristic in enumerate(observations):
                # More recent observations get higher weights
                recency_weight = 1.0 + (i / n_obs)  # Weight increases with recency
                all_characteristics.append(characteristic)
                all_weights.append(recency_weight)
                
        if all_characteristics:
            # Convert to numpy arrays for weighted average calculation
            characteristics_array = np.array(all_characteristics)
            weights_array = np.array(all_weights)
            weights_array = weights_array / np.sum(weights_array)  # Normalize weights
            
            # Calculate weighted average target
            target = np.average(characteristics_array, axis=0, weights=weights_array)
            
            # Update ILT schema with adaptive learning rate
            # Learning rate increases with successful observations
            effective_rate = self.params.ilt_learning_rate * (1.0 + np.log1p(len(all_characteristics)) * 0.1)
            effective_rate = min(0.5, effective_rate)  # Cap at 0.5 to maintain stability
            
            # Update schema with momentum to smooth learning
            self.ilt_schema = self.ilt_schema + effective_rate * (target - self.ilt_schema)
            
            # Ensure schema stays within valid range
            self.ilt_schema = np.clip(self.ilt_schema, 0.0, 100.0)
    
    def get_state(self) -> dict:
        """Get the current state of the agent."""
        state = super().get_state()
        state.update({
            'successful_leaders': {k: [c.tolist() for c in v] for k, v in self.successful_leaders.items()}
        })
        return state


class CognitiveModel(BaseLeadershipModel):
    """Cognitive model where agents learn and adjust their ILTs through observation."""
    
    def __init__(self, params: CognitiveParameters):
        """Initialize the cognitive model."""
        super().__init__(params)
        self.params = params
        self.current_step = 0
        self.recent_interactions = []
        self.max_recent_interactions = 10
        
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
        """Execute one step of the model"""
        self.current_step += 1
        
        # Select agents for interaction
        if self.params.interaction_selection == "sequential":
            agent_i = self.agents[self.current_step % len(self.agents)]
            others = [a for a in self.agents if a != agent_i]
            agent_j = np.random.choice(others)
        else:
            agents = list(self.agents)  # Convert to list for random.sample
            agent_i, agent_j = np.random.choice(agents, size=2, replace=False)
        
        # Before step 20, use base model behavior
        if self.current_step < 20:
            # Use base model claim/grant logic
            claim_made = np.random.random() < self.params.base_claim_probability
            if claim_made:
                grant_made = np.random.random() < self.params.base_claim_probability
            else:
                grant_made = False

            if claim_made and grant_made:
                agent_i.leadership_claims += 1
                agent_j.leadership_grants += 1
                
                # Update perceptions using base model method
                for agent in self.agents:
                    if agent != agent_i:
                        current = agent.perceptions.get(agent_i.id, 0)
                        agent.perceptions[agent_i.id] = current + self.params.success_boost
        
        # After step 20, use cognitive model behavior
        else:
            # Calculate match score for potential leadership claim
            match_score = self._calculate_match_score(agent_i.characteristic, agent_j.ilt_schema)
            
            # Decide whether to make leadership claim based on match score
            claim_made = match_score > self.params.match_threshold
            
            # Process claim if made
            if claim_made:
                grant_made = match_score > self.params.match_threshold
                
                if grant_made:
                    # Record successful interaction
                    self.process_leadership_claim(agent_i, agent_j, True, match_score)
                    
                    # Update recent interactions list
                    interaction = {
                        'step': self.current_step,
                        'claimer': agent_i.id,
                        'target': agent_j.id,
                        'success': True,
                        'match_score': match_score
                    }
                    self.recent_interactions.append(interaction)
                    if len(self.recent_interactions) > self.max_recent_interactions:
                        self.recent_interactions.pop(0)
            else:
                grant_made = False

        return {
            'step': self.current_step,
            'interacting_agents': (agent_i.id, agent_j.id),
            'claims_made': claim_made,
            'grants_made': grant_made,
            'agents': [agent.get_state() for agent in self.agents],
            'recent_interactions': self.recent_interactions
        }

    def _calculate_match_score(self, characteristic: np.ndarray, ilt_schema: np.ndarray) -> float:
        """Calculate how well a characteristic matches an ILT schema."""
        if self.params.match_algorithm == "euclidean":
            distance = np.linalg.norm(ilt_schema - characteristic)
            max_distance = np.linalg.norm(np.array([100, 100]))  # Maximum possible distance
            return 1.0 - (distance / max_distance)
        elif self.params.match_algorithm == "cosine":
            return np.dot(ilt_schema, characteristic) / (np.linalg.norm(ilt_schema) * np.linalg.norm(characteristic))
        elif self.params.match_algorithm == "average":
            # Calculate average difference across dimensions
            diff = np.abs(ilt_schema - characteristic)
            avg_diff = np.mean(diff)
            return 1.0 - (avg_diff / 100.0)  # Normalize to [0,1]
        else:
            raise ValueError(f"Unknown match algorithm: {self.params.match_algorithm}")

    def process_leadership_claim(self, claimer: Agent, target: Agent, success: bool, match_score: float):
        """Process a leadership claim between two agents."""
        # Update perceptions based on claim outcome
        if success:
            # Update ILTs based on successful leadership
            target.update_ilt_from_observation(claimer.id, claimer.characteristic)
            for observer in self.agents:
                if observer.id != claimer.id and observer.id != target.id:
                    observer.update_ilt_from_observation(claimer.id, claimer.characteristic)
            
            # Boost target's perception of claimer based on match quality
            if claimer.id not in target.perceptions:
                target.perceptions[claimer.id] = 50.0
            boost = self.params.success_boost * (1.0 + match_score)
            target.perceptions[claimer.id] = min(100.0, target.perceptions[claimer.id] + boost)
            
            # Slightly decrease target's perception of others
            for other_id in target.perceptions:
                if other_id != claimer.id:
                    target.perceptions[other_id] = max(0.0, target.perceptions[other_id] - boost * 0.1)
            
            # Update observers' perceptions
            for observer in self.agents:
                if observer.id != claimer.id and observer.id != target.id:
                    if claimer.id not in observer.perceptions:
                        observer.perceptions[claimer.id] = 50.0
                    observer_match = self._calculate_match_score(claimer.characteristic, observer.ilt_schema)
                    observer_boost = self.params.success_boost * (1.0 + observer_match) * 0.5
                    observer.perceptions[claimer.id] = min(100.0, observer.perceptions[claimer.id] + observer_boost)
                    
                    # Slightly decrease observer's perception of others
                    for other_id in observer.perceptions:
                        if other_id != claimer.id:
                            observer.perceptions[other_id] = max(0.0, observer.perceptions[other_id] - observer_boost * 0.1)
        else:
            # Apply penalties for failed claims
            if claimer.id not in target.perceptions:
                target.perceptions[claimer.id] = 50.0
            penalty = self.params.failure_penalty * (1.0 + (1.0 - match_score))
            target.perceptions[claimer.id] = max(0.0, target.perceptions[claimer.id] - penalty)
            
            # Update observers' perceptions
            for observer in self.agents:
                if observer.id != claimer.id and observer.id != target.id:
                    if claimer.id not in observer.perceptions:
                        observer.perceptions[claimer.id] = 50.0
                    observer_match = self._calculate_match_score(claimer.characteristic, observer.ilt_schema)
                    observer_penalty = self.params.failure_penalty * (1.0 + (1.0 - observer_match)) * 0.5
                    observer.perceptions[claimer.id] = max(0.0, observer.perceptions[claimer.id] - observer_penalty)
