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
    # Stage transition parameters
    dyadic_interactions_before_switch: int = 10  # Number of dyadic interactions before switching to Stage 2
    
    # Identity update parameters
    identity_update_rate: float = 0.2  # How quickly identities change
    perception_update_rate: float = 0.2  # How quickly perceptions change
    
    # Penalties and boosts
    claim_success_boost: float = 5.0  # Boost to leader identity on successful claim
    grant_success_boost: float = 5.0  # Boost to follower identity on successful grant
    rejection_penalty: float = 3.0  # Penalty for rejected claims
    passivity_penalty: float = 1.0  # Penalty for not claiming when should
    
    def __post_init__(self):
        """Convert parameters to appropriate types after initialization."""
        super().__post_init__()
        
        # Convert parameters
        self.dyadic_interactions_before_switch = int(self.dyadic_interactions_before_switch)
        self.identity_update_rate = float(self.identity_update_rate)
        self.perception_update_rate = float(self.perception_update_rate)
        self.claim_success_boost = float(self.claim_success_boost)
        self.grant_success_boost = float(self.grant_success_boost)
        self.rejection_penalty = float(self.rejection_penalty)
        self.passivity_penalty = float(self.passivity_penalty)
        
        self.validate_parameters()
    
    def validate_parameters(self):
        """Validate model parameters."""
        if not 5 <= self.dyadic_interactions_before_switch <= 20:
            raise ValueError("dyadic_interactions_before_switch must be between 5 and 20")
        
        if not 0.1 <= self.identity_update_rate <= 0.8:
            raise ValueError("identity_update_rate must be between 0.1 and 0.8")
            
        if not 0.1 <= self.perception_update_rate <= 0.8:
            raise ValueError("perception_update_rate must be between 0.1 and 0.8")
            
        if not 3.0 <= self.claim_success_boost <= 15.0:
            raise ValueError("claim_success_boost must be between 3.0 and 15.0")
            
        if not 3.0 <= self.grant_success_boost <= 15.0:
            raise ValueError("grant_success_boost must be between 3.0 and 15.0")
            
        if not 1.0 <= self.rejection_penalty <= 10.0:
            raise ValueError("rejection_penalty must be between 1.0 and 10.0")
            
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
            'granted': False
        }
    
    def decide_claim(self, match_score: float, step: int) -> bool:
        """Decide whether to make leadership claim based on schema match and identity."""
        # Calculate identity-based probability
        identity_prob = self.leader_identity / 100.0
        
        if step < self.params.dyadic_interactions_before_switch:
            # Use only schema matching before switch
            claim_probability = match_score * self.params.base_claim_probability
        else:
            # In Stage 2, blend schema and identity
            schema_weight = 0.2  # Keep 20% schema influence
            identity_weight = 0.8  # 80% identity influence
            
            claim_probability = (
                schema_weight * match_score * self.params.base_claim_probability +
                identity_weight * identity_prob
            )
        
        # Add small noise
        noise = self.rng.normal(0, 0.05)
        claim_probability = np.clip(claim_probability + noise, 0, 1)
        
        # Store last interaction state
        self.last_interaction['match_score'] = match_score
        self.last_interaction['claimed'] = self.rng.random() < claim_probability
        
        return self.last_interaction['claimed']
    
    def decide_grant(self, match_score: float, step: int) -> bool:
        """Decide whether to grant leadership claim based on schema match and identity."""
        # Calculate identity-based probability
        identity_prob = self.follower_identity / 100.0
        
        if step < self.params.dyadic_interactions_before_switch:
            # Use only schema matching before switch
            grant_probability = match_score
        else:
            # In Stage 2, blend schema and identity
            schema_weight = 0.2  # Keep 20% schema influence
            identity_weight = 0.8  # 80% identity influence
            
            grant_probability = (
                schema_weight * match_score +
                identity_weight * identity_prob
            )
        
        # Only grant if above threshold
        if grant_probability <= self.params.match_threshold:
            grant_probability = 0
            
        # Add small noise
        noise = self.rng.normal(0, 0.05)
        grant_probability = np.clip(grant_probability + noise, 0, 1)
        
        # Store last interaction state
        self.last_interaction['granted'] = self.rng.random() < grant_probability
        
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
        """Execute one step of the model following the two-stage process."""
        # Select interaction pair using base model's selection
        agent1, agent2 = self.select_interaction_pair()
        
        # Calculate match scores
        match_score_1 = agent1.calculate_ilt_match(agent2.characteristic)
        match_score_2 = agent2.calculate_ilt_match(agent1.characteristic)
        
        # Track interactions
        recent_interactions = []
        
        # Stage 1: Initial Interaction Based on Leadership Schemas
        if self.time < self.params.dyadic_interactions_before_switch:
            # Decisions based purely on schema matching
            agent1_claims = match_score_1 > self.params.match_threshold and self.rng.random() < self.params.base_claim_probability
            agent2_claims = match_score_2 > self.params.match_threshold and self.rng.random() < self.params.base_claim_probability
            
            if agent1_claims:
                granted = match_score_1 > self.params.match_threshold
                if granted:
                    # Update identities based on successful claim
                    agent1.leader_identity = min(100, agent1.leader_identity + self.params.claim_success_boost)
                    agent2.follower_identity = min(100, agent2.follower_identity + self.params.grant_success_boost)
                    # Update perceptions
                    agent2.update_perception(agent1.id, self.params.claim_success_boost)
                else:
                    # Penalty for rejected claim
                    agent1.leader_identity = max(0, agent1.leader_identity - self.params.rejection_penalty)
                    agent2.update_perception(agent1.id, -self.params.rejection_penalty)
                
                recent_interactions.append({
                    'claimer': agent1.id,
                    'target': agent2.id,
                    'success': granted,
                    'match_score': match_score_1
                })
            
            if agent2_claims:
                granted = match_score_2 > self.params.match_threshold
                if granted:
                    # Update identities based on successful claim
                    agent2.leader_identity = min(100, agent2.leader_identity + self.params.claim_success_boost)
                    agent1.follower_identity = min(100, agent1.follower_identity + self.params.grant_success_boost)
                    # Update perceptions
                    agent1.update_perception(agent2.id, self.params.claim_success_boost)
                else:
                    # Penalty for rejected claim
                    agent2.leader_identity = max(0, agent2.leader_identity - self.params.rejection_penalty)
                    agent1.update_perception(agent2.id, -self.params.rejection_penalty)
                
                recent_interactions.append({
                    'claimer': agent2.id,
                    'target': agent1.id,
                    'success': granted,
                    'match_score': match_score_2
                })
        
        # Stage 2: Contextualized Identity Based Interaction
        else:
            # Decisions based on identities and schemas
            agent1_claims = agent1.decide_claim(match_score_1, self.time)
            agent2_claims = agent2.decide_claim(match_score_2, self.time)
            
            if agent1_claims:
                granted = agent2.decide_grant(match_score_1, self.time)
                if granted:
                    # Update identities and perceptions
                    identity_boost = self.params.claim_success_boost * self.params.identity_update_rate
                    perception_boost = self.params.claim_success_boost * self.params.perception_update_rate
                    
                    # Success reinforces both identities
                    agent1.leader_identity = min(100, agent1.leader_identity + identity_boost)
                    agent2.follower_identity = min(100, agent2.follower_identity + identity_boost)
                    
                    # All agents update their perceptions
                    for observer in self.agents:
                        if observer.id != agent1.id:
                            observer.update_perception(agent1.id, perception_boost)
                else:
                    # Penalty for rejected claim
                    identity_penalty = self.params.rejection_penalty * self.params.identity_update_rate
                    perception_penalty = self.params.rejection_penalty * self.params.perception_update_rate
                    
                    # Failed claim affects both identities
                    agent1.leader_identity = max(0, agent1.leader_identity - identity_penalty)
                    agent1.follower_identity = min(100, agent1.follower_identity + identity_penalty * 0.5)
                    
                    # All agents update their perceptions
                    for observer in self.agents:
                        if observer.id != agent1.id:
                            observer.update_perception(agent1.id, -perception_penalty)
                
                recent_interactions.append({
                    'claimer': agent1.id,
                    'target': agent2.id,
                    'success': granted,
                    'match_score': match_score_1
                })
            
            if agent2_claims:
                granted = agent1.decide_grant(match_score_2, self.time)
                if granted:
                    # Update identities and perceptions
                    identity_boost = self.params.claim_success_boost * self.params.identity_update_rate
                    perception_boost = self.params.claim_success_boost * self.params.perception_update_rate
                    
                    # Success reinforces both identities
                    agent2.leader_identity = min(100, agent2.leader_identity + identity_boost)
                    agent1.follower_identity = min(100, agent1.follower_identity + identity_boost)
                    
                    # All agents update their perceptions
                    for observer in self.agents:
                        if observer.id != agent2.id:
                            observer.update_perception(agent2.id, perception_boost)
                else:
                    # Penalty for rejected claim
                    identity_penalty = self.params.rejection_penalty * self.params.identity_update_rate
                    perception_penalty = self.params.rejection_penalty * self.params.perception_update_rate
                    
                    # Failed claim affects both identities
                    agent2.leader_identity = max(0, agent2.leader_identity - identity_penalty)
                    agent2.follower_identity = min(100, agent2.follower_identity + identity_penalty * 0.5)
                    
                    # All agents update their perceptions
                    for observer in self.agents:
                        if observer.id != agent2.id:
                            observer.update_perception(agent2.id, -perception_penalty)
                
                recent_interactions.append({
                    'claimer': agent2.id,
                    'target': agent1.id,
                    'success': granted,
                    'match_score': match_score_2
                })
        
        self.time += 1
        state = self.get_state()
        state['recent_interactions'] = recent_interactions
        return state
    
    def get_state(self):
        """Get current model state."""
        return {
            'time': self.time,
            'agents': [agent.get_state() for agent in self.agents],
            'recent_interactions': []  # Initialize empty list for recent interactions
        }
    
    def get_metrics(self) -> Dict:
        """Get model metrics."""
        metrics = super().get_metrics()  # Get base metrics
        metrics.update({
            'stage': 2 if self.time >= self.params.dyadic_interactions_before_switch else 1,
            'using_identities': self.time >= self.params.dyadic_interactions_before_switch
        })
        return metrics
