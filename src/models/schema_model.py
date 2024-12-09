"""
Schema-based leadership emergence model.

This model extends the base model by adding:
1. Leadership characteristics for each agent
2. Implicit Leadership Theories (ILTs) that influence perceptions
3. Schema-based decision making for claiming and granting
"""

import numpy as np
from typing import Dict, Tuple, Any
from dataclasses import dataclass

from .base_model import BaseLeadershipModel, Agent

@dataclass
class SchemaAgent(Agent):
    """Agent with leadership schemas and characteristics."""
    characteristics: float = None  # Actual leadership attributes
    ilt: float = None             # Implicit Leadership Theory (ideal leader prototype)
    
    def __post_init__(self):
        super().__post_init__()
        self.schema_history = []  # Track schema updates
        
    def update_schema(self, interaction_success: bool, other_characteristics: float):
        """Update ILT based on interaction outcomes."""
        if interaction_success:
            # Move ILT slightly toward successful interaction partner
            self.ilt = self.ilt + 0.1 * (other_characteristics - self.ilt)
        else:
            # Move ILT slightly away from unsuccessful interaction partner
            self.ilt = self.ilt - 0.1 * (other_characteristics - self.ilt)
        
        # Keep ILT in valid range
        self.ilt = np.clip(self.ilt, 0, 100)
        self.schema_history.append(self.ilt)

class SchemaModel(BaseLeadershipModel):
    """Model incorporating schema-based leadership perceptions."""
    
    def __init__(
        self,
        n_agents: int = 4,
        initial_li_equal: bool = True,
        li_change_rate: float = 2.0,
        schema_weight: float = 0.5,  # Weight for schema vs identity in decisions
        random_seed: int = None
    ):
        super().__init__(
            n_agents=n_agents,
            initial_li_equal=initial_li_equal,
            li_change_rate=li_change_rate,
            random_seed=random_seed
        )
        self.schema_weight = schema_weight
        
        # Initialize agents with characteristics and schemas
        self.agents = [
            SchemaAgent(
                id=i,
                leader_identity=50.0 if initial_li_equal else self.rng.uniform(60, 80),
                characteristics=self.rng.uniform(0, 100),
                ilt=self.rng.uniform(0, 100)
            )
            for i in range(n_agents)
        ]
        
        # Add schema-related tracking to history
        self.history.update({
            'characteristics': [],
            'ilts': [],
            'schema_similarity': []
        })
    
    def _process_interaction(self, agent1: SchemaAgent, agent2: SchemaAgent) -> Tuple[bool, bool]:
        """Process interaction using schema-based decisions."""
        # Calculate schema-based probabilities
        claim_schema_fit = 1 - abs(agent1.characteristics - agent1.ilt) / 100
        grant_schema_fit = 1 - abs(agent1.characteristics - agent2.ilt) / 100
        
        # Combine with identity-based probabilities
        claim_identity = agent1.leader_identity / 100
        grant_identity = agent2.follower_identity / 100
        
        # Weight schema and identity influences
        claim_prob = (1 - self.schema_weight) * claim_identity + self.schema_weight * claim_schema_fit
        grant_prob = (1 - self.schema_weight) * grant_identity + self.schema_weight * grant_schema_fit
        
        # Make decisions
        claiming = self.rng.random() < claim_prob
        granting = self.rng.random() < grant_prob
        
        # Update schemas based on interaction outcome
        interaction_success = claiming and granting
        agent1.update_schema(interaction_success, agent2.characteristics)
        agent2.update_schema(interaction_success, agent1.characteristics)
        
        return claiming, granting
    
    def _track_outcomes(self):
        """Track additional schema-related outcomes."""
        super()._track_outcomes()
        
        # Track characteristics and ILTs
        self.history['characteristics'].append(
            [agent.characteristics for agent in self.agents]
        )
        self.history['ilts'].append(
            [agent.ilt for agent in self.agents]
        )
        
        # Track schema similarity (how well agents agree on leader prototype)
        similarities = []
        for i, agent1 in enumerate(self.agents):
            for j, agent2 in enumerate(self.agents):
                if i < j:  # Only calculate each pair once
                    similarity = 1 - abs(agent1.ilt - agent2.ilt) / 100
                    similarities.append(similarity)
        
        self.history['schema_similarity'].append(np.mean(similarities))
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get current state including schema information."""
        state = super()._get_current_state()
        state.update({
            'characteristics': [agent.characteristics for agent in self.agents],
            'ilts': [agent.ilt for agent in self.agents],
            'schema_similarity': self.history['schema_similarity'][-1]
        })
        return state 