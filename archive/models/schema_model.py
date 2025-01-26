"""
Schema-based leadership emergence model.
"""

from typing import Tuple
from src.models.base_model import BaseLeadershipModel, Agent


class SchemaModel(BaseLeadershipModel):
    """Model that adds schema-based decision making."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add schema-related attributes to agents
        for agent in self.agents:
            agent.characteristics = self.rng.uniform(10, 90)
            agent.ilt = self.rng.uniform(10, 90)
    
    def _process_interaction(self, agent1: Agent, agent2: Agent) -> Tuple[bool, bool]:
        """Override to add schema-based decisions."""
        # Calculate schema-characteristic fit
        claim_similarity = 1 - abs(agent1.characteristics - agent1.ilt) / 100
        grant_similarity = 1 - abs(agent1.characteristics - agent2.ilt) / 100
        
        # Combine with base identity probabilities
        base_claim_prob = agent1.leader_identity / 100
        base_grant_prob = agent2.follower_identity / 100
        
        claim_prob = 0.5 * (base_claim_prob + claim_similarity)
        grant_prob = 0.5 * (base_grant_prob + grant_similarity)
        
        claiming = self.rng.random() < claim_prob
        granting = self.rng.random() < grant_prob
        
        return claiming, granting