"""
Combined schema and network leadership emergence model.
"""

from typing import Tuple
import networkx as nx
from src.models.base_model import Agent
from src.models.network_model import NetworkModel
from src.models.schema_model import SchemaModel


class SchemaNetworkModel(NetworkModel, SchemaModel):
    """Model combining both schema and network effects."""
    
    def _process_interaction(self, agent1: Agent, agent2: Agent) -> Tuple[bool, bool]:
        """Combine schema and network influences."""
        # Get schema-based probabilities
        claim_similarity = 1 - abs(agent1.characteristics - agent1.ilt) / 100
        grant_similarity = 1 - abs(agent1.characteristics - agent2.ilt) / 100
        
        # Get network-based probabilities
        centrality = nx.degree_centrality(self.interaction_network)
        agent1_centrality = centrality.get(agent1.id, 0)
        agent2_centrality = centrality.get(agent2.id, 0)
        
        # Base identity probabilities
        base_claim_prob = agent1.leader_identity / 100
        base_grant_prob = agent2.follower_identity / 100
        
        # Combine all influences equally
        claim_prob = (base_claim_prob + claim_similarity + agent1_centrality) / 3
        grant_prob = (base_grant_prob + grant_similarity + agent2_centrality) / 3
        
        claiming = self.rng.random() < claim_prob
        granting = self.rng.random() < grant_prob
        
        return claiming, granting 