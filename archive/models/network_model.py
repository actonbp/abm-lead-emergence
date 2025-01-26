"""
Network-based leadership emergence model.
"""

from typing import Tuple
import networkx as nx
from src.models.base import BaseLeadershipModel, Agent


class NetworkModel(BaseLeadershipModel):
    """Model that adds network-based interaction selection."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.network_weight = config.get('network_weight', 0.3)
    
    def _select_interaction_pair(self):
        """Override to add network-based selection."""
        if not hasattr(self, 'last_interaction') or self.rng.random() < 0.3:  # Sometimes select random pair
            return super()._select_interaction_pair()
        
        # Prefer interactions with connected agents
        agent1 = self.rng.choice(self.agents)
        neighbors = list(self.interaction_network.neighbors(agent1.id))
        if neighbors:
            agent2_id = self.rng.choice(neighbors)
            agent2 = next(a for a in self.agents if a.id == agent2_id)
        else:
            agent2 = self.rng.choice([a for a in self.agents if a != agent1])
        return agent1, agent2
    
    def step(self):
        """Execute one step with network influence."""
        # Select interaction pair
        agent1, agent2 = self._select_interaction_pair()
        self.last_interaction = (agent1.id, agent2.id)
        
        # Calculate network influence
        centrality = nx.degree_centrality(self.interaction_network)
        agent1_centrality = centrality.get(agent1.id, 0)
        agent2_centrality = centrality.get(agent2.id, 0)
        
        # Get base claim decisions with network influence
        agent1_base_prob = agent1.last_claim_prob if hasattr(agent1, 'last_claim_prob') else 0.5
        agent2_base_prob = agent2.last_claim_prob if hasattr(agent2, 'last_claim_prob') else 0.5
        
        # Adjust probabilities with network position
        agent1_claim_prob = (1 - self.network_weight) * agent1_base_prob + self.network_weight * agent1_centrality
        agent2_claim_prob = (1 - self.network_weight) * agent2_base_prob + self.network_weight * agent2_centrality
        
        # Make claim decisions
        agent1_claims = self.rng.random() < agent1_claim_prob
        agent2_claims = self.rng.random() < agent2_claim_prob
        
        # Store claim decisions
        agent1.last_claimed = agent1_claims
        agent2.last_claimed = agent2_claims
        agent1.last_claim_prob = agent1_claim_prob
        agent2.last_claim_prob = agent2_claim_prob
        
        # Get grant decisions if there were claims
        agent1_grants = False
        agent2_grants = False
        if agent2_claims:
            # Base grant probability from ILT match
            agent1_grants = agent1.evaluate_grant(agent2, self.li_change_rate)
            # Adjust with network influence
            agent1_grant_prob = (1 - self.network_weight) * agent1.last_grant_prob + self.network_weight * agent1_centrality
            agent1_grants = self.rng.random() < agent1_grant_prob
            agent1.last_grant_prob = agent1_grant_prob
            
        if agent1_claims:
            # Base grant probability from ILT match
            agent2_grants = agent2.evaluate_grant(agent1, self.li_change_rate)
            # Adjust with network influence
            agent2_grant_prob = (1 - self.network_weight) * agent2.last_grant_prob + self.network_weight * agent2_centrality
            agent2_grants = self.rng.random() < agent2_grant_prob
            agent2.last_grant_prob = agent2_grant_prob
        
        # Update identities and network based on interaction outcome
        if agent1_claims and agent2_grants:
            agent1.update_identities(True, True, self.li_change_rate)
            agent2.update_identities(False, False, self.li_change_rate)
            # Update network - agent2 perceives agent1 as leader
            if not self.interaction_network.has_edge(agent2.id, agent1.id):
                self.interaction_network.add_edge(agent2.id, agent1.id, weight=1)
            else:
                self.interaction_network[agent2.id][agent1.id]['weight'] += 1
                
        if agent2_claims and agent1_grants:
            agent2.update_identities(True, True, self.li_change_rate)
            agent1.update_identities(False, False, self.li_change_rate)
            # Update network - agent1 perceives agent2 as leader
            if not self.interaction_network.has_edge(agent1.id, agent2.id):
                self.interaction_network.add_edge(agent1.id, agent2.id, weight=1)
            else:
                self.interaction_network[agent1.id][agent2.id]['weight'] += 1
        
        # Store interaction state for UI
        self.last_interaction_state = {
            'agent1': agent1.get_interaction_state(),
            'agent2': agent2.get_interaction_state()
        }
        
        return (agent1_claims and agent2_grants) or (agent2_claims and agent1_grants)