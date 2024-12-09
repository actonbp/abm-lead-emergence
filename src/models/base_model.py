import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class Agent:
    """An agent in the leadership emergence model."""
    id: int
    leader_identity: float = 50.0    # Initial LI (can be uniform or varied)
    follower_identity: float = 50.0  # Initial FI (usually uniform)
    characteristics: float = None     # Leadership characteristics
    ilt: float = None                # Implicit Leadership Theory
    
    def __post_init__(self):
        """Initialize agent's history tracking."""
        self.perceived_leadership = {}  # How others view this agent
        self.interaction_count = 0
        self.claims_history = []
        self.grants_history = []
        self.leader_identity_history = [self.leader_identity]
        self.follower_identity_history = [self.follower_identity]
    
    def update_history(self):
        """Update history after identity changes."""
        self.leader_identity_history.append(self.leader_identity)
        self.follower_identity_history.append(self.follower_identity)

class BaseLeadershipModel:
    """Base class for all leadership emergence models."""
    
    def __init__(
        self,
        n_agents: int = 4,
        initial_li_equal: bool = True,
        li_change_rate: float = 2.0,
        random_seed: int = None
    ):
        self.n_agents = n_agents
        self.time = 0
        self.rng = np.random.default_rng(random_seed)
        self.li_change_rate = li_change_rate
        
        # Initialize network for tracking interactions
        self.interaction_network = nx.DiGraph()
        
        # Track outcomes over time
        self.history = {
            'leader_identities': [],
            'follower_identities': [],
            'centralization': [],
            'density': [],
            'interaction_patterns': []
        }
        
        # Initialize agents with base attributes
        self.agents = [
            Agent(
                id=i,
                leader_identity=50.0 if initial_li_equal else self.rng.uniform(60, 80)
            )
            for i in range(n_agents)
        ]
        for agent in self.agents:
            self.interaction_network.add_node(agent.id)
    
    def step(self) -> Dict:
        """Execute one step of the model."""
        # Select interaction pair
        agent1, agent2 = self._select_interaction_pair()
        
        # Process interaction
        claiming, granting = self._process_interaction(agent1, agent2)
        
        # Update based on interaction
        self._update_identities(agent1, agent2, claiming, granting)
        self._update_network(agent1, agent2, claiming, granting)
        
        # Update histories for all agents
        for agent in self.agents:
            agent.update_history()
        
        # Track outcomes
        self._track_outcomes()
        
        self.time += 1
        return self._get_current_state()
    
    def _select_interaction_pair(self) -> Tuple[Agent, Agent]:
        """Base method for selecting interaction pairs (random)."""
        agents = self.rng.choice(self.agents, size=2, replace=False)
        return agents[0], agents[1]
    
    def _process_interaction(self, agent1: Agent, agent2: Agent) -> Tuple[bool, bool]:
        """Base method for processing interactions (identity-based)."""
        claim_prob = agent1.leader_identity / 100
        grant_prob = agent2.follower_identity / 100
        
        claiming = self.rng.random() < claim_prob
        granting = self.rng.random() < grant_prob
        
        return claiming, granting
    
    def _update_identities(self, agent1: Agent, agent2: Agent, claiming: bool, granting: bool):
        """Base method for updating identities."""
        if claiming and granting:
            agent1.leader_identity = min(100, agent1.leader_identity + self.li_change_rate)
            agent2.follower_identity = min(100, agent2.follower_identity + self.li_change_rate)
        elif claiming and not granting:
            agent1.leader_identity = max(0, agent1.leader_identity - self.li_change_rate)
        elif granting and not claiming:
            agent2.follower_identity = max(0, agent2.follower_identity - self.li_change_rate)
    
    def _update_network(self, agent1: Agent, agent2: Agent, claiming: bool, granting: bool):
        """Base method for updating network."""
        weight = 0.1  # Base interaction weight
        if claiming:
            weight += 0.1
        if granting:
            weight += 0.1
            
        if self.interaction_network.has_edge(agent1.id, agent2.id):
            self.interaction_network[agent1.id][agent2.id]['weight'] += weight
        else:
            self.interaction_network.add_edge(agent1.id, agent2.id, weight=weight)
    
    def _track_outcomes(self):
        """Track various outcomes over time."""
        self.history['leader_identities'].append(
            [agent.leader_identity for agent in self.agents]
        )
        self.history['follower_identities'].append(
            [agent.follower_identity for agent in self.agents]
        )
        
        centralization = self._calculate_centralization()
        density = nx.density(self.interaction_network)
        
        self.history['centralization'].append(centralization)
        self.history['density'].append(density)
    
    def _calculate_centralization(self) -> float:
        """Calculate leadership centralization using network metrics."""
        try:
            centrality = nx.degree_centrality(self.interaction_network)
            return np.var(list(centrality.values()))
        except:
            return 0.0
    
    def _get_current_state(self) -> Dict:
        """Return current state of the model."""
        return {
            'time': self.time,
            'leader_identities': [agent.leader_identity for agent in self.agents],
            'follower_identities': [agent.follower_identity for agent in self.agents],
            'centralization': self.history['centralization'][-1],
            'density': self.history['density'][-1]
        }

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

class NetworkModel(BaseLeadershipModel):
    """Model that adds network-based interaction selection."""
    
    def __init__(self, network_weight: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.network_weight = network_weight
    
    def _select_interaction_pair(self) -> Tuple[Agent, Agent]:
        """Override to add network-based selection."""
        if self.time == 0 or self.rng.random() < 0.3:  # Sometimes select random pair
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
    
    def _process_interaction(self, agent1: Agent, agent2: Agent) -> Tuple[bool, bool]:
        """Override to add network position influence."""
        # Get base probabilities
        base_claim_prob = agent1.leader_identity / 100
        base_grant_prob = agent2.follower_identity / 100
        
        # Add network centrality influence
        centrality = nx.degree_centrality(self.interaction_network)
        agent1_centrality = centrality.get(agent1.id, 0)
        agent2_centrality = centrality.get(agent2.id, 0)
        
        claim_prob = (1 - self.network_weight) * base_claim_prob + self.network_weight * agent1_centrality
        grant_prob = (1 - self.network_weight) * base_grant_prob + self.network_weight * agent2_centrality
        
        claiming = self.rng.random() < claim_prob
        granting = self.rng.random() < grant_prob
        
        return claiming, granting

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