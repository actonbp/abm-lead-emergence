"""
Base model class for leadership emergence simulations.
"""

from typing import Dict, Any, Optional
import json
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field
import networkx as nx


class ModelParameters(BaseModel):
    """Parameter validation schema for leadership models."""
    
    # Agent Properties
    n_agents: int = Field(ge=2, le=100, description="Number of agents")
    initial_li_equal: bool = Field(
        description="Whether agents start with equal leadership identities"
    )
    li_change_rate: float = Field(
        gt=0.0, le=5.0,
        description="Rate of leadership identity change"
    )
    
    # ILT Matching Parameters
    ilt_match_algorithm: str = Field(
        default="euclidean",
        description="Algorithm to use for ILT matching: euclidean, gaussian, sigmoid, threshold"
    )
    ilt_match_params: dict = Field(
        default={
            "sigma": 20.0,  # For gaussian
            "k": 10.0,      # For sigmoid
            "threshold": 15.0  # For threshold
        },
        description="Parameters for the ILT matching algorithm"
    )
    
    # Optional parameters with defaults
    interaction_radius: Optional[float] = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="Radius for agent interactions"
    )
    memory_length: Optional[int] = Field(
        default=0,
        ge=0,
        description="Number of past interactions to remember"
    )


class Agent:
    """Agent in the leadership emergence model."""
    
    def __init__(self, id, rng, match_algorithm="euclidean", match_params=None):
        self.id = id
        self.rng = rng
        self.match_algorithm = match_algorithm
        self.match_params = match_params or {
            "sigma": 20.0,
            "k": 10.0,
            "threshold": 15.0
        }
        
        # Core attributes - random initial values in moderate range
        self.characteristics = rng.uniform(40, 60)  # Leadership characteristics
        self.ilt = rng.uniform(40, 60)  # Implicit Leadership Theory
        self.leader_identity = 50  # Start at neutral leader identity
        self.follower_identity = 50  # Start at neutral follower identity
        
        # Last interaction state
        self.last_interaction = {
            'claim_prob': 0,
            'grant_prob': 0,
            'claimed': False,
            'granted': False,
            'ilt_match': 0
        }
    
    def update_identity(self, leader_change=0, follower_change=0):
        """Update both leader and follower identities."""
        self.leader_identity = max(0, min(100, self.leader_identity + leader_change))
        self.follower_identity = max(0, min(100, self.follower_identity + follower_change))
    
    def calculate_match(self, characteristics, ilt):
        """Calculate match between characteristics and ILT using selected algorithm."""
        if self.match_algorithm == "euclidean":
            # Linear distance-based match (current approach)
            diff = abs(characteristics - ilt) / 100
            return (1 - diff) ** 2
        
        elif self.match_algorithm == "gaussian":
            # Gaussian similarity - emphasizes close matches
            sigma = self.match_params["sigma"]
            return np.exp(-((characteristics - ilt)**2) / (2 * sigma**2))
        
        elif self.match_algorithm == "sigmoid":
            # Sigmoid function - smooth threshold
            k = self.match_params["k"]
            diff = abs(characteristics - ilt) / 100
            return 1 / (1 + np.exp(-k * (1 - diff)))
        
        elif self.match_algorithm == "threshold":
            # Hard threshold-based matching
            threshold = self.match_params["threshold"]
            diff = abs(characteristics - ilt)
            return 1.0 if diff <= threshold else 0.0
    
    def decide_claim(self):
        """Decide whether to claim leadership based on self-perception."""
        # Calculate match using selected algorithm
        self_match = self.calculate_match(self.characteristics, self.ilt)
        
        # Identity influences claim probability
        claim_prob = self_match * (self.leader_identity / 100) * 0.7
        
        self.last_interaction['claim_prob'] = claim_prob
        self.last_interaction['claimed'] = self.rng.random() < claim_prob
        return self.last_interaction['claimed']
    
    def evaluate_grant(self, other_agent):
        """Evaluate whether to grant leadership to another agent."""
        # Calculate match using selected algorithm
        other_match = self.calculate_match(other_agent.characteristics, self.ilt)
        
        # Lower identity and good match needed for granting
        grant_prob = other_match * (1 - self.follower_identity/100) * 0.6
        
        self.last_interaction['ilt_match'] = other_match
        self.last_interaction['grant_prob'] = grant_prob
        self.last_interaction['granted'] = self.rng.random() < grant_prob
        return self.last_interaction['granted']
    
    def get_state(self):
        """Get current agent state."""
        return {
            'id': self.id,
            'characteristics': self.characteristics,
            'ilt': self.ilt,
            'leader_identity': self.leader_identity,
            'follower_identity': self.follower_identity,
            'last_interaction': self.last_interaction.copy()
        }


class BaseLeadershipModel:
    """Base model for leadership emergence simulation."""
    
    def __init__(self, config):
        """Initialize model with given configuration."""
        self.config = config
        self.rng = np.random.default_rng()
        
        # Initialize agents
        self.n_agents = config['n_agents']
        self.agents = [
            Agent(i, self.rng, 
                 match_algorithm=config.get('ilt_match_algorithm', 'euclidean'),
                 match_params=config.get('ilt_match_params', None))
            for i in range(self.n_agents)
        ]
        
        # Initialize interaction network
        self.interaction_network = nx.DiGraph()
        for i in range(self.n_agents):
            self.interaction_network.add_node(i)
        
        # Set initial identities
        if config.get('initial_li_equal', True):
            initial_li = config.get('initial_identity', 50)
            initial_fi = config.get('initial_identity', 50)
            for agent in self.agents:
                agent.leader_identity = initial_li
                agent.follower_identity = initial_fi
        else:
            for agent in self.agents:
                agent.leader_identity = self.rng.uniform(30, 70)
                agent.follower_identity = self.rng.uniform(30, 70)
    
    def _select_interaction_pair(self):
        """Select two agents for interaction."""
        agent_indices = self.rng.choice(self.n_agents, size=2, replace=False)
        return self.agents[agent_indices[0]], self.agents[agent_indices[1]]
    
    def step(self):
        """Execute one step of the simulation."""
        # Select interaction pair
        agent1, agent2 = self._select_interaction_pair()
        self.last_interaction = (agent1.id, agent2.id)
        
        # Calculate match between characteristics and ILTs
        agent1_self_match = agent1.calculate_match(agent1.characteristics, agent1.ilt)
        agent2_self_match = agent2.calculate_match(agent2.characteristics, agent2.ilt)
        
        agent1_other_match = agent1.calculate_match(agent1.characteristics, agent2.ilt)
        agent2_other_match = agent2.calculate_match(agent2.characteristics, agent1.ilt)
        
        # Store match values for visualization
        self.last_agent1_match = agent1_other_match
        self.last_agent2_match = agent2_other_match
        
        # Probabilistic claiming based on self-match and current identity
        agent1_claim_prob = agent1_self_match * (agent1.leader_identity / 100) * self.config.get('claim_multiplier', 0.7)
        agent2_claim_prob = agent2_self_match * (agent2.leader_identity / 100) * self.config.get('claim_multiplier', 0.7)
        
        agent1_claims = self.rng.random() < agent1_claim_prob
        agent2_claims = self.rng.random() < agent2_claim_prob
        
        # Store claim information for visualization
        self.last_agent1_claimed = agent1_claims
        self.last_agent2_claimed = agent2_claims
        self.last_agent1_claim_prob = agent1_claim_prob
        self.last_agent2_claim_prob = agent2_claim_prob
        
        # Calculate grant probabilities based on other-match and current identity
        agent1_grant_prob = 0
        agent2_grant_prob = 0
        if agent1_claims:
            agent2_grant_prob = agent1_other_match * (agent2.follower_identity / 100) * self.config.get('grant_multiplier', 0.6)
        if agent2_claims:
            agent1_grant_prob = agent2_other_match * (agent1.follower_identity / 100) * self.config.get('grant_multiplier', 0.6)
        
        # Make grant decisions probabilistically
        agent1_grants = False
        agent2_grants = False
        if agent2_claims:
            agent1_grants = self.rng.random() < agent1_grant_prob
        if agent1_claims:
            agent2_grants = self.rng.random() < agent2_grant_prob
        
        # Store grant information for visualization
        self.last_agent1_granted = agent1_grants
        self.last_agent2_granted = agent2_grants
        self.last_agent1_grant_prob = agent1_grant_prob
        self.last_agent2_grant_prob = agent2_grant_prob
        
        # Update model state based on claims and grants
        grant_given = (agent1_claims and agent2_grants) or (agent2_claims and agent1_grants)
        self.last_grant_given = grant_given
        
        # Update identities based on interaction outcomes
        identity_change_rate = self.config.get('identity_change_rate', 2.0)
        
        if agent1_claims:
            if agent2_grants:
                # Claim was granted
                agent1.update_identity(identity_change_rate, -identity_change_rate)
                # Update network
                if not self.interaction_network.has_edge(agent2.id, agent1.id):
                    self.interaction_network.add_edge(agent2.id, agent1.id, weight=1)
                else:
                    self.interaction_network[agent2.id][agent1.id]['weight'] += 1
            else:
                # Claim was rejected
                agent1.update_identity(-identity_change_rate, identity_change_rate/2)
        
        if agent2_claims:
            if agent1_grants:
                # Claim was granted
                agent2.update_identity(identity_change_rate, -identity_change_rate)
                # Update network
                if not self.interaction_network.has_edge(agent1.id, agent2.id):
                    self.interaction_network.add_edge(agent1.id, agent2.id, weight=1)
                else:
                    self.interaction_network[agent1.id][agent2.id]['weight'] += 1
            else:
                # Claim was rejected
                agent2.update_identity(-identity_change_rate, identity_change_rate/2)
        
        # Update granting agent identities
        if agent1_grants:
            # Agent 1 gave grant
            agent1.update_identity(-identity_change_rate/2, identity_change_rate)
        elif agent2_claims:  # Agent 1 withheld grant
            agent1.update_identity(identity_change_rate/4, -identity_change_rate/4)
            
        if agent2_grants:
            # Agent 2 gave grant
            agent2.update_identity(-identity_change_rate/2, identity_change_rate)
        elif agent1_claims:  # Agent 2 withheld grant
            agent2.update_identity(identity_change_rate/4, -identity_change_rate/4)
    
    def get_state(self):
        """Get current model state."""
        return {
            'agents': [agent.get_state() for agent in self.agents],
            'network': self.interaction_network.copy(),
            'last_interaction': getattr(self, 'last_interaction', None),
            'last_grant_given': getattr(self, 'last_grant_given', None)
        } 