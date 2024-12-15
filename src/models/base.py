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
        self.identity = 50  # Start at neutral identity
        
        # Last interaction state
        self.last_interaction = {
            'claim_prob': 0,
            'grant_prob': 0,
            'claimed': False,
            'granted': False,
            'ilt_match': 0
        }
    
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
            return 1.0 if abs(characteristics - ilt) <= threshold else 0.0
        
        else:
            # Default to euclidean if unknown algorithm
            diff = abs(characteristics - ilt) / 100
            return 1 - diff
    
    def decide_claim(self):
        """Decide whether to claim leadership based on self-perception."""
        # Calculate match using selected algorithm
        self_match = self.calculate_match(self.characteristics, self.ilt)
        
        # Identity influences claim probability
        claim_prob = self_match * (self.identity / 100) * 0.7
        
        self.last_interaction['claim_prob'] = claim_prob
        self.last_interaction['claimed'] = self.rng.random() < claim_prob
        return self.last_interaction['claimed']
    
    def evaluate_grant(self, other_agent):
        """Evaluate whether to grant leadership to another agent."""
        # Calculate match using selected algorithm
        other_match = self.calculate_match(other_agent.characteristics, self.ilt)
        
        # Lower identity and good match needed for granting
        grant_prob = other_match * (1 - self.identity/100) * 0.6
        
        self.last_interaction['ilt_match'] = other_match
        self.last_interaction['grant_prob'] = grant_prob
        self.last_interaction['granted'] = self.rng.random() < grant_prob
        return self.last_interaction['granted']
    
    def update_identity(self, granted_leadership, claimed_leadership, change_rate):
        """Update identity based on interaction outcome.
        
        Args:
            granted_leadership: Whether this agent was granted leadership by the other
            claimed_leadership: Whether this agent claimed leadership
            change_rate: Base rate for identity changes
        """
        # Successful claim is when both claimed and granted
        successful_claim = claimed_leadership and granted_leadership
        
        if successful_claim:
            # Successful leadership claim (claimed and granted) strengthens identity
            self.identity = min(100, self.identity + change_rate)
        elif claimed_leadership and not granted_leadership:
            # Failed leadership claim (claimed but not granted) strongly weakens identity
            self.identity = max(0, self.identity - 2 * change_rate)
        else:
            # Not claiming leadership reduces identity by base amount
            self.identity = max(0, self.identity - change_rate)
    
    def get_state(self):
        """Get current agent state."""
        return {
            'id': self.id,
            'characteristics': self.characteristics,
            'ilt': self.ilt,
            'identity': self.identity,
            'last_interaction': self.last_interaction.copy()
        }


class BaseLeadershipModel:
    """Base model for leadership emergence simulation."""
    
    def __init__(self, config=None):
        """Initialize the model with given configuration."""
        self.config = config or {}
        self.n_agents = config.get('n_agents', 6)
        self.identity_change_rate = config.get('identity_change_rate', 2.0)
        self.rng = np.random.default_rng()
        
        # Initialize agents
        self.agents = [Agent(i, self.rng) for i in range(self.n_agents)]
        
        # Initialize interaction network
        self.interaction_network = nx.DiGraph()
        for i in range(self.n_agents):
            self.interaction_network.add_node(i)
    
    def _select_interaction_pair(self):
        """Select two random agents for interaction."""
        agents = self.rng.choice(self.agents, size=2, replace=False)
        return agents[0], agents[1]
    
    def step(self):
        """Execute one step of the model."""
        # Select interaction pair
        agent1, agent2 = self._select_interaction_pair()
        self.last_interaction = (agent1.id, agent2.id)
        
        # Get claim decisions
        agent1_claims = agent1.decide_claim()
        agent2_claims = agent2.decide_claim()
        
        # Get grant decisions if there were claims
        agent1_grants = False
        agent2_grants = False
        if agent2_claims:
            agent1_grants = agent1.evaluate_grant(agent2)
        if agent1_claims:
            agent2_grants = agent2.evaluate_grant(agent1)
        
        # Update identities and network based on interaction outcome
        if agent1_claims and agent2_grants:
            agent1.update_identity(True, True, self.identity_change_rate)
            agent2.update_identity(False, False, self.identity_change_rate)
            # Update network - agent2 perceives agent1 as leader
            if not self.interaction_network.has_edge(agent2.id, agent1.id):
                self.interaction_network.add_edge(agent2.id, agent1.id, weight=1)
            else:
                self.interaction_network[agent2.id][agent1.id]['weight'] += 1
                
        if agent2_claims and agent1_grants:
            agent2.update_identity(True, True, self.identity_change_rate)
            agent1.update_identity(False, False, self.identity_change_rate)
            # Update network - agent1 perceives agent2 as leader
            if not self.interaction_network.has_edge(agent1.id, agent2.id):
                self.interaction_network.add_edge(agent1.id, agent2.id, weight=1)
            else:
                self.interaction_network[agent1.id][agent2.id]['weight'] += 1
        
        # Store interaction state for UI
        self.last_interaction_state = {
            'agent1': agent1.get_state(),
            'agent2': agent2.get_state(),
            'grant_given': (agent1_claims and agent2_grants) or (agent2_claims and agent1_grants)
        }
        
        return self.last_interaction_state['grant_given']
    
    def run(self, n_steps: int) -> Dict[str, Any]:
        """Run model for specified number of steps."""
        history = []
        for step in range(n_steps):
            # Execute step and record state
            self.step()
            
            # Record state
            state = {
                'timestep': step,
                'agents': [agent.get_state() for agent in self.agents],
                'network': list(self.interaction_network.edges(data=True)),
                'last_interaction': self.last_interaction_state
            }
            history.append(state)
            
        return {
            'history': history,
            'final_state': {
                'agents': [agent.get_state() for agent in self.agents],
                'network': list(self.interaction_network.edges(data=True))
            },
            'parameters': {
                'n_agents': self.n_agents,
                'identity_change_rate': self.identity_change_rate
            }
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current model state.
        
        Returns:
            Dict containing current state variables
        """
        return {
            'timestep': self.timestep,
            'leader_identities': self.agents.copy(),
            'parameters': self.params.model_dump()
        } 