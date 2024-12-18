"""
Base model class for leadership emergence simulations.
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from pydantic import BaseModel, Field
import yaml


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
        config: Dict[str, Any] = None,
        config_path: Optional[str] = None,
        random_seed: Optional[int] = None
    ):
        """Initialize model with configuration.
        
        Args:
            config: Dictionary of model parameters
            config_path: Path to YAML/JSON configuration file
            random_seed: Random seed for reproducibility
        """
        # Load and validate configuration
        if config_path:
            with open(config_path) as f:
                if config_path.endswith('.yaml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
        
        if config is None:
            config = {
                'n_agents': 4,
                'initial_li_equal': True,
                'li_change_rate': 2.0
            }
            
        self.params = ModelParameters(**config)
        
        # Initialize random state
        self.rng = np.random.default_rng(random_seed)
        
        # Initialize model state
        self.time = 0
        self.li_change_rate = self.params.li_change_rate
        
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
                leader_identity=50.0 if self.params.initial_li_equal 
                else self.rng.uniform(60, 80),
                characteristics=self.rng.uniform(40, 60),  # Random initial characteristics
                ilt=self.rng.uniform(40, 60)  # Random initial ILT
            )
            for i in range(self.params.n_agents)
        ]
        
        # Add nodes to interaction network
        for i in range(self.params.n_agents):
            self.interaction_network.add_node(i)
    
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
    
    def run(self, n_steps: int) -> Dict[str, Any]:
        """Run model for specified number of steps.
        
        Args:
            n_steps: Number of timesteps to run
            
        Returns:
            Dict containing simulation history
        """
        for _ in range(n_steps):
            state = self.step()
            self.history.append(state)
            
        return {
            'history': self.history,
            'parameters': self.params.model_dump()
        }
    
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
            agent1.leader_identity = min(100, agent1.leader_identity + self.params.li_change_rate)
            agent2.follower_identity = min(100, agent2.follower_identity + self.params.li_change_rate)
        elif claiming and not granting:
            agent1.leader_identity = max(0, agent1.leader_identity - self.params.li_change_rate)
        elif granting and not claiming:
            agent2.follower_identity = max(0, agent2.follower_identity - self.params.li_change_rate)
    
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
            'centralization': self.history['centralization'][-1] if self.history['centralization'] else 0.0,
            'density': self.history['density'][-1] if self.history['density'] else 0.0
        }