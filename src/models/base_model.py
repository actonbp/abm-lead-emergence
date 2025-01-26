"""
Base model class for leadership emergence simulations.
Provides core claim-grant logic without perspective-specific features.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from pydantic import BaseModel, Field


class ModelParameters(BaseModel):
    """Core parameter validation schema."""
    
    # Core Parameters
    n_agents: int = Field(ge=2, le=100, description="Number of agents")
    claim_threshold: float = Field(
        default=0.5,
        gt=0.0, le=1.0,
        description="Base threshold for leadership claims"
    )
    grant_threshold: float = Field(
        default=0.4,
        gt=0.0, le=1.0,
        description="Base threshold for granting leadership"
    )
    
    # Interaction Parameters
    claim_multiplier: float = Field(
        default=0.7,
        gt=0.0, le=1.0,
        description="Multiplier for claim probability"
    )
    grant_multiplier: float = Field(
        default=0.6,
        gt=0.0, le=1.0,
        description="Multiplier for grant probability"
    )
    
    # Update Parameters
    success_boost: float = Field(
        default=5.0,
        gt=0.0, le=10.0,
        description="How much leadership score increases on successful claim"
    )
    failure_penalty: float = Field(
        default=3.0,
        gt=0.0, le=10.0,
        description="How much leadership score decreases on failed claim"
    )
    grant_penalty: float = Field(
        default=2.0,
        gt=0.0, le=10.0,
        description="How much leadership score decreases when granting"
    )


@dataclass
class Agent:
    """Base agent with minimal leadership mechanics."""
    id: int
    rng: np.random.Generator
    params: ModelParameters
    
    # Core attributes
    lead_score: float = 50.0  # Basic leadership measure
    
    def __post_init__(self):
        """Initialize agent's state tracking."""
        self.history = {
            'lead_score': [self.lead_score],
            'claims': [],
            'grants': []
        }
        
        self.last_interaction = {
            'claim_prob': 0,
            'grant_prob': 0,
            'claimed': False,
            'granted': False,
            'other_id': None,  # Track who we interacted with
            'role': None       # 'claimer' or 'granter'
        }
    
    def decide_claim(self, other: 'Agent') -> bool:
        """Basic claim decision logic."""
        claim_prob = self.lead_score / 100 * self.params.claim_multiplier
        will_claim = self.rng.random() < claim_prob
        
        # Update interaction state
        self.last_interaction.update({
            'claim_prob': claim_prob,
            'claimed': will_claim,
            'other_id': other.id,
            'role': 'claimer'
        })
        
        self.history['claims'].append(will_claim)
        
        return will_claim
    
    def decide_grant(self, claimer: 'Agent') -> bool:
        """Basic grant decision logic."""
        # Lower leadership score = more likely to grant
        grant_prob = (1 - self.lead_score / 100) * self.params.grant_multiplier
        will_grant = self.rng.random() < grant_prob
        
        # Update interaction state
        self.last_interaction.update({
            'grant_prob': grant_prob,
            'granted': will_grant,
            'other_id': claimer.id,
            'role': 'granter'
        })
        
        self.history['grants'].append(will_grant)
        
        return will_grant
    
    def update_score(self, delta: float):
        """Update leadership score within bounds."""
        self.lead_score = max(0, min(100, self.lead_score + delta))
        self.history['lead_score'].append(self.lead_score)
    
    def get_state(self) -> Dict:
        """Get current agent state."""
        return {
            'id': self.id,
            'lead_score': self.lead_score,
            'last_interaction': self.last_interaction.copy()
        }


class BaseLeadershipModel:
    """Base class with core claim-grant mechanics."""
    
    def __init__(
        self,
        params: Dict[str, Any] = None,
        random_seed: Optional[int] = None
    ):
        """Initialize model with parameters."""
        self.params = ModelParameters(**(params or {}))
        self.rng = np.random.default_rng(random_seed)
        self.time = 0
        
        # Initialize agents
        self.agents = [
            Agent(i, self.rng, self.params)
            for i in range(self.params.n_agents)
        ]
        
        # Track model history
        self.history = []
    
    def step(self) -> Dict[str, Any]:
        """Execute one simulation step."""
        # Select interaction pair
        pair = self._select_interaction_pair()
        
        # Process claim-grant interaction
        interaction = self._process_interaction(pair)
        
        # Update states based on interaction
        self._update_states(pair, interaction)
        
        # Increment time
        self.time += 1
        
        # Track state
        state = self.get_state()
        self.history.append(state)
        
        return state
    
    def run(self, n_steps: int) -> List[Dict[str, Any]]:
        """Run simulation for specified number of steps."""
        history = []
        for _ in range(n_steps):
            state = self.step()
            history.append(state)
        return history
    
    def _select_interaction_pair(self) -> Tuple[Agent, Agent]:
        """Select two agents for interaction."""
        pair = self.rng.choice(self.agents, size=2, replace=False)
        return tuple(pair)
    
    def _process_interaction(self, pair: Tuple[Agent, Agent]) -> Dict:
        """Process claim-grant interaction between agents."""
        claimer, granter = pair
        
        # Get claim and grant decisions
        claimed = claimer.decide_claim(granter)
        granted = granter.decide_grant(claimer) if claimed else False
        
        return {
            'claimed': claimed,
            'granted': granted,
            'claimer_id': claimer.id,
            'granter_id': granter.id
        }
    
    def _update_states(self, pair: Tuple[Agent, Agent], interaction: Dict):
        """Update agent states based on interaction."""
        claimer, granter = pair
        
        if interaction['claimed'] and interaction['granted']:
            # Successful leadership claim
            claimer.update_score(self.params.success_boost)
            granter.update_score(-self.params.grant_penalty)
        elif interaction['claimed'] and not interaction['granted']:
            # Failed leadership claim
            claimer.update_score(-self.params.failure_penalty)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current model state."""
        return {
            'time': self.time,
            'agents': [agent.get_state() for agent in self.agents]
        }