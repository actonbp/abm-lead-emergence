"""
Social Interactionist perspective on leadership emergence.
Focuses on how leadership emerges through social interactions and shared schemas.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np

from ..base_model import BaseLeadershipModel, Agent, ModelParameters
from pydantic import Field


class InteractionistParameters(ModelParameters):
    """Parameters specific to the Social Interactionist perspective."""
    
    # Schema Parameters
    schema_weight: float = Field(
        default=0.7,
        gt=0.0, le=1.0,
        description="Weight given to schema-based decisions"
    )
    identity_weight: float = Field(
        default=0.3,
        gt=0.0, le=1.0,
        description="Weight given to identity-based decisions"
    )
    
    # Perception Parameters
    perception_change_success: float = Field(
        default=2.0,
        gt=0.0, le=5.0,
        description="How much perception increases after successful claim"
    )
    perception_change_reject: float = Field(
        default=3.0,
        gt=0.0, le=5.0,
        description="How much perception decreases after rejected claim"
    )


@dataclass
class InteractionistAgent(Agent):
    """Agent with social interactionist-specific features."""
    
    # Additional attributes
    characteristics: float = field(init=False)  # Leadership characteristics
    schema: float = field(init=False)  # Leadership schema/prototype
    perceptions: Dict[int, float] = field(default_factory=dict)  # Perceptions of others
    
    def __post_init__(self):
        """Initialize social interactionist-specific state."""
        super().__post_init__()
        
        # Initialize random characteristics and schema
        self.characteristics = self.rng.uniform(20, 80)
        self.schema = self.rng.uniform(20, 80)
        
        # Track perceptions
        self.perceptions = {}  # id -> perception score
    
    def calculate_schema_match(self, other: 'InteractionistAgent') -> float:
        """Calculate how well other agent matches leadership schema."""
        diff = abs(other.characteristics - self.schema)
        k = 0.15  # Controls steepness
        midpoint = 20  # Point of steepest change
        
        return 1 / (1 + np.exp(k * (diff - midpoint)))
    
    def decide_claim(self, other: 'InteractionistAgent') -> bool:
        """Decide whether to claim leadership based on schema and identity."""
        # Get schema-based score
        schema_score = self.calculate_schema_match(self)
        
        # Get identity-based score
        identity_score = self.lead_score / 100
        
        # Combine scores using weights
        claim_prob = (
            self.params.schema_weight * schema_score +
            self.params.identity_weight * identity_score
        ) * self.params.claim_multiplier
        
        will_claim = self.rng.random() < claim_prob
        
        # Update interaction state
        self.last_interaction.update({
            'claim_prob': claim_prob,
            'claimed': will_claim,
            'schema_score': schema_score,
            'identity_score': identity_score
        })
        
        self.history['claims'].append(will_claim)
        
        return will_claim
    
    def decide_grant(self, claimer: 'InteractionistAgent') -> bool:
        """Decide whether to grant leadership based on schema and perceptions."""
        # Get schema-based score
        schema_score = self.calculate_schema_match(claimer)
        
        # Get perception-based score
        perception_score = self.perceptions.get(claimer.id, 50) / 100
        
        # Combine scores using weights
        grant_prob = (
            self.params.schema_weight * schema_score +
            self.params.identity_weight * perception_score
        ) * self.params.grant_multiplier
        
        will_grant = self.rng.random() < grant_prob
        
        # Update interaction state
        self.last_interaction.update({
            'grant_prob': grant_prob,
            'granted': will_grant,
            'schema_score': schema_score,
            'perception_score': perception_score
        })
        
        self.history['grants'].append(will_grant)
        
        return will_grant
    
    def update_perception(self, other_id: int, change: float):
        """Update perception of another agent."""
        if other_id not in self.perceptions:
            self.perceptions[other_id] = 50.0  # Start neutral
        
        current = self.perceptions[other_id]
        self.perceptions[other_id] = max(0, min(100, current + change))
    
    def get_state(self) -> Dict:
        """Get current agent state including social interactionist features."""
        state = super().get_state()
        state.update({
            'characteristics': self.characteristics,
            'schema': self.schema,
            'perceptions': self.perceptions.copy()
        })
        return state


class InteractionistModel(BaseLeadershipModel):
    """Leadership emergence model from social interactionist perspective."""
    
    def __init__(self, params: Dict[str, Any] = None, random_seed: int = None):
        """Initialize with social interactionist parameters."""
        # Convert to perspective-specific parameters
        self.params = InteractionistParameters(**(params or {}))
        self.rng = np.random.default_rng(random_seed)
        self.time = 0
        
        # Initialize agents with perspective-specific class
        self.agents = [
            InteractionistAgent(i, self.rng, self.params)
            for i in range(self.params.n_agents)
        ]
        
        # Track model history
        self.history = []
    
    def _update_states(self, pair: tuple[Agent, Agent], interaction: Dict):
        """Update agent states with social interactionist logic."""
        claimer, granter = pair
        
        if interaction['claimed'] and interaction['granted']:
            # Successful leadership claim
            claimer.update_score(2.0)
            granter.update_score(-1.0)
            # Update granter's perception of claimer
            granter.update_perception(
                claimer.id,
                self.params.perception_change_success
            )
        elif interaction['claimed'] and not interaction['granted']:
            # Failed leadership claim
            claimer.update_score(-1.0)
            # Update granter's perception of claimer negatively
            granter.update_perception(
                claimer.id,
                -self.params.perception_change_reject
            )
