"""
Crisis context that modifies leadership dynamics during crisis situations.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field

from .base import Context, ContextParameters


class CrisisParameters(ContextParameters):
    """Parameters for crisis context."""
    
    # Crisis intensity affects how much behaviors are modified
    intensity: float = Field(
        default=0.7,
        gt=0.0, le=1.0,
        description="Intensity of the crisis (higher = stronger effects)"
    )
    
    # How much more likely agents are to claim/grant during crisis
    claim_boost: float = Field(
        default=1.5,
        gt=1.0, le=3.0,
        description="Multiplier for claim probabilities during crisis"
    )
    grant_boost: float = Field(
        default=1.3,
        gt=1.0, le=3.0,
        description="Multiplier for grant probabilities during crisis"
    )
    
    # How much faster identities change during crisis
    update_multiplier: float = Field(
        default=1.5,
        gt=1.0, le=3.0,
        description="Multiplier for state updates during crisis"
    )


class CrisisContext(Context):
    """Context that modifies behavior during crisis situations.
    
    During crises:
    1. Agents are more likely to claim leadership (seeking direction)
    2. Agents are more likely to grant leadership (seeking stability)
    3. Leadership identities change faster (accelerated emergence)
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize crisis context with parameters."""
        self.params = CrisisParameters(**(params or {}))
    
    def modify_claim_probability(self, base_prob: float, agent_state: Dict) -> float:
        """Increase claim probability during crisis."""
        # Higher intensity = stronger boost to claiming
        boost = 1.0 + (self.params.claim_boost - 1.0) * self.params.intensity
        return min(1.0, base_prob * boost)
    
    def modify_grant_probability(self, base_prob: float, agent_state: Dict, claimer_state: Dict) -> float:
        """Increase grant probability during crisis."""
        # Higher intensity = stronger boost to granting
        boost = 1.0 + (self.params.grant_boost - 1.0) * self.params.intensity
        return min(1.0, base_prob * boost)
    
    def modify_state_update(self, base_update: float, agent_state: Dict, interaction: Dict) -> float:
        """Accelerate identity changes during crisis."""
        # Higher intensity = faster state updates
        boost = 1.0 + (self.params.update_multiplier - 1.0) * self.params.intensity
        return base_update * boost 