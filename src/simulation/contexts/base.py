"""
Base context class for modifying model behavior in specific scenarios.
"""

from typing import Dict, Any, Tuple
from pydantic import BaseModel


class ContextParameters(BaseModel):
    """Base parameters for contexts."""
    pass


class Context:
    """Base context class that defines the interface for all contexts."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize context with parameters."""
        self.params = ContextParameters(**(params or {}))
    
    def modify_claim_probability(self, base_prob: float, agent_state: Dict) -> float:
        """Modify an agent's claim probability based on context.
        
        Args:
            base_prob: The base probability calculated by the model
            agent_state: Current state of the agent making the decision
            
        Returns:
            Modified probability
        """
        return base_prob
    
    def modify_grant_probability(self, base_prob: float, agent_state: Dict, claimer_state: Dict) -> float:
        """Modify an agent's grant probability based on context.
        
        Args:
            base_prob: The base probability calculated by the model
            agent_state: Current state of the agent making the decision
            claimer_state: Current state of the agent claiming leadership
            
        Returns:
            Modified probability
        """
        return base_prob
    
    def modify_state_update(self, base_update: float, agent_state: Dict, interaction: Dict) -> float:
        """Modify how agent states are updated based on context.
        
        Args:
            base_update: The base update value calculated by the model
            agent_state: Current state of the agent being updated
            interaction: Details of the interaction that led to the update
            
        Returns:
            Modified update value
        """
        return base_update
    
    def get_state(self) -> Dict[str, Any]:
        """Get current context state."""
        return {
            'type': self.__class__.__name__,
            'params': self.params.dict()
        } 