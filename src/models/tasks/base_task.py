"""
Base Task Interface for Leadership Emergence Models

This module defines the interface for tasks that can be used with any leadership
emergence model. The task integrates information sharing with leadership claims.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np

class BaseTask(ABC):
    """Abstract base class for tasks in the leadership emergence framework.
    
    In this framework:
    - Leadership claims are made by sharing information
    - Granting leadership means accepting/integrating shared information
    - Task success depends on how well information is shared and integrated
    """
    
    @abstractmethod
    def get_shared_info(self, agent_id: int) -> Dict[str, Any]:
        """Get currently shared/visible information for an agent."""
        pass
    
    @abstractmethod
    def get_unique_info(self, agent_id: int) -> Dict[str, Any]:
        """Get information unique to a specific agent (their private info)."""
        pass
    
    @abstractmethod
    def share_info(self, from_agent: int, to_agent: int, granted: bool) -> Dict[str, Any]:
        """Share information from one agent to another through a claim.
        
        Args:
            from_agent: ID of agent making leadership claim (sharing info)
            to_agent: ID of agent receiving claim
            granted: Whether the leadership claim was granted
            
        Returns:
            Dict containing:
            - shared_info: What information was shared
            - success: Whether sharing was successful
            - quality: Quality of information shared (0-1)
        """
        pass
    
    @abstractmethod
    def evaluate_current_solution(self) -> float:
        """Evaluate current group solution based on shared information.
        
        Returns:
            Score between 0-1 indicating solution quality
        """
        pass

class TaskContext:
    """Defines the context in which a task is performed."""
    
    def __init__(self, 
                 context_type: str = "none",
                 time_pressure: float = 0.0,
                 complexity: float = 0.0,
                 uncertainty: float = 0.0):
        """Initialize task context.
        
        Args:
            context_type: "none" (default), "crisis", "routine", or "creative"
            time_pressure: 0-1 scale of time pressure
            complexity: 0-1 scale of task complexity
            uncertainty: 0-1 scale of outcome uncertainty
        """
        self.context_type = context_type
        self.time_pressure = np.clip(time_pressure, 0, 1)
        self.complexity = np.clip(complexity, 0, 1)
        self.uncertainty = np.clip(uncertainty, 0, 1)
    
    def get_modifiers(self) -> Dict[str, float]:
        """Get context-specific modifiers based on context type and parameters."""
        # Base case - no modifiers
        return {
            "claim_threshold_modifier": 1.0,
            "grant_threshold_modifier": 1.0,
            "learning_rate_modifier": 1.0,
            "prototype_influence_modifier": 1.0
        } 