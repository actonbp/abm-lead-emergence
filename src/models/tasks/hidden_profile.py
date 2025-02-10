"""
Hidden Profile Task Implementation

This task simulates a group decision-making scenario where information is distributed
among group members, with some information shared and some unique to each member.
The optimal solution requires combining information from multiple members.
"""

from typing import Dict, Any, List
import numpy as np
from .base_task import BaseTask, TaskContext

class HiddenProfileTask(BaseTask):
    """Implementation of a hidden profile task for leadership emergence."""
    
    def __init__(self, n_agents: int, context: TaskContext):
        """Initialize hidden profile task.
        
        Args:
            n_agents: Number of agents in the group
            context: TaskContext instance defining the task conditions
        """
        self.n_agents = n_agents
        self.context = context
        
        # Generate task information structure
        self._generate_information()
    
    def _generate_information(self):
        """Generate shared and unique information for the task."""
        # Shared information (visible to all)
        self.shared_info = {
            "obvious_features": {
                "feature_1": 0.6,  # Somewhat positive
                "feature_2": 0.4,  # Somewhat negative
            },
            "common_knowledge": [
                "basic_fact_1",
                "basic_fact_2"
            ]
        }
        
        # Unique information (distributed among agents)
        self.unique_info = {}
        critical_features = [
            ("critical_1", 0.8),  # Strong positive
            ("critical_2", 0.2),  # Strong negative
            ("critical_3", 0.9),  # Strong positive
        ]
        
        # Distribute critical information among agents
        for i in range(self.n_agents):
            # Each agent gets some unique critical features
            agent_features = np.random.choice(
                critical_features,
                size=min(2, len(critical_features)),
                replace=False
            )
            self.unique_info[i] = {
                "critical_features": dict(agent_features),
                "private_knowledge": [f"unique_fact_{i}_1"]
            }
    
    def get_shared_info(self) -> Dict[str, Any]:
        """Get information shared among all agents."""
        return self.shared_info
    
    def get_unique_info(self, agent_id: int) -> Dict[str, Any]:
        """Get information unique to a specific agent."""
        return self.unique_info.get(agent_id, {})
    
    def evaluate_solution(self, proposed_solution: Dict[str, Any]) -> float:
        """Evaluate a proposed solution.
        
        The quality of the solution depends on:
        1. How many critical features were identified
        2. Whether the interpretation is correct
        3. How well information was integrated
        
        Returns:
            Float between 0 and 1 indicating solution quality
        """
        # Count critical features identified
        critical_features = set()
        for info in self.unique_info.values():
            critical_features.update(info["critical_features"].keys())
        
        identified_features = set(proposed_solution.get("identified_features", []))
        feature_score = len(identified_features & critical_features) / len(critical_features)
        
        # Check interpretation accuracy
        interpretation_score = 0.0
        if "interpretation" in proposed_solution:
            # Compare proposed interpretation with actual values
            correct = 0
            total = 0
            for feature, value in proposed_solution["interpretation"].items():
                if feature in self.shared_info["obvious_features"]:
                    actual = self.shared_info["obvious_features"][feature]
                    error = abs(value - actual)
                    correct += 1 - min(error, 1.0)
                    total += 1
                for info in self.unique_info.values():
                    if feature in info["critical_features"]:
                        actual = info["critical_features"][feature]
                        error = abs(value - actual)
                        correct += 1 - min(error, 1.0)
                        total += 1
            if total > 0:
                interpretation_score = correct / total
        
        # Weight the components
        total_score = 0.6 * feature_score + 0.4 * interpretation_score
        return total_score
    
    def get_context_modifiers(self) -> Dict[str, float]:
        """Get context-specific modifiers for the leadership model."""
        return self.context.get_modifiers() 