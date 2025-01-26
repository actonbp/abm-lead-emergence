"""
Theory validation component for leadership emergence simulations.
Compares simulation results with theoretical predictions.
"""

from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum

class TheoryType(Enum):
    SIP = "social_interactionist"
    SCP = "social_cognitive"
    SIT = "social_identity"

@dataclass
class TheoryPrediction:
    """Represents theoretical predictions for leadership emergence."""
    emergence_speed: float  # Expected number of steps to stable structure
    stability: float  # Expected stability score (0-1)
    hierarchy_clarity: float  # Expected hierarchy clarity score (0-1)
    role_differentiation: float  # Expected role differentiation score (0-1)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "emergence_speed": self.emergence_speed,
            "stability": self.stability,
            "hierarchy_clarity": self.hierarchy_clarity,
            "role_differentiation": self.role_differentiation
        }

class TheoryValidator:
    """Validates simulation results against theoretical predictions."""
    
    def __init__(self):
        # Define theoretical predictions for each theory
        self.predictions = {
            TheoryType.SIP: TheoryPrediction(
                emergence_speed=0.6,  # Moderate speed
                stability=0.8,  # High stability
                hierarchy_clarity=0.7,  # Clear hierarchy
                role_differentiation=0.8  # Strong differentiation
            ),
            TheoryType.SCP: TheoryPrediction(
                emergence_speed=0.8,  # Fast emergence
                stability=0.6,  # Moderate stability
                hierarchy_clarity=0.8,  # Very clear hierarchy
                role_differentiation=0.7  # Moderate differentiation
            ),
            TheoryType.SIT: TheoryPrediction(
                emergence_speed=0.4,  # Slower emergence
                stability=0.9,  # Very high stability
                hierarchy_clarity=0.6,  # Moderate clarity
                role_differentiation=0.9  # Very strong differentiation
            )
        }
    
    def validate_results(
        self,
        simulation_results: List[Dict],
        theory_type: TheoryType
    ) -> Dict[str, Any]:
        """
        Validate simulation results against theoretical predictions.
        
        Args:
            simulation_results: List of simulation results
            theory_type: Type of theory to validate against
            
        Returns:
            Dictionary of validation metrics
        """
        # Get theoretical predictions
        predictions = self.predictions[theory_type]
        
        # Calculate metrics from simulation results
        metrics = self._calculate_metrics(simulation_results)
        
        # Compare with predictions
        validation_scores = self._compare_with_predictions(metrics, predictions)
        
        # Calculate overall alignment score
        alignment_score = np.mean(list(validation_scores.values()))
        
        return {
            "theory_type": theory_type.value,
            "metrics": metrics,
            "predictions": predictions.to_dict(),
            "validation_scores": validation_scores,
            "overall_alignment": float(alignment_score)
        }
    
    def compare_theories(
        self,
        simulation_results: List[Dict]
    ) -> Dict[str, Any]:
        """
        Compare simulation results with all theories.
        
        Args:
            simulation_results: List of simulation results
            
        Returns:
            Dictionary of comparison results
        """
        comparisons = {}
        best_alignment = 0.0
        best_theory = None
        
        # Validate against each theory
        for theory in TheoryType:
            validation = self.validate_results(simulation_results, theory)
            comparisons[theory.value] = validation
            
            if validation["overall_alignment"] > best_alignment:
                best_alignment = validation["overall_alignment"]
                best_theory = theory
        
        return {
            "theory_comparisons": comparisons,
            "best_fit_theory": best_theory.value,
            "best_fit_score": float(best_alignment)
        }
    
    def _calculate_metrics(self, simulation_results: List[Dict]) -> Dict[str, float]:
        """Calculate metrics from simulation results."""
        all_histories = [result["history"] for result in simulation_results]
        
        # Calculate emergence speed
        emergence_speeds = []
        for history in all_histories:
            speed = self._calculate_emergence_speed(history)
            emergence_speeds.append(speed)
        
        # Calculate stability
        stabilities = []
        for history in all_histories:
            stability = self._calculate_stability(history)
            stabilities.append(stability)
        
        # Calculate hierarchy clarity
        clarities = []
        for history in all_histories:
            clarity = self._calculate_hierarchy_clarity(history)
            clarities.append(clarity)
        
        # Calculate role differentiation
        differentiations = []
        for history in all_histories:
            diff = self._calculate_role_differentiation(history)
            differentiations.append(diff)
        
        return {
            "emergence_speed": float(np.mean(emergence_speeds)),
            "stability": float(np.mean(stabilities)),
            "hierarchy_clarity": float(np.mean(clarities)),
            "role_differentiation": float(np.mean(differentiations))
        }
    
    def _compare_with_predictions(
        self,
        metrics: Dict[str, float],
        predictions: TheoryPrediction
    ) -> Dict[str, float]:
        """Compare metrics with theoretical predictions."""
        scores = {}
        
        for key in metrics:
            # Calculate similarity score (1 - normalized absolute difference)
            pred_value = getattr(predictions, key)
            metric_value = metrics[key]
            
            difference = abs(pred_value - metric_value)
            max_difference = 1.0  # Since all metrics are normalized to [0,1]
            
            similarity = 1.0 - (difference / max_difference)
            scores[key] = float(similarity)
        
        return scores
    
    def _calculate_emergence_speed(self, history: List[Dict]) -> float:
        """Calculate emergence speed from simulation history."""
        # Implementation depends on specific metrics used
        # For now, use a simple proxy based on identity variance
        n_steps = len(history)
        variances = []
        
        for state in history:
            li_values = [agent["leader_identity"] for agent in state]
            variance = np.var(li_values)
            variances.append(variance)
        
        # Normalize to [0,1] where 1 means fast emergence
        return 1.0 - (np.argmax(variances) / n_steps)
    
    def _calculate_stability(self, history: List[Dict]) -> float:
        """Calculate stability from simulation history."""
        # Look at last 25% of history
        cutoff = int(len(history) * 0.75)
        recent_history = history[cutoff:]
        
        # Calculate variance in leader identities
        variances = []
        for state in recent_history:
            li_values = [agent["leader_identity"] for agent in state]
            variance = np.var(li_values)
            variances.append(variance)
        
        # Normalize to [0,1] where 1 means very stable
        return 1.0 - (np.std(variances) / np.mean(variances))
    
    def _calculate_hierarchy_clarity(self, history: List[Dict]) -> float:
        """Calculate hierarchy clarity from simulation history."""
        # Use final state
        final_state = history[-1]
        
        # Calculate separation between leader and follower identities
        li_values = [agent["leader_identity"] for agent in final_state]
        fi_values = [agent["follower_identity"] for agent in final_state]
        
        # Calculate mean difference and normalize
        differences = np.array(li_values) - np.array(fi_values)
        mean_diff = np.mean(np.abs(differences))
        
        # Normalize to [0,1] where 1 means very clear hierarchy
        return min(mean_diff / 50.0, 1.0)  # 50 is max possible difference
    
    def _calculate_role_differentiation(self, history: List[Dict]) -> float:
        """Calculate role differentiation from simulation history."""
        # Use final state
        final_state = history[-1]
        
        # Calculate correlation between leader and follower identities
        li_values = [agent["leader_identity"] for agent in final_state]
        fi_values = [agent["follower_identity"] for agent in final_state]
        
        correlation = np.corrcoef(li_values, fi_values)[0,1]
        
        # Transform to [0,1] where 1 means strong differentiation
        return (1.0 - correlation) / 2.0 