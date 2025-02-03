#!/usr/bin/env python3

"""Run parameter sweep for leadership emergence models."""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import json
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.base_model import BaseLeadershipModel, ModelParameters
from src.models.perspectives import InteractionistModel, InteractionistParameters
from src.models.perspectives.cognitive import CognitiveModel, CognitiveParameters

def convert_to_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

def calculate_hierarchy_strength(state):
    """Calculate entropy of leadership structure.
    
    Lower entropy indicates more organized hierarchy.
    Uses Shannon entropy on the distribution of leadership values.
    """
    n_agents = len(state['agents'])
    perception_matrix = np.zeros((n_agents, n_agents))
    
    # Fill perception matrix
    for i, agent in enumerate(state['agents']):
        for j_str, value in agent['leadership_perceptions'].items():
            j = int(j_str)
            perception_matrix[i,j] = value / 100.0  # Scale to [0,1]
    
    # Calculate mean incoming leadership for each agent
    incoming_leadership = np.mean(perception_matrix, axis=0)
    
    # Calculate entropy using histogram of leadership values
    hist, _ = np.histogram(incoming_leadership, bins=5, range=(0,1), density=True)
    hist = hist / np.sum(hist)  # Normalize to get probabilities
    
    # Remove zero probabilities before calculating entropy
    hist = hist[hist > 0]
    
    # Calculate Shannon entropy
    entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
    
    # Normalize by maximum possible entropy (uniform distribution)
    max_entropy = np.log2(5)  # 5 bins
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return normalized_entropy  # Higher entropy = less hierarchical

def calculate_perception_agreement(state):
    """Calculate agreement in leadership perceptions across agents.
    
    Higher values indicate more consensus in perceptions.
    Uses standard deviation of perceptions for each target,
    then averages across targets. Lower std dev = higher agreement.
    """
    n_agents = len(state['agents'])
    perception_matrix = np.zeros((n_agents, n_agents))
    
    # Fill perception matrix
    for i, agent in enumerate(state['agents']):
        for j_str, value in agent['leadership_perceptions'].items():
            j = int(j_str)
            perception_matrix[i,j] = value / 100.0  # Scale to [0,1]
    
    # For each target, calculate std dev of others' perceptions
    agreement_scores = []
    for j in range(n_agents):
        # Get all perceptions of agent j (excluding self-perception)
        perceptions = [perception_matrix[i,j] for i in range(n_agents) if i != j]
        if perceptions:
            # Convert std dev to agreement score (1 - normalized std dev)
            std_dev = np.std(perceptions)
            # Max possible std dev for values in [0,1] is ~0.5
            agreement = 1.0 - (std_dev * 2.0)  # Scale to [0,1]
            agreement_scores.append(agreement)
    
    # Average agreement across all targets
    return np.mean(agreement_scores) if agreement_scores else 0.0

def calculate_leadership_differentiation(state):
    """Calculate degree of leadership role differentiation.
    
    Higher values indicate clearer distinction between leaders and followers.
    Uses standard deviation of mean incoming leadership perceptions.
    """
    n_agents = len(state['agents'])
    perception_matrix = np.zeros((n_agents, n_agents))
    
    # Fill perception matrix
    for i, agent in enumerate(state['agents']):
        for j_str, value in agent['leadership_perceptions'].items():
            j = int(j_str)
            perception_matrix[i,j] = value / 100.0  # Scale to [0,1]
    
    # Calculate mean incoming leadership for each agent
    mean_incoming = np.mean(perception_matrix, axis=0)
    
    # Calculate standard deviation of mean incoming leadership
    differentiation = np.std(mean_incoming)
    
    return differentiation

def run_parameter_sweep(model_type='base', n_calls=50):
    """Run parameter sweep using Bayesian optimization.
    
    Args:
        model_type: 'base', 'interactionist', or 'cognitive'
        n_calls: Number of optimization iterations
    """
    # Define the search space based on model type
    if model_type == 'base':
        space = [
            Integer(5, 10, name='n_agents'),
            Integer(2, 3, name='schema_dimensions'),
            Categorical(['average', 'minimum'], name='match_algorithm'),
            Real(0.4, 0.7, name='match_threshold'),
            Real(3.0, 8.0, name='success_boost'),
            Real(2.0, 6.0, name='failure_penalty')
        ]
        ModelClass = BaseLeadershipModel
        ParamsClass = ModelParameters
    elif model_type == 'interactionist':
        space = [
            Integer(5, 10, name='n_agents'),
            Integer(2, 3, name='schema_dimensions'),
            Categorical(['average', 'minimum'], name='match_algorithm'),
            Real(0.4, 0.7, name='match_threshold'),
            Real(3.0, 8.0, name='success_boost'),
            Real(2.0, 6.0, name='failure_penalty'),
            Integer(10, 30, name='switch_step'),
            Integer(3, 10, name='identity_transition_steps')
        ]
        ModelClass = InteractionistModel
        ParamsClass = InteractionistParameters
    elif model_type == 'cognitive':  # Add cognitive model
        space = [
            Integer(5, 10, name='n_agents'),
            Integer(2, 3, name='schema_dimensions'),
            Categorical(['average', 'minimum'], name='match_algorithm'),
            Real(0.4, 0.7, name='match_threshold'),
            Real(3.0, 8.0, name='success_boost'),
            Real(2.0, 6.0, name='failure_penalty'),
            Real(0.1, 0.5, name='ilt_learning_rate'),
            Real(0.5, 2.0, name='observation_weight'),
            Real(0.0, 0.3, name='memory_decay'),
            Integer(5, 20, name='max_memory')
        ]
        ModelClass = CognitiveModel
        ParamsClass = CognitiveParameters
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    def objective(x):
        """Objective function for optimization."""
        # Convert parameters to dictionary
        params = dict(zip([dim.name for dim in space], x))
        
        # Add fixed parameters
        params.update({
            'characteristic_distribution': 'normal',
            'ilt_distribution': 'normal',
            'distribution_mean': 50.0,
            'distribution_std': 15.0,
            'interaction_selection': 'random',
            'schema_type': 'continuous',
            'dimension_weights': 'uniform',
            'grant_first': False,
            'allow_mutual_claims': True,
            'allow_self_loops': False,
            'simultaneous_roles': True,
            'base_claim_probability': 0.7,
            'identity_inertia': 0.2
        })
        
        try:
            # Create model parameters
            model_params = ParamsClass(**params)
            
            # Run simulation
            model = ModelClass(model_params)
            entropy_improvements = []
            agreement_improvements = []
            
            for _ in range(5):  # Run 5 times
                entropy_scores = []
                agreement_scores = []
                
                for _ in range(100):  # Run for 100 steps
                    state = model.step()
                    entropy = calculate_hierarchy_strength(state)
                    agreement = calculate_perception_agreement(state)
                    
                    entropy_scores.append(entropy)
                    agreement_scores.append(agreement)
                
                # Calculate improvements
                initial_entropy = np.mean(entropy_scores[:10])
                final_entropy = np.mean(entropy_scores[-10:])
                entropy_improvement = initial_entropy - final_entropy
                
                initial_agreement = np.mean(agreement_scores[:10])
                final_agreement = np.mean(agreement_scores[-10:])
                agreement_improvement = final_agreement - initial_agreement
                
                # Calculate trends
                time_points = np.arange(100)
                entropy_slope, _ = np.polyfit(time_points, entropy_scores, 1)
                agreement_slope, _ = np.polyfit(time_points, agreement_scores, 1)
                
                # Score this run
                entropy_score = entropy_improvement + (-entropy_slope * 100)  # Convert slope to total change
                agreement_score = agreement_improvement + (agreement_slope * 100)
                
                entropy_improvements.append(entropy_score)
                agreement_improvements.append(agreement_score)
            
            # Calculate final score
            mean_entropy_improvement = np.mean(entropy_improvements)
            mean_agreement_improvement = np.mean(agreement_improvements)
            
            # Weight agreement more heavily than entropy
            score = mean_entropy_improvement * 0.3 + mean_agreement_improvement * 0.7
            
            return -score  # Negative because we want to maximize
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 0.0
    
    print(f"Running Bayesian optimization with {n_calls} iterations...")
    
    # Run optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_random_starts=20,
        verbose=True
    )
    
    # Print best parameters
    print("\nBest parameters found:")
    for name, value in zip([dim.name for dim in space], result.x):
        print(f"{name}: {value}")
    print(f"\nBest score: {-result.fun}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/parameter_sweep/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best parameters
    with open(output_dir / "best_params.json", "w") as f:
        params = dict(zip([dim.name for dim in space], result.x))
        params = {k: convert_to_serializable(v) for k, v in params.items()}  # Convert numpy types
        json.dump(params, f, indent=2)
    
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run parameter sweep')
    parser.add_argument('--model', type=str, choices=['base', 'interactionist', 'cognitive'],
                      default='base', help='Model type to optimize')
    args = parser.parse_args()
    run_parameter_sweep(model_type=args.model) 