#!/usr/bin/env python3

"""Visualize the dynamics of the base leadership emergence model."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.base_model import BaseLeadershipModel, ModelParameters
from scripts.run_parameter_sweep import calculate_hierarchy_strength, calculate_perception_agreement, calculate_leadership_differentiation

def run_and_visualize_base_model(n_runs=20, n_steps=100):
    """Run and visualize the base model with best parameters."""
    # Best parameters from sweep
    best_params = {
        'n_agents': 6,
        'schema_dimensions': 2,
        'match_algorithm': 'minimum',
        'match_threshold': 0.41,
        'success_boost': 7.94,
        'failure_penalty': 5.64,
        
        # Fixed parameters
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
    }
    
    # Store results from all runs
    all_entropies = []
    all_agreements = []
    all_differentiations = []
    all_perceptions = []
    
    print(f"Running {n_runs} simulations...")
    
    # Run multiple simulations
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}")
        
        # Create model with best parameters
        model_params = ModelParameters(**best_params)
        model = BaseLeadershipModel(model_params)
        
        # Run simulation
        run_entropies = []
        run_agreements = []
        run_differentiations = []
        run_perceptions = []
        
        for step in range(n_steps):
            state = model.step()
            entropy = calculate_hierarchy_strength(state)
            agreement = calculate_perception_agreement(state)
            differentiation = calculate_leadership_differentiation(state)
            
            run_entropies.append(entropy)
            run_agreements.append(agreement)
            run_differentiations.append(differentiation)
            
            # Store perception matrix at key points
            if step in [0, n_steps//2, n_steps-1]:
                perception_matrix = np.zeros((best_params['n_agents'], best_params['n_agents']))
                for i, agent in enumerate(state['agents']):
                    for j_str, value in agent['leadership_perceptions'].items():
                        j = int(j_str)
                        perception_matrix[i,j] = value / 100.0  # Scale to [0,1]
                run_perceptions.append((step, perception_matrix))
        
        all_entropies.append(run_entropies)
        all_agreements.append(run_agreements)
        all_differentiations.append(run_differentiations)
        all_perceptions.append(run_perceptions)
    
    # Convert to numpy arrays
    all_entropies = np.array(all_entropies)
    all_agreements = np.array(all_agreements)
    all_differentiations = np.array(all_differentiations)
    
    # Calculate means and standard deviations
    mean_entropy = np.mean(all_entropies, axis=0)
    std_entropy = np.std(all_entropies, axis=0)
    mean_agreement = np.mean(all_agreements, axis=0)
    std_agreement = np.std(all_agreements, axis=0)
    mean_differentiation = np.mean(all_differentiations, axis=0)
    std_differentiation = np.std(all_differentiations, axis=0)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 20))
    
    # 1. Entropy reduction over time
    ax1 = plt.subplot(411)
    time_points = np.arange(n_steps)
    
    # Plot all runs in background
    for run_entropy in all_entropies:
        ax1.plot(time_points, run_entropy, 'gray', alpha=0.2, linewidth=1)
    
    # Plot mean and confidence interval
    ax1.plot(time_points, mean_entropy, 'b-', label='Mean Entropy', linewidth=2)
    ax1.fill_between(time_points, 
                     mean_entropy - std_entropy,
                     mean_entropy + std_entropy,
                     alpha=0.2)
    ax1.set_title('Hierarchy Entropy Over Time (Base Model)')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Entropy (lower = more hierarchical)')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Agreement increase over time
    ax2 = plt.subplot(412)
    
    # Plot all runs in background
    for run_agreement in all_agreements:
        ax2.plot(time_points, run_agreement, 'gray', alpha=0.2, linewidth=1)
    
    # Plot mean and confidence interval
    ax2.plot(time_points, mean_agreement, 'g-', label='Mean Agreement', linewidth=2)
    ax2.fill_between(time_points,
                     mean_agreement - std_agreement,
                     mean_agreement + std_agreement,
                     alpha=0.2)
    ax2.set_title('Perception Agreement Over Time (Base Model)')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Agreement Score (higher = more consensus)')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Leadership differentiation over time
    ax3 = plt.subplot(413)
    
    # Plot all runs in background
    for run_diff in all_differentiations:
        ax3.plot(time_points, run_diff, 'gray', alpha=0.2, linewidth=1)
    
    # Plot mean and confidence interval
    ax3.plot(time_points, mean_differentiation, 'm-', label='Mean Differentiation', linewidth=2)
    ax3.fill_between(time_points,
                     mean_differentiation - std_differentiation,
                     mean_differentiation + std_differentiation,
                     alpha=0.2)
    ax3.set_title('Leadership Differentiation Over Time (Base Model)')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Differentiation Score (higher = clearer roles)')
    ax3.grid(True)
    ax3.legend()
    
    # 4. Example perception matrices at key points
    ax4 = plt.subplot(414)
    example_run = all_perceptions[0]  # Use first run as example
    n_timepoints = len(example_run)
    
    for idx, (step, matrix) in enumerate(example_run):
        plt.subplot(4, n_timepoints, 3*n_timepoints + idx + 1)
        plt.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar()
        if step == 0:
            plt.title('Initial Perceptions')
        elif step == n_steps//2:
            plt.title('Mid-point')
        else:
            plt.title('Final Perceptions')
        plt.xlabel('Target Agent')
        plt.ylabel('Perceiving Agent')
    
    # Add parameter details in text
    param_text = (
        f"Best Parameters:\n"
        f"N={best_params['n_agents']}, "
        f"Dims={best_params['schema_dimensions']}\n"
        f"Success={best_params['success_boost']:.2f}, "
        f"Failure={best_params['failure_penalty']:.2f}\n"
        f"Match={best_params['match_threshold']:.2f}, "
        f"Algorithm={best_params['match_algorithm']}"
    )
    fig.text(0.02, 0.02, param_text, fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'outputs/plots/base_model_metrics_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {filename}")
    plt.close()

if __name__ == "__main__":
    run_and_visualize_base_model() 