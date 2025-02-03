#!/usr/bin/env python3

"""
Compare different perspectives on leadership emergence:
- Base Model (Schema-based)
- Interactionist Model (Schema + Identity)
- Cognitive Model (Learning + Adaptation)
- Null Model (Random)

Creates publication-quality visualizations with multiple runs.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.base_model import BaseLeadershipModel, ModelParameters
from src.models.perspectives.interactionist import InteractionistModel
from src.models.perspectives.cognitive import CognitiveModel, CognitiveParameters
from src.models.null_model import NullModel
from scripts.test_single_team import calculate_leadership_emergence_icc

def run_model_comparison(n_steps=50, n_runs=30, print_interval=10):
    """Run multiple simulations comparing different leadership perspectives.
    
    Args:
        n_steps (int): Number of steps per simulation
        n_runs (int): Number of simulation runs for confidence
        print_interval (int): How often to print progress
    """
    # Set up consistent parameters across models
    base_params = {
        'n_agents': 6,
        'schema_dimensions': 2,
        'match_algorithm': 'minimum',
        'match_threshold': 0.41,
        'success_boost': 25.0,
        'failure_penalty': 20.0,
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
    
    # Additional parameters for interactionist model
    interactionist_params = base_params.copy()
    interactionist_params.update({
        'dyadic_interactions_before_switch': 10,
        'identity_update_rate': 0.6,
        'perception_update_rate': 0.6,
        'claim_success_boost': 12.0,
        'grant_success_boost': 12.0,
        'rejection_penalty': 8.0,
        'passivity_penalty': 4.0
    })
    
    # Additional parameters for cognitive model
    cognitive_params = base_params.copy()
    cognitive_params.update({
        'n_agents': 6,
        'schema_dimensions': 2,
        'match_algorithm': 'average',
        'match_threshold': 0.47,
        'success_boost': 7.57,
        'failure_penalty': 4.89,
        'ilt_learning_rate': 0.26,
        'observation_weight': 1.91,
        'memory_decay': 0.074,
        'max_memory': 11
    })
    
    # Storage for multiple runs
    results = {
        'Base': [],
        'Interactionist': [],
        'Cognitive': [],
        'Null': []
    }
    time_points = list(range(5, n_steps))
    
    print(f"Running {n_runs} simulations for each model...")
    for run in range(n_runs):
        # Set different random seed for each run
        run_seed = 42 + run
        base_params['random_seed'] = run_seed
        interactionist_params['random_seed'] = run_seed
        cognitive_params['random_seed'] = run_seed
        
        # Initialize models
        base_model = BaseLeadershipModel(ModelParameters(**base_params))
        interactionist_model = InteractionistModel(interactionist_params)
        cognitive_model = CognitiveModel(CognitiveParameters(**cognitive_params))
        null_model = NullModel(base_params)
        
        # Storage for this run
        run_scores = {
            'Base': [],
            'Interactionist': [],
            'Cognitive': [],
            'Null': []
        }
        
        # Run simulations
        for step in range(n_steps):
            if step >= 5:
                # Base model
                state = base_model.step()
                run_scores['Base'].append(calculate_leadership_emergence_icc(state))
                
                # Interactionist model
                state = interactionist_model.step()
                run_scores['Interactionist'].append(calculate_leadership_emergence_icc(state))
                
                # Cognitive model
                state = cognitive_model.step()
                run_scores['Cognitive'].append(calculate_leadership_emergence_icc(state))
                
                # Null model
                state = null_model.step()
                run_scores['Null'].append(calculate_leadership_emergence_icc(state))
            else:
                base_model.step()
                interactionist_model.step()
                cognitive_model.step()
                null_model.step()
        
        # Store run results
        for model in results:
            results[model].append(run_scores[model])
        
        if (run + 1) % 5 == 0:
            print(f"Completed {run + 1}/{n_runs} runs")
    
    # Convert to numpy arrays
    for model in results:
        results[model] = np.array(results[model])
    
    # Calculate statistics
    stats = {}
    for model in results:
        stats[model] = {
            'mean': np.mean(results[model], axis=0),
            'std': np.std(results[model], axis=0),
            'ci_lower': np.percentile(results[model], 2.5, axis=0),
            'ci_upper': np.percentile(results[model], 97.5, axis=0)
        }
    
    # Create visualization
    create_comparison_plot(time_points, results, stats)
    
    # Print final statistics
    print("\nFinal Model Comparison (Mean ± Std):")
    for model in results:
        final_scores = results[model][:,-1]
        print(f"{model:13s}: {np.mean(final_scores):.3f} ± {np.std(final_scores):.3f}")
        print(f"{'':13s}  95% CI: [{np.percentile(final_scores, 2.5):.3f}, {np.percentile(final_scores, 97.5):.3f}]")

def create_comparison_plot(time_points, results, stats):
    """Create publication-quality comparison plot."""
    plt.style.use('seaborn-v0_8-paper')
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {
        'Base': '#1f77b4',       # Blue
        'Interactionist': '#9467bd',  # Purple
        'Cognitive': '#2ca02c',   # Green
        'Null': '#7f7f7f'        # Gray
    }
    
    model_names = {
        'Base': 'Base Model',
        'Interactionist': 'Interactionist Model',
        'Cognitive': 'Cognitive Model',
        'Null': 'Null Model'
    }
    
    # Plot individual runs with very light transparency
    for model, runs in results.items():
        for run in runs:
            ax.plot(time_points, run, color=colors[model], alpha=0.03, linewidth=0.3)
    
    # Plot means with confidence intervals
    for model in results:
        mean = stats[model]['mean']
        ci_lower = stats[model]['ci_lower']
        ci_upper = stats[model]['ci_upper']
        
        ax.fill_between(time_points, ci_lower, ci_upper, color=colors[model], alpha=0.1,
                       label=f'{model_names[model]} (95% CI)')
        
        if model == 'Null':
            linestyle = '--'
            alpha = 0.8
        else:
            linestyle = '-'
            alpha = 1.0
        
        ax.plot(time_points, mean, color=colors[model], linewidth=2.5, 
                label=f'{model_names[model]}', alpha=alpha, linestyle=linestyle)
    
    # Customize appearance
    ax.set_xlabel('Time Step', fontsize=11, labelpad=10)
    ax.set_ylabel('Leadership Structure Emergence (ICC)', fontsize=11, labelpad=10)
    ax.set_title('Emergence of Leadership Structure', fontsize=13, pad=15)
    
    # Add subtitle
    plt.figtext(0.5, 0.02, 
                'Comparing base model (claims/grants), interactionist model (identity development),\n' +
                'cognitive model (learning/adaptation), and null model (random interactions)', 
                ha='center', fontsize=9, alpha=0.7)
    
    # Customize grid and legend
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    legend = ax.legend(fontsize=9, loc='center right', framealpha=0.9,
                      edgecolor='none', bbox_to_anchor=(1.15, 0.5))
    
    # Set y-axis limits and annotations
    ax.set_ylim(-0.05, 1.05)
    ax.text(0.02, 0.98, 'Strong Structure', transform=ax.transAxes,
            fontsize=9, alpha=0.6, verticalalignment='top')
    ax.text(0.02, 0.02, 'Weak Structure', transform=ax.transAxes,
            fontsize=9, alpha=0.6)
    
    # Adjust layout and save
    plt.subplots_adjust(right=0.82, bottom=0.15)
    plots_dir = Path('outputs/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = plots_dir / f'perspective_comparison_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\nPlot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    run_model_comparison() 