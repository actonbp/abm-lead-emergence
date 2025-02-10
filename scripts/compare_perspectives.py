#!/usr/bin/env python3

"""
Compare different perspectives on leadership emergence:
- Base Model (Schema-based)
- Interactionist Model (Schema + Identity)
- Cognitive Model (Learning + Adaptation)
- Identity Model (Social Identity)
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
from src.models.perspectives import InteractionistModel, InteractionistParameters
from src.models.perspectives.cognitive import CognitiveModel, CognitiveParameters
from src.models.perspectives.identity import IdentityModel, IdentityParameters
from src.models.null_model import NullModel
from scripts.test_single_team import calculate_leadership_emergence_icc

def calculate_gini_coefficient(state):
    """Calculate Gini coefficient of leadership perceptions as a measure of hierarchy strength."""
    n_agents = len(state['agents'])
    perception_matrix = np.zeros((n_agents, n_agents))
    
    # Fill perception matrix
    for i, agent in enumerate(state['agents']):
        for j_str, value in agent['leadership_perceptions'].items():
            j = int(j_str)
            perception_matrix[i,j] = value
    
    # Calculate mean incoming leadership for each agent
    mean_incoming = np.mean(perception_matrix, axis=0)
    
    # Calculate Gini coefficient
    sorted_scores = np.sort(mean_incoming)
    n = len(sorted_scores)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * sorted_scores)) / (n * np.sum(sorted_scores))

def calculate_role_differentiation(state):
    """Calculate role differentiation based on average separation between leader and follower identities.
    
    This metric considers:
    1. Average leader-follower identity gap for each agent
    2. Consistency of role separation across agents
    3. Overall role clarity in the group
    """
    n_agents = len(state['agents'])
    
    # Get all identities
    leader_identities = np.array([agent['leader_identity'] for agent in state['agents']])
    follower_identities = np.array([agent['follower_identity'] for agent in state['agents']])
    
    # Calculate mean leadership perceptions for each agent
    perception_matrix = np.zeros((n_agents, n_agents))
    for i, agent in enumerate(state['agents']):
        for j_str, value in agent['leadership_perceptions'].items():
            j = int(j_str)
            perception_matrix[i,j] = value
    mean_received_leadership = np.mean(perception_matrix, axis=0)
    
    # Weight the identity gaps by received leadership
    normalized_weights = mean_received_leadership / np.sum(mean_received_leadership)
    identity_gaps = leader_identities - follower_identities
    
    # Calculate weighted average gap
    weighted_gap = np.sum(identity_gaps * normalized_weights)
    
    # Scale to [0,1] range
    return np.clip(weighted_gap / 100.0, 0, 1)

def calculate_perception_stability(current_state, previous_state):
    """Calculate stability in leadership rankings between time steps."""
    if previous_state is None:
        return 0.0
        
    n_agents = len(current_state['agents'])
    current_matrix = np.zeros((n_agents, n_agents))
    previous_matrix = np.zeros((n_agents, n_agents))
    
    # Fill matrices
    for i, agent in enumerate(current_state['agents']):
        for j_str, value in agent['leadership_perceptions'].items():
            j = int(j_str)
            current_matrix[i,j] = value
            
    for i, agent in enumerate(previous_state['agents']):
        for j_str, value in agent['leadership_perceptions'].items():
            j = int(j_str)
            previous_matrix[i,j] = value
    
    # Get rankings
    current_ranks = np.argsort(np.mean(current_matrix, axis=0))
    previous_ranks = np.argsort(np.mean(previous_matrix, axis=0))
    
    # Calculate rank correlation
    return np.corrcoef(current_ranks, previous_ranks)[0,1]

def calculate_network_centralization(state):
    """Calculate network centralization as a measure of hierarchical structure."""
    n_agents = len(state['agents'])
    perception_matrix = np.zeros((n_agents, n_agents))
    
    # Fill perception matrix
    for i, agent in enumerate(state['agents']):
        for j_str, value in agent['leadership_perceptions'].items():
            j = int(j_str)
            perception_matrix[i,j] = value
    
    # Calculate normalized centrality scores
    mean_incoming = np.mean(perception_matrix, axis=0)
    max_centrality = np.max(mean_incoming)
    centrality_sum = np.sum(max_centrality - mean_incoming)
    
    # Normalize by theoretical maximum
    max_possible = (n_agents - 1) * (100 - np.mean(mean_incoming))
    return centrality_sum / max_possible if max_possible > 0 else 0

def calculate_leadership_entropy(state, n_bins=10):
    """Calculate entropy of leadership perceptions to measure structure vs disorder.
    
    Lower entropy indicates more structured leadership (clear leaders/followers).
    Higher entropy indicates more distributed/ambiguous leadership.
    
    Args:
        state: Current model state
        n_bins: Number of bins for histogram (default=10)
        
    Returns:
        float: Normalized entropy value between 0 and 1
    """
    n_agents = len(state['agents'])
    perception_matrix = np.zeros((n_agents, n_agents))
    
    # Fill perception matrix
    for i, agent in enumerate(state['agents']):
        for j_str, value in agent['leadership_perceptions'].items():
            j = int(j_str)
            perception_matrix[i,j] = value
    
    # Get mean incoming leadership for each agent
    mean_incoming = np.mean(perception_matrix, axis=0)
    
    # Create histogram of leadership perceptions
    hist, _ = np.histogram(mean_incoming, bins=n_bins, range=(0, 100))
    
    # Calculate probability distribution
    prob_dist = hist / np.sum(hist)
    
    # Remove zero probabilities (log(0) is undefined)
    prob_dist = prob_dist[prob_dist > 0]
    
    # Calculate Shannon entropy
    entropy = -np.sum(prob_dist * np.log2(prob_dist))
    
    # Normalize by maximum possible entropy (uniform distribution)
    max_entropy = np.log2(n_bins)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # Invert so 1 means structured (low entropy) and 0 means disordered (high entropy)
    return 1 - normalized_entropy

def calculate_rank_volatility(current_state, previous_state):
    """Calculate how much leadership rankings and perceptions change between timesteps.
    
    Volatility represents instability in the leadership structure:
    - High values: Leadership roles still being negotiated (early emergence)
    - Medium values: Leadership structure starting to stabilize (mid emergence)
    - Low values: Stable leadership hierarchy (late emergence)
    
    The metric combines:
    1. Rank volatility: How much leadership rankings change
    2. Perception volatility: How much leadership perceptions change
    3. Role clarity: Whether changes maintain or disrupt hierarchy
    """
    if previous_state is None:
        return 1.0  # Maximum volatility at start
        
    n_agents = len(current_state['agents'])
    current_matrix = np.zeros((n_agents, n_agents))
    previous_matrix = np.zeros((n_agents, n_agents))
    
    # Fill matrices
    for i, agent in enumerate(current_state['agents']):
        for j_str, value in agent['leadership_perceptions'].items():
            j = int(j_str)
            current_matrix[i,j] = value
            
    for i, agent in enumerate(previous_state['agents']):
        for j_str, value in agent['leadership_perceptions'].items():
            j = int(j_str)
            previous_matrix[i,j] = value
    
    # Get mean received leadership
    current_leadership = np.mean(current_matrix, axis=0)
    previous_leadership = np.mean(previous_matrix, axis=0)
    
    # Calculate rank volatility
    current_ranks = np.argsort(np.argsort(current_leadership))
    previous_ranks = np.argsort(np.argsort(previous_leadership))
    rank_changes = np.abs(current_ranks - previous_ranks)
    rank_volatility = np.mean(rank_changes) / (n_agents - 1)  # Normalize by max possible rank change
    
    # Calculate perception volatility
    perception_changes = np.abs(current_leadership - previous_leadership)
    perception_volatility = np.mean(perception_changes) / 100.0  # Normalize to [0,1]
    
    # Calculate role clarity
    # High if changes maintain relative order (e.g., leaders stay leaders)
    # Low if changes disrupt hierarchy (e.g., leaders become followers)
    current_top = current_ranks >= n_agents/2  # Top half of ranks
    previous_top = previous_ranks >= n_agents/2
    role_disruption = np.mean(current_top != previous_top)  # How many crossed the median
    
    # Combine components with weights
    # More weight on rank_volatility and role_disruption as they better indicate structural changes
    volatility = (0.4 * rank_volatility + 
                 0.2 * perception_volatility + 
                 0.4 * role_disruption)
    
    return volatility

def calculate_perception_consensus(state):
    """Calculate degree of consensus in leadership perceptions.
    
    This metric combines two factors:
    1. Agreement: How much agents agree on each other's leadership level
    2. Differentiation: Whether perceptions converge on clear leader/follower roles
    
    High values indicate both agreement AND clear role differentiation.
    Low values indicate either disagreement OR ambiguous roles.
    """
    n_agents = len(state['agents'])
    perception_matrix = np.zeros((n_agents, n_agents))
    
    # Fill perception matrix
    for i, agent in enumerate(state['agents']):
        for j_str, value in agent['leadership_perceptions'].items():
            j = int(j_str)
            perception_matrix[i,j] = value
    
    # Calculate agreement component (inverse of perception variance)
    perception_std = np.std(perception_matrix, axis=0)
    agreement = 1 - (np.mean(perception_std) / np.sqrt(np.var([0, 100])))
    
    # Calculate differentiation component (spread of mean perceptions)
    mean_perceptions = np.mean(perception_matrix, axis=0)
    perception_range = np.max(mean_perceptions) - np.min(mean_perceptions)
    differentiation = perception_range / 100.0  # Normalize to [0,1]
    
    # Combine both components - multiply so both need to be high for high consensus
    consensus = agreement * differentiation
    
    return consensus

def create_comparison_plot(metrics):
    """Create multi-panel plot comparing different models."""
    # Set up the figure
    fig = plt.figure(figsize=(12, 10))
    
    # Define colors and labels for models
    colors = {
        'Base': '#1f77b4',     # Blue
        'Interactionist': '#9467bd',  # Purple
        'Cognitive': '#2ca02c',   # Green
        'Identity': '#d62728',    # Red
        'Null': '#7f7f7f'        # Gray
    }
    
    labels = {
        'Base': 'Base Model',
        'Interactionist': 'Interactionist Model',
        'Cognitive': 'Cognitive Model',
        'Identity': 'Identity Model',
        'Null': 'Null Model'
    }
    
    # Panel layout
    panels = {
        'Leadership Structure (ICC)': {'metric': 'ICC', 'pos': 221},
        'Hierarchy Strength (Gini)': {'metric': 'Gini', 'pos': 222},
        'Leadership Structure (Entropy)': {'metric': 'Entropy', 'pos': 223},
        'Leadership Rank Volatility': {'metric': 'Volatility', 'pos': 224}
    }
    
    # Create each panel
    for title, info in panels.items():
        ax = fig.add_subplot(info['pos'])
        metric = info['metric']
        
        for model in ['Base', 'Interactionist', 'Cognitive', 'Identity', 'Null']:
            # Calculate mean and confidence intervals
            data = np.array(metrics[metric][model])
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            ci = 1.96 * std / np.sqrt(len(data))
            
            # Time points
            x = np.arange(len(mean))
            
            # Plot individual runs with low alpha
            for run in data:
                ax.plot(x, run, color=colors[model], alpha=0.05)
            
            # Plot mean line
            ax.plot(x, mean, color=colors[model], label=labels[model], linewidth=2)
            
            # Add confidence interval
            ax.fill_between(x, mean-ci, mean+ci, color=colors[model], alpha=0.2)
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Only add legend to first panel
        if title == 'Leadership Structure (ICC)':
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout and add title
    plt.tight_layout()
    
    # Add overall title
    fig.suptitle('Leadership Emergence Model Comparison', y=1.02, fontsize=14)
    
    # Add description
    plt.figtext(0.5, 0.02, 
                'Comparing base model (claims/grants), interactionist model (identity development),\n' +
                'cognitive model (learning/adaptation), identity model (social identity), and null model (random interactions)',
                ha='center', fontsize=9, alpha=0.7)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = Path("outputs/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f"leadership_emergence_analysis_{timestamp}.png", 
                bbox_inches='tight', dpi=300)
    plt.close()

def run_model_comparison(n_steps=50, n_runs=30, print_interval=10):
    """Run and compare different leadership emergence models."""
    # Set up consistent parameters across models
    base_params = {
        'n_agents': 6,
        'schema_dimensions': 2,
        'match_algorithm': 'average',
        'match_threshold': 0.5,
        'success_boost': 5.0,
        'failure_penalty': 3.0,
        'base_claim_probability': 0.7,
        'identity_inertia': 0.2,
        
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
        'simultaneous_roles': True
    }
    
    print(f"Running {n_runs} simulations for each model...")
    
    # Storage for multiple metrics
    metrics = {
        'ICC': {model: [[] for _ in range(n_runs)] for model in ['Base', 'Interactionist', 'Cognitive', 'Identity', 'Null']},
        'Gini': {model: [[] for _ in range(n_runs)] for model in ['Base', 'Interactionist', 'Cognitive', 'Identity', 'Null']},
        'Entropy': {model: [[] for _ in range(n_runs)] for model in ['Base', 'Interactionist', 'Cognitive', 'Identity', 'Null']},
        'Volatility': {model: [[] for _ in range(n_runs)] for model in ['Base', 'Interactionist', 'Cognitive', 'Identity', 'Null']}
    }
    
    # Storage for ICC scores
    results = {model: [] for model in ['Base', 'Interactionist', 'Cognitive', 'Identity', 'Null']}
    
    for run in range(n_runs):
        # Set different random seed for each run
        run_seed = 42 + run
        base_params['random_seed'] = run_seed
        
        # Initialize models with perspective-specific parameters
        models = {
            'Base': BaseLeadershipModel(ModelParameters(**base_params)),
            'Interactionist': InteractionistModel(InteractionistParameters(
                **base_params,
                dyadic_interactions_before_switch=20,
                identity_update_rate=0.2,
                perception_update_rate=0.2
            )),
            'Cognitive': CognitiveModel(CognitiveParameters(
                **base_params,
                dyadic_interactions_before_switch=20,
                ilt_learning_rate=0.3
            )),
            'Identity': IdentityModel(IdentityParameters(
                **base_params,
                dyadic_interactions_before_switch=20,
                prototype_learning_rate=0.2,
                prototype_influence_weight=0.8
            )),
            'Null': NullModel(ModelParameters(**base_params))
        }
        
        # Storage for this run's ICC scores
        run_scores = {model: [] for model in ['Base', 'Interactionist', 'Cognitive', 'Identity', 'Null']}
        
        # Get initial states
        states = {model: models[model].get_state() for model in models}
        
        # Store initial metrics
        for model in ['Base', 'Interactionist', 'Cognitive', 'Identity', 'Null']:
            icc = calculate_leadership_emergence_icc(states[model])
            run_scores[model].append(icc)
            metrics['ICC'][model][run].append(icc)
            metrics['Gini'][model][run].append(calculate_gini_coefficient(states[model]))
            metrics['Entropy'][model][run].append(calculate_leadership_entropy(states[model]))
            metrics['Volatility'][model][run].append(0)  # No volatility for first step
        
        # Track previous states for volatility calculation
        previous_states = states.copy()
        
        # Run simulation steps
        for step in range(n_steps):
            # Base model
            state = models['Base'].step()
            icc = calculate_leadership_emergence_icc(state)
            run_scores['Base'].append(icc)
            metrics['ICC']['Base'][run].append(icc)
            metrics['Gini']['Base'][run].append(calculate_gini_coefficient(state))
            metrics['Entropy']['Base'][run].append(calculate_leadership_entropy(state))
            metrics['Volatility']['Base'][run].append(calculate_rank_volatility(state, previous_states['Base']))
            previous_states['Base'] = state
            
            # Interactionist model
            state = models['Interactionist'].step()
            icc = calculate_leadership_emergence_icc(state)
            run_scores['Interactionist'].append(icc)
            metrics['ICC']['Interactionist'][run].append(icc)
            metrics['Gini']['Interactionist'][run].append(calculate_gini_coefficient(state))
            metrics['Entropy']['Interactionist'][run].append(calculate_leadership_entropy(state))
            metrics['Volatility']['Interactionist'][run].append(calculate_rank_volatility(state, previous_states['Interactionist']))
            previous_states['Interactionist'] = state
            
            # Cognitive model
            state = models['Cognitive'].step()
            icc = calculate_leadership_emergence_icc(state)
            run_scores['Cognitive'].append(icc)
            metrics['ICC']['Cognitive'][run].append(icc)
            metrics['Gini']['Cognitive'][run].append(calculate_gini_coefficient(state))
            metrics['Entropy']['Cognitive'][run].append(calculate_leadership_entropy(state))
            metrics['Volatility']['Cognitive'][run].append(calculate_rank_volatility(state, previous_states['Cognitive']))
            previous_states['Cognitive'] = state
            
            # Identity model
            state = models['Identity'].step()
            icc = calculate_leadership_emergence_icc(state)
            run_scores['Identity'].append(icc)
            metrics['ICC']['Identity'][run].append(icc)
            metrics['Gini']['Identity'][run].append(calculate_gini_coefficient(state))
            metrics['Entropy']['Identity'][run].append(calculate_leadership_entropy(state))
            metrics['Volatility']['Identity'][run].append(calculate_rank_volatility(state, previous_states['Identity']))
            previous_states['Identity'] = state
            
            # Null model
            state = models['Null'].step()
            icc = calculate_leadership_emergence_icc(state)
            run_scores['Null'].append(icc)
            metrics['ICC']['Null'][run].append(icc)
            metrics['Gini']['Null'][run].append(calculate_gini_coefficient(state))
            metrics['Entropy']['Null'][run].append(calculate_leadership_entropy(state))
            metrics['Volatility']['Null'][run].append(calculate_rank_volatility(state, previous_states['Null']))
            previous_states['Null'] = state
        
        # Store final ICC scores for this run
        for model in results:
            results[model].append(run_scores[model][-1])
        
        if (run + 1) % print_interval == 0:
            print(f"Completed {run + 1} runs")
    
    # Print final results
    print("\nFinal model comparison results:")
    for model in results:
        mean = np.mean(results[model])
        std = np.std(results[model])
        ci = 1.96 * std / np.sqrt(n_runs)  # 95% confidence interval
        print(f"{model}: {mean:.3f} ± {std:.3f} (95% CI: [{mean-ci:.3f}, {mean+ci:.3f}])")
    
    # Create visualization
    create_comparison_plot(metrics)
    
    return results, metrics

if __name__ == "__main__":
    run_model_comparison() 