#!/usr/bin/env python3

"""Test script to examine leadership perception scores for a single team."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.base_model import BaseLeadershipModel, ModelParameters
from scripts.run_parameter_sweep import calculate_hierarchy_strength, calculate_perception_agreement
from src.models.perspectives.interactionist import InteractionistModel, InteractionistParameters

def calculate_gini(perceptions):
    """Calculate Gini coefficient of leadership perceptions."""
    # Sort perceptions in ascending order
    sorted_perceptions = np.sort(perceptions)
    n = len(sorted_perceptions)
    if n == 0:
        return 0
    # Calculate Lorenz curve
    cumsum = np.cumsum(sorted_perceptions)
    # Calculate Gini coefficient
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

def calculate_agreement_measures(state, debug=False):
    """Calculate multiple agreement measures."""
    n_agents = len(state['agents'])
    perception_matrix = np.zeros((n_agents, n_agents))
    
    # Fill perception matrix
    for i, agent in enumerate(state['agents']):
        for j_str, value in agent['leadership_perceptions'].items():
            j = int(j_str)
            perception_matrix[i,j] = value / 100.0  # Scale to [0,1]
    
    if debug:
        print("\nRaw Perception Matrix (scaled to [0,1]):")
        print(pd.DataFrame(perception_matrix).round(3))
    
    measures = {}
    
    # Standard deviation based agreement
    agreement_scores = []
    if debug:
        print("\nStandard Deviation Analysis:")
    for j in range(n_agents):
        perceptions = [perception_matrix[i,j] for i in range(n_agents) if i != j]
        if perceptions:
            std_dev = np.std(perceptions)
            agreement = 1.0 - (std_dev * 2.0)
            agreement_scores.append(agreement)
            if debug:
                print(f"Agent {j} as target:")
                print(f"  Perceptions: {[round(p,3) for p in perceptions]}")
                print(f"  Std Dev: {std_dev:.3f}")
                print(f"  Agreement Score: {agreement:.3f}")
    measures['std_agreement'] = np.mean(agreement_scores) if agreement_scores else 0.0
    
    # Gini coefficient based agreement
    gini_scores = []
    if debug:
        print("\nGini Coefficient Analysis:")
    for j in range(n_agents):
        perceptions = [perception_matrix[i,j] for i in range(n_agents) if i != j]
        if perceptions:
            gini = calculate_gini(perceptions)
            agreement = 1.0 - gini
            gini_scores.append(agreement)
            if debug:
                print(f"Agent {j} as target:")
                print(f"  Perceptions: {[round(p,3) for p in perceptions]}")
                print(f"  Gini: {gini:.3f}")
                print(f"  Agreement Score: {agreement:.3f}")
    measures['gini_agreement'] = np.mean(gini_scores) if gini_scores else 0.0
    
    # Coefficient of variation based agreement
    cv_scores = []
    if debug:
        print("\nCoefficient of Variation Analysis:")
    for j in range(n_agents):
        perceptions = [perception_matrix[i,j] for i in range(n_agents) if i != j]
        if perceptions and np.mean(perceptions) > 0:
            mean = np.mean(perceptions)
            std_dev = np.std(perceptions)
            cv = std_dev / mean
            agreement = 1.0 - min(cv, 1.0)
            cv_scores.append(agreement)
            if debug:
                print(f"Agent {j} as target:")
                print(f"  Perceptions: {[round(p,3) for p in perceptions]}")
                print(f"  Mean: {mean:.3f}")
                print(f"  Std Dev: {std_dev:.3f}")
                print(f"  CV: {cv:.3f}")
                print(f"  Agreement Score: {agreement:.3f}")
    measures['cv_agreement'] = np.mean(cv_scores) if cv_scores else 0.0
    
    if debug:
        print("\nFinal Agreement Scores:")
        for key, value in measures.items():
            print(f"  {key}: {value:.3f}")
    
    return measures

def calculate_hierarchy_entropy(state):
    """Calculate entropy of leadership structure."""
    n_agents = len(state['agents'])
    perception_matrix = np.zeros((n_agents, n_agents))
    
    # Fill perception matrix
    for i, agent in enumerate(state['agents']):
        for j_str, value in agent['leadership_perceptions'].items():
            j = int(j_str)
            perception_matrix[i,j] = value / 100.0
    
    # Calculate average incoming leadership scores
    incoming_leadership = np.mean(perception_matrix, axis=0)
    
    # Create histogram with fixed bins
    hist, _ = np.histogram(incoming_leadership, bins=5, range=(0,1), density=False)
    # Convert to probabilities
    hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
    # Remove zero bins
    hist = hist[hist > 0]
    
    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
    max_entropy = np.log2(5)  # Maximum possible entropy with 5 bins
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return normalized_entropy

def calculate_leadership_stratification(state):
    """Calculate metrics showing how stratified leadership is."""
    n_agents = len(state['agents'])
    perception_matrix = np.zeros((n_agents, n_agents))
    
    # Fill perception matrix
    for i, agent in enumerate(state['agents']):
        for j_str, value in agent['leadership_perceptions'].items():
            j = int(j_str)
            perception_matrix[i,j] = value
    
    # Calculate average incoming leadership for each agent
    mean_incoming = np.mean(perception_matrix, axis=0)
    
    # Sort scores from highest to lowest
    sorted_scores = np.sort(mean_incoming)[::-1]
    
    # Calculate metrics
    metrics = {
        'top_leader_score': sorted_scores[0],
        'bottom_follower_score': sorted_scores[-1],
        'leadership_range': sorted_scores[0] - sorted_scores[-1],
        'quartile_ratio': np.percentile(sorted_scores, 75) / np.percentile(sorted_scores, 25) if np.percentile(sorted_scores, 25) > 0 else 1.0,
        'stratification_index': np.std(sorted_scores) / np.mean(sorted_scores) if np.mean(sorted_scores) > 0 else 0.0
    }
    
    return metrics, sorted_scores

def calculate_leadership_emergence_icc(state, debug=False):
    """Calculate ICC-like measure of leadership emergence."""
    n_agents = len(state['agents'])
    perception_matrix = np.zeros((n_agents, n_agents))
    
    # Fill perception matrix
    for i, agent in enumerate(state['agents']):
        for j_str, value in agent['leadership_perceptions'].items():
            j = int(j_str)
            perception_matrix[i,j] = value
    
    # Calculate variances
    within_agent_vars = []
    agent_means = []
    
    if debug:
        print("\nICC Components Analysis:")
        print("Agent | Mean Perception | Within-Agent Variance")
        print("-" * 45)
    
    for j in range(n_agents):
        perceptions = perception_matrix[:,j][perception_matrix[:,j] != 0]  # Exclude self-perception
        if len(perceptions) > 1:
            within_var = np.var(perceptions)
            mean_perception = np.mean(perceptions)
            within_agent_vars.append(within_var)
            agent_means.append(mean_perception)
            
            if debug:
                print(f"{j:5d} | {mean_perception:14.2f} | {within_var:20.2f}")
    
    if not within_agent_vars or not agent_means:
        return 0.0
        
    # Average within-agent variance with minimum threshold
    avg_within_var = max(np.mean(within_agent_vars), 1.0)  # Minimum variance of 1.0
    
    # Between-agent variance (variance of mean perceptions)
    between_var = np.var(agent_means)
    
    if debug:
        print("\nVariance Components:")
        print(f"Average Within-Agent Variance: {avg_within_var:.2f}")
        print(f"Between-Agent Variance: {between_var:.2f}")
        print(f"Total Variance: {between_var + avg_within_var:.2f}")
    
    # Calculate ICC-like measure
    total_var = between_var + avg_within_var
    if total_var == 0:
        return 0.0
        
    emergence_score = between_var / total_var
    
    if debug:
        print(f"Emergence Score (ICC): {emergence_score:.3f}")
    
    return emergence_score

def print_perception_matrix(state, step):
    """Print the leadership perception matrix in a readable format."""
    n_agents = len(state['agents'])
    perception_matrix = np.zeros((n_agents, n_agents))
    
    # Fill perception matrix
    for i, agent in enumerate(state['agents']):
        for j_str, value in agent['leadership_perceptions'].items():
            j = int(j_str)
            perception_matrix[i,j] = value
    
    # Create DataFrame for prettier printing
    df = pd.DataFrame(perception_matrix)
    df.index = [f"Agent {i}" for i in range(n_agents)]
    df.columns = [f"Agent {i}" for i in range(n_agents)]
    
    print(f"\nStep {step} Leadership Perceptions:")
    print("Rows: perceiving agents, Columns: target agents")
    print(df.round(2))
    
    # Calculate and print summary statistics
    mean_incoming = np.mean(perception_matrix, axis=0)
    std_incoming = np.std(perception_matrix, axis=0)
    print("\nLeadership Score Summary:")
    print("Agent | Avg Score | Std Dev | Min Score | Max Score | Role")
    print("-" * 65)
    for i in range(n_agents):
        scores = perception_matrix[:, i][perception_matrix[:, i] != 0]  # Exclude self-perception
        min_score = np.min(scores) if len(scores) > 0 else 0.0
        max_score = np.max(scores) if len(scores) > 0 else 0.0
        role = "Strong Leader" if mean_incoming[i] > 70 else \
               "Moderate Leader" if mean_incoming[i] > 60 else \
               "Weak Leader" if mean_incoming[i] > 45 else "Follower"
        print(f"  {i:2d}  |   {mean_incoming[i]:6.2f} |  {std_incoming[i]:6.2f} |  {min_score:7.2f} |  {max_score:7.2f} | {role}")
    
    # Print overall metrics
    entropy = calculate_hierarchy_entropy(state)
    measures = calculate_agreement_measures(state)
    print(f"\nHierarchy Entropy: {entropy:.3f} (lower = more hierarchical)")
    print(f"Perception Agreement Measures:")
    for key, value in measures.items():
        print(f"  {key}: {value:.3f}")
    
    # Print recent interactions
    print("\nRecent Interactions:")
    for interaction in state['recent_interactions']:
        print(f"Claimer {interaction['claimer']} -> Target {interaction['target']}: "
              f"{'Success' if interaction['success'] else 'Failed'} "
              f"(Match: {interaction['match_score']:.3f})")
    
    # Add leadership stratification analysis
    strat_metrics, sorted_scores = calculate_leadership_stratification(state)
    print("\nLeadership Hierarchy Analysis:")
    print(f"Top Leader Score: {strat_metrics['top_leader_score']:.2f}")
    print(f"Bottom Follower Score: {strat_metrics['bottom_follower_score']:.2f}")
    print(f"Leadership Range: {strat_metrics['leadership_range']:.2f}")
    print(f"Leadership Quartile Ratio: {strat_metrics['quartile_ratio']:.2f}")
    print(f"Stratification Index: {strat_metrics['stratification_index']:.2f}")
    
    print("\nLeadership Hierarchy (from highest to lowest):")
    agent_rankings = np.argsort([-np.mean(perception_matrix[:,i]) for i in range(n_agents)])
    for rank, agent_idx in enumerate(agent_rankings):
        score = np.mean(perception_matrix[:,agent_idx])
        print(f"Rank {rank+1}: Agent {agent_idx} (Score: {score:.2f})")
    
    # Add emergence score
    emergence_score = calculate_leadership_emergence_icc(state)
    print(f"\nLeadership Emergence Score (ICC): {emergence_score:.3f}")
    print("(Higher scores indicate clearer leadership structure)")
    
    return entropy, measures, emergence_score

def run_single_team_test(model_type='base', n_steps=50, print_interval=10):
    """Run simulation for a single team and print detailed results.
    
    Args:
        model_type (str): Type of model to run ('base' or 'interactionist')
        n_steps (int): Number of simulation steps
        print_interval (int): How often to print detailed state
    """
    # Use parameters that worked well in sweep
    params = {
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
    
    # Add interactionist-specific parameters if needed
    if model_type == 'interactionist':
        params.update({
            'dyadic_interactions_before_switch': 10,
            'identity_update_rate': 0.6,
            'perception_update_rate': 0.6,
            'claim_success_boost': 12.0,
            'grant_success_boost': 12.0,
            'rejection_penalty': 8.0,
            'passivity_penalty': 4.0,
            'success_boost': 25.0,
            'failure_penalty': 20.0,
        })
    
    print("Creating team with parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")
    
    # Create and run appropriate model type
    if model_type == 'interactionist':
        model = InteractionistModel(params)
    else:
        model_params = ModelParameters(**params)
        model = BaseLeadershipModel(model_params)
    
    # Print initial characteristics and ILTs
    print("\nInitial Agent Characteristics:")
    for i, agent in enumerate(model.agents):
        print(f"\nAgent {i}:")
        print(f"Characteristic: {agent.characteristic}")
        print(f"ILT Schema: {agent.ilt_schema}")
        
        # Print match scores with other agents
        print("\nMatch Scores with Others:")
        for j, other in enumerate(model.agents):
            if i != j:
                match = agent.calculate_ilt_match(other.characteristic)
                print(f"  -> Agent {j}: {match:.3f}")
        
        # Print identity scores for interactionist model
        if model_type == 'interactionist':
            print(f"Leader Identity: {agent.leader_identity:.2f}")
            print(f"Follower Identity: {agent.follower_identity:.2f}")
    
    # Track metrics over time
    time_points = []
    entropy_scores = []
    agreement_measures = {
        'std_agreement': [],
        'gini_agreement': [],
        'cv_agreement': []
    }
    # Track individual agent leadership scores
    agent_scores = {i: [] for i in range(params['n_agents'])}
    
    # Add tracking for stratification
    stratification_over_time = []
    
    # Add tracking for emergence score
    emergence_scores = []
    
    # Run simulation
    for step in range(n_steps):
        state = model.step()
        
        # Calculate average incoming leadership for each agent
        n_agents = len(state['agents'])
        perception_matrix = np.zeros((n_agents, n_agents))
        for i, agent in enumerate(state['agents']):
            for j_str, value in agent['leadership_perceptions'].items():
                j = int(j_str)
                perception_matrix[i,j] = value
        
        mean_incoming = np.mean(perception_matrix, axis=0)
        
        # Only start tracking after step 5
        if step >= 5:
            time_points.append(step)
            entropy_scores.append(calculate_hierarchy_entropy(state))
            measures = calculate_agreement_measures(state)
            for key in agreement_measures:
                agreement_measures[key].append(measures[key])
            
            # Track individual scores
            for i in range(n_agents):
                agent_scores[i].append(mean_incoming[i])
        
        # Track stratification
        if step >= 5:
            strat_metrics, _ = calculate_leadership_stratification(state)
            stratification_over_time.append(strat_metrics['stratification_index'])
        
        # Track emergence score
        if step >= 5:
            emergence_scores.append(calculate_leadership_emergence_icc(state))
        
        # Print detailed state at intervals
        if step % print_interval == 0:
            print_perception_matrix(state, step)
            
            # Print additional info for interactionist model
            if model_type == 'interactionist':
                print("\nIdentity Scores:")
                for i, agent in enumerate(model.agents):
                    print(f"Agent {i}:")
                    print(f"  Leader Identity: {agent.leader_identity:.2f}")
                    print(f"  Follower Identity: {agent.follower_identity:.2f}")
    
    # Create plots directory if it doesn't exist
    plots_dir = Path('outputs/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create single emergence plot
    plt.figure(figsize=(12, 8))
    plt.plot(time_points, emergence_scores, color='purple', linewidth=2.5, label='Leadership Structure Emergence')
    
    # Add stage transition line
    if model_type == 'interactionist':
        plt.axvline(x=params['dyadic_interactions_before_switch'], color='gray', linestyle='--', alpha=0.5, 
                   label='Schema → Identity Transition')
    
    # Customize appearance
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Leadership Structure Emergence (ICC)', fontsize=12)
    plt.title('Development of Leadership Structure Over Time', fontsize=14, pad=20)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='lower right')
    
    # Set y-axis limits with some padding
    plt.ylim(-0.05, 1.05)
    
    # Add annotations
    plt.text(0.02, 0.98, 'Strong Structure', transform=plt.gca().transAxes, 
            fontsize=10, alpha=0.7, verticalalignment='top')
    plt.text(0.02, 0.02, 'Weak Structure', transform=plt.gca().transAxes, 
            fontsize=10, alpha=0.7)
    
    # Save plot
    emergence_plot_path = plots_dir / f'{model_type}_emergence_{timestamp}.png'
    plt.savefig(emergence_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print final summary
    print("\nFinal Model State Summary:")
    print(f"Model Type: {model_type}")
    print(f"Number of Steps: {n_steps}")
    print(f"\nFinal Emergence Score: {emergence_scores[-1]:.3f}")
    print("\nPlot saved to:")
    print(f"  {emergence_plot_path}")

def run_comparison_test(n_steps=50, n_runs=20, print_interval=10):
    """Run multiple simulations of base, interactionist, and null models and compare their emergence patterns.
    
    Args:
        n_steps (int): Number of steps per simulation
        n_runs (int): Number of simulation runs
        print_interval (int): How often to print detailed state
    """
    # Base parameters (shared between all models)
    base_params = {
        'n_agents': 6,
        'schema_dimensions': 2,
        'match_algorithm': 'minimum',
        'match_threshold': 0.41,
        'success_boost': 25.0,  # Harmonized with interactionist
        'failure_penalty': 20.0,  # Harmonized with interactionist
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
    
    # Storage for multiple runs
    base_runs = []
    interactionist_runs = []
    null_runs = []
    time_points = list(range(5, n_steps))  # Start from step 5 as before
    
    print("Running simulations...")
    for run in range(n_runs):
        # Set different random seed for each run
        run_seed = 42 + run
        base_params['random_seed'] = run_seed
        interactionist_params['random_seed'] = run_seed
        
        # Run base model
        base_model = BaseLeadershipModel(ModelParameters(**base_params))
        base_scores = []
        
        # Run interactionist model
        interactionist_model = InteractionistModel(interactionist_params)
        interactionist_scores = []
        
        # Run null model
        null_model = NullModel(base_params)
        null_scores = []
        
        # Run all models for n_steps
        for step in range(n_steps):
            # Step and collect scores if past initial period
            if step >= 5:
                # Base model
                state = base_model.step()
                base_scores.append(calculate_leadership_emergence_icc(state))
                
                # Interactionist model
                state = interactionist_model.step()
                interactionist_scores.append(calculate_leadership_emergence_icc(state))
                
                # Null model
                state = null_model.step()
                null_scores.append(calculate_leadership_emergence_icc(state))
            else:
                # Just step without collecting scores
                base_model.step()
                interactionist_model.step()
                null_model.step()
        
        base_runs.append(base_scores)
        interactionist_runs.append(interactionist_scores)
        null_runs.append(null_scores)
        
        if (run + 1) % 5 == 0:
            print(f"Completed {run + 1}/{n_runs} runs")
    
    # Convert to numpy arrays for easier manipulation
    base_runs = np.array(base_runs)
    interactionist_runs = np.array(interactionist_runs)
    null_runs = np.array(null_runs)
    
    # Calculate means and standard deviations
    base_mean = np.mean(base_runs, axis=0)
    interactionist_mean = np.mean(interactionist_runs, axis=0)
    null_mean = np.mean(null_runs, axis=0)
    
    # Create plots directory if it doesn't exist
    plots_dir = Path('outputs/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot individual runs with transparency
    for i in range(n_runs):
        plt.plot(time_points, base_runs[i], color='blue', alpha=0.1, linewidth=0.5)
        plt.plot(time_points, interactionist_runs[i], color='purple', alpha=0.1, linewidth=0.5)
        plt.plot(time_points, null_runs[i], color='gray', alpha=0.1, linewidth=0.5)
    
    # Plot means with solid lines
    plt.plot(time_points, base_mean, color='blue', linewidth=2.5, label='Base Model', alpha=0.8)
    plt.plot(time_points, interactionist_mean, color='purple', linewidth=2.5, label='Interactionist Model', alpha=0.8)
    plt.plot(time_points, null_mean, color='gray', linewidth=2.5, label='Null Model', alpha=0.8)
    
    # Add stage transition line
    plt.axvline(x=interactionist_params['dyadic_interactions_before_switch'], 
                color='gray', linestyle='--', alpha=0.5,
                label='Schema → Identity Transition')
    
    # Customize appearance
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Leadership Structure Emergence (ICC)', fontsize=12)
    plt.title('Comparison of Leadership Structure Development\nAcross Multiple Runs', fontsize=14, pad=20)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='lower right')
    
    # Set y-axis limits with some padding
    plt.ylim(-0.05, 1.05)
    
    # Add annotations
    plt.text(0.02, 0.98, 'Strong Structure', transform=plt.gca().transAxes, 
            fontsize=10, alpha=0.7, verticalalignment='top')
    plt.text(0.02, 0.02, 'Weak Structure', transform=plt.gca().transAxes, 
            fontsize=10, alpha=0.7)
    
    # Save plot
    comparison_plot_path = plots_dir / f'model_comparison_multi_{timestamp}.png'
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print final summary with means and standard deviations
    print("\nFinal Model Comparison (Mean ± Std):")
    print(f"Base Model: {np.mean(base_runs[:,-1]):.3f} ± {np.std(base_runs[:,-1]):.3f}")
    print(f"Interactionist Model: {np.mean(interactionist_runs[:,-1]):.3f} ± {np.std(interactionist_runs[:,-1]):.3f}")
    print(f"Null Model: {np.mean(null_runs[:,-1]):.3f} ± {np.std(null_runs[:,-1]):.3f}")
    print("\nPlot saved to:")
    print(f"  {comparison_plot_path}")

if __name__ == "__main__":
    # Import cognitive model
    from src.models.perspectives.cognitive import CognitiveModel, CognitiveParameters
    
    # Set up parameters using the optimized values we found
    params = {
        'n_agents': 6,
        'schema_dimensions': 2,
        'match_algorithm': 'average',
        'match_threshold': 0.47,
        'success_boost': 7.57,
        'failure_penalty': 4.89,
        'ilt_learning_rate': 0.26,
        'observation_weight': 1.91,
        'memory_decay': 0.074,
        'max_memory': 11,
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
    
    # Initialize model
    model = CognitiveModel(CognitiveParameters(**params))
    
    print("Starting Cognitive Model Test")
    print("=" * 50)
    
    # Run simulation with detailed analysis every 10 steps
    n_steps = 50
    analysis_steps = [0, 10, 20, 30, 40, 49]  # Steps to print detailed analysis
    
    for step in range(n_steps):
        state = model.step()
        
        if step in analysis_steps:
            print(f"\nStep {step}")
            print("-" * 20)
            
            # Print perception matrix
            print_perception_matrix(state, step)
            
            # Calculate and print agreement measures
            print("\nAgreement Analysis:")
            agreement = calculate_agreement_measures(state, debug=True)
            
            # Calculate and print emergence score
            print("\nEmergence Analysis:")
            emergence = calculate_leadership_emergence_icc(state, debug=True)
            print(f"Overall Emergence Score: {emergence:.3f}")
            
            # Calculate and print hierarchy metrics
            print("\nHierarchy Analysis:")
            metrics, scores = calculate_leadership_stratification(state)
            print("Leadership Scores (high to low):", [f"{s:.2f}" for s in scores])
            for metric, value in metrics.items():
                print(f"{metric}: {value:.3f}")
            
            print("\nAgent ILT Analysis:")
            for i, agent in enumerate(model.agents):
                print(f"Agent {i}:")
                print(f"  ILT Schema: {agent.ilt_schema}")
                print(f"  Characteristic: {agent.characteristic}")
                print(f"  Successful Leaders: {agent.successful_leaders}")
            
            print("=" * 50) 