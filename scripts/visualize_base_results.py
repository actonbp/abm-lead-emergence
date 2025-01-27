"""
Generate visualizations of base model results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.models.base_model import BaseLeadershipModel, ModelParameters

def plot_identity_evolution(model_history, output_path):
    """Plot how leader and follower identities evolve over time."""
    plt.figure(figsize=(12, 6))
    
    # Extract identity time series
    leader_ids = []
    follower_ids = []
    for state in model_history:
        leader_ids.append([agent['leader_identity'] for agent in state['agents']])
        follower_ids.append([agent['follower_identity'] for agent in state['agents']])
    
    leader_ids = np.array(leader_ids)
    follower_ids = np.array(follower_ids)
    
    # Plot individual trajectories
    plt.subplot(1, 2, 1)
    for i in range(leader_ids.shape[1]):
        plt.plot(leader_ids[:, i], 'r-', alpha=0.3, label='Leader' if i == 0 else None)
        plt.plot(follower_ids[:, i], 'b-', alpha=0.3, label='Follower' if i == 0 else None)
    
    plt.title('Individual Identity Trajectories')
    plt.xlabel('Time Steps')
    plt.ylabel('Identity Score')
    plt.legend()
    
    # Plot identity distributions over time
    plt.subplot(1, 2, 2)
    data = [leader_ids[-20:].flatten(), follower_ids[-20:].flatten()]
    plt.violinplot(data, positions=[1, 2], showmeans=True)
    plt.xticks([1, 2], ['Leader', 'Follower'])
    plt.title('Final Identity Distributions')
    plt.ylabel('Identity Score')
    
    plt.tight_layout()
    plt.savefig(output_path / 'identity_evolution.png')
    plt.close()

def plot_leadership_structure(model_history, output_path):
    """Plot the emergent leadership structure."""
    plt.figure(figsize=(15, 5))
    
    # Extract final perceptions
    final_state = model_history[-1]
    n_agents = len(final_state['agents'])
    perceptions = np.zeros((n_agents, n_agents))
    for i, agent in enumerate(final_state['agents']):
        for j, perception in agent['leadership_perceptions'].items():
            perceptions[i,int(j)] = perception
    
    # Plot perception network
    plt.subplot(1, 3, 1)
    sns.heatmap(perceptions, cmap='viridis', center=50)
    plt.title('Leadership Perception Network')
    plt.xlabel('Target Agent')
    plt.ylabel('Perceiver Agent')
    
    # Plot hierarchy levels
    plt.subplot(1, 3, 2)
    avg_received = np.mean(perceptions, axis=0)
    plt.bar(range(n_agents), avg_received)
    plt.title('Average Leadership Received')
    plt.xlabel('Agent')
    plt.ylabel('Average Perception')
    
    # Plot perception agreement over time
    plt.subplot(1, 3, 3)
    agreements = []
    for state in model_history:
        perceptions = np.zeros((n_agents, n_agents))
        for i, agent in enumerate(state['agents']):
            for j, perception in agent['leadership_perceptions'].items():
                perceptions[i,int(j)] = perception
        
        # Calculate agreement as correlation between agents' perceptions
        correlations = []
        for i in range(n_agents):
            for j in range(i+1, n_agents):
                r = np.corrcoef(perceptions[i], perceptions[j])[0,1]
                correlations.append(r)
        agreements.append(np.mean(correlations))
    
    plt.plot(agreements)
    plt.title('Perception Agreement Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Agreement')
    
    plt.tight_layout()
    plt.savefig(output_path / 'leadership_structure.png')
    plt.close()

def plot_parameter_sensitivity(output_path):
    """Plot sensitivity analysis of key parameters."""
    # Best parameters from our sweep
    best_params = {
        'n_agents': 5,
        'schema_dimensions': 2,
        'characteristic_distribution': 'normal',
        'ilt_distribution': 'normal',
        'distribution_std': 0.169,
        'match_threshold': 0.570,
        'success_boost': 6.992,
        'failure_penalty': 2.299,
        'n_steps': 100,
        'interaction_selection': 'random',
        'schema_type': 'continuous',
        'dimension_weights': 'uniform',
        'grant_first': False,
        'allow_mutual_claims': False,
        'allow_self_loops': False,
        'simultaneous_roles': False
    }
    
    # Parameters to vary
    param_ranges = {
        'success_boost': np.linspace(5, 9, 5),
        'failure_penalty': np.linspace(2, 4, 5),
        'match_threshold': np.linspace(0.4, 0.6, 5),
        'distribution_std': np.linspace(0.1, 0.2, 5)
    }
    
    plt.figure(figsize=(15, 10))
    
    for i, (param_name, param_values) in enumerate(param_ranges.items()):
        identity_diffs = []
        hierarchy_strengths = []
        
        for value in param_values:
            # Create parameters with this value
            test_params = best_params.copy()
            test_params[param_name] = value
            
            # Run model
            model = BaseLeadershipModel(params=ModelParameters(**test_params))
            history = []
            for _ in range(100):
                state = model.step()
                history.append(state)
            
            # Calculate metrics
            final_state = history[-1]
            leader_ids = [agent['leader_identity'] for agent in final_state['agents']]
            follower_ids = [agent['follower_identity'] for agent in final_state['agents']]
            identity_diff = abs(np.mean(leader_ids) - np.mean(follower_ids))
            
            perceptions = np.zeros((len(final_state['agents']), len(final_state['agents'])))
            for j, agent in enumerate(final_state['agents']):
                for k, perception in agent['leadership_perceptions'].items():
                    perceptions[j,int(k)] = perception
            hierarchy_strength = np.std(np.mean(perceptions, axis=0)) / 100
            
            identity_diffs.append(identity_diff)
            hierarchy_strengths.append(hierarchy_strength)
        
        # Plot results
        plt.subplot(2, 2, i+1)
        plt.plot(param_values, identity_diffs, 'b-', label='Identity Differentiation')
        plt.plot(param_values, hierarchy_strengths, 'r-', label='Hierarchy Strength')
        plt.title(f'Effect of {param_name}')
        plt.xlabel(param_name)
        plt.ylabel('Metric Value')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'parameter_sensitivity.png')
    plt.close()

def main():
    """Generate all visualizations."""
    # Create output directory
    output_dir = Path('outputs/parameter_sweep')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run model with best parameters
    best_params = {
        'n_agents': 5,
        'schema_dimensions': 2,
        'characteristic_distribution': 'normal',
        'ilt_distribution': 'normal',
        'distribution_std': 0.169,
        'match_threshold': 0.570,
        'success_boost': 6.992,
        'failure_penalty': 2.299,
        'n_steps': 100,
        'interaction_selection': 'random',
        'schema_type': 'continuous',
        'dimension_weights': 'uniform',
        'grant_first': False,
        'allow_mutual_claims': False,
        'allow_self_loops': False,
        'simultaneous_roles': False
    }
    
    model = BaseLeadershipModel(params=ModelParameters(**best_params))
    history = []
    for _ in range(100):
        state = model.step()
        history.append(state)
    
    # Generate visualizations
    plot_identity_evolution(history, output_dir)
    plot_leadership_structure(history, output_dir)
    plot_parameter_sensitivity(output_dir)

if __name__ == "__main__":
    main() 