#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.models.base_model import BaseLeadershipModel

def run_configuration(config: dict, n_steps: int = 100, n_replications: int = 10):
    """Run a configuration multiple times and collect emergence metrics."""
    histories = []
    
    for _ in range(n_replications):
        model = BaseLeadershipModel(config=config)
        history = model.run(n_steps=n_steps)
        histories.append(history)
    
    # Average metrics across replications
    avg_metrics = {
        'kendall_w': np.mean([h['kendall_w'] for h in histories], axis=0),
        'krippendorff_alpha': np.mean([h['krippendorff_alpha'] for h in histories], axis=0),
        'normalized_entropy': np.mean([h['normalized_entropy'] for h in histories], axis=0),
        'top_leader_agreement': np.mean([h['top_leader_agreement'] for h in histories], axis=0)
    }
    
    # Calculate standard deviations
    std_metrics = {
        'kendall_w': np.std([h['kendall_w'] for h in histories], axis=0),
        'krippendorff_alpha': np.std([h['krippendorff_alpha'] for h in histories], axis=0),
        'normalized_entropy': np.std([h['normalized_entropy'] for h in histories], axis=0),
        'top_leader_agreement': np.std([h['top_leader_agreement'] for h in histories], axis=0)
    }
    
    return avg_metrics, std_metrics

def plot_emergence_patterns(configs: list, labels: list, n_steps: int = 100, n_replications: int = 10):
    """Plot emergence patterns for multiple configurations."""
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Leadership Emergence Patterns Over Time', fontsize=14)
    
    metrics = ['kendall_w', 'krippendorff_alpha', 'normalized_entropy', 'top_leader_agreement']
    titles = ["Kendall's W", "Krippendorff's Alpha", 'Normalized Entropy', 'Top Leader Agreement']
    
    # Colors for different configurations
    colors = ['b', 'g', 'r']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i // 2, i % 2]
        
        for j, (config, label) in enumerate(zip(configs, labels)):
            # Run simulation
            avg_metrics, std_metrics = run_configuration(config, n_steps, n_replications)
            
            # Plot mean with confidence interval
            time = np.arange(n_steps)
            mean = avg_metrics[metric]
            std = std_metrics[metric]
            
            ax.plot(time, mean, color=colors[j], label=label)
            ax.fill_between(time, mean - std, mean + std, color=colors[j], alpha=0.2)
        
        ax.set_title(title)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Metric Value')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    
    # Create output directory
    output_dir = Path('outputs/emergence_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plot
    plt.savefig(output_dir / 'emergence_patterns.png')
    plt.close()

def main():
    """Main function to analyze emergence patterns."""
    # Define configurations to analyze
    configs = [
        {
            # Original Configuration 1 (Sigmoid transition, moderate group)
            'n_agents': 6,
            'initial_li_equal': False,
            'weight_function': 'sigmoid',
            'li_change_rate': 0.020,
            'schema_weight': 0.112,
            'weight_transition_start': 0.350,
            'weight_transition_end': 0.649,
            'claim_threshold': 0.564,
            'grant_threshold': 0.537,
            'perception_change_success': 0.144,
            'perception_change_reject': 0.109
        },
        {
            # New Optimized Configuration
            'n_agents': 4,
            'initial_li_equal': True,
            'weight_function': 'sigmoid',
            'li_change_rate': 0.2,
            'schema_weight': 0.4,
            'weight_transition_start': 0.3,
            'weight_transition_end': 0.7,
            'claim_threshold': 0.3,
            'grant_threshold': 0.3,
            'perception_change_success': 0.2,
            'perception_change_reject': 0.1
        },
        {
            # Original Configuration 3 (Linear transition, large group)
            'n_agents': 8,
            'initial_li_equal': False,
            'weight_function': 'linear',
            'li_change_rate': 0.166,
            'schema_weight': 0.313,
            'weight_transition_start': 0.360,
            'weight_transition_end': 0.729,
            'claim_threshold': 0.574,
            'grant_threshold': 0.394,
            'perception_change_success': 0.054,
            'perception_change_reject': 0.162
        }
    ]
    
    labels = [
        'Original (n=6, Sigmoid)',
        'Optimized (n=4, Sigmoid)',
        'Original (n=8, Linear)'
    ]
    
    # Plot emergence patterns
    plot_emergence_patterns(configs, labels)

if __name__ == '__main__':
    main() 