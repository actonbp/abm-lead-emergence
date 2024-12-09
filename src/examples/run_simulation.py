"""
Example script demonstrating how to use the leadership emergence model and create visualizations.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base_model import LeadershipEmergenceModel
from src.visualization.plot_outcomes import (
    plot_identity_evolution,
    plot_network_metrics,
    plot_leadership_network,
    create_summary_plots,
    plot_interaction_heatmap
)
import matplotlib.pyplot as plt

def run_example_simulation():
    """Run an example simulation and create visualizations."""
    
    print("Initializing model...")
    # Initialize model with example parameters
    model = LeadershipEmergenceModel(
        n_agents=4,
        initial_li_equal=True,
        li_change_rate=2.0,
        random_seed=42
    )
    
    print("Running simulation...")
    # Run simulation for 100 steps
    for step in range(100):
        if step % 20 == 0:
            print(f"Step {step}/100")
        model.step()
    
    print("\nCreating visualizations...")
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "example_simulation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot and save identity evolution
    plot_identity_evolution(model.history)
    plt.savefig(output_dir / "identity_evolution.png")
    plt.close()
    
    # Plot and save network metrics
    plot_network_metrics(model.history)
    plt.savefig(output_dir / "network_metrics.png")
    plt.close()
    
    # Plot and save leadership network
    plot_leadership_network(model.interaction_network)
    plt.savefig(output_dir / "leadership_network.png")
    plt.close()
    
    # Plot and save interaction heatmap
    plot_interaction_heatmap(model)
    plt.savefig(output_dir / "interaction_heatmap.png")
    plt.close()
    
    # Create and save summary plots
    create_summary_plots(model)
    plt.savefig(output_dir / "summary_plots.png")
    plt.close()
    
    print("\nSimulation complete! Visualizations saved in outputs/example_simulation/")
    
    # Print some summary statistics
    print("\nFinal Statistics:")
    print(f"Mean Leader Identity: {sum(model.agents[-1].leader_identity for agent in model.agents) / model.n_agents:.2f}")
    print(f"Mean Follower Identity: {sum(model.agents[-1].follower_identity for agent in model.agents) / model.n_agents:.2f}")
    print(f"Network Density: {model.history['density'][-1]:.2f}")
    print(f"Leadership Centralization: {model.history['centralization'][-1]:.2f}")

if __name__ == "__main__":
    run_example_simulation() 