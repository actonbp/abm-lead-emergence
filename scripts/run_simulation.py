"""
Run a leadership emergence simulation using a configuration file.
"""

import sys
from pathlib import Path
import yaml
import argparse
from datetime import datetime
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.models.base_model import BaseLeadershipModel
from src.simulation.runner import SimulationRunner
from src.visualization.plot_outcomes import (
    plot_identity_evolution,
    plot_network_metrics,
    plot_leadership_network,
    create_summary_plots
)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_output_dirs(base_dir="data"):
    """Create output directories if they don't exist."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directories
    raw_dir = Path(base_dir) / "raw" / timestamp
    processed_dir = Path(base_dir) / "processed" / timestamp
    
    for dir_path in [raw_dir, processed_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return raw_dir, processed_dir

def run_simulation(config_path):
    """Run simulation using provided configuration."""
    
    # Load configuration
    config = load_config(config_path)
    print(f"Loaded configuration from {config_path}")
    
    # Setup output directories
    raw_dir, processed_dir = setup_output_dirs()
    print(f"Output directories created:\nRaw: {raw_dir}\nProcessed: {processed_dir}")
    
    # Initialize model with configuration
    model = BaseLeadershipModel(config=config)
    
    # Initialize runner
    runner = SimulationRunner(model, config)
    
    print("\nRunning simulation...")
    # Run simulation
    results = runner.run()
    
    # Save raw results
    results_file = raw_dir / "simulation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSimulation complete! Results saved to {results_file}")
    
    # Save configuration used
    config_file = raw_dir / "config_used.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    return results, raw_dir, processed_dir

def main():
    parser = argparse.ArgumentParser(description="Run leadership emergence simulation")
    parser.add_argument("--config", required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    
    results, raw_dir, processed_dir = run_simulation(args.config)
    
    print("\nFinal Statistics:")
    print(f"Raw data directory: {raw_dir}")
    print(f"Processed data directory: {processed_dir}")

if __name__ == "__main__":
    main() 