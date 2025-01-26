#!/usr/bin/env python3

import os
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging
import matplotlib.pyplot as plt
import json

from src.analysis.parameter_exploration import ParameterExplorer
from src.analysis.ml_pipeline import MLPipeline
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.models.base_model import BaseLeadershipModel

def load_sweep_config(config_path):
    """Load parameter sweep configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_output_directory(base_dir: str) -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def evaluate_configuration(
    config: Dict[str, Any],
    n_steps: int,
    n_replications: int
) -> Dict[str, Any]:
    """Evaluate a parameter configuration by running simulations.
    
    Args:
        config: Parameter configuration to evaluate
        n_steps: Number of steps per simulation
        n_replications: Number of replications per configuration
        
    Returns:
        Dictionary containing evaluation results
    """
    # Add n_steps to config
    full_config = config.copy()
    full_config['n_steps'] = n_steps
    
    # Run multiple replications
    metrics_list = []
    histories = []
    
    for _ in range(n_replications):
        # Create and run model
        model = BaseLeadershipModel(config=full_config)
        history = model.run(n_steps=n_steps)
        histories.append(history)
        
        # Calculate metrics for this replication
        metrics = {}
        
        # Get final values (last 10 steps)
        final_slice = slice(-10, None)
        metrics['kendall_w'] = np.mean(history['kendall_w'][final_slice])
        metrics['krippendorff_alpha'] = np.mean(history['krippendorff_alpha'][final_slice])
        metrics['normalized_entropy'] = np.mean(history['normalized_entropy'][final_slice])
        metrics['top_leader_agreement'] = np.mean(history['top_leader_agreement'][final_slice])
        
        metrics_list.append(metrics)
    
    # Average metrics across replications
    avg_metrics = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        avg_metrics[key] = float(np.mean(values))
    
    return {
        'metrics': avg_metrics,
        'histories': histories
    }

def generate_plots(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate plots from analysis results.
    
    Args:
        results: Dictionary containing analysis results
        output_dir: Directory to save plots
    """
    # Create plots directory
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Plot parameter importance
    plt.figure(figsize=(10, 6))
    params = list(results['parameter_importance'].keys())
    importance = list(results['parameter_importance'].values())
    plt.barh(params, importance)
    plt.title('Parameter Importance')
    plt.xlabel('Absolute Correlation with Objective')
    plt.tight_layout()
    plt.savefig(plots_dir / 'parameter_importance.png')
    plt.close()

def save_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Save analysis results to file.
    
    Args:
        results: Dictionary containing analysis results
        output_dir: Directory to save results
    """
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray, list)):
            return [convert_to_serializable(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        return obj
    
    # Save results as JSON
    with open(output_dir / 'analysis_results.json', 'w') as f:
        # Convert numpy arrays to lists
        serializable_results = {
            'best_configs': convert_to_serializable(results['best_configs']),
            'best_scores': convert_to_serializable(results['best_scores']),
            'clusters': convert_to_serializable(results['clusters']),
            'parameter_importance': convert_to_serializable(results['parameter_importance'])
        }
        json.dump(serializable_results, f, indent=2)

def run_parameter_sweep(sweep_config_path: str, base_config_path: str) -> Dict:
    """Run parameter sweep analysis."""
    # Load configurations
    with open(sweep_config_path) as f:
        sweep_config = yaml.safe_load(f)
    
    # Setup output directory
    output_dir = setup_output_directory(sweep_config['output']['output_dir'])
    
    # Initialize logger
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    # Log configurations
    logger.info(f"Loaded sweep configuration from {sweep_config_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize parameter explorer
    explorer = ParameterExplorer(
        parameter_space=sweep_config['parameter_space'],
        n_initial_samples=sweep_config['ml_pipeline']['n_initial_samples']
    )
    
    # Initialize ML pipeline
    pipeline = MLPipeline(
        n_iterations=sweep_config['ml_pipeline']['n_iterations'],
        batch_size=sweep_config['ml_pipeline']['batch_size'],
        n_clusters=sweep_config['ml_pipeline']['n_clusters']
    )
    
    # Run initial Latin Hypercube sampling
    logger.info("Running initial Latin Hypercube sampling...")
    initial_configs = explorer.generate_initial_samples()
    
    # Evaluate initial configurations
    initial_results = []
    for config in initial_configs:
        result = evaluate_configuration(
            config,
            n_steps=sweep_config['simulation']['n_steps'],
            n_replications=sweep_config['simulation']['n_replications']
        )
        initial_results.append(result)
    
    # Run Bayesian optimization
    logger.info("Running Bayesian optimization...")
    optimized_results = pipeline.run_optimization(
        initial_configs=initial_configs,
        initial_results=initial_results,
        parameter_space=sweep_config['parameter_space'],
        n_iterations=sweep_config['ml_pipeline']['n_iterations']
    )
    
    # Analyze results
    logger.info("Analyzing results...")
    analysis_results = pipeline.analyze_results(
        configs=initial_configs + optimized_results['configs'],
        results=initial_results + optimized_results['results']
    )
    
    # Generate plots if specified
    if sweep_config['output']['generate_plots']:
        logger.info("Generating plots...")
        generate_plots(analysis_results, output_dir)
    
    # Save final results
    if sweep_config['output']['save_results']:
        logger.info("Saving results...")
        save_results(analysis_results, output_dir)
    
    return analysis_results

def main():
    """Main function."""
    # Set paths
    sweep_config_path = 'config/parameter_sweep.yaml'
    base_config_path = 'config/base_config.yaml'
    
    # Run parameter sweep
    results = run_parameter_sweep(sweep_config_path, base_config_path)
    
    # Print summary of best configurations
    print("\nBest configurations found:")
    for i, config in enumerate(results['best_configs'][:3]):
        print(f"\nConfiguration {i+1}:")
        for param, value in config.items():
            print(f"{param}: {value}")
        print(f"Score: {results['best_scores'][i]:.3f}")

if __name__ == '__main__':
    main() 