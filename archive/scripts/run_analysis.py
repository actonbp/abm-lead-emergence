#!/usr/bin/env python3
"""
Command-line script for running leadership emergence analyses.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from src.models.base_model import BaseLeadershipModel
from src.models.schema_model import SchemaModel
from src.simulation.runner import SimulationRunner
from src.analysis.pipeline import AnalysisPipeline

def load_config(config_file: str) -> Dict[str, Any]:
    """Load analysis configuration from JSON file."""
    with open(config_file) as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(
        description="Run leadership emergence simulations and analysis"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/analysis_config.json",
        help="Path to analysis configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Directory for output files"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up model
    model_type = config.get("model_type", "base")
    model_class = {
        "base": BaseLeadershipModel,
        "schema": SchemaModel
    }[model_type]
    
    # Run simulations
    print(f"Running {model_type} model simulations...")
    runner = SimulationRunner(
        model_class=model_class,
        parameter_space=config["parameter_space"],
        n_steps=config["n_steps"],
        n_replications=config["n_replications"],
        output_dir=f"{args.output_dir}/raw",
        random_seed=args.random_seed
    )
    
    results_file = runner.run_batch(
        n_processes=config.get("n_processes")
    )
    
    # Load simulation results
    with open(results_file) as f:
        simulation_results = json.load(f)["results"]
    
    # Run analysis
    print("\nAnalyzing results...")
    pipeline = AnalysisPipeline(
        output_dir=f"{args.output_dir}/analysis",
        random_seed=args.random_seed
    )
    
    analysis_results = pipeline.run_analysis(
        simulation_results,
        config["analysis_params"]
    )
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}/analysis")

if __name__ == "__main__":
    main() 