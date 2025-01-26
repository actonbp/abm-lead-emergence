"""
Main script for running ML-driven analysis of leadership emergence simulations.
"""

import argparse
from pathlib import Path
import json
import yaml
from typing import Dict, Any

from src.models.base_model import BaseLeadershipModel
from src.analysis.ml_pipeline import MLPipeline
from src.analysis.theory_validation import TheoryValidator, TheoryType

def load_config(config_path: str) -> Dict[str, Any]:
    """Load analysis configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_analysis(config: Dict[str, Any], output_dir: str):
    """Run complete ML analysis pipeline."""
    # Initialize components
    ml_pipeline = MLPipeline(output_dir=output_dir)
    theory_validator = TheoryValidator()
    
    # Define parameter space
    parameter_space = {
        'n_agents': (4, 16),
        'initial_li_equal': (0, 1),  # Boolean treated as float for GP
        'li_change_rate': (0.1, 5.0),
        'interaction_radius': (0.5, 2.0),
        'memory_length': (0, 10),
        'claim_threshold': (0.3, 0.7),
        'grant_threshold': (0.4, 0.8),
        'schema_weight': (0.0, 1.0),
        'social_identity_influence': (0.0, 1.0)
    }
    
    # Initial parameter exploration
    print("Starting parameter exploration...")
    initial_params = ml_pipeline.explore_parameter_space(
        parameter_space,
        n_initial=config['n_initial_samples']
    )
    
    # Run initial simulations
    print("Running initial simulations...")
    simulation_results = []
    for params in initial_params:
        model = BaseLeadershipModel(params)
        history = model.run_simulation(config['n_steps'])
        simulation_results.append({
            "parameters": params,
            "history": history
        })
    
    # Iterative improvement
    for iteration in range(config['n_iterations']):
        print(f"Starting iteration {iteration + 1}/{config['n_iterations']}")
        
        # Analyze patterns
        pattern_results = ml_pipeline.analyze_patterns(
            simulation_results,
            n_clusters=config['n_clusters']
        )
        
        # Validate against theories
        theory_results = theory_validator.compare_theories(simulation_results)
        
        # Update surrogate model
        objectives = []
        for result in simulation_results:
            # Calculate overall score based on pattern clarity and theory alignment
            pattern_score = pattern_results['cluster_stats'][0]['size'] / len(simulation_results)
            theory_score = theory_results['best_fit_score']
            objectives.append({
                'overall_score': 0.7 * pattern_score + 0.3 * theory_score
            })
        
        ml_pipeline.update_surrogate_model(initial_params, objectives)
        
        # Select next parameters
        next_params = ml_pipeline.select_next_parameters(
            parameter_space,
            batch_size=config['batch_size']
        )
        
        # Run additional simulations
        for params in next_params:
            model = BaseLeadershipModel(params)
            history = model.run_simulation(config['n_steps'])
            simulation_results.append({
                "parameters": params,
                "history": history
            })
            initial_params.append(params)
    
    # Final analysis
    print("Running final analysis...")
    final_patterns = ml_pipeline.analyze_patterns(
        simulation_results,
        n_clusters=config['n_clusters']
    )
    
    final_theory_comparison = theory_validator.compare_theories(simulation_results)
    
    # Save results
    results = {
        "metadata": {
            "n_simulations": len(simulation_results),
            "parameter_space": parameter_space,
            "config": config
        },
        "pattern_analysis": final_patterns,
        "theory_comparison": final_theory_comparison,
        "parameter_importance": ml_pipeline.parameter_importance
    }
    
    output_path = Path(output_dir) / "ml_analysis_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis complete. Results saved to {output_path}")
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total simulations: {len(simulation_results)}")
    print(f"Best-fit theory: {final_theory_comparison['best_fit_theory']}")
    print(f"Theory alignment score: {final_theory_comparison['best_fit_score']:.3f}")
    print("\nTop parameter importance:")
    for i, (param, importance) in enumerate(zip(parameter_space.keys(), ml_pipeline.parameter_importance)):
        print(f"{i+1}. {param}: {importance:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Run ML analysis of leadership emergence")
    parser.add_argument("--config", type=str, default="config/analysis_config.yaml",
                      help="Path to analysis configuration file")
    parser.add_argument("--output-dir", type=str, default="data/ml_analysis",
                      help="Directory for analysis outputs")
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_analysis(config, args.output_dir)

if __name__ == "__main__":
    main() 