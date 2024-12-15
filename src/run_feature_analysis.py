#!/usr/bin/env python3
"""
Script to run leadership emergence simulations and extract features.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
import logging
import shutil
import os

from models.schema_model import SchemaModel
from simulation.runner import BatchRunner, SimulationConfig
from features.time_series import TimeSeriesFeatureExtractor
from features.batch_extractor import BatchFeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories(base_dir: str) -> tuple[Path, Path, Path]:
    """Setup directory structure for experiment.
    
    Args:
        base_dir: Base directory path
        
    Returns:
        Tuple of (base_dir, sim_dir, feature_dir) Paths
    """
    base_dir = Path(base_dir)
    sim_dir = base_dir / "simulations"
    feature_dir = base_dir / "features"
    
    # Remove existing directories if they exist
    if base_dir.exists():
        logger.info(f"Removing existing directory: {base_dir}")
        shutil.rmtree(base_dir)
    
    # Create fresh directories
    for d in [base_dir, sim_dir, feature_dir]:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {d}")
    
    return base_dir, sim_dir, feature_dir


def create_parameter_grid():
    """Create grid of parameter combinations to explore."""
    return {
        "n_agents": [4, 6, 8],
        "initial_li_equal": [True, False],
        "li_change_rate": [0.1, 0.3, 0.5],
        "schema_weight": [0.3, 0.5, 0.7],
        "claim_threshold": [0.6],
        "grant_threshold": [0.4]
    }


def run_experiment(output_dir: str, n_replications: int = 5):
    """Run batch of simulations and extract features.
    
    Args:
        output_dir: Directory for output files
        n_replications: Number of replications per parameter combination
    """
    # Setup directories
    base_dir, sim_dir, feature_dir = setup_directories(output_dir)
    
    # Generate parameter combinations
    param_grid = create_parameter_grid()
    param_combinations = []
    
    # Create all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for values in product(*param_values):
        params = dict(zip(param_names, values))
        for rep in range(n_replications):
            config = SimulationConfig(
                model_params=params,
                n_steps=100,
                random_seed=42 + len(param_combinations)
            )
            param_combinations.append(config)
    
    # Run simulations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_id = f"experiment_{timestamp}"
    
    logger.info(f"Running {len(param_combinations)} simulations...")
    runner = BatchRunner(
        model_class=SchemaModel,
        output_dir=sim_dir,
        n_jobs=os.cpu_count() // 2  # Use half of available cores
    )
    
    result_files = runner.run_batch(param_combinations, batch_id)
    batch_dir = sim_dir / f"batch_{batch_id}"
    logger.info(f"Simulations completed. Results saved in: {batch_dir}")
    
    if not result_files:
        logger.error("No successful simulations to analyze")
        return None, None
    
    # Extract features
    logger.info("Extracting features...")
    feature_extractor = TimeSeriesFeatureExtractor(stability_threshold=0.05)
    batch_extractor = BatchFeatureExtractor(
        feature_extractor,
        n_jobs=os.cpu_count() // 2  # Use half of available cores
    )
    
    features_file = feature_dir / f"features_{batch_id}.csv"
    features_df = batch_extractor.extract_batch_features(
        str(batch_dir),
        str(features_file)
    )
    
    if features_df.empty:
        logger.error("No features could be extracted")
        return None, None
    
    # Generate summary statistics
    logger.info("Generating summary statistics...")
    summary_stats = analyze_features(features_df)
    
    summary_file = feature_dir / f"summary_{batch_id}.json"
    with open(summary_file, "w") as f:
        json.dump(summary_stats, f, indent=2)
    
    logger.info(f"Results saved:")
    logger.info(f"- Features: {features_file}")
    logger.info(f"- Summary: {summary_file}")
    
    return features_df, summary_stats


def analyze_features(df: pd.DataFrame) -> dict:
    """Generate summary statistics from extracted features.
    
    Args:
        df: DataFrame containing extracted features
        
    Returns:
        Dict containing summary statistics
    """
    # Group by parameter combinations
    group_cols = [col for col in ['n_agents', 'initial_li_equal', 'li_change_rate', 'schema_weight']
                 if col in df.columns]
    
    # Calculate statistics for each feature
    feature_cols = [col for col in [
        'mean_final_li', 'mean_final_fi',
        'time_to_li_stability', 'time_to_fi_stability',
        'mean_role_diff', 'role_polarization',
        'emergence_speed', 'leader_consistency'
    ] if col in df.columns]
    
    if not feature_cols:
        logger.error("No feature columns found in DataFrame")
        return {}
    
    summary = {
        'parameter_effects': {},
        'correlations': {},
        'overall_stats': {},
        'feature_importance': {}
    }
    
    # Parameter effects
    if group_cols:
        for param in group_cols:
            effects = {}
            for feature in feature_cols:
                # Calculate mean and std feature value for each parameter value
                param_stats = df.groupby(param)[feature].agg(['mean', 'std']).round(3)
                effects[feature] = param_stats.to_dict('index')
            summary['parameter_effects'][param] = effects
    
    # Feature correlations
    if len(feature_cols) > 1:
        corr_matrix = df[feature_cols].corr().round(3)
        for f1 in feature_cols:
            for f2 in feature_cols:
                if f1 < f2:  # Only store upper triangle
                    corr = corr_matrix.loc[f1, f2]
                    if abs(corr) > 0.3:  # Store moderate to strong correlations
                        key = f"{f1}_vs_{f2}"
                        summary['correlations'][key] = float(corr)
    
    # Overall statistics
    for feature in feature_cols:
        stats = df[feature].describe().round(3)
        summary['overall_stats'][feature] = {
            'mean': float(stats['mean']),
            'std': float(stats['std']),
            'min': float(stats['min']),
            'max': float(stats['max']),
            'q1': float(stats['25%']),
            'median': float(stats['50%']),
            'q3': float(stats['75%'])
        }
    
    # Feature importance (variance explained)
    total_var = df[feature_cols].var().sum()
    if total_var > 0:
        var_explained = (df[feature_cols].var() / total_var).round(3)
        summary['feature_importance'] = var_explained.to_dict()
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run leadership emergence simulations and extract features"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/experiments",
        help="Directory for output files"
    )
    parser.add_argument(
        "--n-replications",
        type=int,
        default=5,
        help="Number of replications per parameter combination"
    )
    args = parser.parse_args()
    
    features_df, summary_stats = run_experiment(
        args.output_dir,
        args.n_replications
    )
    
    if features_df is None or summary_stats is None:
        logger.error("Analysis failed")
        return
    
    # Print key findings
    logger.info("\nKey findings:")
    
    # Most important features
    if summary_stats.get('feature_importance'):
        logger.info("\nFeature importance (variance explained):")
        for feature, importance in sorted(
            summary_stats['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            logger.info(f"- {feature}: {importance:.3f}")
    
    # Strong correlations
    if summary_stats.get('correlations'):
        logger.info("\nStrong feature correlations:")
        for pair, corr in sorted(
            summary_stats['correlations'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ):
            logger.info(f"- {pair}: {corr:.3f}")
    
    # Parameter effects
    if summary_stats.get('parameter_effects'):
        logger.info("\nNotable parameter effects:")
        for param, effects in summary_stats['parameter_effects'].items():
            for feature, stats in effects.items():
                # Calculate effect size using max difference in means
                means = [s['mean'] for s in stats.values()]
                effect_size = max(means) - min(means)
                if effect_size > 0.1:  # Only show notable effects
                    logger.info(f"- {param} on {feature}: {effect_size:.3f}")
                    # Show parameter values with highest/lowest effect
                    max_val = max(stats.items(), key=lambda x: x[1]['mean'])
                    min_val = min(stats.items(), key=lambda x: x[1]['mean'])
                    logger.info(f"  Best {param}={max_val[0]} (mean={max_val[1]['mean']:.3f})")
                    logger.info(f"  Worst {param}={min_val[0]} (mean={min_val[1]['mean']:.3f})")


if __name__ == "__main__":
    main() 