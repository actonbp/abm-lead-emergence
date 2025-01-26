#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr

def load_results(results_path):
    """Load results from JSON file."""
    with open(results_path) as f:
        return json.load(f)

def analyze_metrics(data):
    """Analyze metric statistics."""
    metrics = ['emergence_slope', 'early_diversity', 'convergence_quality', 'emergence_stability']
    stats = {}
    
    print("Metric Statistics:")
    print("-----------------")
    for metric in metrics:
        values = [r['metrics'][metric] for r in data]
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
        print(f"\n{metric}:")
        print(f"  Mean: {stats[metric]['mean']:.4f}")
        print(f"  Std:  {stats[metric]['std']:.4f}")
        print(f"  Min:  {stats[metric]['min']:.4f}")
        print(f"  Max:  {stats[metric]['max']:.4f}")
    
    return stats

def analyze_parameters(data):
    """Analyze parameter distributions and correlations."""
    # Extract parameters and convert to DataFrame
    params = []
    metrics = []
    for result in data:
        params.append(result['parameters'])
        metrics.append(result['metrics'])
    
    param_df = pd.DataFrame(params)
    metric_df = pd.DataFrame(metrics)
    
    print("\nParameter Statistics:")
    print("--------------------")
    for col in param_df.columns:
        if pd.api.types.is_numeric_dtype(param_df[col]):
            print(f"\n{col}:")
            print(f"  Mean: {param_df[col].mean():.4f}")
            print(f"  Std:  {param_df[col].std():.4f}")
            print(f"  Min:  {param_df[col].min():.4f}")
            print(f"  Max:  {param_df[col].max():.4f}")
        else:
            counts = param_df[col].value_counts()
            print(f"\n{col}:")
            for val, count in counts.items():
                print(f"  {val}: {count} times ({count/len(param_df)*100:.1f}%)")
    
    # Calculate correlations between parameters and metrics
    print("\nParameter-Metric Correlations:")
    print("-----------------------------")
    for param in param_df.columns:
        if pd.api.types.is_numeric_dtype(param_df[param]):
            print(f"\n{param}:")
            for metric in metric_df.columns:
                corr, p = spearmanr(param_df[param], metric_df[metric])
                if p < 0.05:  # Only show significant correlations
                    print(f"  {metric}: r={corr:.3f} (p={p:.3f})")
    
    return param_df, metric_df

def analyze_best_configurations(data, metric_weights=None):
    """Analyze top performing configurations."""
    if metric_weights is None:
        metric_weights = {
            'emergence_slope': 0.3,
            'early_diversity': 0.2,
            'convergence_quality': 0.3,
            'emergence_stability': 0.2
        }
    
    # Calculate weighted scores
    scores = []
    for result in data:
        score = sum(
            metric_weights[metric] * result['metrics'][metric]
            for metric in metric_weights
        )
        scores.append((score, result))
    
    # Sort by score
    scores.sort(reverse=True)
    
    print("\nTop 5 Configurations:")
    print("--------------------")
    for i, (score, result) in enumerate(scores[:5]):
        print(f"\nRank {i+1} (Score: {score:.4f}):")
        print("Parameters:")
        for param, value in result['parameters'].items():
            print(f"  {param}: {value}")
        print("Metrics:")
        for metric, value in result['metrics'].items():
            print(f"  {metric}: {value:.4f}")
    
    return scores

def main():
    """Main analysis function."""
    # Find most recent results directory
    results_dir = Path('outputs/parameter_sweep')
    latest_dir = sorted(results_dir.glob('*'))[-1]
    results_path = latest_dir / 'final_results.json'
    
    print(f"Analyzing results from: {latest_dir}")
    print("=" * 50)
    
    # Load and analyze results
    data = load_results(results_path)
    metric_stats = analyze_metrics(data)
    param_df, metric_df = analyze_parameters(data)
    best_configs = analyze_best_configurations(data)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 