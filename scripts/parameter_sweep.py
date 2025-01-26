"""
Parameter sweep script for exploring leadership emergence patterns.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import json
from datetime import datetime
from itertools import product
import pandas as pd
from typing import Dict, List, Tuple, Any

from src.models.base_model import BaseLeadershipModel

def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj

def calculate_metrics(history: List[Dict]) -> Dict:
    """Calculate various metrics from simulation history."""
    # Time to first leader (first agent to reach score > 75)
    time_to_first_leader = None
    for step, state in enumerate(history):
        scores = [agent['lead_score'] for agent in state['agents']]
        if any(score > 75 for score in scores):
            time_to_first_leader = step
            break
    
    # Get final state
    final_state = history[-1]
    final_scores = [agent['lead_score'] for agent in final_state['agents']]
    
    # Number of leaders at end (score > 75)
    num_leaders = sum(1 for score in final_scores if score > 75)
    
    # Leadership concentration (std dev of scores)
    score_std = np.std(final_scores)
    
    # Total successful claims and grants
    total_claims = 0
    successful_claims = 0
    for state in history:
        for agent in state['agents']:
            if agent['last_interaction']['role'] == 'claimer':
                if agent['last_interaction']['claimed']:
                    total_claims += 1
                    if agent['last_interaction']['granted']:
                        successful_claims += 1
    
    # Average grant rate
    grant_rate = successful_claims / total_claims if total_claims > 0 else 0
    
    return {
        'time_to_first_leader': time_to_first_leader,
        'num_leaders': num_leaders,
        'score_std': score_std,
        'total_claims': total_claims,
        'successful_claims': successful_claims,
        'grant_rate': grant_rate
    }

def run_parameter_sweep(
    output_dir: str = "outputs/parameter_sweep",
    n_steps: int = 100,
    n_replications: int = 5
):
    """Run parameter sweep and save results."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define parameter grid
    param_grid = {
        "n_agents": [4, 6, 8],
        "claim_multiplier": [0.5, 0.7, 0.9],
        "grant_multiplier": [0.4, 0.6, 0.8],
        "success_boost": [3.0, 5.0, 7.0],
        "failure_penalty": [2.0, 3.0, 4.0],
        "grant_penalty": [1.0, 2.0, 3.0]
    }
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    # Store all results
    all_results = []
    
    # Run simulations
    total_sims = len(combinations) * n_replications
    sim_count = 0
    
    for combo in combinations:
        params = dict(zip(param_names, combo))
        
        for rep in range(n_replications):
            sim_count += 1
            print(f"\nRunning simulation {sim_count}/{total_sims}")
            print(f"Parameters: {params}")
            print(f"Replication: {rep + 1}/{n_replications}")
            
            # Run simulation
            model = BaseLeadershipModel(
                params=params,
                random_seed=rep
            )
            
            history = []
            for _ in range(n_steps):
                state = model.step()
                history.append(state)
            
            # Calculate metrics
            metrics = calculate_metrics(history)
            
            # Store results
            result = {
                'parameters': params,
                'replication': rep,
                'metrics': metrics,
                'final_state': model.get_state()
            }
            all_results.append(result)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"sweep_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    # Create summary DataFrame
    summary_data = []
    for result in all_results:
        row = {}
        row.update(result['parameters'])
        row.update({
            'replication': result['replication'],
            **{f"metric_{k}": v for k, v in result['metrics'].items()}
        })
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Calculate means and std devs across replications
    summary_stats = df.groupby(param_names).agg({
        'metric_time_to_first_leader': ['mean', 'std'],
        'metric_num_leaders': ['mean', 'std'],
        'metric_score_std': ['mean', 'std'],
        'metric_grant_rate': ['mean', 'std']
    }).round(2)
    
    # Save summary stats
    summary_file = output_dir / f"sweep_summary_{timestamp}.csv"
    summary_stats.to_csv(summary_file)
    
    print(f"\nResults saved to {output_dir}")
    return df, summary_stats


if __name__ == "__main__":
    df, summary = run_parameter_sweep()
    
    # Print some interesting findings
    print("\nTop 5 parameter combinations for fast leadership emergence:")
    fastest = df['metric_time_to_first_leader'] >= 0  # Filter out no-leader cases
    fastest_combos = df[fastest].groupby(list(df.columns[:6]))['metric_time_to_first_leader'].mean()
    print(fastest_combos.sort_values()[:5])
    
    print("\nTop 5 parameter combinations for multiple leaders:")
    most_leaders = df.groupby(list(df.columns[:6]))['metric_num_leaders'].mean()
    print(most_leaders.sort_values(ascending=False)[:5]) 