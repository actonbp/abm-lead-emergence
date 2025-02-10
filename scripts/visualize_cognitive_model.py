"""
Visualization script for the Cognitive Model.

This script runs the cognitive model and creates visualizations to show:
1. Leadership emergence patterns
2. ILT convergence over time
3. Learning dynamics and schema adaptation
4. ICC analysis of leadership emergence
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.perspectives.cognitive import CognitiveModel, CognitiveParameters
from src.models.base_model import ModelParameters

def run_cognitive_model(n_steps: int = 1000) -> Tuple[CognitiveModel, List[Dict]]:
    """Run the cognitive model and collect step data."""
    # Initialize model with default parameters
    params = CognitiveParameters(
        n_agents=10,
        schema_dimensions=2,
        schema_type="continuous",
        characteristic_distribution="normal",
        ilt_distribution="normal",
        distribution_mean=50,
        distribution_std=20,
        match_algorithm="average",
        base_claim_probability=0.3,
        failure_penalty=5.0,
        ilt_learning_rate=0.3,
        observation_weight=1.5,
        memory_decay=0.2,
        max_memory=15,
        success_boost=15.0,
        random_seed=42
    )
    
    model = CognitiveModel(params)
    step_data = []
    
    # Run model
    for step in range(n_steps):
        step_result = model.step()
        if step == 0:  # Print first step data to understand structure
            print("First step data structure:")
            print(step_result)
        step_data.append(step_result)
    
    return model, step_data

def plot_leadership_emergence(step_data: List[Dict], save_path: str = None):
    """Plot leadership emergence patterns."""
    # Extract leadership claims and grants
    steps = [d['step'] for d in step_data]
    claims = [len(d['claims_made']) for d in step_data]
    grants = [len(d['grants_made']) for d in step_data]
    
    # Calculate moving averages
    window = 50
    claims_ma = pd.Series(claims).rolling(window=window).mean()
    grants_ma = pd.Series(grants).rolling(window=window).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, claims_ma, label='Leadership Claims (MA)', alpha=0.8)
    plt.plot(steps, grants_ma, label='Leadership Grants (MA)', alpha=0.8)
    plt.xlabel('Step')
    plt.ylabel('Number of Claims/Grants')
    plt.title('Leadership Emergence Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_ilt_convergence(model: CognitiveModel, step_data: List[Dict], save_path: str = None):
    """Plot ILT schema convergence over time."""
    # Get final ILT schemas
    final_ilts = np.array([agent.ilt_schema for agent in model.agents])
    
    # Calculate pairwise distances between ILTs
    n_agents = len(model.agents)
    distances = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j:
                distances[i,j] = np.linalg.norm(final_ilts[i] - final_ilts[j])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(distances, cmap='viridis', annot=True, fmt='.2f')
    plt.title('Pairwise Distances Between Agent ILT Schemas')
    plt.xlabel('Agent ID')
    plt.ylabel('Agent ID')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_learning_dynamics(step_data: List[Dict], save_path: str = None):
    """Plot learning dynamics and schema adaptation."""
    # Extract successful interactions
    successful_steps = []
    successful_weights = []
    
    for step in step_data:
        if step['recent_interactions']:
            for interaction in step['recent_interactions']:
                if interaction['success']:
                    successful_steps.append(step['step'])
                    successful_weights.append(interaction['weight'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: Successful interactions over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(successful_steps, successful_weights, alpha=0.5, s=30)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Interaction Weight')
    ax1.set_title('Successful Leadership Interactions')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of interaction weights
    ax2 = fig.add_subplot(gs[1, 0])
    sns.histplot(successful_weights, bins=20, ax=ax2)
    ax2.set_xlabel('Interaction Weight')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Interaction Weights')
    
    # Plot 3: Cumulative successful interactions
    ax3 = fig.add_subplot(gs[1, 1])
    cumulative = np.cumsum([1 for _ in successful_steps])
    ax3.plot(successful_steps, cumulative)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Cumulative Successful Interactions')
    ax3.set_title('Accumulation of Leadership Experience')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def calculate_icc(ratings_matrix):
    """
    Calculate ICC(1) from a matrix of ratings where rows are targets and columns are raters.
    """
    n_targets, n_raters = ratings_matrix.shape
    
    # Calculate means
    grand_mean = np.mean(ratings_matrix)
    target_means = np.mean(ratings_matrix, axis=1)
    
    # Calculate sums of squares
    between_target_ss = n_raters * np.sum((target_means - grand_mean) ** 2)
    total_ss = np.sum((ratings_matrix - grand_mean) ** 2)
    within_target_ss = total_ss - between_target_ss
    
    # Calculate degrees of freedom
    between_target_df = n_targets - 1
    within_target_df = n_targets * (n_raters - 1)
    
    # Calculate mean squares
    between_target_ms = between_target_ss / between_target_df if between_target_df > 0 else 0
    within_target_ms = within_target_ss / within_target_df if within_target_df > 0 else 0
    
    # Calculate ICC(1)
    icc = (between_target_ms - within_target_ms) / (between_target_ms + (n_raters - 1) * within_target_ms)
    
    return float(icc)  # Ensure we return a float

def plot_icc_over_time(step_data: List[Dict], model: CognitiveModel, save_path: str = None):
    """Plot ICC analysis of leadership emergence over time."""
    n_agents = len(model.agents)
    window_size = 50  # Window for calculating ICC
    step_indices = range(0, len(step_data), window_size)
    icc_values = []
    
    for start_idx in step_indices:
        end_idx = min(start_idx + window_size, len(step_data))
        
        # Get the last step in this window to analyze leadership perceptions
        step = step_data[end_idx - 1]
        
        # Create ratings matrix where each row is a target and each column is a rater
        ratings_matrix = np.zeros((n_agents, n_agents))
        
        # First pass: collect all ratings except self-ratings
        for agent_data in step['agents']:
            rater_id = agent_data['id']
            for target_id, rating in agent_data['leadership_perceptions'].items():
                target_id = int(target_id)
                ratings_matrix[target_id, rater_id] = rating
        
        # Second pass: fill diagonal with mean rating received
        for i in range(n_agents):
            # Calculate mean rating received by agent i (excluding self-rating)
            ratings_received = ratings_matrix[i, :]
            mean_rating = np.mean(ratings_received[ratings_received > 0])
            if not np.isnan(mean_rating):
                ratings_matrix[i, i] = mean_rating
            else:
                # If no ratings received yet, use neutral rating
                ratings_matrix[i, i] = 50.0
        
        # Print first window's ratings for debugging
        if start_idx == 0:
            print("\nFirst window ratings matrix:")
            print("Mean rating:", np.mean(ratings_matrix))
            print("Std rating:", np.std(ratings_matrix))
            print("Min rating:", np.min(ratings_matrix))
            print("Max rating:", np.max(ratings_matrix))
            print("\nSample of ratings matrix:")
            print(ratings_matrix[:5, :5])  # Show top-left 5x5 corner
        
        # Calculate ICC for this window
        icc = calculate_icc(ratings_matrix)
        icc_values.append(icc)
    
    # Plot ICC over time
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(step_indices) + window_size/2, icc_values, '-o')
    plt.xlabel('Time Step')
    plt.ylabel('ICC(1)')
    plt.title('Leadership Emergence: Intraclass Correlation Over Time')
    plt.grid(True, alpha=0.3)
    
    # Add a horizontal line at ICC = 0.1 (typical threshold for emergence)
    plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Emergence Threshold (ICC=0.1)')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    """Run model and create visualizations."""
    # Create output directory if it doesn't exist
    output_dir = 'output/cognitive_model'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run model
    print("Running cognitive model simulation...")
    model, step_data = run_cognitive_model(n_steps=1000)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_leadership_emergence(step_data, os.path.join(output_dir, 'leadership_emergence.png'))
    plot_ilt_convergence(model, step_data, os.path.join(output_dir, 'ilt_convergence.png'))
    plot_learning_dynamics(step_data, os.path.join(output_dir, 'learning_dynamics.png'))
    plot_icc_over_time(step_data, model, os.path.join(output_dir, 'icc_analysis.png'))
    print("Visualizations saved to output/cognitive_model/")

if __name__ == "__main__":
    main() 