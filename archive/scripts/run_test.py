import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from scipy.stats import entropy
from sklearn.preprocessing import normalize

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.base_model import BaseLeadershipModel

def plot_identities(history, save_path):
    """Plot leader and follower identities over time."""
    plt.figure(figsize=(12, 6))
    
    # Plot leader identities
    leader_ids = np.array(history['leader_identities'])
    plt.subplot(1, 2, 1)
    for i in range(leader_ids.shape[1]):
        plt.plot(leader_ids[:, i], alpha=0.5, label=f'Agent {i}')
    plt.title('Leader Identities Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Leader Identity')
    plt.grid(True)
    
    # Plot follower identities
    follower_ids = np.array(history['follower_identities'])
    plt.subplot(1, 2, 2)
    for i in range(follower_ids.shape[1]):
        plt.plot(follower_ids[:, i], alpha=0.5, label=f'Agent {i}')
    plt.title('Follower Identities Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Follower Identity')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'identities.png'))
    plt.close()

def plot_team_distributions(team_stats: List[Dict], save_path: Path):
    """Plot distributions of outcomes across teams."""
    plt.figure(figsize=(15, 10))
    
    # Leader Identity Distribution
    plt.subplot(2, 2, 1)
    leader_means = [stats['leader_mean'] for stats in team_stats]
    leader_stds = [stats['leader_std'] for stats in team_stats]
    sns.histplot(leader_means, bins=20)
    plt.title('Distribution of Team Leader Identity Means')
    plt.xlabel('Leader Identity Mean')
    
    # Follower Identity Distribution
    plt.subplot(2, 2, 2)
    follower_means = [stats['follower_mean'] for stats in team_stats]
    follower_stds = [stats['follower_std'] for stats in team_stats]
    sns.histplot(follower_means, bins=20)
    plt.title('Distribution of Team Follower Identity Means')
    plt.xlabel('Follower Identity Mean')
    
    # Network Metrics Distribution
    plt.subplot(2, 2, 3)
    densities = [stats['density'] for stats in team_stats]
    sns.histplot(densities, bins=20)
    plt.title('Distribution of Network Densities')
    plt.xlabel('Network Density')
    
    plt.subplot(2, 2, 4)
    centralizations = [stats['centralization'] for stats in team_stats]
    sns.histplot(centralizations, bins=20)
    plt.title('Distribution of Network Centralizations')
    plt.xlabel('Network Centralization')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'team_distributions.png'))
    plt.close()

def plot_individual_distributions(all_individuals: Dict, save_path: Path):
    """Plot distributions of individual outcomes across all teams."""
    plt.figure(figsize=(15, 10))
    
    # Leader Identity Distribution
    plt.subplot(2, 1, 1)
    sns.histplot(all_individuals['leader_identities'], bins=30)
    plt.axvline(x=50, color='r', linestyle='--', alpha=0.5, label='Initial Value')
    plt.title('Distribution of Individual Leader Identities')
    plt.xlabel('Leader Identity')
    plt.ylabel('Count')
    
    # Follower Identity Distribution
    plt.subplot(2, 1, 2)
    sns.histplot(all_individuals['follower_identities'], bins=30)
    plt.axvline(x=50, color='r', linestyle='--', alpha=0.5, label='Initial Value')
    plt.title('Distribution of Individual Follower Identities')
    plt.xlabel('Follower Identity')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'individual_distributions.png'))
    plt.close()

def run_team() -> Dict:
    """Run a single team simulation and return statistics."""
    model = BaseLeadershipModel(config_path='config/test_minimal.yaml')
    results = model.run(n_steps=100)
    
    final_leader_ids = np.array(model.history['leader_identities'][-1])
    final_follower_ids = np.array(model.history['follower_identities'][-1])
    
    return {
        'leader_identities': final_leader_ids,
        'follower_identities': final_follower_ids,
        'density': model.history['density'][-1],
        'centralization': model.history['centralization'][-1],
        'history': model.history  # Add full history
    }

def analyze_perceptions(all_perceptions: np.ndarray) -> Dict:
    """Analyze the final perception matrix across all teams."""
    # Flatten all non-self perceptions
    flat_perceptions = []
    n_agents = all_perceptions.shape[1]
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j:  # Exclude self-perceptions
                flat_perceptions.extend(all_perceptions[:, i, j])
    
    flat_perceptions = np.array(flat_perceptions)
    return {
        'mean': np.mean(flat_perceptions),
        'std': np.std(flat_perceptions),
        'min': np.min(flat_perceptions),
        '25%': np.percentile(flat_perceptions, 25),
        'median': np.median(flat_perceptions),
        '75%': np.percentile(flat_perceptions, 75),
        'max': np.max(flat_perceptions)
    }

def plot_perception_heatmap(perceptions: np.ndarray, save_path: Path):
    """Plot heatmap of final leadership perceptions."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(perceptions, annot=True, fmt='.1f', cmap='RdYlBu_r',
                vmin=0, vmax=100, center=50)
    plt.title('Final Leadership Perceptions\n(How much each agent perceives others as leaders)')
    plt.xlabel('Target Agent')
    plt.ylabel('Perceiving Agent')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'perception_heatmap.png'))
    plt.close()

def calculate_kendall_w(perception_matrix: np.ndarray) -> float:
    """Calculate Kendall's W (coefficient of concordance) for leadership rankings.
    
    Higher values indicate stronger agreement among agents about leadership rankings.
    Range: 0 (no agreement) to 1 (perfect agreement)
    """
    # Convert perceptions to rankings for each agent
    rankings = np.zeros_like(perception_matrix)
    for i in range(perception_matrix.shape[0]):
        rankings[i] = stats.rankdata(-perception_matrix[i])  # Negative to rank highest perceptions first
    
    # Calculate Kendall's W
    n = rankings.shape[0]  # number of raters
    k = rankings.shape[1]  # number of items being ranked
    
    # Calculate mean of each column (item)
    mean_ranks = np.mean(rankings, axis=0)
    overall_mean = np.mean(mean_ranks)
    
    # Calculate S (sum of squared deviations)
    S = np.sum((mean_ranks - overall_mean) ** 2)
    
    # Calculate maximum possible S
    max_S = (n ** 2 * (k ** 3 - k)) / 12
    
    # Kendall's W
    W = S / max_S if max_S > 0 else 0
    return W

def calculate_krippendorff_alpha(perception_matrix: np.ndarray) -> float:
    """Calculate Krippendorff's alpha for leadership perceptions.
    
    This is a simplified version for interval data.
    Range: < 0 (poor agreement) to 1 (perfect agreement)
    """
    n_raters, n_items = perception_matrix.shape
    
    # Calculate observed disagreement
    Do = 0
    n_pairs = 0
    for i in range(n_items):
        for j in range(i + 1, n_items):
            for r1 in range(n_raters):
                for r2 in range(r1 + 1, n_raters):
                    diff = perception_matrix[r1, i] - perception_matrix[r2, j]
                    Do += diff ** 2
                    n_pairs += 1
    
    Do = Do / n_pairs if n_pairs > 0 else 0
    
    # Calculate expected disagreement
    all_values = perception_matrix.flatten()
    De = np.var(all_values) if len(all_values) > 1 else 0
    
    # Calculate alpha
    alpha = 1 - (Do / De) if De > 0 else 0
    return alpha

def calculate_entropy_metrics(perception_matrix: np.ndarray) -> Dict[str, float]:
    """Calculate entropy-based leadership concentration metrics.
    
    Returns:
        Dict with entropy and normalized entropy values.
        Lower values indicate more concentrated leadership (stronger consensus).
    """
    # Normalize perceptions to sum to 1 (like probabilities)
    normalized_perceptions = normalize(perception_matrix, norm='l1', axis=1)
    
    # Calculate entropy for each rater's perceptions
    entropies = np.array([entropy(row) for row in normalized_perceptions])
    
    # Calculate normalized entropy (divided by log(n) for n items)
    max_entropy = np.log(perception_matrix.shape[1])
    normalized_entropies = entropies / max_entropy if max_entropy > 0 else entropies
    
    return {
        'mean_entropy': np.mean(entropies),
        'normalized_entropy': np.mean(normalized_entropies)
    }

def identify_top_leaders(perception_matrix: np.ndarray, threshold: float = 0.8) -> Dict[str, float]:
    """Analyze top leader identification metrics.
    
    Args:
        perception_matrix: Matrix of leadership perceptions
        threshold: Threshold for considering someone a "top" leader (percentile)
    
    Returns:
        Dict with various top leader metrics
    """
    n_raters, n_items = perception_matrix.shape
    
    # Calculate threshold value for "top" leader
    threshold_value = np.percentile(perception_matrix, threshold * 100)
    
    # Count how many agents are considered top leaders by each rater
    top_leaders_per_rater = np.sum(perception_matrix >= threshold_value, axis=1)
    
    # Calculate agreement on top leader
    top_leader_indices = np.argmax(perception_matrix, axis=1)
    unique_top_leaders, counts = np.unique(top_leader_indices, return_counts=True)
    max_agreement = np.max(counts) / n_raters if len(counts) > 0 else 0
    
    return {
        'n_top_leaders': len(unique_top_leaders),
        'max_agreement_ratio': max_agreement,
        'mean_top_leaders_per_rater': np.mean(top_leaders_per_rater)
    }

def calculate_perception_agreement(perception_matrix: np.ndarray) -> float:
    """Calculate how much agents agree about who the leaders are.
    
    Higher values mean agents have similar perceptions about who is a leader.
    Lower values mean agents disagree about who is a leader.
    """
    # For each potential leader (column), calculate variance in how they're perceived
    perception_variances = np.var(perception_matrix, axis=0)
    # Low variance means high agreement, so invert
    agreement = 1 / (1 + perception_variances)
    # Return mean agreement across all agents
    return np.mean(agreement)

def analyze_emergence(history: Dict) -> Dict:
    """Analyze leadership emergence over time using multiple metrics."""
    perception_matrices = np.array(history['leadership_perceptions'])
    n_timesteps = len(perception_matrices)
    
    # Initialize metric arrays
    agreements = []
    kendall_ws = []
    krippendorff_alphas = []
    entropies = []
    normalized_entropies = []
    top_leader_metrics = []
    
    # Calculate metrics at each timestep
    for t in range(n_timesteps):
        matrix = perception_matrices[t]
        
        # Basic agreement
        agreement = calculate_perception_agreement(matrix)
        agreements.append(agreement)
        
        # Kendall's W
        kendall_w = calculate_kendall_w(matrix)
        kendall_ws.append(kendall_w)
        
        # Krippendorff's alpha
        alpha = calculate_krippendorff_alpha(matrix)
        krippendorff_alphas.append(alpha)
        
        # Entropy metrics
        entropy_metrics = calculate_entropy_metrics(matrix)
        entropies.append(entropy_metrics['mean_entropy'])
        normalized_entropies.append(entropy_metrics['normalized_entropy'])
        
        # Top leader metrics
        top_metrics = identify_top_leaders(matrix)
        top_leader_metrics.append(top_metrics)
    
    # Calculate early and late metrics
    early_slice = slice(0, 10)
    late_slice = slice(-10, None)
    
    return {
        'agreements': agreements,
        'early_agreement': np.mean(agreements[early_slice]),
        'late_agreement': np.mean(agreements[late_slice]),
        'agreement_change': np.mean(agreements[late_slice]) - np.mean(agreements[early_slice]),
        
        'kendall_ws': kendall_ws,
        'early_kendall_w': np.mean(kendall_ws[early_slice]),
        'late_kendall_w': np.mean(kendall_ws[late_slice]),
        'kendall_w_change': np.mean(kendall_ws[late_slice]) - np.mean(kendall_ws[early_slice]),
        
        'krippendorff_alphas': krippendorff_alphas,
        'early_alpha': np.mean(krippendorff_alphas[early_slice]),
        'late_alpha': np.mean(krippendorff_alphas[late_slice]),
        'alpha_change': np.mean(krippendorff_alphas[late_slice]) - np.mean(krippendorff_alphas[early_slice]),
        
        'entropies': entropies,
        'early_entropy': np.mean(entropies[early_slice]),
        'late_entropy': np.mean(entropies[late_slice]),
        'entropy_change': np.mean(entropies[late_slice]) - np.mean(entropies[early_slice]),
        
        'normalized_entropies': normalized_entropies,
        'early_norm_entropy': np.mean(normalized_entropies[early_slice]),
        'late_norm_entropy': np.mean(normalized_entropies[late_slice]),
        'norm_entropy_change': np.mean(normalized_entropies[late_slice]) - np.mean(normalized_entropies[early_slice]),
        
        'top_leader_metrics': top_leader_metrics,
        'early_top_leaders': np.mean([m['n_top_leaders'] for m in top_leader_metrics[:10]]),
        'late_top_leaders': np.mean([m['n_top_leaders'] for m in top_leader_metrics[-10:]]),
        'early_max_agreement': np.mean([m['max_agreement_ratio'] for m in top_leader_metrics[:10]]),
        'late_max_agreement': np.mean([m['max_agreement_ratio'] for m in top_leader_metrics[-10:]])
    }

def plot_emergence(emergence_data: List[Dict], save_path: Path):
    """Plot emergence metrics over time."""
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Leadership Emergence Metrics Over Time', fontsize=14)
    
    # Plot 1: Basic Agreement and Kendall's W
    ax = axes[0, 0]
    for team_data in emergence_data:
        ax.plot(team_data['agreements'], alpha=0.1, color='blue')
        ax.plot(team_data['kendall_ws'], alpha=0.1, color='red')
    
    # Plot mean trajectories
    mean_agreements = np.mean([d['agreements'] for d in emergence_data], axis=0)
    mean_kendalls = np.mean([d['kendall_ws'] for d in emergence_data], axis=0)
    ax.plot(mean_agreements, color='blue', linewidth=2, label='Mean Agreement')
    ax.plot(mean_kendalls, color='red', linewidth=2, label='Mean Kendall\'s W')
    ax.set_title('Agreement Metrics')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Metric Value')
    ax.grid(True)
    ax.legend()
    
    # Plot 2: Krippendorff's Alpha
    ax = axes[0, 1]
    for team_data in emergence_data:
        ax.plot(team_data['krippendorff_alphas'], alpha=0.1, color='green')
    
    mean_alphas = np.mean([d['krippendorff_alphas'] for d in emergence_data], axis=0)
    ax.plot(mean_alphas, color='green', linewidth=2, label='Mean Alpha')
    ax.set_title('Krippendorff\'s Alpha')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Alpha Value')
    ax.grid(True)
    ax.legend()
    
    # Plot 3: Entropy Metrics
    ax = axes[1, 0]
    for team_data in emergence_data:
        ax.plot(team_data['normalized_entropies'], alpha=0.1, color='purple')
    
    mean_entropies = np.mean([d['normalized_entropies'] for d in emergence_data], axis=0)
    ax.plot(mean_entropies, color='purple', linewidth=2, label='Mean Normalized Entropy')
    ax.set_title('Leadership Distribution Entropy\n(Lower = More Concentrated Leadership)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Normalized Entropy')
    ax.grid(True)
    ax.legend()
    
    # Plot 4: Top Leader Metrics
    ax = axes[1, 1]
    mean_n_leaders = np.mean([[m['n_top_leaders'] for m in d['top_leader_metrics']] 
                            for d in emergence_data], axis=0)
    mean_max_agreement = np.mean([[m['max_agreement_ratio'] for m in d['top_leader_metrics']] 
                                for d in emergence_data], axis=0)
    
    ax.plot(mean_n_leaders, color='orange', linewidth=2, label='Avg # Top Leaders')
    ax.plot(mean_max_agreement, color='brown', linewidth=2, label='Max Agreement Ratio')
    ax.set_title('Top Leader Metrics')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Metric Value')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'emergence.png'))
    plt.close()

def main():
    # Create output directory
    output_dir = Path('outputs/test_run')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run multiple teams
    n_teams = 50
    print(f"\nRunning {n_teams} teams...")
    team_stats = []
    emergence_data = []
    
    # Collect all individual outcomes
    all_leader_ids = []
    all_follower_ids = []
    all_final_perceptions = []
    
    for i in range(n_teams):
        if i % 10 == 0:
            print(f"Running team {i+1}/{n_teams}")
        results = run_team()
        team_stats.append(results)
        
        # Analyze emergence for this team
        emergence_stats = analyze_emergence(results['history'])
        emergence_data.append(emergence_stats)
        
        all_leader_ids.extend(results['leader_identities'])
        all_follower_ids.extend(results['follower_identities'])
        final_perceptions = results['history']['leadership_perceptions'][-1]
        all_final_perceptions.append(final_perceptions)
    
    # Print emergence statistics
    print("\nEmergence Statistics:")
    
    print("\nBasic Agreement Metrics:")
    early_agreements = [d['early_agreement'] for d in emergence_data]
    late_agreements = [d['late_agreement'] for d in emergence_data]
    agreement_changes = [d['agreement_change'] for d in emergence_data]
    
    print("Early Agreement (first 10 steps):")
    print(f"  Mean: {np.mean(early_agreements):.3f}")
    print(f"  Std:  {np.std(early_agreements):.3f}")
    
    print("\nLate Agreement (last 10 steps):")
    print(f"  Mean: {np.mean(late_agreements):.3f}")
    print(f"  Std:  {np.std(late_agreements):.3f}")
    
    print("\nAgreement Change:")
    print(f"  Mean: {np.mean(agreement_changes):.3f}")
    print(f"  Std:  {np.std(agreement_changes):.3f}")
    
    print("\nKendall's W Statistics:")
    early_ws = [d['early_kendall_w'] for d in emergence_data]
    late_ws = [d['late_kendall_w'] for d in emergence_data]
    w_changes = [d['kendall_w_change'] for d in emergence_data]
    
    print("Early Kendall's W:")
    print(f"  Mean: {np.mean(early_ws):.3f}")
    print(f"  Std:  {np.std(early_ws):.3f}")
    
    print("\nLate Kendall's W:")
    print(f"  Mean: {np.mean(late_ws):.3f}")
    print(f"  Std:  {np.std(late_ws):.3f}")
    
    print("\nKrippendorff's Alpha Statistics:")
    early_alphas = [d['early_alpha'] for d in emergence_data]
    late_alphas = [d['late_alpha'] for d in emergence_data]
    alpha_changes = [d['alpha_change'] for d in emergence_data]
    
    print("Early Alpha:")
    print(f"  Mean: {np.mean(early_alphas):.3f}")
    print(f"  Std:  {np.std(early_alphas):.3f}")
    
    print("\nLate Alpha:")
    print(f"  Mean: {np.mean(late_alphas):.3f}")
    print(f"  Std:  {np.std(late_alphas):.3f}")
    
    print("\nEntropy Statistics:")
    early_entropies = [d['early_norm_entropy'] for d in emergence_data]
    late_entropies = [d['late_norm_entropy'] for d in emergence_data]
    entropy_changes = [d['norm_entropy_change'] for d in emergence_data]
    
    print("Early Normalized Entropy:")
    print(f"  Mean: {np.mean(early_entropies):.3f}")
    print(f"  Std:  {np.std(early_entropies):.3f}")
    
    print("\nLate Normalized Entropy:")
    print(f"  Mean: {np.mean(late_entropies):.3f}")
    print(f"  Std:  {np.std(late_entropies):.3f}")
    
    print("\nTop Leader Statistics:")
    early_top = [d['early_top_leaders'] for d in emergence_data]
    late_top = [d['late_top_leaders'] for d in emergence_data]
    early_max = [d['early_max_agreement'] for d in emergence_data]
    late_max = [d['late_max_agreement'] for d in emergence_data]
    
    print("Early Number of Top Leaders:")
    print(f"  Mean: {np.mean(early_top):.3f}")
    print(f"  Std:  {np.std(early_top):.3f}")
    
    print("\nLate Number of Top Leaders:")
    print(f"  Mean: {np.mean(late_top):.3f}")
    print(f"  Std:  {np.std(late_top):.3f}")
    
    print("\nEarly Max Agreement Ratio:")
    print(f"  Mean: {np.mean(early_max):.3f}")
    print(f"  Std:  {np.std(early_max):.3f}")
    
    print("\nLate Max Agreement Ratio:")
    print(f"  Mean: {np.mean(late_max):.3f}")
    print(f"  Std:  {np.std(late_max):.3f}")
    
    # Plot emergence over time
    plot_emergence(emergence_data, output_dir)
    
    # Convert to numpy arrays and calculate remaining statistics
    all_leader_ids = np.array(all_leader_ids)
    all_follower_ids = np.array(all_follower_ids)
    all_final_perceptions = np.array(all_final_perceptions)
    
    # Calculate individual-level statistics
    print("\nIndividual-Level Statistics:")
    
    print("\nLeader Identities:")
    print(f"  Mean:  {np.mean(all_leader_ids):.2f}")
    print(f"  Std:   {np.std(all_leader_ids):.2f}")
    print(f"  Min:   {np.min(all_leader_ids):.2f}")
    print(f"  25%:   {np.percentile(all_leader_ids, 25):.2f}")
    print(f"  Median:{np.median(all_leader_ids):.2f}")
    print(f"  75%:   {np.percentile(all_leader_ids, 75):.2f}")
    print(f"  Max:   {np.max(all_leader_ids):.2f}")
    
    print("\nFollower Identities:")
    print(f"  Mean:  {np.mean(all_follower_ids):.2f}")
    print(f"  Std:   {np.std(all_follower_ids):.2f}")
    print(f"  Min:   {np.min(all_follower_ids):.2f}")
    print(f"  25%:   {np.percentile(all_follower_ids, 25):.2f}")
    print(f"  Median:{np.median(all_follower_ids):.2f}")
    print(f"  75%:   {np.percentile(all_follower_ids, 75):.2f}")
    print(f"  Max:   {np.max(all_follower_ids):.2f}")
    
    print("\nLeadership Perceptions:")
    perception_stats = analyze_perceptions(all_final_perceptions)
    print(f"  Mean:  {perception_stats['mean']:.2f}")
    print(f"  Std:   {perception_stats['std']:.2f}")
    print(f"  Min:   {perception_stats['min']:.2f}")
    print(f"  25%:   {perception_stats['25%']:.2f}")
    print(f"  Median:{perception_stats['median']:.2f}")
    print(f"  75%:   {perception_stats['75%']:.2f}")
    print(f"  Max:   {perception_stats['max']:.2f}")
    
    # Plot distributions and heatmap
    all_individuals = {
        'leader_identities': all_leader_ids,
        'follower_identities': all_follower_ids
    }
    plot_individual_distributions(all_individuals, output_dir)
    
    # Plot average perception heatmap
    avg_perceptions = np.mean(all_final_perceptions, axis=0)
    plot_perception_heatmap(avg_perceptions, output_dir)
    
    # Network statistics
    print("\nTeam-Level Network Metrics:")
    densities = [stats['density'] for stats in team_stats]
    centralizations = [stats['centralization'] for stats in team_stats]
    print(f"  Mean Density: {np.mean(densities):.2f}")
    print(f"  Mean Centralization: {np.mean(centralizations):.2f}")

if __name__ == "__main__":
    main() 