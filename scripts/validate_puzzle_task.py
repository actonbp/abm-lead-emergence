#!/usr/bin/env python3

"""
Validation script for puzzle task implementation.
Compares base model (no emergence) with social interactionist model (identity-based emergence)
to verify that task mechanics reflect expected leadership patterns.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.base_model import BaseLeadershipModel, ModelParameters
from src.models.perspectives import InteractionistModel, InteractionistParameters
from src.models.tasks.puzzle_task import PuzzleTask

def run_puzzle_validation(n_steps=100):
    """
    Run simulations with both base and interactionist models to verify mechanics.
    Tracks detailed metrics about leadership emergence and task performance.
    """
    # Common parameters
    base_params = {
        'n_agents': 6,
        'match_threshold': 0.41,
        'base_claim_probability': 0.7,
        'schema_dimensions': 2
    }
    
    # Create models
    base_model = BaseLeadershipModel(ModelParameters(**base_params))
    interactionist_model = InteractionistModel(InteractionistParameters(
        **base_params,
        identity_growth_rate=0.15,     # Faster growth rate
        max_identity_weight=0.9,       # Higher max influence
        identity_growth_exponent=2.0,  # Exponential growth
        identity_update_rate=0.3,      # Faster identity updates
        perception_update_rate=0.3,    # Faster perception updates
        success_boost=10.0,           # Stronger success reinforcement
        failure_penalty=5.0           # Stronger failure penalty
    ))
    
    # Create tasks for each model
    base_task = PuzzleTask(
        num_agents=base_params['n_agents'],
        puzzle_size=10,
        num_stages=3
    )
    interactionist_task = PuzzleTask(
        num_agents=base_params['n_agents'],
        puzzle_size=10,
        num_stages=3
    )
    
    base_model.set_task(base_task)
    interactionist_model.set_task(interactionist_task)
    
    # Track metrics for both models
    logs = {
        'base': {
            'stage': [], 
            'resource': [], 
            'time': [], 
            'interactions': [],
            'leadership': [],  # Track leadership metrics
            'task': []        # Track task performance metrics
        },
        'interactionist': {
            'stage': [], 
            'resource': [], 
            'time': [], 
            'interactions': [],
            'leadership': [],  # Track leadership metrics
            'task': []        # Track task performance metrics
        }
    }
    
    print("Starting puzzle validation run...")
    print("\nInitial States:")
    print("\nBase Model:")
    print_puzzle_state(base_task)
    print("\nInteractionist Model:")
    print_puzzle_state(interactionist_task)
    
    # Run simulations
    for step in range(n_steps):
        # Base model step
        base_state = base_model.step()
        
        # Track stage and quality metrics
        logs['base']['stage'].append({
            'current_stage': base_task.current_stage,
            'stage_quality': base_task._evaluate_stage_understanding(),
            'overall_quality': base_task._evaluate_understanding()
        })
        
        # Track resource usage
        logs['base']['resource'].append({
            'resources': base_task.resources,
            'time_spent': base_task.time_spent
        })
        
        # Track interactions
        logs['base']['interactions'].append(base_state.get('recent_interactions', []))
        
        # Track leadership metrics
        leadership_metrics = calculate_leadership_metrics(base_state)
        logs['base']['leadership'].append(leadership_metrics)
        
        # Track task performance metrics
        task_metrics = calculate_task_metrics(base_task, base_state)
        logs['base']['task'].append(task_metrics)
        
        # Interactionist model step
        inter_state = interactionist_model.step()
        
        # Track stage and quality metrics
        logs['interactionist']['stage'].append({
            'current_stage': interactionist_task.current_stage,
            'stage_quality': interactionist_task._evaluate_stage_understanding(),
            'overall_quality': interactionist_task._evaluate_understanding()
        })
        
        # Track resource usage
        logs['interactionist']['resource'].append({
            'resources': interactionist_task.resources,
            'time_spent': interactionist_task.time_spent
        })
        
        # Track interactions
        logs['interactionist']['interactions'].append(inter_state.get('recent_interactions', []))
        
        # Track leadership metrics
        leadership_metrics = calculate_leadership_metrics(inter_state)
        logs['interactionist']['leadership'].append(leadership_metrics)
        
        # Track task performance metrics
        task_metrics = calculate_task_metrics(interactionist_task, inter_state)
        logs['interactionist']['task'].append(task_metrics)
        
        # Print detailed info every 10 steps
        if step % 10 == 0:
            print(f"\nStep {step}:")
            print("\nBase Model:")
            print_puzzle_state(base_task)
            print_recent_interactions(base_state)
            print_leadership_metrics(leadership_metrics)
            print("\nInteractionist Model:")
            print_puzzle_state(interactionist_task)
            print_recent_interactions(inter_state)
            print_leadership_metrics(leadership_metrics)
    
    # Print final summary
    print("\nFinal Summary:")
    print_final_summary(logs)
    
    # Plot comparison results
    plot_validation_results(logs)
    
    return logs

def calculate_leadership_metrics(state):
    """Calculate detailed metrics about leadership emergence."""
    n_agents = len(state['agents'])
    
    # Calculate leadership perception matrix
    perception_matrix = np.zeros((n_agents, n_agents))
    for i, agent in enumerate(state['agents']):
        for j_str, value in agent['leadership_perceptions'].items():
            j = int(j_str)
            perception_matrix[i,j] = value / 100.0  # Scale to [0,1]
    
    # Calculate mean incoming leadership for each agent
    mean_incoming = np.mean(perception_matrix, axis=0)
    
    # Calculate leadership metrics
    metrics = {
        # Hierarchy strength (Gini coefficient of mean incoming leadership)
        'hierarchy_strength': calculate_gini(mean_incoming),
        
        # Leadership concentration (std dev of mean incoming leadership)
        'leadership_concentration': np.std(mean_incoming),
        
        # Agreement on leadership using ICC-like measure
        # High when agents agree about each leader (low within variance)
        # AND there's clear differentiation (high between variance)
        'perception_agreement': calculate_leadership_icc(perception_matrix),
        
        # Identify emergent leaders (agents with high mean incoming leadership)
        'emergent_leaders': [
            i for i, score in enumerate(mean_incoming)
            if score > np.mean(mean_incoming) + np.std(mean_incoming)
        ],
        
        # Mean leader and follower identities
        'mean_leader_identity': np.mean([
            agent['leader_identity'] for agent in state['agents']
        ]),
        'mean_follower_identity': np.mean([
            agent['follower_identity'] for agent in state['agents']
        ])
    }
    
    return metrics

def calculate_leadership_icc(perception_matrix):
    """Calculate ICC-like measure for leadership perceptions.
    
    This measures both:
    1. Agreement about each agent's leadership (low within-agent variance)
    2. Differentiation between agents (high between-agent variance)
    
    Returns:
        float: ICC score between 0 and 1
        - 0: No agreement/differentiation
        - 1: Perfect agreement and clear differentiation
    """
    n_agents = perception_matrix.shape[0]
    
    # Calculate variance components
    grand_mean = np.mean(perception_matrix)
    agent_means = np.mean(perception_matrix, axis=0)  # Mean rating received by each agent
    
    # Between-agent variance (variance of agent means)
    between_var = np.sum((agent_means - grand_mean) ** 2) / (n_agents - 1)
    
    # Within-agent variance (average variance of ratings for each agent)
    within_vars = []
    for j in range(n_agents):
        # Get all ratings for agent j (excluding self-rating)
        ratings = [perception_matrix[i,j] for i in range(n_agents) if i != j]
        if len(ratings) > 1:  # Need at least 2 ratings to calculate variance
            within_vars.append(np.var(ratings))
    
    if not within_vars:  # No valid variances
        return 0.0
        
    within_var = np.mean(within_vars)
    
    # Calculate ICC
    # ICC = between / (between + within)
    # This will be high when:
    # - Agents agree about each leader (low within_var)
    # - Clear differentiation between agents (high between_var)
    if between_var + within_var == 0:  # Avoid division by zero
        return 0.0
        
    return between_var / (between_var + within_var)

def calculate_task_metrics(task, state):
    """Calculate detailed metrics about task performance."""
    metrics = {
        # Stage completion
        'stage_completion': [
            len([p for p in task.shared_pieces if p[0] == stage]) / (task.puzzle_size // task.num_stages)
            for stage in range(task.num_stages)
        ],
        
        # Quality by stage
        'stage_qualities': [
            task._evaluate_stage_understanding() if stage == task.current_stage
            else task._evaluate_understanding()
            for stage in range(task.num_stages)
        ],
        
        # Resource efficiency
        'resource_efficiency': task.resources / task.initial_resources,
        
        # Time efficiency
        'time_efficiency': 1.0 - (task.time_spent / (task.num_stages * task.puzzle_size)),
        
        # Current stage leaders
        'stage_leaders': task.stage_leaders.copy()
    }
    
    return metrics

def calculate_gini(values):
    """Calculate Gini coefficient as a measure of inequality."""
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return np.sum((2 * index - n - 1) * values) / (n * np.sum(values))

def print_leadership_metrics(metrics):
    """Print leadership emergence metrics."""
    print("\nLeadership Metrics:")
    print(f"Hierarchy Strength: {metrics['hierarchy_strength']:.3f}")
    print(f"Leadership Concentration: {metrics['leadership_concentration']:.3f}")
    print(f"Perception Agreement: {metrics['perception_agreement']:.3f}")
    print(f"Emergent Leaders: {metrics['emergent_leaders']}")
    print(f"Mean Leader Identity: {metrics['mean_leader_identity']:.1f}")
    print(f"Mean Follower Identity: {metrics['mean_follower_identity']:.1f}")

def print_final_summary(logs):
    """Print final summary comparing both models."""
    print("\nBase Model Summary:")
    base_final = {
        'stages_completed': logs['base']['stage'][-1]['current_stage'],
        'overall_quality': logs['base']['stage'][-1]['overall_quality'],
        'resources_remaining': logs['base']['resource'][-1]['resources'],
        'time_spent': logs['base']['resource'][-1]['time_spent'],
        'hierarchy_strength': logs['base']['leadership'][-1]['hierarchy_strength'],
        'perception_agreement': logs['base']['leadership'][-1]['perception_agreement']
    }
    
    print("\nInteractionist Model Summary:")
    inter_final = {
        'stages_completed': logs['interactionist']['stage'][-1]['current_stage'],
        'overall_quality': logs['interactionist']['stage'][-1]['overall_quality'],
        'resources_remaining': logs['interactionist']['resource'][-1]['resources'],
        'time_spent': logs['interactionist']['resource'][-1]['time_spent'],
        'hierarchy_strength': logs['interactionist']['leadership'][-1]['hierarchy_strength'],
        'perception_agreement': logs['interactionist']['leadership'][-1]['perception_agreement']
    }
    
    print("\nComparison:")
    for metric in base_final:
        print(f"{metric}:")
        print(f"  Base: {base_final[metric]:.3f}")
        print(f"  Interactionist: {inter_final[metric]:.3f}")
        print(f"  Difference: {inter_final[metric] - base_final[metric]:.3f}")

def print_puzzle_state(task):
    """Print current state of puzzle task."""
    print(f"Current Stage: {task.current_stage + 1}/{task.num_stages}")
    print(f"Resources Remaining: {task.resources:.1f}/{task.initial_resources}")
    print(f"Time Spent: {task.time_spent}")
    
    print("\nStage Progress:")
    for stage in range(task.num_stages):
        quality = np.mean(np.abs(
            task.current_understanding[stage] - 
            task.true_solutions[stage]
        ))
        print(f"Stage {stage + 1}: {(1-quality)*100:.1f}% complete")
        
        if stage in task.stage_leaders:
            print(f"  Led by Agent {task.stage_leaders[stage]}")
    
    print("\nShared Information:")
    shared_by_stage = {}
    for stage, piece in task.shared_pieces:
        shared_by_stage[stage] = shared_by_stage.get(stage, 0) + 1
    
    for stage in range(task.num_stages):
        pieces_shared = shared_by_stage.get(stage, 0)
        total_pieces = len(task.true_solutions[stage])
        print(f"Stage {stage + 1}: {pieces_shared}/{total_pieces} pieces shared")

def print_recent_interactions(state):
    """Print details of recent interactions."""
    print("\nRecent Interactions:")
    for interaction in state.get('recent_interactions', []):
        print(f"Agent {interaction['claimer']} → Agent {interaction['target']}:")
        print(f"- Success: {interaction['success']}")
        if interaction['success']:
            if 'shared_info' in interaction:
                print(f"- Stage: {interaction['shared_info'][0] + 1}")
            print(f"- Quality: {interaction.get('quality', 'N/A')}")
            print(f"- Cost: {interaction.get('cost', 'N/A')}")
            print(f"- Time: {interaction.get('time', 'N/A')}")
            if 'moves_closer' in interaction:
                print(f"- Improves Solution: {interaction['moves_closer']}")

def plot_validation_results(logs):
    """Create validation plots comparing both models."""
    fig = plt.figure(figsize=(15, 15))
    
    # 1. Stage Quality Over Time
    ax1 = plt.subplot(421)
    steps = range(len(logs['base']['stage']))
    
    # Base model
    base_qualities = [log['stage_quality'] for log in logs['base']['stage']]
    base_overall = [log['overall_quality'] for log in logs['base']['stage']]
    ax1.plot(steps, base_qualities, 'b-', label='Base - Stage Quality', alpha=0.7)
    ax1.plot(steps, base_overall, 'b--', label='Base - Overall Quality', alpha=0.7)
    
    # Interactionist model
    inter_qualities = [log['stage_quality'] for log in logs['interactionist']['stage']]
    inter_overall = [log['overall_quality'] for log in logs['interactionist']['stage']]
    ax1.plot(steps, inter_qualities, 'r-', label='Interactionist - Stage Quality', alpha=0.7)
    ax1.plot(steps, inter_overall, 'r--', label='Interactionist - Overall Quality', alpha=0.7)
    
    ax1.set_title('Solution Quality Over Time')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Quality')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Stage Progression
    ax2 = plt.subplot(422)
    base_stages = [log['current_stage'] for log in logs['base']['stage']]
    inter_stages = [log['current_stage'] for log in logs['interactionist']['stage']]
    
    ax2.plot(steps, base_stages, 'b-', label='Base Model', alpha=0.7)
    ax2.plot(steps, inter_stages, 'r-', label='Interactionist Model', alpha=0.7)
    ax2.set_title('Stage Progression')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Current Stage')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Resource Usage
    ax3 = plt.subplot(423)
    base_resources = [log['resources'] for log in logs['base']['resource']]
    inter_resources = [log['resources'] for log in logs['interactionist']['resource']]
    
    ax3.plot(steps, base_resources, 'b-', label='Base Model', alpha=0.7)
    ax3.plot(steps, inter_resources, 'r-', label='Interactionist Model', alpha=0.7)
    ax3.set_title('Resource Usage')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Resources Remaining')
    ax3.grid(True)
    ax3.legend()
    
    # 4. Time Usage
    ax4 = plt.subplot(424)
    base_times = [log['time_spent'] for log in logs['base']['resource']]
    inter_times = [log['time_spent'] for log in logs['interactionist']['resource']]
    
    ax4.plot(steps, base_times, 'b-', label='Base Model', alpha=0.7)
    ax4.plot(steps, inter_times, 'r-', label='Interactionist Model', alpha=0.7)
    ax4.set_title('Time Usage')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Time Spent')
    ax4.grid(True)
    ax4.legend()
    
    # 5. Hierarchy Strength
    ax5 = plt.subplot(425)
    base_hierarchy = [log['hierarchy_strength'] for log in logs['base']['leadership']]
    inter_hierarchy = [log['hierarchy_strength'] for log in logs['interactionist']['leadership']]
    
    ax5.plot(steps, base_hierarchy, 'b-', label='Base Model', alpha=0.7)
    ax5.plot(steps, inter_hierarchy, 'r-', label='Interactionist Model', alpha=0.7)
    ax5.set_title('Hierarchy Strength')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Gini Coefficient')
    ax5.grid(True)
    ax5.legend()
    
    # 6. Perception Agreement
    ax6 = plt.subplot(426)
    base_agreement = [log['perception_agreement'] for log in logs['base']['leadership']]
    inter_agreement = [log['perception_agreement'] for log in logs['interactionist']['leadership']]
    
    ax6.plot(steps, base_agreement, 'b-', label='Base Model', alpha=0.7)
    ax6.plot(steps, inter_agreement, 'r-', label='Interactionist Model', alpha=0.7)
    ax6.set_title('Perception Agreement')
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Mean Correlation')
    ax6.grid(True)
    ax6.legend()
    
    # 7. Leader/Follower Identity Development
    ax7 = plt.subplot(427)
    base_leader_id = [log['mean_leader_identity'] for log in logs['base']['leadership']]
    base_follower_id = [log['mean_follower_identity'] for log in logs['base']['leadership']]
    inter_leader_id = [log['mean_leader_identity'] for log in logs['interactionist']['leadership']]
    inter_follower_id = [log['mean_follower_identity'] for log in logs['interactionist']['leadership']]
    
    ax7.plot(steps, base_leader_id, 'b-', label='Base - Leader', alpha=0.7)
    ax7.plot(steps, base_follower_id, 'b--', label='Base - Follower', alpha=0.7)
    ax7.plot(steps, inter_leader_id, 'r-', label='Inter - Leader', alpha=0.7)
    ax7.plot(steps, inter_follower_id, 'r--', label='Inter - Follower', alpha=0.7)
    ax7.set_title('Identity Development')
    ax7.set_xlabel('Step')
    ax7.set_ylabel('Mean Identity')
    ax7.grid(True)
    ax7.legend()
    
    # 8. Successful Interactions
    ax8 = plt.subplot(428)
    base_successes = [sum(1 for i in ints if i['success']) for ints in logs['base']['interactions']]
    inter_successes = [sum(1 for i in ints if i['success']) for ints in logs['interactionist']['interactions']]
    
    # Calculate moving averages
    window = 5
    base_ma = np.convolve(base_successes, np.ones(window)/window, mode='valid')
    inter_ma = np.convolve(inter_successes, np.ones(window)/window, mode='valid')
    ma_steps = range(len(base_ma))
    
    ax8.plot(ma_steps, base_ma, 'b-', label='Base Model', alpha=0.7)
    ax8.plot(ma_steps, inter_ma, 'r-', label='Interactionist Model', alpha=0.7)
    ax8.set_title('Successful Interactions (Moving Average)')
    ax8.set_xlabel('Step')
    ax8.set_ylabel('Number of Successful Interactions')
    ax8.grid(True)
    ax8.legend()
    
    # Add vertical line for interactionist transition
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5, label='Identity Transition')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'outputs/plots/puzzle_validation_comparison_{timestamp}.png')
    plt.close()

if __name__ == "__main__":
    # Run validation
    results = run_puzzle_validation() 