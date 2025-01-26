"""
Test script to verify base model functionality.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.base_model import BaseLeadershipModel
import numpy as np


def print_interaction(step: int, state: dict):
    """Pretty print an interaction."""
    print(f"\nStep {step}:")
    print("-" * 40)
    
    # Find agents involved in interaction
    claimer = None
    granter = None
    
    for agent in state['agents']:
        if agent['last_interaction']['role'] == 'claimer':
            claimer = agent
        elif agent['last_interaction']['role'] == 'granter':
            granter = agent
    
    if not (claimer and granter):
        print("No interaction this step")
        return
    
    # Print interaction details
    print(f"Claimer (Agent {claimer['id']}, score={claimer['lead_score']:.1f}):")
    print(f"  - Claim probability: {claimer['last_interaction']['claim_prob']:.2f}")
    print(f"  - Claimed: {claimer['last_interaction']['claimed']}")
    
    print(f"\nGranter (Agent {granter['id']}, score={granter['lead_score']:.1f}):")
    print(f"  - Grant probability: {granter['last_interaction']['grant_prob']:.2f}")
    print(f"  - Granted: {granter['last_interaction']['granted']}")
    
    # Print outcome
    if claimer['last_interaction']['claimed']:
        if granter['last_interaction']['granted']:
            print("\nOutcome: Successful leadership claim!")
            print(f"  - Agent {claimer['id']} leadership increased")
            print(f"  - Agent {granter['id']} leadership decreased")
        else:
            print("\nOutcome: Failed leadership claim")
            print(f"  - Agent {claimer['id']} leadership decreased")
    else:
        print("\nOutcome: No leadership claim made")


def run_test():
    """Run basic model test."""
    print("Testing Base Leadership Model")
    print("=" * 40)
    
    # Initialize model with fixed seed for reproducibility
    params = {
        "n_agents": 4,
        "claim_multiplier": 0.7,
        "grant_multiplier": 0.6,
        "success_boost": 5.0,
        "failure_penalty": 3.0,
        "grant_penalty": 2.0
    }
    model = BaseLeadershipModel(params=params, random_seed=42)
    
    # Print initial state
    print("\nInitial State:")
    print("-" * 40)
    for agent in model.agents:
        print(f"Agent {agent.id}: score = {agent.lead_score:.1f}")
    
    # Run for a few steps
    n_steps = 5
    for step in range(n_steps):
        state = model.step()
        print_interaction(step + 1, state)
    
    # Print final state
    print("\nFinal State:")
    print("-" * 40)
    for agent in model.agents:
        print(f"Agent {agent.id}: score = {agent.lead_score:.1f}")
        print(f"  - Claims made: {sum(agent.history['claims'])}")
        print(f"  - Grants given: {sum(agent.history['grants'])}")
        print(f"  - Score history: {[f'{score:.1f}' for score in agent.history['lead_score']]}")


if __name__ == "__main__":
    run_test() 