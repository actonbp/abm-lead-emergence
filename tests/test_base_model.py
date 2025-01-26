"""Essential tests for base leadership model."""

import pytest
from src.models.base_model import BaseLeadershipModel

@pytest.fixture
def base_model():
    """Create a base model instance with test parameters."""
    params = {
        "n_agents": 4,
        "claim_threshold": 0.5,
        "grant_threshold": 0.4,
        "claim_multiplier": 0.7,
        "grant_multiplier": 0.6,
        "success_boost": 5.0,
        "failure_penalty": 3.0,
        "grant_penalty": 2.0
    }
    return BaseLeadershipModel(params)

def test_model_initialization(base_model):
    """Test model initialization."""
    assert len(base_model.agents) == 4
    assert base_model.time == 0
    # Check initial scores
    assert all(agent.lead_score == 50.0 for agent in base_model.agents)

def test_step_execution(base_model):
    """Test single step execution."""
    state = base_model.step()
    
    # Verify state structure
    assert 'time' in state
    assert 'agents' in state
    assert len(state['agents']) == 4
    
    # Verify time increment
    assert state['time'] == 1

def test_agent_state(base_model):
    """Test agent state tracking."""
    state = base_model.step()
    agent_state = state['agents'][0]
    
    # Check agent state structure
    assert 'id' in agent_state
    assert 'lead_score' in agent_state
    assert 'last_interaction' in agent_state
    
    # Check interaction tracking
    interaction = agent_state['last_interaction']
    assert 'role' in interaction  # Either 'claimer' or 'granter'
    assert 'other_id' in interaction

def test_run_simulation(base_model):
    """Test running full simulation."""
    n_steps = 5
    history = base_model.run(n_steps)
    
    # Check history length
    assert len(history) == n_steps
    
    # Check state structure in history
    for state in history:
        assert 'time' in state
        assert 'agents' in state
        assert len(state['agents']) == 4 