"""
Tests for schema-based leadership model.
"""

import pytest
import numpy as np
from src.models.schema_model import SchemaModel, SchemaModelParameters


def test_schema_parameters():
    """Test schema model parameter validation."""
    # Test valid parameters
    valid_params = {
        "n_agents": 10,
        "initial_li_equal": True,
        "li_change_rate": 1.0,
        "schema_weight": 0.5,
        "claim_threshold": 0.6,
        "grant_threshold": 0.4
    }
    params = SchemaModelParameters(**valid_params)
    assert params.schema_weight == 0.5
    
    # Test invalid schema_weight
    with pytest.raises(ValueError):
        SchemaModelParameters(
            **{**valid_params, "schema_weight": 1.5}  # Too high
        )


def test_schema_initialization():
    """Test schema model initialization."""
    config = {
        "n_agents": 10,
        "initial_li_equal": True,
        "li_change_rate": 1.0,
        "schema_weight": 0.5,
        "claim_threshold": 0.6,
        "grant_threshold": 0.4
    }
    
    model = SchemaModel(config=config, random_seed=42)
    
    # Check dimensions
    assert len(model.schemas) == 10
    assert len(model.follower_identities) == 10
    
    # Check value ranges
    assert np.all((model.schemas >= 0) & (model.schemas <= 1))
    assert np.all((model.follower_identities >= 0) & (model.follower_identities <= 1))


def test_interaction_mechanics():
    """Test leadership claim and grant mechanics."""
    config = {
        "n_agents": 4,
        "initial_li_equal": True,
        "li_change_rate": 1.0,
        "schema_weight": 0.5,
        "claim_threshold": 0.6,
        "grant_threshold": 0.4
    }
    
    model = SchemaModel(config=config, random_seed=42)
    
    # Force specific values for testing
    model.agents[0] = 0.8  # High leader identity
    model.schemas[0] = 0.7  # Similar schema
    
    # Test leadership claim
    assert model._make_leadership_claim(0)  # Should claim
    
    # Test leadership grant
    assert model._grant_leadership(1, 0)  # Should grant


def test_identity_updates():
    """Test identity updates after interactions."""
    config = {
        "n_agents": 4,
        "initial_li_equal": True,
        "li_change_rate": 0.1,
        "schema_weight": 0.5,
        "claim_threshold": 0.6,
        "grant_threshold": 0.4
    }
    
    model = SchemaModel(config=config, random_seed=42)
    
    # Store initial values
    initial_leader_identity = model.agents[0]
    initial_follower_identity = model.follower_identities[1]
    
    # Simulate successful interaction
    model._update_identities(0, 1, claim_granted=True)
    
    # Check identities increased
    assert model.agents[0] > initial_leader_identity
    assert model.follower_identities[1] > initial_follower_identity
    
    # Simulate failed interaction
    initial_leader_identity = model.agents[0]
    initial_follower_identity = model.follower_identities[1]
    
    model._update_identities(0, 1, claim_granted=False)
    
    # Check identities decreased
    assert model.agents[0] < initial_leader_identity
    assert model.follower_identities[1] < initial_follower_identity


def test_model_step():
    """Test full model step."""
    config = {
        "n_agents": 4,
        "initial_li_equal": True,
        "li_change_rate": 0.1,
        "schema_weight": 0.5,
        "claim_threshold": 0.6,
        "grant_threshold": 0.4
    }
    
    model = SchemaModel(config=config, random_seed=42)
    
    # Run one step
    state = model.step()
    
    # Check state contents
    assert 'leader_identities' in state
    assert 'follower_identities' in state
    assert 'schemas' in state
    assert len(state['leader_identities']) == 4
    
    # Check value ranges
    assert np.all((state['leader_identities'] >= 0) & 
                 (state['leader_identities'] <= 1))
    assert np.all((state['follower_identities'] >= 0) & 
                 (state['follower_identities'] <= 1)) 