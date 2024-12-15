"""
Tests for base leadership model.
"""

import pytest
import numpy as np
from src.models.base import BaseLeadershipModel, ModelParameters
from pydantic import ValidationError


class SimpleModel(BaseLeadershipModel):
    """Simple model implementation for testing."""
    
    def step(self):
        """Implement simple random walk."""
        self.agents += self.random.normal(0, 0.1, size=len(self.agents))
        self.agents = np.clip(self.agents, 0, 1)
        return self.get_state()


def test_parameter_validation():
    """Test parameter validation."""
    # Valid parameters
    valid_params = {
        "n_agents": 10,
        "initial_li_equal": True,
        "li_change_rate": 1.0
    }
    params = ModelParameters(**valid_params)
    assert params.n_agents == 10
    
    # Invalid n_agents
    with pytest.raises(ValidationError):
        ModelParameters(
            n_agents=1,  # Too small
            initial_li_equal=True,
            li_change_rate=1.0
        )
    
    # Invalid li_change_rate
    with pytest.raises(ValidationError):
        ModelParameters(
            n_agents=10,
            initial_li_equal=True,
            li_change_rate=-1.0  # Negative rate
        )


def test_model_initialization():
    """Test model initialization."""
    config = {
        "n_agents": 10,
        "initial_li_equal": True,
        "li_change_rate": 1.0
    }
    
    # Test with config dict
    model = SimpleModel(config=config)
    assert len(model.agents) == 10
    assert model.timestep == 0
    
    # Test with equal initial conditions
    assert np.allclose(model.agents, 0.5)
    
    # Test with random initial conditions
    model = SimpleModel(
        config={**config, "initial_li_equal": False},
        random_seed=42
    )
    assert not np.allclose(model.agents, 0.5)
    assert np.all((model.agents >= 0) & (model.agents <= 1))


def test_model_running():
    """Test model execution."""
    config = {
        "n_agents": 5,
        "initial_li_equal": True,
        "li_change_rate": 1.0
    }
    
    model = SimpleModel(config=config)
    
    # Test single step
    state = model.step()
    assert model.timestep == 0
    assert "leader_identities" in state
    assert len(state["leader_identities"]) == 5
    
    # Test multiple steps
    results = model.run(n_steps=10)
    assert model.timestep == 10
    assert len(results["history"]) == 10
    assert "parameters" in results


def test_reproducibility():
    """Test random seed reproducibility."""
    config = {
        "n_agents": 10,
        "initial_li_equal": False,
        "li_change_rate": 1.0
    }
    
    # Create two models with same seed
    model1 = SimpleModel(config=config, random_seed=42)
    model2 = SimpleModel(config=config, random_seed=42)
    
    # Run both models
    results1 = model1.run(n_steps=5)
    results2 = model2.run(n_steps=5)
    
    # Compare histories
    for state1, state2 in zip(results1["history"], results2["history"]):
        assert np.allclose(
            state1["leader_identities"],
            state2["leader_identities"]
        ) 