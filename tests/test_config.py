"""Essential tests for configuration loading."""

import pytest
from src.utils.config import load_config

# Minimal test config
VALID_CONFIG = """
simulation:
  n_agents: 4
  claim_rate: 0.5
  n_steps: 100

parameters:
  success_boost: 2.0
  failure_penalty: 1.0
  grant_rate: 0.6

output:
  save_frequency: 10
  log_interactions: true
"""

@pytest.fixture
def temp_config(tmp_path):
    """Create a temporary config file."""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(VALID_CONFIG)
    return config_path

def test_valid_config(temp_config):
    """Test loading a valid configuration."""
    config = load_config(str(temp_config))
    # Check core parameters are loaded correctly
    assert config["simulation"]["n_agents"] == 4
    assert config["parameters"]["success_boost"] == 2.0

def test_missing_file():
    """Test loading non-existent config file."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")

def test_base_config():
    """Test loading actual base config file."""
    base_config = load_config("config/base.yaml")
    # Verify essential parameters exist and have valid values
    assert isinstance(base_config["simulation"]["n_agents"], int)
    assert 0 <= base_config["simulation"]["claim_rate"] <= 1 