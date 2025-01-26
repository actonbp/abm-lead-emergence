"""Configuration utilities for leadership emergence model."""

import yaml
from pathlib import Path
from typing import Dict, Any, Union, Optional
from datetime import datetime

# Required configuration keys and their types/ranges
CONFIG_SCHEMA = {
    'simulation': {
        'n_agents': (int, (2, 100)),      # (type, (min, max))
        'claim_rate': (float, (0.0, 1.0)),
        'n_steps': (int, (1, 10000))
    },
    'parameters': {
        'success_boost': (float, (0.0, None)),  # None means no upper bound
        'failure_penalty': (float, (0.0, None)),
        'grant_rate': (float, (0.0, 1.0))
    },
    'output': {
        'save_frequency': (int, (1, None)),
        'log_interactions': (bool, None)  # None means no range check
    }
}

def validate_value(value: Any, expected_type: type, value_range: Optional[tuple] = None) -> bool:
    """Validate a config value against its expected type and range."""
    if not isinstance(value, expected_type):
        return False
    if value_range is not None:
        min_val, max_val = value_range
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
    return True

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing validated configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate against schema
        for section, keys in CONFIG_SCHEMA.items():
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
            
            for key, (expected_type, value_range) in keys.items():
                if key not in config[section]:
                    raise ValueError(f"Missing required key: {section}.{key}")
                
                value = config[section][key]
                if not validate_value(value, expected_type, value_range):
                    raise ValueError(
                        f"Invalid value for {section}.{key}: {value}. "
                        f"Expected type {expected_type.__name__}"
                        + (f" in range {value_range}" if value_range else "")
                    )
        
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format: {e}")

def save_results_json(results: Dict[str, Any], config: Dict[str, Any], output_path: str) -> None:
    """Save simulation results and config to JSON.
    
    Args:
        results: Simulation results dictionary
        config: Configuration dictionary used for simulation
        output_path: Path to save JSON output
    """
    import json
    
    output = {
        "simulation_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config_used": config,
        "results": results
    }
    
    # Ensure directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2) 