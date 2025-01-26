"""
Parameter exploration module for leadership emergence analysis.
Implements Latin Hypercube Sampling and parameter evaluation.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from scipy.stats import qmc
import json
from pathlib import Path

class ParameterExplorer:
    """Class for exploring parameter space using various sampling methods."""
    
    def __init__(
        self,
        parameter_space: Dict[str, Dict[str, Any]],
        n_initial_samples: int = 50
    ):
        """Initialize parameter explorer.
        
        Args:
            parameter_space: Dictionary defining parameter space
            n_initial_samples: Number of initial samples to generate
        """
        self.parameter_space = parameter_space
        self.n_initial_samples = n_initial_samples
        
        # Separate continuous and discrete parameters
        self.continuous_params = {}
        self.discrete_params = {}
        
        for param_name, param_info in parameter_space.items():
            if param_info['type'] == 'continuous':
                self.continuous_params[param_name] = param_info['range']
            elif param_info['type'] == 'discrete':
                self.discrete_params[param_name] = param_info['values']
    
    def generate_initial_samples(self) -> List[Dict[str, Any]]:
        """Generate initial parameter combinations using Latin Hypercube Sampling."""
        # Initialize sampler for continuous parameters
        n_continuous = len(self.continuous_params)
        if n_continuous > 0:
            sampler = qmc.LatinHypercube(d=n_continuous, seed=42)
            samples = sampler.random(n=self.n_initial_samples)
            samples = samples.reshape(self.n_initial_samples, n_continuous)
            
            # Scale samples to parameter ranges
            for i, (param_name, (low, high)) in enumerate(self.continuous_params.items()):
                samples[:, i] = low + (high - low) * samples[:, i]
        
        # Generate configurations
        configs = []
        for i in range(self.n_initial_samples):
            config = {}
            
            # Add continuous parameters
            if n_continuous > 0:
                for j, param_name in enumerate(self.continuous_params.keys()):
                    config[param_name] = float(samples[i, j])
            
            # Add discrete parameters
            for param_name, values in self.discrete_params.items():
                config[param_name] = np.random.choice(values)
            
            configs.append(config)
        
        return configs
    
    def save_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2) 