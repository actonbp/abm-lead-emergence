"""
Simulation execution engine for running batches of leadership emergence models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Type, Any
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import json

from ..models.base_model import BaseLeadershipModel

class SimulationRunner:
    """Manages batch execution of leadership emergence simulations."""
    
    def __init__(
        self,
        model_class: Type[BaseLeadershipModel],
        parameter_space: Dict[str, List[Any]],
        n_steps: int = 100,
        n_replications: int = 10,
        output_dir: str = None,
        random_seed: int = None
    ):
        self.model_class = model_class
        self.parameter_space = parameter_space
        self.n_steps = n_steps
        self.n_replications = n_replications
        self.output_dir = Path(output_dir or "data/raw")
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_single_simulation(
        self,
        params: Dict[str, Any],
        replication: int
    ) -> Dict[str, Any]:
        """Run a single simulation with given parameters."""
        # Set random seed for reproducibility
        seed = self.rng.integers(0, 2**32) if self.random_seed is None else self.random_seed + replication
        
        # Initialize model
        model = self.model_class(random_seed=seed, **params)
        
        # Run simulation
        history = []
        for _ in range(self.n_steps):
            state = model.step()
            history.append(state)
        
        # Return results
        return {
            "parameters": params,
            "replication": replication,
            "history": history,
            "final_state": state
        }
    
    def run_batch(self, n_processes: int = None) -> str:
        """Run a batch of simulations with parameter combinations."""
        # Generate parameter combinations
        param_combinations = []
        for replication in range(self.n_replications):
            for params in self._generate_parameter_combinations():
                param_combinations.append((params, replication))
        
        # Run simulations in parallel
        results = []
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            futures = [
                executor.submit(self.run_single_simulation, params, rep)
                for params, rep in param_combinations
            ]
            for future in futures:
                results.append(future.result())
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"simulation_batch_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump({
                "metadata": {
                    "model_class": self.model_class.__name__,
                    "n_steps": self.n_steps,
                    "n_replications": self.n_replications,
                    "parameter_space": self.parameter_space,
                    "random_seed": self.random_seed,
                    "timestamp": timestamp
                },
                "results": results
            }, f, indent=2)
        
        return str(output_file)
    
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters from parameter space."""
        # Get all parameter names and values
        param_names = list(self.parameter_space.keys())
        param_values = list(self.parameter_space.values())
        
        # Generate combinations
        combinations = []
        for values in np.ndindex(*[len(v) for v in param_values]):
            combination = {
                name: param_values[i][values[i]]
                for i, name in enumerate(param_names)
            }
            combinations.append(combination)
        
        return combinations 