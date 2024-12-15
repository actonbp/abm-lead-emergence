"""
Simulation execution engine for running batches of leadership emergence models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Type, Any, Optional
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import multiprocessing
import os
from dataclasses import dataclass

from src.models.base_model import BaseLeadershipModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy arrays and model objects."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, '__dict__'):
            # For model objects, just save their state
            return {
                'class': obj.__class__.__name__,
                'state': {
                    k: v for k, v in obj.__dict__.items()
                    if not k.startswith('_')
                }
            }
        return super().default(obj)


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""
    model_params: Dict[str, Any]
    n_steps: int
    random_seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")


class BatchRunner:
    """Manages batch execution of leadership emergence simulations."""
    
    def __init__(
        self,
        model_class: Type[BaseLeadershipModel],
        output_dir: Path,
        n_jobs: int = -1
    ):
        """Initialize batch runner.
        
        Args:
            model_class: Class of the model to run
            output_dir: Directory for output files
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.model_class = model_class
        self.output_dir = Path(output_dir)
        
        # Set number of jobs
        if n_jobs < 0:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = min(n_jobs, multiprocessing.cpu_count())
        
        # Create output directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary output directories."""
        dirs = ['raw', 'processed', 'logs']
        for dir_name in dirs:
            dir_path = self.output_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
    
    def run_single_simulation(
        self,
        config: SimulationConfig,
        run_id: str,
        batch_dir: Path
    ) -> str:
        """Run a single simulation with given configuration.
        
        Args:
            config: Simulation configuration
            run_id: Unique identifier for this run
            batch_dir: Directory for batch results
            
        Returns:
            Path to result file
        """
        try:
            # Initialize model
            model = self.model_class(
                config=config.model_params,
                random_seed=config.random_seed
            )
            
            # Run simulation
            history = []
            for timestep in range(config.n_steps):
                state = model.step()
                state['timestep'] = timestep
                history.append(state)
            
            # Get final state
            final_state = history[-1]
            
            # Extract network state
            network_state = {
                'nodes': list(model.interaction_network.nodes()),
                'edges': [(u, v, d) for u, v, d in model.interaction_network.edges(data=True)]
            }
            
            # Prepare results
            results = {
                'parameters': config.model_params,
                'history': history,
                'final_state': final_state,
                'leader_identities': final_state['leader_identities'],
                'follower_identities': final_state.get('follower_identities', [0.0] * len(final_state['leader_identities'])),
                'schemas': final_state.get('schemas', [0.0] * len(final_state['leader_identities'])),
                'model_state': {
                    'class': model.__class__.__name__,
                    'time': model.time,
                    'n_agents': len(model.agents),
                    'interaction_network': network_state,
                    'agent_states': [
                        {
                            'id': agent.id,
                            'leader_identity': agent.leader_identity,
                            'follower_identity': agent.follower_identity,
                            'characteristics': agent.characteristics,
                            'ilt': agent.ilt,
                            'leader_identity_history': agent.leader_identity_history,
                            'follower_identity_history': agent.follower_identity_history
                        }
                        for agent in model.agents
                    ]
                },
                'metadata': {
                    'run_id': run_id,
                    'timestamp': datetime.now().isoformat(),
                    'model_class': self.model_class.__name__,
                    'n_steps': config.n_steps,
                    'random_seed': config.random_seed
                }
            }
            
            # Save results
            result_file = batch_dir / f"{run_id}.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
            
            return str(result_file)
            
        except Exception as e:
            logger.error(f"Error in run {run_id}: {str(e)}")
            raise
    
    def run_batch(
        self,
        configs: List[SimulationConfig],
        batch_id: str
    ) -> List[str]:
        """Run a batch of simulations with different configurations.
        
        Args:
            configs: List of simulation configurations
            batch_id: Unique identifier for this batch
            
        Returns:
            List of paths to result files
        """
        # Create batch directory
        batch_dir = self.output_dir / f"batch_{batch_id}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Save batch metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'model_class': self.model_class.__name__,
            'n_configurations': len(configs),
            'n_jobs': self.n_jobs,
            'configurations': [
                {
                    'model_params': config.model_params,
                    'n_steps': config.n_steps,
                    'random_seed': config.random_seed
                }
                for config in configs
            ]
        }
        
        with open(batch_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)
        
        # Run simulations in parallel
        result_files = []
        
        logger.info(f"Running {len(configs)} simulations with {self.n_jobs} workers")
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for i, config in enumerate(configs):
                run_id = f"{batch_id}_run_{i:04d}"
                future = executor.submit(
                    self.run_single_simulation,
                    config,
                    run_id,
                    batch_dir
                )
                futures.append((run_id, future))
            
            # Process results as they complete
            for i, (run_id, future) in enumerate(futures):
                try:
                    result_file = future.result()
                    result_files.append(result_file)
                    logger.info(f"Completed run {run_id} ({i+1}/{len(configs)})")
                except Exception as e:
                    logger.error(f"Failed run {run_id}: {str(e)}")
        
        return result_files


def create_parameter_grid(
    param_ranges: Dict[str, List[Any]]
) -> List[Dict[str, Any]]:
    """Create grid of parameter combinations.
    
    Args:
        param_ranges: Dict mapping parameter names to lists of values
        
    Returns:
        List of parameter combinations
    """
    # Get parameter names and values
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    # Generate combinations
    combinations = []
    for values in np.ndindex(*[len(v) for v in param_values]):
        combination = {
            name: param_values[i][values[i]]
            for i, name in enumerate(param_names)
        }
        combinations.append(combination)
    
    return combinations 