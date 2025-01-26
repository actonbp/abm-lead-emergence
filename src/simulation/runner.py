"""
Enhanced simulation runner for leadership emergence models.
Supports different theoretical perspectives and optional contexts.
"""

from typing import Dict, Any, Optional, Type
import logging
from datetime import datetime
from pathlib import Path

from ..models.base_model import BaseLeadershipModel
from .contexts.base import Context

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimulationRunner:
    """Runs leadership emergence simulations with different perspectives and contexts."""
    
    def __init__(
        self,
        model_class: Type[BaseLeadershipModel],
        context: Optional[Context] = None,
        output_dir: Optional[str] = None
    ):
        """Initialize runner with model class and optional context.
        
        Args:
            model_class: The perspective-specific model class to use
            context: Optional context to modify model behavior
            output_dir: Directory to save results (if None, don't save)
        """
        self.model_class = model_class
        self.context = context
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_single(
        self,
        params: Dict[str, Any],
        n_steps: int,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run a single simulation with given parameters.
        
        Args:
            params: Model parameters
            n_steps: Number of steps to run
            random_seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary containing simulation results
        """
        logger.info(f"Starting simulation for {n_steps} steps")
        logger.info(f"Model class: {self.model_class.__name__}")
        if self.context:
            logger.info(f"Context: {self.context.__class__.__name__}")
        
        # Initialize model with perspective-specific class
        model = self.model_class(params=params, random_seed=random_seed)
        
        # Run simulation
        history = []
        for step in range(n_steps):
            if step % 10 == 0:
                logger.info(f"Step {step}/{n_steps}")
            
            # Get current state
            state = model.step()
            
            # Apply context modifications if present
            if self.context:
                state['context'] = self.context.get_state()
            
            history.append(state)
        
        # Prepare results
        results = {
            'model_class': self.model_class.__name__,
            'context_class': self.context.__class__.__name__ if self.context else None,
            'parameters': params,
            'n_steps': n_steps,
            'random_seed': random_seed,
            'history': history,
            'final_state': model.get_state(),
            'metadata': {
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Save results if output directory specified
        if self.output_dir:
            self._save_results(results)
        
        logger.info("Simulation complete")
        return results
    
    def run_parameter_sweep(
        self,
        param_grid: Dict[str, list],
        n_steps: int,
        n_replications: int = 1,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run multiple simulations sweeping over parameter combinations.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            n_steps: Number of steps per simulation
            n_replications: Number of replications per parameter combination
            random_seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary containing results for all simulations
        """
        logger.info(f"Starting parameter sweep")
        logger.info(f"Parameter grid: {param_grid}")
        logger.info(f"Replications per combination: {n_replications}")
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        # Run simulations for each combination
        all_results = []
        for params in param_combinations:
            logger.info(f"Running combination: {params}")
            
            # Run replications
            for rep in range(n_replications):
                rep_seed = random_seed + rep if random_seed else None
                results = self.run_single(
                    params=params,
                    n_steps=n_steps,
                    random_seed=rep_seed
                )
                results['replication'] = rep
                all_results.append(results)
        
        # Prepare sweep results
        sweep_results = {
            'param_grid': param_grid,
            'n_steps': n_steps,
            'n_replications': n_replications,
            'base_random_seed': random_seed,
            'results': all_results,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_class': self.model_class.__name__,
                'context_class': self.context.__class__.__name__ if self.context else None
            }
        }
        
        # Save sweep results if output directory specified
        if self.output_dir:
            self._save_sweep_results(sweep_results)
        
        logger.info("Parameter sweep complete")
        return sweep_results
    
    def _generate_param_combinations(self, param_grid: Dict[str, list]) -> list:
        """Generate all combinations of parameters from grid."""
        import itertools
        
        # Get parameter names and values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generate all combinations
        combinations = []
        for values in itertools.product(*param_values):
            combination = dict(zip(param_names, values))
            combinations.append(combination)
        
        return combinations
    
    def _save_results(self, results: Dict[str, Any]):
        """Save single simulation results."""
        import json
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sim_{results['model_class']}_{timestamp}.json"
        
        # Save to file
        with open(self.output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _save_sweep_results(self, results: Dict[str, Any]):
        """Save parameter sweep results."""
        import json
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sweep_{results['metadata']['model_class']}_{timestamp}.json"
        
        # Save to file
        with open(self.output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2) 