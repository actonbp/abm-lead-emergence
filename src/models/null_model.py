"""
Null model for leadership emergence - uses random decisions as a control condition.
"""

from typing import Dict, Any, Tuple
import numpy as np
from .base_model import BaseLeadershipModel, Agent, ModelParameters

class NullAgent(Agent):
    """Agent that makes random leadership decisions."""
    
    def decide_claim(self, match_score: float) -> bool:
        """Make random leadership claim."""
        claim_probability = 0.5  # Pure random chance
        noise = self.rng.normal(0, 0.05)
        claim_probability = np.clip(claim_probability + noise, 0, 1)
        
        self.last_interaction['match_score'] = match_score
        self.last_interaction['claimed'] = self.rng.random() < claim_probability
        
        return self.last_interaction['claimed']
    
    def decide_grant(self, match_score: float) -> bool:
        """Make random granting decision."""
        grant_probability = 0.5  # Pure random chance
        noise = self.rng.normal(0, 0.05)
        grant_probability = np.clip(grant_probability + noise, 0, 1)
        
        self.last_interaction['granted'] = self.rng.random() < grant_probability
        
        return self.last_interaction['granted']

class NullModel(BaseLeadershipModel):
    """Null model with random leadership decisions."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize with parameters."""
        if isinstance(params, dict):
            self.params = ModelParameters(**params)
        else:
            self.params = params if params else ModelParameters()
        
        # Initialize random number generator
        self.rng = np.random.default_rng(self.params.random_seed)
        self.time = 0
        
        # Initialize agents using NullAgent
        self.agents = [
            NullAgent(i, self.rng, self.params)
            for i in range(self.params.n_agents)
        ]
        
        # Track model state
        self.history = [] 