from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class ModelParameters(BaseModel):
    """Parameters for leadership emergence model."""
    
    # Basic simulation parameters
    n_agents: int = Field(default=6, ge=2, description="Number of agents")
    n_steps: int = Field(default=100, ge=1, description="Number of simulation steps")
    
    # Initial conditions
    initial_li_equal: bool = Field(default=True, description="Whether all agents start with equal leadership identities")
    initial_identity: float = Field(default=50.0, ge=0.0, le=100.0, description="Initial identity value if equal")
    
    # Identity change parameters
    li_change_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Rate of leadership identity change")
    
    # Schema parameters
    schema_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Final weight for schema influence")
    weight_transition_start: float = Field(default=0.2, ge=0.0, le=1.0, description="When to start transition from schema to identity")
    weight_transition_end: float = Field(default=0.8, ge=0.0, le=1.0, description="When to end transition from schema to identity")
    weight_function: str = Field(default="linear", description="Function type for schema-identity transition")
    
    # Interaction thresholds
    claim_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Threshold for leadership claims")
    grant_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Threshold for granting leadership")
    
    # Perception change parameters
    perception_change_success: float = Field(default=0.1, ge=0.0, le=1.0, description="Perception increase for successful claims")
    perception_change_reject: float = Field(default=0.1, ge=0.0, le=1.0, description="Perception decrease for rejected claims")
    perception_change_noclaim: float = Field(default=0.05, ge=0.0, le=1.0, description="Perception decrease for no claim")
    
    # ILT matching parameters
    ilt_match_algorithm: str = Field(default="euclidean", description="Algorithm for ILT matching")
    ilt_match_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for ILT matching algorithm")
    
    # Behavioral flags
    penalize_no_claim: bool = Field(default=True, description="Whether to penalize agents for not claiming")
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        arbitrary_types_allowed = True
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ModelParameters':
        """Create parameters from configuration dictionary."""
        # Ensure n_steps is included
        if 'n_steps' not in config:
            config['n_steps'] = cls.__fields__['n_steps'].default
        return cls(**config) 