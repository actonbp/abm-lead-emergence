"""
Puzzle Task Implementation

This module implements a staged puzzle task where agents must collectively discover 
a true pattern through coordinated information sharing. Leadership emergence affects
efficiency and resource usage across multiple stages.
"""

from typing import Dict, Any, List
import numpy as np
from .base_task import BaseTask

class PuzzleTask(BaseTask):
    def __init__(self, num_agents: int, puzzle_size: int = 9, num_stages: int = 3):
        """Initialize puzzle task.
        
        Args:
            num_agents: Number of agents participating
            puzzle_size: Number of total information pieces (default 9 for 3 per stage)
            num_stages: Number of stages in the puzzle
        """
        self.num_agents = num_agents
        self.puzzle_size = puzzle_size
        self.num_stages = num_stages
        
        # Resource tracking
        self.initial_resources = 100.0
        self.resources = self.initial_resources
        self.time_spent = 0
        
        # Stage tracking
        self.current_stage = 0
        self.stage_leaders = {}  # Track who led each stage
        self.stage_costs = {}    # Track resource usage per stage
        
        # Initialize the task
        self._initialize_puzzle()
        
    def _initialize_puzzle(self):
        """Set up the puzzle by creating and distributing information."""
        # Generate the true solution as multiple stages
        # Each stage has its own pattern to discover
        self.true_solutions = []
        self.current_understanding = []
        
        pieces_per_stage = self.puzzle_size // self.num_stages
        
        for stage in range(self.num_stages):
            # Generate true pattern for this stage
            stage_solution = np.random.normal(0.5, 0.2, pieces_per_stage)
            stage_solution = np.clip(stage_solution, 0, 1)
            self.true_solutions.append(stage_solution)
            
            # Initialize understanding at 0
            stage_understanding = np.zeros(pieces_per_stage)
            self.current_understanding.append(stage_understanding)
        
        # Distribute information pieces among agents
        all_pieces = []
        for stage in range(self.num_stages):
            stage_pieces = [(stage, i) for i in range(pieces_per_stage)]
            all_pieces.extend(stage_pieces)
        np.random.shuffle(all_pieces)
        
        # Give each agent some pieces that are close to true values
        # and some that are misleading
        self.agent_pieces = {}
        pieces_per_agent = len(all_pieces) // self.num_agents
        
        for i in range(self.num_agents):
            agent_pieces = []
            start_idx = i * pieces_per_agent
            pieces = all_pieces[start_idx:start_idx + pieces_per_agent]
            
            for stage, piece_idx in pieces:
                # Each agent gets some accurate and some inaccurate info
                true_value = self.true_solutions[stage][piece_idx]
                if np.random.random() < 0.8:  # 80% accurate
                    value = true_value + np.random.normal(0, 0.05)  # Less noise
                else:  # 20% misleading
                    value = 1 - true_value + np.random.normal(0, 0.05)  # Less noise
                value = np.clip(value, 0, 1)
                agent_pieces.append((stage, piece_idx, value))
            
            self.agent_pieces[i] = agent_pieces
        
        # Track what's been shared
        self.shared_pieces = set()
        
    def get_shared_info(self, agent_id: int) -> Dict[str, Any]:
        """Get currently shared/visible information."""
        return {
            "current_stage": self.current_stage,
            "current_understanding": [u.copy() for u in self.current_understanding],
            "shared_pieces": list(self.shared_pieces),
            "stage_quality": self._evaluate_stage_understanding(),
            "overall_quality": self._evaluate_understanding(),
            "resources_remaining": self.resources,
            "time_spent": self.time_spent,
            "stage_leaders": self.stage_leaders.copy()
        }
        
    def get_unique_info(self, agent_id: int) -> Dict[str, Any]:
        """Get information unique to an agent."""
        # Only return pieces relevant to current stage
        current_pieces = [
            (s, p, v) for s, p, v in self.agent_pieces.get(agent_id, [])
            if s == self.current_stage and (s, p) not in self.shared_pieces
        ]
        return {
            "pieces": current_pieces,
            "num_unshared": len(current_pieces)
        }
    
    def _calculate_efficiency_multiplier(self) -> float:
        """Calculate resource efficiency multiplier based on leadership structure.
        
        Returns a multiplier that reduces costs based on number of leaders, with
        diminishing returns for each additional leader.
        
        No leaders: 1.0 (base cost)
        1 leader: 0.15 (85% reduction)
        2 leaders: 0.08 (92% reduction)
        3 leaders: 0.05 (95% reduction)
        """
        num_leaders = len(self.stage_leaders)
        if num_leaders == 0:
            return 1.0  # No efficiency gain without leaders
        
        # Exponential decay of costs with diminishing returns
        # base_cost * (0.15 ^ num_leaders) but with a floor
        return max(0.05, 0.15 ** num_leaders)
        
    def share_info(self, from_agent: int, to_agent: int, granted: bool) -> Dict[str, Any]:
        """Share information through a leadership claim."""
        # Get unshared pieces for current stage
        agent_pieces = [
            (s, p, v) for s, p, v in self.agent_pieces[from_agent]
            if s == self.current_stage and (s, p) not in self.shared_pieces
        ]
        
        if not agent_pieces:
            return {
                "shared_info": None,
                "success": False,
                "quality": 0.0,
                "cost": 0.0,
                "time": 0.0,
                "stage_complete": False,
                "moves_closer": False
            }
        
        # Choose a piece to share
        stage, piece_idx, piece_value = agent_pieces[np.random.randint(len(agent_pieces))]
        
        # High base costs that get reduced by leadership
        base_cost = 25.0  # Even higher initial cost
        time_cost = 5.0   # Even higher time cost
        
        # Calculate efficiency multiplier based on leadership structure
        efficiency_multiplier = self._calculate_efficiency_multiplier()
        
        # Early stage bonus to help get started
        if self.current_stage == 0 and len(self.shared_pieces) < 2:
            efficiency_multiplier *= 0.5  # 50% discount on first stage
        
        if granted:
            # Update collective understanding
            old_understanding = self.current_understanding[stage].copy()
            self.current_understanding[stage][piece_idx] = piece_value
            self.shared_pieces.add((stage, piece_idx))
            
            # Calculate quality improvement
            old_quality = self._evaluate_stage_understanding(old_understanding)
            new_quality = self._evaluate_stage_understanding()
            quality = max(0, new_quality - old_quality)
            
            # Apply efficiency to costs
            final_cost = base_cost * efficiency_multiplier
            final_time = time_cost * efficiency_multiplier
            
            # Update resources and time
            self.resources = max(0, self.resources - final_cost)
            self.time_spent += final_time
            
            # Check if this stage is complete - lower threshold for first stage
            completion_threshold = 0.4 if self.current_stage == 0 else 0.5
            stage_complete = self._evaluate_stage_understanding() > completion_threshold
            
            if stage_complete and self.current_stage not in self.stage_leaders:
                self.stage_leaders[self.current_stage] = from_agent
            
            # If stage complete, move to next stage
            if stage_complete:
                self.current_stage = min(self.current_stage + 1, self.num_stages - 1)
            
            return {
                "shared_info": (stage, piece_idx),
                "value": piece_value,
                "success": True,
                "quality": quality,
                "cost": final_cost,
                "time": final_time,
                "stage_complete": stage_complete,
                "moves_closer": new_quality > old_quality
            }
        else:
            # Failed claims still cost resources but less
            failed_cost = base_cost * 0.5  # 50% of base cost
            failed_time = time_cost * 0.5  # 50% of time cost
            
            # Apply efficiency to failed costs too
            final_failed_cost = failed_cost * efficiency_multiplier
            final_failed_time = failed_time * efficiency_multiplier
            
            # Update resources and time
            self.resources = max(0, self.resources - final_failed_cost)
            self.time_spent += final_failed_time
            
            return {
                "shared_info": (stage, piece_idx),
                "value": piece_value,
                "success": False,
                "quality": 0.0,
                "cost": final_failed_cost,
                "time": final_failed_time,
                "stage_complete": False,
                "moves_closer": False
            }
    
    def _evaluate_stage_understanding(self, understanding=None) -> float:
        """Evaluate how close current stage understanding is to truth."""
        if understanding is None:
            understanding = self.current_understanding[self.current_stage]
            
        # Calculate quality based on shared pieces only
        pieces_per_stage = self.puzzle_size // self.num_stages
        shared_pieces = [(s, p) for s, p in self.shared_pieces if s == self.current_stage]
        shared_count = len(shared_pieces)
        
        if shared_count == 0:
            return 0.0  # No progress if no pieces shared
            
        # Calculate error only on shared pieces
        shared_indices = [p for _, p in shared_pieces]
        shared_understanding = understanding[shared_indices]
        shared_truth = self.true_solutions[self.current_stage][shared_indices]
        
        error = np.mean(np.abs(shared_understanding - shared_truth))
        quality = 1 - error
        
        # Weight by completion
        completion = shared_count / pieces_per_stage
        return quality * completion
    
    def _evaluate_understanding(self) -> float:
        """Evaluate overall solution quality across all stages."""
        stage_scores = []
        for stage in range(self.num_stages):
            # Get shared pieces for this stage
            shared_pieces = [(s, p) for s, p in self.shared_pieces if s == stage]
            shared_count = len(shared_pieces)
            
            if shared_count == 0:
                stage_scores.append(0.0)
                continue
                
            # Calculate error only on shared pieces
            shared_indices = [p for _, p in shared_pieces]
            shared_understanding = self.current_understanding[stage][shared_indices]
            shared_truth = self.true_solutions[stage][shared_indices]
            
            error = np.mean(np.abs(shared_understanding - shared_truth))
            quality = 1 - error
            
            # Weight by completion
            pieces_per_stage = self.puzzle_size // self.num_stages
            completion = shared_count / pieces_per_stage
            stage_scores.append(quality * completion)
        
        return np.mean(stage_scores)
        
    def evaluate_current_solution(self) -> float:
        """Evaluate current group solution."""
        # Calculate base quality score
        quality_score = self._evaluate_understanding()
        
        # Calculate resource efficiency
        resource_score = self.resources / self.initial_resources
        
        # Calculate time efficiency
        time_efficiency = 1.0 - (self.time_spent / (self.num_stages * self.puzzle_size))
        
        # Weight quality more heavily but consider efficiency
        return 0.6 * quality_score + 0.2 * resource_score + 0.2 * time_efficiency 