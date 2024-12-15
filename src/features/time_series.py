"""
Feature extraction from leadership emergence simulation time series data.
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from scipy import stats
import pandas as pd
import warnings
from collections import defaultdict


@dataclass
class TimeSeriesFeatures:
    """Container for extracted time series features."""
    # Basic statistics
    mean_final_li: float
    std_final_li: float
    mean_final_fi: float
    std_final_fi: float
    
    # Convergence features
    time_to_li_stability: int
    time_to_fi_stability: int
    final_li_variance: float
    final_fi_variance: float
    
    # Role differentiation
    mean_role_diff: float
    max_role_diff: float
    role_diff_variance: float
    role_polarization: float
    
    # Trend features
    li_trend_slope: float
    fi_trend_slope: float
    li_trend_r2: float
    fi_trend_r2: float
    
    # Emergence features
    emergence_speed: float
    leader_consistency: float
    follower_consistency: float


class TimeSeriesFeatureExtractor:
    """Extracts features from simulation time series data."""
    
    def __init__(self, stability_threshold: float = 0.1):
        """Initialize feature extractor.
        
        Args:
            stability_threshold: Threshold for considering a time series stable
        """
        self.stability_threshold = stability_threshold
    
    def extract_features(self, simulation_results: Dict[str, Any]) -> TimeSeriesFeatures:
        """Extract features from simulation results.
        
        Args:
            simulation_results: Dictionary containing simulation results
            
        Returns:
            TimeSeriesFeatures object containing extracted features
        """
        history = simulation_results.get('history', [])
        
        # Handle empty history
        if not history:
            # Use final state if available
            if 'final_state' in simulation_results:
                final_state = simulation_results['final_state']
                return self._handle_single_step(
                    final_state['leader_identities'],
                    final_state.get('follower_identities', [0.0] * len(final_state['leader_identities']))
                )
            else:
                # Use top-level state
                return self._handle_single_step(
                    simulation_results['leader_identities'],
                    simulation_results.get('follower_identities', [0.0] * len(simulation_results['leader_identities']))
                )
        
        # Convert history to numpy arrays
        li_history = np.array([
            state['leader_identities'] for state in history
        ])
        fi_history = np.array([
            state.get('follower_identities', [0.0] * len(state['leader_identities']))
            for state in history
        ])
        
        # Add final state if available and different from last history state
        if 'final_state' in simulation_results:
            final_li = simulation_results['final_state']['leader_identities']
            final_fi = simulation_results['final_state'].get('follower_identities', [0.0] * len(final_li))
            
            # Only add if different from last state
            if len(li_history) == 0 or not np.array_equal(final_li, li_history[-1]):
                li_history = np.vstack([li_history, final_li]) if len(li_history) > 0 else np.array([final_li])
                fi_history = np.vstack([fi_history, final_fi]) if len(fi_history) > 0 else np.array([final_fi])
        
        # Add top-level state if available and different from last state
        if 'leader_identities' in simulation_results:
            top_li = simulation_results['leader_identities']
            top_fi = simulation_results.get('follower_identities', [0.0] * len(top_li))
            
            # Only add if different from last state
            if len(li_history) == 0 or not np.array_equal(top_li, li_history[-1]):
                li_history = np.vstack([li_history, top_li]) if len(li_history) > 0 else np.array([top_li])
                fi_history = np.vstack([fi_history, top_fi]) if len(fi_history) > 0 else np.array([top_fi])
        
        # Handle single step case
        if len(li_history) == 1:
            return self._handle_single_step(li_history[0], fi_history[0])
        
        # Extract all feature types
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            basic_stats = self._calculate_basic_stats(li_history, fi_history)
            convergence = self._calculate_convergence_features(li_history, fi_history)
            role_features = self._calculate_role_features(li_history, fi_history)
            trend_features = self._calculate_trend_features(li_history, fi_history)
            emergence = self._calculate_emergence_features(li_history, fi_history)
        
        # Combine all features
        return TimeSeriesFeatures(
            **basic_stats,
            **convergence,
            **role_features,
            **trend_features,
            **emergence
        )
    
    def _handle_single_step(
        self,
        li_final: np.ndarray,
        fi_final: np.ndarray
    ) -> TimeSeriesFeatures:
        """Handle the special case of single step simulation."""
        return TimeSeriesFeatures(
            # Basic statistics
            mean_final_li=float(np.mean(li_final)),
            std_final_li=float(np.std(li_final)),
            mean_final_fi=float(np.mean(fi_final)),
            std_final_fi=float(np.std(fi_final)),
            
            # Convergence features (no convergence possible)
            time_to_li_stability=0,
            time_to_fi_stability=0,
            final_li_variance=float(np.var(li_final)),
            final_fi_variance=float(np.var(fi_final)),
            
            # Role differentiation
            mean_role_diff=float(np.mean(li_final - fi_final)),
            max_role_diff=float(np.max(np.abs(li_final - fi_final))),
            role_diff_variance=float(np.var(li_final - fi_final)),
            role_polarization=float(self._calculate_polarization(li_final - fi_final)),
            
            # Trend features (no trend possible)
            li_trend_slope=0.0,
            fi_trend_slope=0.0,
            li_trend_r2=0.0,
            fi_trend_r2=0.0,
            
            # Emergence features (no emergence possible)
            emergence_speed=0.0,
            leader_consistency=1.0,
            follower_consistency=1.0
        )
    
    def _calculate_basic_stats(
        self,
        li_history: np.ndarray,
        fi_history: np.ndarray
    ) -> Dict[str, float]:
        """Calculate basic statistical features."""
        return {
            'mean_final_li': float(np.mean(li_history[-1])),
            'std_final_li': float(np.std(li_history[-1])),
            'mean_final_fi': float(np.mean(fi_history[-1])),
            'std_final_fi': float(np.std(fi_history[-1]))
        }
    
    def _calculate_convergence_features(
        self,
        li_history: np.ndarray,
        fi_history: np.ndarray
    ) -> Dict[str, float]:
        """Calculate features related to convergence and stability."""
        # Calculate variances over time
        li_var = np.var(li_history, axis=1)
        fi_var = np.var(fi_history, axis=1)
        
        # Find when variance stabilizes
        li_stable = np.where(li_var < self.stability_threshold)[0]
        fi_stable = np.where(fi_var < self.stability_threshold)[0]
        
        return {
            'time_to_li_stability': int(li_stable[0] if len(li_stable) > 0 else len(li_var)),
            'time_to_fi_stability': int(fi_stable[0] if len(fi_stable) > 0 else len(fi_var)),
            'final_li_variance': float(li_var[-1]),
            'final_fi_variance': float(fi_var[-1])
        }
    
    def _calculate_role_features(
        self,
        li_history: np.ndarray,
        fi_history: np.ndarray
    ) -> Dict[str, float]:
        """Calculate features related to role differentiation."""
        # Calculate role differences
        role_diff = li_history - fi_history
        final_diff = role_diff[-1]
        
        return {
            'mean_role_diff': float(np.mean(final_diff)),
            'max_role_diff': float(np.max(np.abs(final_diff))),
            'role_diff_variance': float(np.var(final_diff)),
            'role_polarization': float(self._calculate_polarization(final_diff))
        }
    
    def _calculate_trend_features(
        self,
        li_history: np.ndarray,
        fi_history: np.ndarray
    ) -> Dict[str, float]:
        """Calculate features related to identity trends."""
        time = np.arange(len(li_history))
        
        # Calculate mean trends
        li_mean = np.mean(li_history, axis=1)
        fi_mean = np.mean(fi_history, axis=1)
        
        # Handle constant values
        if np.all(li_mean == li_mean[0]):
            li_slope, li_r_value = 0.0, 0.0
        else:
            li_slope, _, li_r_value, _, _ = stats.linregress(time, li_mean)
        
        if np.all(fi_mean == fi_mean[0]):
            fi_slope, fi_r_value = 0.0, 0.0
        else:
            fi_slope, _, fi_r_value, _, _ = stats.linregress(time, fi_mean)
        
        return {
            'li_trend_slope': float(li_slope),
            'fi_trend_slope': float(fi_slope),
            'li_trend_r2': float(li_r_value ** 2),
            'fi_trend_r2': float(fi_r_value ** 2)
        }
    
    def _calculate_emergence_features(
        self,
        li_history: np.ndarray,
        fi_history: np.ndarray
    ) -> Dict[str, float]:
        """Calculate features related to leadership emergence."""
        # Calculate emergence speed (time to first leader)
        leader_threshold = 0.7
        leader_mask = li_history >= leader_threshold
        
        if np.any(leader_mask):
            first_leader = np.where(np.any(leader_mask, axis=1))[0][0]
            emergence_speed = 1.0 / (first_leader + 1)  # Add 1 to avoid division by zero
        else:
            emergence_speed = 0.0
        
        # Calculate consistency of leader/follower roles
        leader_consistency = self._calculate_role_consistency(li_history >= leader_threshold)
        follower_consistency = self._calculate_role_consistency(fi_history >= leader_threshold)
        
        return {
            'emergence_speed': float(emergence_speed),
            'leader_consistency': float(leader_consistency),
            'follower_consistency': float(follower_consistency)
        }
    
    def _calculate_polarization(self, values: np.ndarray) -> float:
        """Calculate polarization as distance from uniform distribution."""
        hist, _ = np.histogram(values, bins=10, range=(-1, 1), density=True)
        uniform = np.ones_like(hist) / len(hist)
        return float(np.sqrt(np.mean((hist - uniform) ** 2)))
    
    def _calculate_role_consistency(self, role_mask: np.ndarray) -> float:
        """Calculate consistency of roles over time."""
        if len(role_mask) <= 1:
            return 1.0
            
        transitions = np.sum(role_mask[1:] != role_mask[:-1], axis=0)
        max_transitions = len(role_mask) - 1
        consistency = 1 - transitions / max_transitions
        return float(np.mean(consistency))


class BatchFeatureExtractor:
    """Extracts features from batches of simulation results."""
    
    def __init__(self, feature_extractor: TimeSeriesFeatureExtractor):
        """Initialize batch feature extractor.
        
        Args:
            feature_extractor: Feature extractor for individual simulations
        """
        self.feature_extractor = feature_extractor
    
    def extract_batch_features(
        self,
        batch_dir: str,
        output_file: str = None
    ) -> pd.DataFrame:
        """Extract features from all simulations in a batch.
        
        Args:
            batch_dir: Directory containing batch simulation results
            output_file: Optional path to save features DataFrame
            
        Returns:
            DataFrame containing extracted features for all simulations
        """
        import glob
        import json
        from pathlib import Path
        
        # Load all result files
        result_files = glob.glob(str(Path(batch_dir) / "*.json"))
        features_list = []
        
        for file_path in result_files:
            # Skip metadata file
            if "metadata" in file_path:
                continue
                
            # Load simulation results
            with open(file_path) as f:
                results = json.load(f)
            
            try:
                # Extract features
                features = self.feature_extractor.extract_features(results)
                
                # Get parameters from top level or fallback to final state/history
                parameters = results.get('parameters', {})
                if not parameters and 'final_state' in results:
                    parameters = results['final_state'].get('parameters', {})
                elif not parameters and results.get('history', []):
                    parameters = results['history'][0].get('parameters', {})
                
                # Add metadata
                features_dict = {
                    'run_id': Path(file_path).stem,
                    **parameters,
                    **features.__dict__
                }
                features_list.append(features_dict)
                
                if len(features_list) % 10 == 0:
                    print(f"Processed {len(features_list)} simulations...")
                
            except Exception as e:
                warnings.warn(f"Failed to extract features from {file_path}: {str(e)}")
                print(f"Error details: {str(e)}")  # Add more detailed error reporting
        
        if not features_list:
            warnings.warn("No features could be extracted from any simulation")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(features_list)
        
        # Save if output file specified
        if output_file:
            df.to_csv(output_file, index=False)
        
        return df 