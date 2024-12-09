"""
Feature extraction from time series data of leadership emergence simulations.
"""

import numpy as np
from typing import Dict, List, Any
from scipy import stats

def extract_time_series_features(history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Extract features from simulation time series data."""
    
    # Convert history to numpy arrays
    li_history = np.array([
        state['leader_identities'] for state in history
    ])
    fi_history = np.array([
        state['follower_identities'] for state in history
    ])
    
    features = {}
    
    # Basic statistics
    features.update({
        'mean_final_li': np.mean(li_history[-1]),
        'mean_final_fi': np.mean(fi_history[-1]),
        'std_final_li': np.std(li_history[-1]),
        'std_final_fi': np.std(fi_history[-1])
    })
    
    # Time to stability
    features.update(_calculate_stability_features(li_history, fi_history))
    
    # Role differentiation
    features.update(_calculate_role_features(li_history, fi_history))
    
    # Trend features
    features.update(_calculate_trend_features(li_history, fi_history))
    
    return features

def _calculate_stability_features(
    li_history: np.ndarray,
    fi_history: np.ndarray,
    threshold: float = 0.1
) -> Dict[str, float]:
    """Calculate features related to stability of identities."""
    
    # Calculate variances over time
    li_var = np.var(li_history, axis=1)
    fi_var = np.var(fi_history, axis=1)
    
    # Find when variance stabilizes
    li_stable = np.where(li_var < threshold)[0]
    fi_stable = np.where(fi_var < threshold)[0]
    
    features = {
        'time_to_li_stability': li_stable[0] if len(li_stable) > 0 else len(li_var),
        'time_to_fi_stability': fi_stable[0] if len(fi_stable) > 0 else len(fi_var),
        'final_li_variance': li_var[-1],
        'final_fi_variance': fi_var[-1]
    }
    
    return features

def _calculate_role_features(
    li_history: np.ndarray,
    fi_history: np.ndarray
) -> Dict[str, float]:
    """Calculate features related to role differentiation."""
    
    # Calculate role differences
    role_diff = li_history - fi_history
    
    features = {
        'mean_role_diff': np.mean(role_diff[-1]),
        'max_role_diff': np.max(np.abs(role_diff[-1])),
        'role_diff_variance': np.var(role_diff[-1]),
        'role_polarization': _calculate_polarization(role_diff[-1])
    }
    
    return features

def _calculate_trend_features(
    li_history: np.ndarray,
    fi_history: np.ndarray
) -> Dict[str, float]:
    """Calculate features related to identity trends."""
    
    # Calculate trends using linear regression
    time = np.arange(len(li_history))
    
    features = {}
    
    # Leader identity trends
    for i in range(li_history.shape[1]):
        slope, _, r_value, _, _ = stats.linregress(time, li_history[:, i])
        features[f'li_trend_agent_{i}'] = slope
        features[f'li_trend_r2_agent_{i}'] = r_value**2
    
    # Follower identity trends
    for i in range(fi_history.shape[1]):
        slope, _, r_value, _, _ = stats.linregress(time, fi_history[:, i])
        features[f'fi_trend_agent_{i}'] = slope
        features[f'fi_trend_r2_agent_{i}'] = r_value**2
    
    return features

def _calculate_polarization(values: np.ndarray) -> float:
    """Calculate polarization as distance from uniform distribution."""
    hist, _ = np.histogram(values, bins=10, density=True)
    uniform = np.ones_like(hist) / len(hist)
    return np.sum(np.abs(hist - uniform)) 