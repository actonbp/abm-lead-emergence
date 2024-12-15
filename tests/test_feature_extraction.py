"""
Tests for feature extraction functionality.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import json

from src.features.time_series import (
    TimeSeriesFeatureExtractor,
    BatchFeatureExtractor,
    TimeSeriesFeatures
)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


def create_mock_simulation_results(
    n_agents: int = 10,
    n_steps: int = 100,
    random_seed: int = None
):
    """Create mock simulation results for testing."""
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Create mock time series data
    time = np.linspace(0, 1, n_steps)
    li_history = np.zeros((n_steps, n_agents))
    fi_history = np.zeros((n_steps, n_agents))
    
    # Add some patterns
    for i in range(n_agents):
        # Leader identities increase over time with noise
        li_history[:, i] = 0.5 + 0.3 * time + 0.1 * np.random.randn(n_steps)
        # Follower identities decrease over time with noise
        fi_history[:, i] = 0.5 - 0.2 * time + 0.1 * np.random.randn(n_steps)
    
    # Clip values to [0, 1]
    li_history = np.clip(li_history, 0, 1)
    fi_history = np.clip(fi_history, 0, 1)
    
    # Create history
    history = []
    for t in range(n_steps):
        history.append({
            'leader_identities': li_history[t],
            'follower_identities': fi_history[t],
            'schemas': np.random.rand(n_agents)  # Random schemas
        })
    
    return {
        'history': history,
        'parameters': {
            'n_agents': n_agents,
            'li_change_rate': 0.1,
            'schema_weight': 0.5
        },
        'metadata': {
            'run_id': 'test_run',
            'n_steps': n_steps,
            'random_seed': random_seed
        }
    }


def test_feature_extraction():
    """Test basic feature extraction."""
    results = create_mock_simulation_results(random_seed=42)
    extractor = TimeSeriesFeatureExtractor()
    
    features = extractor.extract_features(results)
    
    # Check feature types
    assert isinstance(features, TimeSeriesFeatures)
    
    # Check basic statistics
    assert 0 <= features.mean_final_li <= 1
    assert 0 <= features.mean_final_fi <= 1
    assert features.std_final_li >= 0
    assert features.std_final_fi >= 0
    
    # Check convergence features
    assert features.time_to_li_stability >= 0
    assert features.time_to_fi_stability >= 0
    assert features.final_li_variance >= 0
    assert features.final_fi_variance >= 0
    
    # Check trend features
    assert isinstance(features.li_trend_slope, float)
    assert isinstance(features.fi_trend_slope, float)
    assert 0 <= features.li_trend_r2 <= 1
    assert 0 <= features.fi_trend_r2 <= 1


def test_batch_feature_extraction():
    """Test batch feature extraction."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock batch results
        batch_dir = Path(temp_dir) / "batch_test"
        batch_dir.mkdir()
        
        # Create multiple simulation results
        n_sims = 3
        for i in range(n_sims):
            results = create_mock_simulation_results(
                random_seed=42 + i,
                n_steps=50  # Smaller for faster tests
            )
            results['metadata']['run_id'] = f'test_run_{i}'
            
            with open(batch_dir / f"sim_{i}.json", 'w') as f:
                json.dump(results, f, cls=NumpyEncoder)
        
        # Create metadata file
        with open(batch_dir / "metadata.json", 'w') as f:
            json.dump({'n_simulations': n_sims}, f)
        
        # Extract features
        extractor = TimeSeriesFeatureExtractor()
        batch_extractor = BatchFeatureExtractor(extractor)
        
        # Test without saving
        df = batch_extractor.extract_batch_features(str(batch_dir))
        assert len(df) == n_sims
        assert 'run_id' in df.columns
        assert 'mean_final_li' in df.columns
        
        # Test with saving
        output_file = batch_dir / "features.csv"
        df = batch_extractor.extract_batch_features(
            str(batch_dir),
            str(output_file)
        )
        assert output_file.exists()


def test_feature_stability():
    """Test stability of feature extraction."""
    # Create two identical simulations
    results1 = create_mock_simulation_results(random_seed=42)
    results2 = create_mock_simulation_results(random_seed=42)
    
    extractor = TimeSeriesFeatureExtractor()
    
    features1 = extractor.extract_features(results1)
    features2 = extractor.extract_features(results2)
    
    # Check that features are identical
    for field in features1.__dataclass_fields__:
        val1 = getattr(features1, field)
        val2 = getattr(features2, field)
        assert np.allclose(val1, val2), f"Feature {field} not stable"


def test_edge_cases():
    """Test feature extraction with edge cases."""
    # Test with minimal simulation
    min_results = create_mock_simulation_results(
        n_agents=2,
        n_steps=10,
        random_seed=42
    )
    extractor = TimeSeriesFeatureExtractor()
    features = extractor.extract_features(min_results)
    assert isinstance(features, TimeSeriesFeatures)
    
    # Test with constant values
    const_results = create_mock_simulation_results(
        n_agents=3,
        n_steps=10,
        random_seed=42
    )
    for state in const_results['history']:
        state['leader_identities'] = np.array([0.5, 0.5, 0.5])
        state['follower_identities'] = np.array([0.3, 0.3, 0.3])
    
    features = extractor.extract_features(const_results)
    assert features.std_final_li == 0
    assert features.std_final_fi == 0
    assert features.li_trend_slope == 0
    
    # Test with single step
    single_step = create_mock_simulation_results(
        n_agents=3,
        n_steps=1,
        random_seed=42
    )
    features = extractor.extract_features(single_step)
    assert isinstance(features, TimeSeriesFeatures)
    assert features.li_trend_slope == 0
    assert features.fi_trend_slope == 0
    
    # Test with constant values
    const_results = create_mock_simulation_results(
        n_agents=3,
        n_steps=10,
        random_seed=42
    )
    for state in const_results['history']:
        state['leader_identities'] = np.array([0.5, 0.5, 0.5])
        state['follower_identities'] = np.array([0.3, 0.3, 0.3])
    
    features = extractor.extract_features(const_results)
    assert features.std_final_li == 0
    assert features.std_final_fi == 0
    assert features.li_trend_slope == 0
    
    # Test with single step
    single_step = create_mock_simulation_results(
        n_agents=3,
        n_steps=1,
        random_seed=42
    )
    features = extractor.extract_features(single_step)
    assert isinstance(features, TimeSeriesFeatures)
    assert features.li_trend_slope == 0
    assert features.fi_trend_slope == 0