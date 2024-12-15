"""
Integration tests for the leadership emergence analysis system.
"""

import pytest
import tempfile
from pathlib import Path
import json
import pandas as pd
import numpy as np
import shutil

from models.schema_model import SchemaModel
from simulation.runner import BatchRunner, SimulationConfig
from features.time_series import TimeSeriesFeatureExtractor
from features.batch_extractor import BatchFeatureExtractor


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_configs():
    """Create test simulation configurations."""
    base_config = {
        "n_agents": 4,
        "initial_li_equal": False,
        "li_change_rate": 0.1,
        "schema_weight": 0.5,
        "claim_threshold": 0.6,
        "grant_threshold": 0.4
    }
    
    configs = []
    for n_agents in [4, 6]:
        for li_equal in [True, False]:
            config = SimulationConfig(
                model_params={
                    **base_config,
                    "n_agents": n_agents,
                    "initial_li_equal": li_equal
                },
                n_steps=10,
                random_seed=42
            )
            configs.append(config)
    
    return configs


def test_end_to_end_workflow(temp_output_dir, test_configs):
    """Test complete workflow from simulation to analysis."""
    # 1. Run batch simulations
    runner = BatchRunner(
        model_class=SchemaModel,
        output_dir=temp_output_dir,
        n_jobs=2
    )
    
    batch_id = "test_batch"
    result_files = runner.run_batch(test_configs, batch_id)
    
    # Verify simulation outputs
    assert len(result_files) == len(test_configs)
    batch_dir = temp_output_dir / f"batch_{batch_id}"
    assert batch_dir.exists()
    
    metadata_file = batch_dir / "metadata.json"
    assert metadata_file.exists()
    
    with open(metadata_file) as f:
        metadata = json.load(f)
        assert metadata["n_configurations"] == len(test_configs)
        assert metadata["model_class"] == "SchemaModel"
    
    # 2. Extract features
    feature_extractor = TimeSeriesFeatureExtractor()
    batch_extractor = BatchFeatureExtractor(
        feature_extractor,
        n_jobs=2
    )
    
    features_file = temp_output_dir / "features.csv"
    features_df = batch_extractor.extract_batch_features(
        str(batch_dir),
        str(features_file)
    )
    
    # Verify feature extraction
    assert not features_df.empty
    assert len(features_df) == len(test_configs)
    assert features_file.exists()
    
    # Check required columns
    required_columns = {
        'n_agents', 'initial_li_equal', 'li_change_rate',
        'mean_final_li', 'mean_final_fi', 
        'time_to_li_stability', 'time_to_fi_stability'
    }
    assert required_columns.issubset(features_df.columns)
    
    # Verify feature values
    assert features_df['n_agents'].nunique() == 2
    assert features_df['initial_li_equal'].nunique() == 2
    assert all(features_df['mean_final_li'] >= 0)
    assert all(features_df['mean_final_li'] <= 1)


def test_failed_simulation_handling(temp_output_dir):
    """Test system's handling of failed simulations."""
    # Create invalid config that should fail
    invalid_config = SimulationConfig(
        model_params={
            "n_agents": -1,  # Invalid value
            "initial_li_equal": True,
            "li_change_rate": 0.1
        },
        n_steps=10
    )
    
    runner = BatchRunner(
        model_class=SchemaModel,
        output_dir=temp_output_dir
    )
    
    # Run should complete but log errors
    result_files = runner.run_batch([invalid_config], "failed_test")
    assert len(result_files) == 0


def test_interrupted_workflow_recovery(temp_output_dir, test_configs):
    """Test system's ability to handle interruptions and recover."""
    runner = BatchRunner(
        model_class=SchemaModel,
        output_dir=temp_output_dir
    )
    
    # Run first half of configs
    first_half = test_configs[:len(test_configs)//2]
    batch_id = "interrupted_test"
    first_results = runner.run_batch(first_half, batch_id)
    
    # Simulate interruption by removing some files
    batch_dir = temp_output_dir / f"batch_{batch_id}"
    for file in list(batch_dir.glob("*_run_*.json"))[:2]:
        file.unlink()
    
    # Run feature extraction
    feature_extractor = TimeSeriesFeatureExtractor()
    batch_extractor = BatchFeatureExtractor(feature_extractor)
    
    features_df = batch_extractor.extract_batch_features(str(batch_dir))
    
    # Should have features for remaining files
    expected_count = len(first_half) - 2
    assert len(features_df) == expected_count


def test_parameter_space_coverage(temp_output_dir):
    """Test that analysis covers the intended parameter space."""
    # Create configs covering parameter space
    param_space = {
        "n_agents": [4, 6, 8],
        "initial_li_equal": [True, False],
        "li_change_rate": [0.1, 0.3, 0.5],
        "schema_weight": [0.3, 0.5, 0.7]
    }
    
    configs = []
    for n_agents in param_space["n_agents"]:
        for li_equal in param_space["initial_li_equal"]:
            for rate in param_space["li_change_rate"]:
                for weight in param_space["schema_weight"]:
                    config = SimulationConfig(
                        model_params={
                            "n_agents": n_agents,
                            "initial_li_equal": li_equal,
                            "li_change_rate": rate,
                            "schema_weight": weight,
                            "claim_threshold": 0.6,
                            "grant_threshold": 0.4
                        },
                        n_steps=10,
                        random_seed=42
                    )
                    configs.append(config)
    
    runner = BatchRunner(
        model_class=SchemaModel,
        output_dir=temp_output_dir,
        n_jobs=2
    )
    
    result_files = runner.run_batch(configs, "coverage_test")
    
    # Verify all parameter combinations were tested
    batch_dir = temp_output_dir / "batch_coverage_test"
    
    feature_extractor = TimeSeriesFeatureExtractor()
    batch_extractor = BatchFeatureExtractor(feature_extractor)
    features_df = batch_extractor.extract_batch_features(str(batch_dir))
    
    # Check parameter coverage
    for param, values in param_space.items():
        assert set(features_df[param].unique()) == set(values)
    
    # Verify expected number of combinations
    expected_count = (
        len(param_space["n_agents"]) *
        len(param_space["initial_li_equal"]) *
        len(param_space["li_change_rate"]) *
        len(param_space["schema_weight"])
    )
    assert len(features_df) == expected_count


def test_reproducibility(temp_output_dir):
    """Test that results are reproducible with same random seed."""
    config = SimulationConfig(
        model_params={
            "n_agents": 4,
            "initial_li_equal": False,
            "li_change_rate": 0.1,
            "schema_weight": 0.5,
            "claim_threshold": 0.6,
            "grant_threshold": 0.4
        },
        n_steps=10,
        random_seed=42
    )
    
    # Run simulation twice with same seed
    runner1 = BatchRunner(
        model_class=SchemaModel,
        output_dir=temp_output_dir / "run1"
    )
    runner2 = BatchRunner(
        model_class=SchemaModel,
        output_dir=temp_output_dir / "run2"
    )
    
    results1 = runner1.run_batch([config], "repro_test1")
    results2 = runner2.run_batch([config], "repro_test2")
    
    # Load and compare results
    with open(results1[0]) as f1, open(results2[0]) as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)
    
    # Compare histories
    for state1, state2 in zip(data1["history"], data2["history"]):
        assert np.allclose(
            state1["leader_identities"],
            state2["leader_identities"]
        )
        assert np.allclose(
            state1["follower_identities"],
            state2["follower_identities"]
        ) 