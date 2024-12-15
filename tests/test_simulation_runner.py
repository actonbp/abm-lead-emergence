"""
Tests for simulation runner.
"""

import pytest
import tempfile
from pathlib import Path
import json
import numpy as np

from models.schema_model import SchemaModel
from simulation.runner import BatchRunner, SimulationConfig, create_parameter_grid


def test_simulation_config():
    """Test simulation configuration validation."""
    # Valid config
    config = SimulationConfig(
        model_params={
            "n_agents": 10,
            "initial_li_equal": True,
            "li_change_rate": 0.1
        },
        n_steps=10,
        random_seed=42
    )
    assert config.n_steps == 10
    assert config.random_seed == 42
    
    # Invalid n_steps
    with pytest.raises(ValueError):
        SimulationConfig(
            model_params={"n_agents": 10},
            n_steps=-1
        )


def test_batch_runner_initialization():
    """Test batch runner initialization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = BatchRunner(
            model_class=SchemaModel,
            output_dir=Path(temp_dir)
        )
        
        # Check directory creation
        assert (Path(temp_dir) / "raw").exists()
        assert (Path(temp_dir) / "processed").exists()
        assert (Path(temp_dir) / "logs").exists()


def test_single_simulation():
    """Test running a single simulation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = BatchRunner(
            model_class=SchemaModel,
            output_dir=Path(temp_dir)
        )
        
        config = SimulationConfig(
            model_params={
                "n_agents": 10,
                "initial_li_equal": True,
                "li_change_rate": 0.1,
                "schema_weight": 0.5,
                "claim_threshold": 0.6,
                "grant_threshold": 0.4
            },
            n_steps=10,
            random_seed=42
        )
        
        # Create batch directory
        batch_dir = Path(temp_dir) / "batch_test"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = runner.run_single_simulation(config, "test_run", batch_dir)
        
        # Check result file
        assert Path(result_file).exists()
        
        # Load and check contents
        with open(result_file) as f:
            results = json.load(f)
            assert len(results["history"]) == 10
            assert "leader_identities" in results
            assert "parameters" in results


def test_batch_execution():
    """Test running a batch of simulations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = BatchRunner(
            model_class=SchemaModel,
            output_dir=Path(temp_dir),
            n_jobs=2
        )
        
        # Create multiple configurations
        configs = [
            SimulationConfig(
                model_params={
                    "n_agents": 10,
                    "initial_li_equal": True,
                    "li_change_rate": rate,
                    "schema_weight": 0.5,
                    "claim_threshold": 0.6,
                    "grant_threshold": 0.4
                },
                n_steps=10,
                random_seed=42 + i
            )
            for i, rate in enumerate([0.1, 0.2])
        ]
        
        result_files = runner.run_batch(configs, "test_batch")
        
        # Check results
        assert len(result_files) == 2
        assert all(Path(f).exists() for f in result_files)
        
        # Check metadata file
        batch_dir = Path(temp_dir) / "batch_test_batch"
        metadata_file = batch_dir / "metadata.json"
        assert metadata_file.exists()
        
        # Check metadata contents
        with open(metadata_file) as f:
            metadata = json.load(f)
            assert metadata["n_configurations"] == 2
            assert metadata["model_class"] == "SchemaModel"


def test_parameter_grid():
    """Test parameter grid generation."""
    param_grid = {
        "n_agents": [4, 6],
        "initial_li_equal": [True, False],
        "li_change_rate": [0.1, 0.2]
    }
    
    # Create all combinations
    configs = []
    for n_agents in param_grid["n_agents"]:
        for li_equal in param_grid["initial_li_equal"]:
            for rate in param_grid["li_change_rate"]:
                config = SimulationConfig(
                    model_params={
                        "n_agents": n_agents,
                        "initial_li_equal": li_equal,
                        "li_change_rate": rate,
                        "schema_weight": 0.5,
                        "claim_threshold": 0.6,
                        "grant_threshold": 0.4
                    },
                    n_steps=10
                )
                configs.append(config)
    
    # Run batch
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = BatchRunner(
            model_class=SchemaModel,
            output_dir=Path(temp_dir)
        )
        
        result_files = runner.run_batch(configs, "grid_test")
        
        # Check number of results
        assert len(result_files) == (
            len(param_grid["n_agents"]) *
            len(param_grid["initial_li_equal"]) *
            len(param_grid["li_change_rate"])
        ) 


def test_simulation_config_validation():
    """Test comprehensive validation of simulation configuration."""
    # Test valid configurations
    valid_config = SimulationConfig(
        model_params={"n_agents": 10},
        n_steps=5,
        random_seed=42
    )
    assert valid_config.n_steps == 5
    assert valid_config.random_seed == 42

    # Test invalid n_steps
    with pytest.raises(ValueError, match="n_steps must be positive"):
        SimulationConfig(
            model_params={"n_agents": 10},
            n_steps=0
        )
    
    with pytest.raises(ValueError, match="n_steps must be positive"):
        SimulationConfig(
            model_params={"n_agents": 10},
            n_steps=-5
        )


def test_batch_runner_error_handling():
    """Test error handling in batch execution."""
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = BatchRunner(
            model_class=SchemaModel,
            output_dir=Path(temp_dir)
        )
        
        # Test with invalid model parameters
        invalid_config = SimulationConfig(
            model_params={"invalid_param": 10},  # This should cause model initialization to fail
            n_steps=5,
            random_seed=42
        )
        
        with pytest.raises(Exception):  # Should catch model initialization error
            runner.run_single_simulation(
                invalid_config,
                "error_test",
                Path(temp_dir)
            )


def test_parameter_grid_generation():
    """Test parameter grid generation with various input types."""
    # Test basic parameter grid
    param_ranges = {
        "n_agents": [5, 10],
        "li_change_rate": [0.1, 0.2],
        "initial_li_equal": [True, False]
    }
    
    combinations = create_parameter_grid(param_ranges)
    assert len(combinations) == 8  # 2 x 2 x 2 combinations
    
    # Verify all combinations are unique
    unique_combinations = {
        frozenset(combo.items())
        for combo in combinations
    }
    assert len(unique_combinations) == len(combinations)
    
    # Verify all parameter values are present
    n_agents_values = {combo["n_agents"] for combo in combinations}
    assert n_agents_values == set([5, 10])
    
    li_rates = {combo["li_change_rate"] for combo in combinations}
    assert li_rates == set([0.1, 0.2])
    
    li_equal_values = {combo["initial_li_equal"] for combo in combinations}
    assert li_equal_values == set([True, False])


def test_batch_runner_parallel_execution():
    """Test parallel execution of batch runner."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with different numbers of workers
        for n_jobs in [1, 2]:
            runner = BatchRunner(
                model_class=SchemaModel,
                output_dir=Path(temp_dir),
                n_jobs=n_jobs
            )
            
            # Create multiple configurations
            configs = [
                SimulationConfig(
                    model_params={
                        "n_agents": 5,
                        "initial_li_equal": True,
                        "li_change_rate": 0.1,
                        "schema_weight": 0.5,
                        "claim_threshold": 0.6,
                        "grant_threshold": 0.4
                    },
                    n_steps=5,
                    random_seed=42 + i
                )
                for i in range(4)  # Run 4 simulations
            ]
            
            result_files = runner.run_batch(configs, f"parallel_test_{n_jobs}")
            
            # Check all simulations completed
            assert len(result_files) == 4
            
            # Check results are valid
            for file_path in result_files:
                with open(file_path) as f:
                    results = json.load(f)
                    assert "history" in results
                    assert len(results["history"]) == 5  # n_steps
                    assert "parameters" in results
                    assert "metadata" in results


def test_batch_runner_reproducibility():
    """Test reproducibility of simulation results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = BatchRunner(
            model_class=SchemaModel,
            output_dir=Path(temp_dir)
        )
        
        # Run same configuration twice
        config = SimulationConfig(
            model_params={
                "n_agents": 5,
                "initial_li_equal": True,
                "li_change_rate": 0.1,
                "schema_weight": 0.5,
                "claim_threshold": 0.6,
                "grant_threshold": 0.4
            },
            n_steps=5,
            random_seed=42
        )
        
        result1 = runner.run_single_simulation(
            config,
            "reproducibility_test_1",
            Path(temp_dir)
        )
        
        result2 = runner.run_single_simulation(
            config,
            "reproducibility_test_2",
            Path(temp_dir)
        )
        
        # Load and compare results
        with open(result1) as f1, open(result2) as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)
            
            # Compare histories
            assert data1["history"] == data2["history"]
            
            # Compare final states
            assert data1["final_state"] == data2["final_state"]
            
            # Compare leader identities
            assert data1["leader_identities"] == data2["leader_identities"]