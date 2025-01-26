# Scripts Directory

This directory contains entry point scripts for running simulations, analysis, and parameter sweeps. These scripts use the core functionality implemented in `src/`.

## Script Overview

- `test_base_model.py`: Runs basic tests of the leadership emergence model
  - Validates core claim-grant mechanics
  - Prints detailed interaction logs
  - Useful for quick verification of model behavior

- `parameter_sweep.py`: Explores parameter space systematically
  - Runs simulations with different parameter combinations
  - Saves results for analysis
  - Uses parameter configurations from `config/parameter_sweep.yaml`

- `analyze_emergence.py`: Analyzes leadership emergence patterns
  - Processes simulation results
  - Generates emergence metrics
  - Creates visualization plots
  - Uses analysis components from `src/analysis/`

- `analyze_results.py`: Detailed analysis of simulation results
  - Computes summary statistics
  - Generates plots and visualizations
  - Uses analysis tools from `src/analysis/`

- `run_simulation.py`: Basic simulation runner
  - Runs single simulation with specified parameters
  - Saves results and generates basic plots
  - Good starting point for understanding the model

## Usage Pattern

These scripts are entry points that compose functionality from `src/`. They should:
1. Parse command line arguments or config files
2. Set up logging and output directories
3. Call appropriate functions from `src/`
4. Save results and generate visualizations

Core logic and reusable components should live in `src/`, while these scripts provide convenient ways to use that functionality. 