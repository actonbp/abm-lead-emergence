# Leadership Emergence Simulation (Work in Progress)

**Author**: Bryan Acton

This project simulates how leadership emerges in groups through a simple "claim and grant" process: people can claim leadership, and others can choose to grant or deny those claims.

## Quick Start

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: 
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install requirements: `pip install -r requirements.txt`

## File Formats Explained

We use two main file formats:

1. **YAML Files** (`.yaml`): For human-readable settings
   - Found in: `config/*.yaml`
   - Used for: Setting up simulations
   - Example:
   ```yaml
   # This controls how many agents are in the simulation
   n_agents: 4
   
   # This affects how often agents try to claim leadership
   claim_rate: 0.5
   ```

2. **JSON Files** (`.json`): For storing results
   - Found in: `outputs/*.json`
   - Used for: Saving simulation data
   - Example:
   ```json
   {
     "time_step": 1,
     "leader_score": 0.75
   }
   ```

## Project Structure

- `src/`: Main code
  - `models/`: The simulation models
  - `simulation/`: Running simulations
  - `analysis/`: Analyzing results
  - `app/`: Interactive visualization (work in progress)

- `scripts/`: Ready-to-use analysis tools
  - `run_simulation.py`: Run a basic simulation
  - `parameter_sweep.py`: Try different settings
  - `analyze_results.py`: Look at the results

- `config/`: Settings files (in YAML)
  - `base.yaml`: Basic settings
  - `variants/`: Different experiment settings

- `data/`: Where data is stored
  - `raw/`: Original simulation outputs
  - `processed/`: Analyzed results

## Running Experiments

1. Basic simulation:
   ```bash
   python scripts/run_simulation.py
   ```

2. Try different parameters:
   ```bash
   python scripts/parameter_sweep.py
   ```

3. Look at results:
   ```bash
   python scripts/analyze_results.py
   ```

## Development Status

This project is actively being developed. The Shiny app for visualization is still in progress.

## AI Tool Usage

This project was developed with assistance from AI tools including GPT models and Claude (via Cursor).

## Questions or Issues?

Feel free to open an issue or contact the author directly.

## License

[Insert License Information]
