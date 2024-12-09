# Leadership Emergence Model Analysis Framework

## Project Overview
This framework provides a comprehensive system for simulating, analyzing, and understanding leadership emergence patterns in social groups. It combines interactive exploration through a web interface with robust batch processing capabilities for machine learning analysis.

## System Architecture

### 1. Core Components

#### Model Layer (`src/models/`)
- `base_model.py`: Abstract base class defining the leadership model interface
- `schema_model.py`: Implementation of schema-based leadership emergence model
- Extensible for additional model implementations

#### Simulation Engine (`src/simulation/`)
- `runner.py`: Manages batch execution of simulations
- Handles parameter space exploration
- Provides parallel processing capabilities
- Saves raw simulation data

#### Feature Engineering (`src/features/`)
- `time_series.py`: Extracts meaningful features from simulation trajectories
- Captures:
  - Basic statistics (means, variances)
  - Stability metrics
  - Role differentiation
  - Temporal trends

#### Analysis Pipeline (`src/analysis/`)
- `pipeline.py`: Orchestrates the analysis workflow
- `clustering.py`: Initial pattern detection
- `ml_analysis.py`: Advanced machine learning analysis
  - Principal Component Analysis (PCA)
  - K-means clustering
  - t-SNE visualization
  - Parameter distribution analysis

#### Web Interface (`src/app/`)
- `app.py`: Interactive web application
- Real-time simulation visualization
- Step-by-step model exploration
- Parameter tuning interface

## Workflow

### 1. Interactive Exploration
```bash
python src/app/app.py
```
- Launch web interface
- Explore model behavior
- Test different parameters
- Visualize single simulations

### 2. Batch Analysis
```bash
python src/run_analysis.py --output-dir data/results
```
- Run multiple simulations
- Explore parameter space
- Generate raw data
- Extract features

### 3. ML Analysis
```bash
python src/analysis/ml_analysis.py --results-dir data/results --output-dir data/ml_analysis
```
- Process simulation results
- Identify patterns
- Generate visualizations
- Produce insights

## Directory Structure
```
.
├── config/
│   └── analysis_config.json    # Analysis configuration
├── src/
│   ├── app/                    # Web interface
│   ├── models/                 # Model implementations
│   ├── simulation/             # Simulation engine
│   ├── features/               # Feature extraction
│   └── analysis/               # Analysis tools
├── data/
│   ├── raw/                    # Raw simulation data
│   ├── results/                # Processed results
│   └── ml_analysis/           # ML outputs
└── README.md                   # Project documentation
```

## Data Flow
1. **Model Definition**
   - Define leadership emergence models
   - Implement model dynamics
   - Specify parameters

2. **Data Generation**
   - Run simulations across parameter space
   - Store raw trajectories
   - Extract features

3. **Analysis Pipeline**
   - Load simulation results
   - Perform feature engineering
   - Run ML analyses
   - Generate visualizations

4. **Insights Generation**
   - Identify emergence patterns
   - Analyze parameter effects
   - Visualize relationships
   - Generate reports

## Configuration
The system is configured through `config/analysis_config.json`:
- Model parameters to explore
- Number of simulation steps
- Number of replications
- Analysis parameters

## Extending the Framework
1. **New Models**
   - Inherit from `BaseLeadershipModel`
   - Implement required methods
   - Add to model registry

2. **New Features**
   - Add extraction functions to `time_series.py`
   - Update analysis pipeline

3. **New Analyses**
   - Add analysis modules
   - Extend ML pipeline
   - Create visualizations

## Output Products
1. **Raw Data**
   - Simulation trajectories
   - Parameter combinations
   - Feature vectors

2. **Analysis Results**
   - PCA components
   - Cluster assignments
   - Statistical measures

3. **Visualizations**
   - t-SNE plots
   - Parameter distributions
   - Cluster characteristics

## Usage Examples
See `README.md` for detailed usage examples and tutorials. 