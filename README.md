# Leadership Emergence in Teams: Machine Learning Analysis of Agent-Based Models

This repository explores leadership emergence in teams using agent-based modeling (ABM) and machine learning techniques. We aim to identify key conditions and mechanisms that lead to stable leadership patterns across different theoretical scenarios.

## Project Structure

```
├── src/
│   ├── models/                    # Core model implementations
│   │   ├── base_model.py         # Base leadership emergence model
│   │   ├── schema_model.py       # Schema-based extensions
│   │   ├── memory_model.py       # Models with memory effects
│   │   └── structural_model.py   # Models with structural preferences
│   │
│   ├── simulation/               # Simulation management
│   │   ├── runner.py            # Simulation execution engine
│   │   ├── parameter_space.py   # Parameter space definition
│   │   └── data_collector.py    # Data collection utilities
│   │
│   ├── features/                 # Feature extraction
│   │   ├── time_series.py       # Time series feature extraction
│   │   ├── network.py           # Network-based features
│   │   └── stability.py         # Stability metrics
│   │
│   ├── analysis/                 # Analysis tools
│   │   ├── dimensionality.py    # Dimensionality reduction
│   │   ├── clustering.py        # Clustering analysis
│   │   └── importance.py        # Feature importance analysis
│   │
│   ├── visualization/            # Visualization tools
│   │   ├── network_viz.py       # Network visualizations
│   │   ├── time_series_viz.py   # Time series plots
│   │   └── pattern_viz.py       # Pattern visualization
│   │
│   └── app/                      # Interactive applications
│       ├── simulation_app.py     # Simulation interface
│       └── analysis_app.py       # Analysis dashboard
│
├── notebooks/                    # Analysis notebooks
│   ├── model_exploration/       # Model behavior exploration
│   ├── feature_analysis/        # Feature extraction analysis
│   └── pattern_analysis/        # Pattern discovery analysis
│
├── data/                        # Data storage
│   ├── raw/                     # Raw simulation outputs
│   ├── processed/               # Processed features
│   └── results/                 # Analysis results
│
├── tests/                       # Test suite
│   ├── model_tests/            # Model unit tests
│   ├── simulation_tests/       # Simulation tests
│   └── analysis_tests/         # Analysis pipeline tests
│
└── docs/                        # Documentation
    ├── models/                  # Model documentation
    ├── features/               # Feature documentation
    └── analysis/               # Analysis documentation
```

## Core Components

### 1. Models
- Base leadership emergence model with pairwise interactions
- Extensions for different theoretical assumptions:
  - Schema-based decision making
  - Memory effects and learning
  - Structural preferences
  - Network effects

### 2. Simulation Framework
- Parameter space exploration
- Batch simulation execution
- Data collection and storage
- Progress tracking and monitoring

### 3. Feature Extraction
- Time series features
- Network metrics
- Stability measures
- Pattern indicators

### 4. Analysis Pipeline
- Dimensionality reduction (PCA, UMAP)
- Clustering (k-means, HDBSCAN)
- Feature importance analysis
- Pattern discovery

### 5. Visualization Tools
- Interactive network visualization
- Time series plotting
- Pattern visualization
- Analysis dashboards

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/leadership-emergence-ml.git
cd leadership-emergence-ml
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run tests:
```bash
python -m pytest tests/
```

5. Start the simulation app:
```bash
python src/app/simulation_app.py
```

## Usage Examples

See the `notebooks/` directory for example analyses and the `docs/` directory for detailed documentation.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
