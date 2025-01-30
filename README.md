# Leadership Emergence Simulation

> ⚠️ **Work in Progress Notice**: This repository is currently under active development. Many models and code components referenced in the documentation are still being implemented or refined. Additional theoretical perspectives and optimization features will be added soon. Please check back for updates or follow the repository for notifications.

An agent-based model exploring how leadership naturally emerges in groups through a "claim and grant" process, with multiple theoretical perspectives and Bayesian parameter optimization.

## 🚀 Quick Start

```bash
# Clone and enter the repository
git clone https://github.com/bacton/abm-lead-emergence.git
cd abm-lead-emergence

# Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run Bayesian optimization for model parameters
python scripts/optimize.py --model base
python scripts/optimize.py --model interactionist

# Run comparison with optimized parameters
python scripts/compare_perspectives.py

# Visualize results
python scripts/visualize.py
```

## 🔬 Framework Overview

```
                        OPTIMIZED MODEL FRAMEWORK
                        ========================

┌─────────────────────────────────────────────────────────────┐
│                 Bayesian Parameter Search                    │
│ ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│ │   Define    │ ─► │   Search    │ ─► │  Optimal    │      │
│ │Search Space │    │  Process    │    │ Parameters   │      │
│ └─────────────┘    └─────────────┘    └─────────────┘      │
└───────────────────────────┬─────────────────────────────────┘
                           ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Model Layer   │     │   Simulation    │     │    Analysis     │
│  (Optimized)    │     │    Engine       │     │     Layer       │
│ ┌─────────────┐ │     │                 │     │                 │
│ │ Base Model  │ │     │    Runs the     │     │  Processes &    │
│ └─────────────┘ │     │   simulation    │     │   Analyzes      │
│ ┌─────────────┐ │ ──► │   over time     │ ──► │    Results      │
│ │Interactionist│ │     │                 │     │                 │
│ └─────────────┘ │     │  Updates state   │     │ - ICC Metrics   │
│ ┌─────────────┐ │     │  & interactions  │     │ - Comparisons   │
│ │ Null Model  │ │     │                 │     │ - Visualization │
│ └─────────────┘ │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Framework Components

1. **Bayesian Optimization**
   - Efficient parameter search
   - Performance-based optimization
   - Reproducible parameter selection

2. **Model Layer**
   - Base Model (Claims & Grants)
   - Social Interactionist Model
   - Null Model (Control)

3. **Simulation Engine**
   - Time-step based evolution
   - Agent interactions
   - State management

4. **Analysis Layer**
   - ICC metrics calculation
   - Model comparisons
   - Result visualization

## 📊 Model Perspectives

The project implements multiple theoretical perspectives on leadership emergence:

1. **Base Model**: Pure schema matching
   - Leaders and followers emerge through schema matching
   - Simple claim and grant mechanics
   - No identity effects

2. **Social Interactionist**: Schema -> Identity transition
   - Starts with schema matching
   - Transitions to identity-based decisions
   - Models how repeated interactions shape identity

3. **Null Model**: Random interaction control
   - Baseline comparison
   - Random leadership claims/grants
   - No underlying mechanism

4. **Future Extensions**:
   - Cognitive: Learning and adaptation
   - Identity: Group prototypes and social identity
   - Context: Crisis and environmental effects

## 📁 Project Structure

```
src/
├── models/           # Model Implementations
│   ├── base_model.py   # Base schema matching
│   ├── null_model.py   # Random interaction control
│   └── perspectives/   # Different theoretical views
│       ├── interactionist.py  # Schema -> Identity
│       ├── cognitive.py       # Learning (future)
│       └── identity.py        # Group identity (future)
├── simulation/      # Simulation infrastructure
├── analysis/       # Analysis tools
└── utils/          # Shared utilities

scripts/            # Execution Scripts
├── optimize.py       # Bayesian parameter optimization
├── compare_perspectives.py  # Model comparison
└── visualize.py      # Result visualization

config/             # Configuration Files
├── parameters/       # Model parameters
└── optimization/    # Search space definitions

outputs/            # Results
├── plots/           # Generated visualizations
├── optimization/    # Optimization results
└── data/           # Simulation data
```

## 🔄 Analysis Pipeline

1. **Parameter Optimization**
   ```bash
   # Find optimal parameters using Bayesian search
   python scripts/optimize.py --model base
   python scripts/optimize.py --model interactionist
   ```

2. **Model Comparison**
   ```bash
   # Compare models with optimized parameters
   python scripts/compare_perspectives.py
   ```

3. **Visualization**
   ```bash
   # Generate detailed visualizations
   python scripts/visualize.py --models base interactionist null
   ```

## 📚 Documentation

- [Model Documentation](docs/MODELS.md) - Theoretical perspectives
- [Development Status](docs/ROADMAP.md) - Current progress and plans
- [Optimization Guide](docs/OPTIMIZATION.md) - Parameter search details

## 🤝 Contributing

Contributions welcome! See [Contributing Guide](CONTRIBUTING.md) for guidelines.

## 📚 Citation

If you use this code or framework in your research, please cite:

```bibtex
@misc{acton2024leadership,
  author = {Acton, Bryan},
  title = {Leadership Emergence Simulation: An Agent-Based Modeling Framework},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/bacton/abm-lead-emergence}
}
```

Or in text:
> Acton, B. (2024). Leadership Emergence Simulation: An Agent-Based Modeling Framework. GitHub. https://github.com/bacton/abm-lead-emergence

## 📝 License

MIT License - see [LICENSE](LICENSE)
