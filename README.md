# Leadership Emergence Simulation

A simple agent-based model exploring how leadership naturally emerges in groups through a "claim and grant" process.

## 🚀 Quick Start

```bash
# Clone and enter the repository
git clone https://github.com/bacton/abm-lead-emergence.git
cd abm-lead-emergence

# Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run a basic simulation
python scripts/run_simulation.py
```

## 📊 Latest Results

The base model (v1.0) shows how leadership emerges through simple interactions:
- Leaders and followers naturally differentiate over time
- Stable (but weak) hierarchies form
- Optimal parameters identified for strongest emergence

[See full results here](results/base_model_results.md)

## 📁 Project Structure

```
src/           # Core model code
├── models/      # Different model implementations
├── simulation/  # Running simulations
└── analysis/    # Analyzing results

scripts/       # Ready-to-use analysis tools
├── run_simulation.py     # Run a basic simulation
└── analyze_results.py    # Look at the results

config/        # Model settings (YAML files)
results/       # Analysis and findings
outputs/       # Generated visualizations
```

## 📚 Documentation

- [Quick Start Guide](QUICKSTART.md) - Get started in 5 minutes
- [Model Documentation](MODELS.md) - How the model works
- [Development Status](ROADMAP.md) - Current progress and plans

## 🤝 Contributing

Contributions welcome! See [Contributing Guide](CONTRIBUTING.md) for guidelines.

## 📝 License

MIT License - see [LICENSE](LICENSE)
