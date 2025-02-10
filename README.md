# Leadership Emergence Simulation

> ⚠️ **Work in Progress Notice**: This repository explores how different theoretical mechanisms can explain leadership emergence in groups. The base model provides core interaction mechanics, while three distinct perspectives implement different emergence mechanisms. We're adding contextual factors to show how each mechanism may be more effective in different situations.

An agent-based model exploring how leadership naturally emerges in groups through a "claim and grant" process, comparing different theoretical mechanisms across contexts.

## 🚀 Quick Start

```bash
# Clone and enter the repository
git clone https://github.com/bacton/abm-lead-emergence.git
cd abm-lead-emergence

# Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run parameter optimization for each model
python scripts/run_parameter_sweep.py --model base
python scripts/run_parameter_sweep.py --model interactionist
python scripts/run_parameter_sweep.py --model cognitive
python scripts/run_parameter_sweep.py --model identity

# Compare models with optimized parameters
python scripts/compare_perspectives.py
```

## 📊 Model Structure

### Foundation Models
1. **Base Model**: Core interaction mechanics
   - Basic claim-grant process
   - Schema-based decisions
   - Does not produce emergence alone
   - Foundation for other perspectives

2. **Null Model**: Random baseline
   - Random claim/grant decisions
   - No underlying mechanism
   - Control condition

### Emergence Perspectives
Each perspective builds on the base model, adding a distinct mechanism for leadership emergence:

1. **Social Interactionist**: Identity Development
   - Two-stage process:
     1. Schema-based interactions
     2. Identity-based decisions
   - Shows how repeated interactions create stable identities
   - Emergence through role crystallization

2. **Cognitive**: Learning & Adaptation
   - ILT adaptation through observation
   - Recent interaction tracking
   - Social learning from successful leaders
   - Emergence through prototype convergence

3. **Identity**: Group Prototypes (In Progress)
   - Collective identity influence
   - Shared leadership prototypes
   - Group-level processes
   - Emergence through prototype alignment

### Model Comparison

| Model          | Core Mechanism              | Decision Basis                  | Learning Process              | Task Context |
|----------------|----------------------------|--------------------------------|------------------------------|--------------|
| Base           | Schema Matching            | Pure characteristic-ILT match  | None (static schemas)        | None         |
| Interactionist | Identity Development       | Contextualized role identities | Identity crystallization     | None         |
| Cognitive      | Schema Adaptation          | Adapted ILT matching          | Learning from observations   | None         |
| Identity       | Collective Influence       | Group prototype alignment     | Prototype evolution         | None         |

### Task Framework

The models are designed to work with different task contexts through a modular task framework:

1. **Base Task Interface**
   - Shared/unique information handling
   - Solution evaluation metrics
   - Context-specific modifiers
   - Performance tracking

2. **Context Types**
   | Context  | Time Pressure | Complexity | Uncertainty | Key Features |
   |----------|---------------|------------|-------------|--------------|
   | None     | Low          | Low        | Low         | Basic interaction |
   | Crisis   | High         | High       | High        | Quick decisions |
   | Routine  | Low          | Moderate   | Low         | Stable patterns |
   | Creative | Moderate     | High       | High        | Novel solutions |

3. **Hidden Profile Task**
   - Distributed information
   - Shared vs. unique knowledge
   - Integration requirements
   - Solution quality metrics

4. **Context Effects**
   - Modify decision thresholds
   - Adjust learning rates
   - Influence prototype formation
   - Shape interaction patterns

### Contextual Differentiation (Planned)
Adding task/situation contexts to show when each mechanism is most effective:
- Crisis vs. Routine Tasks
- Creative vs. Structured Work
- Short-term vs. Long-term Groups

### Control Model
- Random claim/grant decisions
- No underlying mechanism
- Baseline for comparing emergence patterns

## 🔍 Key Findings

Our models demonstrate clear evidence of leadership emergence beyond the base model:

### Emergence Patterns
1. **Strong Leadership Structure**
   - Cognitive & Identity models: Strong emergence (ICC > 0.8)
   - Interactionist model: Moderate emergence (ICC ≈ 0.35)
   - Base & Null models: No emergence (ICC ≈ 0.2)

2. **Hierarchy Development**
   - Clear hierarchical structures emerge in perspective models
   - Significant entropy reduction after initial interactions
   - Stable but not extreme differentiation
   - Base model remains unstructured

3. **Collective Dynamics**
   - Group-level patterns match empirical observations
   - Natural transition from individual to collective processes
   - Stable leadership structures emerge by step 20-30
   - Consistent with real-world team development

4. **Model Validation**
   - Results align with theoretical predictions
   - Each perspective shows distinct emergence patterns
   - Clear differentiation from base/null models
   - Replicates key empirical findings

## 🔬 Framework Components

```
                     MODEL FRAMEWORK
                     ===============

┌─────────────────────────────────────────────────────┐
│                Parameter Optimization                │
│  - Optimal parameters for each perspective          │
│  - Context-specific optimization                    │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Base Models    │  │   Emergence     │  │    Analysis     │
│                 │  │  Perspectives   │  │                 │
│ - Core Process  │  │ - Interactionist│  │ - Comparisons  │
│ - Null Model    │◄─┤ - Cognitive     ├─►│ - Contexts     │
│                 │  │ - Identity      │  │ - Effectiveness │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## 📁 Project Structure

```
src/
├── models/           # Model Implementations
│   ├── base_model.py   # Core mechanics (no emergence)
│   ├── null_model.py   # Random baseline
│   └── perspectives/   # Emergence mechanisms
│       ├── interactionist.py  # Identity development
│       ├── cognitive.py       # Learning/adaptation
│       └── identity.py        # Group prototypes
├── simulation/      # Simulation infrastructure
├── analysis/       # Analysis tools
└── utils/          # Shared utilities

scripts/            # Execution Scripts
├── run_parameter_sweep.py    # Parameter optimization
├── compare_perspectives.py   # Mechanism comparison
└── visualize/               # Analysis visualization
    ├── base_model.py         # Base dynamics
    ├── interactionist.py     # Identity emergence
    └── cognitive.py          # Learning patterns

config/             # Configuration Files
├── parameters.yaml           # Default parameters
└── parameter_sweep.yaml      # Optimization settings
```

## 🔄 Analysis Pipeline

1. **Parameter Optimization**
   ```bash
   # Find optimal parameters for each mechanism
   python scripts/run_parameter_sweep.py --model base
   python scripts/run_parameter_sweep.py --model interactionist
   python scripts/run_parameter_sweep.py --model cognitive
   ```

2. **Model Comparison**
   ```bash
   # Compare emergence mechanisms
   python scripts/compare_perspectives.py
   ```

3. **Individual Analysis**
   ```bash
   # Analyze specific mechanisms
   python scripts/visualize/interactionist.py
   python scripts/visualize/cognitive.py
   ```

## 📚 Documentation

- [Parameter Reference](docs/parameter_reference.md) - Parameter documentation
- [Project Overview](docs/PROJECT_OVERVIEW.md) - Detailed description
- [Model Documentation](docs/MODELS.md) - Theoretical background

## 🤝 Contributing

See [Contributing Guide](CONTRIBUTING.md) for guidelines.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{acton2024leadership,
  author = {Acton, Bryan},
  title = {Leadership Emergence Simulation: An Agent-Based Modeling Framework},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/bacton/abm-lead-emergence}
}
```

## 📝 License

MIT License - see [LICENSE](LICENSE)
