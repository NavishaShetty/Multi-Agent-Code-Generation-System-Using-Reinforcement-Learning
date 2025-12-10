# RL Code Generation Workflow Agent

A Reinforcement Learning-based orchestrator for a multi-agent code generation system. The RL agent learns to optimally coordinate four specialized LLM agents (Planner, Coder, Tester, Debugger) to solve coding tasks efficiently.

## Core Concept

- **LLM agents** (via OpenRouter) handle the actual coding tasks
- **RL orchestrator** (Q-Learning + Thompson Sampling) learns WHICH agent to invoke and WHEN
- The LLM weights are NEVER updated - only the RL policy weights are trained

## Features

- **4 Specialized Agents**: Planner, Coder, Tester, Debugger
- **2 RL Methods**: Q-Learning (value-based) + Thompson Sampling (exploration)
- **Blackboard Communication**: Shared message passing between agents
- **Custom Tool**: Code Complexity Analyzer with multiple metrics
- **Fast Simulation**: ~100k episodes/second for training
- **Visualization**: Learning curves and Q-table heatmaps

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd code-gen-rl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenRouter API key:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## Quick Start

### 1. Sanity Check (verify API works)
```bash
python scripts/sanity_check.py
```

### 2. Train the RL Agent (on simulation)
```bash
python training/train_simulated.py --episodes 5000
```

### 3. Generate Visualizations
```bash
python visualization/learning_curves.py
python visualization/q_table_viz.py
```

### 4. Collect Baseline (fixed pipeline)
```bash
python scripts/collect_baseline.py --tasks 5
```

### 5. Validate with Real LLM
```bash
python training/validate_real.py --tasks 3
```

### 6. Compare Results
```bash
python training/evaluate.py
```

## Testing the Agents

To verify the agents are working correctly:

**Test the full agent pipeline (Planner → Coder → Tester → Debugger):**
```bash
python orchestrator/fixed_pipeline.py
```

**Run the interactive demo:**
```bash
python demo/demo.py
```

**Collect baseline metrics across multiple tasks:**
```bash
python scripts/collect_baseline.py
```

## Project Structure

```
code-gen-rl/
├── agents/                 # LLM-powered agents
│   ├── base_agent.py       # Abstract base class
│   ├── planner_agent.py    # Task planning
│   ├── coder_agent.py      # Code generation
│   ├── tester_agent.py     # Code analysis
│   └── debugger_agent.py   # Bug fixing
│
├── communication/          # Agent communication
│   └── blackboard.py       # Shared message board
│
├── tools/                  # Tools for agents
│   ├── code_executor.py    # Safe code execution
│   ├── test_runner.py      # Test generation and running
│   └── complexity_analyzer.py  # CUSTOM TOOL
│
├── rl/                     # Reinforcement Learning
│   ├── q_learning.py       # Q-Learning implementation
│   ├── thompson_sampling.py # Thompson Sampling
│   └── combined_agent.py   # Q-Learning + Thompson Sampling
│
├── environment/            # RL Environment
│   ├── state.py            # State representation
│   ├── rewards.py          # Reward function
│   ├── simulated_env.py    # Fast simulation
│   └── coding_env.py       # Real LLM environment
│
├── training/               # Training scripts
│   ├── train_simulated.py  # Train on simulation
│   ├── validate_real.py    # Test with real LLM
│   └── evaluate.py         # Compare policies
│
├── visualization/          # Plotting
│   ├── learning_curves.py  # Training progress
│   └── q_table_viz.py      # Q-value visualization
│
├── experiments/results/    # Saved results
│   ├── q_table.json        # Trained Q-table
│   ├── learning_curves.png # Training plots
│   └── comparison.json     # Policy comparison
│
└── scripts/                # Utility scripts
    ├── sanity_check.py     # API verification
    └── collect_baseline.py # Baseline metrics
```

## RL Formulation

### State Space (64 states)
- `has_plan`: bool (2 values)
- `has_code`: bool (2 values)
- `has_error`: bool (2 values)
- `tests_pass`: bool (2 values)
- `iteration_bucket`: 0-3 (4 values)

Total: 2 × 2 × 2 × 2 × 4 = 64 states

### Action Space (4 actions)
- `planner`: Generate task plan
- `coder`: Write code
- `tester`: Analyze/test code
- `debugger`: Fix errors

### Reward Function
| Event | Reward |
|-------|--------|
| Task success | +10.0 |
| Task timeout | -5.0 |
| Progress (plan) | +0.2 |
| Progress (code) | +0.3 |
| Error fixed | +0.5 |
| Redundant action | -0.5 |
| Invalid action | -0.3 |
| Step cost | -0.1 |

### Q-Learning Update
```
Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
```

### Thompson Sampling
Action selection via Beta distribution sampling:
```
θ(s,a) ~ Beta(α_sa, β_sa)
a* = argmax_a θ_sample(s,a)
```

## Configuration

Edit `config.yaml` to customize:
- API settings (model, temperature)
- RL hyperparameters (α, γ, episodes)
- Simulation probabilities

## Results

After training, the RL agent learns an effective policy:
- **Success rate**: ~85-90% on simulated tasks
- **Training time**: <1 second for 5000 episodes
- **Key insight**: Agent learns to plan first, then code, test, and debug as needed

## Custom Tool: Complexity Analyzer

Located in `tools/complexity_analyzer.py`, this tool provides:
- Cyclomatic complexity
- Lines of code
- Function count
- Max nesting depth
- Cognitive complexity

```python
from tools.complexity_analyzer import ComplexityAnalyzer

analyzer = ComplexityAnalyzer()
metrics = analyzer.analyze(code)
print(f"Complexity score: {metrics.overall_score()}")
```

## License

MIT License
