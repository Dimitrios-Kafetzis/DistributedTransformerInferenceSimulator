# Transformer Inference Simulator

A Python-based simulator for distributed transformer inference across edge, cluster and hybrid cloud environments.

## Overview

This project provides a simulation framework for studying and optimizing transformer model inference across distributed computing environments. It supports:

- Different deployment scenarios (edge clusters, distributed edge, hybrid cloud-edge)
- Resource-aware component distribution
- Network topology and bandwidth modeling  
- Performance metrics collection and analysis

## Features

- **Multiple Deployment Scenarios**
  - Edge cluster (8 devices)
  - Distributed edge (16 devices)  
  - Hybrid cloud-edge (24 devices)

- **Resource Management**
  - Memory tracking and allocation
  - Compute capacity modeling
  - Cache placement optimization

- **Network Simulation**
  - Configurable network topologies
  - Bandwidth and latency modeling
  - Communication overhead analysis

- **Performance Analysis**
  - Latency measurement
  - Resource utilization tracking
  - Network metrics collection
  - Visualization tools

## Complete Setup and Usage Guide

### 1. Setup and Installation

#### Clone the Repository
```bash
git clone https://github.com/username/transformer_inference_simulator.git
cd transformer_inference_simulator
```

#### Create and Activate Virtual Environment
```bash
# Create virtual environment
python3 -m venv transformer_inference_simulator_env

# Activate virtual environment
# On Linux/macOS:
source transformer_inference_simulator_env/bin/activate
# On Windows:
transformer_inference_simulator_env\Scripts\activate
```

#### Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 2. Running Tests

#### Run All Tests with Coverage
```bash
# Full test suite with coverage report
pytest tests/ --cov=src --cov-report=html
```

#### Run Specific Test Categories
```bash
# Core components tests
pytest tests/test_core/

# Algorithm tests
pytest tests/test_algorithms/

# Environment tests
pytest tests/test_environment/

# Simulation tests
pytest tests/test_simulation/

# Utilities tests
pytest tests/test_utils/
```

### 3. Running Experiments

#### Baseline Comparisons
```bash
python run_experiments.py --config experiments/configs/baseline_comparison.yaml --output-dir results/baseline
```

#### Edge Cluster Experiments
```bash
python run_experiments.py \
    --config experiments/configs/edge_cluster.yaml \
    --output-dir results/edge_cluster
```

#### Distributed Edge Experiments
```bash
python run_experiments.py \
    --config experiments/configs/distributed_edge.yaml \
    --output-dir results/distributed_edge
```

#### Hybrid Cloud-Edge Experiments
```bash
python run_experiments.py \
    --config experiments/configs/hybrid_cloud_edge.yaml \
    --output-dir results/hybrid_cloud
```

### 4. Analyzing Results

#### Analyze All Results
```bash
python -c "from analysis import analyze_results; analyze_results('results', 'analysis/plots')"
```

#### Analyze Specific Scenarios
```bash
# Edge cluster analysis
python -c "from analysis import analyze_results; analyze_results('results/edge_cluster', 'analysis/plots/edge_cluster')"

# Distributed edge analysis
python -c "from analysis import analyze_results; analyze_results('results/distributed_edge', 'analysis/plots/distributed_edge')"

# Hybrid cloud analysis
python -c "from analysis import analyze_results; analyze_results('results/hybrid_cloud', 'analysis/plots/hybrid_cloud')"
```

## Project Structure

```
transformer_inference_simulator/
├── src/                     # Source code
│   ├── algorithms/         # Distribution algorithms
│   ├── core/              # Core components
│   ├── environment/       # Simulation environment
│   ├── simulation/        # Simulation engine
│   └── utils/             # Utilities
├── experiments/            # Experiment configurations
│   ├── configs/           # YAML configuration files
│   ├── scenarios/         # Scenario implementations
│   └── results/           # Experiment results
├── analysis/              # Analysis tools
│   ├── metrics/          # Metrics calculation
│   ├── plotting/         # Visualization
│   └── utils/            # Analysis utilities
└── tests/                 # Test suite
```

## Configuration

Experiment configurations are defined in YAML files under `experiments/configs/`. Example configuration:

```yaml
network:
  topology_type: "edge_cluster"
  num_devices: 8
  min_bandwidth: 1.0    # 1 Gbps
  max_bandwidth: 10.0   # 10 Gbps

resources:
  memory_mu: 2.0
  memory_sigma: 0.5
  memory_min: 2.0    # 2GB RAM
  memory_max: 16.0   # 16GB RAM
  compute_mu: 5.0
  compute_sigma: 0.5
  compute_min: 10.0  # 10 GFLOPS
  compute_max: 100.0 # 100 GFLOPS

workload:
  model_type: "SMALL"  # 8 attention heads, D=512
  initial_sequence_lengths: [128, 256]
  generation_steps: [32, 64]
```

## File Formats

### Metrics File (.jsonl)
The metrics are stored in JSONL (JSON Lines) format, where each line is a valid JSON object:
```jsonl
{"timestamp": "2024-03-01T10:00:00", "metrics": {"latency": 100, "memory_usage": 0.8}}
{"timestamp": "2024-03-01T10:00:01", "metrics": {"latency": 105, "memory_usage": 0.85}}
```

To read metrics files:
```python
import pandas as pd
metrics_df = pd.read_json('metrics.jsonl', lines=True)
```

## Troubleshooting

### Common Issues and Solutions

1. Virtual Environment Issues
```bash
# If venv creation fails, ensure you have python3-venv installed
sudo apt-get install python3-venv  # On Ubuntu/Debian
```

2. Missing Dependencies
```bash
# If import errors occur, verify installation
pip list
pip install -r requirements.txt --force-reinstall
```

3. Permission Issues
```bash
# If permission denied when creating directories
chmod +x run_experiments.py
sudo chown -R $USER:$USER results/
```

### Viewing Results

Check the log files for detailed information:
```bash
# View latest logs
tail -f results/*/logs/*.log
```

View the test coverage report:
```bash
# On macOS:
open htmlcov/index.html
# On Linux:
xdg-open htmlcov/index.html
# On Windows:
start htmlcov/index.html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.