# Distributed Transformer Inference Simulator

A Python-based simulation framework for resource-aware, distributed Transformer (https://arxiv.org/html/1706.03762v7) inference. This simulator is designed to model and optimize the inference of Transformer models across heterogeneous environments such as edge clusters, distributed edge networks, and hybrid cloud-edge deployments.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture and Project Structure](#architecture-and-project-structure)
  - [Algorithms](#algorithms)
  - [Core Components](#core-components)
  - [Simulation Engine](#simulation-engine)
  - [Utilities](#utilities)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Experiments](#running-experiments)
  - [Simulation Workflow](#simulation-workflow)
  - [Metrics and Analysis](#metrics-and-analysis)
- [Configuration](#configuration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

The **Distributed Transformer Inference Simulator** is a modular framework that emulates the distributed inference process for Transformer models. Built with a discrete-event simulation core, it models key aspects such as:

- **Device Resource Management:** Models compute devices with specified memory and compute capacities, and tracks resource allocation.
- **Transformer Model Simulation:** Implements components like attention heads, projection layers, and feed-forward networks; estimates memory requirements and computational costs.
- **Event-Driven Simulation:** Utilizes an event queue to schedule and process events (compute, transfer, generation steps, cache updates) in chronological order.
- **Execution Scheduling:** Includes a scheduler that creates execution plans, resolves component dependencies, and manages inter-device transfers.
- **Metrics Collection:** Gathers detailed performance, resource, and communication metrics for analysis.
- **Algorithmic Experimentation:** Supports multiple strategies for distributing inference workloads, including baseline, edge sharding, galaxy-inspired, and resource-aware algorithms.

---

## Key Features

- **Multiple Distribution Algorithms:**
  - *Baselines:* Simple strategies serving as benchmarks.
  - State of the Art algorithmic frameworks:
    - *EdgeShard:* Techniques optimized for splitting workloads across edge devices. (https://arxiv.org/pdf/2405.14371)
    - *Galaxy:* Hierarchical approaches for large-scale distributed inference. (https://arxiv.org/pdf/2405.17245)
  - Our proposed algorithmic framework:
    - *Resource-Aware:* Dynamic allocation based on real-time resource availability.
  
- **Device and Resource Modeling:**
  - **Device Class:** Represents a compute node with methods for resource allocation, deallocation, and tracking usage histories.
  - **ResourceState:** Provides current availability and utilization percentages for memory and compute resources.
  
- **Discrete-Event Simulation:**
  - **Event & EventQueue:** Defines events (e.g., compute start/complete, transfer start/complete, generation steps, cache updates) and ensures proper ordering.
  - **Simulation Engine:** Processes events, manages simulation time, and enforces termination conditions (max steps, time limits).
  
- **Transformer Model Simulation:**
  - **TransformerConfig:** Stores parameters such as number of attention heads, embedding dimension, initial sequence length, and precision.
  - **Transformer Components:**
    - *AttentionHead:* Computes memory requirements (for QKV matrices and cache), and FLOPs.
    - *ProjectionLayer:* Handles outputs from attention heads.
    - *FeedForwardNetwork (FFN):* Implements FFN computations with associated resource and FLOP estimates.
  - **Transformer Class:** Aggregates components, manages sequence progression, and calculates total resource and computation demands.
  
- **Execution Scheduling:**
  - **EventScheduler & ExecutionPlan:** Constructs execution plans per generation step, builds dependency graphs, schedules component computations, and manages cross-device transfers.
  
- **Metrics Collection and Analysis:**
  - **PerformanceMetrics:** Tracks simulation time, generation step latencies, and compute/communication durations.
  - **ResourceMetrics:** Monitors device resource utilizations, peaks, and idle times.
  - **CommunicationMetrics:** Logs data transfers, migration counts, and link bandwidth usage.
  - **MetricsCollector:** Aggregates metrics in real time, provides summary statistics, and supports exporting to JSON.
  
- **Utilities:**
  - **Configuration Management:** Simplifies experiment setup via YAML files.
  - **Logging:** Provides detailed logging for simulation events and resource updates.
  - **Visualization:** Contains hooks for plotting metrics (to be extended as needed).

---

## Architecture and Project Structure
```
transformer_inference_simulator/
├── analysis/ # Analysis and plotting of simulation results 
├── experiments/ # YAML configurations and experiment scripts 
├── src/ 
│   ├── algorithms/ # Distribution algorithms: 
│   │   ├── baselines.py # Baseline distribution strategies 
│   │   ├── edgeshard.py # Edge sharding for partitioning tasks 
│   │   ├── galaxy.py # Hierarchical (galaxy-inspired) strategies 
│   │   ├── resource_aware.py # Resource-aware scheduling algorithms 
│   │   └── utils.py # Helper functions for algorithms 
│   ├── core/ # Core simulation components: 
│   │   ├── device.py # Device class & resource tracking 
│   │   ├── event.py # Event definitions and ordering 
│   │   ├── network.py # Network topology, bandwidth & latency modeling 
│   │   └── transformer.py # Transformer model components and config 
│   ├── simulation/ # Simulation engine and scheduling: 
│   │   ├── engine.py # Main simulation engine (event loop) 
│   │   ├── metrics.py # Metrics collection (performance, resource, communication) 
│   │   └── scheduler.py # Execution scheduling and dependency resolution 
│   └── utils/ # Utility modules: 
│       ├── config.py # Experiment configuration utilities 
│       ├── logging.py # Logging helper for simulation events 
│       └── visualization.py # (Optional) routines for visualizing metrics 
├── tests/ # Unit and integration tests for all modules 
├── .gitignore 
├── LICENSE 
├── README.md # This file 
├── requirements.txt # Python dependencies 
├── run_experiments.py # Script to run simulation experiments 
└── setup.py # Package setup script
```

### Algorithms

The `src/algorithms` directory provides various strategies to distribute Transformer inference tasks:
- **baselines.py:** Implements basic distribution strategies.
- **edgeshard.py:** Implements sharding techniques optimized for edge devices.
- **galaxy.py:** Implements hierarchical distribution methods for large-scale inference.
- **resource_aware.py:** Allocates tasks dynamically based on resource availability.
- **utils.py:** Contains common helper functions.

### Core Components

- **Device (device.py):**  
  Encapsulates a compute device with memory and compute capacities. It offers methods to:
  - Check if resources can be accommodated.
  - Allocate and deallocate resources (including cache).
  - Maintain usage histories for analysis.

- **Event System (event.py):**  
  Defines an `Event` class and an `EventQueue` that together:
  - Represent discrete events (e.g., compute, transfer, generation steps, cache updates).
  - Order events by time and priority.
  
- **Transformer Model (transformer.py):**  
  Defines the Transformer configuration and its components:
  - **TransformerConfig:** Stores model parameters.
  - **AttentionHead, ProjectionLayer, FeedForwardNetwork:** Calculate memory requirements and FLOPs.
  - **Transformer Class:** Manages components and sequence progression.

### Simulation Engine

- **Engine (engine.py):**  
  Drives the simulation by:
  - Processing events from the `EventQueue`.
  - Scheduling generation steps and component executions.
  - Handling compute and transfer events.
  - Checking termination conditions (max steps and time limits).
  
- **Metrics (metrics.py):**  
  Collects detailed metrics including:
  - Generation step latencies.
  - Computation and transfer durations.
  - Device resource utilization and network bandwidth usage.
  
- **Scheduler (scheduler.py):**  
  Manages the execution plan for each generation step:
  - Builds dependency graphs among Transformer components.
  - Schedules computations and cross-device transfers.
  - Estimates component completion times.

### Utilities

- **Configuration (config.py):** Manages experiment parameters.
- **Logging (logging.py):** Provides structured logging for simulation events.
- **Visualization (visualization.py):** Contains routines to plot metrics (extension point).

---

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

### 2. Usage

#### Running Experiments
##### Example: Baseline Comparison:
```bash
python run_experiments.py --config experiments/configs/baseline_comparison.yaml --output-dir results/baseline
```
#### Simulation Workflow
1. Initialization:
The simulation engine initializes devices, loads the Transformer model, and schedules the first generation step.

2. Event Processing:
The engine processes events in sequence (compute starts/completes, transfers, cache updates), advancing the simulation time accordingly.

3. Execution Scheduling:
The scheduler builds an execution plan for each generation step, resolves component dependencies, and schedules cross-device transfers if necessary.

4. Metrics Collection:
Throughout the simulation, detailed performance, resource, and communication metrics are collected for post-run analysis.

### 3. Configuration

Experiment parameters are defined in YAML files under experiments/configs/. Typical settings include:
- Network Settings: Topology, device count, bandwidth, and latency.
- Resource Specifications: Memory and compute capacities (with statistical parameters).
- Workload Settings: Transformer model parameters (e.g., number of heads, embedding dimensions, sequence lengths).
Customize these files to simulate various deployment scenarios and hardware constraints.

## Author & Contact

For questions, suggestions, or issues, please open an issue on GitHub or contact the maintainer and author (Dimitrios Kafetzis) at dimitrioskafetzis@gmail.com or kafetzis@aueb.gr.


## License

This project is licensed under the MIT License - see the LICENSE file for details.
