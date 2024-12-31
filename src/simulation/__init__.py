# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Dimitrios Kafetzis
#
# This file is part of the Transformer Inference Simulator project.
# Licensed under the MIT License; you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#   https://opensource.org/licenses/MIT
#
# Author:  Dimitrios Kafetzis (dimitrioskafetzis@gmail.com)
# File:    src/simulation/__init__.py
# Description:
#   Initializes the simulation package that coordinates event-driven
#   operations for distributed Transformer inference, including the
#   simulation engine, scheduler, and metrics collection.
#
# ---------------------------------------------------------------------------

"""
Initializes the simulation subpackage, providing the core simulation engine,
event scheduler, and metrics collectors for analyzing performance, resource
utilization, and communication overhead in distributed Transformer inference.
"""

from .engine import (
    SimulationEngine,
    SimulationState,
    SimulationConfig
)
from .scheduler import (
    EventScheduler,
    ExecutionPlan,
    ComponentState
)
from .metrics import (
    MetricsCollector,
    PerformanceMetrics,
    ResourceMetrics,
    CommunicationMetrics
)

# Version information
__version__ = '0.1.0'

# Define public interface
__all__ = [
    # Simulation engine components
    'SimulationEngine',
    'SimulationState',
    'SimulationConfig',
    
    # Scheduler components
    'EventScheduler',
    'ExecutionPlan',
    'ComponentState',
    
    # Metrics components
    'MetricsCollector',
    'PerformanceMetrics',
    'ResourceMetrics',
    'CommunicationMetrics'
]

# Module level documentation
SimulationEngine.__doc__ = """
Main simulation engine that coordinates the execution of the distributed inference simulation.
Handles event processing, state management, and metrics collection.
"""

EventScheduler.__doc__ = """
Manages the scheduling and execution of simulation events, including computation and communication.
Ensures correct ordering and timing of operations.
"""

MetricsCollector.__doc__ = """
Collects and analyzes performance metrics throughout the simulation.
Tracks resource utilization, latency, and communication overhead.
"""