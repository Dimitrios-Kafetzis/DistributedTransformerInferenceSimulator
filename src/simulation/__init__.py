"""
Simulation components for transformer inference.
This module provides the simulation engine, event scheduling,
and metrics collection for analyzing distributed transformer inference.
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