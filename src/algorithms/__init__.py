"""
Algorithms for distributed transformer inference.
This module provides implementations of resource-aware distribution
and baseline algorithms for comparison.
"""

from .resource_aware import (
    ResourceAwareDistributor,
    ScoringFunction,
    AssignmentResult
)

from .baselines import (
    GreedyDistributor,
    RoundRobinDistributor,
    StaticDistributor,
    DynamicMigrationDistributor
)

from .utils import (
    ResourceRequirements,
    CommunicationCost,
    validate_assignment
)

# Version information
__version__ = '0.1.0'

# Define public interface
__all__ = [
    # Main algorithm
    'ResourceAwareDistributor',
    'ScoringFunction',
    'AssignmentResult',
    
    # Baseline algorithms
    'GreedyDistributor',
    'RoundRobinDistributor',
    'StaticDistributor',
    'DynamicMigrationDistributor',
    
    # Utilities
    'ResourceRequirements',
    'CommunicationCost',
    'validate_assignment'
]

# Module level documentation
ResourceAwareDistributor.__doc__ = """
Main implementation of the resource-aware distribution algorithm.
Optimizes component placement and cache management for distributed transformer inference.
"""

GreedyDistributor.__doc__ = """
Baseline implementation using greedy placement strategy.
Assigns components to first available device with sufficient resources.
"""

RoundRobinDistributor.__doc__ = """
Baseline implementation using round-robin distribution.
Distributes components evenly across available devices.
"""

StaticDistributor.__doc__ = """
Baseline implementation using static partitioning.
Fixes component assignments based on initial conditions.
"""

DynamicMigrationDistributor.__doc__ = """
Baseline implementation with dynamic migration.
Reactively reassigns components based on resource thresholds.
"""