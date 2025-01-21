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
# File:    src/algorithms/__init__.py
# Description:
#   Initializes the algorithms module, providing resource-aware and
#   baseline distribution strategies for distributed transformer
#   inference simulation.
#
# ---------------------------------------------------------------------------

"""
Initializes the algorithms subpackage, exposing classes and functions for
various distribution strategies (resource-aware, greedy, round-robin, etc.)
for distributed transformer inference.
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
    DynamicMigrationDistributor,
    ExactOptimalDistributor
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
    'ExactOptimalDistributor',
    
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