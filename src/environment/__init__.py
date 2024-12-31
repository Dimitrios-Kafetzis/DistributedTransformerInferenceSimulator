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
# File:    src/environment/__init__.py
# Description:
#   Initializes the environment module, providing functionality for
#   resource distributions, workload generation, and network topology
#   configurations for distributed inference scenarios.
#
# ---------------------------------------------------------------------------

"""
Initializes the environment subpackage. Includes classes and utilities
to generate or manage device capabilities, network topologies, and
inference workloads for the simulator.
"""

from .topology import (
    NetworkTopologyGenerator,
    EdgeClusterTopology,
    DistributedEdgeTopology,
    HybridCloudEdgeTopology,
    TopologyConfig
)

from .resources import (
    ResourceDistributor,
    LogNormalDistribution,
    DeviceCapabilities,
    BandwidthManager
)

from .workload import (
    WorkloadGenerator,
    TransformerWorkload,
    SequenceConfig,
    WorkloadType
)

# Version information
__version__ = '0.1.0'

# Define public interface
__all__ = [
    # Topology generation
    'NetworkTopologyGenerator',
    'EdgeClusterTopology',
    'DistributedEdgeTopology',
    'HybridCloudEdgeTopology',
    'TopologyConfig',
    
    # Resource distribution
    'ResourceDistributor',
    'LogNormalDistribution',
    'DeviceCapabilities',
    'BandwidthManager',
    
    # Workload generation
    'WorkloadGenerator',
    'TransformerWorkload',
    'SequenceConfig',
    'WorkloadType'
]

# Module level documentation
NetworkTopologyGenerator.__doc__ = """
Creates different network topology configurations for simulation scenarios.
Supports edge cluster, distributed edge, and hybrid cloud-edge setups.
"""

ResourceDistributor.__doc__ = """
Manages the distribution of compute and memory resources across devices.
Uses log-normal distributions to model heterogeneous capabilities.
"""

WorkloadGenerator.__doc__ = """
Generates transformer inference workloads with varying sequence lengths
and generation steps for evaluation scenarios.
"""