"""
Environment generation and configuration for transformer inference simulation.
This module provides tools for creating network topologies, managing resource
distributions, and generating workloads.
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