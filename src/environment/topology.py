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
# File:    src/environment/topology.py
# Description:
#   Provides classes for generating various network topologies (edge cluster,
#   distributed edge, hybrid cloud-edge), as well as general topology config
#   structures for the simulator.
#
# ---------------------------------------------------------------------------

"""
Defines the NetworkTopologyGenerator base class along with specific
implementations like EdgeClusterTopology, DistributedEdgeTopology,
and HybridCloudEdgeTopology. Each topology generator creates a network
graph with nodes, edges, and bandwidth assignments used by the simulator.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
from ..core import Network, Device

@dataclass
class TopologyConfig:
    """Configuration for network topology generation"""
    num_devices: int
    min_bandwidth: float  # Gbps
    max_bandwidth: float  # Gbps
    edge_probability: float = 0.3  # For random graph generation
    seed: Optional[int] = None

class NetworkTopologyGenerator:
    """Base class for topology generation"""
    
    def __init__(self, config: TopologyConfig):
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
            
    def generate(self) -> Network:
        """Generate network topology"""
        raise NotImplementedError
        
    def _assign_bandwidths(self, network: Network) -> None:
        """Assign random bandwidths to network links"""
        for (source, target) in network.topology.edges():
            bandwidth = np.random.uniform(
                self.config.min_bandwidth,
                self.config.max_bandwidth
            )
            network.add_link(source, target, bandwidth)

class EdgeClusterTopology(NetworkTopologyGenerator):
    """
    Generates edge cluster topology with 8 devices in a hierarchical setup
    """
    
    def generate(self) -> Network:
        network = Network()
        
        # Create nodes
        for i in range(self.config.num_devices):
            network.add_device(f"device_{i}")
            
        # Create hierarchical structure
        # One high-capacity node (0) connected to several mid-tier nodes (1-3)
        # which are in turn connected to edge nodes (4-7)
        core_connections = [(0, i) for i in range(1, 4)]
        edge_connections = [
            (1, 4), (1, 5),
            (2, 5), (2, 6),
            (3, 6), (3, 7)
        ]
        
        # Add edges with high bandwidth for core connections
        for source, target in core_connections:
            bandwidth = np.random.uniform(5.0, 10.0)  # 5-10 Gbps
            network.add_link(f"device_{source}", f"device_{target}", bandwidth)
            
        # Add edges with medium bandwidth for edge connections
        for source, target in edge_connections:
            bandwidth = np.random.uniform(1.0, 5.0)  # 1-5 Gbps
            network.add_link(f"device_{source}", f"device_{target}", bandwidth)
            
        return network

class DistributedEdgeTopology(NetworkTopologyGenerator):
    """
    Generates distributed edge topology with 16 devices in a mesh network
    """
    
    def generate(self) -> Network:
        network = Network()
        
        # Create nodes
        for i in range(self.config.num_devices):
            network.add_device(f"device_{i}")
            
        # Create mesh topology with geographic clustering
        # Divide devices into 4 geographic clusters
        clusters = [
            list(range(0, 4)),
            list(range(4, 8)),
            list(range(8, 12)),
            list(range(12, 16))
        ]
        
        # Connect devices within clusters (higher bandwidth)
        for cluster in clusters:
            for i in cluster:
                for j in cluster:
                    if i < j:
                        bandwidth = np.random.uniform(0.5, 1.0)  # 500 Mbps - 1 Gbps
                        network.add_link(f"device_{i}", f"device_{j}", bandwidth)
                        
        # Connect clusters (lower bandwidth); existing backbone or cross-links
        inter_cluster_links = [
            (0, 4), (4, 8), (8, 12),  # Main backbone
            (0, 8), (4, 12),          # Cross connections
            (1, 5), (5, 9), (9, 13),  # Secondary paths
            (2, 6), (6, 10), (10, 14) # Redundant paths
        ]
        
        for source, target in inter_cluster_links:
            bandwidth = np.random.uniform(0.1, 0.5)  # 100 Mbps - 500 Mbps
            network.add_link(f"device_{source}", f"device_{target}", bandwidth)
        
        # --- NEW: Ensure at least one direct cross-edge between each pair of clusters ---
        # so that each cluster has a guaranteed link to every other cluster.
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dev_i = clusters[i][0]  # pick first device of cluster i
                dev_j = clusters[j][0]  # pick first device of cluster j
                # Check if there's already an edge between device_i and device_j
                if not network.topology.has_edge(f"device_{dev_i}", f"device_{dev_j}") \
                   and not network.topology.has_edge(f"device_{dev_j}", f"device_{dev_i}"):
                    # Add at least one link
                    bandwidth = np.random.uniform(0.1, 0.5)  # 100 Mbps - 500 Mbps
                    network.add_link(f"device_{dev_i}", f"device_{dev_j}", bandwidth)
        
        return network

class HybridCloudEdgeTopology(NetworkTopologyGenerator):
    """
    Generates hybrid cloud-edge topology with 24 devices
    """
    
    def generate(self) -> Network:
        network = Network()
        
        # Create nodes
        for i in range(self.config.num_devices):
            network.add_device(f"device_{i}")
            
        # Divide into cloud (0-3), regional (4-11), and edge (12-23) nodes
        cloud_nodes = list(range(4))
        regional_nodes = list(range(4, 12))
        edge_nodes = list(range(12, 24))
        
        # Connect cloud nodes (high bandwidth)
        for i in cloud_nodes:
            for j in cloud_nodes:
                if i < j:
                    bandwidth = np.random.uniform(20.0, 40.0)  # 20-40 Gbps
                    network.add_link(f"device_{i}", f"device_{j}", bandwidth)
                    
        # Connect regional nodes to nearest cloud node
        cloud_regional_mapping = {
            4: [0], 5: [0], 6: [1], 7: [1],
            8: [2], 9: [2], 10: [3], 11: [3]
        }
        
        for regional, clouds in cloud_regional_mapping.items():
            for cloud in clouds:
                bandwidth = np.random.uniform(5.0, 10.0)  # 5-10 Gbps
                network.add_link(f"device_{cloud}", f"device_{regional}", bandwidth)
                
        # Connect edge nodes to regional nodes
        regional_edge_mapping = {
            4: [12, 13, 14], 5: [15, 16, 17],
            6: [18, 19], 7: [20, 21],
            8: [22, 23], 9: [], 10: [], 11: []
        }
        
        for regional, edges in regional_edge_mapping.items():
            for edge in edges:
                bandwidth = np.random.uniform(0.1, 1.0)  # 100Mbps-1Gbps
                network.add_link(f"device_{regional}", f"device_{edge}", bandwidth)
                
        return network

def create_topology(
    topology_type: str,
    config: TopologyConfig
) -> Network:
    """Factory function to create topology based on type"""
    generators = {
        'edge_cluster': EdgeClusterTopology,
        'distributed_edge': DistributedEdgeTopology,
        'hybrid_cloud_edge': HybridCloudEdgeTopology
    }
    
    if topology_type not in generators:
        raise ValueError(
            f"Unknown topology type: {topology_type}. "
            f"Available types: {list(generators.keys())}"
        )
        
    generator = generators[topology_type](config)
    return generator.generate()

def validate_topology(network: Network) -> bool:
    # If no nodes or edges, it's invalid
    if network.topology.number_of_nodes() == 0:
        return False
    if network.topology.number_of_edges() == 0:
        return False

    # Then check connectivity
    if not nx.is_connected(network.topology):
        return False
    
    # Also check that each edge is in network.links
    for (source, target) in network.topology.edges():
        # or check both directions in an undirected network
        if (source, target) not in network.links and (target, source) not in network.links:
            return False
    
    return True