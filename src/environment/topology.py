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
from typing import Optional
import networkx as nx
import numpy as np


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
            
    def generate(self):
        """
        Generate network topology. Must return a Network instance.
        We'll do a local import inside the method to avoid top-level import of core.Network.
        """
        raise NotImplementedError
        
    def _assign_bandwidths(self, network_obj) -> None:
        """
        Assign random bandwidths to network links.
        Note: 'network_obj' is expected to be an instance of core.Network,
              but we can't reference that class at top-level; we do local import.
        """
        for (source, target) in network_obj.topology.edges():
            bandwidth = np.random.uniform(
                self.config.min_bandwidth,
                self.config.max_bandwidth
            )
            network_obj.add_link(source, target, bandwidth)

class EdgeClusterTopology(NetworkTopologyGenerator):
    """
    Generates edge cluster topology with some hierarchical structure.
    """
    
    def generate(self):
        """
        Actually create a Network object, add devices, and set up edges.
        """
        # Local import from ..core
        from ..core import Network
        
        network_obj = Network()
        
        # Create nodes
        for i in range(self.config.num_devices):
            network_obj.add_device(f"device_{i}")
            
        # Example hierarchical connections
        core_connections = [(0, i) for i in range(1, 4)]
        edge_connections = [
            (1, 4), (1, 5),
            (2, 5), (2, 6),
            (3, 6), (3, 7)
        ]
        
        # Add edges for "core" connections
        for source, target in core_connections:
            bandwidth = np.random.uniform(5.0, 10.0)  # example 5-10 Gbps
            network_obj.add_link(f"device_{source}", f"device_{target}", bandwidth)
        
        # Add edges for "edge" connections
        for source, target in edge_connections:
            bandwidth = np.random.uniform(1.0, 5.0)  # example 1-5 Gbps
            network_obj.add_link(f"device_{source}", f"device_{target}", bandwidth)
        
        return network_obj

class DistributedEdgeTopology(NetworkTopologyGenerator):
    """
    Generates distributed edge topology (mesh-like) with multiple clusters.
    """
    
    def generate(self):
        from ..core import Network
        
        network_obj = Network()
        
        # Create nodes
        for i in range(self.config.num_devices):
            network_obj.add_device(f"device_{i}")
            
        # Example cluster-based mesh
        clusters = [
            list(range(0, 4)),
            list(range(4, 8)),
            list(range(8, 12)),
            list(range(12, 16))
        ]
        
        # Connect devices within clusters
        for cluster in clusters:
            for i in cluster:
                for j in cluster:
                    if i < j:
                        bw = np.random.uniform(0.5, 1.0)
                        network_obj.add_link(f"device_{i}", f"device_{j}", bw)
        
        # Connect clusters with some backbone/cross-links
        inter_cluster_links = [
            (0, 4), (4, 8), (8, 12),
            (0, 8), (4, 12),
            (1, 5), (5, 9), (9, 13),
            (2, 6), (6, 10), (10, 14)
        ]
        for source, target in inter_cluster_links:
            bw = np.random.uniform(0.1, 0.5)
            network_obj.add_link(f"device_{source}", f"device_{target}", bw)
        
        # Possibly ensure cross-edges between clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dev_i = clusters[i][0]
                dev_j = clusters[j][0]
                if not network_obj.topology.has_edge(f"device_{dev_i}", f"device_{dev_j}"):
                    bw = np.random.uniform(0.1, 0.5)
                    network_obj.add_link(f"device_{dev_i}", f"device_{dev_j}", bw)
        
        return network_obj

class HybridCloudEdgeTopology(NetworkTopologyGenerator):
    """
    Generates a hybrid cloud-edge topology with cloud, regional, and edge nodes.
    """
    
    def generate(self):
        from ..core import Network
        
        network_obj = Network()
        
        # Create nodes
        for i in range(self.config.num_devices):
            network_obj.add_device(f"device_{i}")
            
        cloud_nodes = list(range(4))
        regional_nodes = list(range(4, 12))
        edge_nodes = list(range(12, 24))
        
        # Connect cloud nodes (high bandwidth)
        for i in cloud_nodes:
            for j in cloud_nodes:
                if i < j:
                    bw = np.random.uniform(20.0, 40.0)
                    network_obj.add_link(f"device_{i}", f"device_{j}", bw)
        
        # Connect regional to cloud
        cloud_regional_mapping = {
            4: [0], 5: [0], 6: [1], 7: [1],
            8: [2], 9: [2], 10: [3], 11: [3]
        }
        for regional, clouds in cloud_regional_mapping.items():
            for cnode in clouds:
                bw = np.random.uniform(5.0, 10.0)
                network_obj.add_link(f"device_{cnode}", f"device_{regional}", bw)
        
        # Connect edge to regional
        regional_edge_mapping = {
            4: [12, 13, 14], 5: [15, 16, 17],
            6: [18, 19], 7: [20, 21],
            8: [22, 23], 9: [], 10: [], 11: []
        }
        for rnode, edges in regional_edge_mapping.items():
            for enode in edges:
                bw = np.random.uniform(0.1, 1.0)
                network_obj.add_link(f"device_{rnode}", f"device_{enode}", bw)
        
        return network_obj

def create_topology(topology_type: str, config: TopologyConfig):
    """
    Factory function to create a specific topology based on the type string.
    """
    generator_map = {
        'edge_cluster': EdgeClusterTopology,
        'distributed_edge': DistributedEdgeTopology,
        'hybrid_cloud_edge': HybridCloudEdgeTopology
    }
    
    if topology_type not in generator_map:
        raise ValueError(
            f"Unknown topology type: {topology_type}. "
            f"Available: {list(generator_map.keys())}"
        )
    
    gen_cls = generator_map[topology_type](config)
    return gen_cls.generate()

def validate_topology(network_obj) -> bool:
    """
    Validate if the network_obj is connected and edges are consistent.
    'network_obj' is presumably an instance of core.Network but
    we do not name-import it at top-level.
    """
    # If no nodes or edges, it's invalid
    if network_obj.topology.number_of_nodes() == 0:
        return False
    if network_obj.topology.number_of_edges() == 0:
        return False
    
    import networkx as nx
    if not nx.is_connected(network_obj.topology):
        return False
    
    # Check that each edge is in network_obj.links
    for (src, tgt) in network_obj.topology.edges():
        if (src, tgt) not in network_obj.links and (tgt, src) not in network_obj.links:
            return False
    
    return True
