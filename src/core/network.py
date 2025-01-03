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
# File:    src/core/network.py
# Description:
#   Implements the Network class for modeling device connectivity, managing
#   bandwidth-limited links, and tracking active data transfers in the simulator.
#
# ---------------------------------------------------------------------------

"""
Contains the Network class responsible for storing a graph-based topology,
adding/removing devices and links, calculating transfer times over available bandwidth,
and maintaining states for ongoing transfers within the simulation.
"""

import networkx as nx
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class Link:
    """Represents a network link with bandwidth and current utilization."""
    bandwidth: float  # in Gbps
    used_bandwidth: float = 0.0

    @property
    def available_bandwidth(self) -> float:
        """Return how much bandwidth remains available."""
        return max(0.0, self.bandwidth - self.used_bandwidth)

class UndirectedGraphWithIsConnected(nx.Graph):
    """
    A small subclass so we can call graph.is_connected().
    The environment tests do: assert network.topology.is_connected()
    """
    def is_connected(self):
        # For undirected graphs, we can use nx.is_connected(...)
        return nx.is_connected(self)

class Network:
    """Represents the network topology and manages communication between devices."""
    
    def __init__(self, directed: bool = False):
        """
        :param directed: If True, use a directed topology (nx.DiGraph).
                         If False, use undirected (nx.Graph).
        """
        # If your test_network test insists on a DiGraph, set directed=True.
        # If your environment wants is_connected( ), set directed=False.
        if directed:
            self.topology = nx.DiGraph()
        else:
            self.topology = UndirectedGraphWithIsConnected()
        
        self.links: Dict[Tuple[str, str], Link] = {}
        self.active_transfers: List[Dict] = []
        
        # Performance metrics
        self.total_data_transferred = 0.0
        self.peak_bandwidth_usage = 0.0

    def add_device(self, device_id: str) -> None:
        """Add a device (node) to the network."""
        self.topology.add_node(device_id)
        
    def add_link(
        self,
        source: str,
        target: str,
        bandwidth: float,
        bidirectional: bool = False
    ) -> None:
        """
        Add a network link between devices.
        For an undirected graph, store both (source, target) and (target, source).
        
        :param source:      ID of the source device
        :param target:      ID of the target device
        :param bandwidth:   Link bandwidth in Gbps
        :param bidirectional:  If True, create link in both directions 
                               (for a DiGraph, that means separate edges).
        """
        # Always add source -> target
        self.topology.add_edge(source, target, bandwidth=bandwidth)
        self.links[(source, target)] = Link(bandwidth=bandwidth)
        
        # If using an undirected graph, or user specified bidirectional in a DiGraph:
        if not self.topology.is_directed() or bidirectional:
            self.topology.add_edge(target, source, bandwidth=bandwidth)
            self.links[(target, source)] = Link(bandwidth=bandwidth)
    
    def get_path(self, source: str, target: str) -> List[str]:
        """Find shortest path between two devices."""
        try:
            return nx.shortest_path(self.topology, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Raise ValueError for either missing node or no route
            raise ValueError(f"No path exists between {source} and {target}")
            
    def calculate_transfer_time(
        self,
        source: str,
        target: str,
        data_size: float  # in GB
    ) -> float:
        """
        Calculate expected transfer time in seconds,
        based on the lowest (bottleneck) available bandwidth along the path.
        data_size in GB, bandwidth in Gbps => time in seconds.
        """
        if source == target:
            return 0.0

        path = self.get_path(source, target)
        min_bandwidth = float('inf')
        
        for i in range(len(path) - 1):
            link_obj = self.links[(path[i], path[i+1])]
            if link_obj.available_bandwidth < min_bandwidth:
                min_bandwidth = link_obj.available_bandwidth
        
        return data_size / min_bandwidth if min_bandwidth > 0 else float('inf')
        
    def start_transfer(
        self,
        source: str,
        target: str,
        data_size: float,
        start_time: float
    ) -> Dict:
        """
        Begin a data transfer between devices.
        
        :return: A dictionary with transfer metadata,
                 including the path and the expected completion time.
        """
        path = self.get_path(source, target)
        transfer_time = self.calculate_transfer_time(source, target, data_size)
        
        # Increase used_bandwidth along each link in the path
        if transfer_time < float('inf'):  # i.e. not zero bandwidth
            required_bandwidth = data_size / transfer_time
            for i in range(len(path) - 1):
                link_obj = self.links[(path[i], path[i+1])]
                link_obj.used_bandwidth += required_bandwidth

                # Track peak usage
                if link_obj.used_bandwidth > self.peak_bandwidth_usage:
                    self.peak_bandwidth_usage = link_obj.used_bandwidth
        
        transfer = {
            'source': source,
            'target': target,
            'data_size': data_size,
            'path': path,
            'start_time': start_time,
            'end_time': start_time + transfer_time
        }
        
        self.active_transfers.append(transfer)
        self.total_data_transferred += data_size
        
        return transfer
        
    def complete_transfer(self, transfer: Dict) -> None:
        """
        Mark a transfer as complete and free up the bandwidth it used along the path.
        """
        path = transfer['path']
        data_size = transfer['data_size']
        duration = transfer['end_time'] - transfer['start_time']

        if duration > 0:
            required_bandwidth = data_size / duration
            for i in range(len(path) - 1):
                link_obj = self.links[(path[i], path[i+1])]
                link_obj.used_bandwidth -= required_bandwidth

        # Remove from active transfers
        self.active_transfers.remove(transfer)
        
    def get_network_state(self) -> Dict:
        """
        Return a snapshot of the network state:
          - total data transferred
          - peak bandwidth usage
          - count of active transfers
          - number of nodes/edges
        """
        return {
            'total_data_transferred': self.total_data_transferred,
            'peak_bandwidth_usage': self.peak_bandwidth_usage,
            'active_transfers': len(self.active_transfers),
            'topology_stats': {
                'nodes': self.topology.number_of_nodes(),
                'edges': self.topology.number_of_edges()
            }
        }
