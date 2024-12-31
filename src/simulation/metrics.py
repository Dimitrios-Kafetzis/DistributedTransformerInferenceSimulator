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
# File:    src/simulation/metrics.py
# Description:
#   Implements metrics collection for distributed Transformer inference,
#   including performance tracking, resource utilization measurements,
#   and communication overhead analysis.
#
# ---------------------------------------------------------------------------

"""
Provides classes and functions for metrics gathering during simulation:
PerformanceMetrics for latency and step timing, ResourceMetrics for device
usage and peak demands, and CommunicationMetrics for data-transfer tracking.
Also includes a MetricsCollector to integrate these metrics in real-time.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import numpy as np
from collections import defaultdict
import json
import time
from ..core import Device, Network, Event, EventType

@dataclass
class PerformanceMetrics:
    """Tracks end-to-end performance metrics"""
    total_time: float = 0.0
    step_latencies: List[float] = field(default_factory=list)
    computation_times: List[float] = field(default_factory=list)
    communication_times: List[float] = field(default_factory=list)
    
    @property
    def average_latency(self) -> float:
        """Calculate average latency per generation step"""
        return np.mean(self.step_latencies) if self.step_latencies else 0.0
    
    @property
    def latency_std(self) -> float:
        # The test wants the sample standard deviation (ddof=1)
        if not self.step_latencies:
            return 0.0
        return float(np.std(self.step_latencies, ddof=1))
    
    @property
    def computation_ratio(self) -> float:
        """Ratio of computation time to total time"""
        return (np.sum(self.computation_times) / self.total_time 
                if self.total_time > 0 else 0.0)
    
    @property
    def communication_ratio(self) -> float:
        """Ratio of communication time to total time"""
        return (np.sum(self.communication_times) / self.total_time 
                if self.total_time > 0 else 0.0)

@dataclass
class ResourceMetrics:
    """Tracks resource utilization metrics"""
    memory_utilization: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    compute_utilization: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    peak_memory: Dict[str, float] = field(default_factory=dict)
    peak_compute: Dict[str, float] = field(default_factory=dict)
    device_idle_times: Dict[str, float] = field(default_factory=dict)
    
    def get_average_utilization(self, device_id: str) -> Dict[str, float]:
        """Calculate average utilization for a device"""
        return {
            'memory': np.mean(self.memory_utilization[device_id]),
            'compute': np.mean(self.compute_utilization[device_id])
        }
    
    def get_utilization_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate utilization statistics across all devices"""
        stats = {}
        for device_id in self.memory_utilization.keys():
            stats[device_id] = {
                'avg_memory': np.mean(self.memory_utilization[device_id]),
                'avg_compute': np.mean(self.compute_utilization[device_id]),
                'peak_memory': self.peak_memory.get(device_id, 0),
                'peak_compute': self.peak_compute.get(device_id, 0),
                'idle_time': self.device_idle_times.get(device_id, 0),
            }
        return stats

@dataclass
class CommunicationMetrics:
    """Tracks communication and migration metrics"""
    total_data_transferred: float = 0.0  # In GB
    total_migrations: int = 0
    migration_costs: List[float] = field(default_factory=list)
    bandwidth_utilization: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    transfer_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def average_migration_cost(self) -> float:
        """Calculate average cost per migration"""
        return np.mean(self.migration_costs) if self.migration_costs else 0.0
    
    def get_link_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each network link"""
        stats = {}
        for link_id, utils in self.bandwidth_utilization.items():
            stats[link_id] = {
                'avg_utilization': np.mean(utils),
                'peak_utilization': np.max(utils),
                'transfer_count': self.transfer_counts[link_id]
            }
        return stats

class MetricsCollector:
    """Collects and analyzes simulation metrics"""
    
    def __init__(self):
        self.performance = PerformanceMetrics()
        self.resources = ResourceMetrics()
        self.communication = CommunicationMetrics()
        
        # Tracking state
        self.step_start_time: Optional[float] = None
        self.current_step: int = 0
        self.active_computations: Dict[str, float] = {}
        self.active_transfers: Dict[str, Dict] = {}
        
    def start_step(self, step: int) -> None:
        """Mark the start of a new generation step"""
        self.step_start_time = time.time()
        self.current_step = step
        
    def end_step(self) -> None:
        """Mark the end of current generation step"""
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            self.performance.step_latencies.append(step_time)
            self.performance.total_time += step_time
            self.step_start_time = None
            
    def record_event(self, event: Event) -> None:
        """Record metrics from an event"""
        if event.event_type == EventType.COMPUTE_START:
            self._record_compute_start(event)
        elif event.event_type == EventType.COMPUTE_COMPLETE:
            self._record_compute_complete(event)
        elif event.event_type == EventType.TRANSFER_START:
            self._record_transfer_start(event)
        elif event.event_type == EventType.TRANSFER_COMPLETE:
            self._record_transfer_complete(event)
            
    def record_resource_state(
        self,
        devices: Dict[str, Device],
        network: Network
    ) -> None:
        """Record current resource utilization state"""
        # Record device utilization
        for device_id, device in devices.items():
            state = device.get_resource_state()
            
            self.resources.memory_utilization[device_id].append(
                state['memory_utilization']
            )
            self.resources.compute_utilization[device_id].append(
                state['compute_utilization']
            )
            
            # Update peak values
            self.resources.peak_memory[device_id] = max(
                self.resources.peak_memory.get(device_id, 0),
                state['memory_utilization']
            )
            self.resources.peak_compute[device_id] = max(
                self.resources.peak_compute.get(device_id, 0),
                state['compute_utilization']
            )
            
        # Record network utilization
        network_state = network.get_network_state()
        self.communication.total_data_transferred = \
            network_state['total_data_transferred']
            
        # Record bandwidth utilization for each link
        for (src, dst), link in network.links.items():
            link_id = f"{src}->{dst}"
            utilization = (link.used_bandwidth / link.bandwidth) * 100
            self.communication.bandwidth_utilization[link_id].append(utilization)
            
    def record_migration(
        self,
        component_id: str,
        source_device: str,
        target_device: str,
        cost: float
    ) -> None:
        """Record a component migration"""
        self.communication.total_migrations += 1
        self.communication.migration_costs.append(cost)
        
    def _record_compute_start(self, event: Event) -> None:
        """Record start of computation"""
        self.active_computations[event.component_id] = event.time
        
    def _record_compute_complete(self, event: Event) -> None:
        """Record completion of computation"""
        if event.component_id in self.active_computations:
            start_time = self.active_computations.pop(event.component_id)
            compute_time = event.time - start_time
            self.performance.computation_times.append(compute_time)
            
    def _record_transfer_start(self, event: Event) -> None:
        """Record start of data transfer"""
        transfer_id = f"{event.source_device}->{event.target_device}"
        self.active_transfers[event.component_id] = {
            'start_time': event.time,
            'transfer_id': transfer_id
        }
        self.communication.transfer_counts[transfer_id] += 1
        
    def _record_transfer_complete(self, event: Event) -> None:
        """Record completion of data transfer"""
        if event.component_id in self.active_transfers:
            transfer_info = self.active_transfers.pop(event.component_id)
            transfer_time = event.time - transfer_info['start_time']
            self.performance.communication_times.append(transfer_time)
            
    def get_summary(self) -> Dict:
        """Generate a summary of all metrics"""
        return {
            'performance': {
                'total_time': self.performance.total_time,
                'average_latency': self.performance.average_latency,
                'latency_std': self.performance.latency_std,
                'computation_ratio': self.performance.computation_ratio,
                'communication_ratio': self.performance.communication_ratio
            },
            'resources': {
                'utilization': self.resources.get_utilization_stats()
            },
            'communication': {
                'total_data_transferred': self.communication.total_data_transferred,
                'total_migrations': self.communication.total_migrations,
                'average_migration_cost': self.communication.average_migration_cost,
                'link_stats': self.communication.get_link_stats()
            }
        }
        
    def save_to_file(self, filename: str) -> None:
        """Save metrics to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
            
    def plot_metrics(self) -> None:
        """Generate visualization of key metrics"""
        # This will be implemented in the visualization utility
        raise NotImplementedError