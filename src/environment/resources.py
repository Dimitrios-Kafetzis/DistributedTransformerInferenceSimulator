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
# File:    src/environment/resources.py
# Description:
#   Defines the ResourceDistributor for assigning memory and compute
#   capacities to devices, and other structures for modeling device
#   capabilities and bandwidth in simulation.
#
# ---------------------------------------------------------------------------

"""
Implements resource distribution and modeling, including log-normal
distributions for generating heterogeneous device capabilities, a
ResourceDistributor for provisioning memory and compute, and a
BandwidthManager for assigning link bandwidths in the network.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class DeviceCapabilities:
    """Represents the capabilities of a device"""
    memory_capacity: float  # GB
    compute_capacity: float  # GFLOPS
    is_source: bool = False
    device_tier: str = "edge"  # One of: cloud, regional, edge

@dataclass
class LogNormalDistribution:
    """Parameters for log-normal distribution"""
    mu: float
    sigma: float
    min_value: float
    max_value: float
    
    def sample(self, size: int = 1) -> np.ndarray:
        """Generate samples from the distribution"""
        samples = np.random.lognormal(self.mu, self.sigma, size)
        # Clip to specified range
        samples = np.clip(samples, self.min_value, self.max_value)
        return samples

class ResourceDistributor:
    """
    Manages the distribution of compute and memory resources
    across devices using log-normal distributions
    """
    
    def __init__(
        self,
        num_devices: int,
        memory_distribution: LogNormalDistribution,
        compute_distribution: LogNormalDistribution,
        seed: Optional[int] = None
    ):
        self.num_devices = num_devices
        self.memory_dist = memory_distribution
        self.compute_dist = compute_distribution
        
        if seed is not None:
            np.random.seed(seed)
            
        # Default tier proportions (example)
        self.tier_proportions = {
            'cloud': 0.1,      # 10% cloud nodes
            'regional': 0.3,   # 30% regional
            'edge': 0.6        # 60% edge
        }
        
    def generate_capabilities(self) -> Dict[str, DeviceCapabilities]:
        """Generate capabilities for all devices"""
        capabilities = {}
        
        # For tiering:
        num_cloud = max(1, int(self.num_devices * self.tier_proportions['cloud']))
        num_regional = max(1, int(self.num_devices * self.tier_proportions['regional']))
        num_edge = self.num_devices - num_cloud - num_regional
        
        # Cloud
        cloud_caps = self._generate_tier_capabilities(
            num_cloud, memory_multiplier=4.0, compute_multiplier=4.0, tier="cloud"
        )
        # Regional
        regional_caps = self._generate_tier_capabilities(
            num_regional, memory_multiplier=2.0, compute_multiplier=2.0, tier="regional"
        )
        # Edge
        edge_caps = self._generate_tier_capabilities(
            num_edge, memory_multiplier=1.0, compute_multiplier=1.0, tier="edge"
        )
        
        device_id = 0
        for tier_list in [cloud_caps, regional_caps, edge_caps]:
            for item in tier_list:
                capabilities[f"device_{device_id}"] = item
                device_id += 1
        
        # Mark the first device as source
        if capabilities:
            capabilities["device_0"].is_source = True
        
        return capabilities
        
    def _generate_tier_capabilities(
        self,
        num_devices: int,
        memory_multiplier: float,
        compute_multiplier: float,
        tier: str
    ) -> List[DeviceCapabilities]:
        capabilities_list = []
        
        mem_vals = np.clip(
            self.memory_dist.sample(num_devices) * memory_multiplier,
            2.0, 16.0
        )
        comp_vals = np.clip(
            self.compute_dist.sample(num_devices) * compute_multiplier,
            1.0e10, 1.0e11
        )
        
        for i in range(num_devices):
            capabilities_list.append(DeviceCapabilities(
                memory_capacity=float(mem_vals[i]),
                compute_capacity=float(comp_vals[i]),
                device_tier=tier
            ))
        return capabilities_list

class BandwidthManager:
    """
    Manages bandwidth assignments for links, e.g. cloud > regional > edge.
    """
    
    def __init__(
        self,
        cloud_bandwidth: Tuple[float, float] = (20.0, 40.0),
        regional_bandwidth: Tuple[float, float] = (5.0, 10.0),
        edge_bandwidth: Tuple[float, float] = (0.1, 1.0)
    ):
        self.bandwidth_ranges = {
            'cloud': cloud_bandwidth,
            'regional': regional_bandwidth,
            'edge': edge_bandwidth
        }
        
    def get_bandwidth(self, source_tier: str, target_tier: str) -> float:
        """
        Return some random bandwidth for a link between source_tier and target_tier.
        Uses the "lower" tier as the limiting factor.
        """
        tier_order = ['edge', 'regional', 'cloud']
        lower_tier = min(source_tier, target_tier, key=lambda x: tier_order.index(x))
        bw_range = self.bandwidth_ranges[lower_tier]
        return np.random.uniform(bw_range[0], bw_range[1])

def create_devices(capabilities: Dict[str, DeviceCapabilities]):
    """
    Create actual Device instances from the capabilities.
    We do a local import of `Device` here to avoid top-level import from core.
    """
    from ..core import Device
    
    devices = {}
    for dev_id, caps in capabilities.items():
        devices[dev_id] = Device(
            device_id=dev_id,
            memory_capacity=caps.memory_capacity,
            compute_capacity=caps.compute_capacity,
            is_source=caps.is_source
        )
    return devices

def validate_resource_distribution(capabilities: Dict[str, DeviceCapabilities]) -> bool:
    """Check if we have at least one source, positive resources, valid tiers, etc."""
    if not capabilities:
        return False
    
    # Must have at least one source
    if not any(caps.is_source for caps in capabilities.values()):
        return False
    
    # All must have positive resource
    for caps in capabilities.values():
        if caps.memory_capacity <= 0 or caps.compute_capacity <= 0:
            return False
    
    valid_tiers = {'cloud', 'regional', 'edge'}
    if not all(caps.device_tier in valid_tiers for caps in capabilities.values()):
        return False
    
    return True

def get_tier_statistics(capabilities: Dict[str, DeviceCapabilities]):
    """Compute stats per tier."""
    tier_stats = {}
    for tier in ['cloud', 'regional', 'edge']:
        tier_list = [c for c in capabilities.values() if c.device_tier == tier]
        if tier_list:
            mem_vals = [c.memory_capacity for c in tier_list]
            cmp_vals = [c.compute_capacity for c in tier_list]
            tier_stats[tier] = {
                'count': len(tier_list),
                'avg_memory': float(np.mean(mem_vals)),
                'std_memory': float(np.std(mem_vals)),
                'avg_compute': float(np.mean(cmp_vals)),
                'std_compute': float(np.std(cmp_vals)),
            }
    return tier_stats
