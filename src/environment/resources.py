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
from ..core import Device

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
            
        # Default tier proportions (can be customized)
        self.tier_proportions = {
            'cloud': 0.1,    # 10% cloud nodes
            'regional': 0.3, # 30% regional nodes
            'edge': 0.6      # 60% edge nodes
        }
        
    def generate_capabilities(self) -> Dict[str, DeviceCapabilities]:
        """Generate capabilities for all devices"""
        capabilities = {}
        
        # Calculate number of devices per tier
        num_cloud = max(1, int(self.num_devices * self.tier_proportions['cloud']))
        num_regional = max(1, int(self.num_devices * self.tier_proportions['regional']))
        num_edge = self.num_devices - num_cloud - num_regional
        
        # Generate cloud node capabilities (highest resources)
        cloud_capabilities = self._generate_tier_capabilities(
            num_cloud,
            memory_multiplier=4.0,
            compute_multiplier=4.0,
            tier="cloud"
        )
        
        # Generate regional node capabilities (medium resources)
        regional_capabilities = self._generate_tier_capabilities(
            num_regional,
            memory_multiplier=2.0,
            compute_multiplier=2.0,
            tier="regional"
        )
        
        # Generate edge node capabilities (base resources)
        edge_capabilities = self._generate_tier_capabilities(
            num_edge,
            memory_multiplier=1.0,
            compute_multiplier=1.0,
            tier="edge"
        )
        
        # Combine all capabilities
        device_id = 0
        for caps in [cloud_capabilities, regional_capabilities, edge_capabilities]:
            for capability in caps:
                capabilities[f"device_{device_id}"] = capability
                device_id += 1
                
        # Designate first device as source node
        if capabilities:
            first_device = capabilities["device_0"]
            first_device.is_source = True
            
        return capabilities
        
    def _generate_tier_capabilities(
        self,
        num_devices: int,
        memory_multiplier: float,
        compute_multiplier: float,
        tier: str
    ) -> List[DeviceCapabilities]:
        """Generate capabilities for devices in a specific tier"""
        capabilities = []
        
        # After sampling, clamp to [2,16]
        # Generate memory capacities
        memory_capacities = np.clip(
            self.memory_dist.sample(num_devices) * memory_multiplier,
            2.0, 16.0
        )
        # Generate compute capacities
        compute_capacities = np.clip(
            self.compute_dist.sample(num_devices) * compute_multiplier,
            10.0e9, 100.0e9  # whatever range you want for compute between 10 GFLOPs to 100GFLOPs
        )
        
        # Create DeviceCapabilities objects
        for i in range(num_devices):
            capabilities.append(DeviceCapabilities(
                memory_capacity=float(memory_capacities[i]),
                compute_capacity=float(compute_capacities[i]),
                device_tier=tier
            ))
            
        return capabilities

class BandwidthManager:
    """Manages bandwidth assignments for network links"""
    
    def __init__(
        self,
        cloud_bandwidth: Tuple[float, float] = (20.0, 40.0),    # Gbps
        regional_bandwidth: Tuple[float, float] = (5.0, 10.0),   # Gbps
        edge_bandwidth: Tuple[float, float] = (0.1, 1.0)        # Gbps
    ):
        self.bandwidth_ranges = {
            'cloud': cloud_bandwidth,
            'regional': regional_bandwidth,
            'edge': edge_bandwidth
        }
        
    def get_bandwidth(
        self,
        source_tier: str,
        target_tier: str
    ) -> float:
        """Get bandwidth for a link between two tiers"""
        # Use the lower tier's bandwidth range for inter-tier connections
        if source_tier == target_tier:
            bandwidth_range = self.bandwidth_ranges[source_tier]
        else:
            lower_tier = min(
                source_tier,
                target_tier,
                key=lambda x: ['edge', 'regional', 'cloud'].index(x)
            )
            bandwidth_range = self.bandwidth_ranges[lower_tier]
            
        return np.random.uniform(bandwidth_range[0], bandwidth_range[1])

def create_devices(
    capabilities: Dict[str, DeviceCapabilities]
) -> Dict[str, Device]:
    """Create Device objects from capabilities"""
    devices = {}
    
    for device_id, capability in capabilities.items():
        devices[device_id] = Device(
            device_id=device_id,
            memory_capacity=capability.memory_capacity,
            compute_capacity=capability.compute_capacity,
            is_source=capability.is_source
        )
        
    return devices

def validate_resource_distribution(
    capabilities: Dict[str, DeviceCapabilities]
) -> bool:
    """Validate resource distribution properties"""
    if not capabilities:
        return False
        
    # Ensure at least one source device
    if not any(cap.is_source for cap in capabilities.values()):
        return False
        
    # Ensure positive resource values
    for cap in capabilities.values():
        if cap.memory_capacity <= 0 or cap.compute_capacity <= 0:
            return False
            
    # Ensure valid device tiers
    valid_tiers = {'cloud', 'regional', 'edge'}
    if not all(cap.device_tier in valid_tiers for cap in capabilities.values()):
        return False
        
    return True

def get_tier_statistics(
    capabilities: Dict[str, DeviceCapabilities]
) -> Dict[str, Dict[str, float]]:
    """Calculate statistics for each device tier"""
    tier_stats = {}
    
    for tier in ['cloud', 'regional', 'edge']:
        tier_devices = [
            cap for cap in capabilities.values()
            if cap.device_tier == tier
        ]
        
        if tier_devices:
            memory_values = [dev.memory_capacity for dev in tier_devices]
            compute_values = [dev.compute_capacity for dev in tier_devices]
            
            tier_stats[tier] = {
                'count': len(tier_devices),
                'avg_memory': np.mean(memory_values),
                'std_memory': np.std(memory_values),
                'avg_compute': np.mean(compute_values),
                'std_compute': np.std(compute_values)
            }
            
    return tier_stats