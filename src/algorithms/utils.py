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
# File:    src/algorithms/utils.py
# Description:
#   Contains helper classes and functions for distribution algorithms,
#   including resource requirement calculations, communication cost
#   estimations, and assignment validation utilities.
#
# ---------------------------------------------------------------------------

"""
Provides utility methods and classes for the algorithms subpackage, such
as resource requirement computation, communication cost calculation, and
assignment validation logic for distributed transformer inference.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from ..core import Device, Network, Transformer, TransformerComponent

@dataclass
class ResourceRequirements:
    """Represents resource requirements for a component"""
    memory_gb: float
    compute_flops: float
    cache_gb: Optional[float] = None
    
    @property
    def total_memory_gb(self) -> float:
        """Total memory including cache if present"""
        return self.memory_gb + (self.cache_gb or 0.0)

@dataclass
class CommunicationCost:
    """Represents communication cost between components"""
    source_component: str
    target_component: str
    data_size_gb: float
    source_device: str
    target_device: str

def calculate_resource_requirements(
    component: TransformerComponent,
    transformer: Transformer,
    generation_step: int
) -> ResourceRequirements:
    """Calculate resource requirements for a component at current step"""
    memory_req = component.compute_memory_requirements(
        transformer.current_sequence_length
    )
    compute_req = component.compute_flops(
        transformer.current_sequence_length
    )
    
    cache_req = None
    if hasattr(component, 'compute_cache_memory'):
        cache_req = component.compute_cache_memory(generation_step)
        
    return ResourceRequirements(
        memory_gb=memory_req,
        compute_flops=compute_req,
        cache_gb=cache_req
    )

def get_component_dependencies(
    component_id: str,
    transformer: Transformer
) -> Set[str]:
    """Get set of component IDs that this component depends on"""
    dependencies = set()
    
    if component_id == "projection":
        # Projection layer depends on all attention heads
        dependencies.update(
            head.component_id for head in transformer.attention_heads
        )
    elif component_id == "ffn":
        # FFN depends on projection
        dependencies.add("projection")
    elif component_id.startswith("head_"):
        # Attention heads might depend on previous cache state
        pass  # Cache dependencies handled separately
        
    return dependencies

def calculate_communication_costs(
    assignments: Dict[str, str],
    transformer: Transformer,
    network: Network
) -> List[CommunicationCost]:
    """Calculate communication costs for current assignments"""
    costs = []
    
    # Process each component
    for comp_id, dev_id in assignments.items():
        # Get dependencies
        deps = get_component_dependencies(comp_id, transformer)
        
        # Calculate costs for each dependency
        for dep_id in deps:
            if dep_id in assignments and assignments[dep_id] != dev_id:
                data_size = estimate_transfer_size(
                    dep_id,
                    comp_id,
                    transformer
                )
                
                costs.append(CommunicationCost(
                    source_component=dep_id,
                    target_component=comp_id,
                    data_size_gb=data_size,
                    source_device=assignments[dep_id],
                    target_device=dev_id
                ))
                
    return costs

def estimate_transfer_size(
    source_id: str,
    target_id: str,
    transformer: Transformer
) -> float:
    """Estimate size of data transfer between components in GB"""
    if source_id.startswith("head_") and target_id == "projection":
        # Transfer attention head output
        return (transformer.current_sequence_length * 
                transformer.config.head_dim * 
                transformer.config.precision_bytes) / (1024**3)
    elif source_id == "projection" and target_id == "ffn":
        # Transfer projection output
        return (transformer.current_sequence_length * 
                transformer.config.embedding_dim * 
                transformer.config.precision_bytes) / (1024**3)
    return 0.0

def check_device_capacity(
    device: Device,
    requirements: ResourceRequirements
) -> bool:
    """Check if device can accommodate resource requirements"""
    return (device.memory.available >= requirements.total_memory_gb and
            device.compute.available >= requirements.compute_flops)

def validate_assignment(
    assignments: Dict[str, str],
    cache_assignments: Dict[str, str],
    transformer: Transformer,
    devices: Dict[str, Device],
    network: Network,
    generation_step: int
) -> bool:
    """
    Validate if assignment is feasible considering:
    - Resource constraints
    - Communication constraints
    - Privacy constraints
    """
    # Check if all components are assigned
    if len(assignments) != len(transformer.get_all_components()):
        return False
        
    # Track resource usage per device
    device_memory_usage = {dev_id: 0.0 for dev_id in devices}
    device_compute_usage = {dev_id: 0.0 for dev_id in devices}
    
    # Check each component's assignment
    for comp_id, dev_id in assignments.items():
        component = transformer.get_component(comp_id)
        device = devices[dev_id]
        
        # Calculate resource requirements
        requirements = calculate_resource_requirements(
            component,
            transformer,
            generation_step
        )
        
        # Update device usage
        device_memory_usage[dev_id] += requirements.memory_gb
        device_compute_usage[dev_id] += requirements.compute_flops
        
        # Add cache usage if applicable
        if comp_id in cache_assignments:
            if requirements.cache_gb is not None:
                device_memory_usage[dev_id] += requirements.cache_gb
                
    # Verify device capacities aren't exceeded
    for dev_id, device in devices.items():
        if (device_memory_usage[dev_id] > device.memory.capacity or
            device_compute_usage[dev_id] > device.compute.capacity):
            return False
            
    # Check network feasibility
    comm_costs = calculate_communication_costs(assignments, transformer, network)
    for cost in comm_costs:
        # Check if network can support the required transfer
        if network.calculate_transfer_time(
            cost.source_device,
            cost.target_device,
            cost.data_size_gb
        ) == float('inf'):
            return False
            
    # All checks passed
    return True

def calculate_total_latency(
    assignments: Dict[str, str],
    cache_assignments: Dict[str, str],
    transformer: Transformer,
    devices: Dict[str, Device],
    network: Network,
    generation_step: int
) -> float:
    """Calculate total end-to-end latency for current assignment"""
    # Calculate computation time for each device
    compute_times = {dev_id: 0.0 for dev_id in devices}
    for comp_id, dev_id in assignments.items():
        component = transformer.get_component(comp_id)
        requirements = calculate_resource_requirements(
            component,
            transformer,
            generation_step
        )
        compute_times[dev_id] += (requirements.compute_flops / 
                                devices[dev_id].compute.capacity)
        
    # Calculate communication time
    comm_costs = calculate_communication_costs(assignments, transformer, network)
    comm_time = sum(
        network.calculate_transfer_time(
            cost.source_device,
            cost.target_device,
            cost.data_size_gb
        )
        for cost in comm_costs
    )
    
    # Total latency is max computation time plus communication time
    max_compute_time = max(compute_times.values())
    return max_compute_time + comm_time

def estimate_migration_cost(
    component: TransformerComponent,
    source_device: str,
    target_device: str,
    transformer: Transformer,
    network: Network,
    generation_step: int
) -> float:
    """Estimate cost of migrating a component between devices"""
    # Calculate total data that needs to be transferred
    requirements = calculate_resource_requirements(
        component,
        transformer,
        generation_step
    )
    
    total_data = requirements.memory_gb
    if requirements.cache_gb is not None:
        total_data += requirements.cache_gb
        
    # Calculate transfer time
    return network.calculate_transfer_time(
        source_device,
        target_device,
        total_data
    )