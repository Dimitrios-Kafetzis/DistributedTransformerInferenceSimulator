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
# File:    src/core/device.py
# Description:
#   Defines the Device class and associated ResourceState tracking for
#   memory and compute usage, essential for distributed inference.
#
# ---------------------------------------------------------------------------

"""
Implements the Device class that encapsulates memory and compute resource states,
providing methods to allocate/deallocate resources for transformer components
and to maintain usage histories.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np

@dataclass
class ResourceState:
    """Tracks current resource utilization."""
    used: float = 0.0
    capacity: float = 0.0
    
    @property
    def available(self) -> float:
        """Returns how much of the resource is still available."""
        return max(0.0, self.capacity - self.used)
    
    @property
    def utilization(self) -> float:
        """Returns current utilization percentage (0-100%)."""
        return (self.used / self.capacity * 100) if self.capacity > 0 else 100.0

class Device:
    """
    Represents a compute device with specific memory (GB) and compute (GFLOPS/s) capacity.
    Provides methods to check/allocate/deallocate resources for assigned components.
    """
    
    def __init__(
        self,
        device_id: str,
        memory_capacity: float,   # in GB
        compute_capacity: float,  # in GFLOPS/s
        is_source: bool = False
    ):
        self.device_id = device_id
        self.is_source = is_source
        
        # Track memory/compute usage
        self.memory = ResourceState(capacity=memory_capacity)
        self.compute = ResourceState(capacity=compute_capacity)
        
        # Each assigned component stores (memory_req, compute_req)
        self.assigned_components: Dict[str, Tuple[float, float]] = {}
        # Cache usage (extra memory only)
        self.cache_assignments: Dict[str, float] = {}
        
        # History for debugging or analysis
        self.memory_history: List[float] = []
        self.compute_history: List[float] = []
        
    def can_accommodate(self, memory_req: float, compute_req: float) -> bool:
        """
        Check if this device has enough available memory and compute resources
        for the requested amounts.
        """
        return (
            self.memory.available >= memory_req and
            self.compute.available >= compute_req
        )
    
    def allocate_resources(
        self,
        component_id: str,
        memory_req: float,
        compute_req: float,
        cache_size: Optional[float] = None
    ) -> bool:
        """
        Attempt to allocate the requested resources for the given component.
        Optionally adds cache_size to memory usage.
        Returns True if successful; False otherwise.
        """
        if not self.can_accommodate(memory_req, compute_req):
            return False
        
        # Allocate for main memory/compute usage
        self.memory.used += memory_req
        self.compute.used += compute_req
        self.assigned_components[component_id] = (memory_req, compute_req)
        
        # If there's a cache requirement, add that memory usage too
        if cache_size is not None and cache_size > 0.0:
            self.memory.used += cache_size
            self.cache_assignments[component_id] = cache_size
        
        # Record utilization in history
        self.memory_history.append(self.memory.utilization)
        self.compute_history.append(self.compute.utilization)
        
        return True
        
    def deallocate_resources(self, component_id: str) -> None:
        """
        Deallocate resources used by the given component, restoring memory
        and compute capacity accordingly. Cache usage is also removed if present.
        """
        # If the component was using memory+compute
        if component_id in self.assigned_components:
            mem_used, comp_used = self.assigned_components[component_id]
            self.memory.used -= mem_used
            self.compute.used -= comp_used
            del self.assigned_components[component_id]
                
        # If the component was using extra cache memory
        if component_id in self.cache_assignments:
            cache_used = self.cache_assignments[component_id]
            self.memory.used -= cache_used
            del self.cache_assignments[component_id]
        
        # Make sure we don't go negative due to floating-point quirks
        self.memory.used = max(0.0, self.memory.used)
        self.compute.used = max(0.0, self.compute.used)
        
        # Record updated utilization
        self.memory_history.append(self.memory.utilization)
        self.compute_history.append(self.compute.utilization)
    
    def get_resource_state(self) -> Dict[str, float]:
        """
        Return a snapshot of current resource usage, useful for scheduling or debugging.
        """
        return {
            "device_id": self.device_id,
            "memory_used": self.memory.used,
            "memory_capacity": self.memory.capacity,
            "memory_utilization": self.memory.utilization,
            "compute_used": self.compute.used,
            "compute_capacity": self.compute.capacity,
            "compute_utilization": self.compute.utilization,
            "is_source": self.is_source,
        }
