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
from typing import Dict, Optional, Tuple, List, Union
import numpy as np
from src.utils.logging import SimulationLogger, LogLevel

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
        """Returns current utilization percentage (0..100)."""
        return (self.used / self.capacity * 100) if self.capacity > 0 else 100.0


class Device:
    """
    Represents a compute device with specific memory (GB) and compute (GFLOPS/s) capacity.
    Provides methods to check/allocate/deallocate resources for assigned components.
    
    assigned_components[comp_id] = {
        "memory": float,     # how much memory we allocated for main usage
        "compute": float,    # how much compute usage allocated
        "ephemeral": bool    # True => free next step; False => keep allocated
    }
    
    cache_assignments[comp_id] = {
        "memory": float,     # additional memory for cache usage
        "ephemeral": bool
    }
    """

    def __init__(
        self,
        device_id: str,
        memory_capacity: float,   # in GB
        compute_capacity: float,  # in GFLOPS/s
        is_source: bool = False,
        logger: Optional[SimulationLogger] = None
    ):
        self.device_id = device_id
        self.is_source = is_source
        
        # Track memory/compute usage
        self.memory = ResourceState(capacity=memory_capacity)
        self.compute = ResourceState(capacity=compute_capacity)
        
        # Each assigned component: comp_id -> { "memory": float, "compute": float, "ephemeral": bool }
        self.assigned_components: Dict[str, Dict[str, Union[float, bool]]] = {}
        
        # Each cache assignment: comp_id -> { "memory": float, "ephemeral": bool }
        self.cache_assignments: Dict[str, Dict[str, Union[float, bool]]] = {}
        
        # History for debugging or analysis
        self.memory_history: List[float] = []
        self.compute_history: List[float] = []
        
        self.logger = logger  # for logging if desired

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
        cache_size: Optional[float] = None,
        ephemeral: bool = True
    ) -> bool:
        """
        Attempt to allocate the requested resources for the given component.
        :param ephemeral: if True, this usage can be freed on next step. 
                          if False, it remains across steps (like heads, caches).
        Returns True if successful; False otherwise.
        """
        if not self.can_accommodate(memory_req, compute_req):
            return False
        
        # Allocate main memory & compute
        self.memory.used += memory_req
        self.compute.used += compute_req
        
        self.assigned_components[component_id] = {
            "memory": memory_req,
            "compute": compute_req,
            "ephemeral": ephemeral
        }
        
        # Allocate cache memory if any
        if cache_size and cache_size > 0.0:
            if self.memory.available < cache_size:
                # revert main memory & compute if cache can't fit
                self.memory.used -= memory_req
                self.compute.used -= compute_req
                del self.assigned_components[component_id]
                return False
            self.memory.used += cache_size
            self.cache_assignments[component_id] = {
                "memory": cache_size,
                "ephemeral": ephemeral
            }
        
        # Record utilization in history
        self.memory_history.append(self.memory.utilization)
        self.compute_history.append(self.compute.utilization)
        
        # Optional log
        if self.logger:
            self.logger.log_event(
                "device_alloc",
                f"Device={self.device_id}: Allocated comp={component_id}, ephemeral={ephemeral}, "
                f"mem_req={memory_req:.4f} + (cache={cache_size or 0:.4f}), compute_req={compute_req:.4f}. "
                f"Now used mem={self.memory.used:.4f}/{self.memory.capacity}, compute={self.compute.used:.4f}/{self.compute.capacity}",
                level=LogLevel.DEBUG
            )
        
        return True
        
    def deallocate_resources(self, component_id: str, force: bool = True) -> None:
        """
        Deallocate resources used by the given component, restoring memory
        and compute capacity accordingly. Cache usage is also removed if present.
        
        :param force: if True, forcibly remove the component even if ephemeral=False.
                      Usually for partial approach, we only call this if ephemeral=True or we do want to free it.
        """
        # 1) If the component was using memory+compute:
        if component_id in self.assigned_components:
            info = self.assigned_components[component_id]
            ephemeral_flag = bool(info.get("ephemeral", True))
            if force or ephemeral_flag:
                mem_used = float(info["memory"])
                comp_used = float(info["compute"])
                self.memory.used -= mem_used
                self.compute.used -= comp_used
                del self.assigned_components[component_id]
        
        # 2) If the component has extra cache usage
        if component_id in self.cache_assignments:
            cache_info = self.cache_assignments[component_id]
            ephemeral_flag = bool(cache_info.get("ephemeral", True))
            if force or ephemeral_flag:
                cache_used = float(cache_info["memory"])
                self.memory.used -= cache_used
                del self.cache_assignments[component_id]
        
        # Make sure we don't go negative due to floating-point quirks
        self.memory.used = max(0.0, self.memory.used)
        self.compute.used = max(0.0, self.compute.used)
        
        # Record updated utilization
        self.memory_history.append(self.memory.utilization)
        self.compute_history.append(self.compute.utilization)

        if self.logger:
            self.logger.log_event(
                "device_dealloc",
                f"Device={self.device_id}: Freed resources for comp={component_id} (force={force}). "
                f"Now used mem={self.memory.used:.4f}/{self.memory.capacity}, "
                f"compute={self.compute.used:.4f}/{self.compute.capacity}",
                level=LogLevel.DEBUG
            )
    
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
