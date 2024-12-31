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
# File:    src/algorithms/baselines.py
# Description:
#   Implements baseline distribution algorithms such as greedy, round-robin,
#   static, and dynamic migration strategies for comparison with the
#   resource-aware approach.
#
# ---------------------------------------------------------------------------

"""
Defines baseline or simpler distribution algorithms, including GreedyDistributor,
RoundRobinDistributor, StaticDistributor, and DynamicMigrationDistributor,
showcasing different strategies for distributed transformer inference.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set
from ..core import Device, Network, Transformer, AttentionHead, TransformerComponent
from .utils import ResourceRequirements, validate_assignment
from .resource_aware import AssignmentResult

class BaseDistributor(ABC):
    """Base class for all distribution algorithms"""
    
    def __init__(
        self,
        transformer: Transformer,
        network: Network,
        devices: Dict[str, Device]
    ):
        self.transformer = transformer
        self.network = network
        self.devices = devices
        
    @abstractmethod
    def compute_assignment(
        self,
        generation_step: int,
        previous_assignments: Optional[Dict[str, str]] = None,
        previous_cache: Optional[Dict[str, str]] = None
    ) -> AssignmentResult:
        """Compute component assignments for current generation step"""
        pass
        
    def _get_resource_usage(self) -> Dict[str, Dict[str, float]]:
        """Calculate current resource usage for all devices"""
        usage = {}
        for dev_id, device in self.devices.items():
            usage[dev_id] = {
                'memory_used': device.memory.used,
                'memory_capacity': device.memory.capacity,
                'compute_used': device.compute.used,
                'compute_capacity': device.compute.capacity
            }
        return usage

class GreedyDistributor(BaseDistributor):
    """
    Greedy placement strategy that assigns components to the first
    available device with sufficient resources
    """
    
    def compute_assignment(
        self,
        generation_step: int,
        previous_assignments: Optional[Dict[str, str]] = None,
        previous_cache: Optional[Dict[str, str]] = None
    ) -> AssignmentResult:
        assignments = {}
        cache_assignments = {}
        
        # Process components in order
        for component in self.transformer.get_all_components():
            assigned = False
            
            # Try each device in order
            for device in self.devices.values():
                # Calculate resource requirements
                memory_req = component.compute_memory_requirements(
                    self.transformer.current_sequence_length
                )
                compute_req = component.compute_flops(
                    self.transformer.current_sequence_length
                )
                
                # Add cache requirement if applicable
                if hasattr(component, 'compute_cache_memory'):
                    memory_req += component.compute_cache_memory(generation_step)
                
                # Check if device can accommodate
                if device.can_accommodate(memory_req, compute_req):
                    assignments[component.component_id] = device.device_id
                    if hasattr(component, 'compute_cache_memory'):
                        cache_assignments[component.component_id] = device.device_id
                    assigned = True
                    break
                    
            if not assigned:
                return AssignmentResult(
                    component_assignments=assignments,
                    cache_assignments=cache_assignments,
                    estimated_latency=float('inf'),
                    resource_usage=self._get_resource_usage(),
                    is_feasible=False
                )
                
        # Validate assignment
        is_feasible = validate_assignment(
            assignments,
            cache_assignments,
            self.transformer,
            self.devices,
            self.network,
            generation_step
        )
        
        # Estimate latency (using same method as resource-aware)
        latency = self._estimate_latency(
            assignments,
            cache_assignments,
            generation_step
        )
        
        return AssignmentResult(
            component_assignments=assignments,
            cache_assignments=cache_assignments,
            estimated_latency=latency,
            resource_usage=self._get_resource_usage(),
            is_feasible=is_feasible
        )

class RoundRobinDistributor(BaseDistributor):
    """
    Round-robin distribution strategy that distributes components
    evenly across available devices
    """
    
    def compute_assignment(
        self,
        generation_step: int,
        previous_assignments: Optional[Dict[str, str]] = None,
        previous_cache: Optional[Dict[str, str]] = None
    ) -> AssignmentResult:
        assignments = {}
        cache_assignments = {}
        device_list = list(self.devices.values())
        current_device_idx = 0
        
        for component in self.transformer.get_all_components():
            # Try devices in round-robin order
            attempts = 0
            assigned = False
            
            while attempts < len(device_list) and not assigned:
                device = device_list[current_device_idx]
                
                # Calculate resource requirements
                memory_req = component.compute_memory_requirements(
                    self.transformer.current_sequence_length
                )
                compute_req = component.compute_flops(
                    self.transformer.current_sequence_length
                )
                
                # Add cache requirement if applicable
                if hasattr(component, 'compute_cache_memory'):
                    memory_req += component.compute_cache_memory(generation_step)
                
                # Check if device can accommodate
                if device.can_accommodate(memory_req, compute_req):
                    assignments[component.component_id] = device.device_id
                    if hasattr(component, 'compute_cache_memory'):
                        cache_assignments[component.component_id] = device.device_id
                    assigned = True
                
                current_device_idx = (current_device_idx + 1) % len(device_list)
                attempts += 1
                
            if not assigned:
                return AssignmentResult(
                    component_assignments=assignments,
                    cache_assignments=cache_assignments,
                    estimated_latency=float('inf'),
                    resource_usage=self._get_resource_usage(),
                    is_feasible=False
                )
                
        return AssignmentResult(
            component_assignments=assignments,
            cache_assignments=cache_assignments,
            estimated_latency=self._estimate_latency(
                assignments,
                cache_assignments,
                generation_step
            ),
            resource_usage=self._get_resource_usage(),
            is_feasible=True
        )

class StaticDistributor(BaseDistributor):
    """
    Static partitioning strategy that maintains fixed assignments
    based on initial conditions
    """
    
    def __init__(
        self,
        transformer: Transformer,
        network: Network,
        devices: Dict[str, Device]
    ):
        super().__init__(transformer, network, devices)
        self.initial_assignments: Optional[Dict[str, str]] = None
        self.initial_cache: Optional[Dict[str, str]] = None
        
    def compute_assignment(
        self,
        generation_step: int,
        previous_assignments: Optional[Dict[str, str]] = None,
        previous_cache: Optional[Dict[str, str]] = None
    ) -> AssignmentResult:
        # Use initial assignment if available, otherwise create one
        if self.initial_assignments is None:
            # Create initial assignment using greedy strategy
            greedy = GreedyDistributor(self.transformer, self.network, self.devices)
            initial_result = greedy.compute_assignment(0, None, None)
            
            if not initial_result.is_feasible:
                return initial_result
                
            self.initial_assignments = initial_result.component_assignments
            self.initial_cache = initial_result.cache_assignments
            
        # Validate if static assignment is still feasible
        is_feasible = validate_assignment(
            self.initial_assignments,
            self.initial_cache,
            self.transformer,
            self.devices,
            self.network,
            generation_step
        )
        
        return AssignmentResult(
            component_assignments=self.initial_assignments,
            cache_assignments=self.initial_cache,
            estimated_latency=self._estimate_latency(
                self.initial_assignments,
                self.initial_cache,
                generation_step
            ) if is_feasible else float('inf'),
            resource_usage=self._get_resource_usage(),
            is_feasible=is_feasible
        )

class DynamicMigrationDistributor(BaseDistributor):
    """
    Dynamic migration strategy that reassigns components when
    resource utilization exceeds thresholds
    """
    
    def __init__(
        self,
        transformer: Transformer,
        network: Network,
        devices: Dict[str, Device],
        memory_threshold: float = 0.9,  # 90% utilization threshold
        compute_threshold: float = 0.9
    ):
        super().__init__(transformer, network, devices)
        self.memory_threshold = memory_threshold
        self.compute_threshold = compute_threshold
        
    def compute_assignment(
        self,
        generation_step: int,
        previous_assignments: Optional[Dict[str, str]] = None,
        previous_cache: Optional[Dict[str, str]] = None
    ) -> AssignmentResult:
        if previous_assignments is None:
            # Initial assignment using greedy strategy
            greedy = GreedyDistributor(self.transformer, self.network, self.devices)
            return greedy.compute_assignment(generation_step, None, None)
            
        assignments = previous_assignments.copy()
        cache_assignments = previous_cache.copy() if previous_cache else {}
        
        # Check for overloaded devices
        overloaded_devices = self._find_overloaded_devices(
            assignments,
            cache_assignments,
            generation_step
        )
        
        if overloaded_devices:
            # Attempt to migrate components from overloaded devices
            success = self._migrate_components(
                overloaded_devices,
                assignments,
                cache_assignments,
                generation_step
            )
            
            if not success:
                return AssignmentResult(
                    component_assignments=assignments,
                    cache_assignments=cache_assignments,
                    estimated_latency=float('inf'),
                    resource_usage=self._get_resource_usage(),
                    is_feasible=False
                )
                
        # Validate final assignment
        is_feasible = validate_assignment(
            assignments,
            cache_assignments,
            self.transformer,
            self.devices,
            self.network,
            generation_step
        )
        
        return AssignmentResult(
            component_assignments=assignments,
            cache_assignments=cache_assignments,
            estimated_latency=self._estimate_latency(
                assignments,
                cache_assignments,
                generation_step
            ) if is_feasible else float('inf'),
            resource_usage=self._get_resource_usage(),
            is_feasible=is_feasible
        )
        
    def _find_overloaded_devices(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> Set[str]:
        """Identify devices exceeding resource thresholds"""
        overloaded = set()
        
        for device_id, device in self.devices.items():
            # Calculate current utilization
            memory_used = 0.0
            compute_used = 0.0
            
            for comp_id, dev_id in assignments.items():
                if dev_id == device_id:
                    component = self.transformer.get_component(comp_id)
                    memory_used += component.compute_memory_requirements(
                        self.transformer.current_sequence_length
                    )
                    compute_used += component.compute_flops(
                        self.transformer.current_sequence_length
                    )
                    
            # Add cache usage
            for comp_id, dev_id in cache_assignments.items():
                if dev_id == device_id:
                    component = self.transformer.get_component(comp_id)
                    if hasattr(component, 'compute_cache_memory'):
                        memory_used += component.compute_cache_memory(
                            generation_step
                        )
                        
            # Check thresholds
            if (memory_used / device.memory.capacity > self.memory_threshold or
                compute_used / device.compute.capacity > self.compute_threshold):
                overloaded.add(device_id)
                
        return overloaded
        
    def _migrate_components(
        self,
        overloaded_devices: Set[str],
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> bool:
        """Attempt to migrate components from overloaded devices"""
        # Get components on overloaded devices
        components_to_migrate = {
            comp_id: dev_id
            for comp_id, dev_id in assignments.items()
            if dev_id in overloaded_devices
        }
        
        # Try to find new assignments for these components
        for comp_id in components_to_migrate:
            success = False
            component = self.transformer.get_component(comp_id)
            
            # Calculate resource requirements
            memory_req = component.compute_memory_requirements(
                self.transformer.current_sequence_length
            )
            compute_req = component.compute_flops(
                self.transformer.current_sequence_length
            )
            
            # Add cache requirement if applicable
            if hasattr(component, 'compute_cache_memory'):
                memory_req += component.compute_cache_memory(generation_step)
                
            # Try each non-overloaded device
            for device_id, device in self.devices.items():
                if device_id not in overloaded_devices:
                    if device.can_accommodate(memory_req, compute_req):
                        # Migrate component
                        assignments[comp_id] = device_id
                        if comp_id in cache_assignments:
                            cache_assignments[comp_id] = device_id
                        success = True
                        break
                        
            if not success:
                return False
                
        return True
        
    def _estimate_latency(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> float:
        """Estimate end-to-end latency including migration costs"""
        # Base latency calculation
        base_latency = super()._estimate_latency(
            assignments,
            cache_assignments,
            generation_step
        )
        
        # Add migration overhead if components were moved
        migration_overhead = self._calculate_migration_overhead(
            assignments,
            cache_assignments
        )
        
        return base_latency + migration_overhead
        
    def _calculate_migration_overhead(
        self,
        new_assignments: Dict[str, str],
        new_cache: Dict[str, str]
    ) -> float:
        """Calculate overhead from component migrations"""
        if not hasattr(self, '_previous_assignments'):
            self._previous_assignments = new_assignments.copy()
            self._previous_cache = new_cache.copy()
            return 0.0
            
        total_overhead = 0.0
        
        # Check for migrations
        for comp_id, dev_id in new_assignments.items():
            if (comp_id in self._previous_assignments and
                dev_id != self._previous_assignments[comp_id]):
                # Component was migrated, calculate transfer cost
                component = self.transformer.get_component(comp_id)
                data_size = component.compute_memory_requirements(
                    self.transformer.current_sequence_length
                )
                
                if comp_id in new_cache:
                    data_size += component.compute_cache_memory(
                        self.transformer.current_sequence_length - 
                        self.transformer.config.initial_sequence_length
                    )
                    
                # Calculate transfer time
                transfer_time = self.network.calculate_transfer_time(
                    self._previous_assignments[comp_id],
                    dev_id,
                    data_size
                )
                total_overhead += transfer_time
                
        # Update previous assignments for next iteration
        self._previous_assignments = new_assignments.copy()
        self._previous_cache = new_cache.copy()
        
        return total_overhead

    def _estimate_latency(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> float:
        """Estimate total latency for current assignments"""
        # Calculate computation time for each device
        compute_times = {dev_id: 0.0 for dev_id in self.devices}
        for comp_id, dev_id in assignments.items():
            component = self.transformer.get_component(comp_id)
            flops = component.compute_flops(self.transformer.current_sequence_length)
            compute_times[dev_id] += flops / self.devices[dev_id].compute.capacity
            
        # Calculate communication time between dependent components
        comm_time = 0.0
        for comp_id, dev_id in assignments.items():
            component = self.transformer.get_component(comp_id)
            
            # Add communication cost for attention heads to projection
            if isinstance(component, AttentionHead):
                if assignments["projection"] != dev_id:
                    data_size = (self.transformer.current_sequence_length * 
                               self.transformer.config.head_dim * 
                               self.transformer.config.precision_bytes) / (1024**3)
                    comm_time += self.network.calculate_transfer_time(
                        dev_id,
                        assignments["projection"],
                        data_size
                    )
                    
            # Add communication cost from projection to FFN
            elif comp_id == "projection":
                if assignments["ffn"] != dev_id:
                    data_size = (self.transformer.current_sequence_length * 
                               self.transformer.config.embedding_dim * 
                               self.transformer.config.precision_bytes) / (1024**3)
                    comm_time += self.network.calculate_transfer_time(
                        dev_id,
                        assignments["ffn"],
                        data_size
                    )
                    
        # Total latency is max computation time plus communication time
        max_compute_time = max(compute_times.values())
        return max_compute_time + comm_time