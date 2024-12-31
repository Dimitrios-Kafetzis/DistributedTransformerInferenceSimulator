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
# File:    src/simulation/scheduler.py
# Description:
#   Provides classes and methods for scheduling computation and communication
#   events in a distributed Transformer inference scenario, including an
#   ExecutionPlan, event ordering, and resource contention resolution.
#
# ---------------------------------------------------------------------------

"""
Contains the EventScheduler, ExecutionPlan, and related logic for arranging
Transformer components, transfers, and dependencies over the course of
the simulation. Manages when computations start, how data is transferred,
and the final completion times for each generation step.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

# Core imports (assuming your code structure):
from ..core import (
    Event, EventQueue, Transformer, Network, Device,
    TransformerComponent, AttentionHead, EventType
)

class ComponentState(Enum):
    """Possible states of a transformer component during execution."""
    PENDING = "pending"         # Waiting for dependencies
    READY = "ready"             # Ready to execute
    EXECUTING = "executing"     # Currently executing
    TRANSFERRING = "transferring"  # Data being transferred
    COMPLETED = "completed"     # Execution completed

@dataclass
class ExecutionPlan:
    """Represents the execution plan for a generation step."""
    step: int
    sequence_length: int
    dependencies: Dict[str, Set[str]]  # component_id -> set of dependency ids
    device_assignments: Dict[str, str]  # component_id -> device_id
    
    # Remove the old static 'estimated_completion_time' field and
    # use a computed property below.
    component_states: Dict[str, ComponentState] = field(default_factory=dict)
    completion_times: Dict[str, float] = field(default_factory=dict)

    @property
    def estimated_completion_time(self) -> float:
        """Dynamically compute the plan’s overall completion time."""
        if not self.completion_times:
            return 0.0
        return max(self.completion_times.values())


class EventScheduler:
    """Manages scheduling and execution of transformer components."""
    
    def __init__(
        self,
        transformer: Transformer,
        network: Network,
        devices: Dict[str, Device],
        event_queue: EventQueue
    ):
        self.transformer = transformer
        self.network = network
        self.devices = devices
        self.event_queue = event_queue
        
        # Track execution state
        self.current_plan: Optional[ExecutionPlan] = None
        self.active_transfers: Dict[str, Dict] = {}
        
    def create_execution_plan(
        self,
        step: int,
        device_assignments: Dict[str, str]
    ) -> ExecutionPlan:
        """
        Create an execution plan for the current generation step.
        Builds dependencies, initializes states, and calculates completion times.
        """
        plan = ExecutionPlan(
            step=step,
            sequence_length=self.transformer.current_sequence_length,
            dependencies=self._build_dependency_graph(),
            device_assignments=device_assignments
        )
        
        # Initialize all components as PENDING
        for component in self.transformer.get_all_components():
            plan.component_states[component.component_id] = ComponentState.PENDING
            
        # Calculate initial completion times
        self._calculate_completion_times(plan)
        
        return plan
        
    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """Construct the dependency graph for transformer components."""
        dependencies: Dict[str, Set[str]] = {}
        
        # Initialize empty sets for each component
        for component in self.transformer.get_all_components():
            dependencies[component.component_id] = set()
        
        # Only add “cache_...” dependencies if there’s a prior step
        # (i.e. if self.current_plan exists and step > 0).
        if self.current_plan and self.current_plan.step > 0:
            for head in self.transformer.attention_heads:
                dependencies[head.component_id].add(f"cache_{head.component_id}")
        
        # Projection depends on all attention heads
        for head in self.transformer.attention_heads:
            dependencies["projection"].add(head.component_id)
            
        # FFN depends on projection
        dependencies["ffn"].add("projection")
        
        return dependencies
        
    def _calculate_completion_times(self, plan: ExecutionPlan) -> None:
        """
        Calculate an approximate completion time for each component
        based on device speed and any inter-device transfers.
        """
        completion_times: Dict[str, float] = {}
        current_time = self.event_queue.current_time
        
        # We'll process until all dependencies are resolved
        components_to_process = set(plan.dependencies.keys())
        
        while components_to_process:
            # Components whose dependencies are all in completion_times
            ready_components = {
                comp_id
                for comp_id in components_to_process
                if all(dep in completion_times for dep in plan.dependencies[comp_id])
            }
            
            for comp_id in ready_components:
                component = self.transformer.get_component(comp_id)
                device_id = plan.device_assignments[comp_id]
                device = self.devices[device_id]
                
                # Compute flops and device speed => compute_time
                compute_time = (
                    component.compute_flops(plan.sequence_length) / device.compute.capacity
                )
                
                # Sum transfer times if dependencies are on different devices
                transfer_time = 0.0
                for dep_id in plan.dependencies[comp_id]:
                    if plan.device_assignments[dep_id] != device_id:
                        data_size = self._estimate_transfer_size(
                            dep_id, comp_id, plan.sequence_length
                        )
                        ttime = self.network.calculate_transfer_time(
                            plan.device_assignments[dep_id],
                            device_id,
                            data_size
                        )
                        transfer_time += ttime
                
                # Earliest possible start time = max of all dependencies
                start_after = max(
                    [completion_times[dep] for dep in plan.dependencies[comp_id]]
                    or [current_time]
                )
                completion_times[comp_id] = start_after + transfer_time + compute_time
            
            components_to_process -= ready_components
        
        plan.completion_times = completion_times
        # plan.estimated_completion_time is now a @property => no direct assignment
    
    def _estimate_transfer_size(
        self,
        source_id: str,
        target_id: str,
        sequence_length: int
    ) -> float:
        """
        Estimate data size in GB between two components, based on the
        transformer's config (embedding_dim, head_dim, etc.).
        """
        source = self.transformer.get_component(source_id)
        target = self.transformer.get_component(target_id)
        
        # Example logic:
        if isinstance(source, AttentionHead) and target_id == "projection":
            return (
                sequence_length
                * self.transformer.config.head_dim
                * self.transformer.config.precision_bytes
                / (1024**3)
            )
        elif source_id == "projection" and target_id == "ffn":
            return (
                sequence_length
                * self.transformer.config.embedding_dim
                * self.transformer.config.precision_bytes
                / (1024**3)
            )
        return 0.0
        
    def schedule_execution(self, plan: ExecutionPlan) -> None:
        """
        Entry point to schedule all components in the plan.
        Typically called at the beginning of each generation step.
        """
        self.current_plan = plan
        
        # Find initial ready components
        ready_components = self._get_ready_components()
        
        # Schedule them
        for comp_id in ready_components:
            self._schedule_component(comp_id)
            
    def _get_ready_components(self) -> Set[str]:
        """Return the set of components whose dependencies are completed."""
        ready = set()
        if not self.current_plan:
            return ready
        
        for comp_id, state in self.current_plan.component_states.items():
            if state == ComponentState.PENDING:
                # Check if all dependencies are completed
                deps_completed = all(
                    self.current_plan.component_states[dep] == ComponentState.COMPLETED
                    for dep in self.current_plan.dependencies[comp_id]
                )
                if deps_completed:
                    ready.add(comp_id)
        return ready
        
    def _schedule_component(self, component_id: str):
        """Schedule either transfers or the computation for the given component."""
        if not self.current_plan:
            return
        
        # Figure out if we need cross-device transfers
        transfers_required = False
        for dep_id in self.current_plan.dependencies[component_id]:
            if self.current_plan.device_assignments[dep_id] != \
               self.current_plan.device_assignments[component_id]:
                transfers_required = True
                self._schedule_transfer(dep_id, component_id)
        
        if not transfers_required:
            # No cross-device dependency => schedule computation
            self._schedule_computation(component_id)
            
    def _schedule_computation(self, component_id: str) -> None:
        """Enqueue compute_start and compute_complete for the given component."""
        if not self.current_plan:
            return
        
        device_id = self.current_plan.device_assignments[component_id]
        component = self.transformer.get_component(component_id)
        device = self.devices[device_id]
        
        # Compute flops => compute_time
        flops = component.compute_flops(self.current_plan.sequence_length)
        compute_time = flops / device.compute.capacity
        
        # Insert compute events into the queue
        self.event_queue.schedule_computation(
            component_id=component_id,
            device_id=device_id,
            computation_time=compute_time,
            metadata={"flops": flops}
        )
        self.current_plan.component_states[component_id] = ComponentState.EXECUTING
        
    def _schedule_transfer(self, source_id: str, target_id: str) -> None:
        """Enqueue transfer events for cross-device dependencies."""
        if not self.current_plan:
            return
        
        source_device = self.current_plan.device_assignments[source_id]
        target_device = self.current_plan.device_assignments[target_id]
        
        data_size = self._estimate_transfer_size(
            source_id,
            target_id,
            self.current_plan.sequence_length
        )
        transfer_time = self.network.calculate_transfer_time(
            source_device,
            target_device,
            data_size
        )
        
        # We schedule the transfer
        self.event_queue.schedule_transfer(
            component_id=f"{source_id}_to_{target_id}",
            source_device=source_device,
            target_device=target_device,
            data_size=data_size,
            transfer_time=transfer_time
        )
        self.current_plan.component_states[target_id] = ComponentState.TRANSFERRING
        
    def handle_event_completion(self, event: Event) -> None:
        """React when a compute or transfer event completes, possibly scheduling next steps."""
        if not self.current_plan:
            return
        
        if event.event_type == EventType.COMPUTE_COMPLETE:
            # Mark the component as COMPLETED
            self.current_plan.component_states[event.component_id] = ComponentState.COMPLETED
            
        elif event.event_type == EventType.TRANSFER_COMPLETE:
            # Once the transfer to a component is done, check if we can start its compute
            target_id = event.component_id.split("_to_")[1]
            if self._are_transfers_complete(target_id):
                self._schedule_computation(target_id)
        
        # Possibly other components now become ready:
        ready_components = self._get_ready_components()
        for comp_id in ready_components:
            self._schedule_component(comp_id)
            
    def _are_transfers_complete(self, component_id: str) -> bool:
        """
        Quick check if all cross-device transfers for a component are done.
        This might need more robust logic if a component has multiple dependencies
        from different devices.
        """
        if not self.current_plan:
            return False
        
        # Simplistic approach: if we've scheduled a transfer for each dependency
        # that was on a different device, we’d wait for each to produce a
        # TRANSFER_COMPLETE event. If none remain active => done.
        for dep_id in self.current_plan.dependencies[component_id]:
            if (dep_id != component_id  # not needed, but just in case
                and self.current_plan.device_assignments[dep_id]
                != self.current_plan.device_assignments[component_id]):
                
                # If there's still an entry in active_transfers, not done yet
                xfer_id = f"{dep_id}_to_{component_id}"
                if xfer_id in self.active_transfers:
                    return False
        return True
