# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Dimitrios Kafetzis
#
# This file is part of the Transformer Inference Simulator project.
# Licensed under the MIT License; you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#   https://opensource.org/licenses/MIT
#
# Author: Dimitrios Kafetzis (dimitrioskafetzis@gmail.com)
# File: src/algorithms/resource_aware.py
# Description:
#   This script implements the resource-aware distribution algorithm for
#   transformer inference, as described in Section IV of the paper:
#
#       "Large Language Model Partitioning for Low-Latency Inference at the Edge"
#
#   The algorithm partitions the decoder blocks of a Transformer at the attention-head
#   level (and associated K/V caches), along with the projection and feed-forward layers.
#   It assigns each block to an edge device by computing a scoring function S(i, j, t)
#   that combines memory, compute, and communication constraints. Two scoring modes are
#   provided:
#
#     (1) "max" mode: S(i,j,t) = max { memory_ratio, compute_ratio, communication_ratio }.
#     (2) "weighted_sum" mode: S(i,j,t) = alpha * compute_ratio + beta * memory_ratio + gamma * comm_ratio.
#
#   A block is deemed feasible on a device if S(i, j, t) <= 1, ensuring that memory and compute
#   constraints are not violated, and communication overhead remains acceptable.
#
#   The algorithm is myopic (token-by-token) and reassigns blocks dynamically at each generation step,
#   incorporating migration costs when a block is moved from one device to another. This implementation
#   follows the pseudocode and description in the paper, where the objective is to minimize the total delay
#   (inference delay plus migration delay) at each step while satisfying resource constraints.
#
# ---------------------------------------------------------------------------

"""
Contains the main ResourceAwareDistributor class and related data structures,
implementing a multi-dimensional scoring function and constraints to achieve
optimized distributed assignments for transformer inference.

This module corresponds to the resource-aware algorithm described in the attached
paper, which uses attention head-level partitioning for autoregressive LLM inference.
The scoring function S(i, j, t) is used to decide the best device for each block,
and the algorithm handles migrations and backtracking if necessary.
"""

import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.utils.logging import SimulationLogger, LogLevel  # adapt import path if necessary
from ..core import Device, Network, Transformer, TransformerComponent
from .utils import ResourceRequirements, CommunicationCost, validate_assignment, compute_3phase_latency

@dataclass
class AssignmentResult:
    """
    Data structure representing the outcome of the resource-aware block assignment.

    Attributes:
        component_assignments: Mapping from Transformer component IDs to device IDs.
        cache_assignments: Mapping from attention head IDs (or their caches) to device IDs.
        estimated_latency: The estimated total latency (including compute, communication, and migration costs).
        resource_usage: Snapshot of resource usage on devices after assignment.
        is_feasible: Boolean flag indicating whether a feasible assignment was found.
        error: Optional error message if assignment failed.
        migrations: Optional list of migration tuples (from_device, to_device, component_id).
        communication_time: Total communication time incurred.
        data_transferred_gb: Total data transferred (in GB) due to inter-device communications.
    """
    component_assignments: Dict[str, str]
    cache_assignments: Dict[str, str]
    estimated_latency: float
    resource_usage: Dict[str, Dict[str, float]]
    is_feasible: bool
    error: Optional[str] = None
    migrations: Optional[List[tuple]] = None
    communication_time: float = 0.0
    data_transferred_gb: float = 0.0

@dataclass
class ScoringFunction:
    """
    Implements the scoring function S(i, j, t) for a given component i and candidate device j at generation step t.
    
    There are two modes for computing the score:
      (1) 'max' mode: Returns max(compute_ratio, memory_ratio, comm_ratio).
          This mode penalizes the device based on the worst-case resource bottleneck.
      (2) 'weighted_sum' mode: Returns alpha * compute_ratio + beta * memory_ratio + gamma * comm_ratio.
          This mode allows adjusting the relative importance of compute, memory, and communication constraints.
    
    The score is used to choose the best device for placing a Transformer component such that
    the placement is resource-feasible (S(i, j, t) <= 1) and minimizes overall latency.
    """
    use_weighted_sum: bool = False  # If True, use weighted sum; else, use max(...)
    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3

    def compute(
        self,
        component: TransformerComponent,
        device: Device,
        network: Network,
        transformer: Transformer,
        current_assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> float:
        """
        Compute the scoring function S(i, j, t) for assigning the given component to the device.

        Returns infinity if the assignment is infeasible or if a policy forbids placement
        (e.g., non-input components on a source device). Otherwise, computes three ratios:
          - Compute ratio: The ratio of the component's FLOPs to the device's compute capacity.
          - Memory ratio: The ratio of the component's memory requirement (including cache if applicable)
            to the device's memory capacity.
          - Communication ratio: An estimate of the additional communication delay if the component is placed
            on this device relative to its dependencies.
          
        Depending on the mode:
          - If use_weighted_sum is False, returns the maximum of the three ratios.
          - If True, returns a weighted sum using the parameters alpha, beta, and gamma.
        
        This scoring function corresponds to the formulation in the paper (see Section IV-A),
        where the goal is to choose a device that minimizes the maximum load and communication cost.
        
        Args:
            component: The Transformer component being assigned.
            device: Candidate Device for assignment.
            network: The network object for computing transfer times.
            transformer: The transformer model with current configuration.
            current_assignments: Existing mapping of components to devices.
            cache_assignments: Existing cache assignments.
            generation_step: Current generation step (token index).
        
        Returns:
            A float score representing the cost of placing the component on the device.
            Lower scores indicate better placement.
        """
        # Example policy: if the device is marked as a "source" device and the component is not an input or positional component,
        # then we do not allow assignment (return infinity).
        if device.is_source and component.component_id not in ["input", "position"]:
            return float('inf')

        # Compute the three ratios
        compute_ratio = _compute_ratio(component, device, transformer)
        memory_ratio = _memory_ratio(component, device, transformer, generation_step)
        comm_ratio = _communication_ratio(
            component, device, network, transformer, current_assignments, cache_assignments
        )

        if not self.use_weighted_sum:
            # In 'max' mode, we penalize the device based on the worst-case ratio.
            return max(compute_ratio, memory_ratio, comm_ratio)
        else:
            # In weighted-sum mode, we combine the ratios linearly.
            return self.alpha * compute_ratio + self.beta * memory_ratio + self.gamma * comm_ratio

def _compute_ratio(component: TransformerComponent, device: Device, transformer: Transformer) -> float:
    """
    Calculate the compute ratio: the fraction of device compute capacity required by the component.
    
    Uses the component's estimated FLOPs (from compute_flops) at the current sequence length.
    Returns infinity if the device's compute capacity is zero.
    
    Args:
        component: Transformer component.
        device: Device under consideration.
        transformer: Transformer model (to get current sequence length).
        
    Returns:
        Compute ratio as a float.
    """
    flops = component.compute_flops(transformer.current_sequence_length)
    if device.compute.capacity <= 0:
        return float('inf')
    return flops / device.compute.capacity

def _memory_ratio(
    component: TransformerComponent,
    device: Device,
    transformer: Transformer,
    generation_step: int
) -> float:
    """
    Calculate the memory ratio: the fraction of device memory required by the component.
    
    This function computes the memory needed for the component using compute_memory_requirements.
    For components with a K/V cache (e.g., attention heads), it adds the cache memory requirement.
    Returns infinity if the device's memory capacity is zero.
    
    Args:
        component: Transformer component.
        device: Device under consideration.
        transformer: Transformer model.
        generation_step: Current generation step (affects cache size).
        
    Returns:
        Memory ratio as a float.
    """
    memory_req = component.compute_memory_requirements(transformer.current_sequence_length)
    if hasattr(component, 'compute_cache_memory'):
        memory_req += component.compute_cache_memory(generation_step)
    if device.memory.capacity <= 0:
        return float('inf')
    return memory_req / device.memory.capacity

def _communication_ratio(
    component: TransformerComponent,
    device: Device,
    network: Network,
    transformer: Transformer,
    current_assignments: Dict[str, str],
    cache_assignments: Dict[str, str]
) -> float:
    """
    Estimate the communication ratio, which approximates the extra delay incurred due to transferring
    data between devices when the component's dependencies are located on different devices.
    
    For each dependency (e.g., for 'projection' that depends on attention heads), if the dependency is assigned
    to a different device than the current one, we estimate the transfer size and use the network's transfer
    time calculation to sum a total communication cost.
    
    Args:
        component: Transformer component.
        device: Candidate device.
        network: Network object to calculate transfer times.
        transformer: Transformer model.
        current_assignments: Current mapping of components to devices.
        cache_assignments: Current cache assignments (if any).
        
    Returns:
        Communication cost as a float (total estimated transfer time).
    """
    total_comm_cost = 0.0
    deps = _get_dependencies_global(component.component_id, transformer)
    for dep_id in deps:
        if dep_id in current_assignments:
            src_dev = current_assignments[dep_id]
            if src_dev != device.device_id:
                data_size_gb = _estimate_transfer_size_global(dep_id, component.component_id, transformer)
                ttime = network.calculate_transfer_time(src_dev, device.device_id, data_size_gb)
                total_comm_cost += ttime
    return total_comm_cost

def _get_dependencies_global(component_id: str, transformer: Transformer) -> Set[str]:
    """
    Determine the global dependencies for a given component.
    
    For example, the 'projection' block depends on all attention heads,
    and the 'ffn' block depends on the 'projection' block.
    
    Args:
        component_id: The ID of the component.
        transformer: The transformer model.
        
    Returns:
        A set of component IDs that the given component depends on.
    """
    deps = set()
    if component_id == "projection":
        # Projection depends on all attention heads.
        deps.update(h.component_id for h in transformer.attention_heads)
    elif component_id == "ffn":
        deps.add("projection")
    return deps

def _estimate_transfer_size_global(source_id: str, target_id: str, transformer: Transformer) -> float:
    """
    Estimate the size of data to be transferred (in GB) between two components.
    
    The function uses the transformer's current sequence length and configuration parameters.
    For example, transferring outputs from an attention head to the projection layer.
    
    Args:
        source_id: ID of the source component.
        target_id: ID of the target component.
        transformer: Transformer model (to access current sequence length, head_dim, etc.).
        
    Returns:
        Estimated transfer size in GB as a float.
    """
    if source_id.startswith("head_") and target_id == "projection":
        return (transformer.current_sequence_length *
                transformer.config.head_dim *
                transformer.config.precision_bytes) / (1024**3)
    elif source_id == "projection" and target_id == "ffn":
        return (transformer.current_sequence_length *
                transformer.config.embedding_dim *
                transformer.config.precision_bytes) / (1024**3)
    return 0.0

class ResourceAwareDistributor:
    """
    Implements the resource-aware distribution algorithm for LLM block assignment.

    This class implements the myopic, token-by-token placement and migration procedure
    described in Section IV of the paper "Large Language Model Partitioning for Low-Latency
    Inference at the Edge". In this approach:
      - Each Transformer block (e.g., an attention head and its K/V cache, projection, and FFN)
        is assigned to a device by minimizing a score that combines memory, compute, and communication costs.
      - The scoring function (see ScoringFunction) can be computed as the maximum of the three ratios
        or as a weighted sum.
      - Ephemeral components (those that can be freed every step) are deallocated at the start of each
        generation step, while non-ephemeral components (e.g., attention heads with caches) persist.
      - If a block's optimal assignment changes from the previous step, a migration cost is incurred.
      - The algorithm includes mechanisms to resolve resource overloads and backtrack if constraints are violated.
    
    This method is designed to adapt to the dynamic nature of autoregressive LLM inference, where the
    key-value caches grow with each generated token.
    """
    def __init__(
        self,
        transformer: Transformer,
        network: Network,
        devices: Dict[str, Device],
        logger: Optional[SimulationLogger] = None,
        use_weighted_sum: bool = False,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
        use_partial_latency_check: bool = False
    ):
        """
        Initialize the ResourceAwareDistributor.
        
        Args:
            transformer: The Transformer model to be partitioned.
            network: The network object used to compute transfer times.
            devices: Dictionary of available devices.
            logger: Optional logger for simulation events.
            use_weighted_sum: If True, use the weighted sum approach in the scoring function.
            alpha, beta, gamma: Weights for compute, memory, and communication ratios in weighted sum mode.
            use_partial_latency_check: If True, perform a mini-latency simulation to further refine device selection.
        """
        self.transformer = transformer
        self.network = network
        self.devices = devices
        self.logger = logger

        self.use_partial_latency_check = use_partial_latency_check

        # Initialize the scoring function with the given mode and weights.
        self.scoring = ScoringFunction(
            use_weighted_sum=use_weighted_sum,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )

    def compute_assignment(
        self,
        generation_step: int,
        previous_assignments: Optional[Dict[str, str]] = None,
        previous_cache: Optional[Dict[str, str]] = None
    ) -> AssignmentResult:
        """
        Compute a new assignment for Transformer blocks at the current generation step.
        
        This method implements the core of the resource-aware distribution algorithm.
        It follows these high-level steps:
          1) Deallocate ephemeral components from the previous step.
          2) Start with previous assignments if available.
          3) Sort Transformer blocks by descending resource demand.
          4) For each block, select the best device based on the scoring function.
          5) Attempt to allocate the block (and its cache, if applicable) on the chosen device.
          6) Validate the final assignment against all constraints.
          7) Compute the overall latency and communication cost of the assignment.
        
        Returns:
            An AssignmentResult object encapsulating the block-to-device mapping,
            estimated latency, resource usage, and feasibility status.
        """
        if self.logger:
            self.logger.log_event(
                "resource_aware_compute",
                f"[ResourceAwareDistributor] compute_assignment called, step={generation_step}",
                level=LogLevel.DEBUG
            )

        # 1) Deallocate ephemeral components from previous step.
        self._reset_device_states_for_step()

        # 2) Initialize assignments from previous step if available.
        assignments = {} if not previous_assignments else dict(previous_assignments)
        cache_assignments = {} if not previous_cache else dict(previous_cache)

        # 3) Sort components by their resource demand (largest first).
        try:
            components = self._sort_by_resource_demand()
            if not components:
                if self.logger:
                    self.logger.log_error(
                        "resource_aware_no_components",
                        "No components found in the transformer to assign."
                    )
                return AssignmentResult(
                    component_assignments={},
                    cache_assignments={},
                    estimated_latency=float('inf'),
                    resource_usage=self._get_resource_usage(),
                    is_feasible=False
                )
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "resource_aware_sort_fail",
                    f"Exception sorting components by demand: {str(ex)}"
                )
            return AssignmentResult(
                component_assignments={},
                cache_assignments={},
                estimated_latency=float('inf'),
                resource_usage=self._get_resource_usage(),
                is_feasible=False,
                error=f"Sort error: {str(ex)}"
            )

        # 4) Iterate over sorted components and assign each one.
        try:
            for component in components:
                comp_id = component.component_id
                mem_req = component.compute_memory_requirements(self.transformer.current_sequence_length)
                flops_req = component.compute_flops(self.transformer.current_sequence_length)
                cache_req = 0.0
                if hasattr(component, 'compute_cache_memory'):
                    cache_req = component.compute_cache_memory(generation_step)

                # Mark component as ephemeral unless it is an attention head or a cache block.
                ephemeral = not (comp_id.startswith("head_") or comp_id.endswith("_cache"))

                best_device = self._find_best_device(
                    component, assignments, cache_assignments, generation_step
                )
                if not best_device:
                    # No feasible device was found for this component.
                    if self.logger:
                        self.logger.log_error(
                            "resource_aware_no_device",
                            f"No feasible device for {comp_id}"
                        )
                    return AssignmentResult(
                        component_assignments=assignments,
                        cache_assignments=cache_assignments,
                        estimated_latency=float('inf'),
                        resource_usage=self._get_resource_usage(),
                        is_feasible=False,
                        error=f"No device for {comp_id}"
                    )

                # Attempt to allocate main resources and cache (if needed).
                ok_main = best_device.allocate_resources(comp_id, mem_req, flops_req, ephemeral=ephemeral)
                ok_cache = True
                if cache_req > 0 and ok_main:
                    ok_cache = best_device.allocate_resources(f"{comp_id}_cache", cache_req, 0.0, ephemeral=ephemeral)

                if not (ok_main and ok_cache):
                    # Allocation failed; revert partial allocations.
                    if ok_main:
                        best_device.deallocate_resources(comp_id, force=True)
                    if ok_cache:
                        best_device.deallocate_resources(f"{comp_id}_cache", force=True)

                    if self.logger:
                        self.logger.log_error(
                            "resource_aware_alloc_fail",
                            f"Could not allocate {comp_id} on {best_device.device_id}"
                        )
                    return AssignmentResult(
                        component_assignments=assignments,
                        cache_assignments=cache_assignments,
                        estimated_latency=float('inf'),
                        resource_usage=self._get_resource_usage(),
                        is_feasible=False,
                        error=f"Alloc fail for {comp_id}"
                    )

                # Record the assignment.
                assignments[comp_id] = best_device.device_id
                if cache_req > 0:
                    cache_assignments[comp_id] = best_device.device_id

        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "resource_aware_assign_loop_fail",
                    f"Error in assignment loop: {str(ex)}"
                )
            return AssignmentResult(
                component_assignments=assignments,
                cache_assignments=cache_assignments,
                estimated_latency=float('inf'),
                resource_usage=self._get_resource_usage(),
                is_feasible=False,
                error=f"Assign loop exception: {str(ex)}"
            )

        # 5) Validate the final assignment to ensure all resource constraints are met.
        try:
            feasible = validate_assignment(
                assignments,
                cache_assignments,
                self.transformer,
                self.devices,
                self.network,
                generation_step
            )
            if not feasible:
                if self.logger:
                    self.logger.log_error(
                        "resource_aware_infeasible",
                        "Final assignment not feasible"
                    )
                return AssignmentResult(
                    component_assignments=assignments,
                    cache_assignments=cache_assignments,
                    estimated_latency=float('inf'),
                    resource_usage=self._get_resource_usage(),
                    is_feasible=False
                )
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "resource_aware_validation_fail",
                    f"Validate assignment error: {str(ex)}"
                )
            return AssignmentResult(
                component_assignments=assignments,
                cache_assignments=cache_assignments,
                estimated_latency=float('inf'),
                resource_usage=self._get_resource_usage(),
                is_feasible=False,
                error=f"Validation exception: {str(ex)}"
            )

        # 6) Compute the estimated total latency and communication time for the assignment.
        try:
            latency, comm_time, data_gb = self._compute_latency_and_comm(assignments, cache_assignments, generation_step)
            usage = self._get_resource_usage()
            return AssignmentResult(
                component_assignments=assignments,
                cache_assignments=cache_assignments,
                estimated_latency=latency,
                resource_usage=usage,
                is_feasible=True,
                migrations=[],
                communication_time=comm_time,
                data_transferred_gb=data_gb
            )
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "resource_aware_latency_comm_fail",
                    f"Exception in latency/comm calc: {str(ex)}"
                )
            return AssignmentResult(
                component_assignments=assignments,
                cache_assignments=cache_assignments,
                estimated_latency=float('inf'),
                resource_usage=self._get_resource_usage(),
                is_feasible=False,
                error=f"Latency/comm exception: {str(ex)}"
            )

    # ------------------------------------------------------------------------
    # Implementation details for helper methods.
    # ------------------------------------------------------------------------

    def _reset_device_states_for_step(self) -> None:
        """
        Deallocate ephemeral components from devices at the beginning of each generation step.
        
        Ephemeral components (those that can be reallocated at every step) are deallocated,
        while non-ephemeral ones (e.g., attention heads with persistent caches) remain allocated.
        """
        for device in self.devices.values():
            comps_to_remove = []
            for comp_id, info in device.assigned_components.items():
                if info.get("ephemeral", True) is True:
                    comps_to_remove.append(comp_id)
            caches_to_remove = []
            for comp_id, cinfo in device.cache_assignments.items():
                if cinfo.get("ephemeral", True) is True:
                    caches_to_remove.append(comp_id)

            for cid in comps_to_remove:
                device.deallocate_resources(cid, force=True)
            for cid in caches_to_remove:
                device.deallocate_resources(cid, force=True)

    def _compute_latency_and_comm(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> Tuple[float, float, float]:
        """
        Compute the total estimated latency and communication cost for the current assignment.
        
        Uses a concurrency-based latency model (e.g., compute_3phase_latency) and adds up
        communication delays for transfers required between blocks located on different devices.
        
        Returns:
            A tuple (total_latency, total_comm_time, total_data_transferred_gb).
        """
        total_latency = compute_3phase_latency(
            self.transformer,
            self.devices,
            self.network,
            assignments,
            generation_step,
            concurrency_mode="sum"  # or "max"/"hybrid" if desired
        )
        comm_time, data_gb = self._compute_comm_stats_separately(assignments, cache_assignments, generation_step)

        if self.logger:
            self.logger.log_event(
                "resource_aware_latency_comm",
                f"ResourceAware => lat={total_latency:.4f}, comm_time={comm_time:.4f}, data_gb={data_gb:.4f}",
                level=LogLevel.DEBUG
            )
        return (total_latency, comm_time, data_gb)

    def _compute_comm_stats_separately(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> Tuple[float, float]:
        """
        Compute separate communication statistics: total communication time and total data transferred (in GB).
        
        For each component, if its dependencies are assigned to a different device,
        the method sums the transfer time based on estimated transfer size and network bandwidth.
        
        Returns:
            A tuple (total_comm_time, total_data_transferred_gb).
        """
        total_comm_time = 0.0
        total_data_gb = 0.0
        try:
            for comp_id, dev_id in assignments.items():
                deps = self.__get_dependencies(comp_id)
                for dep_id in deps:
                    if dep_id in assignments:
                        src_dev = assignments[dep_id]
                        if src_dev != dev_id:
                            data_size_gb = self.__estimate_transfer_size(dep_id, comp_id)
                            total_data_gb += data_size_gb
                            ttime = self.network.calculate_transfer_time(src_dev, dev_id, data_size_gb)
                            total_comm_time += ttime

                            if self.logger and data_size_gb > 0:
                                self.logger.log_event(
                                    "resource_aware_comm",
                                    f"transfer {data_size_gb:.6f}GB from {src_dev}->{dev_id} for dep={dep_id}->{comp_id}, "
                                    f"time={ttime:.4f}",
                                    level=LogLevel.DEBUG
                                )
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "resource_aware_comm_stats_fail",
                    f"Exception: {str(ex)}"
                )
        return (total_comm_time, total_data_gb)

    def _sort_by_resource_demand(self) -> List[TransformerComponent]:
        """
        Sort Transformer components by their resource demand (memory and compute), in descending order.
        
        This sorting ensures that components with the highest demand are assigned first,
        which is critical for avoiding overload and for effective resource balancing.
        
        Returns:
            A list of TransformerComponent objects sorted by demand.
        """
        try:
            comps = self.transformer.get_all_components()
            if not comps:
                return []
            demands = []
            max_mem = max(d.memory.capacity for d in self.devices.values())
            max_cmp = max(d.compute.capacity for d in self.devices.values() or [1.0])

            for c in comps:
                mem = c.compute_memory_requirements(self.transformer.current_sequence_length)
                flp = c.compute_flops(self.transformer.current_sequence_length)
                score = (mem / max_mem) + (flp / max_cmp)
                demands.append((score, c))

            demands.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in demands]
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "resource_aware_sort_exception",
                    f"Exception sorting resource demand: {str(ex)}"
                )
            return []

    def _find_best_device(
        self,
        component: TransformerComponent,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> Optional[Device]:
        """
        Find the best device for placing a given Transformer component.
        
        For each feasible device (i.e., one that can accommodate the component's memory and compute requirements),
        the method computes a score using the scoring function defined earlier. Optionally, if use_partial_latency_check
        is enabled, the method performs a mini-latency simulation by temporarily allocating the component to the device,
        computing a concurrency-based latency estimate, and then reverting the allocation.
        
        The device with the lowest (best) score is selected.
        
        Args:
            component: The Transformer component to be assigned.
            assignments: Current component assignments mapping.
            cache_assignments: Current cache assignments mapping.
            generation_step: Current generation step.
        
        Returns:
            The Device object with the best (lowest) computed score, or None if no feasible device is found.
        """
        best_score = float('inf')
        best_dev = None

        mem_req = component.compute_memory_requirements(self.transformer.current_sequence_length)
        flops_req = component.compute_flops(self.transformer.current_sequence_length)
        if hasattr(component, 'compute_cache_memory'):
            mem_req += component.compute_cache_memory(generation_step)

        if self.logger:
            self.logger.log_event(
                "resource_aware_find_device",
                f"Finding device for {component.component_id}, mem={mem_req:.4f}, flops={flops_req:.4f}",
                level=LogLevel.DEBUG
            )

        for dev in self.devices.values():
            # Quick feasibility check based on available resources.
            if not dev.can_accommodate(mem_req, flops_req):
                if self.logger:
                    self.logger.log_event(
                        "resource_aware_device_skip",
                        f"Device={dev.device_id} can't fit {component.component_id}",
                        level=LogLevel.DEBUG
                    )
                continue

            # 1) Compute score using the resource-aware scoring function.
            ratio_score = self.scoring.compute(
                component, dev, self.network, self.transformer,
                assignments, cache_assignments, generation_step
            )
            if ratio_score == float('inf'):
                # Skip this device if it is deemed infeasible by policy.
                continue

            if not self.use_partial_latency_check:
                # Use the computed ratio score directly.
                final_score = ratio_score
            else:
                # 2) If enabled, perform a partial latency simulation:
                # Temporarily allocate the component to the device to compute a concurrency-based latency estimate.
                ephemeral = not (component.component_id.startswith("head_") or component.component_id.endswith("_cache"))
                ok_main = dev.allocate_resources(component.component_id, mem_req, flops_req, ephemeral=ephemeral)
                if not ok_main:
                    continue

                tmp_assignments = dict(assignments)
                tmp_assignments[component.component_id] = dev.device_id

                # Compute approximate latency if this assignment is made.
                lat_if_assigned, _, _ = self._compute_latency_and_comm(
                    tmp_assignments, cache_assignments, generation_step
                )

                # Revert temporary allocation.
                dev.deallocate_resources(component.component_id, force=True)

                final_score = lat_if_assigned

            if final_score < best_score:
                best_score = final_score
                best_dev = dev

        if self.logger and best_dev:
            self.logger.log_event(
                "resource_aware_best_device",
                f"Best device for {component.component_id} => {best_dev.device_id}, score={best_score:.4f}",
                level=LogLevel.DEBUG
            )
        return best_dev

    # ------------------------------------------------------------------------
    # Helper methods for dependency and transfer size estimation.
    # ------------------------------------------------------------------------

    def _get_resource_usage(self) -> Dict[str, Dict[str, float]]:
        """
        Collect the current resource usage statistics for all devices.
        
        Returns:
            A dictionary mapping device IDs to their resource usage details (memory used, compute used, etc.).
        """
        usage = {}
        for dev_id, dev in self.devices.items():
            usage[dev_id] = {
                "memory_used": dev.memory.used,
                "memory_capacity": dev.memory.capacity,
                "compute_used": dev.compute.used,
                "compute_capacity": dev.compute.capacity,
                "compute_utilization": (
                    dev.compute.used / dev.compute.capacity if dev.compute.capacity > 0 else 1.0
                ),
                "memory_utilization": (
                    dev.memory.used / dev.memory.capacity if dev.memory.capacity > 0 else 1.0
                )
            }
        return usage

    def __get_dependencies(self, comp_id: str) -> List[str]:
        """
        Get the local dependency list for a given component.
        
        For example, the 'projection' block depends on all attention heads,
        and the 'ffn' block depends on the 'projection' block.
        
        Args:
            comp_id: The component ID.
        
        Returns:
            A list of component IDs on which comp_id depends.
        """
        deps = []
        if comp_id == "projection":
            if hasattr(self.transformer, 'attention_heads'):
                for head in self.transformer.attention_heads:
                    deps.append(head.component_id)
        elif comp_id == "ffn":
            deps.append("projection")
        return deps

    def __estimate_transfer_size(self, source_id: str, target_id: str) -> float:
        """
        Estimate the data transfer size (in GB) between two components.
        
        The calculation is based on the transformer’s current sequence length and configuration.
        It applies specific formulas for transferring data from attention heads to projection,
        or from projection to the feed-forward network.
        
        Args:
            source_id: ID of the source component.
            target_id: ID of the target component.
        
        Returns:
            Estimated data transfer size in GB.
        """
        if source_id.startswith("head_") and target_id == "projection":
            return (
                self.transformer.current_sequence_length *
                self.transformer.config.head_dim *
                self.transformer.config.precision_bytes
            ) / (1024**3)
        elif source_id == "projection" and target_id == "ffn":
            return (
                self.transformer.current_sequence_length *
                self.transformer.config.embedding_dim *
                self.transformer.config.precision_bytes
            ) / (1024**3)
        return 0.0
