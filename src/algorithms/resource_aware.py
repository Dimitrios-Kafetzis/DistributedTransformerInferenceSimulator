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
# File:    src/algorithms/resource_aware.py
# Description:
#   Provides the resource-aware distribution algorithm for transformer
#   inference, incorporating memory, compute, and communication
#   constraints to optimize component placements.
#
# ---------------------------------------------------------------------------

"""
Contains the main ResourceAwareDistributor class and related data structures,
implementing a multi-dimensional scoring function and constraints to achieve
optimized distributed assignments for transformer inference.
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
    """Results from the distribution algorithm."""
    component_assignments: Dict[str, str]  # component_id -> device_id
    cache_assignments: Dict[str, str]      # head_id -> device_id
    estimated_latency: float
    resource_usage: Dict[str, Dict[str, float]]
    is_feasible: bool
    error: Optional[str] = None  # In case we want to store a message
    migrations: Optional[List[tuple]] = None
    # Fields to store per-step communication overhead/time
    communication_time: float = 0.0
    data_transferred_gb: float = 0.0


@dataclass
class ScoringFunction:
    """Implementation of the scoring function S(i,j,t) from the paper"""

    @staticmethod
    def compute(
        component: TransformerComponent,
        device: Device,
        network: Network,
        transformer: Transformer,
        current_assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> float:
        """
        Compute the scoring function S(i,j,t) as defined in Section IV.
        Returns infinity for infeasible assignments or source constraints (depending on policy).
        """
        # If device is source and component not "input"/"position", treat as infeasible (example policy).
        if device.is_source and component.component_id not in ["input", "position"]:
            return float('inf')

        # Compute base scoring function from memory, compute, communication
        compute_ratio = _compute_ratio(component, device, transformer)
        memory_ratio = _memory_ratio(component, device, transformer, generation_step)
        comm_ratio = _communication_ratio(
            component, device, network, transformer,
            current_assignments, cache_assignments
        )

        return max(compute_ratio, memory_ratio, comm_ratio)


def _compute_ratio(
    component: TransformerComponent,
    device: Device,
    transformer: Transformer
) -> float:
    """Calculate computation ratio (eq. 7-8 from the paper)."""
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
    """Calculate memory ratio including cache requirements."""
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
    Calculate communication ratio based on dependencies and device bandwidth.
    Summation of the transfer_time for each dependency that’s on a different device.
    """
    total_comm_cost = 0.0
    deps = _get_dependencies(component.component_id, transformer)

    for dep_id in deps:
        if dep_id in current_assignments:
            source_dev = current_assignments[dep_id]
            if source_dev != device.device_id:
                data_size_gb = _estimate_transfer_size(dep_id, component.component_id, transformer)
                transfer_time = network.calculate_transfer_time(source_dev, device.device_id, data_size_gb)
                total_comm_cost += transfer_time

    return total_comm_cost


def _get_dependencies(
    component_id: str,
    transformer: Transformer
) -> Set[str]:
    """Get dependencies for a component: e.g., projection depends on attention heads."""
    dependencies = set()
    if component_id == "projection":
        dependencies.update(head.component_id for head in transformer.attention_heads)
    elif component_id == "ffn":
        dependencies.add("projection")
    return dependencies


def _estimate_transfer_size(
    source_id: str,
    target_id: str,
    transformer: Transformer
) -> float:
    """Estimate size of data transfer between components in GB."""
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
    Implementation of the resource-aware distribution algorithm
    from Section IV of the paper.
    """

    def __init__(
        self,
        transformer: Transformer,
        network: Network,
        devices: Dict[str, Device],
        logger: Optional[SimulationLogger] = None
    ):
        """
        :param transformer: the Transformer model to be distributed
        :param network: the network topology
        :param devices: dict of device_id -> Device objects
        :param logger: optional SimulationLogger for debug/error messages
        """
        self.transformer = transformer
        self.network = network
        self.devices = devices
        self.scoring = ScoringFunction()
        self.logger = logger  # for structured logging

    def compute_assignment(
        self,
        generation_step: int,
        previous_assignments: Optional[Dict[str, str]] = None,
        previous_cache: Optional[Dict[str, str]] = None
    ) -> AssignmentResult:
        """Compute optimal component assignments for current generation step."""

        if self.logger:
            self.logger.log_event(
                "resource_aware_compute",
                f"[ResourceAwareDistributor] compute_assignment called, step={generation_step}",
                level=LogLevel.DEBUG
            )

        # Copy or create fresh assignment dicts
        assignments = {} if previous_assignments is None else previous_assignments.copy()
        cache_assignments = {} if previous_cache is None else previous_cache.copy()

        # Reset device states (for a fresh assignment pass)
        try:
            for device in self.devices.values():
                for comp_id in list(device.assigned_components.keys()):
                    device.deallocate_resources(comp_id)
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "resource_aware_deallocate_fail",
                    f"Error deallocating device resources before assignment pass: {str(ex)}"
                )
            traceback.print_exc()

        # Sort components by resource demand
        try:
            components = self._sort_by_resource_demand()
            if not components:
                if self.logger:
                    self.logger.log_error(
                        "resource_aware_no_components",
                        f"No components found in the transformer to assign."
                    )
                return AssignmentResult(
                    component_assignments={},
                    cache_assignments={},
                    estimated_latency=float('inf'),
                    resource_usage=self._get_resource_usage(),
                    is_feasible=False,
                    error="No components",
                    migrations=[],
                    communication_time=0.0,
                    data_transferred_gb=0.0
                )
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "resource_aware_sort_fail",
                    f"Exception sorting components by resource demand: {str(ex)}"
                )
            traceback.print_exc()
            # fallback: treat as no components
            return AssignmentResult(
                component_assignments={},
                cache_assignments={},
                estimated_latency=float('inf'),
                resource_usage=self._get_resource_usage(),
                is_feasible=False,
                error=f"Exception in sorting: {str(ex)}",
                migrations=[],
                communication_time=0.0,
                data_transferred_gb=0.0
            )

        # Try to assign each component
        try:
            for component in components:
                comp_id = component.component_id
                memory_req = component.compute_memory_requirements(self.transformer.current_sequence_length)
                flops_req = component.compute_flops(self.transformer.current_sequence_length)

                cache_req = 0.0
                if hasattr(component, 'compute_cache_memory'):
                    cache_req = component.compute_cache_memory(generation_step)

                best_device = self._find_best_device(
                    component,
                    assignments,
                    cache_assignments,
                    generation_step
                )
                if best_device is None:
                    # no feasible device
                    if self.logger:
                        self.logger.log_error(
                            "resource_aware_no_device",
                            f"No suitable device found for component={comp_id}"
                        )
                    return AssignmentResult(
                        component_assignments=assignments,
                        cache_assignments=cache_assignments,
                        estimated_latency=float('inf'),
                        resource_usage=self._get_resource_usage(),
                        is_feasible=False,
                        error=f"No suitable device for {comp_id}",
                        migrations=[],
                        communication_time=0.0,
                        data_transferred_gb=0.0
                    )

                # Allocate resources on best_device
                ok_main = best_device.allocate_resources(comp_id, memory_req, flops_req)
                ok_cache = True
                if cache_req > 0.0 and ok_main:
                    ok_cache = best_device.allocate_resources(f"{comp_id}_cache", cache_req, 0.0)

                if not (ok_main and ok_cache):
                    if ok_main:
                        best_device.deallocate_resources(comp_id)
                    if ok_cache:
                        best_device.deallocate_resources(f"{comp_id}_cache")
                    if self.logger:
                        self.logger.log_error(
                            "resource_aware_allocation_fail",
                            f"Failed to allocate resources for component={comp_id} on device={best_device.device_id}"
                        )
                    return AssignmentResult(
                        component_assignments=assignments,
                        cache_assignments=cache_assignments,
                        estimated_latency=float('inf'),
                        resource_usage=self._get_resource_usage(),
                        is_feasible=False,
                        error=f"Failed allocating {comp_id}",
                        migrations=[],
                        communication_time=0.0,
                        data_transferred_gb=0.0
                    )

                # success => record assignment
                assignments[comp_id] = best_device.device_id
                if cache_req > 0:
                    cache_assignments[comp_id] = best_device.device_id

        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "resource_aware_assign_loop_fail",
                    f"Error during assignment loop: {str(ex)}"
                )
            traceback.print_exc()
            return AssignmentResult(
                component_assignments=assignments,
                cache_assignments=cache_assignments,
                estimated_latency=float('inf'),
                resource_usage=self._get_resource_usage(),
                is_feasible=False,
                error=f"Exception in assignment loop: {str(ex)}",
                migrations=[],
                communication_time=0.0,
                data_transferred_gb=0.0
            )

        # Now validate
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
                        f"Assignment is not feasible after final validation."
                    )
                return AssignmentResult(
                    component_assignments=assignments,
                    cache_assignments=cache_assignments,
                    estimated_latency=float('inf'),
                    resource_usage=self._get_resource_usage(),
                    is_feasible=False,
                    migrations=[],
                    communication_time=0.0,
                    data_transferred_gb=0.0
                )
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "resource_aware_validation_fail",
                    f"Exception in validate_assignment: {str(ex)}"
                )
            traceback.print_exc()
            return AssignmentResult(
                component_assignments=assignments,
                cache_assignments=cache_assignments,
                estimated_latency=float('inf'),
                resource_usage=self._get_resource_usage(),
                is_feasible=False,
                error=f"Exception in validation: {str(ex)}",
                migrations=[],
                communication_time=0.0,
                data_transferred_gb=0.0
            )

        # If feasible => compute latency + communication
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
                    f"Exception computing latency/communication: {str(ex)}"
                )
            traceback.print_exc()
            usage = self._get_resource_usage()
            return AssignmentResult(
                component_assignments=assignments,
                cache_assignments=cache_assignments,
                estimated_latency=float('inf'),
                resource_usage=usage,
                is_feasible=False,
                error=f"Exception computing latency/comm: {str(ex)}",
                migrations=[],
                communication_time=0.0,
                data_transferred_gb=0.0
            )

    def _compute_comm_stats_separately(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> Tuple[float, float]:
        """
        Returns (comm_time, data_transferred_gb) by summing
        explicit data transfers from assignment => using device bandwidth,
        ignoring concurrency or pipeline overlap.
        """
        total_comm_time = 0.0
        total_data_gb = 0.0

        try:
            for comp_id, dev_id in assignments.items():
                component = self.transformer.get_component(comp_id)
                deps = self._get_dependencies(comp_id)

                for dep_id in deps:
                    if dep_id in assignments:
                        src_dev = assignments[dep_id]
                        if src_dev != dev_id:
                            data_size_gb = self._estimate_transfer_size(dep_id, comp_id)
                            total_data_gb += data_size_gb
                            ttime = self.network.calculate_transfer_time(src_dev, dev_id, data_size_gb)
                            total_comm_time += ttime
                            if self.logger and data_size_gb > 0.0:
                                self.logger.log_event(
                                    "resource_aware_comm",
                                    f"Transferring {data_size_gb:.6f}GB from {src_dev} -> {dev_id} for dep={dep_id}->{comp_id}, time={ttime:.6f}s",
                                    level=LogLevel.DEBUG
                                )
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "resource_aware_comm_stats_fail",
                    f"Error in _compute_comm_stats_separately: {str(ex)}"
                )
            traceback.print_exc()

        return (total_comm_time, total_data_gb)

    def _compute_latency_and_comm(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> Tuple[float, float, float]:
        """
        We'll compute concurrency-based total latency from compute_3phase_latency(),
        then separately compute comm_time + data_gb in a second pass.
        """
        try:
            total_latency = compute_3phase_latency(
                self.transformer,
                self.devices,
                self.network,
                assignments,
                generation_step,
                concurrency_mode="sum"  # or "max" or "hybrid"
            )
            comm_time, data_gb = self._compute_comm_stats_separately(
                assignments, cache_assignments, generation_step
            )
            if self.logger:
                self.logger.log_event(
                    "resource_aware_latency_comm",
                    f"Final concurrency-based latency={total_latency:.4f}, comm_time={comm_time:.4f}, data={data_gb:.4f}GB",
                    level=LogLevel.DEBUG
                )
            return (total_latency, comm_time, data_gb)
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "resource_aware_latency_comm_exception",
                    f"Exception in _compute_latency_and_comm: {str(ex)}"
                )
            traceback.print_exc()
            return (float('inf'), 0.0, 0.0)

    def _sort_by_resource_demand(self) -> List[TransformerComponent]:
        """Sort components by (memory+compute) demand in descending order."""
        try:
            comps = self.transformer.get_all_components()
            if not comps:
                return []
            demands = []
            max_mem = max(dev.memory.capacity for dev in self.devices.values())
            max_cmp = max(dev.compute.capacity for dev in self.devices.values()) or 1.0

            for c in comps:
                mem = c.compute_memory_requirements(self.transformer.current_sequence_length)
                flp = c.compute_flops(self.transformer.current_sequence_length)
                # quick approach: sum normalized
                score = (mem / max_mem) + (flp / max_cmp)
                demands.append((score, c))

            demands.sort(key=lambda x: x[0], reverse=True)
            sorted_comps = [d[1] for d in demands]
            if self.logger:
                self.logger.log_event(
                    "resource_aware_sort",
                    "Sorted components by resource demand (descending)",
                    level=LogLevel.DEBUG,
                    sorted_list=[(c.component_id, sc) for sc, c in demands]
                )
            return sorted_comps
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "resource_aware_sort_exception",
                    f"Exception sorting by resource demand: {str(ex)}"
                )
            traceback.print_exc()
            return []

    def _find_best_device(
        self,
        component: TransformerComponent,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> Optional[Device]:
        """
        Pick the device with the minimal scoring function value (S(i,j,t)).
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
                f"Trying to find best device for component={component.component_id}, mem_req={mem_req:.4f}, flops_req={flops_req:.4f}",
                level=LogLevel.DEBUG
            )

        try:
            for dev in self.devices.values():
                dev_id = dev.device_id
                # quick check if feasible ignoring communication
                if not dev.can_accommodate(mem_req, flops_req):
                    if self.logger:
                        self.logger.log_event(
                            "resource_aware_device_skip",
                            f"Device={dev_id} cannot accommodate comp={component.component_id}, mem_req={mem_req:.4f}, flops_req={flops_req:.4f}",
                            level=LogLevel.DEBUG
                        )
                    continue

                score_val = ScoringFunction.compute(
                    component,
                    dev,
                    self.network,
                    self.transformer,
                    assignments,
                    cache_assignments,
                    generation_step
                )
                if self.logger:
                    self.logger.log_event(
                        "resource_aware_score",
                        f"Device={dev_id}, S(i,j,t)={score_val:.4f}",
                        level=LogLevel.DEBUG
                    )

                if score_val < best_score:
                    best_score = score_val
                    best_dev = dev

            if best_dev and self.logger:
                self.logger.log_event(
                    "resource_aware_best_device",
                    f"Best device for component={component.component_id} is {best_dev.device_id} with score={best_score:.4f}",
                    level=LogLevel.DEBUG
                )
            elif not best_dev and self.logger:
                self.logger.log_event(
                    "resource_aware_no_best",
                    f"No suitable device for component={component.component_id}, mem_req={mem_req:.4f}, flops_req={flops_req:.4f}",
                    level=LogLevel.DEBUG
                )
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "resource_aware_find_device_exception",
                    f"Exception in _find_best_device for comp={component.component_id}: {str(ex)}"
                )
            traceback.print_exc()

        return best_dev

    def _get_resource_usage(self) -> Dict[str, Dict[str, float]]:
        """
        Build a dictionary describing memory/compute usage and utilization on each device.
        """
        usage = {}
        try:
            for dev_id, dev in self.devices.items():
                usage[dev_id] = {
                    'memory_used': dev.memory.used,
                    'memory_capacity': dev.memory.capacity,
                    'compute_used': dev.compute.used,
                    'compute_capacity': dev.compute.capacity,
                    'compute_utilization': (
                        dev.compute.used / dev.compute.capacity
                        if dev.compute.capacity > 0 else 1.0
                    ),
                    'memory_utilization': (
                        dev.memory.used / dev.memory.capacity
                        if dev.memory.capacity > 0 else 1.0
                    )
                }
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "resource_aware_resource_usage_fail",
                    f"Exception in _get_resource_usage: {str(ex)}"
                )
            traceback.print_exc()

        return usage
