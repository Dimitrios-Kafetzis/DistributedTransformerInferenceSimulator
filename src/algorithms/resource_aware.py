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
    error: Optional[str] = None
    migrations: Optional[List[tuple]] = None
    communication_time: float = 0.0
    data_transferred_gb: float = 0.0


@dataclass
class ScoringFunction:
    """
    Implements the scoring function S(i,j,t). 
    We have two modes:
      (1) 'max' mode => returns max(compute_ratio, memory_ratio, comm_ratio)
      (2) 'weighted_sum' mode => alpha * compute_ratio + beta * memory_ratio + gamma * comm_ratio.
    """
    use_weighted_sum: bool = False  # If True, do the weighted sum approach; otherwise do max(...)
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
        Returns infinity if infeasible or if we have a policy that forbids device for certain comps.
        Otherwise:
          - If use_weighted_sum=False => returns max(compute_ratio, memory_ratio, comm_ratio).
          - If use_weighted_sum=True  => returns alpha*compute_ratio + beta*memory_ratio + gamma*comm_ratio.
        """
        # Example policy: skip device if it's a "source" device but comp not input/position
        if device.is_source and component.component_id not in ["input", "position"]:
            return float('inf')

        # Gather the 3 ratio components
        compute_ratio = _compute_ratio(component, device, transformer)
        memory_ratio = _memory_ratio(component, device, transformer, generation_step)
        comm_ratio = _communication_ratio(
            component, device, network, transformer, current_assignments, cache_assignments
        )

        if not self.use_weighted_sum:
            # Old approach
            return max(compute_ratio, memory_ratio, comm_ratio)
        else:
            # Weighted sum approach
            return self.alpha * compute_ratio + self.beta * memory_ratio + self.gamma * comm_ratio


def _compute_ratio(component: TransformerComponent, device: Device, transformer: Transformer) -> float:
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
    deps = set()
    if component_id == "projection":
        # depends on all attention heads
        deps.update(h.component_id for h in transformer.attention_heads)
    elif component_id == "ffn":
        deps.add("projection")
    return deps


def _estimate_transfer_size_global(source_id: str, target_id: str, transformer: Transformer) -> float:
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
    Resource-aware approach from Section IV. 
    We free ephemeral components each step, preserving head_* or *_cache across steps.
    
    This class can do two new modifications:
      1) Weighted sum vs. max(...) in the scoring function (ScoringFunction has a flag).
      2) Optional mini-latency check in `_find_best_device`, enabled by `use_partial_latency_check`.
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
        :param use_weighted_sum: if True, do weighted sum in scoring; otherwise do max.
        :param alpha, beta, gamma: weights if we do weighted sum
        :param use_partial_latency_check: if True, we do a mini-latency simulation in `_find_best_device`.
        """
        self.transformer = transformer
        self.network = network
        self.devices = devices
        self.logger = logger

        self.use_partial_latency_check = use_partial_latency_check

        # Build the scoring function object
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

        if self.logger:
            self.logger.log_event(
                "resource_aware_compute",
                f"[ResourceAwareDistributor] compute_assignment called, step={generation_step}",
                level=LogLevel.DEBUG
            )

        # 1) Deallocate ephemeral from previous step
        self._reset_device_states_for_step()

        # 2) Start from previous assignments or empty
        assignments = {} if not previous_assignments else dict(previous_assignments)
        cache_assignments = {} if not previous_cache else dict(previous_cache)

        # 3) Sort components by resource demand (descending)
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

        # 4) Assign each component in order
        try:
            for component in components:
                comp_id = component.component_id
                mem_req = component.compute_memory_requirements(self.transformer.current_sequence_length)
                flops_req = component.compute_flops(self.transformer.current_sequence_length)
                cache_req = 0.0
                if hasattr(component, 'compute_cache_memory'):
                    cache_req = component.compute_cache_memory(generation_step)

                # ephemeral unless it's head_* or *_cache
                ephemeral = not (comp_id.startswith("head_") or comp_id.endswith("_cache"))

                best_device = self._find_best_device(
                    component, assignments, cache_assignments, generation_step
                )
                if not best_device:
                    # No feasible device found
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

                # Attempt allocation
                ok_main = best_device.allocate_resources(comp_id, mem_req, flops_req, ephemeral=ephemeral)
                ok_cache = True
                if cache_req > 0 and ok_main:
                    ok_cache = best_device.allocate_resources(f"{comp_id}_cache", cache_req, 0.0, ephemeral=ephemeral)

                if not (ok_main and ok_cache):
                    # revert partial success
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

                # Record
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

        # 5) Validate final assignment
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

        # 6) If feasible => compute concurrency-based latency + comm
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
    # Implementation details
    # ------------------------------------------------------------------------

    def _reset_device_states_for_step(self) -> None:
        """
        Only deallocate ephemeral comps (projection, ffn, etc.). 
        Keep non-ephemeral (heads, caches).
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
        # concurrency-based total latency
        total_latency = compute_3phase_latency(
            self.transformer,
            self.devices,
            self.network,
            assignments,
            generation_step,
            concurrency_mode="sum"  # or "max"/"hybrid" if desired
        )
        # Then sum up the data xfers
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
        Sort components by (memory + compute) usage, descending.
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
        For each feasible device, compute:
          - ratio-based or weighted-sum score from ScoringFunction, 
          - optionally a partial concurrency-based estimate,
        and pick the device with the best (lowest) combined measure.
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
            # Quick feasibility
            if not dev.can_accommodate(mem_req, flops_req):
                if self.logger:
                    self.logger.log_event(
                        "resource_aware_device_skip",
                        f"Device={dev.device_id} can't fit {component.component_id}",
                        level=LogLevel.DEBUG
                    )
                continue

            # 1) Ratio-based or weighted-sum scoring
            ratio_score = self.scoring.compute(
                component, dev, self.network, self.transformer,
                assignments, cache_assignments, generation_step
            )
            if ratio_score == float('inf'):
                # not feasible or policy skip
                continue

            if not self.use_partial_latency_check:
                # simpler approach: just use the ratio_score
                final_score = ratio_score
            else:
                # 2) Do a local concurrency-based partial check
                # Temporarily allocate this component to dev, measure concurrency latency
                # Then revert it.
                ephemeral = not (component.component_id.startswith("head_") or component.component_id.endswith("_cache"))
                ok_main = dev.allocate_resources(component.component_id, mem_req, flops_req, ephemeral=ephemeral)
                # no separate cache check here for partial approach; you can do it if needed
                if not ok_main:
                    # revert any partial
                    continue

                # Record the assignment in a temporary dict
                tmp_assignments = dict(assignments)
                tmp_assignments[component.component_id] = dev.device_id

                # Compute concurrency-based latency in a partial scenario
                # We do not finalize or validate the entire assignment, 
                # just measure approximate new latency with concurrency.
                #   Possibly we do a partial 'compute_3phase_latency' with concurrency_mode, 
                #   or do the entire existing + this one component assigned.
                # For simplicity, let's do a partial approach:
                lat_if_assigned, _, _ = self._compute_latency_and_comm(
                    tmp_assignments, cache_assignments, generation_step
                )

                # Revert
                dev.deallocate_resources(component.component_id, force=True)

                # Combine ratio_score with the partial concurrency measure (or use just concurrency)
                # Here we take a simple approach: final_score = lat_if_assigned
                # Or we can do some combination if desired.
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
    # Helper methods for dependencies and data size
    # ------------------------------------------------------------------------
    def _get_resource_usage(self) -> Dict[str, Dict[str, float]]:
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
        e.g. 'projection' depends on heads, 'ffn' depends on 'projection'.
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
        Same logic: head->projection or projection->ffn
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
