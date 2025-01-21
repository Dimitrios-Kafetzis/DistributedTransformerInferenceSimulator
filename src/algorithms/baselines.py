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
# File: src/algorithms/baselines.py
# Description:
#   Implements baseline distribution algorithms such as greedy, round-robin,
#   static, and dynamic migration strategies for comparison with the
#   resource-aware approach.
#
# ---------------------------------------------------------------------------

"""
Defines baseline or simpler distribution algorithms, including GreedyDistributor,
RoundRobinDistributor, StaticDistributor, and DynamicMigrationDistributor,
to compare against the resource-aware approach.

Now updated so that only ephemeral blocks are freed each step, while K/V caches
and attention heads remain allocated across steps (ephemeral=False).
"""

import traceback
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from src.utils.logging import SimulationLogger, LogLevel  # adapt import path if needed
from ..core import Device, Network, Transformer, AttentionHead, TransformerComponent
from .utils import validate_assignment, compute_3phase_latency
from .resource_aware import AssignmentResult


class BaseDistributor(ABC):
    """Base class for all distribution algorithms"""

    def __init__(
        self,
        transformer: Transformer,
        network: Network,
        devices: Dict[str, Device],
        logger: Optional[SimulationLogger] = None
    ):
        self.transformer = transformer
        self.network = network
        self.devices = devices
        self.logger = logger
        
    @abstractmethod
    def compute_assignment(
        self,
        generation_step: int,
        previous_assignments: Optional[Dict[str, str]] = None,
        previous_cache: Optional[Dict[str, str]] = None
    ) -> AssignmentResult:
        pass
        
    def _get_resource_usage(self) -> Dict[str, Dict[str, float]]:
        usage = {}
        for dev_id, device in self.devices.items():
            usage[dev_id] = {
                'memory_used': device.memory.used,
                'memory_capacity': device.memory.capacity,
                'compute_used': device.compute.used,
                'compute_capacity': device.compute.capacity,
                'memory_utilization': (
                    device.memory.used / device.memory.capacity if device.memory.capacity > 0 else 1.0
                ),
                'compute_utilization': (
                    device.compute.used / device.compute.capacity if device.compute.capacity > 0 else 1.0
                ),
            }
        return usage

    def _reset_device_states_for_step(self) -> None:
        """
        Deallocate ephemeral components only. 
        We skip anything marked ephemeral=False (like K/V caches or attention heads).
        """
        for device in self.devices.values():
            comps_to_remove = []
            for comp_id, info in device.assigned_components.items():
                if info.get("ephemeral", True) is True:
                    comps_to_remove.append(comp_id)

            # For cache assignments
            caches_to_remove = []
            for comp_id, cinfo in device.cache_assignments.items():
                if cinfo.get("ephemeral", True) is True:
                    caches_to_remove.append(comp_id)

            # Actually remove them if ephemeral
            for cid in comps_to_remove:
                device.deallocate_resources(cid, force=True)
            for cid in caches_to_remove:
                device.deallocate_resources(cid, force=True)

    def _is_ephemeral_component(self, comp_id: str) -> bool:
        """
        E.g., attention heads are ephemeral=False, caches ephemeral=False.
        Return True for everything else (projection, ffn).
        """
        if comp_id.startswith("head_") or comp_id.endswith("_cache"):
            return False
        return True

    def _estimate_latency(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> float:
        return 0.0

    def _estimate_comm_time(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> float:
        return 0.0

    def _get_dependencies(self, component_id: str) -> List[str]:
        deps = []
        if component_id == "projection":
            # depends on all heads
            if hasattr(self.transformer, 'attention_heads'):
                for head in self.transformer.attention_heads:
                    deps.append(head.component_id)
        elif component_id == "ffn":
            # depends on projection
            deps.append("projection")
        return deps

    def _estimate_transfer_size(self, source_id: str, target_id: str) -> float:
        # Simple method. Head->projection or projection->ffn
        if source_id.startswith("head_") and target_id == "projection":
            return (self.transformer.current_sequence_length *
                    self.transformer.config.head_dim *
                    self.transformer.config.precision_bytes) / (1024**3)
        elif source_id == "projection" and target_id == "ffn":
            return (self.transformer.current_sequence_length *
                    self.transformer.config.embedding_dim *
                    self.transformer.config.precision_bytes) / (1024**3)
        return 0.0

    def _compute_latency_and_comm(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> Tuple[float, float, float]:
        """
        Default (no real concurrency). Subclasses can override or do 3-phase logic.
        """
        total_latency = self._estimate_latency(assignments, cache_assignments, generation_step)
        comm_time = self._estimate_comm_time(assignments, cache_assignments, generation_step)
        data_gb = 0.0
        return (total_latency, comm_time, data_gb)


class GreedyDistributor(BaseDistributor):
    """Greedy: assign each component to the first device that can fit it."""
    def compute_assignment(
        self,
        generation_step: int,
        previous_assignments: Optional[Dict[str, str]] = None,
        previous_cache: Optional[Dict[str, str]] = None
    ) -> AssignmentResult:

        self._reset_device_states_for_step()

        if self.logger:
            self.logger.log_event(
                "greedy_compute",
                f"[GreedyDistributor] compute_assignment called, step={generation_step}",
                level=LogLevel.DEBUG
            )

        try:
            assignments = {}
            cache_assignments = {}

            for component in self.transformer.get_all_components():
                comp_id = component.component_id
                memory_req = component.compute_memory_requirements(self.transformer.current_sequence_length)
                compute_req = component.compute_flops(self.transformer.current_sequence_length)
                cache_mem = 0.0
                if hasattr(component, 'compute_cache_memory'):
                    cache_mem = component.compute_cache_memory(generation_step)

                # ephemeral or not
                is_ephemeral = self._is_ephemeral_component(comp_id)

                assigned = False
                for device in self.devices.values():
                    ok = device.allocate_resources(
                        comp_id,
                        memory_req,
                        compute_req,
                        cache_size=cache_mem,
                        ephemeral=is_ephemeral
                    )
                    if ok:
                        assignments[comp_id] = device.device_id
                        if cache_mem > 0.0:
                            cache_assignments[comp_id] = device.device_id
                        if self.logger:
                            self.logger.log_component_assignment(
                                generation_step,
                                comp_id,
                                device.device_id,
                                assignment_type="greedy"
                            )
                        assigned = True
                        break

                if not assigned:
                    usage = self._get_resource_usage()
                    if self.logger:
                        self.logger.log_error(
                            "greedy_assignment_fail",
                            (
                                f"Could not find device for {comp_id}, "
                                f"mem_req={memory_req:.4f}, flops_req={compute_req:.4f}, ephemeral={is_ephemeral}. "
                                f"Usage:\n{usage}"
                            )
                        )
                    return AssignmentResult(
                        component_assignments=assignments,
                        cache_assignments=cache_assignments,
                        estimated_latency=float('inf'),
                        resource_usage=usage,
                        is_feasible=False,
                        migrations=[],
                        communication_time=0.0,
                        data_transferred_gb=0.0
                    )

            # Validate
            is_feasible = validate_assignment(
                assignments,
                cache_assignments,
                self.transformer,
                self.devices,
                self.network,
                generation_step
            )

            latency, comm_time, data_gb = self._compute_latency_and_comm(
                assignments,
                cache_assignments,
                generation_step
            )
            usage = self._get_resource_usage()

            return AssignmentResult(
                component_assignments=assignments,
                cache_assignments=cache_assignments,
                estimated_latency=latency,
                resource_usage=usage,
                is_feasible=is_feasible,
                migrations=[],
                communication_time=comm_time,
                data_transferred_gb=data_gb
            )

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    "greedy_assignment_exception",
                    f"Exception in GreedyDistributor: {str(e)}"
                )
            traceback.print_exc()
            usage = self._get_resource_usage()
            return AssignmentResult(
                component_assignments={},
                cache_assignments={},
                estimated_latency=float('inf'),
                resource_usage=usage,
                is_feasible=False,
                migrations=[],
                communication_time=0.0,
                data_transferred_gb=0.0
            )

    # Example: separate pass to sum data xfer times
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
                deps = self._get_dependencies(comp_id)
                for dep_id in deps:
                    if dep_id in assignments:
                        src_dev = assignments[dep_id]
                        if src_dev != dev_id:
                            data_size_gb = self._estimate_transfer_size(dep_id, comp_id)
                            total_data_gb += data_size_gb
                            ttime = self.network.calculate_transfer_time(src_dev, dev_id, data_size_gb)
                            total_comm_time += ttime
                            if self.logger:
                                self.logger.log_communication(
                                    step=generation_step,
                                    source_component=dep_id,
                                    target_component=comp_id,
                                    data_size=data_size_gb,
                                    transfer_time=ttime
                                )
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "greedy_comm_stats_fail",
                    f"Error in comm_stats_separately: {str(ex)}"
                )
            traceback.print_exc()

        return (total_comm_time, total_data_gb)

    def _compute_latency_and_comm(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> Tuple[float, float, float]:
        try:
            total_latency = compute_3phase_latency(
                self.transformer, self.devices, self.network,
                assignments, generation_step, concurrency_mode="sum"
            )
            comm_time, data_gb = self._compute_comm_stats_separately(
                assignments, cache_assignments, generation_step
            )
            return (total_latency, comm_time, data_gb)
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "greedy_latency_comm_fail",
                    f"Exception: {str(ex)}"
                )
            traceback.print_exc()
            return (float('inf'), 0.0, 0.0)


class RoundRobinDistributor(BaseDistributor):
    """Round-robin: cycle across devices for each component in order."""
    def compute_assignment(
        self,
        generation_step: int,
        previous_assignments: Optional[Dict[str, str]] = None,
        previous_cache: Optional[Dict[str, str]] = None
    ) -> AssignmentResult:

        self._reset_device_states_for_step()

        if self.logger:
            self.logger.log_event(
                "round_robin_compute",
                f"[RoundRobinDistributor] compute_assignment called, step={generation_step}",
                level=LogLevel.DEBUG
            )

        try:
            assignments = {}
            cache_assignments = {}
            device_list = list(self.devices.values())
            idx = 0

            for component in self.transformer.get_all_components():
                comp_id = component.component_id
                mem_req = component.compute_memory_requirements(self.transformer.current_sequence_length)
                comp_req = component.compute_flops(self.transformer.current_sequence_length)
                cache_mem = 0.0
                if hasattr(component, 'compute_cache_memory'):
                    cache_mem = component.compute_cache_memory(generation_step)

                ephemeral = self._is_ephemeral_component(comp_id)

                assigned = False
                attempts = 0
                while attempts < len(device_list) and not assigned:
                    dev = device_list[idx]
                    ok = dev.allocate_resources(
                        comp_id,
                        mem_req,
                        comp_req,
                        cache_size=cache_mem,
                        ephemeral=ephemeral
                    )
                    if ok:
                        assignments[comp_id] = dev.device_id
                        if cache_mem > 0.0:
                            cache_assignments[comp_id] = dev.device_id
                        if self.logger:
                            self.logger.log_component_assignment(
                                generation_step,
                                comp_id,
                                dev.device_id,
                                assignment_type="round_robin"
                            )
                        assigned = True
                    idx = (idx + 1) % len(device_list)
                    attempts += 1

                if not assigned:
                    usage = self._get_resource_usage()
                    if self.logger:
                        self.logger.log_error(
                            "round_robin_assignment_fail",
                            f"Cannot place {comp_id}, ephemeral={ephemeral}, usage:\n{usage}"
                        )
                    return AssignmentResult(
                        component_assignments=assignments,
                        cache_assignments=cache_assignments,
                        estimated_latency=float('inf'),
                        resource_usage=usage,
                        is_feasible=False,
                        migrations=[],
                        communication_time=0.0,
                        data_transferred_gb=0.0
                    )

            is_feasible = validate_assignment(
                assignments, cache_assignments,
                self.transformer, self.devices,
                self.network, generation_step
            )
            latency, comm_time, data_gb = self._compute_latency_and_comm(
                assignments, cache_assignments, generation_step
            )
            usage = self._get_resource_usage()

            return AssignmentResult(
                component_assignments=assignments,
                cache_assignments=cache_assignments,
                estimated_latency=latency,
                resource_usage=usage,
                is_feasible=is_feasible,
                migrations=[],
                communication_time=comm_time,
                data_transferred_gb=data_gb
            )

        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "round_robin_assignment_exception",
                    f"Exception: {str(ex)}"
                )
            traceback.print_exc()
            usage = self._get_resource_usage()
            return AssignmentResult(
                component_assignments={},
                cache_assignments={},
                estimated_latency=float('inf'),
                resource_usage=usage,
                is_feasible=False,
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
        """Similar approach: sum data xfers ignoring concurrency."""
        total_comm_time = 0.0
        total_data_gb = 0.0
        try:
            for comp_id, dev_id in assignments.items():
                deps = self._get_dependencies(comp_id)
                for dep_id in deps:
                    if dep_id in assignments:
                        src_dev = assignments[dep_id]
                        if src_dev != dev_id:
                            data_size_gb = self._estimate_transfer_size(dep_id, comp_id)
                            total_data_gb += data_size_gb
                            ttime = self.network.calculate_transfer_time(src_dev, dev_id, data_size_gb)
                            total_comm_time += ttime
                            if self.logger:
                                self.logger.log_communication(
                                    step=generation_step,
                                    source_component=dep_id,
                                    target_component=comp_id,
                                    data_size=data_size_gb,
                                    transfer_time=ttime
                                )
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "round_robin_comm_stats_fail",
                    f"Error in comm_stats_separately: {str(ex)}"
                )
            traceback.print_exc()
        return (total_comm_time, total_data_gb)

    def _compute_latency_and_comm(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> Tuple[float, float, float]:
        try:
            total_latency = compute_3phase_latency(
                self.transformer, self.devices, self.network,
                assignments, generation_step, concurrency_mode="sum"
            )
            comm_time, data_gb = self._compute_comm_stats_separately(
                assignments, cache_assignments, generation_step
            )
            return (total_latency, comm_time, data_gb)
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "round_robin_latency_comm_fail",
                    f"Exception: {str(ex)}"
                )
            traceback.print_exc()
            return (float('inf'), 0.0, 0.0)


class StaticDistributor(BaseDistributor):
    """
    Static: pick an initial assignment at step=0 (via Greedy), then reuse it every step.
    Only ephemeral components get freed each step, but we do not move them around.
    """
    def __init__(
        self,
        transformer: Transformer,
        network: Network,
        devices: Dict[str, Device],
        logger: Optional[SimulationLogger] = None
    ):
        super().__init__(transformer, network, devices, logger)
        self.initial_assignments: Optional[Dict[str, str]] = None
        self.initial_cache: Optional[Dict[str, str]] = None

    def compute_assignment(
        self,
        generation_step: int,
        previous_assignments: Optional[Dict[str, str]] = None,
        previous_cache: Optional[Dict[str, str]] = None
    ) -> AssignmentResult:

        self._reset_device_states_for_step()

        if self.logger:
            self.logger.log_event(
                "static_compute",
                f"[StaticDistributor] step={generation_step}",
                level=LogLevel.DEBUG
            )

        try:
            # On step=0, do a one-time greedy if we have no initial_assignments
            if self.initial_assignments is None:
                from .baselines import GreedyDistributor
                gd = GreedyDistributor(self.transformer, self.network, self.devices, logger=self.logger)
                init_res = gd.compute_assignment(0, None, None)
                if not init_res.is_feasible:
                    usage = self._get_resource_usage()
                    if self.logger:
                        self.logger.log_error(
                            "static_initial_fail",
                            f"Initial Greedy fail at step=0. usage:\n{usage}"
                        )
                    return init_res
                self.initial_assignments = init_res.component_assignments
                self.initial_cache = init_res.cache_assignments

            # Check feasibility of reusing the same assignment
            is_feasible = validate_assignment(
                self.initial_assignments,
                self.initial_cache,
                self.transformer,
                self.devices,
                self.network,
                generation_step
            )

            latency, comm_time, data_gb = self._compute_latency_and_comm(
                self.initial_assignments,
                self.initial_cache,
                generation_step
            )
            usage = self._get_resource_usage()

            return AssignmentResult(
                component_assignments=self.initial_assignments,
                cache_assignments=self.initial_cache,
                estimated_latency=latency,
                resource_usage=usage,
                is_feasible=is_feasible,
                migrations=[],
                communication_time=comm_time,
                data_transferred_gb=data_gb
            )

        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "static_assignment_exception",
                    f"Exception in StaticDistributor: {str(ex)}"
                )
            traceback.print_exc()
            usage = self._get_resource_usage()
            return AssignmentResult(
                component_assignments={},
                cache_assignments={},
                estimated_latency=float('inf'),
                resource_usage=usage,
                is_feasible=False,
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
        """Sum comm times ignoring concurrency."""
        total_comm_time = 0.0
        total_data_gb = 0.0
        try:
            for comp_id, dev_id in assignments.items():
                deps = self._get_dependencies(comp_id)
                for dep_id in deps:
                    if dep_id in assignments:
                        src_dev = assignments[dep_id]
                        if src_dev != dev_id:
                            data_size_gb = self._estimate_transfer_size(dep_id, comp_id)
                            total_data_gb += data_size_gb
                            ttime = self.network.calculate_transfer_time(src_dev, dev_id, data_size_gb)
                            total_comm_time += ttime
                            if self.logger:
                                self.logger.log_communication(
                                    step=generation_step,
                                    source_component=dep_id,
                                    target_component=comp_id,
                                    data_size=data_size_gb,
                                    transfer_time=ttime
                                )
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "static_comm_stats_fail",
                    f"Error in comm_stats_separately: {str(ex)}"
                )
            traceback.print_exc()
        return (total_comm_time, total_data_gb)

    def _compute_latency_and_comm(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> Tuple[float, float, float]:
        try:
            total_latency = compute_3phase_latency(
                self.transformer, self.devices, self.network,
                assignments, generation_step, concurrency_mode="sum"
            )
            comm_time, data_gb = self._compute_comm_stats_separately(
                assignments, cache_assignments, generation_step
            )
            return (total_latency, comm_time, data_gb)
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "static_latency_comm_fail",
                    f"Exception in _compute_latency_and_comm: {str(ex)}"
                )
            traceback.print_exc()
            return (float('inf'), 0.0, 0.0)


class DynamicMigrationDistributor(BaseDistributor):
    """
    Dynamic approach: re-check feasibility at each step, possibly reassign.
    (Placeholder: in practice we only do ephemeral frees, or rely on thresholds.)
    """
    def __init__(
        self,
        transformer: Transformer,
        network: Network,
        devices: Dict[str, Device],
        memory_threshold: float = 0.9,
        compute_threshold: float = 0.9,
        logger: Optional[SimulationLogger] = None
    ):
        super().__init__(transformer, network, devices, logger)
        self.memory_threshold = memory_threshold
        self.compute_threshold = compute_threshold
        self.initial_assignments: Optional[Dict[str, str]] = None
        self.initial_cache: Optional[Dict[str, str]] = None

    def compute_assignment(
        self,
        generation_step: int,
        previous_assignments: Optional[Dict[str, str]] = None,
        previous_cache: Optional[Dict[str, str]] = None
    ) -> AssignmentResult:

        self._reset_device_states_for_step()

        if self.logger:
            self.logger.log_event(
                "dynamic_compute",
                f"[DynamicMigrationDistributor] compute_assignment called, step={generation_step}",
                level=LogLevel.DEBUG
            )

        try:
            if self.initial_assignments is None and previous_assignments is None:
                # do a one-time Greedy
                from .baselines import GreedyDistributor
                gd = GreedyDistributor(self.transformer, self.network, self.devices, logger=self.logger)
                init_res = gd.compute_assignment(generation_step, None, None)
                init_res.migrations = init_res.migrations or []
                if not init_res.is_feasible:
                    usage = self._get_resource_usage()
                    if self.logger:
                        self.logger.log_error(
                            "dynamic_initial_greedy_fail",
                            "Initial assignment via Greedy not feasible.\n"
                            f"Device usage:\n{usage}"
                        )
                self.initial_assignments = init_res.component_assignments
                self.initial_cache = init_res.cache_assignments
                return init_res

            if self.initial_assignments is None:
                # fallback if we had a previous
                self.initial_assignments = dict(previous_assignments or {})
                self.initial_cache = dict(previous_cache or {})

            # Minimal approach: just re-validate
            is_feasible = validate_assignment(
                self.initial_assignments,
                self.initial_cache,
                self.transformer,
                self.devices,
                self.network,
                generation_step
            )
            latency, comm_time, data_gb = self._compute_latency_and_comm(
                self.initial_assignments,
                self.initial_cache,
                generation_step
            )
            usage = self._get_resource_usage()

            return AssignmentResult(
                component_assignments=self.initial_assignments,
                cache_assignments=self.initial_cache,
                estimated_latency=latency,
                resource_usage=usage,
                is_feasible=is_feasible,
                migrations=[],
                communication_time=comm_time,
                data_transferred_gb=data_gb
            )

        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "dynamic_assignment_exception",
                    f"Exception: {str(ex)}"
                )
            traceback.print_exc()
            usage = self._get_resource_usage()
            return AssignmentResult(
                component_assignments={},
                cache_assignments={},
                estimated_latency=float('inf'),
                resource_usage=usage,
                is_feasible=False,
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
        """Similar to others: sum data from each dep."""
        total_comm_time = 0.0
        total_data_gb = 0.0
        try:
            for comp_id, dev_id in assignments.items():
                deps = self._get_dependencies(comp_id)
                for dep_id in deps:
                    if dep_id in assignments:
                        src_dev = assignments[dep_id]
                        if src_dev != dev_id:
                            data_size_gb = self._estimate_transfer_size(dep_id, comp_id)
                            total_data_gb += data_size_gb
                            ttime = self.network.calculate_transfer_time(src_dev, dev_id, data_size_gb)
                            total_comm_time += ttime
                            if self.logger:
                                self.logger.log_communication(
                                    step=generation_step,
                                    source_component=dep_id,
                                    target_component=comp_id,
                                    data_size=data_size_gb,
                                    transfer_time=ttime
                                )
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "dynamic_comm_stats_fail",
                    f"Error in comm_stats_separately: {str(ex)}"
                )
            traceback.print_exc()
        return (total_comm_time, total_data_gb)

    def _compute_latency_and_comm(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> Tuple[float, float, float]:
        try:
            total_latency = compute_3phase_latency(
                self.transformer, self.devices, self.network,
                assignments, generation_step, concurrency_mode="sum"
            )
            comm_time, data_gb = self._compute_comm_stats_separately(
                assignments, cache_assignments, generation_step
            )
            return (total_latency, comm_time, data_gb)
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "dynamic_latency_comm_fail",
                    f"Exception: {str(ex)}"
                )
            traceback.print_exc()
            return (float('inf'), 0.0, 0.0)



class ExactOptimalDistributor(BaseDistributor):
    """
    Exact approach: enumerates all possible assignments of each component to
    each device (for small networks) and picks the assignment with minimal
    concurrency-based 3-phase latency.
    This is only feasible for small #devices and small #components.
    """

    def compute_assignment(
        self,
        generation_step: int,
        previous_assignments: Optional[Dict[str, str]] = None,
        previous_cache: Optional[Dict[str, str]] = None
    ) -> AssignmentResult:
        """
        Steps:
          1) reset ephemeral usage
          2) collect all components
          3) for each feasible assignment (where memory/compute not exceeded),
             compute concurrency-based latency => keep best
        """
        self._reset_device_states_for_step()
        if self.logger:
            self.logger.log_event(
                "exact_optimal_compute",
                f"[ExactOptimalDistributor] step={generation_step}",
                level=LogLevel.DEBUG
            )

        # get components
        components = self.transformer.get_all_components()
        comp_ids = [c.component_id for c in components]

        # We'll store the best scenario
        best_latency = float('inf')
        best_assignments: Dict[str, str] = {}
        best_cache: Dict[str, str] = {}
        best_usage: Dict[str, Dict[str, float]] = {}
        feasible_found = False

        # We'll do a brute force: For each component, pick a device
        # This is device_count^component_count enumerations
        device_keys = list(self.devices.keys())
        import itertools

        # We'll create ephemeral flags
        ephemeral_map = {
            c.component_id: self._is_ephemeral_component(c.component_id)
            for c in components
        }

        # We'll define a helper function that tries to allocate everything for a given assignment
        def try_assignment(assignment_map: Dict[str, str]):
            """
            Returns:
              (feasible: bool, cache_map: dict, usage: dict, concurrency_latency: float)
            """
            # We must copy device states or do a temporary approach
            # We'll do a 'temp usage' approach
            # Easiest is to clone memory/compute usage from each device, allocate, check feasibility.
            # Then if feasible, compute concurrency-based latency.
            # We'll also build a cache_assignments for the heads that need cache (if any).
            from copy import deepcopy

            # snapshot states
            saved_used = {
                d_id: (dev.memory.used, dev.compute.used,
                       dict(dev.assigned_components), dict(dev.cache_assignments))
                for d_id, dev in self.devices.items()
            }

            # zero ephemeral usage
            for d_id, dev in self.devices.items():
                dev.assigned_components = {
                    k: v for (k, v) in dev.assigned_components.items()
                    if not v.get("ephemeral", True)  # keep ephemeral=False
                }
                dev.cache_assignments = {
                    k: c for (k, c) in dev.cache_assignments.items()
                    if not c.get("ephemeral", True)
                }

                # recalc used
                dev.memory.used = sum(float(x["memory"]) for x in dev.assigned_components.values())
                dev.memory.used += sum(float(x["memory"]) for x in dev.cache_assignments.values())
                dev.compute.used = sum(float(x["compute"]) for x in dev.assigned_components.values())

            # Now allocate each component
            local_cache_assign = {}
            for cid in comp_ids:
                # find the component
                comp_obj = next(c for c in components if c.component_id == cid)
                mem_req = comp_obj.compute_memory_requirements(self.transformer.current_sequence_length)
                comp_req = comp_obj.compute_flops(self.transformer.current_sequence_length)
                ccache_req = 0.0
                if hasattr(comp_obj, "compute_cache_memory"):
                    ccache_req = comp_obj.compute_cache_memory(generation_step)
                ephemeral_flag = ephemeral_map[cid]

                # which device
                dev_id = assignment_map[cid]
                d = self.devices[dev_id]
                ok_main = d.allocate_resources(cid, mem_req, comp_req, ephemeral=ephemeral_flag)
                if not ok_main:
                    # revert
                    # restore
                    for dd_id, (m_used, c_used, assigned_c, cache_c) in saved_used.items():
                        self.devices[dd_id].memory.used = m_used
                        self.devices[dd_id].compute.used = c_used
                        self.devices[dd_id].assigned_components = assigned_c
                        self.devices[dd_id].cache_assignments = cache_c
                    return (False, {}, {}, float('inf'))

                if ccache_req > 0:
                    ok_cache = d.allocate_resources(f"{cid}_cache", ccache_req, 0.0, ephemeral=ephemeral_flag)
                    if not ok_cache:
                        # revert
                        for dd_id, (m_used, c_used, assigned_c, cache_c) in saved_used.items():
                            self.devices[dd_id].memory.used = m_used
                            self.devices[dd_id].compute.used = c_used
                            self.devices[dd_id].assigned_components = assigned_c
                            self.devices[dd_id].cache_assignments = cache_c
                        return (False, {}, {}, float('inf'))
                    local_cache_assign[cid] = dev_id

            # if we get here => feasible
            # compute concurrency-based latency
            assign_for_latency = dict(assignment_map)  # comp_id->device
            concurrency_latency = compute_3phase_latency(
                self.transformer, self.devices, self.network, assign_for_latency, generation_step, concurrency_mode="sum"
            )
            usage_now = self._get_resource_usage()
            # revert or keep? We want to keep it if we found best? Let's always revert after measure, then re-allocate the best
            # but we'll just revert for safety
            final_usage = usage_now

            for dd_id, (m_used, c_used, assigned_c, cache_c) in saved_used.items():
                self.devices[dd_id].memory.used = m_used
                self.devices[dd_id].compute.used = c_used
                self.devices[dd_id].assigned_components = assigned_c
                self.devices[dd_id].cache_assignments = cache_c

            return (True, local_cache_assign, final_usage, concurrency_latency)

        # building all device assignments for each component
        # We'll skip ephemeral re-assign for heads that are non-ephemeral? Actually we won't skip. We can place heads anywhere
        # because the scenario might want it. But if you want the privacy constraint or something, you can add it. We'll skip.

        import itertools
        dev_count = len(device_keys)

        # The brute force approach: cartesian product of device_keys for comp_ids
        # e.g. for comp in comp_ids, pick device in device_keys => dev_count^len(comp_ids) combos
        for combo in itertools.product(device_keys, repeat=len(comp_ids)):
            # combo is a tuple, e.g. ('device_0','device_1','device_1') for 3 comps
            assignment_map = {}
            for i, cid in enumerate(comp_ids):
                assignment_map[cid] = combo[i]

            feasible_flag, local_cache_map, usage_now, concurrency_lat = try_assignment(assignment_map)
            if feasible_flag and concurrency_lat < best_latency:
                best_latency = concurrency_lat
                best_assignments = dict(assignment_map)
                best_cache = dict(local_cache_map)
                best_usage = usage_now
                feasible_found = True

        if not feasible_found:
            # no feasible assignment
            usage = self._get_resource_usage()
            return AssignmentResult(
                component_assignments={},
                cache_assignments={},
                estimated_latency=float('inf'),
                resource_usage=usage,
                is_feasible=False,
                migrations=[],
                communication_time=0.0,
                data_transferred_gb=0.0
            )
        else:
            # re-allocate for best assignment so the device states are consistent
            # or we can skip if we don't rely on device states after this
            # We'll do a final pass that actually assigns them so if next step uses them, it's correct
            self._reset_device_states_for_step()
            for cid in comp_ids:
                comp_obj = next(c for c in components if c.component_id == cid)
                mem_req = comp_obj.compute_memory_requirements(self.transformer.current_sequence_length)
                comp_req = comp_obj.compute_flops(self.transformer.current_sequence_length)
                ephemeral_flag = ephemeral_map[cid]
                dev_id = best_assignments[cid]
                dev = self.devices[dev_id]
                dev.allocate_resources(cid, mem_req, comp_req, ephemeral=ephemeral_flag)
                if cid in best_cache:
                    ccache_req = 0.0
                    if hasattr(comp_obj, 'compute_cache_memory'):
                        ccache_req = comp_obj.compute_cache_memory(generation_step)
                    dev.allocate_resources(f"{cid}_cache", ccache_req, 0.0, ephemeral=ephemeral_flag)

            final_usage = self._get_resource_usage()

            # We'll do no advanced comm_time for now
            # If you want to sum data, you can do a separate pass
            # We'll do concurrency-based approach => best_latency, comm_time=0
            # or we can do a separate sum of data
            # For clarity, let's keep comm_time=0
            # Data transferred is 0 for a single-step? We can do more advanced if we want
            return AssignmentResult(
                component_assignments=best_assignments,
                cache_assignments=best_cache,
                estimated_latency=best_latency,
                resource_usage=final_usage,
                is_feasible=True,
                migrations=[],
                communication_time=0.0,
                data_transferred_gb=0.0
            )

