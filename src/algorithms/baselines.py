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
        """
        :param transformer: The Transformer model instance (with config, layers, etc.).
        :param network: The simulated network topology.
        :param devices: A dict of device_id -> Device objects.
        :param logger: (Optional) SimulationLogger for structured logging.
        """
        self.transformer = transformer
        self.network = network
        self.devices = devices
        self.logger = logger  # store the logger if provided
        
    @abstractmethod
    def compute_assignment(
        self,
        generation_step: int,
        previous_assignments: Optional[Dict[str, str]] = None,
        previous_cache: Optional[Dict[str, str]] = None
    ) -> AssignmentResult:
        """
        Subclasses must implement. Returns an AssignmentResult that includes:
          - Assignments (component->device)
          - Cache assignments if any
          - Estimated latency
          - Resource usage
          - Feasibility
          - Migrations (if relevant)
          - Communication overhead info
        """
        pass
        
    def _get_resource_usage(self) -> Dict[str, Dict[str, float]]:
        """Collect current usage from each device. Useful for debugging or building partial results."""
        usage = {}
        for dev_id, device in self.devices.items():
            mem_used = device.memory.used
            mem_cap = device.memory.capacity
            comp_used = device.compute.used
            comp_cap = device.compute.capacity
            usage[dev_id] = {
                'memory_used': mem_used,
                'memory_capacity': mem_cap,
                'compute_used': comp_used,
                'compute_capacity': comp_cap,
                'memory_utilization': (mem_used / mem_cap) if mem_cap > 0 else 1.0,
                'compute_utilization': (comp_used / comp_cap) if comp_cap > 0 else 1.0
            }
        return usage
    
    def _estimate_latency(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> float:
        """Default minimal or naive latency for baseline. Override if needed."""
        return 0.0

    def _estimate_comm_time(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> float:
        """Default minimal communication overhead for baseline. Override if needed."""
        return 0.0

    def _get_dependencies(self, component_id: str) -> List[str]:
        """
        Return a list of dependencies for a given component. 
        Usually: "projection" depends on all attention heads, "ffn" depends on "projection".
        """
        deps = []
        if component_id == "projection":
            if hasattr(self.transformer, 'attention_heads'):
                for head in self.transformer.attention_heads:
                    deps.append(head.component_id)
        elif component_id == "ffn":
            deps.append("projection")
        return deps

    def _estimate_transfer_size(
        self,
        source_id: str,
        target_id: str
    ) -> float:
        """
        Estimate data size in GB for transferring outputs from source_id to target_id.
        This is a naive placeholder; real logic can be more elaborate.
        """
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
        Can be overridden or used as a fallback for concurrency-based logic, etc.
        Returns (latency, comm_time, data_gb).
        """
        total_latency = self._estimate_latency(assignments, cache_assignments, generation_step)
        comm_time = self._estimate_comm_time(assignments, cache_assignments, generation_step)
        data_gb = 0.0
        return (total_latency, comm_time, data_gb)


class GreedyDistributor(BaseDistributor):
    """Greedy strategy: assign each component to the first feasible device."""

    def compute_assignment(
        self,
        generation_step: int,
        previous_assignments: Optional[Dict[str, str]] = None,
        previous_cache: Optional[Dict[str, str]] = None
    ) -> AssignmentResult:

        if self.logger:
            self.logger.log_event(
                "greedy_compute",
                f"[GreedyDistributor] compute_assignment called, step={generation_step}",
                level=LogLevel.DEBUG
            )

        try:
            assignments = {}
            cache_assignments = {}

            # For each component in the transformer
            for component in self.transformer.get_all_components():
                comp_id = component.component_id
                memory_req = component.compute_memory_requirements(self.transformer.current_sequence_length)
                compute_req = component.compute_flops(self.transformer.current_sequence_length)
                if hasattr(component, 'compute_cache_memory'):
                    memory_req += component.compute_cache_memory(generation_step)

                assigned = False
                for device in self.devices.values():
                    if device.can_accommodate(memory_req, compute_req):
                        assignments[comp_id] = device.device_id
                        if hasattr(component, 'compute_cache_memory'):
                            cache_assignments[comp_id] = device.device_id
                        assigned = True
                        # Optionally log this assignment
                        if self.logger:
                            self.logger.log_component_assignment(
                                generation_step,
                                comp_id,
                                device.device_id,
                                assignment_type="greedy"
                            )
                        break

                if not assigned:
                    # Failure: not feasible
                    if self.logger:
                        self.logger.log_error(
                            "greedy_assignment_fail",
                            f"Could not find a device for component={comp_id}, mem_req={memory_req:.4f}, flops_req={compute_req:.4f}"
                        )
                    usage = self._get_resource_usage()
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

            # Validate final assignment
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
                    f"Exception in GreedyDistributor compute_assignment: {str(e)}"
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
        """
        Returns (comm_time, data_gb) by summing data transfers
        from each dependency, ignoring concurrency or pipeline overlap.
        """
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
                    f"Error in _compute_comm_stats_separately: {str(ex)}"
                )
            traceback.print_exc()

        return total_comm_time, total_data_gb
    
    def _compute_latency_and_comm(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> Tuple[float, float, float]:
        """
        We'll compute concurrency-based total latency from compute_3phase_latency(),
        then do a separate pass for comm_time + data_gb.
        """
        try:
            total_latency = compute_3phase_latency(
                self.transformer,
                self.devices,
                self.network,
                assignments,
                generation_step,
                concurrency_mode="sum"
            )
            comm_time, data_gb = self._compute_comm_stats_separately(
                assignments,
                cache_assignments,
                generation_step
            )
            return (total_latency, comm_time, data_gb)
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "greedy_latency_comm_fail",
                    f"Exception in _compute_latency_and_comm: {str(ex)}"
                )
            traceback.print_exc()
            return (float('inf'), 0.0, 0.0)


class RoundRobinDistributor(BaseDistributor):
    """Round-robin distribution: cycle through devices in order."""

    def compute_assignment(
        self,
        generation_step: int,
        previous_assignments: Optional[Dict[str, str]] = None,
        previous_cache: Optional[Dict[str, str]] = None
    ) -> AssignmentResult:

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
                if hasattr(component, 'compute_cache_memory'):
                    mem_req += component.compute_cache_memory(generation_step)

                assigned = False
                attempts = 0
                while attempts < len(device_list) and not assigned:
                    dev = device_list[idx]
                    if dev.can_accommodate(mem_req, comp_req):
                        assignments[comp_id] = dev.device_id
                        if hasattr(component, 'compute_cache_memory'):
                            cache_assignments[comp_id] = dev.device_id
                        assigned = True
                        if self.logger:
                            self.logger.log_component_assignment(
                                generation_step,
                                comp_id,
                                dev.device_id,
                                assignment_type="round_robin"
                            )
                    idx = (idx + 1) % len(device_list)
                    attempts += 1

                if not assigned:
                    if self.logger:
                        self.logger.log_error(
                            "round_robin_assignment_fail",
                            f"Cannot place component={comp_id}, mem_req={mem_req:.4f}, compute_req={comp_req:.4f}"
                        )
                    usage = self._get_resource_usage()
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
                    "round_robin_assignment_exception",
                    f"Exception in RoundRobinDistributor compute_assignment: {str(e)}"
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
        """Similar logic for comm/time as the Greedy approach."""
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
                    f"Error in _compute_comm_stats_separately: {str(ex)}"
                )
            traceback.print_exc()

        return total_comm_time, total_data_gb
    
    def _compute_latency_and_comm(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> Tuple[float, float, float]:
        """Use concurrency-based function plus separate pass for comm/time."""
        try:
            total_latency = compute_3phase_latency(
                self.transformer,
                self.devices,
                self.network,
                assignments,
                generation_step,
                concurrency_mode="sum"
            )
            comm_time, data_gb = self._compute_comm_stats_separately(
                assignments, cache_assignments, generation_step
            )
            return (total_latency, comm_time, data_gb)
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "round_robin_latency_comm_fail",
                    f"Exception in _compute_latency_and_comm: {str(ex)}"
                )
            traceback.print_exc()
            return (float('inf'), 0.0, 0.0)


class StaticDistributor(BaseDistributor):
    """Static partitioning: fix the assignment at step 0 and reuse it every step."""

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

        if self.logger:
            self.logger.log_event(
                "static_compute",
                f"[StaticDistributor] compute_assignment called, step={generation_step}",
                level=LogLevel.DEBUG
            )

        try:
            # If no initial assignment, do a one-time Greedy
            if self.initial_assignments is None:
                from .baselines import GreedyDistributor
                gd = GreedyDistributor(self.transformer, self.network, self.devices, logger=self.logger)
                init_res = gd.compute_assignment(0, None, None)
                if not init_res.is_feasible:
                    if self.logger:
                        self.logger.log_error(
                            "static_initial_fail",
                            f"Failed initial assignment with Greedy at step=0"
                        )
                    return init_res
                self.initial_assignments = init_res.component_assignments
                self.initial_cache = init_res.cache_assignments

            # Validate
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

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    "static_assignment_exception",
                    f"Exception in StaticDistributor compute_assignment: {str(e)}"
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
        """Sum communication times and data for each dependency (no concurrency)."""
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
                    f"Error in _compute_comm_stats_separately: {str(ex)}"
                )
            traceback.print_exc()

        return total_comm_time, total_data_gb
    
    def _compute_latency_and_comm(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> Tuple[float, float, float]:
        """Concurrency-based total latency + separate pass for comm."""
        try:
            total_latency = compute_3phase_latency(
                self.transformer,
                self.devices,
                self.network,
                assignments,
                generation_step,
                concurrency_mode="sum"
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
    """Dynamic migration that reassigns on threshold exceed (placeholder logic)."""

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

        if self.logger:
            self.logger.log_event(
                "dynamic_compute",
                f"[DynamicMigrationDistributor] compute_assignment called, step={generation_step}",
                level=LogLevel.DEBUG
            )

        try:
            # If we don't have an initial assignment, or no prev, do Greedy
            if self.initial_assignments is None and previous_assignments is None:
                from .baselines import GreedyDistributor
                gd = GreedyDistributor(self.transformer, self.network, self.devices, logger=self.logger)
                init_res = gd.compute_assignment(generation_step, None, None)
                init_res.migrations = init_res.migrations or []
                if not init_res.is_feasible and self.logger:
                    self.logger.log_error(
                        "dynamic_initial_greedy_fail",
                        "Could not do initial assignment via Greedy"
                    )
                self.initial_assignments = init_res.component_assignments
                self.initial_cache = init_res.cache_assignments
                return init_res

            # Otherwise, just re-validate old assignment (placeholder).
            if self.initial_assignments is None:
                # possibly we have previous_assignments from prior step
                self.initial_assignments = dict(previous_assignments)
                self.initial_cache = dict(previous_cache) if previous_cache else {}

            # Check feasibility, thresholds, etc. (not implemented fully)
            is_feasible = validate_assignment(
                self.initial_assignments,
                self.initial_cache,
                self.transformer,
                self.devices,
                self.network,
                generation_step
            )

            # concurrency-based total
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
        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    "dynamic_assignment_exception",
                    f"Exception in DynamicMigrationDistributor compute_assignment: {str(e)}"
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
        """Same approach as in other classes for comm/time, ignoring concurrency."""
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
                    f"Error in _compute_comm_stats_separately: {str(ex)}"
                )
            traceback.print_exc()

        return total_comm_time, total_data_gb
    
    def _compute_latency_and_comm(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> Tuple[float, float, float]:
        """Compute concurrency-based total latency, plus separate pass for comm/time."""
        try:
            total_latency = compute_3phase_latency(
                self.transformer,
                self.devices,
                self.network,
                assignments,
                generation_step,
                concurrency_mode="sum"
            )
            comm_time, data_gb = self._compute_comm_stats_separately(
                assignments, cache_assignments, generation_step
            )
            return (total_latency, comm_time, data_gb)
        except Exception as ex:
            if self.logger:
                self.logger.log_error(
                    "dynamic_latency_comm_fail",
                    f"Exception in _compute_latency_and_comm: {str(ex)}"
                )
            traceback.print_exc()
            return (float('inf'), 0.0, 0.0)
