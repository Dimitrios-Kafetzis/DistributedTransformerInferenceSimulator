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
# File:    src/algorithms/galaxy.py
# Description:
#   Implements a simplified "Galaxy"-style hybrid parallel distribution
#   approach for Transformer inference, inspired by:
#
#       Ye, M., Shen, B., Li, W., Li, X., & Guan, J. (2023).
#       "Galaxy: A Hybrid-Parallel Approach for Training & Inference
#       in Large Language Models."
#       https://arxiv.org/pdf/2405.17245
#
#   Galaxy proposes a multi-dimensional parallel scheme (pipeline, tensor,
#   sequence parallelism) to distribute Transformer blocks across devices
#   more flexibly than purely pipeline-based or data-parallel strategies.
#   Here, we implement a simplified version focused on pipeline + tensor
#   parallel for inference. In short:
#
#     1) We partition the Transformer’s decoder layers into pipeline
#        stages (shards).
#     2) For each stage, we also replicate or split the big matrix
#        multiplications among multiple devices (tensor parallel dimension).
#     3) (Optional) If we had large inference batches, we could add
#        sequence parallelism, but single-sequence autoregressive
#        inference typically doesn't benefit from that dimension.
#
#   The code below mirrors the pipeline logic from EdgeShard plus a
#   basic notion of "tensor shards" within each pipeline stage. We
#   do not replicate the advanced ILP or dynamic programming that Galaxy
#   uses to find an optimal partition. Instead, we demonstrate a
#   static or semi-static approach that merges these two parallel
#   dimensions. This is sufficient to compare to your resource-aware
#   method and other baselines in the same simulation framework.
#
# ------------------------------------------------------------------------------

"""
GalaxyDistributor:
------------------

This module introduces the GalaxyDistributor class, which attempts to
combine pipeline partitioning (layer-wise) with tensor splitting for
each layer. Conceptually, each pipeline stage (a contiguous set of
layers) is assigned to a "stage group" of devices that collectively
perform the layer’s computations in parallel at the tensor level.

Key Steps:
----------
1) **Pipeline Partition**:
   - We split the model’s layers into N_stages contiguous shards, 
     just like pipeline parallelism. E.g., if there are 24 layers
     and we want 3 pipeline stages, we might do layers [0..7], [8..15],
     [16..23].
2) **Tensor Split**:
   - For each pipeline stage, we have M_devices assigned as a "tensor
     group." Instead of a single device hosting that entire shard,
     we distribute the big matrix multiplications across these M_devices
     at every forward pass. We approximate that by dividing the 
     memory + compute requirements among the devices in that group.
   - If the group has M_devices, each device in the group 
     effectively handles 1/M of the FLOPs and 1/M of the memory 
     for the major linear layers. (We still store all K/V caches
     for that entire stage across these M devices, but you might
     replicate them or split them. Below we do a naive "split" 
     to keep memory usage consistent, but real Galaxy might replicate 
     or partially replicate certain states.)
3) **Forward/Inference**:
   - During inference, tokens flow from stage 1 -> stage 2 -> ... -> stage N
     (pipeline). Within each stage, the relevant "tensor group" does 
     the big linear ops in parallel. 
   - The simulator’s concurrency-based approach can partially capture
     the pipeline concurrency, but the *intra-stage* concurrency 
     from tensor parallel might require a custom method to reflect
     that all M_devices in a stage group can operate simultaneously.
   - For simplicity, we approximate it by dividing the stage's compute 
     load by M. Then we pick the slowest device in that stage group 
     to define the final compute time for that stage (a "synchronization" 
     at each layer).
4) **Comparison**:
   - This approach can reduce per-stage compute time if you have 
     multiple devices in each stage group. However, you do pay some 
     overhead for partial synchronization or communication among 
     the tensor group. The code below includes a naive 
     `_intra_stage_comm()` to reflect the all-reduce or gather 
     steps typically required in tensor parallel.

In short, this file gives a partial view of how Galaxy’s multi-dimensional
parallelism might be simulated. Real Galaxy also leverages ILP to find
optimal partitions, auto-chunking, dynamic re-distribution, and so on.
But the following class is enough for a “Galaxy-like” test that you can
compare with your other algorithms in the simulator.
"""

import traceback
from typing import Dict, List, Optional, Tuple

from src.utils.logging import SimulationLogger, LogLevel
from ..core import Device, Network, Transformer
from .baselines import BaseDistributor
from .resource_aware import AssignmentResult
from .utils import validate_assignment, compute_3phase_latency


class GalaxyDistributor(BaseDistributor):
    """
    Approximate Galaxy approach with pipeline + tensor parallelism.
    We define:
      - num_stages: how many pipeline shards
      - devices_per_stage: how many devices in each stage group
    We then do:
      1) partition model's layers into contiguous shards (#shards=num_stages),
      2) for each shard, pick a group of (devices_per_stage) devices,
      3) split the compute among those devices for that shard.

    Because the simulator is not fully designed for "group-based"
    parallel allocations, we'll do a simplified approach:
      - For each stage i, we have a list of device_ids that belong 
        to that stage group.
      - The memory & compute for that stage is divided by the group size 
        for an approximate concurrency benefit.
      - We sum up the time across pipeline stages in a concurrency-based
        or pipeline-based manner. 
      - Communication overhead includes stage->stage transfer + 
        an approximate "intra-stage" overhead for all-gather/all-reduce.

    This code is purely for demonstration. Feel free to refine it to 
    better approximate real Galaxy behavior.
    """

    def __init__(
        self,
        transformer: Transformer,
        network: Network,
        devices: Dict[str, Device],
        logger: Optional[SimulationLogger] = None,
        num_stages: int = 2,
        devices_per_stage: int = 2,
    ):
        super().__init__(transformer, network, devices, logger)
        self.num_stages = num_stages
        self.devices_per_stage = devices_per_stage

        # We store pipeline shards: each shard is a list of layer indices
        self.shards: List[List[int]] = []
        # For each shard i, we store a device group: e.g. [devA, devB, devC]
        self.shard_device_groups: List[List[str]] = []

        # We track if we've built the pipeline partition
        self._partition_built = False
        # We also track if we've done the actual memory/compute allocation
        self._has_allocated_stages = False

        # Keep the assignment from the first time we allocate 
        # so we don't re-allocate every step
        self._previous_result: Optional[AssignmentResult] = None

    def compute_assignment(
        self,
        generation_step: int,
        previous_assignments: Optional[Dict[str, str]] = None,
        previous_cache: Optional[Dict[str, str]] = None
    ) -> AssignmentResult:
        """
        Main method called by the simulator each step in autoregressive decoding.

        - If we've already allocated shards (and we're beyond step=0),
          skip re-allocation to avoid stacking persistent usage repeatedly.
        - Otherwise, build the pipeline shards, device groups, and 
          do the stage allocations once.
        """
        # 1) Free ephemeral usage from the previous step
        self._reset_device_states_for_step()

        if self.logger:
            self.logger.log_event(
                "galaxy_compute",
                f"[GalaxyDistributor] step={generation_step}, "
                f"partition_built={self._partition_built}, "
                f"allocated_stages={self._has_allocated_stages}",
                level=LogLevel.DEBUG
            )

        # =========== If we ALREADY allocated the stages for previous steps ===========
        if self._has_allocated_stages and generation_step > 0 and self._previous_result is not None:
            # We skip the repeated stage re-allocation. Just re-calc concurrency if needed.
            # This ensures we won't blow up memory usage by double-allocating persistent shards.

            try:
                total_latency, comm_time, data_gb = self._compute_latency_and_comm(
                    self._previous_result.component_assignments,
                    self._previous_result.cache_assignments,
                    generation_step
                )
            except Exception as ex:
                usage = self._get_resource_usage()
                err_str = f"Galaxy: concurrency-based latency exception at step={generation_step}: {str(ex)}"
                if self.logger:
                    self.logger.log_error("galaxy_latency_exception", err_str)
                    traceback.print_exc()
                return AssignmentResult(
                    component_assignments=self._previous_result.component_assignments,
                    cache_assignments=self._previous_result.cache_assignments,
                    estimated_latency=float('inf'),
                    resource_usage=usage,
                    is_feasible=False,
                    error=err_str
                )

            usage = self._get_resource_usage()
            return AssignmentResult(
                component_assignments=self._previous_result.component_assignments,
                cache_assignments=self._previous_result.cache_assignments,
                estimated_latency=total_latency,
                resource_usage=usage,
                is_feasible=True,
                migrations=[],
                communication_time=comm_time,
                data_transferred_gb=data_gb
            )

        # =========== Otherwise, we do the FIRST ALLOCATION (step=0 or first time) ===========

        # 2) If not yet built the partition of layers -> pipeline stages, do so
        if not self._partition_built:
            try:
                self._build_pipeline_shards()
                self._assign_device_groups()
                self._partition_built = True
            except Exception as ex:
                usage = self._get_resource_usage()
                err_str = f"Galaxy: failed to build partition => {str(ex)}"
                if self.logger:
                    self.logger.log_error("galaxy_partition_fail", err_str)
                return AssignmentResult(
                    component_assignments={},
                    cache_assignments={},
                    estimated_latency=float('inf'),
                    resource_usage=usage,
                    is_feasible=False,
                    error=err_str
                )

        # 3) Perform the stage allocations for the first time
        try:
            assignments, cache_assignments, feasible = self._allocate_stages(generation_step)
            if not feasible:
                usage = self._get_resource_usage()
                return AssignmentResult(
                    component_assignments=assignments,
                    cache_assignments=cache_assignments,
                    estimated_latency=float('inf'),
                    resource_usage=usage,
                    is_feasible=False,
                    error="Galaxy: stage allocation infeasible"
                )

            # 4) Validate
            is_ok = validate_assignment(
                assignments, 
                cache_assignments, 
                self.transformer, 
                self.devices, 
                self.network, 
                generation_step
            )
            if not is_ok:
                usage = self._get_resource_usage()
                return AssignmentResult(
                    component_assignments=assignments,
                    cache_assignments=cache_assignments,
                    estimated_latency=float('inf'),
                    resource_usage=usage,
                    is_feasible=False,
                    error="Galaxy: assignment validation failed"
                )

            # 5) Approx concurrency-based pipeline + partial tensor parallel
            total_latency, comm_time, data_gb = self._compute_latency_and_comm(
                assignments,
                cache_assignments,
                generation_step
            )
            usage = self._get_resource_usage()

            result = AssignmentResult(
                component_assignments=assignments,
                cache_assignments=cache_assignments,
                estimated_latency=total_latency,
                resource_usage=usage,
                is_feasible=True,
                migrations=[],
                communication_time=comm_time,
                data_transferred_gb=data_gb
            )

            # Mark that we've allocated the shards so that at next step
            # we won't re-allocate and cause double usage
            self._has_allocated_stages = True
            self._previous_result = result

            return result

        except Exception as ex:
            usage = self._get_resource_usage()
            err_str = f"Galaxy: exception in stage allocation => {str(ex)}"
            if self.logger:
                self.logger.log_error("galaxy_allocation_exception", err_str)
                traceback.print_exc()
            return AssignmentResult(
                component_assignments={},
                cache_assignments={},
                estimated_latency=float('inf'),
                resource_usage=usage,
                is_feasible=False,
                error=err_str
            )


    # ----------------------------------------------------------------------
    # Partition logic
    # ----------------------------------------------------------------------
    def _build_pipeline_shards(self) -> None:
        """
        Gather decoder layers from `self.transformer`, 
        split them into `self.num_stages` contiguous shards.
        e.g. if 24 layers, num_stages=3 => 8 layers per shard.
        We'll store them in self.shards as a list of lists:
          e.g. [[0,1,2,3,4,5,6,7], [8..15], [16..23]]
        """
        decoder_layers = getattr(self.transformer, "decoder_layers", [])
        n_layers = len(decoder_layers)
        if n_layers == 0:
            self.shards = []
            return

        chunk_size = max(n_layers // self.num_stages, 1)
        self.shards = [[] for _ in range(self.num_stages)]
        for i, layer_obj in enumerate(decoder_layers):
            shard_index = i // chunk_size
            if shard_index >= self.num_stages:
                shard_index = self.num_stages - 1
            self.shards[shard_index].append(layer_obj.layer_id)

    def _assign_device_groups(self) -> None:
        """
        We have `num_stages` pipeline shards. For each shard, pick 
        `devices_per_stage` devices from the pool. If we run out of devices,
        we might reuse some or fail. For simplicity, we'll do a naive approach:
          - sort device_ids,
          - chunk them for each stage group,
          - if not enough devices, we cycle.

        We'll store them in self.shard_device_groups as a list of lists of device_ids.
        e.g. if we have 6 devices, devices_per_stage=2, num_stages=3 => 
             stage0 => [dev0, dev1], stage1 => [dev2, dev3], stage2 => [dev4, dev5]
        If we only have 4 devices, stage2 might reuse dev0, dev1, etc. 
        """
        device_ids = list(self.devices.keys())
        device_ids.sort()
        n_devs = len(device_ids)

        self.shard_device_groups = []
        dev_index = 0
        for stage_i in range(self.num_stages):
            group = []
            for _ in range(self.devices_per_stage):
                if dev_index >= n_devs:
                    dev_index = 0  # cycle or raise an error
                group.append(device_ids[dev_index])
                dev_index += 1
            self.shard_device_groups.append(group)


    # ----------------------------------------------------------------------
    # Allocation logic
    # ----------------------------------------------------------------------
    def _allocate_stages(
        self, generation_step: int
    ) -> Tuple[Dict[str,str], Dict[str,str], bool]:
        """
        For each shard, we gather the memory & compute from all layers in that shard,
        then split that usage among the device group. We approximate that each device
        in the group handles 1/(group_size) of the load. We also store a single 
        "logical" assignment that says each layer belongs to "the group." 
        But in the simulator, we must pick one device_id for each layer's 'component_id' 
        for the final dictionary. We'll just pick the "first" device in the group 
        as the 'owner' for the purpose of the assignment dictionary, but in reality, 
        the entire group cooperates.

        For K/V caches, we similarly split the memory among the group (or replicate it, 
        if you want to reflect actual Galaxy design). By default, let's do a split 
        to keep memory usage from exploding.

        Return: (assignments, cache_assignments, feasible)
        """
        assignments = {}
        cache_assignments = {}
        feasible = True

        decoder_layers = getattr(self.transformer, "decoder_layers", [])
        # Build a quick index: layer_id -> layer_object
        layer_map = {layer_obj.layer_id: layer_obj for layer_obj in decoder_layers}

        for stage_idx, layer_ids in enumerate(self.shards):
            group = self.shard_device_groups[stage_idx]
            group_size = len(group)
            # Summation of memory & flops across all layers in this shard
            total_mem = 0.0
            total_flops = 0.0
            total_cache = 0.0

            for lid in layer_ids:
                layer_obj = layer_map[lid]
                mem_req = layer_obj.compute_memory_requirements(self.transformer.current_sequence_length)
                flops_req = layer_obj.compute_flops(self.transformer.current_sequence_length)
                cache_req = 0.0
                if hasattr(layer_obj, "compute_cache_memory"):
                    cache_req = layer_obj.compute_cache_memory(generation_step)

                total_mem += mem_req
                total_flops += flops_req
                total_cache += cache_req

            # Now the portion each device must hold
            mem_per_dev = total_mem / group_size
            flops_per_dev = total_flops / group_size
            cache_per_dev = total_cache / group_size

            # Each stage is persistent for entire inference
            ephemeral_flag = False

            # Attempt allocation on each device in the group
            for dev_id in group:
                dev = self.devices[dev_id]
                ok_main = dev.allocate_resources(
                    f"galaxy_stage_{stage_idx}_main",
                    mem_per_dev,
                    flops_per_dev,
                    ephemeral=ephemeral_flag
                )
                if not ok_main:
                    feasible = False
                    break

                if cache_per_dev > 0.0:
                    ok_cache = dev.allocate_resources(
                        f"galaxy_stage_{stage_idx}_cache",
                        cache_per_dev,
                        0.0,
                        ephemeral=ephemeral_flag
                    )
                    if not ok_cache:
                        feasible = False
                        break

            if not feasible:
                break

            # Meanwhile, produce a final 'assignments' mapping 
            # so the simulator can track per-layer ownership. 
            # We'll pick the first device in the group as the "owner."
            if group:
                main_dev_id = group[0]
                for lid in layer_ids:
                    layer_obj = layer_map[lid]
                    comp_id = layer_obj.component_id
                    assignments[comp_id] = main_dev_id
                    if hasattr(layer_obj, "compute_cache_memory"):
                        cache_assignments[comp_id] = main_dev_id

        return assignments, cache_assignments, feasible

    # ----------------------------------------------------------------------
    # Concurrency + Communication
    # ----------------------------------------------------------------------
    def _compute_latency_and_comm(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> Tuple[float, float, float]:
        """
        We'll do a multi-step approach:
          1) Approx pipeline concurrency with `compute_3phase_latency`.
             Our partial flops allocation in _allocate_stages means each
             "representative device" sees smaller usage => effectively
             some speedup from tensor parallel.
          2) Add an "intra-stage" gather overhead if needed. 
             Real Galaxy does an all-reduce or gather step for partial outputs 
             inside each stage group. We'll do a naive `_intra_stage_comm()`.
          3) Sum up pipeline stage->stage transfers with `_sum_pipeline_comm()`.

        We return (total_latency + comm_overhead, total_comm_time, data_xferred).
        """
        total_latency = compute_3phase_latency(
            self.transformer,
            self.devices,
            self.network,
            assignments,
            generation_step,
            concurrency_mode="sum"
        )

        # Intra-stage comm overhead for tensor parallel
        intra_stage_time, intra_data_gb = self._intra_stage_comm()
        # Pipeline stage->stage overhead
        pipeline_time, pipeline_data_gb = self._sum_pipeline_comm(assignments)

        comm_time = intra_stage_time + pipeline_time
        data_gb = intra_data_gb + pipeline_data_gb

        # We'll sum up total_latency + comm_time 
        # though you might want a more refined concurrency approach 
        # if the comm can overlap with compute
        return (total_latency + comm_time, comm_time, data_gb)

    def _intra_stage_comm(self) -> Tuple[float, float]:
        """
        Naive approximation of communication overhead within each 
        stage group for tensor parallel. Typically an all-reduce 
        or gather for partial outputs. We do a ring-based approach 
        to model some allreduce cost for each group. 
        """
        total_time = 0.0
        total_data = 0.0

        # Suppose each stage triggers ~2 GB of data all-reduce 
        # for demonstration
        allreduce_gb_per_stage = 2.0

        for group in self.shard_device_groups:
            group_size = len(group)
            if group_size <= 1:
                continue

            main_dev = group[0]
            # ring-based or star-based approach, let's do star-based 
            # from main_dev -> each other dev
            for i in range(1, group_size):
                dev_i = group[i]
                data_gb = allreduce_gb_per_stage
                ttime = self.network.calculate_transfer_time(main_dev, dev_i, data_gb)
                total_time += ttime
                total_data += data_gb

        return (total_time, total_data)

    def _sum_pipeline_comm(
        self,
        assignments: Dict[str, str]
    ) -> Tuple[float, float]:
        """
        Summation of stage->stage data transfer. 
        Similar to edgeshard's logic: 
          - pick "representative device" for each stage 
            (the first layer's assigned device),
          - if stage i's device != stage i+1's device, we do a 
            partial hidden state transfer.

        This is naive, but consistent with the EdgeShard example.
        """
        total_time = 0.0
        total_data = 0.0

        for i in range(len(self.shards) - 1):
            stage_i_layers = self.shards[i]
            stage_j_layers = self.shards[i+1]
            if not stage_i_layers or not stage_j_layers:
                continue

            lid_i = stage_i_layers[0]  # first layer in stage i
            lid_j = stage_j_layers[0]  # first layer in stage i+1

            comp_id_i = self._get_component_id(lid_i)
            comp_id_j = self._get_component_id(lid_j)
            dev_i = assignments.get(comp_id_i, None)
            dev_j = assignments.get(comp_id_j, None)

            if dev_i and dev_j and dev_i != dev_j:
                data_size_gb = self._estimate_hidden_state_size()
                ttime = self.network.calculate_transfer_time(dev_i, dev_j, data_size_gb)
                total_time += ttime
                total_data += data_size_gb

                if self.logger and data_size_gb > 0:
                    self.logger.log_event(
                        "galaxy_pipeline_comm",
                        f"Stage{i}->Stage{i+1} xfer dev={dev_i} -> dev={dev_j}, "
                        f"data={data_size_gb:.6f}GB, time={ttime:.4f}",
                        level=LogLevel.DEBUG
                    )

        return (total_time, total_data)

    def _estimate_hidden_state_size(self) -> float:
        """
        Similar to edgeshard's approach:
        hidden_dim * seq_len * precision_bytes => convert to GB
        """
        D = self.transformer.config.embedding_dim
        seq_len = self.transformer.current_sequence_length
        b = self.transformer.config.precision_bytes
        return (D * seq_len * b) / (1024 ** 3)

    def _get_component_id(self, layer_id: int) -> str:
        """
        Convert a layer_id to the actual layer's component_id
        the same way the simulator does. 
        """
        decoder_layers = getattr(self.transformer, "decoder_layers", [])
        for layer_obj in decoder_layers:
            if layer_obj.layer_id == layer_id:
                return layer_obj.component_id
        return f"layer_{layer_id}"  # fallback
