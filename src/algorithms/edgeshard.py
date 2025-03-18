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
# File:    src/algorithms/edgeshard.py
# Description:
#   Implements an "EdgeShard"-style distributor for Transformer inference,
#   approximating the main ideas of the EdgeShard approach described in:
#
#       Zhang, H., Huang, P., Wang, D., & Li, M. (2023).
#       "EdgeShard: Empowering Resource-Constrained Devices via Hybrid 
#       Model-Parallel Inference."
#       https://arxiv.org/pdf/2405.14371
#
#   EdgeShard operates at the layer/block level rather than the individual
#   head/FFN components. It uses a static pipeline-parallel (layer-wise)
#   distribution of the Transformer’s blocks across multiple devices, each
#   hosting a contiguous "shard" of layers. During autoregressive inference,
#   tokens (or micro-batches) flow sequentially from shard to shard (pipeline
#   parallelism), reducing device idling while still being simpler to manage
#   than fully flexible or head-level partitions.
#
#   In the original EdgeShard paper, the approach can also incorporate 
#   tensor parallelism (splitting each matrix multiplication across multiple
#   devices in parallel). For simplicity, this file focuses primarily on the
#   pipeline-parallel partition—i.e., each device is responsible for a 
#   contiguous portion of the Transformer (e.g., layers 1..k on device1, 
#   layers k+1..m on device2, etc.). One could extend this to partial 
#   tensor splitting as well, but the fundamental logic remains the same.
#
#   Key points:
#     - We treat each decoder layer (or block) as a single "component"
#       to be assigned to devices (instead of the fine-grained approach
#       that places individual attention heads or FFN modules).
#     - A block and its key-value (KV) cache remain on the same device
#       for the entire inference process (ephemeral=False in the simulator).
#     - The assignment is computed once (e.g., uniform partition or
#       a memory-based partition) and remains static throughout 
#       autoregressive decoding.
#     - When simulating per-token inference, the pipeline scheduling can
#       be approximated with a concurrency-based or pipeline-based latency
#       function. Here, we use the simulator’s existing concurrency-based
#       method (`compute_3phase_latency`), but note that a more advanced
#       pipeline micro-batching model could give more faithful EdgeShard 
#       timings if desired.
#
#   This implementation closely mirrors the static, layer-wise partitioning
#   portion of the EdgeShard approach. It does not fully replicate all
#   advanced scheduling optimizations (e.g., bubble filling, advanced 
#   micro-batching, or partial tensor slicing), but it provides a 
#   representative comparison for how a "layer-based, pipeline parallel"
#   scheme performs relative to a "head-based, resource-adaptive" scheme
#   (e.g., the ResourceAwareDistributor).
#
# -----------------------------------------------------------------------------

"""
EdgeShardDistributor:
---------------------

In this module, we introduce the EdgeShardDistributor class, which implements
a static, pipeline-like block-level partition strategy for Transformer inference.

1) **Layer Partitioning**:
   We divide the entire Transformer’s decoder into `num_shards` contiguous chunks 
   (shards). For example, if the model has 24 layers and we specify `num_shards=3`,
   we might allocate:
       - Layers 0..7   -> Device A  (first shard)
       - Layers 8..15  -> Device B  (second shard)
       - Layers 16..23 -> Device C  (third shard)
   The exact partition depends on memory feasibility and device availability.

2) **Static Assignment**:
   Once assigned, these shards remain on their designated devices for the duration
   of autoregressive inference. The K/V caches associated with each layer are also
   placed on the same device. This approach mirrors the primary pipeline mechanism
   in EdgeShard.

3) **Inference Execution**:
   At runtime, input tokens (or partial states) pass from shard 1 -> shard 2 -> ...
   -> shard N in a pipeline. We approximate this pipeline concurrency by using the
   simulator’s `compute_3phase_latency` for concurrency-based scheduling. For more
   accurate pipeline micro-batching, you could add a custom pipeline scheduling 
   routine if desired.

4) **Comparison**:
   Because EdgeShard uses a coarser granularity (layer-level) and static partition,
   it typically reduces overhead from frequent migrations or fine-grained 
   assignment. However, it can be less flexible and may not fully utilize device 
   resources in highly heterogeneous environments. The ResourceAwareDistributor’s 
   approach, by contrast, can adapt each step and even assign individual heads 
   or caches to different devices.

This file can thus be used to compare an EdgeShard-style approach side-by-side 
with other distribution algorithms (Greedy, RoundRobin, Static, DynamicMigration,
ResourceAware) within the Transformer Inference Simulator.
"""

import traceback
from typing import Dict, Optional, Tuple

from src.utils.logging import SimulationLogger, LogLevel
from ..core import Device, Network, Transformer, TransformerComponent
from .baselines import BaseDistributor
from .utils import validate_assignment, compute_3phase_latency
from .resource_aware import AssignmentResult


class EdgeShardDistributor(BaseDistributor):
    """
    Implements a block-level, pipeline-parallel style distribution analogous
    to the "EdgeShard" approach, partitioning the Transformer’s layers
    (decoder blocks) into contiguous shards.

    :param transformer: The Transformer model instance.
    :param network: The underlying network (devices + links).
    :param devices: Dictionary of device_id -> Device objects.
    :param logger: Optional logger for debug and error logging.
    :param num_shards: How many shards (contiguous sets of layers) to create.
    :param shard_map: If provided, a user-specified mapping {layer_index -> device_id}.
                      If None, the `_build_partition()` method tries to compute one.
    
    Usage:
    ------
    1) Initialize EdgeShardDistributor with a given `num_shards` or a custom `shard_map`.
    2) On each generation step, call `compute_assignment(step, prev_assignments, prev_cache)`.

       *But if you want to allocate once and keep it the same*, see the 
       logic below in `compute_assignment()` that checks if shards are 
       already assigned. If so and it's not the first step, we skip 
       reallocation and simply recalculate concurrency-based pipeline 
       latency (optional).
       
    Note that we treat each layer as non-ephemeral for the entire inference.
    The K/V cache for that layer also remains on the same device. 
    """

    def __init__(
        self,
        transformer: Transformer,
        network: Network,
        devices: Dict[str, Device],
        logger: Optional[SimulationLogger] = None,
        num_shards: int = 2,
        shard_map: Optional[Dict[int, str]] = None
    ):
        super().__init__(transformer, network, devices, logger)
        self.num_shards = num_shards
        self.shard_map = shard_map or {}
        # Flag to indicate whether we've already assigned shards.
        self._shards_assigned = False

        # Stores the entire result from the first time we do a successful assignment:
        self._previous_result: Optional[AssignmentResult] = None

    def compute_assignment(
        self,
        generation_step: int,
        previous_assignments: Optional[Dict[str, str]] = None,
        previous_cache: Optional[Dict[str, str]] = None
    ) -> AssignmentResult:
        """
        Distribute each decoder layer (block) to its designated device.

        Steps:
        ------
        0) If shards have already been assigned, and we are not on step=0,
           skip re-allocation. Just deallocate ephemeral usage, recalc concurrency
           if desired, and return the stored result from the first assignment.
           
        1) Otherwise:
           - Deallocate ephemeral usage from previous step,
           - Build a shard_map (if needed),
           - Allocate each layer on the assigned device (non-ephemeral),
           - Validate feasibility,
           - Compute concurrency-based latency,
           - Save & return the result.
        """
        # === (A) If we've already assigned shards and it's not step 0, skip re-allocation ===
        #     This ensures we do not double-allocate the same layers each step.
        if self._shards_assigned and generation_step > 0 and self._previous_result is not None:
            # Deallocate ephemeral from the last step (but keep the static shards).
            self._reset_device_states_for_step()

            # Optionally re-calc concurrency-based pipeline latency for step>0
            # because the token length might change, or we might want an updated time.
            try:
                total_latency, comm_time, data_gb = self._compute_latency_and_comm(
                    self._previous_result.component_assignments,
                    self._previous_result.cache_assignments,
                    generation_step
                )
            except Exception as ex:
                usage = self._get_resource_usage()
                err_str = f"EdgeShard: concurrency-based latency exception: {str(ex)}"
                if self.logger:
                    self.logger.log_error("edgeshard_latency_exception", err_str)
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
            # Return the original assignments, but updated pipeline concurrency data
            return AssignmentResult(
                component_assignments=self._previous_result.component_assignments,
                cache_assignments=self._previous_result.cache_assignments,
                estimated_latency=total_latency,
                resource_usage=usage,
                is_feasible=True,
                migrations=[],   # static approach => no new migrations
                communication_time=comm_time,
                data_transferred_gb=data_gb
            )

        # === (B) The first time we allocate (or if shard_map was not set yet) ===
        # 1) Deallocate ephemeral from last step
        self._reset_device_states_for_step()

        if self.logger:
            self.logger.log_event(
                "edgeshard_compute",
                f"[EdgeShardDistributor] step={generation_step}, shards_assigned={self._shards_assigned}",
                level=LogLevel.DEBUG
            )

        # 2) Possibly build shard map if not yet assigned
        if not self.shard_map and not self._shards_assigned:
            self.shard_map = self._build_partition()
            self._shards_assigned = True

        # 3) Allocate layers
        assignments, cache_assignments, feasible = self._allocate_shards(generation_step)
        if not feasible:
            usage = self._get_resource_usage()
            return AssignmentResult(
                component_assignments=assignments,
                cache_assignments=cache_assignments,
                estimated_latency=float('inf'),
                resource_usage=usage,
                is_feasible=False,
                error="EdgeShard: layer allocation infeasible"
            )

        # 4) Validate
        is_ok = validate_assignment(
            assignments, cache_assignments,
            self.transformer, self.devices,
            self.network, generation_step
        )
        if not is_ok:
            usage = self._get_resource_usage()
            return AssignmentResult(
                component_assignments=assignments,
                cache_assignments=cache_assignments,
                estimated_latency=float('inf'),
                resource_usage=usage,
                is_feasible=False,
                error="EdgeShard: assignment validation failed"
            )

        # 5) Compute concurrency-based pipeline latency
        try:
            total_latency, comm_time, data_gb = self._compute_latency_and_comm(
                assignments, cache_assignments, generation_step
            )
        except Exception as ex:
            usage = self._get_resource_usage()
            err_str = f"EdgeShard: concurrency-based latency exception: {str(ex)}"
            if self.logger:
                self.logger.log_error("edgeshard_latency_exception", err_str)
                traceback.print_exc()
            return AssignmentResult(
                component_assignments=assignments,
                cache_assignments=cache_assignments,
                estimated_latency=float('inf'),
                resource_usage=usage,
                is_feasible=False,
                error=err_str
            )

        usage = self._get_resource_usage()

        result = AssignmentResult(
            component_assignments=assignments,
            cache_assignments=cache_assignments,
            estimated_latency=total_latency,
            resource_usage=usage,
            is_feasible=True,
            migrations=[],   # static approach => no migrations
            communication_time=comm_time,
            data_transferred_gb=data_gb
        )

        # Save this result so subsequent steps can simply reuse it
        self._previous_result = result

        return result

    def _build_partition(self) -> Dict[int, str]:
        """
        Create a naive partition of decoder layers into `num_shards` contiguous shards.

        Implementation details:
        -----------------------
        - We retrieve the transformer's decoder layers (or blocks).
        - We split them in contiguous chunks. 
        - We assign each chunk to a device in a round-robin or naive manner.
        - Return a dict mapping: layer_index -> device_id

        In practice, you might want a more advanced memory or load-based 
        partitioning to balance usage across devices.
        """
        partition_map = {}
        device_ids = list(self.devices.keys())
        if not device_ids:
            return partition_map

        decoder_layers = getattr(self.transformer, "decoder_layers", [])
        num_layers = len(decoder_layers)
        if num_layers == 0:
            return partition_map

        chunk_size = max(num_layers // self.num_shards, 1)

        for i, layer_obj in enumerate(decoder_layers):
            layer_id = layer_obj.layer_id
            shard_index = i // chunk_size
            if shard_index >= self.num_shards:
                shard_index = self.num_shards - 1
            dev_id = device_ids[shard_index % len(device_ids)]
            partition_map[layer_id] = dev_id

        return partition_map

    def _allocate_shards(
        self,
        generation_step: int
    ) -> Tuple[Dict[str, str], Dict[str, str], bool]:
        """
        Allocate memory & compute for each decoder layer on its assigned device.

        :return: (assignments, cache_assignments, feasible) 
                 where `assignments` is a dict {component_id -> device_id},
                 `cache_assignments` is for K/V caches, 
                 and feasible is True if all layers fit successfully.
        """
        assignments = {}
        cache_assignments = {}
        feasible = True

        decoder_layers = getattr(self.transformer, "decoder_layers", [])
        for layer_obj in decoder_layers:
            layer_idx = layer_obj.layer_id
            comp_id = layer_obj.component_id

            if layer_idx not in self.shard_map:
                feasible = False
                break

            device_id = self.shard_map[layer_idx]
            if device_id not in self.devices:
                feasible = False
                break

            dev = self.devices[device_id]

            mem_req = layer_obj.compute_memory_requirements(self.transformer.current_sequence_length)
            flops_req = layer_obj.compute_flops(self.transformer.current_sequence_length)

            # KV cache
            cache_req = 0.0
            if hasattr(layer_obj, "compute_cache_memory"):
                cache_req = layer_obj.compute_cache_memory(generation_step)

            # Each layer is persistent (ephemeral=False)
            ephemeral_flag = False

            ok_main = dev.allocate_resources(
                comp_id,
                mem_req,
                flops_req,
                ephemeral=ephemeral_flag
            )
            if not ok_main:
                feasible = False
                break

            ok_cache = True
            if cache_req > 0.0:
                ok_cache = dev.allocate_resources(
                    comp_id + "_cache",
                    cache_req,
                    0.0,
                    ephemeral=ephemeral_flag
                )
                if not ok_cache:
                    feasible = False
                    break

            assignments[comp_id] = device_id
            if cache_req > 0.0:
                cache_assignments[comp_id] = device_id

            if self.logger and (ok_main and ok_cache):
                self.logger.log_event(
                    "edgeshard_layer_alloc",
                    f"Assigned layer {comp_id} (layer_idx={layer_idx}) "
                    f"to device={device_id}, mem_req={mem_req:.2f}, flops={flops_req:.2f}",
                    level=LogLevel.DEBUG
                )

            if not (ok_main and ok_cache):
                break

        return assignments, cache_assignments, feasible

    def _compute_latency_and_comm(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> Tuple[float, float, float]:
        """
        Approximate pipeline-parallel inference time using concurrency logic.
        We reuse the simulator’s concurrency-based 3-phase approach 
        (`compute_3phase_latency`).

        :return: (total_latency, communication_time, data_gb_transferred)
        """
        total_latency = compute_3phase_latency(
            self.transformer,
            self.devices,
            self.network,
            assignments,
            generation_step,
            concurrency_mode="sum"  
        )
        comm_time, data_gb = self._sum_communication(assignments, generation_step)
        return (total_latency, comm_time, data_gb)

    def _sum_communication(
        self,
        assignments: Dict[str, str],
        generation_step: int
    ) -> Tuple[float, float]:
        """
        Very simplified approach to summing data transfer times for 
        partial hidden states passed from one layer to the next.
        """
        total_comm_time = 0.0
        total_data_gb = 0.0

        decoder_layers = getattr(self.transformer, "decoder_layers", [])
        sorted_layers = sorted(decoder_layers, key=lambda x: x.layer_id)

        for i in range(len(sorted_layers) - 1):
            layer_curr = sorted_layers[i]
            layer_next = sorted_layers[i+1]
            dev_curr = assignments.get(layer_curr.component_id, None)
            dev_next = assignments.get(layer_next.component_id, None)
            if dev_curr and dev_next and dev_curr != dev_next:
                data_size_gb = self._estimate_hidden_state_size()
                total_data_gb += data_size_gb
                ttime = self.network.calculate_transfer_time(dev_curr, dev_next, data_size_gb)
                total_comm_time += ttime
                if self.logger and data_size_gb > 0:
                    self.logger.log_event(
                        "edgeshard_comm",
                        f"Transfer from layer {layer_curr.component_id} (dev={dev_curr}) "
                        f"to layer {layer_next.component_id} (dev={dev_next}), "
                        f"data_size={data_size_gb:.6f}GB, ttime={ttime:.4f}",
                        level=LogLevel.DEBUG
                    )
        return (total_comm_time, total_data_gb)

    def _estimate_hidden_state_size(self) -> float:
        """
        Rough estimate of partial hidden state size in GB.
        """
        D = self.transformer.config.embedding_dim
        seq_len = self.transformer.current_sequence_length
        b = self.transformer.config.precision_bytes
        return (D * seq_len * b) / (1024 ** 3)
