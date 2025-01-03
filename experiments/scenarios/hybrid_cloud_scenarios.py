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
# File:    experiments/scenarios/hybrid_cloud_scenarios.py
# Description:
#   Defines scenario classes that incorporate both cloud-based high-capacity
#   nodes and edge devices with stricter constraints, modeling realistic
#   hybrid cloud-edge deployments for transformer inference.
#
# ---------------------------------------------------------------------------

"""
Implements scenario classes where a set of powerful cloud nodes collaborate
with resource-limited edge devices to run large-scale transformer inference
tasks, highlighting the performance trade-offs and optimization potential
in hybrid environments.
"""

from typing import Dict, List, Optional, Set
import numpy as np
from datetime import datetime
from collections import defaultdict

from .common import (
    BaseScenario,
    ScenarioResult,
    validate_scenario_requirements,
    collect_scenario_metrics
)
from src.core import Network, Device, Transformer
from src.algorithms import ResourceAwareDistributor
from src.environment import (
    NetworkTopologyGenerator,
    LogNormalDistribution,
    ResourceDistributor,
    WorkloadGenerator,
    HybridCloudEdgeTopology,
    WorkloadType
)
# Make sure SequenceConfig is available:
from src.environment.workload import SequenceConfig


class HybridCloudBaseScenario(BaseScenario):
    """
    Base scenario class for hybrid cloud-edge setups.
    Provides a shared setup_basic_environment() to be reused by
    specialized hybrid cloud scenarios.
    """

    def setup_basic_environment(self) -> None:
        """
        Create the hybrid cloud-edge topology, resource distribution,
        multiple or single workloads, and categorize devices by tier.
        """
        if self.logger:
            self.logger.log_event("setup", "Setting up basic environment for hybrid cloud-edge")

        # 1. Create network topology
        topo_generator = HybridCloudEdgeTopology(self.config.network)
        self.network = topo_generator.generate()

        # 2. Convert config.resources into LogNormalDistribution
        mem_dist = LogNormalDistribution(
            mu=self.config.resources.memory_mu,
            sigma=self.config.resources.memory_sigma,
            min_value=self.config.resources.memory_min,
            max_value=self.config.resources.memory_max
        )
        comp_dist = LogNormalDistribution(
            mu=self.config.resources.compute_mu,
            sigma=self.config.resources.compute_sigma,
            min_value=self.config.resources.compute_min,
            max_value=self.config.resources.compute_max
        )

        resource_distributor = ResourceDistributor(
            num_devices=self.config.network.num_devices,  # e.g. 24
            memory_distribution=mem_dist,
            compute_distribution=comp_dist,
            seed=self.config.resources.seed
        )
        self.device_capabilities = resource_distributor.generate_capabilities()

        # 3. Create Device objects
        self.devices = {}
        for device_id, caps in self.device_capabilities.items():
            self.devices[device_id] = Device(
                device_id=device_id,
                memory_capacity=caps.memory_capacity,
                compute_capacity=caps.compute_capacity,
                is_source=caps.is_source
            )

        # 4. Categorize devices by tier
        #    You can define any logic you want. For demonstration, we pick
        #    the first 4 as cloud, next 8 as regional, rest as edge.
        device_ids_sorted = sorted(self.devices.keys())  # e.g. device_0..device_23
        self.device_tiers = {
            'cloud': device_ids_sorted[:4],
            'regional': device_ids_sorted[4:12],
            'edge': device_ids_sorted[12:]
        }

        # 5. Workload generation
        self.workload_generator = WorkloadGenerator(seed=self.config.workload.seed)
        self.workloads = []

        # Check if multiple or single model types
        if hasattr(self.config.workload, "model_types") and self.config.workload.model_types:
            # multiple
            for mtype in self.config.workload.model_types:
                # define a default SequenceConfig from the first elements
                seq_cfg = SequenceConfig(
                    initial_length=self.config.workload.initial_sequence_lengths[0],
                    num_steps=self.config.workload.generation_steps[0],
                    precision_bytes=self.config.workload.precision_bytes
                )
                w = self.workload_generator.generate_workload(
                    workload_type=WorkloadType[mtype],
                    sequence_config=seq_cfg
                )
                self.workloads.append(w)
        else:
            # single
            # define default SequenceConfig from e.g. first element
            if hasattr(self.config.workload, "model_type"):
                single_type = self.config.workload.model_type
            else:
                single_type = WorkloadType.SMALL  # fallback
            seq_cfg = SequenceConfig(
                initial_length=self.config.workload.initial_sequence_lengths[0],
                num_steps=self.config.workload.generation_steps[0],
                precision_bytes=self.config.workload.precision_bytes
            )
            w = self.workload_generator.generate_workload(
                workload_type=single_type,
                sequence_config=seq_cfg
            )
            self.workloads.append(w)

        # 6. Validate scenario requirements with a sample transformer
        test_transformer = Transformer(self.workloads[0].transformer.config)
        if not validate_scenario_requirements(
            self.config,
            self.network,
            self.devices,
            test_transformer
        ):
            raise ValueError("Scenario requirements not met in hybrid cloud-edge environment setup.")

        # 7. Initialize ResourceAwareDistributor with the first workload's transformer
        self.distributor = ResourceAwareDistributor(
            self.workloads[0].transformer,
            self.network,
            self.devices
        )


class HybridCloudBasicScenario(HybridCloudBaseScenario):
    """
    Basic hybrid cloud-edge scenario testing operation with 24 devices
    across cloud, regional, and edge tiers
    """

    def setup(self) -> None:
        """Set up the hybrid cloud-edge environment"""
        if self.logger:
            self.logger.log_event("setup", "Setting up hybrid cloud-edge basic scenario")
        self.setup_basic_environment()

    def run(self) -> ScenarioResult:
        """Run the basic hybrid cloud-edge scenario"""
        metrics = {
            'resource_metrics': {},
            'communication_metrics': {},
            'performance_metrics': {},
            'tier_metrics': defaultdict(dict)
        }

        try:
            # We'll run each workload in self.workloads
            for idx, workload in enumerate(self.workloads):
                self.distributor.transformer = workload.transformer
                w_metrics = self._run_workload_with_tier_tracking(workload, idx)

                metrics[f'workload_{idx}'] = w_metrics
                self._update_tier_metrics(w_metrics, metrics['tier_metrics'])

            final_metrics = collect_scenario_metrics(
                resource_metrics=metrics['resource_metrics'],
                communication_metrics=metrics['communication_metrics'],
                performance_metrics=metrics['performance_metrics']
            )
            # Optionally attach the tier_metrics or the entire metrics dict
            final_metrics['tier_metrics'] = metrics['tier_metrics']

            return ScenarioResult(
                scenario_name=self.__class__.__name__,
                start_time=datetime.now(),
                end_time=datetime.now(),
                metrics=final_metrics,
                success=True
            )

        except Exception as e:
            if self.logger:
                self.logger.log_error("scenario_error", str(e))
            return ScenarioResult(
                scenario_name=self.__class__.__name__,
                start_time=datetime.now(),
                end_time=datetime.now(),
                metrics={},
                success=False,
                error=str(e)
            )

    def _update_tier_metrics(
        self,
        workload_metrics: Dict,
        tier_metrics: Dict
    ) -> None:
        """
        Given the metrics from one run of a workload,
        aggregate stats by tier for the final scenario result.
        """
        for tier, devices in self.device_tiers.items():
            # Example: compute average compute utilization across all steps & devices
            util_vals = []
            # 'resource_metrics' is step -> {dev_id: {...}} in workload_metrics
            for step_data in workload_metrics['resource_metrics'].values():
                for d_id in devices:
                    if d_id in step_data:
                        u = step_data[d_id].get('compute_utilization', 0.0)
                        util_vals.append(u)

            if util_vals:
                avg_util = float(np.mean(util_vals))
            else:
                avg_util = 0.0

            # store in tier_metrics e.g. tier_metrics[tier]['utilization'] = ...
            if 'utilization' not in tier_metrics[tier]:
                tier_metrics[tier]['utilization'] = []
            tier_metrics[tier]['utilization'].append(avg_util)

            # Example: track number of assigned components
            if 'assignments' not in tier_metrics[tier]:
                tier_metrics[tier]['assignments'] = 0

            # each step has 'component_assignments'
            if 'component_assignments' in workload_metrics:
                for step_idx, cassign in workload_metrics['component_assignments'].items():
                    # cassign is a dict of comp -> device
                    for comp, dev_id in cassign.items():
                        if dev_id in devices:
                            tier_metrics[tier]['assignments'] += 1

    def cleanup(self) -> None:
        """Clean up resources"""
        if self.logger:
            self.logger.log_event("cleanup", "Cleaning up hybrid cloud basic scenario")


    def _run_workload_with_tier_tracking(
        self,
        workload,
        workload_idx: int
    ) -> Dict:
        """Run a workload while tracking tier-specific metrics"""
        wmetrics = {
            'resource_metrics': {},
            'performance_metrics': {},
            'component_assignments': {}
        }

        for step in range(workload.sequence_config.num_steps):
            assignment_result = self.distributor.compute_assignment(
                generation_step=step,
                previous_assignments=wmetrics.get('previous_assignments'),
                previous_cache=wmetrics.get('previous_cache')
            )

            if not assignment_result.is_feasible:
                raise RuntimeError(
                    f"Infeasible assignment at step {step} for workload {workload_idx}"
                )

            # Record resource usage
            wmetrics['resource_metrics'][step] = assignment_result.resource_usage
            wmetrics['component_assignments'][step] = assignment_result.component_assignments
            wmetrics['previous_assignments'] = assignment_result.component_assignments
            wmetrics['previous_cache'] = assignment_result.cache_assignments

            # Performance
            wmetrics['performance_metrics'][step] = {
                'latency': assignment_result.estimated_latency,
                'step': step
            }

        return wmetrics


class HybridCloudTierBalancingScenario(HybridCloudBaseScenario):
    """
    Tests workload balancing across cloud, regional, and edge tiers
    """

    def setup(self) -> None:
        """Set up tier balancing test environment"""
        if self.logger:
            self.logger.log_event("setup", "Setting up hybrid cloud tier balancing scenario")
        self.setup_basic_environment()

        # Example: Tier tracker or extra data if needed
        self.tier_tracker = {
            tier: {
                'compute_usage': [],
                'memory_usage': [],
                'component_assignments': defaultdict(int)
            }
            for tier in ['cloud', 'regional', 'edge']
        }

    def run(self) -> ScenarioResult:
        """Run tier balancing analysis"""
        metrics = {
            'resource_metrics': {},
            'communication_metrics': {},
            'performance_metrics': {},
            'tier_balance_metrics': {}
        }

        try:
            for workload in self.workloads:
                self.distributor.transformer = workload.transformer
                tmetrics = self._run_tier_balanced_workload(workload)

                metrics['tier_balance_metrics'][workload.workload_type.name] = \
                    self._analyze_tier_balance(tmetrics)

            final_metrics = collect_scenario_metrics(
                resource_metrics=metrics['resource_metrics'],
                communication_metrics=metrics['communication_metrics'],
                performance_metrics=metrics['performance_metrics']
            )
            final_metrics['tier_balance_metrics'] = metrics['tier_balance_metrics']

            return ScenarioResult(
                scenario_name=self.__class__.__name__,
                start_time=datetime.now(),
                end_time=datetime.now(),
                metrics=final_metrics,
                success=True
            )

        except Exception as e:
            if self.logger:
                self.logger.log_error("scenario_error", str(e))
            return ScenarioResult(
                scenario_name=self.__class__.__name__,
                start_time=datetime.now(),
                end_time=datetime.now(),
                metrics={},
                success=False,
                error=str(e)
            )

    def cleanup(self) -> None:
        """Clean up resources"""
        if self.logger:
            self.logger.log_event("cleanup", "Cleaning up hybrid cloud tier balancing scenario")

    def _run_tier_balanced_workload(self, workload) -> Dict:
        """
        Run the workload, collecting data about tier usage.
        For example, we track compute usage by tier at each step.
        """
        tmetrics = defaultdict(lambda: defaultdict(list))
        # so tmetrics[tier]['compute_usage'] -> list
        # tmetrics[tier]['memory_usage'] -> list
        # tmetrics[tier]['component_assignments'] -> defaultdict(int), etc.

        for step in range(workload.sequence_config.num_steps):
            assignment_result = self.distributor.compute_assignment(
                generation_step=step,
                previous_assignments=tmetrics.get('previous_assignments'),
                previous_cache=tmetrics.get('previous_cache')
            )

            # track usage
            for tier, dev_ids in self.device_tiers.items():
                cu_list = []
                mu_list = []
                for d_id in dev_ids:
                    usage = assignment_result.resource_usage.get(d_id, {})
                    cu = usage.get('compute_utilization', 0.0)
                    mu = usage.get('memory_utilization', 0.0)
                    cu_list.append(cu)
                    mu_list.append(mu)
                avg_cu = float(np.mean(cu_list)) if cu_list else 0.0
                avg_mu = float(np.mean(mu_list)) if mu_list else 0.0
                tmetrics[tier]['compute_usage'].append(avg_cu)
                tmetrics[tier]['memory_usage'].append(avg_mu)

                # track assignment distribution
                for comp_id, dev_id in assignment_result.component_assignments.items():
                    if dev_id in dev_ids:
                        tmetrics[tier]['component_assignments'][comp_id] += 1

            tmetrics['previous_assignments'] = assignment_result.component_assignments
            tmetrics['previous_cache'] = assignment_result.cache_assignments

        return tmetrics

    def _analyze_tier_balance(self, tmetrics: Dict) -> Dict:
        """
        Perform some analysis on usage distribution and assignment patterns.
        """
        results = {}
        for tier, data in tmetrics.items():
            if tier in ['previous_assignments', 'previous_cache']:
                continue
            # data['compute_usage'], data['memory_usage'], data['component_assignments']
            if isinstance(data, dict):
                cu_vals = data.get('compute_usage', [])
                mu_vals = data.get('memory_usage', [])
                comp_dict = data.get('component_assignments', {})

                results[tier] = {
                    'average_compute': float(np.mean(cu_vals)) if cu_vals else 0.0,
                    'average_memory': float(np.mean(mu_vals)) if mu_vals else 0.0,
                    'assignment_distribution': dict(comp_dict),
                    'balance_index': self._calculate_balance_index(cu_vals)
                }
        return results

    def _calculate_balance_index(self, utilization_values: List[float]) -> float:
        """
        Calculate some measure of how balanced the usage is.
        E.g., a ratio of standard deviation to mean.
        """
        if not utilization_values:
            return 0.0
        mean_val = float(np.mean(utilization_values))
        if mean_val == 0:
            return 0.0
        return float(np.std(utilization_values) / mean_val)


class HybridCloudLatencyScenario(HybridCloudBaseScenario):
    """
    Tests latency characteristics across different tiers and bandwidths
    """

    def setup(self) -> None:
        """Set up latency test environment"""
        if self.logger:
            self.logger.log_event("setup", "Setting up hybrid cloud latency scenario")
        self.setup_basic_environment()

        # Initialize latency tracking if needed
        self.latency_tracker = {
            'intra_tier': defaultdict(list),
            'inter_tier': defaultdict(list),
            'edge_to_cloud': defaultdict(list)
        }

    def run(self) -> ScenarioResult:
        """Run latency analysis"""
        metrics = {
            'resource_metrics': {},
            'communication_metrics': {},
            'performance_metrics': {},
            'latency_analysis': {}
        }

        try:
            # Example: run each workload
            for workload in self.workloads:
                self.distributor.transformer = workload.transformer
                lat_metrics = self._run_latency_analysis(workload)
                metrics['latency_analysis'][workload.workload_type.name] = \
                    self._analyze_latency_patterns(lat_metrics)

            final_metrics = collect_scenario_metrics(
                resource_metrics=metrics['resource_metrics'],
                communication_metrics=metrics['communication_metrics'],
                performance_metrics=metrics['performance_metrics']
            )
            final_metrics['latency_analysis'] = metrics['latency_analysis']

            return ScenarioResult(
                scenario_name=self.__class__.__name__,
                start_time=datetime.now(),
                end_time=datetime.now(),
                metrics=final_metrics,
                success=True
            )

        except Exception as e:
            if self.logger:
                self.logger.log_error("scenario_error", str(e))
            return ScenarioResult(
                scenario_name=self.__class__.__name__,
                start_time=datetime.now(),
                end_time=datetime.now(),
                metrics={},
                success=False,
                error=str(e)
            )

    def cleanup(self) -> None:
        """Clean up resources"""
        if self.logger:
            self.logger.log_event("cleanup", "Cleaning up hybrid cloud latency scenario")

    def _run_latency_analysis(self, workload) -> Dict:
        """
        For each generation step, gather assignments, measure comm latencies, etc.
        Return a structure from which we can compute stats (mean, p95, etc.).
        """
        lat_metrics = {
            'communication_latency': defaultdict(list),
            'processing_latency': defaultdict(list),
            'total_latency': defaultdict(list),
            'previous_assignments': {},
            'previous_cache': {}
        }

        for step in range(workload.sequence_config.num_steps):
            assignment_result = self.distributor.compute_assignment(
                generation_step=step,
                previous_assignments=lat_metrics['previous_assignments'],
                previous_cache=lat_metrics['previous_cache']
            )

            # track latencies
            self._track_latencies(step, assignment_result, lat_metrics)

            lat_metrics['previous_assignments'] = assignment_result.component_assignments
            lat_metrics['previous_cache'] = assignment_result.cache_assignments

        return lat_metrics

    def _track_latencies(
        self,
        step: int,
        assignment_result,
        lat_metrics
    ) -> None:
        """
        Example: track communication latencies by tier transitions, etc.
        """
        for comp_id, dev_id in assignment_result.component_assignments.items():
            source_tier = self._get_device_tier(dev_id)
            # Suppose we track dependencies:
            for dep_id in self._get_component_dependencies(comp_id):
                if dep_id in assignment_result.component_assignments:
                    dep_dev = assignment_result.component_assignments[dep_id]
                    dep_tier = self._get_device_tier(dep_dev)
                    # compute data size
                    data_size = self._estimate_transfer_size(comp_id, dep_id)
                    # compute link time
                    link_time = self.network.calculate_transfer_time(
                        dev_id, dep_dev, data_size
                    )

                    if source_tier == dep_tier:
                        lat_metrics['communication_latency']['intra_tier'].append(link_time)
                    else:
                        lat_metrics['communication_latency']['inter_tier'].append(link_time)
                        if source_tier == 'edge' and dep_tier == 'cloud':
                            lat_metrics['communication_latency']['edge_to_cloud'].append(link_time)

    def _analyze_latency_patterns(self, lat_metrics: Dict) -> Dict:
        """
        Summarize stats for the recorded latencies in lat_metrics['communication_latency'].
        """
        comm_lat = lat_metrics['communication_latency']
        analysis = {}

        for key in ['intra_tier', 'inter_tier', 'edge_to_cloud']:
            arr = comm_lat[key]
            if arr:
                arr_np = np.array(arr, dtype=float)
                analysis[f"{key}_mean"] = float(np.mean(arr_np))
                analysis[f"{key}_std"] = float(np.std(arr_np))
                analysis[f"{key}_p95"] = float(np.percentile(arr_np, 95))
            else:
                analysis[f"{key}_mean"] = 0.0
                analysis[f"{key}_std"] = 0.0
                analysis[f"{key}_p95"] = 0.0

        return analysis

    def _get_device_tier(self, device_id: str) -> str:
        """
        Return 'cloud', 'regional', or 'edge' if device_id is in that group,
        else 'unknown'.
        """
        for tier, dev_ids in self.device_tiers.items():
            if device_id in dev_ids:
                return tier
        return 'unknown'

    def _get_component_dependencies(self, component_id: str) -> List[str]:
        """
        Example logic for dependencies (similar to your code).
        """
        dependencies = []
        if component_id.startswith('head_'):
            dependencies.extend([
                f'cache_{component_id}',
                'projection'
            ])
        elif component_id == 'projection':
            # depends on all heads
            heads_count = self.distributor.transformer.config.num_heads
            dependencies.extend([f'head_{i}' for i in range(heads_count)])
        elif component_id == 'ffn':
            dependencies.append('projection')
        return dependencies

    def _estimate_transfer_size(
        self,
        comp_id: str,
        dep_id: str
    ) -> float:
        """
        Example logic for data size in GB.
        """
        transformer = self.distributor.transformer
        seq_len = transformer.current_sequence_length
        head_dim = transformer.config.head_dim
        emb_dim = transformer.config.embedding_dim
        prec_bytes = transformer.config.precision_bytes

        if dep_id.startswith('cache_'):
            # K/V cache
            return (seq_len * head_dim * prec_bytes * 2) / (1024**3)
        elif comp_id == 'projection' and dep_id.startswith('head_'):
            return (seq_len * head_dim * prec_bytes) / (1024**3)
        elif comp_id == 'ffn' and dep_id == 'projection':
            return (seq_len * emb_dim * prec_bytes) / (1024**3)
        return 0.0
