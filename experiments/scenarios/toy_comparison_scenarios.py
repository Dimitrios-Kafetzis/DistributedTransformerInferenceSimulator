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
# File: experiments/scenarios/toy_comparison_scenarios.py
#
#   A small toy scenario that compares multiple baseline algorithms
#   (Greedy, RoundRobin, Static, DynamicMigration) against ResourceAware,
#   in a 3-device environment with random background usage and
#   step-by-step communication & migration metrics.

import random
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional
import pprint

from .common import (
    ScenarioResult,
    validate_scenario_requirements,
    collect_scenario_metrics
)
from .baseline_scenarios import BaselineComparisonBaseScenario

# 5 algorithms (including static, dynamic)
from src.algorithms import (
    GreedyDistributor,
    RoundRobinDistributor,
    StaticDistributor,
    DynamicMigrationDistributor,
    ResourceAwareDistributor,
)
from src.core import Network, Device, Transformer
from src.environment import (
    ResourceDistributor,
    LogNormalDistribution,
    WorkloadGenerator
)
from src.utils import SimulationLogger, SimulationConfig, LogLevel


class ToyComparisonScenario(BaselineComparisonBaseScenario):
    """
    Compare all 5 algorithms (greedy, round_robin, static, dynamic, resource_aware)
    in a small environment. Each step:
      1) Add random background usage
      2) For each algorithm, compute_assignment(...)
      3) Collect resource usage, communication overhead, migrations, latency
      4) Also store distribution info
      5) Aggregate results => final summary
    """

    def setup(self) -> None:
        if self.logger:
            self.logger.log_event("setup", "Setting up ToyComparisonScenario environment", level=LogLevel.DEBUG)

        self._create_network_and_devices()

        # 2) Create a small workload
        self.workload_generator = WorkloadGenerator(seed=self.config.workload.seed)
        self.workload = self.workload_generator.generate_workload(
            workload_type=self.config.workload.model_type,
            sequence_config=None
        )
        if self.workload.sequence_config:
            self.workload.transformer.current_sequence_length = self.workload.sequence_config.initial_length
        else:
            self.workload.transformer.current_sequence_length = 32

        # 3) Validate scenario
        test_transformer = self.workload.transformer
        if not validate_scenario_requirements(self.config, self.network, self.devices, test_transformer):
            raise ValueError("Scenario requirements not met in ToyComparisonScenario setup")

        # 4) Instantiate all 5
        self.greedy = GreedyDistributor(test_transformer, self.network, self.devices, self.logger)
        self.round_robin = RoundRobinDistributor(test_transformer, self.network, self.devices, self.logger)
        self.static = StaticDistributor(test_transformer, self.network, self.devices, self.logger)
        self.dynamic = DynamicMigrationDistributor(test_transformer, self.network, self.devices, 
                                                   memory_threshold=self.config.algorithm.migration_threshold, 
                                                   compute_threshold=self.config.algorithm.migration_threshold, 
                                                   logger=self.logger)
        self.resource_aware = ResourceAwareDistributor(test_transformer, self.network, self.devices, self.logger)

        self.algorithms = {
            "greedy": self.greedy,
            "round_robin": self.round_robin,
            "static": self.static,
            "dynamic": self.dynamic,
            "resource_aware": self.resource_aware,
        }

        if self.logger:
            self.logger.log_event(
                "setup",
                f"ToyComparisonScenario setup complete with {len(self.devices)} devices.",
                level=LogLevel.DEBUG
            )

    def _create_network_and_devices(self):
        mem_dist = LogNormalDistribution(
            mu=self.config.resources.memory_mu,
            sigma=self.config.resources.memory_sigma,
            min_value=self.config.resources.memory_min,
            max_value=self.config.resources.memory_max,
        )
        comp_dist = LogNormalDistribution(
            mu=self.config.resources.compute_mu,
            sigma=self.config.resources.compute_sigma,
            min_value=self.config.resources.compute_min,
            max_value=self.config.resources.compute_max,
        )
        rd = ResourceDistributor(
            num_devices=self.config.network.num_devices,
            memory_distribution=mem_dist,
            compute_distribution=comp_dist,
            seed=self.config.resources.seed
        )
        caps = rd.generate_capabilities()
        self.devices = {}
        for d_id, c in caps.items():
            self.devices[d_id] = Device(
                device_id=d_id,
                memory_capacity=c.memory_capacity,
                compute_capacity=c.compute_capacity,
                is_source=c.is_source
            )
        self.network = Network()
        random.seed(self.config.network.seed)
        for d1 in self.devices:
            self.network.add_device(d1)
        for d1 in self.devices:
            for d2 in self.devices:
                if d1 != d2:
                    bw = random.uniform(self.config.network.min_bandwidth, self.config.network.max_bandwidth)
                    self.network.add_link(d1, d2, bandwidth=bw)

    def run(self) -> ScenarioResult:
        self.logger.log_event("run", "ToyComparisonScenario begins running.", level=LogLevel.DEBUG)
        if not self.workload or not self.workload.transformer:
            raise RuntimeError("No valid workload or transformer in ToyComparisonScenario")

        default_steps = 5
        steps_val = getattr(self.config.workload, 'generation_steps', None)
        if isinstance(steps_val, list) and len(steps_val) > 0:
            total_steps = steps_val[0]
        elif isinstance(steps_val, int) and steps_val > 0:
            total_steps = steps_val
        else:
            total_steps = default_steps

        from collections import defaultdict
        resource_metrics = defaultdict(dict)
        communication_metrics = defaultdict(dict)
        migration_metrics = defaultdict(dict)
        performance_metrics = defaultdict(dict)

        # NEW: store distribution info (which device each block + cache is on)
        distribution_metrics = defaultdict(dict)  # step -> {algo: {"components": {}, "caches": {}}}

        # comparison metrics
        comparison_metrics = {
            "latency": defaultdict(list),
            "resource_utilization": defaultdict(list),
            "communication_overhead": defaultdict(list),
            "migration_counts": defaultdict(int)
        }

        total_latency = {a: 0.0 for a in self.algorithms}
        feasible_counts = {a: 0 for a in self.algorithms}
        total_migrations = {a: 0 for a in self.algorithms}

        prev_assignments = {a: None for a in self.algorithms}
        prev_cache = {a: None for a in self.algorithms}

        pp = pprint.PrettyPrinter(indent=2, width=120)

        try:
            for step in range(total_steps):
                if self.logger:
                    self.logger.log_event("debug", f"=== Step {step} start ===", level=LogLevel.DEBUG)

                # random usage
                self._inject_random_background_usage()
                if self.logger:
                    self.logger.log_event("debug", f"Step {step}: Random background usage injected.", level=LogLevel.DEBUG)

                for algo_name, distributor in self.algorithms.items():
                    if self.logger:
                        self.logger.log_event(
                            "debug", f"Step {step}, Algo='{algo_name}': Starting compute_assignment...",
                            level=LogLevel.DEBUG
                        )

                    usage_dict = {}
                    algo_latency = float('inf')
                    is_feasible = False
                    migrations = 0
                    comm_time = 0.0
                    data_tx = 0.0  # ADDED: track data transferred

                    try:
                        result = distributor.compute_assignment(
                            generation_step=step,
                            previous_assignments=prev_assignments[algo_name],
                            previous_cache=prev_cache[algo_name]
                        )
                        usage_dict = result.resource_usage
                        algo_latency = result.estimated_latency
                        is_feasible = result.is_feasible
                        if result.migrations:
                            migrations = len(result.migrations)
                        comm_time = result.communication_time  # newly added
                        data_tx = result.data_transferred_gb  # newly introduced in resource_aware.py

                        # store distribution info
                        distribution_metrics[step].setdefault(algo_name, {})
                        distribution_metrics[step][algo_name]["components"] = dict(result.component_assignments)
                        distribution_metrics[step][algo_name]["caches"] = dict(result.cache_assignments)

                        # update prev
                        prev_assignments[algo_name] = result.component_assignments
                        prev_cache[algo_name] = result.cache_assignments

                    except Exception as e:
                        usage_dict = self._collect_current_device_usage()
                        if self.logger:
                            self.logger.log_error(
                                "toy_comparison_algorithm_error",
                                f"Step {step}, Algo='{algo_name}' error: {str(e)}",
                                level=LogLevel.DEBUG
                            )

                    # store per-step metrics
                    resource_metrics[step][algo_name] = usage_dict
                    communication_metrics[step].setdefault(algo_name, {})
                    communication_metrics[step][algo_name]["comm_time"] = comm_time
                    communication_metrics[step][algo_name]["data_gb"] = data_tx  # ADDED: store data transferred
                    migration_metrics[step][algo_name] = migrations
                    performance_metrics[step][algo_name] = {"latency": algo_latency}

                    # aggregator
                    if is_feasible and algo_latency < float('inf'):
                        total_latency[algo_name] += algo_latency
                        feasible_counts[algo_name] += 1
                    total_migrations[algo_name] += migrations

                    # also update comparison_metrics
                    comparison_metrics["latency"][algo_name].append(algo_latency)
                    if usage_dict and len(usage_dict) > 0:
                        avg_util = sum(d.get("compute_utilization", 0.0) for d in usage_dict.values()) / len(usage_dict)
                    else:
                        avg_util = 0.0
                    comparison_metrics["resource_utilization"][algo_name].append(avg_util)
                    comparison_metrics["migration_counts"][algo_name] += migrations
                    comparison_metrics["communication_overhead"][algo_name].append(comm_time)

                    if self.logger:
                        self.logger.log_event(
                            "debug",
                            f"Step {step}, Algo='{algo_name}': avg_util={avg_util:.4f}, "
                            f"migrations={migrations}, comm_time={comm_time:.4f}",
                            level=LogLevel.DEBUG
                        )

                # optional log
                if self.logger:
                    dev_states = self._collect_current_device_usage()
                    lat_per_algo = {a: performance_metrics[step][a]["latency"] for a in self.algorithms}
                    self.logger.log_metrics({
                        "toy_comparison_step": step,
                        "device_usage": dev_states,
                        "latencies_per_algo": lat_per_algo
                    })

            # build aggregator
            aggregates = {
                "average_latency": {},
                "total_migrations": {}
            }
            for algo_name in self.algorithms.keys():
                if feasible_counts[algo_name] > 0:
                    aggregates["average_latency"][algo_name] = (
                        total_latency[algo_name] / feasible_counts[algo_name]
                    )
                else:
                    aggregates["average_latency"][algo_name] = float('inf')
                aggregates["total_migrations"][algo_name] = total_migrations[algo_name]

            if self.logger:
                self.logger.log_event("debug", "Aggregates => " + pp.pformat(aggregates), level=LogLevel.DEBUG)

            final_resource = dict(resource_metrics)
            final_communication = dict(communication_metrics)
            final_performance = dict(performance_metrics)
            final_migrations = dict(migration_metrics)
            final_distribution = dict(distribution_metrics)

            # Insert aggregator
            final_performance["aggregates"] = aggregates

            scenario_metrics = collect_scenario_metrics(
                resource_metrics=final_resource,
                communication_metrics=final_communication,
                performance_metrics=final_performance,
            )
            scenario_metrics["migration_metrics"] = final_migrations

            # Also store distribution info
            scenario_metrics["distribution_metrics"] = final_distribution

            # Also attach comparison_metrics
            comparison_metrics_out = {}
            for key, submap in comparison_metrics.items():
                if key == "migration_counts":
                    comparison_metrics_out[key] = dict(submap)
                else:
                    comparison_metrics_out[key] = {algo: list(vals) for algo, vals in submap.items()}
            scenario_metrics["comparison_metrics"] = comparison_metrics_out

            # Overwrite summary fields
            scenario_metrics["summary"]["average_latency"] = aggregates["average_latency"]
            scenario_metrics["summary"]["total_migrations"] = aggregates["total_migrations"]

            # NEW: Summation of data transferred per algorithm
            total_data_for_algo = {}
            for step_idx, algo_map in final_communication.items():
                for algo, cdict in algo_map.items():
                    data_val = cdict.get("data_gb", 0.0)
                    if data_val is not None:
                        total_data_for_algo.setdefault(algo, 0.0)
                        total_data_for_algo[algo] += data_val

            scenario_metrics["summary"]["total_data_transferred_each_algo"] = total_data_for_algo

            if self.logger:
                self.logger.log_event(
                    "debug",
                    "Final scenario_metrics => \n" + pp.pformat(scenario_metrics),
                    level=LogLevel.DEBUG
                )

            return ScenarioResult(
                scenario_name=self.__class__.__name__,
                start_time=datetime.now(),
                end_time=datetime.now(),
                metrics=scenario_metrics,
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

    def _inject_random_background_usage(self) -> None:
        """For each device, add random fraction of memory/compute usage (0..30%)."""
        for dev_id, device in self.devices.items():
            mem_bg = random.uniform(0.0, 0.3) * device.memory.capacity
            comp_bg = random.uniform(0.0, 0.3) * device.compute.capacity
            device.memory.used = min(device.memory.used + mem_bg, device.memory.capacity)
            device.compute.used = min(device.compute.used + comp_bg, device.compute.capacity)

    def _collect_current_device_usage(self) -> Dict[str, Dict[str, float]]:
        usage = {}
        for dev_id, dev in self.devices.items():
            mem_used = dev.memory.used
            mem_cap = dev.memory.capacity
            comp_used = dev.compute.used
            comp_cap = dev.compute.capacity
            usage[dev_id] = {
                "memory_used": mem_used,
                "memory_capacity": mem_cap,
                "compute_used": comp_used,
                "compute_capacity": comp_cap,
                "memory_utilization": (mem_used / mem_cap) if mem_cap > 0 else 1.0,
                "compute_utilization": (comp_used / comp_cap) if comp_cap > 0 else 1.0
            }
        return usage

    def cleanup(self) -> None:
        if self.logger:
            self.logger.log_event("cleanup", "Cleaning up ToyComparisonScenario")
        super().cleanup()
