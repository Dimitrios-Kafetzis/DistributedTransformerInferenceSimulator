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
#   step-by-step communication & migration metrics. Includes also an ExactOptimal approach for small device counts.

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
    ExactOptimalDistributor
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
      1) Reset device usage to reflect only persistent allocations
      2) Add random background usage
      3) For each algorithm, compute_assignment(...)
      4) Collect resource usage, communication overhead, migrations, latency
      5) Also store distribution info
      6) Aggregate results => final summary
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
        self.dynamic = DynamicMigrationDistributor(
            test_transformer,
            self.network,
            self.devices,
            memory_threshold=self.config.algorithm.migration_threshold,
            compute_threshold=self.config.algorithm.migration_threshold,
            logger=self.logger
        )
        #self.resource_aware = ResourceAwareDistributor(test_transformer, self.network, self.devices, self.logger)
        self.resource_aware = ResourceAwareDistributor(
            transformer=test_transformer,
            network=self.network,
            devices=self.devices,
            logger=self.logger,
            use_weighted_sum=True,     # pick weighted sum
            alpha=0.5,                 # example weighting
            beta=0.3,
            gamma=0.2,
            use_partial_latency_check=False  # enable the mini-latency concurrency check
        )

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
                is_source=c.is_source,
                logger=self.logger
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

        # Store distribution info (which device each block + cache is on)
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

                # 1) Reset usage to only persistent usage, then 2) add random usage
                self._inject_random_background_usage()

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
                    data_tx = 0.0

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
                        comm_time = result.communication_time
                        data_tx = result.data_transferred_gb

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

                    # Store per-step metrics
                    resource_metrics[step][algo_name] = usage_dict
                    communication_metrics[step].setdefault(algo_name, {})
                    communication_metrics[step][algo_name]["comm_time"] = comm_time
                    communication_metrics[step][algo_name]["data_gb"] = data_tx
                    migration_metrics[step][algo_name] = migrations
                    performance_metrics[step][algo_name] = {"latency": algo_latency}

                    # aggregator
                    if is_feasible and algo_latency < float('inf'):
                        total_latency[algo_name] += algo_latency
                        feasible_counts[algo_name] += 1
                    total_migrations[algo_name] += migrations

                    # comparison metrics
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

            # Aggregation
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

            final_performance["aggregates"] = aggregates

            scenario_metrics = collect_scenario_metrics(
                resource_metrics=final_resource,
                communication_metrics=final_communication,
                performance_metrics=final_performance
            )
            scenario_metrics["migration_metrics"] = final_migrations
            scenario_metrics["distribution_metrics"] = final_distribution

            # also attach comparison_metrics
            comparison_metrics_out = {}
            for key, submap in comparison_metrics.items():
                if key == "migration_counts":
                    comparison_metrics_out[key] = dict(submap)
                else:
                    comparison_metrics_out[key] = {algo: list(vals) for algo, vals in submap.items()}
            scenario_metrics["comparison_metrics"] = comparison_metrics_out

            scenario_metrics["summary"]["average_latency"] = aggregates["average_latency"]
            scenario_metrics["summary"]["total_migrations"] = aggregates["total_migrations"]

            # Summation of data transferred per algorithm
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
        """
        Reset each device's used memory/compute to reflect only persistent usage,
        then add random usage for this step.

        We rely on assigned_components[...] storing 'memory' and 'compute' keys for usage.
        """
        for dev_id, device in self.devices.items():
            # 1) Sum up usage from currently assigned components 
            #    (any ephemeral ones presumably have been removed by the distributors).
            persistent_mem = 0.0
            persistent_comp = 0.0
            for comp_id, comp_info in device.assigned_components.items():
                # comp_info = {"memory": <float>, "compute": <float>, "ephemeral": <bool>}
                mem_req = float(comp_info["memory"])
                comp_req = float(comp_info["compute"])
                persistent_mem += mem_req
                persistent_comp += comp_req

            # Set usage to the persistent-only total
            device.memory.used = min(persistent_mem, device.memory.capacity)
            device.compute.used = min(persistent_comp, device.compute.capacity)

            # 2) Add new background usage
            mem_bg = random.uniform(0.0, 0.3) * device.memory.capacity
            comp_bg = random.uniform(0.0, 0.3) * device.compute.capacity

            device.memory.used = min(device.memory.used + mem_bg, device.memory.capacity)
            device.compute.used = min(device.compute.used + comp_bg, device.compute.capacity)

            if self.logger:
                self.logger.log_event(
                    "bg_usage_injection",
                    (
                        f"Device={dev_id}: persistent_mem={persistent_mem:.4f}, persistent_comp={persistent_comp:.4f}, "
                        f"+mem_bg={mem_bg:.4f}, +comp_bg={comp_bg:.4f} => final mem_used={device.memory.used:.4f}, "
                        f"comp_used={device.compute.used:.4f}"
                    ),
                    level=LogLevel.DEBUG
                )

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

class ToyOptimalComparisonScenario(BaselineComparisonBaseScenario):
    """
    Similar to ToyComparisonScenario, but includes the EXACT approach
    for small device counts to measure how close we get to the optimal.

    We'll forcibly limit to 2 or 3 devices so that the enumerations remain feasible.
    We'll define a small model with e.g. 2 heads, smaller dims, so that the brute force
    doesn't blow up in runtime.
    """

    def setup(self) -> None:
        if self.logger:
            self.logger.log_event("setup", "Setting up ToyOptimalComparisonScenario environment", level=LogLevel.DEBUG)

        self._create_network_and_devices()

        # create a small workload
        self.workload_generator = WorkloadGenerator(seed=self.config.workload.seed)
        self.workload = self.workload_generator.generate_workload(
            workload_type=self.config.workload.model_type,
            sequence_config=None
        )
        if self.workload.sequence_config:
            self.workload.transformer.current_sequence_length = self.workload.sequence_config.initial_length
        else:
            # e.g. small default
            self.workload.transformer.current_sequence_length = 16

        # validate
        test_transformer = self.workload.transformer
        if not validate_scenario_requirements(self.config, self.network, self.devices, test_transformer):
            raise ValueError("Scenario requirements not met in ToyOptimalComparisonScenario setup")

        # now define the distributors
        self.greedy = GreedyDistributor(test_transformer, self.network, self.devices, self.logger)
        self.round_robin = RoundRobinDistributor(test_transformer, self.network, self.devices, self.logger)
        self.static = StaticDistributor(test_transformer, self.network, self.devices, self.logger)
        self.dynamic = DynamicMigrationDistributor(
            test_transformer,
            self.network,
            self.devices,
            memory_threshold=self.config.algorithm.migration_threshold,
            compute_threshold=self.config.algorithm.migration_threshold,
            logger=self.logger
        )
        self.resource_aware = ResourceAwareDistributor(
            transformer=test_transformer,
            network=self.network,
            devices=self.devices,
            logger=self.logger,
            use_weighted_sum=True,
            alpha=0.3,
            beta=0.4,
            gamma=0.4,
            use_partial_latency_check=False
        )
        # the new exact approach:
        self.exact_optimal = ExactOptimalDistributor(
            test_transformer, self.network, self.devices, self.logger
        )

        self.algorithms = {
            "greedy": self.greedy,
            "round_robin": self.round_robin,
            "static": self.static,
            "dynamic": self.dynamic,
            "resource_aware": self.resource_aware,
            "exact_optimal": self.exact_optimal,
        }

        if self.logger:
            self.logger.log_event("setup", f"Setup complete with {len(self.devices)} devices for ToyOptimalComparisonScenario", level=LogLevel.DEBUG)


    def _create_network_and_devices(self):
        """
        We'll fix e.g. 2 or 3 devices in the config. We'll follow the same approach as you do in the toy scenario:
        - random bandwidth in [0.1..0.5] or so
        - lognormal distribution for memory/compute
        """
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
        rd = ResourceDistributor(
            num_devices=self.config.network.num_devices,
            memory_distribution=mem_dist,
            compute_distribution=comp_dist,
            seed=self.config.resources.seed
        )
        caps = rd.generate_capabilities()
        self.devices = {}
        random.seed(self.config.network.seed)

        for d_id, c in caps.items():
            self.devices[d_id] = Device(
                device_id=d_id,
                memory_capacity=c.memory_capacity,
                compute_capacity=c.compute_capacity,
                is_source=c.is_source,
                logger=self.logger
            )

        self.network = Network()
        for d1 in self.devices:
            self.network.add_device(d1)
        dev_keys = list(self.devices.keys())
        for i in range(len(dev_keys)):
            for j in range(i+1, len(dev_keys)):
                bw = random.uniform(
                    self.config.network.min_bandwidth, self.config.network.max_bandwidth
                )
                self.network.add_link(dev_keys[i], dev_keys[j], bandwidth=bw)


    def run(self) -> ScenarioResult:
        """
        We'll do a similar multi-step run with random background usage,
        collecting latency, etc. But the key is we also see the 'exact_optimal' result
        and can measure the difference.
        """
        from collections import defaultdict
        self.logger.log_event("run", "ToyOptimalComparisonScenario starts", level=LogLevel.DEBUG)

        total_steps = 5
        if isinstance(self.config.workload.generation_steps, list) and len(self.config.workload.generation_steps) > 0:
            total_steps = self.config.workload.generation_steps[0]
        elif isinstance(self.config.workload.generation_steps, int):
            total_steps = self.config.workload.generation_steps

        resource_metrics = defaultdict(dict)
        communication_metrics = defaultdict(dict)
        migration_metrics = defaultdict(dict)
        performance_metrics = defaultdict(dict)
        distribution_metrics = defaultdict(dict)

        # comparison data
        comparison = {
            "latency": defaultdict(list),
            "avg_util": defaultdict(list),
            "migrations": defaultdict(int),
        }

        # keep track of aggregator
        total_latency = {a: 0.0 for a in self.algorithms}
        feasible_counts = {a: 0 for a in self.algorithms}

        prev_assignments = {a: None for a in self.algorithms}
        prev_cache = {a: None for a in self.algorithms}

        for step in range(total_steps):
            self._inject_random_bg_usage()

            for algo_name, dist in self.algorithms.items():
                try:
                    result = dist.compute_assignment(step, prev_assignments[algo_name], prev_cache[algo_name])
                    resource_metrics[step][algo_name] = result.resource_usage
                    migration_metrics[step][algo_name] = len(result.migrations or [])
                    performance_metrics[step][algo_name] = {"latency": result.estimated_latency}
                    distribution_metrics[step][algo_name] = {
                        "assignments": dict(result.component_assignments),
                        "caches": dict(result.cache_assignments)
                    }

                    if result.is_feasible and result.estimated_latency < float('inf'):
                        total_latency[algo_name] += result.estimated_latency
                        feasible_counts[algo_name] += 1

                    # store
                    prev_assignments[algo_name] = result.component_assignments
                    prev_cache[algo_name] = result.cache_assignments

                    # average utilization
                    usage_dict = result.resource_usage
                    if usage_dict:
                        avg_util = sum(d.get("compute_utilization", 0.0) for d in usage_dict.values()) / len(usage_dict)
                    else:
                        avg_util = 0.0
                    comparison["avg_util"][algo_name].append(avg_util)
                    comparison["latency"][algo_name].append(result.estimated_latency)
                    comparison["migrations"][algo_name] += len(result.migrations or [])

                except Exception as e:
                    # log error
                    pass

        # build scenario metrics
        final_resource = dict(resource_metrics)
        final_communication = dict(communication_metrics)
        final_performance = dict(performance_metrics)
        final_migrations = dict(migration_metrics)

        from experiments.scenarios.common import collect_scenario_metrics
        scn_metrics = collect_scenario_metrics(
            resource_metrics=final_resource,
            communication_metrics=final_communication,
            performance_metrics=final_performance
        )
        scn_metrics["migration_metrics"] = final_migrations
        scn_metrics["distribution_metrics"] = dict(distribution_metrics)

        # aggregator
        aggregates = {"avg_latency": {}, "success": {}}
        for a in self.algorithms:
            if feasible_counts[a] > 0:
                aggregates["avg_latency"][a] = total_latency[a]/feasible_counts[a]
            else:
                aggregates["avg_latency"][a] = float('inf')
            aggregates["success"][a] = feasible_counts[a]

        scn_metrics["summary"]["aggregates"] = aggregates
        # also store the comparison dictionary
        scn_metrics["comparison_metrics"] = {
            "latency": {a: comparison["latency"][a] for a in self.algorithms},
            "avg_util": {a: comparison["avg_util"][a] for a in self.algorithms},
            "total_migrations": {a: comparison["migrations"][a] for a in self.algorithms},
        }

        return ScenarioResult(
            scenario_name=self.__class__.__name__,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metrics=scn_metrics,
            success=True
        )

    def _inject_random_bg_usage(self):
        """Similar to your existing approach, or simpler."""
        for dev in self.devices.values():
            # remove ephemeral usage
            ephemeral_memory = 0.0
            ephemeral_compute = 0.0
            to_remove = []
            for cid, info in dev.assigned_components.items():
                if info.get("ephemeral", True):
                    to_remove.append(cid)
            for ccache, cinfo in dev.cache_assignments.items():
                if cinfo.get("ephemeral", True):
                    to_remove.append(ccache)

            for c in to_remove:
                dev.deallocate_resources(c, force=True)

            # now usage is only persistent
            # add random
            mem_bg = random.uniform(0.0, 0.2)*dev.memory.capacity
            comp_bg = random.uniform(0.0, 0.2)*dev.compute.capacity
            dev.memory.used = min(dev.memory.used + mem_bg, dev.memory.capacity)
            dev.compute.used = min(dev.compute.used + comp_bg, dev.compute.capacity)


    def cleanup(self) -> None:
        if self.logger:
            self.logger.log_event("cleanup", "Cleaning up ToyOptimalComparisonScenario", level=LogLevel.INFO)
        super().cleanup()