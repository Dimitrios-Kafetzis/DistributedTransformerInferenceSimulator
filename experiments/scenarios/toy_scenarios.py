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
# File: experiments/scenarios/toy_scenarios.py
#
# Description:
#   Defines a minimal 'ToyScenario' for demonstration/testing. This scenario
#   randomly assigns resource usage at each step, capturing metrics in a
#   straightforward manner, without running actual "transformer" logic.
#   Also defines a 'ToyComparisonScenario' that compares baseline strategies
#   (greedy, round-robin) vs resource-aware while simulating random background
#   usage on devices each step.

import random
from datetime import datetime
from typing import Dict
from pathlib import Path

# Import your base scenario classes and data structures
from .common import (
    BaseScenario,
    ScenarioResult,
    validate_scenario_requirements,
    collect_scenario_metrics
)
from experiments.scenarios.baseline_scenarios import BaselineComparisonScenario
from src.core import Network, Device, Transformer
from src.algorithms import (
    GreedyDistributor,
    RoundRobinDistributor,
    ResourceAwareDistributor
)
from src.environment import (
    NetworkTopologyGenerator,
    ResourceDistributor,
    WorkloadGenerator,
    LogNormalDistribution,
    WorkloadType
)
from src.utils import SimulationLogger, SimulationConfig


class ToyScenario(BaseScenario):
    """
    A toy scenario that sets up a very small network and devices,
    then for each step, randomly 'uses' some fraction of each device's resources.
    """

    def setup(self) -> None:
        """
        Sets up a small environment. In a real scenario, you might:
          - Parse config
          - Build a small 'toy' network
          - Distribute resources
          - Possibly create a small "workload"
        Here, we'll replicate a simplified environment setup similar
        to how your other scenarios do it, but with minimal logic.
        """
        if self.logger:
            self.logger.log_event("setup", "Setting up ToyScenario environment")

        # (A) Build resource distributions from the config
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
            num_devices=self.config.network.num_devices,
            memory_distribution=mem_dist,
            compute_distribution=comp_dist,
            seed=self.config.resources.seed
        )
        capabilities = resource_distributor.generate_capabilities()

        # (B) Build 'Device' objects
        self.devices = {}
        for device_id, caps in capabilities.items():
            self.devices[device_id] = Device(
                device_id=device_id,
                memory_capacity=caps.memory_capacity,
                compute_capacity=caps.compute_capacity,
                is_source=caps.is_source
            )

        # (C) Build a minimal 'Network'
        self.network = Network()
        random.seed(self.config.network.seed)
        for dev_id in self.devices:
            self.network.add_device(dev_id)

        # Simple full mesh with random bandwidth
        for dev1 in self.devices:
            for dev2 in self.devices:
                if dev1 == dev2:
                    continue
                bandwidth = random.uniform(
                    self.config.network.min_bandwidth,
                    self.config.network.max_bandwidth
                )
                self.network.add_link(dev1, dev2, bandwidth=bandwidth)

        # (D) Create a small "Transformer" from the workload
        self.workload_generator = WorkloadGenerator(seed=self.config.workload.seed)
        self.workload = self.workload_generator.generate_workload(
            workload_type=self.config.workload.model_type,
            sequence_config=None  # or a small config
        )
        test_transformer = self.workload.transformer

        # (E) Validate scenario requirements if desired
        if not validate_scenario_requirements(self.config, self.network, self.devices, test_transformer):
            raise ValueError("Scenario requirements not met in ToyScenario setup")

        if self.logger:
            self.logger.log_event("setup", f"ToyScenario setup complete with {len(self.devices)} devices.")

    def run(self) -> ScenarioResult:
        """
        Runs the toy scenario by simulating a few steps, randomly generating
        resource usage for each device. We store usage in resource_metrics,
        and also create a trivial 'performance_metrics' structure with
        random latency. This scenario doesn't do a real 'assignment'.
        """
        if self.logger:
            self.logger.log_event("run", "Starting ToyScenario run")

        resource_metrics = {}
        performance_metrics = {}

        try:
            if not hasattr(self.config.workload, 'generation_steps'):
                total_steps = 5
            else:
                total_steps = self.config.workload.generation_steps[0] if self.config.workload.generation_steps else 5

            for step in range(total_steps):
                usage_for_this_step = {}
                for dev_id, device in self.devices.items():
                    mem_used = random.uniform(0.0, device.memory.capacity / 2.0)
                    mem_util = mem_used / device.memory.capacity

                    comp_used = random.uniform(0.0, device.compute.capacity / 2.0)
                    comp_util = comp_used / device.compute.capacity

                    usage_for_this_step[dev_id] = {
                        "memory_used": mem_used,
                        "memory_utilization": mem_util,
                        "compute_used": comp_used,
                        "compute_utilization": comp_util
                    }

                resource_metrics[step] = usage_for_this_step

                perf = {
                    "latency": random.uniform(0.001, 0.01) * (step + 1),
                    "step": step
                }
                performance_metrics[step] = perf

                if self.logger:
                    self.logger.log_metrics({
                        "toy_step": step,
                        "toy_usage": usage_for_this_step,
                        "toy_perf": perf
                    })

            scenario_metrics = collect_scenario_metrics(
                resource_metrics=resource_metrics,
                communication_metrics={},
                performance_metrics=performance_metrics
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

    def cleanup(self) -> None:
        """Cleanup if needed."""
        if self.logger:
            self.logger.log_event("cleanup", "Cleaning up ToyScenario")
        # Nothing special here—no open resources
        pass


