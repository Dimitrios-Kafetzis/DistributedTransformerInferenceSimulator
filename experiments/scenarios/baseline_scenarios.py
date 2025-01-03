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
# File:    experiments/scenarios/baseline_scenarios.py
# Description:
#   Provides scenarios that compare baseline strategies (e.g., greedy,
#   round-robin, static partitioning, dynamic migration) against the
#   resource-aware approach for distributed Transformer inference.
#
# ---------------------------------------------------------------------------

"""
Implements baseline scenario classes for comparing naive or simpler distribution
methods, such as greedy or round-robin, with the resource-aware approach in
transformer inference across multiple edge or hybrid devices.
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import json

from .common import (
    BaseScenario,
    ScenarioResult,
    validate_scenario_requirements,
    collect_scenario_metrics
)
from src.core import Network, Device, Transformer
from src.algorithms import (
    GreedyDistributor,
    RoundRobinDistributor,
    StaticDistributor,
    DynamicMigrationDistributor,
    ResourceAwareDistributor
)
from src.environment import (
    NetworkTopologyGenerator,
    ResourceDistributor,
    WorkloadGenerator,
    EdgeClusterTopology,
    DistributedEdgeTopology,
    HybridCloudEdgeTopology,
    LogNormalDistribution,
    WorkloadType
)
from src.utils import (
    SimulationLogger,
    SimulationConfig
)

from experiments.scenarios.common import load_scenario_config


class BaselineComparisonBaseScenario(BaseScenario):
    """
    Base scenario class that sets up the environment (network, devices,
    single or multiple workloads) according to the config's topology_type.
    Child baseline scenarios inherit from this and only need to implement
    'run()' (and 'cleanup()' if needed).
    """

    def setup(self) -> None:
        """Set up the environment based on config, creating network/devices/workload."""
        if self.logger:
            self.logger.log_event("setup", "Setting up baseline comparison environment")

        # 1. Create the network topology based on config.network.topology_type
        topo_type = self.config.network.topology_type
        if topo_type == "edge_cluster":
            topology_generator = EdgeClusterTopology(self.config.network)
        elif topo_type == "distributed_edge":
            topology_generator = DistributedEdgeTopology(self.config.network)
        elif topo_type == "hybrid_cloud_edge":
            topology_generator = HybridCloudEdgeTopology(self.config.network)
        else:
            raise ValueError(f"Unknown topology_type: {topo_type}")

        self.network = topology_generator.generate()

        # 2. Create resource distributions from config
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

        # 4. Create workloads
        self.workload_generator = WorkloadGenerator(seed=self.config.workload.seed)
        # For simplicity, assume a single model_type scenario for baseline
        # If you have multiple model_types, you can adapt similarly.
        if getattr(self.config.workload, 'model_types', None):
            # Just pick the first or handle multiple if desired
            chosen_type = self.config.workload.model_types[0]
            self.workload = self.workload_generator.generate_workload(
                workload_type=WorkloadType[chosen_type],
                sequence_config=None
            )
        else:
            # single model_type
            single_mtype = getattr(self.config.workload, 'model_type', WorkloadType.SMALL)
            self.workload = self.workload_generator.generate_workload(
                workload_type=single_mtype,
                sequence_config=None
            )

        # 5. Validate scenario if desired
        test_transformer = Transformer(self.workload.transformer.config)
        if not validate_scenario_requirements(
            self.config, self.network, self.devices, test_transformer
        ):
            raise ValueError(
                "Scenario requirements not met in baseline environment setup"
            )

    def run(self) -> ScenarioResult:
        """
        Child classes must override 'run()'.
        This is left as a placeholder so we can be non-abstract.
        """
        raise NotImplementedError("Subclasses must implement run()")

    def cleanup(self) -> None:
        """
        Non-abstract cleanup so that child classes do not remain abstract.
        Subclasses can override or extend this if needed.
        """
        if self.logger:
            self.logger.log_event(
                "cleanup",
                "Cleaning up baseline comparison scenario"
            )


class BaselineComparisonScenario(BaselineComparisonBaseScenario):
    """
    A secondary base that adds some shared methods for
    metric initialization and metric recording.
    """

    def _initialize_metrics(self) -> Dict:
        """Initialize metrics tracking structure"""
        return {
            'resource_metrics': defaultdict(dict),
            'communication_metrics': defaultdict(dict),
            'performance_metrics': defaultdict(dict),
            'comparison_metrics': {
                'latency': defaultdict(list),
                'resource_utilization': defaultdict(list),
                'communication_overhead': defaultdict(list),
                'migration_counts': defaultdict(int)
            }
        }

    def _record_comparison_metrics(
        self,
        algorithm_name: str,
        step: int,
        assignment_result,
        metrics: Dict
    ) -> None:
        """Record metrics for algorithm comparison"""
        metrics['latency'][algorithm_name].append(
            assignment_result.estimated_latency
        )
        # Resource utilization
        if assignment_result.resource_usage:
            avg_util = np.mean([
                usage.get('compute_utilization', 0.0)
                for usage in assignment_result.resource_usage.values()
            ])
        else:
            avg_util = 0.0
        metrics['resource_utilization'][algorithm_name].append(avg_util)

        # Migrations if available
        if hasattr(assignment_result, 'migrations'):
            metrics['migration_counts'][algorithm_name] += len(assignment_result.migrations)

    def run(self) -> ScenarioResult:
        # Remains abstract unless you want a default no-op
        raise NotImplementedError("Child classes must implement run().")

    def cleanup(self) -> None:
        """Clean up resources if needed"""
        if self.logger:
            self.logger.log_event("cleanup", f"Cleaning up {self.__class__.__name__}")
        # If there's anything special to do, do it here.
        super().cleanup()


class GreedyBaselineScenario(BaselineComparisonScenario):
    """
    Tests greedy placement strategy that assigns components
    to first available device with sufficient resources.
    """

    def setup(self) -> None:
        super().setup()  # sets up environment
        if self.logger:
            self.logger.log_event("setup", "Setting up Greedy baseline")

        self.greedy_distributor = GreedyDistributor(
            self.workload.transformer,
            self.network,
            self.devices
        )
        self.resource_aware_distributor = ResourceAwareDistributor(
            self.workload.transformer,
            self.network,
            self.devices
        )

    def run(self) -> ScenarioResult:
        """Run greedy baseline comparison"""
        metrics = self._initialize_metrics()
        try:
            seq_config = self.workload.sequence_config
            if seq_config is None:
                raise ValueError("No sequence_config defined in workload for GreedyBaselineScenario.")

            for step in range(seq_config.num_steps):
                # Greedy
                greedy_result = self.greedy_distributor.compute_assignment(
                    generation_step=step,
                    previous_assignments=metrics.get('greedy_previous'),
                    previous_cache=metrics.get('greedy_cache')
                )
                # Resource-Aware
                resource_aware_result = self.resource_aware_distributor.compute_assignment(
                    generation_step=step,
                    previous_assignments=metrics.get('resource_aware_previous'),
                    previous_cache=metrics.get('resource_aware_cache')
                )

                # Record
                self._record_comparison_metrics('greedy', step, greedy_result, metrics['comparison_metrics'])
                self._record_comparison_metrics('resource_aware', step, resource_aware_result, metrics['comparison_metrics'])

                # Update
                metrics['greedy_previous'] = greedy_result.component_assignments
                metrics['greedy_cache'] = greedy_result.cache_assignments
                metrics['resource_aware_previous'] = resource_aware_result.component_assignments
                metrics['resource_aware_cache'] = resource_aware_result.cache_assignments

            scenario_metrics = collect_scenario_metrics(
                resource_metrics=metrics['resource_metrics'],
                communication_metrics=metrics['communication_metrics'],
                performance_metrics=metrics['performance_metrics']
            )
            scenario_metrics['comparison_metrics'] = metrics['comparison_metrics']

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
        """Explicitly implement cleanup to satisfy the abstract base."""
        if self.logger:
            self.logger.log_event("cleanup", f"Cleaning up {self.__class__.__name__}")
        super().cleanup()


class RoundRobinBaselineScenario(BaselineComparisonScenario):
    """
    Tests round-robin distribution strategy that distributes components
    evenly across available devices
    """

    def setup(self) -> None:
        super().setup()
        if self.logger:
            self.logger.log_event("setup", "Setting up Round-Robin baseline")

        self.round_robin_distributor = RoundRobinDistributor(
            self.workload.transformer,
            self.network,
            self.devices
        )
        self.resource_aware_distributor = ResourceAwareDistributor(
            self.workload.transformer,
            self.network,
            self.devices
        )

    def run(self) -> ScenarioResult:
        metrics = self._initialize_metrics()
        try:
            seq_config = self.workload.sequence_config
            if seq_config is None:
                raise ValueError("No sequence_config in workload for RoundRobinBaselineScenario.")

            for step in range(seq_config.num_steps):
                rr_result = self.round_robin_distributor.compute_assignment(
                    generation_step=step,
                    previous_assignments=metrics.get('round_robin_previous'),
                    previous_cache=metrics.get('round_robin_cache')
                )
                ra_result = self.resource_aware_distributor.compute_assignment(
                    generation_step=step,
                    previous_assignments=metrics.get('resource_aware_previous'),
                    previous_cache=metrics.get('resource_aware_cache')
                )

                self._record_comparison_metrics('round_robin', step, rr_result, metrics['comparison_metrics'])
                self._record_comparison_metrics('resource_aware', step, ra_result, metrics['comparison_metrics'])

                metrics['round_robin_previous'] = rr_result.component_assignments
                metrics['round_robin_cache'] = rr_result.cache_assignments
                metrics['resource_aware_previous'] = ra_result.component_assignments
                metrics['resource_aware_cache'] = ra_result.cache_assignments

            scenario_metrics = collect_scenario_metrics(
                resource_metrics=metrics['resource_metrics'],
                communication_metrics=metrics['communication_metrics'],
                performance_metrics=metrics['performance_metrics']
            )
            scenario_metrics['comparison_metrics'] = metrics['comparison_metrics']

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
        """Explicitly implement cleanup."""
        if self.logger:
            self.logger.log_event("cleanup", f"Cleaning up {self.__class__.__name__}")
        super().cleanup()


class StaticBaselineScenario(BaselineComparisonScenario):
    """
    Tests static partitioning strategy that maintains fixed
    assignments based on initial conditions
    """

    def setup(self) -> None:
        super().setup()
        if self.logger:
            self.logger.log_event("setup", "Setting up Static baseline")

        self.static_distributor = StaticDistributor(
            self.workload.transformer,
            self.network,
            self.devices
        )
        self.resource_aware_distributor = ResourceAwareDistributor(
            self.workload.transformer,
            self.network,
            self.devices
        )

    def run(self) -> ScenarioResult:
        metrics = self._initialize_metrics()
        try:
            # Create initial static assignment
            initial_result = self.static_distributor.compute_assignment(generation_step=0)
            if not initial_result.is_feasible:
                raise RuntimeError("Initial static assignment infeasible for static baseline")

            seq_config = self.workload.sequence_config
            if seq_config is None:
                raise ValueError("No sequence_config in workload for StaticBaselineScenario.")

            for step in range(seq_config.num_steps):
                static_result = self.static_distributor.compute_assignment(
                    generation_step=step,
                    previous_assignments=initial_result.component_assignments,
                    previous_cache=initial_result.cache_assignments
                )
                ra_result = self.resource_aware_distributor.compute_assignment(
                    generation_step=step,
                    previous_assignments=metrics.get('resource_aware_previous'),
                    previous_cache=metrics.get('resource_aware_cache')
                )

                self._record_comparison_metrics('static', step, static_result, metrics['comparison_metrics'])
                self._record_comparison_metrics('resource_aware', step, ra_result, metrics['comparison_metrics'])

                metrics['resource_aware_previous'] = ra_result.component_assignments
                metrics['resource_aware_cache'] = ra_result.cache_assignments

            scenario_metrics = collect_scenario_metrics(
                resource_metrics=metrics['resource_metrics'],
                communication_metrics=metrics['communication_metrics'],
                performance_metrics=metrics['performance_metrics']
            )
            scenario_metrics['comparison_metrics'] = metrics['comparison_metrics']

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
        """Explicitly implement cleanup."""
        if self.logger:
            self.logger.log_event("cleanup", f"Cleaning up {self.__class__.__name__}")
        super().cleanup()


class DynamicMigrationBaselineScenario(BaselineComparisonScenario):
    """
    Tests dynamic migration strategy that reassigns components
    when resource utilization exceeds thresholds
    """

    def setup(self) -> None:
        super().setup()
        if self.logger:
            self.logger.log_event("setup", "Setting up Dynamic Migration baseline")

        self.dynamic_distributor = DynamicMigrationDistributor(
            self.workload.transformer,
            self.network,
            self.devices,
            memory_threshold=self.config.algorithm.migration_threshold,
            compute_threshold=self.config.algorithm.migration_threshold
        )
        self.resource_aware_distributor = ResourceAwareDistributor(
            self.workload.transformer,
            self.network,
            self.devices
        )
        self.migration_history = {
            'dynamic': [],
            'resource_aware': []
        }

    def run(self) -> ScenarioResult:
        metrics = self._initialize_metrics()
        try:
            seq_config = self.workload.sequence_config
            if seq_config is None:
                raise ValueError("No sequence_config in workload for DynamicMigrationBaselineScenario.")

            for step in range(seq_config.num_steps):
                dynamic_result = self.dynamic_distributor.compute_assignment(
                    generation_step=step,
                    previous_assignments=metrics.get('dynamic_previous'),
                    previous_cache=metrics.get('dynamic_cache')
                )
                ra_result = self.resource_aware_distributor.compute_assignment(
                    generation_step=step,
                    previous_assignments=metrics.get('resource_aware_previous'),
                    previous_cache=metrics.get('resource_aware_cache')
                )

                self._record_comparison_metrics('dynamic', step, dynamic_result, metrics['comparison_metrics'])
                self._record_comparison_metrics('resource_aware', step, ra_result, metrics['comparison_metrics'])

                self._track_migrations(step, dynamic_result, ra_result,
                                       metrics.get('dynamic_previous'),
                                       metrics.get('resource_aware_previous'))

                metrics['dynamic_previous'] = dynamic_result.component_assignments
                metrics['dynamic_cache'] = dynamic_result.cache_assignments
                metrics['resource_aware_previous'] = ra_result.component_assignments
                metrics['resource_aware_cache'] = ra_result.cache_assignments

            scenario_metrics = collect_scenario_metrics(
                resource_metrics=metrics['resource_metrics'],
                communication_metrics=metrics['communication_metrics'],
                performance_metrics=metrics['performance_metrics']
            )
            scenario_metrics['comparison_metrics'] = metrics['comparison_metrics']

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

    def _track_migrations(
        self,
        step: int,
        dynamic_result,
        resource_aware_result,
        prev_dynamic: Optional[Dict],
        prev_resource_aware: Optional[Dict]
    ) -> None:
        """Track component migrations for both algorithms."""
        if not prev_dynamic or not prev_resource_aware:
            return

        dynamic_migrations = self._get_migrations(prev_dynamic, dynamic_result.component_assignments)
        for comp_id, (old_dev, new_dev) in dynamic_migrations.items():
            self.migration_history['dynamic'].append({
                'step': step,
                'component': comp_id,
                'from': old_dev,
                'to': new_dev
            })

        ra_migrations = self._get_migrations(prev_resource_aware, resource_aware_result.component_assignments)
        for comp_id, (old_dev, new_dev) in ra_migrations.items():
            self.migration_history['resource_aware'].append({
                'step': step,
                'component': comp_id,
                'from': old_dev,
                'to': new_dev
            })

    def _get_migrations(
        self,
        prev_assignments: Dict[str, str],
        new_assignments: Dict[str, str]
    ) -> Dict[str, Tuple[str, str]]:
        """Identify component migrations between assignments."""
        migrations = {}
        for comp_id, new_dev in new_assignments.items():
            if comp_id in prev_assignments:
                old_dev = prev_assignments[comp_id]
                if old_dev != new_dev:
                    migrations[comp_id] = (old_dev, new_dev)
        return migrations

    def cleanup(self) -> None:
        """Explicitly implement cleanup."""
        if self.logger:
            self.logger.log_event("cleanup", f"Cleaning up {self.__class__.__name__}")
        # Save migration history or additional cleanup actions
        if self.migration_history:
            # e.g., self.logger.log_metrics({"migration_history": self.migration_history})
            pass

        super().cleanup()


def run_all_baselines(
    config_path: Union[str, Path],
    output_dir: Union[str, Path],
    logger: Optional[SimulationLogger] = None
) -> Dict[str, ScenarioResult]:
    """
    Run all baseline comparisons.
    1) Load the config from 'config_path'.
    2) Instantiate each baseline scenario class with (config, scenario_output_dir, logger).
    3) Execute each scenario, store results in a dict.
    """
    # 1) Load config
    config = load_scenario_config(config_path)

    # 2) Prepare scenarios
    baseline_scenarios = [
        GreedyBaselineScenario,
        RoundRobinBaselineScenario,
        StaticBaselineScenario,
        DynamicMigrationBaselineScenario
    ]

    results = {}
    for scenario_class in baseline_scenarios:
        scenario_name = scenario_class.__name__
        scenario_out_dir = Path(output_dir) / scenario_name.lower()
        scenario_out_dir.mkdir(parents=True, exist_ok=True)

        if logger:
            logger.log_event("baseline_comparison", f"Running {scenario_name}")

        try:
            scenario = scenario_class(config, scenario_out_dir, logger)
            scenario_result = scenario.execute()
            results[scenario_name] = scenario_result

        except Exception as e:
            if logger:
                logger.log_error("baseline_error", f"Error running {scenario_name}: {str(e)}")
            results[scenario_name] = ScenarioResult(
                scenario_name=scenario_name,
                start_time=datetime.now(),
                end_time=datetime.now(),
                metrics={},
                success=False,
                error=str(e)
            )

    return results


def analyze_baseline_results(
    results: Dict[str, dict],           # now scenario_name -> dictionary
    output_dir: Union[str, Path],
    logger: Optional[SimulationLogger] = None
) -> Dict:
    """
    Analyze and compare baseline results, returning analysis dictionary.

    Args:
        results: A dictionary keyed by scenario name (e.g. "GreedyBaselineScenario"),
                 each value is a dictionary with fields like:
                   {
                     "scenario_name": ...,
                     "start_time": ...,
                     "end_time": ...,
                     "metrics": { ... },
                     "success": bool,
                     "error": ...
                   }
        output_dir: Where to store 'baseline_analysis.json'
        logger: Optional SimulationLogger for structured logging.

    Returns:
        A dictionary containing analysis results (latencies, resource efficiency,
        communication overhead, migration analysis, and summary).
    """
    analysis = {
        'performance_comparison': {},
        'resource_efficiency': {},
        'communication_overhead': {},
        'migration_analysis': {},
        'summary': {}
    }

    try:
        # Performance comparison
        latencies = {}
        for name, scenario_dict in results.items():
            # scenario_dict is a dictionary (not ScenarioResult) now
            if not scenario_dict.get('success', False):
                # skip if scenario was not successful
                continue
            # scenario_dict["metrics"] is presumably a dictionary
            pm = scenario_dict["metrics"].get('performance_metrics', {})
            latencies[name] = pm.get('average_latency', 0)

        analysis['performance_comparison']['latency'] = latencies

        # Resource efficiency
        utilizations = {}
        for name, scenario_dict in results.items():
            if not scenario_dict.get('success', False):
                continue
            rm = scenario_dict["metrics"].get('resource_metrics', {})
            utilizations[name] = rm.get('average_utilization', 0)

        analysis['resource_efficiency']['utilization'] = utilizations

        # Communication overhead
        comm_overhead = {}
        for name, scenario_dict in results.items():
            if not scenario_dict.get('success', False):
                continue
            cm = scenario_dict["metrics"].get('communication_metrics', {})
            comm_overhead[name] = cm.get('total_data_transferred', 0)

        analysis['communication_overhead']['data_transferred'] = comm_overhead

        # Migration analysis
        migrations = {}
        for name, scenario_dict in results.items():
            if not scenario_dict.get('success', False):
                continue
            ms = scenario_dict["metrics"].get('migration_statistics', {})
            migrations[name] = ms

        analysis['migration_analysis'] = migrations

        # Calculate summary
        if latencies:
            best_latency = min(latencies.items(), key=lambda x: x[1])
        else:
            best_latency = None

        if utilizations:
            best_util = max(utilizations.items(), key=lambda x: x[1])
        else:
            best_util = None

        if comm_overhead:
            lowest_comm = min(comm_overhead.items(), key=lambda x: x[1])
        else:
            lowest_comm = None

        analysis['summary'] = {
            'best_latency': best_latency,
            'best_utilization': best_util,
            'lowest_communication': lowest_comm
        }

        # Save to file
        output_path = Path(output_dir) / 'baseline_analysis.json'
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)

        # Log final analysis
        if logger:
            logger.log_metrics({'baseline_analysis': analysis})

        return analysis

    except Exception as e:
        if logger:
            logger.log_error("analysis_error", f"Error analyzing baseline results: {str(e)}")
        return {'error': str(e)}

