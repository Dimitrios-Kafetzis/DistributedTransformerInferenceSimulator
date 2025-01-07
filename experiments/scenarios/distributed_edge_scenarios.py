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
# File:    experiments/scenarios/distributed_edge_scenarios.py
# Description:
#   Defines scenario classes for a distributed edge setting with many
#   devices, each having varying capabilities and limited interconnect
#   bandwidth, to test Transformer inference in wide-area edge networks.
#
# ---------------------------------------------------------------------------

"""
Implements classes that model distributed edge environments, where multiple
geographically dispersed devices must collaboratively run transformer inference
under bandwidth and latency constraints.
"""

from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

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
    DistributedEdgeTopology,
    WorkloadType
)
# Make sure SequenceConfig is available:
from src.environment.workload import SequenceConfig


class DistributedEdgeBaseScenario(BaseScenario):
    """
    Base scenario class for distributed edge experiments.
    Provides a shared setup_basic_environment() to avoid code duplication.
    """

    def setup_basic_environment(self) -> None:
        """
        Creates a distributed edge topology, resource distribution,
        and sets up workloads (possibly multiple) based on config.
        """
        if self.logger:
            self.logger.log_event("setup", "Setting up basic environment for distributed edge")

        # 1. Create network topology
        topo_generator = DistributedEdgeTopology(self.config.network)
        self.network = topo_generator.generate()

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
            num_devices=self.config.network.num_devices,  # e.g. 16
            memory_distribution=mem_dist,
            compute_distribution=comp_dist,
            seed=self.config.resources.seed
        )
        self.device_capabilities = resource_distributor.generate_capabilities()

        # 3. Create devices
        self.devices = {}
        for device_id, caps in self.device_capabilities.items():
            self.devices[device_id] = Device(
                device_id=device_id,
                memory_capacity=caps.memory_capacity,
                compute_capacity=caps.compute_capacity,
                is_source=caps.is_source
            )

        # 4. Workload generation
        self.workload_generator = WorkloadGenerator(seed=self.config.workload.seed)

        # Some distributed_edge configs have multiple model_types (e.g. ["SMALL", "MEDIUM"])
        # If that’s the case, we store them in self.workloads:
        self.workloads = []
        # Attempt to read e.g. self.config.workload.model_types
        # If not present, fallback to a single model_type = self.config.workload.model_type
        # Then define or pick a default SequenceConfig.
        if hasattr(self.config.workload, 'model_types') and self.config.workload.model_types:
            # multiple model types
            for mtype in self.config.workload.model_types:
                # define a default SequenceConfig (first length, first num_steps) if none
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
            # single model type
            # define default SequenceConfig if needed
            seq_cfg = SequenceConfig(
                initial_length=self.config.workload.initial_sequence_lengths[0],
                num_steps=self.config.workload.generation_steps[0],
                precision_bytes=self.config.workload.precision_bytes
            )
            single_type = (self.config.workload.model_type
                           if hasattr(self.config.workload, 'model_type')
                           else WorkloadType.SMALL)
            w = self.workload_generator.generate_workload(
                workload_type=single_type,
                sequence_config=seq_cfg
            )
            self.workloads.append(w)

        # 5. Validate scenario requirements with a fallback transformer
        test_transformer = Transformer(self.workloads[0].transformer.config)
        if not validate_scenario_requirements(
            self.config,
            self.network,
            self.devices,
            test_transformer
        ):
            raise ValueError("Scenario requirements not met in distributed edge environment setup.")

        # 6. Initialize a ResourceAwareDistributor for the first workload
        #    (We can reassign .transformer for others if multiple)
        self.distributor = ResourceAwareDistributor(
            self.workloads[0].transformer,
            self.network,
            self.devices
        )


class DistributedEdgeBasicScenario(DistributedEdgeBaseScenario):
    """
    Basic distributed edge scenario testing operation with N=16 devices.
    """

    def setup(self) -> None:
        """Set up the distributed edge environment"""
        if self.logger:
            self.logger.log_event("setup", "Setting up distributed edge basic scenario")
        self.setup_basic_environment()
        
    def run(self) -> ScenarioResult:
        """
        Run the basic distributed edge scenario across possibly multiple workloads,
        merging their resource/performance metrics into top-level dicts.
        """
        metrics = {
            'resource_metrics': {},
            'communication_metrics': {},
            'performance_metrics': {},
            'model_comparison': {}
        }
    
        try:
            # If self.workloads has multiple (e.g. SMALL, MEDIUM), run each
            for idx, workload in enumerate(self.workloads):
                # reassign transformer
                self.distributor.transformer = workload.transformer
    
                # run the workload
                w_metrics = self._run_single_workload(workload, idx)
    
                # Merge sub-workload resource/performance metrics
                for step, usage in w_metrics['resource_metrics'].items():
                    metrics['resource_metrics'][(idx, step)] = usage
                for step, usage in w_metrics['communication_metrics'].items():
                    metrics['communication_metrics'][(idx, step)] = usage
                for step, usage in w_metrics['performance_metrics'].items():
                    metrics['performance_metrics'][(idx, step)] = usage
    
                # Compute an example average latency
                lat_list = [m['latency'] for m in w_metrics['performance_metrics'].values()]
                avg_latency = float(np.mean(lat_list)) if lat_list else 0.0
    
                model_name = workload.workload_type.name
                # For demonstration, store a "resource_usage_sample" from step 0
                usage_sample = w_metrics['resource_metrics'].get(0, {})
    
                metrics['model_comparison'][f"model_{idx}"] = {
                    'type': model_name,
                    'average_latency': avg_latency,
                    'resource_usage_sample': usage_sample
                }
    
            final_metrics = collect_scenario_metrics(
                resource_metrics=metrics['resource_metrics'],
                communication_metrics=metrics['communication_metrics'],
                performance_metrics=metrics['performance_metrics']
            )
            # Add the sub-dict
            final_metrics['model_comparison'] = metrics['model_comparison']
    
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
            self.logger.log_event("cleanup", "Cleaning up distributed edge basic scenario")

    def _run_single_workload(self, workload, idx: int) -> Dict:
        """
        Runs a single workload from step=0..num_steps-1, returns step-wise metrics.
        """
        w_metrics = {
            'resource_metrics': {},
            'communication_metrics': {},
            'performance_metrics': {}
        }

        # Ensure workload has sequence_config
        if not workload.sequence_config:
            raise RuntimeError(f"Workload index={idx} has no sequence_config; cannot run steps.")

        for step in range(workload.sequence_config.num_steps):
            assignment_result = self.distributor.compute_assignment(
                generation_step=step,
                previous_assignments=w_metrics.get('previous_assignments'),
                previous_cache=w_metrics.get('previous_cache')
            )
            if not assignment_result.is_feasible:
                raise RuntimeError(f"Infeasible assignment at step {step} for workload idx={idx}")

            # record resource usage
            w_metrics['resource_metrics'][step] = assignment_result.resource_usage
            w_metrics['previous_assignments'] = assignment_result.component_assignments
            w_metrics['previous_cache'] = assignment_result.cache_assignments
            w_metrics['performance_metrics'][step] = {
                'latency': assignment_result.estimated_latency,
                'step': step
            }

        return w_metrics


class DistributedEdgeCommunicationScenario(DistributedEdgeBaseScenario):
    """
    Tests communication patterns and bandwidth utilization in a distributed setup
    """

    def setup(self) -> None:
        """Set up communication test environment"""
        if self.logger:
            self.logger.log_event("setup", "Setting up distributed edge communication scenario")
        self.setup_basic_environment()

        # Example: track communication or bandwidth usage
        self.communication_tracker = {
            'transfers': [],
            'bandwidth_usage': {},
            'network_congestion': {}
        }

        # Initialize placeholders for link usage
        for (src, dst), link in self.network.links.items():
            self.communication_tracker['bandwidth_usage'][(src, dst)] = []
            self.communication_tracker['network_congestion'][(src, dst)] = []

        # For simplicity, pick the first workload as the main scenario workload
        # or define multiple if you prefer
        if self.workloads:
            self.main_workload = self.workloads[0]
            self.distributor.transformer = self.main_workload.transformer
        else:
            self.main_workload = None
            self.distributor = None

    def run(self) -> ScenarioResult:
        """Run communication pattern analysis"""
        metrics = {
            'resource_metrics': {},
            'communication_metrics': {},
            'performance_metrics': {},
            'network_analysis': {}
        }

        if not self.distributor or not self.main_workload:
            return ScenarioResult(
                scenario_name=self.__class__.__name__,
                start_time=datetime.now(),
                end_time=datetime.now(),
                metrics={},
                success=False,
                error="No main workload or distributor defined."
            )

        try:
            for step in range(self.main_workload.sequence_config.num_steps):
                # Before assignment, record link states
                self._record_network_state()

                assignment_result = self.distributor.compute_assignment(
                    generation_step=step,
                    previous_assignments=metrics.get('previous_assignments'),
                    previous_cache=metrics.get('previous_cache')
                )

                # record potential "migration" or "component" movements
                self._track_communication(
                    step,
                    assignment_result.component_assignments,
                    metrics.get('previous_assignments', {})
                )

                # update metrics
                metrics['resource_metrics'][step] = assignment_result.resource_usage
                metrics['previous_assignments'] = assignment_result.component_assignments
                metrics['previous_cache'] = assignment_result.cache_assignments
                metrics['performance_metrics'][step] = {
                    'latency': assignment_result.estimated_latency,
                    'step': step
                }

            # After all steps, analyze
            metrics['network_analysis'] = self._analyze_communication_patterns()

            final_metrics = collect_scenario_metrics(
                resource_metrics=metrics['resource_metrics'],
                communication_metrics=metrics['communication_metrics'],
                performance_metrics=metrics['performance_metrics']
            )
            # add "network_analysis" if you want
            final_metrics['network_analysis'] = metrics['network_analysis']

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
            self.logger.log_event("cleanup", "Cleaning up distributed edge communication scenario")

    def _record_network_state(self) -> None:
        """Record current network usage / congestion before assignment"""
        for (src, dst), link in self.network.links.items():
            used_ratio = link.used_bandwidth / link.bandwidth if link.bandwidth > 0 else 0
            congestion = max(0, 1 - link.available_bandwidth / link.bandwidth) if link.bandwidth > 0 else 1
            self.communication_tracker['bandwidth_usage'][(src, dst)].append(used_ratio)
            self.communication_tracker['network_congestion'][(src, dst)].append(congestion)

    def _track_communication(
        self,
        step: int,
        current_assignments: Dict[str, str],
        previous_assignments: Dict[str, str]
    ) -> None:
        """Identify if any component was migrated and record it."""
        for comp_id, dev_id in current_assignments.items():
            if comp_id in previous_assignments:
                prev_dev = previous_assignments[comp_id]
                if dev_id != prev_dev:
                    # record migration
                    self.communication_tracker['transfers'].append({
                        'step': step,
                        'component': comp_id,
                        'source': prev_dev,
                        'target': dev_id
                    })

    def _analyze_communication_patterns(self) -> Dict:
        """Compute summary stats from self.communication_tracker"""
        total_migrations = len(self.communication_tracker['transfers'])
        bandwidth_usage = {
            link: float(np.mean(usages))
            for link, usages in self.communication_tracker['bandwidth_usage'].items()
        }
        congestion_map = {
            link: float(np.mean(vals))
            for link, vals in self.communication_tracker['network_congestion'].items()
        }
        return {
            'total_transfers': total_migrations,
            'bandwidth_utilization': bandwidth_usage,
            'congestion_hotspots': congestion_map
        }


class DistributedEdgeHeterogeneityScenario(DistributedEdgeBaseScenario):
    """
    Tests system behavior with heterogeneous device capabilities
    """

    def setup(self) -> None:
        """Set up heterogeneous environment"""
        if self.logger:
            self.logger.log_event("setup", "Setting up distributed edge heterogeneity scenario")
        self.setup_basic_environment()

        # For demonstration, group devices by some threshold
        self.device_groups = {
            'high_capacity': [],
            'medium_capacity': [],
            'low_capacity': []
        }

        # Suppose you define a threshold for compute capacity:
        # (you can tune these thresholds as you like)
        for device_id, dev in self.devices.items():
            if dev.compute.capacity >= 200:  # e.g. 200 GFLOPS
                self.device_groups['high_capacity'].append(device_id)
            elif dev.compute.capacity >= 100:
                self.device_groups['medium_capacity'].append(device_id)
            else:
                self.device_groups['low_capacity'].append(device_id)

        # If multiple workloads, pick just the first for demonstration
        if self.workloads:
            self.main_workload = self.workloads[0]
            self.distributor.transformer = self.main_workload.transformer
        else:
            self.main_workload = None
            self.distributor = None

    def run(self) -> ScenarioResult:
        """Run heterogeneity analysis"""
        metrics = {
            'resource_metrics': {},
            'communication_metrics': {},
            'performance_metrics': {},
            'heterogeneity_analysis': {}
        }

        if not self.distributor or not self.main_workload:
            return ScenarioResult(
                scenario_name=self.__class__.__name__,
                start_time=datetime.now(),
                end_time=datetime.now(),
                metrics={},
                success=False,
                error="No main workload or distributor to run."
            )
        
        try:
            # track group metrics
            group_metrics = {
                group: {
                    'utilization': [],
                    'assignments': []
                }
                for group in self.device_groups
            }

            for step in range(self.main_workload.sequence_config.num_steps):
                assignment_result = self.distributor.compute_assignment(
                    generation_step=step,
                    previous_assignments=metrics.get('previous_assignments'),
                    previous_cache=metrics.get('previous_cache')
                )

                if not assignment_result.is_feasible:
                    raise RuntimeError(f"Infeasible assignment at step {step} in heterogeneity test.")

                # record resource usage
                metrics['resource_metrics'][step] = assignment_result.resource_usage
                metrics['previous_assignments'] = assignment_result.component_assignments
                metrics['previous_cache'] = assignment_result.cache_assignments
                metrics['performance_metrics'][step] = {
                    'latency': assignment_result.estimated_latency,
                    'step': step
                }

                # record group-level data
                self._record_group_metrics(assignment_result, group_metrics)

            # analyze heterogeneity
            metrics['heterogeneity_analysis'] = self._analyze_heterogeneity(group_metrics)

            final_metrics = collect_scenario_metrics(
                resource_metrics=metrics['resource_metrics'],
                communication_metrics=metrics['communication_metrics'],
                performance_metrics=metrics['performance_metrics']
            )
            # embed the analysis
            final_metrics['heterogeneity_analysis'] = metrics['heterogeneity_analysis']

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
            self.logger.log_event("cleanup", "Cleaning up distributed edge heterogeneity scenario")

    def _record_group_metrics(self, assignment_result, group_metrics):
        """Aggregate usage or assignment stats by group."""
        # compute utilization by group (e.g. average compute utilization):
        for group_name, device_ids in self.device_groups.items():
            if not device_ids:
                group_metrics[group_name]['utilization'].append(0.0)
                group_metrics[group_name]['assignments'].append(0)
                continue

            # average compute_utilization for those devices
            util_vals = []
            for d_id in device_ids:
                dev_usage = assignment_result.resource_usage.get(d_id, {})
                if 'compute_utilization' in dev_usage:
                    util_vals.append(dev_usage['compute_utilization'])
            avg_util = float(np.mean(util_vals)) if util_vals else 0.0
            group_metrics[group_name]['utilization'].append(avg_util)

            # count how many components assigned
            assign_count = sum(
                1
                for comp, dev_id in assignment_result.component_assignments.items()
                if dev_id in device_ids
            )
            group_metrics[group_name]['assignments'].append(assign_count)

    def _analyze_heterogeneity(self, group_metrics) -> Dict:
        """Analyze group-level stats to see if distribution is balanced."""
        analysis = {}
        for group_name, data in group_metrics.items():
            util_arr = np.array(data['utilization']) if data['utilization'] else np.array([0])
            assign_arr = np.array(data['assignments']) if data['assignments'] else np.array([0])

            analysis[group_name] = {
                'average_utilization': float(np.mean(util_arr)),
                'utilization_std': float(np.std(util_arr)),
                'average_assignment': float(np.mean(assign_arr)),
                'assignment_std': float(np.std(assign_arr))
            }
        return analysis
