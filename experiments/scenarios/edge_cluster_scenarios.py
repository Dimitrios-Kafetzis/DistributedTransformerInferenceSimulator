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
# File:    experiments/scenarios/edge_cluster_scenarios.py
# Description:
#   Implements scenario classes focused on edge cluster deployments, where
#   a small number of devices are interconnected with moderate to high
#   bandwidth, simulating compact edge data centers for Transformer inference.
#
# ---------------------------------------------------------------------------

"""
Defines scenario classes for evaluating distributed inference on a small to
medium-scale edge cluster, ensuring resource constraints and hierarchical
topologies are properly tested.
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
    EdgeClusterTopology,
    WorkloadType
)
# Make sure you have a SequenceConfig (or similar) in your code:
from src.environment.workload import SequenceConfig


class EdgeClusterBaseScenario(BaseScenario):
    """
    Base scenario for edge cluster that provides a shared
    setup_basic_environment() method used by multiple specialized scenarios.
    """

    def setup_basic_environment(self) -> None:
        """
        Create network topology, resource distribution, and a default workload.
        This method is called by other edge cluster scenarios to avoid code duplication.
        """
        if self.logger:
            self.logger.log_event("setup", "Setting up basic environment for edge cluster")

        # 1. Create network topology
        topology_generator = EdgeClusterTopology(self.config.network)
        self.network = topology_generator.generate()

        # 2. Set up resource distribution
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
            num_devices=self.config.network.num_devices,  # e.g. 8
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

        # 4. Set up a default single workload if config has a single model_type
        self.workload_generator = WorkloadGenerator(seed=self.config.workload.seed)

        # If the config explicitly has a single "model_type":
        # (like edge_cluster.yaml does)
        if hasattr(self.config.workload, "model_type"):
            # For example, define a default SequenceConfig if none is in your data
            default_seq_config = SequenceConfig(
                initial_length=self.config.workload.initial_sequence_lengths[0],
                num_steps=self.config.workload.generation_steps[0],
                precision_bytes=self.config.workload.precision_bytes
            )
            self.workload = self.workload_generator.generate_workload(
                workload_type=self.config.workload.model_type,
                sequence_config=default_seq_config
            )
        else:
            self.workload = None

        # 5. Validate scenario requirements
        # If we have a single workload, we can pass its transformer's config
        # or a placeholder:
        temp_transformer = None
        if self.workload:
            temp_transformer = self.workload.transformer
        else:
            # Or create a small fallback transformer
            temp_transformer = Transformer(
                config=self.workload_generator.model_configs.get(
                    WorkloadType.SMALL
                )
            )

        if not validate_scenario_requirements(
            self.config,
            self.network,
            self.devices,
            temp_transformer
        ):
            raise ValueError("Scenario requirements not met in basic environment setup")

        # 6. Create a ResourceAwareDistributor if we have a single workload
        if self.workload:
            self.distributor = ResourceAwareDistributor(
                self.workload.transformer,
                self.network,
                self.devices
            )
        else:
            self.distributor = None


class EdgeClusterBasicScenario(EdgeClusterBaseScenario):
    """
    Basic edge cluster scenario testing standard operation with 8 devices
    """
    
    def setup(self) -> None:
        """Set up the edge cluster environment"""
        if self.logger:
            self.logger.log_event("setup", "Setting up edge cluster basic scenario")
            
        self.setup_basic_environment()
            
    def run(self) -> ScenarioResult:
        """Run the basic scenario"""
        if self.logger:
            self.logger.log_event("run", "Starting edge cluster basic scenario")
            
        if not hasattr(self, 'distributor') or not self.distributor:
            raise RuntimeError("No distributor is defined. Did you define a workload?")

        if not self.workload or not self.workload.sequence_config:
            raise RuntimeError("No valid workload or sequence_config found.")

        metrics = {
            'resource_metrics': {},
            'communication_metrics': {},
            'performance_metrics': {}
        }
        
        try:
            # Example usage: run through generation steps
            for step in range(self.workload.sequence_config.num_steps):
                assignment_result = self.distributor.compute_assignment(
                    generation_step=step,
                    previous_assignments=metrics.get('previous_assignments'),
                    previous_cache=metrics.get('previous_cache')
                )
                
                if not assignment_result.is_feasible:
                    raise RuntimeError(f"Infeasible assignment at step {step}")
                    
                # Record metrics
                metrics['resource_metrics'][step] = assignment_result.resource_usage
                metrics['previous_assignments'] = assignment_result.component_assignments
                metrics['previous_cache'] = assignment_result.cache_assignments
                
                # Performance
                metrics['performance_metrics'][step] = {
                    'latency': assignment_result.estimated_latency,
                    'step': step
                }
                
                if self.logger:
                    self.logger.log_metrics({
                        'step': step,
                        'latency': assignment_result.estimated_latency,
                        'resource_usage': assignment_result.resource_usage
                    })
                    
            # Collect final scenario metrics
            return ScenarioResult(
                scenario_name=self.__class__.__name__,
                start_time=datetime.now(),
                end_time=datetime.now(),
                metrics=collect_scenario_metrics(
                    resource_metrics=metrics['resource_metrics'],
                    communication_metrics=metrics['communication_metrics'],
                    performance_metrics=metrics['performance_metrics']
                ),
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
            self.logger.log_event("cleanup", "Cleaning up edge cluster basic scenario")
        # Cleanup resources if needed
        pass


class EdgeClusterScalabilityScenario(EdgeClusterBaseScenario):
    """
    Tests scalability of edge cluster by gradually increasing workload
    """
    
    def setup(self) -> None:
        """Set up scalability test environment"""
        if self.logger:
            self.logger.log_event("setup", "Setting up scalability environment")
        self.setup_basic_environment()

        # Suppose you want multiple workloads of different sequence lengths
        self.workloads = []
        # We assume the user wants to scale from e.g. [128, 256, 512]
        # or read from config
        sequence_lengths = self.config.workload.initial_sequence_lengths
        for length in sequence_lengths:
            seq_cfg = SequenceConfig(
                initial_length=length,
                num_steps=self.config.workload.generation_steps[0],
                precision_bytes=self.config.workload.precision_bytes
            )
            # e.g. always WorkloadType.SMALL for demonstration
            w = self.workload_generator.generate_workload(
                workload_type=self.config.workload.model_type,
                sequence_config=seq_cfg
            )
            self.workloads.append(w)

        # Create a single distributor referencing the first workload's transformer
        # We'll reassign the transform for each workload in run()
        if self.workloads:
            self.distributor = ResourceAwareDistributor(
                self.workloads[0].transformer,
                self.network,
                self.devices
            )
        else:
            self.distributor = None

    def run(self) -> ScenarioResult:
        """Run scalability tests"""
        if not self.distributor:
            raise RuntimeError("No distributor is defined. Possibly no workloads generated.")

        metrics = {
            'resource_metrics': {},
            'communication_metrics': {},
            'performance_metrics': {},
            'scalability_metrics': {}
        }
        
        try:
            # Test each workload size
            for idx, workload in enumerate(self.workloads):
                # Reassign the transform for the distributor
                self.distributor.transformer = workload.transformer

                # Actually run the workload
                workload_metrics = self._run_single_workload(workload, idx)

                # Suppose we store some aggregated info
                latencies = [m['latency'] for m in workload_metrics['performance_metrics'].values()]
                average_latency = float(np.mean(latencies)) if latencies else 0.0

                resource_usages = workload_metrics['resource_metrics'].values()
                # example: peak memory usage across steps
                # let's define "peak_mem" as the sum of device usage at the step's maximum
                # or something. We'll do a simple approach:
                peak_mem = 0.0
                for usage_dict in resource_usages:
                    # usage_dict is e.g. { 'device_0': {'memory_used':..., 'compute_used':...}... }
                    step_total = 0.0
                    for dev_id, usage in usage_dict.items():
                        step_total += usage.get('memory_used', 0)
                    if step_total > peak_mem:
                        peak_mem = step_total

                metrics['scalability_metrics'][idx] = {
                    'sequence_length': workload.sequence_config.initial_length,
                    'average_latency': average_latency,
                    'peak_memory': peak_mem
                }

            # Convert partial metrics to scenario result
            scenario_metrics = collect_scenario_metrics(
                resource_metrics=metrics['resource_metrics'],
                communication_metrics=metrics['communication_metrics'],
                performance_metrics=metrics['performance_metrics']
            )
            # Merge in the 'scalability_metrics' if you want:
            scenario_metrics['scalability'] = metrics['scalability_metrics']

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
        """Clean up resources"""
        if self.logger:
            self.logger.log_event("cleanup", "Cleaning up edge cluster basic scenario")
        # Additional cleanup if needed
        pass
        
    def _run_single_workload(self, workload, workload_idx: int) -> Dict:
        """Run a single workload configuration, collecting step-by-step metrics"""
        metrics = {
            'resource_metrics': {},
            'communication_metrics': {},
            'performance_metrics': {}
        }

        if not workload.sequence_config:
            raise RuntimeError(f"Workload idx={workload_idx} has no sequence_config")

        for step in range(workload.sequence_config.num_steps):
            assignment_result = self.distributor.compute_assignment(
                generation_step=step,
                previous_assignments=metrics.get('previous_assignments'),
                previous_cache=metrics.get('previous_cache')
            )
            
            if not assignment_result.is_feasible:
                raise RuntimeError(
                    f"Infeasible assignment at step {step} for workload {workload_idx}"
                )
                
            # Record resource usage
            metrics['resource_metrics'][step] = assignment_result.resource_usage
            metrics['previous_assignments'] = assignment_result.component_assignments
            metrics['previous_cache'] = assignment_result.cache_assignments

            # Record performance
            metrics['performance_metrics'][step] = {
                'latency': assignment_result.estimated_latency,
                'step': step
            }
            
        return metrics


class EdgeClusterFailureScenario(EdgeClusterBaseScenario):
    """
    Tests edge cluster behavior under device failures
    """
    
    def setup(self) -> None:
        """Set up failure test environment"""
        if self.logger:
            self.logger.log_event("setup", "Setting up basic environment for edge cluster with failures")
        self.setup_basic_environment()
        
        # Define failure scenarios
        self.failure_schedule = [
            {'step': 10, 'device': 'device_4'},  # Edge device failure
            {'step': 20, 'device': 'device_2'},  # Mid-tier device failure
            {'step': 30, 'device': 'device_1'}   # Another mid-tier failure
        ]

    def run(self) -> ScenarioResult:
        """Run failure scenario tests"""
        if not hasattr(self, 'distributor') or not self.distributor:
            raise RuntimeError("No distributor is defined. Did you define a workload?")

        if not self.workload or not self.workload.sequence_config:
            raise RuntimeError("No valid workload or sequence_config found for failure scenario.")

        metrics = {
            'resource_metrics': {},
            'communication_metrics': {},
            'performance_metrics': {},
            'failure_metrics': {}
        }
        
        try:
            # Run through generation steps with failures
            for step in range(self.workload.sequence_config.num_steps):
                # Check for device failures
                self._handle_failures(step)
                
                # Compute assignment
                assignment_result = self.distributor.compute_assignment(
                    generation_step=step,
                    previous_assignments=metrics.get('previous_assignments'),
                    previous_cache=metrics.get('previous_cache')
                )
                
                if not assignment_result.is_feasible:
                    # Log failure recovery attempt
                    metrics['failure_metrics'][step] = {
                        'recovery_attempted': True,
                        'recovery_successful': False
                    }
                    raise RuntimeError(f"Failed to recover at step {step}")
                    
                # Record metrics
                metrics['resource_metrics'][step] = assignment_result.resource_usage
                metrics['previous_assignments'] = assignment_result.component_assignments
                metrics['previous_cache'] = assignment_result.cache_assignments

                # Performance
                metrics['performance_metrics'][step] = {
                    'latency': assignment_result.estimated_latency,
                    'step': step
                }
                
                # Record failure recovery if applicable
                if step in [f['step'] for f in self.failure_schedule]:
                    metrics['failure_metrics'][step] = {
                        'recovery_attempted': True,
                        'recovery_successful': True,
                        'recovery_latency': assignment_result.estimated_latency
                    }
                    
            # Gather final scenario metrics
            final_metrics = collect_scenario_metrics(
                resource_metrics=metrics['resource_metrics'],
                communication_metrics=metrics['communication_metrics'],
                performance_metrics=metrics['performance_metrics']
            )
            # Merge in failure_metrics if needed
            final_metrics['failure_metrics'] = metrics['failure_metrics']

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
            self.logger.log_event("cleanup", "Cleaning up edge cluster basic scenario")
        # Additional cleanup if needed
        pass
        
    def _handle_failures(self, step: int) -> None:
        """Handle scheduled device failures"""
        for failure in self.failure_schedule:
            if failure['step'] == step:
                device_id = failure['device']
                if self.logger:
                    self.logger.log_event(
                        "device_failure",
                        f"Device {device_id} failed at step {step}"
                    )
                    
                # Remove device from available devices
                if device_id in self.devices:
                    del self.devices[device_id]
                    
                # Update network topology
                if device_id in self.network.topology.nodes:
                    self.network.topology.remove_node(device_id)

