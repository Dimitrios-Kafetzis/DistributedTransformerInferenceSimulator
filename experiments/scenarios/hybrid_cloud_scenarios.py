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

from .common import BaseScenario, ScenarioResult, validate_scenario_requirements, collect_scenario_metrics
from src.core import Network, Device, Transformer
from src.algorithms import ResourceAwareDistributor
from src.environment import (
    NetworkTopologyGenerator,
    ResourceDistributor,
    WorkloadGenerator,
    HybridCloudEdgeTopology,
    WorkloadType
)

class HybridCloudBasicScenario(BaseScenario):
    """
    Basic hybrid cloud-edge scenario testing operation with 24 devices
    across cloud, regional, and edge tiers
    """
    
    def setup(self) -> None:
        """Set up the hybrid cloud-edge environment"""
        if self.logger:
            self.logger.log_event("setup", "Setting up hybrid cloud-edge basic scenario")
            
        # Create network topology
        topology_generator = HybridCloudEdgeTopology(self.config.network)
        self.network = topology_generator.generate()
        
        # Set up resource distribution
        resource_distributor = ResourceDistributor(
            num_devices=24,
            memory_distribution=self.config.resources.memory_distribution,
            compute_distribution=self.config.resources.compute_distribution
        )
        self.device_capabilities = resource_distributor.generate_capabilities()
        
        # Create devices
        self.devices = {
            device_id: Device(
                device_id=device_id,
                memory_capacity=caps.memory_capacity,
                compute_capacity=caps.compute_capacity,
                is_source=caps.is_source
            )
            for device_id, caps in self.device_capabilities.items()
        }
        
        # Categorize devices by tier
        self.device_tiers = {
            'cloud': self.devices.keys()[:4],     # First 4 devices are cloud
            'regional': self.devices.keys()[4:12], # Next 8 are regional
            'edge': self.devices.keys()[12:]       # Last 12 are edge
        }
        
        # Set up workloads - using all model sizes
        self.workload_generator = WorkloadGenerator()
        self.workloads = [
            self.workload_generator.generate_workload(
                workload_type,
                self.config.workload.sequence_config
            )
            for workload_type in WorkloadType
        ]
        
        # Initialize algorithm
        self.distributor = ResourceAwareDistributor(
            self.workloads[0].transformer,  # Start with small model
            self.network,
            self.devices
        )
        
    def run(self) -> ScenarioResult:
        """Run the basic hybrid cloud-edge scenario"""
        metrics = {
            'resource_metrics': {},
            'communication_metrics': {},
            'performance_metrics': {},
            'tier_metrics': defaultdict(dict)
        }
        
        try:
            # Run each workload type
            for workload_idx, workload in enumerate(self.workloads):
                self.distributor.transformer = workload.transformer
                workload_metrics = self._run_workload_with_tier_tracking(
                    workload,
                    workload_idx
                )
                
                # Record workload-specific metrics
                metrics[f'workload_{workload_idx}'] = workload_metrics
                
                # Update tier-specific metrics
                self._update_tier_metrics(workload_metrics, metrics['tier_metrics'])
                
            return ScenarioResult(
                scenario_name=self.__class__.__name__,
                start_time=datetime.now(),
                end_time=datetime.now(),
                metrics=collect_scenario_metrics(**metrics),
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
        """Update metrics for each tier"""
        for tier, devices in self.device_tiers.items():
            # Calculate tier utilization
            tier_metrics[tier]['utilization'] = np.mean([
                workload_metrics['resource_metrics'][step][device_id]['compute_utilization']
                for step in workload_metrics['resource_metrics']
                for device_id in devices
            ])
            
            # Calculate tier assignment frequency
            tier_metrics[tier]['assignments'] = sum(
                1 for assignments in workload_metrics['component_assignments'].values()
                for device_id in assignments.values()
                if device_id in devices
            )

class HybridCloudTierBalancingScenario(BaseScenario):
    """
    Tests workload balancing across cloud, regional, and edge tiers
    """
    
    def setup(self) -> None:
        """Set up tier balancing test environment"""
        super().setup()
        
        # Initialize tier tracking
        self.tier_tracker = {
            tier: {
                'compute_usage': [],
                'memory_usage': [],
                'bandwidth_usage': [],
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
            # Test each model size
            for workload in self.workloads:
                self.distributor.transformer = workload.transformer
                tier_metrics = self._run_tier_balanced_workload(workload)
                
                # Record tier-specific metrics
                metrics['tier_balance_metrics'][workload.workload_type.name] = \
                    self._analyze_tier_balance(tier_metrics)
                
            return ScenarioResult(
                scenario_name=self.__class__.__name__,
                start_time=datetime.now(),
                end_time=datetime.now(),
                metrics=collect_scenario_metrics(**metrics),
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
            
    def _run_tier_balanced_workload(self, workload) -> Dict:
        """Run workload with tier balancing analysis"""
        tier_metrics = defaultdict(lambda: defaultdict(list))
        
        for step in range(workload.sequence_config.num_steps):
            assignment_result = self.distributor.compute_assignment(
                step,
                previous_assignments=tier_metrics.get('previous_assignments'),
                previous_cache=tier_metrics.get('previous_cache')
            )
            
            # Track tier-specific metrics
            for tier, devices in self.device_tiers.items():
                self._track_tier_metrics(
                    tier,
                    devices,
                    assignment_result,
                    tier_metrics[tier]
                )
                
            # Update assignments
            tier_metrics['previous_assignments'] = assignment_result.component_assignments
            tier_metrics['previous_cache'] = assignment_result.cache_assignments
            
        return tier_metrics
            
    def _track_tier_metrics(
        self,
        tier: str,
        devices: Set[str],
        assignment_result: Dict,
        metrics: Dict
    ) -> None:
        """Track metrics for a specific tier"""
        # Resource usage
        metrics['compute_usage'].append(np.mean([
            assignment_result.resource_usage[dev_id]['compute_utilization']
            for dev_id in devices
        ]))
        
        metrics['memory_usage'].append(np.mean([
            assignment_result.resource_usage[dev_id]['memory_utilization']
            for dev_id in devices
        ]))
        
        # Component assignments
        for comp_id, dev_id in assignment_result.component_assignments.items():
            if dev_id in devices:
                metrics['component_assignments'][comp_id] += 1
                
    def _analyze_tier_balance(self, tier_metrics: Dict) -> Dict:
        """Analyze tier balance metrics"""
        return {
            tier: {
                'average_compute': np.mean(metrics['compute_usage']),
                'average_memory': np.mean(metrics['memory_usage']),
                'assignment_distribution': dict(metrics['component_assignments']),
                'load_balance_index': self._calculate_balance_index(
                    metrics['compute_usage']
                )
            }
            for tier, metrics in tier_metrics.items()
        }
        
    def _calculate_balance_index(self, utilization_values: List[float]) -> float:
        """Calculate load balance index (0 = perfect balance, 1 = imbalanced)"""
        if not utilization_values:
            return 0.0
        mean_util = np.mean(utilization_values)
        if mean_util == 0:
            return 0.0
        return np.std(utilization_values) / mean_util

class HybridCloudLatencyScenario(BaseScenario):
    """
    Tests latency characteristics across different tiers and bandwidths
    """
    
    def setup(self) -> None:
        """Set up latency test environment"""
        super().setup()
        
        # Initialize latency tracking
        self.latency_tracker = {
            'intra_tier': defaultdict(list),   # Within same tier
            'inter_tier': defaultdict(list),   # Between tiers
            'edge_to_cloud': defaultdict(list) # Edge to cloud specific
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
            # Run workloads with latency tracking
            for workload in self.workloads:
                self.distributor.transformer = workload.transformer
                latency_metrics = self._run_latency_analysis(workload)
                
                # Record latency metrics
                metrics['latency_analysis'][workload.workload_type.name] = \
                    self._analyze_latency_patterns(latency_metrics)
                
            return ScenarioResult(
                scenario_name=self.__class__.__name__,
                start_time=datetime.now(),
                end_time=datetime.now(),
                metrics=collect_scenario_metrics(**metrics),
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
            
    def _run_latency_analysis(self, workload) -> Dict:
        """Run workload with detailed latency tracking"""
        latency_metrics = {
            'communication_latency': defaultdict(list),
            'processing_latency': defaultdict(list),
            'total_latency': defaultdict(list)
        }
        
        for step in range(workload.sequence_config.num_steps):
            # Get assignments
            assignment_result = self.distributor.compute_assignment(
                step,
                previous_assignments=latency_metrics.get('previous_assignments'),
                previous_cache=latency_metrics.get('previous_cache')
            )
            
            # Track latencies
            self._track_latencies(
                step,
                assignment_result,
                latency_metrics
            )
            
            # Update assignments
            latency_metrics['previous_assignments'] = \
                assignment_result.component_assignments
            latency_metrics['previous_cache'] = assignment_result.cache_assignments
            
        return latency_metrics
            
    def _track_latencies(
        self,
        step: int,
        assignment_result: Dict,
        metrics: Dict
    ) -> None:
        """Track various types of latencies"""
        # Track communication latencies between tiers
        for comp_id, dev_id in assignment_result.component_assignments.items():
            source_tier = self._get_device_tier(dev_id)
            
            # Track dependencies
            for dep_id in self._get_component_dependencies(comp_id):
                if dep_id in assignment_result.component_assignments:
                    dep_dev_id = assignment_result.component_assignments[dep_id]
                    dep_tier = self._get_device_tier(dep_dev_id)
                    
                    # Calculate communication latency
                    latency = self.network.calculate_transfer_time(
                        dev_id,
                        dep_dev_id,
                        self._estimate_transfer_size(comp_id, dep_id)
                    )
                    
                    if source_tier == dep_tier:
                        metrics['communication_latency']['intra_tier'].append(latency)
                    else:
                        metrics['communication_latency']['inter_tier'].append(latency)
                        
                        if source_tier == 'edge' and dep_tier == 'cloud':
                            metrics['communication_latency']['edge_to_cloud'].append(latency)
                            
    def _get_device_tier(self, device_id: str) -> str:
        """Get the tier of a device"""
        for tier, devices in self.device_tiers.items():
            if device_id in devices:
                return tier
        return 'unknown'
        
    def _analyze_latency_patterns(self, metrics: Dict) -> Dict:
        """Analyze latency patterns across tiers"""
        return {
            'intra_tier_latency': {
                'mean': np.mean(metrics['communication_latency']['intra_tier']),
                'std': np.std(metrics['communication_latency']['intra_tier']),
                'p95': np.percentile(metrics['communication_latency']['intra_tier'], 95)
            },
            'inter_tier_latency': {
                'mean': np.mean(metrics['communication_latency']['inter_tier']),
                'std': np.std(metrics['communication_latency']['inter_tier']),
                'p95': np.percentile(metrics['communication_latency']['inter_tier'], 95)
            },
            'edge_to_cloud_latency': {
                'mean': np.mean(metrics['communication_latency']['edge_to_cloud']),
                'std': np.std(metrics['communication_latency']['edge_to_cloud']),
                'p95': np.percentile(metrics['communication_latency']['edge_to_cloud'], 95)
            }
        }
    
    def _get_component_dependencies(self, component_id: str) -> List[str]:
        """Get dependencies for a component"""
        dependencies = []
        
        if component_id.startswith('head_'):
            # Attention head dependencies
            dependencies.extend([
                f'cache_{component_id}',  # K/V cache dependency
                'projection'              # Output goes to projection layer
            ])
        elif component_id == 'projection':
            # Projection layer depends on all attention heads
            dependencies.extend([
                f'head_{i}' 
                for i in range(self.distributor.transformer.config.num_heads)
            ])
        elif component_id == 'ffn':
            # FFN depends on projection output
            dependencies.append('projection')
            
        return dependencies
        
    def _estimate_transfer_size(
        self,
        component_id: str,
        dependency_id: str
    ) -> float:
        """Estimate size of data transfer between components in GB"""
        transformer = self.distributor.transformer
        
        if dependency_id.startswith('cache_'):
            # K/V cache transfer size
            return (transformer.current_sequence_length * 
                   transformer.config.head_dim * 
                   transformer.config.precision_bytes * 2) / (1024**3)
            
        elif component_id == 'projection' and dependency_id.startswith('head_'):
            # Attention head output size
            return (transformer.current_sequence_length * 
                   transformer.config.head_dim * 
                   transformer.config.precision_bytes) / (1024**3)
            
        elif component_id == 'ffn' and dependency_id == 'projection':
            # Projection output size
            return (transformer.current_sequence_length * 
                   transformer.config.embedding_dim * 
                   transformer.config.precision_bytes) / (1024**3)
            
        return 0.0
        
    def _run_workload_with_tier_tracking(
        self,
        workload,
        workload_idx: int
    ) -> Dict:
        """Run a workload while tracking tier-specific metrics"""
        metrics = {
            'resource_metrics': {},
            'communication_metrics': {},
            'performance_metrics': {},
            'component_assignments': {},
            'tier_utilization': defaultdict(list)
        }
        
        for step in range(workload.sequence_config.num_steps):
            # Get assignments
            assignment_result = self.distributor.compute_assignment(
                step,
                previous_assignments=metrics.get('previous_assignments'),
                previous_cache=metrics.get('previous_cache')
            )
            
            if not assignment_result.is_feasible:
                raise RuntimeError(
                    f"Infeasible assignment at step {step} for workload {workload_idx}"
                )
                
            # Record metrics
            metrics['resource_metrics'][step] = assignment_result.resource_usage
            metrics['component_assignments'][step] = assignment_result.component_assignments
            metrics['previous_assignments'] = assignment_result.component_assignments
            metrics['previous_cache'] = assignment_result.cache_assignments
            
            # Record tier utilization
            for tier, devices in self.device_tiers.items():
                tier_util = np.mean([
                    assignment_result.resource_usage[dev_id]['compute_utilization']
                    for dev_id in devices
                ])
                metrics['tier_utilization'][tier].append(tier_util)
                
            # Record performance metrics
            metrics['performance_metrics'][step] = {
                'latency': assignment_result.estimated_latency,
                'step': step
            }
            
        return metrics