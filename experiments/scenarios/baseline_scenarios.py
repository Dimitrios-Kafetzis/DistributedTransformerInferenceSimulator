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
transformer inference across multiple edge devices.
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import json
from src.utils import SimulationLogger
from .common import BaseScenario, ScenarioResult, validate_scenario_requirements, collect_scenario_metrics
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
    WorkloadType
)



class BaselineComparisonScenario(BaseScenario):
    """Base class for baseline comparison scenarios"""
    
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
        assignment_result: Dict,
        metrics: Dict
    ) -> None:
        """Record metrics for algorithm comparison"""
        metrics['latency'][algorithm_name].append(
            assignment_result.estimated_latency
        )
        
        # Resource utilization
        avg_util = np.mean([
            usage['compute_utilization']
            for usage in assignment_result.resource_usage.values()
        ])
        metrics['resource_utilization'][algorithm_name].append(avg_util)
        
        # Track migrations if available
        if hasattr(assignment_result, 'migrations'):
            metrics['migration_counts'][algorithm_name] += \
                len(assignment_result.migrations)

class GreedyBaselineScenario(BaselineComparisonScenario):
    """
    Tests greedy placement strategy that assigns components to
    first available device with sufficient resources
    """
    
    def setup(self) -> None:
        """Set up greedy baseline environment"""
        super().setup()
        
        # Initialize both greedy and resource-aware distributors
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
            for step in range(self.workload.sequence_config.num_steps):
                # Run both algorithms
                greedy_result = self.greedy_distributor.compute_assignment(
                    step,
                    previous_assignments=metrics.get('greedy_previous'),
                    previous_cache=metrics.get('greedy_cache')
                )
                
                resource_aware_result = self.resource_aware_distributor.compute_assignment(
                    step,
                    previous_assignments=metrics.get('resource_aware_previous'),
                    previous_cache=metrics.get('resource_aware_cache')
                )
                
                # Record metrics
                self._record_comparison_metrics(
                    'greedy',
                    step,
                    greedy_result,
                    metrics['comparison_metrics']
                )
                
                self._record_comparison_metrics(
                    'resource_aware',
                    step,
                    resource_aware_result,
                    metrics['comparison_metrics']
                )
                
                # Update previous assignments
                metrics['greedy_previous'] = greedy_result.component_assignments
                metrics['greedy_cache'] = greedy_result.cache_assignments
                metrics['resource_aware_previous'] = resource_aware_result.component_assignments
                metrics['resource_aware_cache'] = resource_aware_result.cache_assignments
                
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

class RoundRobinBaselineScenario(BaselineComparisonScenario):
    """
    Tests round-robin distribution strategy that distributes
    components evenly across available devices
    """
    
    def setup(self) -> None:
        """Set up round-robin baseline environment"""
        super().setup()
        
        # Initialize round-robin and resource-aware distributors
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
        """Run round-robin baseline comparison"""
        metrics = self._initialize_metrics()
        
        try:
            for step in range(self.workload.sequence_config.num_steps):
                # Run both algorithms
                round_robin_result = self.round_robin_distributor.compute_assignment(
                    step,
                    previous_assignments=metrics.get('round_robin_previous'),
                    previous_cache=metrics.get('round_robin_cache')
                )
                
                resource_aware_result = self.resource_aware_distributor.compute_assignment(
                    step,
                    previous_assignments=metrics.get('resource_aware_previous'),
                    previous_cache=metrics.get('resource_aware_cache')
                )
                
                # Record metrics
                self._record_comparison_metrics(
                    'round_robin',
                    step,
                    round_robin_result,
                    metrics['comparison_metrics']
                )
                
                self._record_comparison_metrics(
                    'resource_aware',
                    step,
                    resource_aware_result,
                    metrics['comparison_metrics']
                )
                
                # Update previous assignments
                metrics['round_robin_previous'] = round_robin_result.component_assignments
                metrics['round_robin_cache'] = round_robin_result.cache_assignments
                metrics['resource_aware_previous'] = resource_aware_result.component_assignments
                metrics['resource_aware_cache'] = resource_aware_result.cache_assignments
                
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

class StaticBaselineScenario(BaselineComparisonScenario):
    """
    Tests static partitioning strategy that maintains fixed
    assignments based on initial conditions
    """
    
    def setup(self) -> None:
        """Set up static baseline environment"""
        super().setup()
        
        # Initialize static and resource-aware distributors
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
        """Run static baseline comparison"""
        metrics = self._initialize_metrics()
        
        try:
            # Create initial static assignment
            initial_assignment = self.static_distributor.compute_assignment(0)
            if not initial_assignment.is_feasible:
                raise RuntimeError("Initial static assignment infeasible")
                
            for step in range(self.workload.sequence_config.num_steps):
                # Static distributor reuses initial assignment
                static_result = self.static_distributor.compute_assignment(
                    step,
                    previous_assignments=initial_assignment.component_assignments,
                    previous_cache=initial_assignment.cache_assignments
                )
                
                resource_aware_result = self.resource_aware_distributor.compute_assignment(
                    step,
                    previous_assignments=metrics.get('resource_aware_previous'),
                    previous_cache=metrics.get('resource_aware_cache')
                )
                
                # Record metrics
                self._record_comparison_metrics(
                    'static',
                    step,
                    static_result,
                    metrics['comparison_metrics']
                )
                
                self._record_comparison_metrics(
                    'resource_aware',
                    step,
                    resource_aware_result,
                    metrics['comparison_metrics']
                )
                
                # Update resource-aware previous assignments
                metrics['resource_aware_previous'] = resource_aware_result.component_assignments
                metrics['resource_aware_cache'] = resource_aware_result.cache_assignments
                
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

class DynamicMigrationBaselineScenario(BaselineComparisonScenario):
    """
    Tests dynamic migration strategy that reassigns components
    when resource utilization exceeds thresholds
    """
    
    def setup(self) -> None:
        """Set up dynamic migration baseline environment"""
        super().setup()
        
        # Initialize dynamic migration and resource-aware distributors
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
        
        # Track migrations
        self.migration_history = {
            'dynamic': [],
            'resource_aware': []
        }
        
    def run(self) -> ScenarioResult:
        """Run dynamic migration baseline comparison"""
        metrics = self._initialize_metrics()
        
        try:
            for step in range(self.workload.sequence_config.num_steps):
                # Run both algorithms
                dynamic_result = self.dynamic_distributor.compute_assignment(
                    step,
                    previous_assignments=metrics.get('dynamic_previous'),
                    previous_cache=metrics.get('dynamic_cache')
                )
                
                resource_aware_result = self.resource_aware_distributor.compute_assignment(
                    step,
                    previous_assignments=metrics.get('resource_aware_previous'),
                    previous_cache=metrics.get('resource_aware_cache')
                )
                
                # Record metrics
                self._record_comparison_metrics(
                    'dynamic',
                    step,
                    dynamic_result,
                    metrics['comparison_metrics']
                )
                
                self._record_comparison_metrics(
                    'resource_aware',
                    step,
                    resource_aware_result,
                    metrics['comparison_metrics']
                )
                
                # Track migrations
                self._track_migrations(
                    step,
                    dynamic_result,
                    resource_aware_result,
                    metrics.get('dynamic_previous'),
                    metrics.get('resource_aware_previous')
                )
                
                # Update previous assignments
                metrics['dynamic_previous'] = dynamic_result.component_assignments
                metrics['dynamic_cache'] = dynamic_result.cache_assignments
                metrics['resource_aware_previous'] = resource_aware_result.component_assignments
                metrics['resource_aware_cache'] = resource_aware_result.cache_assignments
                
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
            
    def _track_migrations(
        self,
        step: int,
        dynamic_result: Dict,
        resource_aware_result: Dict,
        prev_dynamic: Optional[Dict],
        prev_resource_aware: Optional[Dict]
    ) -> None:
        """Track component migrations for both algorithms"""
        if prev_dynamic and prev_resource_aware:
            # Track dynamic migrations
            dynamic_migrations = self._get_migrations(
                prev_dynamic,
                dynamic_result.component_assignments
            )
            self.migration_history['dynamic'].extend([
                {
                    'step': step,
                    'component': comp_id,
                    'from': src,
                    'to': dst
                }
                for comp_id, (src, dst) in dynamic_migrations.items()
            ])
            
            # Track resource-aware migrations
            resource_aware_migrations = self._get_migrations(
                prev_resource_aware,
                resource_aware_result.component_assignments
            )
            self.migration_history['resource_aware'].extend([
                {
                    'step': step,
                    'component': comp_id,
                    'from': src,
                    'to': dst
                }
                for comp_id, (src, dst) in resource_aware_migrations.items()
            ])
            
    def _get_migrations(
        self,
        prev_assignments: Dict[str, str],
        new_assignments: Dict[str, str]
    ) -> Dict[str, Tuple[str, str]]:
        """Identify component migrations between assignments"""
        migrations = {}
        for comp_id, new_dev in new_assignments.items():
            if comp_id in prev_assignments:
                prev_dev = prev_assignments[comp_id]
                if prev_dev != new_dev:
                    migrations[comp_id] = (prev_dev, new_dev)
        return migrations
        
    def cleanup(self) -> None:
        """Clean up baseline comparison resources"""
        if self.logger:
            self.logger.log_event(
                "cleanup",
                f"Cleaning up {self.__class__.__name__}"
            )
        
        # Save migration history if exists
        if hasattr(self, 'migration_history'):
            migration_stats = {
                algorithm: {
                    'total_migrations': len(migrations),
                    'migration_frequency': len(migrations) / 
                        self.workload.sequence_config.num_steps,
                    'migration_pattern': self._analyze_migration_pattern(migrations)
                }
                for algorithm, migrations in self.migration_history.items()
            }
            
            if self.logger:
                self.logger.log_metrics({
                    'migration_statistics': migration_stats
                })
                
    def _analyze_migration_pattern(self, migrations: List[Dict]) -> Dict:
        """Analyze patterns in component migrations"""
        if not migrations:
            return {}
            
        pattern_analysis = {
            'component_frequency': defaultdict(int),
            'device_pairs': defaultdict(int),
            'step_distribution': defaultdict(int)
        }
        
        for migration in migrations:
            pattern_analysis['component_frequency'][migration['component']] += 1
            pattern_analysis['device_pairs'][
                (migration['from'], migration['to'])
            ] += 1
            pattern_analysis['step_distribution'][migration['step']] += 1
            
        return {
            'most_migrated_components': dict(sorted(
                pattern_analysis['component_frequency'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            'common_migration_paths': dict(sorted(
                pattern_analysis['device_pairs'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            'temporal_distribution': dict(sorted(
                pattern_analysis['step_distribution'].items()
            ))
        }

def run_all_baselines(
    config_path: Union[str, Path],
    output_dir: Union[str, Path],
    logger: Optional[SimulationLogger] = None
) -> Dict[str, ScenarioResult]:
    """Run all baseline comparisons"""
    baseline_scenarios = [
        GreedyBaselineScenario,
        RoundRobinBaselineScenario,
        StaticBaselineScenario,
        DynamicMigrationBaselineScenario
    ]
    
    results = {}
    for scenario_class in baseline_scenarios:
        scenario_name = scenario_class.__name__
        if logger:
            logger.log_event(
                "baseline_comparison",
                f"Running {scenario_name}"
            )
            
        try:
            scenario = scenario_class(
                config_path=config_path,
                output_dir=Path(output_dir) / scenario_name.lower(),
                logger=logger
            )
            results[scenario_name] = scenario.execute()
            
        except Exception as e:
            if logger:
                logger.log_error(
                    "baseline_error",
                    f"Error running {scenario_name}: {str(e)}"
                )
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
    results: Dict[str, ScenarioResult],
    output_dir: Union[str, Path],
    logger: Optional[SimulationLogger] = None
) -> Dict:
    """Analyze and compare baseline results"""
    analysis = {
        'performance_comparison': {},
        'resource_efficiency': {},
        'communication_overhead': {},
        'migration_analysis': {},
        'summary': {}
    }
    
    try:
        # Performance comparison
        latencies = {
            name: result.metrics.get('performance_metrics', {}).get('average_latency', 0)
            for name, result in results.items()
            if result.success
        }
        analysis['performance_comparison']['latency'] = latencies
        
        # Resource efficiency
        utilizations = {
            name: result.metrics.get('resource_metrics', {}).get('average_utilization', 0)
            for name, result in results.items()
            if result.success
        }
        analysis['resource_efficiency']['utilization'] = utilizations
        
        # Communication overhead
        communication = {
            name: result.metrics.get('communication_metrics', {}).get('total_data_transferred', 0)
            for name, result in results.items()
            if result.success
        }
        analysis['communication_overhead']['data_transferred'] = communication
        
        # Migration analysis
        migrations = {
            name: result.metrics.get('migration_statistics', {})
            for name, result in results.items()
            if result.success
        }
        analysis['migration_analysis'] = migrations
        
        # Calculate summary statistics
        analysis['summary'] = {
            'best_latency': min(latencies.items(), key=lambda x: x[1]) if latencies else None,
            'best_utilization': max(utilizations.items(), key=lambda x: x[1]) if utilizations else None,
            'lowest_communication': min(communication.items(), key=lambda x: x[1]) if communication else None
        }
        
        # Save analysis
        output_path = Path(output_dir) / 'baseline_analysis.json'
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        if logger:
            logger.log_metrics({
                'baseline_analysis': analysis
            })
            
        return analysis
        
    except Exception as e:
        if logger:
            logger.log_error(
                "analysis_error",
                f"Error analyzing baseline results: {str(e)}"
            )
        return {
            'error': str(e)
        }