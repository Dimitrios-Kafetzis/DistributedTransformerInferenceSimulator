from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

from .common import BaseScenario, ScenarioResult, validate_scenario_requirements, collect_scenario_metrics
from src.core import Network, Device, Transformer
from src.algorithms import ResourceAwareDistributor
from src.environment import (
    NetworkTopologyGenerator,
    ResourceDistributor,
    WorkloadGenerator,
    DistributedEdgeTopology,
    WorkloadType
)

class DistributedEdgeBasicScenario(BaseScenario):
    """
    Basic distributed edge scenario testing operation with 16 devices
    """
    
    def setup(self) -> None:
        """Set up the distributed edge environment"""
        if self.logger:
            self.logger.log_event("setup", "Setting up distributed edge basic scenario")
            
        # Create network topology
        topology_generator = DistributedEdgeTopology(self.config.network)
        self.network = topology_generator.generate()
        
        # Set up resource distribution
        resource_distributor = ResourceDistributor(
            num_devices=16,
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
        
        # Set up workload - using both small and medium models
        self.workload_generator = WorkloadGenerator()
        self.workloads = [
            self.workload_generator.generate_workload(
                WorkloadType.SMALL,
                self.config.workload.sequence_config
            ),
            self.workload_generator.generate_workload(
                WorkloadType.MEDIUM,
                self.config.workload.sequence_config
            )
        ]
        
        # Initialize algorithm
        self.distributor = ResourceAwareDistributor(
            self.workloads[0].transformer,  # Start with small model
            self.network,
            self.devices
        )
        
    def run(self) -> ScenarioResult:
        """Run the basic distributed edge scenario"""
        metrics = {
            'resource_metrics': {},
            'communication_metrics': {},
            'performance_metrics': {},
            'model_comparison': {}
        }
        
        try:
            # Run each workload type
            for workload_idx, workload in enumerate(self.workloads):
                self.distributor.transformer = workload.transformer
                workload_metrics = self._run_single_workload(workload, workload_idx)
                
                # Record model-specific metrics
                metrics['model_comparison'][f'model_{workload_idx}'] = {
                    'type': workload.workload_type.name,
                    'average_latency': np.mean([
                        m['latency'] for m in 
                        workload_metrics['performance_metrics'].values()
                    ]),
                    'resource_utilization': np.mean([
                        list(m.values())[0] for m in 
                        workload_metrics['resource_metrics'].values()
                    ])
                }
                
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

class DistributedEdgeCommunicationScenario(BaseScenario):
    """
    Tests communication patterns and bandwidth utilization in distributed setup
    """
    
    def setup(self) -> None:
        """Set up communication test environment"""
        # Basic setup
        super().setup()
        
        # Track communication patterns
        self.communication_tracker = {
            'transfers': [],
            'bandwidth_usage': {},
            'network_congestion': {}
        }
        
        # Initialize network monitoring
        for (src, dst), link in self.network.links.items():
            self.communication_tracker['bandwidth_usage'][(src, dst)] = []
            self.communication_tracker['network_congestion'][(src, dst)] = []
            
    def run(self) -> ScenarioResult:
        """Run communication pattern analysis"""
        metrics = {
            'resource_metrics': {},
            'communication_metrics': {},
            'performance_metrics': {},
            'network_analysis': {}
        }
        
        try:
            for step in range(self.workload.sequence_config.num_steps):
                # Track network state before assignment
                self._record_network_state()
                
                # Get assignments
                assignment_result = self.distributor.compute_assignment(
                    step,
                    previous_assignments=metrics.get('previous_assignments'),
                    previous_cache=metrics.get('previous_cache')
                )
                
                # Record transfers and communication patterns
                self._track_communication(
                    step,
                    assignment_result.component_assignments,
                    metrics.get('previous_assignments', {})
                )
                
                # Update metrics
                metrics['resource_metrics'][step] = assignment_result.resource_usage
                metrics['previous_assignments'] = assignment_result.component_assignments
                metrics['previous_cache'] = assignment_result.cache_assignments
                metrics['performance_metrics'][step] = {
                    'latency': assignment_result.estimated_latency,
                    'step': step
                }
                
            # Analyze communication patterns
            metrics['network_analysis'] = self._analyze_communication_patterns()
            
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
            
    def _record_network_state(self) -> None:
        """Record current network state"""
        for (src, dst), link in self.network.links.items():
            self.communication_tracker['bandwidth_usage'][(src, dst)].append(
                link.used_bandwidth / link.bandwidth
            )
            self.communication_tracker['network_congestion'][(src, dst)].append(
                max(0, 1 - link.available_bandwidth / link.bandwidth)
            )
            
    def _track_communication(
        self,
        step: int,
        current_assignments: Dict[str, str],
        previous_assignments: Dict[str, str]
    ) -> None:
        """Track communication events and patterns"""
        for comp_id, device_id in current_assignments.items():
            if comp_id in previous_assignments:
                if device_id != previous_assignments[comp_id]:
                    # Record component migration
                    self.communication_tracker['transfers'].append({
                        'step': step,
                        'type': 'migration',
                        'component': comp_id,
                        'source': previous_assignments[comp_id],
                        'target': device_id
                    })
                    
    def _analyze_communication_patterns(self) -> Dict:
        """Analyze recorded communication patterns"""
        return {
            'total_transfers': len(self.communication_tracker['transfers']),
            'bandwidth_utilization': {
                link: np.mean(usage)
                for link, usage in 
                self.communication_tracker['bandwidth_usage'].items()
            },
            'congestion_hotspots': {
                link: np.mean(congestion)
                for link, congestion in 
                self.communication_tracker['network_congestion'].items()
            }
        }

class DistributedEdgeHeterogeneityScenario(BaseScenario):
    """
    Tests system behavior with heterogeneous device capabilities
    """
    
    def setup(self) -> None:
        """Set up heterogeneous environment"""
        super().setup()
        
        # Create device groups with varying capabilities
        self.device_groups = {
            'high_capacity': [],
            'medium_capacity': [],
            'low_capacity': []
        }
        
        # Categorize devices based on capabilities
        for device_id, device in self.devices.items():
            if device.compute.capacity >= 200:  # GFLOPS
                self.device_groups['high_capacity'].append(device_id)
            elif device.compute.capacity >= 100:
                self.device_groups['medium_capacity'].append(device_id)
            else:
                self.device_groups['low_capacity'].append(device_id)
                
    def run(self) -> ScenarioResult:
        """Run heterogeneity analysis"""
        metrics = {
            'resource_metrics': {},
            'communication_metrics': {},
            'performance_metrics': {},
            'heterogeneity_analysis': {}
        }
        
        try:
            # Track metrics per device group
            group_metrics = {
                group: {
                    'utilization': [],
                    'latency': [],
                    'assignments': []
                }
                for group in self.device_groups.keys()
            }
            
            # Run workload
            for step in range(self.workload.sequence_config.num_steps):
                assignment_result = self.distributor.compute_assignment(
                    step,
                    previous_assignments=metrics.get('previous_assignments'),
                    previous_cache=metrics.get('previous_cache')
                )
                
                # Record group-specific metrics
                self._record_group_metrics(
                    step,
                    assignment_result,
                    group_metrics
                )
                
                # Update general metrics
                metrics['resource_metrics'][step] = assignment_result.resource_usage
                metrics['previous_assignments'] = assignment_result.component_assignments
                metrics['previous_cache'] = assignment_result.cache_assignments
                metrics['performance_metrics'][step] = {
                    'latency': assignment_result.estimated_latency,
                    'step': step
                }
                
            # Analyze heterogeneity impact
            metrics['heterogeneity_analysis'] = self._analyze_heterogeneity(
                group_metrics
            )
            
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
            
    def _record_group_metrics(
        self,
        step: int,
        assignment_result: Dict,
        group_metrics: Dict
    ) -> None:
        """Record metrics for each device group"""
        for group, devices in self.device_groups.items():
            # Calculate group utilization
            group_util = np.mean([
                assignment_result.resource_usage[dev_id]['compute_utilization']
                for dev_id in devices
            ])
            group_metrics[group]['utilization'].append(group_util)
            
            # Count assignments to group
            assignments = sum(
                1 for dev_id in assignment_result.component_assignments.values()
                if dev_id in devices
            )
            group_metrics[group]['assignments'].append(assignments)
            
    def _analyze_heterogeneity(self, group_metrics: Dict) -> Dict:
        """Analyze impact of device heterogeneity"""
        return {
            group: {
                'average_utilization': np.mean(metrics['utilization']),
                'utilization_std': np.std(metrics['utilization']),
                'assignment_frequency': np.mean(metrics['assignments']),
                'assignment_stability': np.std(metrics['assignments'])
            }
            for group, metrics in group_metrics.items()
        }