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
    EdgeClusterTopology,
    WorkloadType
)

class EdgeClusterBasicScenario(BaseScenario):
    """
    Basic edge cluster scenario testing standard operation with 8 devices
    """
    
    def setup(self) -> None:
        """Set up the edge cluster environment"""
        if self.logger:
            self.logger.log_event("setup", "Setting up edge cluster basic scenario")
            
        # Create network topology
        topology_generator = EdgeClusterTopology(self.config.network)
        self.network = topology_generator.generate()
        
        # Set up resource distribution
        resource_distributor = ResourceDistributor(
            num_devices=8,
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
        
        # Set up workload
        self.workload_generator = WorkloadGenerator()
        self.workload = self.workload_generator.generate_workload(
            WorkloadType.SMALL,  # 8 heads, D=512
            self.config.workload.sequence_config
        )
        
        # Initialize algorithm
        self.distributor = ResourceAwareDistributor(
            self.workload.transformer,
            self.network,
            self.devices
        )
        
        # Validate setup
        if not validate_scenario_requirements(
            self.config,
            self.network,
            self.devices,
            self.workload.transformer
        ):
            raise ValueError("Scenario requirements not met")
            
    def run(self) -> ScenarioResult:
        """Run the basic scenario"""
        if self.logger:
            self.logger.log_event("run", "Starting edge cluster basic scenario")
            
        metrics = {
            'resource_metrics': {},
            'communication_metrics': {},
            'performance_metrics': {}
        }
        
        try:
            # Run through generation steps
            for step in range(self.workload.sequence_config.num_steps):
                # Get assignments
                assignment_result = self.distributor.compute_assignment(
                    step,
                    previous_assignments=metrics.get('previous_assignments'),
                    previous_cache=metrics.get('previous_cache')
                )
                
                if not assignment_result.is_feasible:
                    raise RuntimeError(f"Infeasible assignment at step {step}")
                    
                # Record metrics
                metrics['resource_metrics'][step] = assignment_result.resource_usage
                metrics['previous_assignments'] = assignment_result.component_assignments
                metrics['previous_cache'] = assignment_result.cache_assignments
                
                # Record performance
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
            
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.logger:
            self.logger.log_event("cleanup", "Cleaning up edge cluster basic scenario")
        # Cleanup resources if needed
        pass

class EdgeClusterScalabilityScenario(BaseScenario):
    """
    Tests scalability of edge cluster by gradually increasing workload
    """
    
    def setup(self) -> None:
        """Set up scalability test environment"""
        # Similar to basic setup but with different workload configurations
        self.setup_basic_environment()
        
        # Create multiple workloads with increasing sizes
        self.workloads = []
        sequence_lengths = [128, 256, 512]
        for length in sequence_lengths:
            workload = self.workload_generator.generate_workload(
                WorkloadType.SMALL,
                sequence_config=self.config.workload.sequence_config._replace(
                    initial_length=length
                )
            )
            self.workloads.append(workload)
            
    def run(self) -> ScenarioResult:
        """Run scalability tests"""
        metrics = {
            'resource_metrics': {},
            'communication_metrics': {},
            'performance_metrics': {},
            'scalability_metrics': {}
        }
        
        try:
            # Test each workload size
            for idx, workload in enumerate(self.workloads):
                self.distributor.transformer = workload.transformer
                
                # Run workload
                workload_metrics = self._run_single_workload(workload, idx)
                
                # Record scalability metrics
                metrics['scalability_metrics'][idx] = {
                    'sequence_length': workload.sequence_config.initial_length,
                    'average_latency': np.mean(
                        [m['latency'] for m in workload_metrics['performance_metrics'].values()]
                    ),
                    'peak_memory': max(
                        sum(usage.values()) 
                        for usage in workload_metrics['resource_metrics'].values()
                    )
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
            
    def cleanup(self) -> None:
        """Clean up resources"""
        pass
        
    def _run_single_workload(self, workload, workload_idx: int) -> Dict:
        """Run a single workload configuration"""
        metrics = {
            'resource_metrics': {},
            'communication_metrics': {},
            'performance_metrics': {}
        }
        
        for step in range(workload.sequence_config.num_steps):
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
            metrics['previous_assignments'] = assignment_result.component_assignments
            metrics['previous_cache'] = assignment_result.cache_assignments
            metrics['performance_metrics'][step] = {
                'latency': assignment_result.estimated_latency,
                'step': step
            }
            
        return metrics

class EdgeClusterFailureScenario(BaseScenario):
    """
    Tests edge cluster behavior under device failures
    """
    
    def setup(self) -> None:
        """Set up failure test environment"""
        self.setup_basic_environment()
        
        # Define failure scenarios
        self.failure_schedule = [
            {'step': 10, 'device': 'device_4'},  # Edge device failure
            {'step': 20, 'device': 'device_2'},  # Mid-tier device failure
            {'step': 30, 'device': 'device_1'}   # Another mid-tier failure
        ]
        
    def run(self) -> ScenarioResult:
        """Run failure scenario tests"""
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
                
                # Get assignments
                assignment_result = self.distributor.compute_assignment(
                    step,
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
            
    def cleanup(self) -> None:
        """Clean up resources"""
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
                self.network.topology.remove_node(device_id)