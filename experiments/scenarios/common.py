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
# File:    experiments/scenarios/common.py
# Description:
#   Contains base classes and utility functions for experiment scenarios,
#   including scenario execution pipelines, shared data structures, and
#   scenario validation methods for distributed inference.
#
# ---------------------------------------------------------------------------

"""
Defines core abstract classes and helper functions for constructing and
executing various experiment scenarios. Provides common data structures
and logic that baseline or advanced scenario classes can inherit or use.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path
import yaml
import time
import json
from datetime import datetime

from src.core import Transformer, Network, Device
from src.algorithms import ResourceAwareDistributor
from src.environment import (
    NetworkTopologyGenerator,
    ResourceDistributor,
    WorkloadGenerator
)
from src.utils import (
    SimulationConfig,
    SimulationLogger,
    VisualizationManager
)

@dataclass
class ScenarioResult:
    """Results from running a scenario"""
    scenario_name: str
    start_time: datetime
    end_time: datetime
    metrics: Dict
    success: bool
    error: Optional[str] = None

class BaseScenario(ABC):
    """Base class for all experiment scenarios"""
    
    def __init__(
        self,
        config: SimulationConfig,
        output_dir: Union[str, Path],
        logger: Optional[SimulationLogger] = None
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.output_dir / 'metrics'
        self.logs_dir = self.output_dir / 'logs'
        self.viz_dir = self.output_dir / 'visualizations'
        
        for directory in [self.metrics_dir, self.logs_dir, self.viz_dir]:
            directory.mkdir(exist_ok=True)
            
        # Initialize visualization manager
        self.viz = VisualizationManager(self.viz_dir)
        
    @abstractmethod
    def setup(self) -> None:
        """Set up the scenario environment"""
        pass
        
    @abstractmethod
    def run(self) -> ScenarioResult:
        """Run the scenario"""
        pass
        
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up after scenario execution"""
        pass
        
    def execute(self) -> ScenarioResult:
        """Execute the complete scenario"""
        start_time = datetime.now()
        
        try:
            # Setup environment
            self.setup()
            
            # Run scenario
            result = self.run()
            
            # Add timing information
            result.start_time = start_time
            result.end_time = datetime.now()
            
            # Generate visualizations
            self._generate_visualizations(result)
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.log_error("scenario_error", str(e))
                
            return ScenarioResult(
                scenario_name=self.__class__.__name__,
                start_time=start_time,
                end_time=datetime.now(),
                metrics={},
                success=False,
                error=str(e)
            )
            
        finally:
            self.cleanup()
            
    def _generate_visualizations(self, result: ScenarioResult) -> None:
        """Generate visualizations for scenario results"""
        if not result.success:
            return
            
        try:
            # Create standard visualizations
            self.viz.plot_resource_utilization(
                result.metrics['resource_metrics'],
                result.metrics['device_ids']
            )
            
            self.viz.plot_latency_comparison(
                result.metrics['algorithms'],
                result.metrics['latencies'],
                result.metrics['workload_types']
            )
            
            self.viz.plot_communication_patterns(
                result.metrics['communication_data']
            )
            
            # Save performance report
            self.viz.create_performance_report(
                result.metrics,
                self.output_dir / 'performance_report.json'
            )
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    "visualization_error",
                    f"Error generating visualizations: {e}"
                )

def load_scenario_config(config_path: Union[str, Path]) -> SimulationConfig:
    """Load scenario configuration from file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        if config_path.suffix == '.yaml':
            config_dict = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError("Configuration file must be .yaml or .json")
            
    return SimulationConfig.from_dict(config_dict)

def run_scenario(
    scenario_class: type,
    config_path: Union[str, Path],
    output_dir: Union[str, Path],
    logger: Optional[SimulationLogger] = None
) -> ScenarioResult:
    """Run a specific scenario with given configuration"""
    # Load configuration
    config = load_scenario_config(config_path)
    
    # Create and execute scenario
    scenario = scenario_class(config, output_dir, logger)
    return scenario.execute()

def validate_scenario_requirements(
    config: SimulationConfig,
    network: Network,
    devices: Dict[str, Device],
    transformer: Transformer
) -> bool:
    """Validate if scenario requirements are met"""
    try:
        # Check network connectivity
        if not network.topology.is_connected():
            return False
            
        # Check device resources
        total_memory = sum(dev.memory.capacity for dev in devices.values())
        min_memory_required = transformer.get_total_memory_requirement()
        if total_memory < min_memory_required:
            return False
            
        # Check bandwidth requirements
        min_bandwidth = min(
            link.bandwidth for link in network.links.values()
        )
        if min_bandwidth < config.network.min_bandwidth:
            return False
            
        return True
        
    except Exception:
        return False

def collect_scenario_metrics(
    resource_metrics: Dict,
    communication_metrics: Dict,
    performance_metrics: Dict
) -> Dict:
    """Collect and organize scenario metrics"""
    return {
        'timestamp': datetime.now().isoformat(),
        'resource_metrics': resource_metrics,
        'communication_metrics': communication_metrics,
        'performance_metrics': performance_metrics,
        'summary': {
            'total_runtime': performance_metrics.get('total_time', 0),
            'average_latency': performance_metrics.get('average_latency', 0),
            'peak_memory': resource_metrics.get('peak_memory', 0),
            'total_data_transferred': 
                communication_metrics.get('total_data_transferred', 0)
        }
    }