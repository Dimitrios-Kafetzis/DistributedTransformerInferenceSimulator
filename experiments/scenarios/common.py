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
import pandas as pd
import matplotlib.pyplot as plt

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

    def to_dict(self) -> Dict:
        return {
            "scenario_name": self.scenario_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "metrics": self.metrics,
            "success": self.success,
            "error": self.error
        }

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
            
        # Visualization manager
        self.viz = VisualizationManager(self.viz_dir)
        
    @abstractmethod
    def setup(self) -> None:
        pass
        
    @abstractmethod
    def run(self) -> ScenarioResult:
        pass
        
    @abstractmethod
    def cleanup(self) -> None:
        pass
        
    def execute(self) -> ScenarioResult:
        start_time = datetime.now()
        try:
            # Setup
            self.setup()
            
            # Run
            result = self.run()
            
            # Add timing info
            result.start_time = start_time
            result.end_time = datetime.now()

            # Insert device_ids if not present
            if 'device_ids' not in result.metrics:
                if hasattr(self, 'devices'):
                    result.metrics['device_ids'] = sorted(list(self.devices.keys()))
                else:
                    result.metrics['device_ids'] = []

            # Convert any tuple keys to strings
            result.metrics = self._convert_tuple_keys(result.metrics)

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
        # If scenario unsuccessful, skip
        if not result.success:
            return
    
        import ast
        import pprint
        pprinter = pprint.PrettyPrinter(indent=2, width=120)
        
        # Debug: log entire metrics
        self.logger.log_event(
            "debug_scenario_result_metrics",
            "ScenarioResult.metrics = \n" + pprinter.pformat(result.metrics)
        )
        
        # Example: try resource_metrics -> DataFrame
        if "resource_metrics" in result.metrics:
            resource_dict = result.metrics["resource_metrics"]
            # parse into DF
            rows = []
            for step, device_map in resource_dict.items():
                if isinstance(step, str) and step.startswith("(") and step.endswith(")"):
                    try:
                        parsed = ast.literal_eval(step)
                        step_val = parsed[-1]  # last item
                    except:
                        step_val = step
                else:
                    step_val = step
                try:
                    step_val = int(step_val)
                except:
                    pass

                for dev_id, usage in device_map.items():
                    mem_util = usage.get("memory_utilization", 0.0)
                    comp_util = usage.get("compute_utilization", 0.0)
                    rows.append({
                        "step": step_val,
                        "device_id": dev_id,
                        "memory_utilization": mem_util,
                        "compute_utilization": comp_util
                    })
            df = pd.DataFrame(rows)
            if not df.empty:
                try:
                    fig = self.viz.plot_resource_utilization(df, df["device_id"].unique())
                    self.viz.save_plot(fig, "resource_utilization")
                    plt.close(fig)
                except KeyError as ke:
                    self.logger.log_error(
                        "visualization_error",
                        f"plot_resource_utilization KeyError: {ke}"
                    )

    def _convert_tuple_keys(self, data):
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                if isinstance(k, tuple):
                    new_key = str(k)
                else:
                    new_key = k
                new_dict[new_key] = self._convert_tuple_keys(v)
            return new_dict
        elif isinstance(data, list):
            return [self._convert_tuple_keys(item) for item in data]
        else:
            return data


def load_scenario_config(config_path: Union[str, Path]) -> SimulationConfig:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    if config_path.suffix == '.yaml':
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
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
    config = load_scenario_config(config_path)
    scenario = scenario_class(config, output_dir, logger)
    return scenario.execute()


def validate_scenario_requirements(
    config: SimulationConfig,
    network: Network,
    devices: Dict[str, Device],
    transformer: Transformer
) -> bool:
    try:
        if not network.topology.is_connected():
            return False
        total_memory = sum(dev.memory.capacity for dev in devices.values())
        min_memory_required = transformer.get_total_memory_requirement()
        if total_memory < min_memory_required:
            return False
        min_bandwidth = min(link.bandwidth for link in network.links.values())
        if min_bandwidth < config.network.min_bandwidth:
            return False
        return True
    except:
        return False


def collect_scenario_metrics(
    resource_metrics: Dict,
    communication_metrics: Dict,
    performance_metrics: Dict
) -> Dict:
    return {
        'timestamp': datetime.now().isoformat(),
        'resource_metrics': resource_metrics,
        'communication_metrics': communication_metrics,
        'performance_metrics': performance_metrics,
        'summary': {
            'total_runtime': performance_metrics.get('total_time', 0),
            'average_latency': performance_metrics.get('average_latency', 0),
            'peak_memory': resource_metrics.get('peak_memory', 0),
            'total_data_transferred': communication_metrics.get('total_data_transferred', 0)
        }
    }
