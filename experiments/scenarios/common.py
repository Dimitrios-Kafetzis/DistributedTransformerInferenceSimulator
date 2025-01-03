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
        """
        Convert this ScenarioResult into a dictionary that is safe to serialize as JSON.
        Adjust the 'metrics' field if it contains objects that aren't JSON-friendly.
        """
        return {
            "scenario_name": self.scenario_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "metrics": self.metrics,   # If nested objects are not JSON-friendly, convert them here
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

            # --- 1) Insert device_ids if not present ---
            # If your scenario has self.devices, let's add them:
            if 'device_ids' not in result.metrics:
                # Gather from self.devices if that is available:
                if hasattr(self, 'devices'):
                    result.metrics['device_ids'] = sorted(list(self.devices.keys()))
                else:
                    result.metrics['device_ids'] = []

            # --- 2) Convert any tuple keys to strings in result.metrics, if needed ---
            # We can define a helper:
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
        """
        Generate visualizations for scenario results.
     
        1) Converts 'resource_metrics' into a DataFrame, calls plot_resource_utilization().
        2) If 'communication_data' is present, attempts to flatten into a DataFrame
           with columns suitable for plot_communication_patterns().
        3) If 'algorithms'/'latencies'/'workload_types' exist, calls plot_latency_comparison().
        4) Attempts to create a performance report via create_performance_report(),
           using a fallback DataFrame if necessary to avoid KeyError.
        """
        if not result.success:
            return
        
        import matplotlib.pyplot as plt  # Ensure plt is imported
        import pandas as pd  # We'll need pandas if not already imported
     
        function_name = "_generate_visualizations"
        file_name = "experiments/scenarios/common.py"
     
        try:
            ##############################
            # 1) RESOURCE METRICS -> DF
            ##############################
            resource_df = None
            if "resource_metrics" in result.metrics:
                resource_dict = result.metrics["resource_metrics"]  # e.g. { step: { device_0: {...}, ...}, ... }
     
                # Flatten into rows for DataFrame
                rows = []
                for step, device_map in resource_dict.items():
                    for dev_id, usage in device_map.items():
                        mem_util = usage.get("memory_utilization", 0.0)
                        cmp_util = usage.get("compute_utilization", 0.0)
                        rows.append({
                            "step": step,
                            "device_id": dev_id,
                            "memory_utilization": mem_util,
                            "compute_utilization": cmp_util
                        })
     
                resource_df = pd.DataFrame(rows)
     
                # If we have a valid DF with 'device_id' in columns, we can call plot_resource_utilization
                if not resource_df.empty and "device_id" in resource_df.columns:
                    device_ids = resource_df["device_id"].unique().tolist()
     
                    # Debugging log info
                    self.logger.log_metrics({
                        "viz_debug": {
                            "function": function_name,
                            "resource_df_shape": resource_df.shape,
                            "device_ids_list": device_ids
                        }
                    })
     
                    # Attempt to plot:
                    try:
                        fig_res = self.viz.plot_resource_utilization(resource_df, device_ids)
                        self.viz.save_plot(fig_res, "resource_utilization")  # optionally save
                        plt.close(fig_res)
                    except KeyError as ke:
                        self.logger.log_error(
                            "visualization_error",
                            f"[{file_name} -> {function_name}] plot_resource_utilization KeyError: {ke}"
                        )
                else:
                    # Resource DF is empty or missing 'device_id'—skip or log
                    self.logger.log_event(
                        "viz_debug",
                        "resource_df is empty or missing 'device_id'; skipping plot_resource_utilization"
                    )
     
            ##############################
            # 2) LATENCY COMPARISON
            ##############################
            if (
                "algorithms" in result.metrics
                and "latencies" in result.metrics
                and "workload_types" in result.metrics
            ):
                try:
                    fig_lat = self.viz.plot_latency_comparison(
                        result.metrics["algorithms"],
                        result.metrics["latencies"],
                        result.metrics["workload_types"]
                    )
                    self.viz.save_plot(fig_lat, "latency_comparison")
                    plt.close(fig_lat)
                except KeyError as ke:
                    self.logger.log_error(
                        "visualization_error",
                        f"[{file_name} -> {function_name}] plot_latency_comparison KeyError: {ke}"
                    )
     
            ##############################
            # 3) COMMUNICATION PATTERNS
            ##############################
            if "communication_data" in result.metrics:
                comm_data = result.metrics["communication_data"]
     
                # Attempt to flatten or confirm it's a DataFrame
                try:
                    if isinstance(comm_data, pd.DataFrame):
                        comm_df = comm_data.copy()
                    elif isinstance(comm_data, list):
                        # e.g. a list of dicts
                        comm_df = pd.DataFrame(comm_data)
                    elif isinstance(comm_data, dict):
                        # e.g. { step -> {source_device:..., data_volume:...} }
                        rows = []
                        for k, v in comm_data.items():
                            if isinstance(v, dict):
                                row = dict(v)
                                row["comm_key"] = k
                                rows.append(row)
                            else:
                                # If it's something else, adapt or skip
                                pass
                        comm_df = pd.DataFrame(rows)
                    else:
                        comm_df = pd.DataFrame()
     
                    if not comm_df.empty:
                        needed_cols = {"source_device", "data_volume", "transfer_time"}
                        if needed_cols.issubset(set(comm_df.columns)):
                            fig_comm = self.viz.plot_communication_patterns(comm_df)
                            self.viz.save_plot(fig_comm, "communication_patterns")
                            plt.close(fig_comm)
                        else:
                            missing = needed_cols - set(comm_df.columns)
                            self.logger.log_event(
                                "viz_debug",
                                f"communication_data missing columns {missing}; skipping plot_communication_patterns"
                            )
                    else:
                        self.logger.log_event(
                            "viz_debug",
                            "communication_data is empty or unrecognized; skipping plot_communication_patterns"
                        )
     
                except Exception as e:
                    self.logger.log_error(
                        "visualization_error",
                        f"[{file_name} -> {function_name}] plot_communication_patterns error: {str(e)}"
                    )
     
            ##############################
            # 4) PERFORMANCE REPORT
            ##############################
            try:
                # If create_performance_report expects a DataFrame with 'event_type' etc.,
                # we can do a fallback approach if resource_df is missing columns.
                if isinstance(resource_df, pd.DataFrame) and not resource_df.empty:
                    # If 'event_type' not in columns, add it as an empty col to avoid KeyError
                    if "event_type" not in resource_df.columns:
                        resource_df["event_type"] = ""  # blank
                    self.viz.create_performance_report(
                        resource_df,
                        self.output_dir / "performance_report.json"
                    )
                else:
                    # fallback: create an empty DF with at least 'event_type'
                    fallback_df = pd.DataFrame(columns=["event_type"])
                    self.viz.create_performance_report(
                        fallback_df,
                        self.output_dir / "performance_report.json"
                    )
     
            except KeyError as ke:
                self.logger.log_error(
                    "visualization_error",
                    f"[{file_name} -> {function_name}] create_performance_report KeyError: {ke}"
                )
     
        except Exception as e:
            # Catch any unforeseen errors
            self.logger.log_error(
                "visualization_error",
                f"[{file_name} -> {function_name}]: Unexpected error: {str(e)}"
            )

 
    def resource_dict_to_df(resource_dict: dict) -> pd.DataFrame:
        """
        Convert the nested dict {step -> {device_id -> {metric_key -> metric_val}}}
        into a DataFrame with columns = [step, device_id, <metric_keys>].
        """
        rows = []
        for step, device_map in resource_dict.items():
            # device_map is e.g. { "device_0": {...}, "device_1": {...} }
            for dev_id, usage_dict in device_map.items():
                # usage_dict is e.g. {"memory_utilization": float, "compute_utilization": float, ...}
                row = {
                    "step": step,
                    "device_id": dev_id,
                }
                row.update(usage_dict)  # merge usage_dict keys/vals into the row
                rows.append(row)
        df = pd.DataFrame(rows)
        return df

    def _convert_tuple_keys(self, data):
        """
        Recursively convert any dict keys that are tuples into strings,
        so we can safely JSON-serialize them. 
        """
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                # if k is a tuple, convert to string
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
