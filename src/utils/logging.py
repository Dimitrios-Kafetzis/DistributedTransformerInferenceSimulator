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
# File:    src/utils/logging.py
# Description:
#   Defines logging utilities for the Transformer Inference Simulator,
#   including console/file logging, structured metrics logging, and
#   customizable log levels.
#
# ---------------------------------------------------------------------------

"""
Offers structured logging facilities to capture simulation events, errors,
and metrics. Includes SimulationLogger, which manages background threads for
logging JSON metrics, and the NullLogger for disabling logs when required.
"""

import logging
import sys
from enum import Enum
from typing import Dict, Optional, Union
from pathlib import Path
import json
import time
from datetime import datetime
import threading
from queue import Queue
import atexit

class LogLevel(Enum):
    """Log levels for simulation events"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

class SimulationLogger:
    """Thread-safe logger for simulation events and metrics"""
    
    def __init__(
        self,
        name: str,
        log_dir: Union[str, Path],
        level: LogLevel = LogLevel.INFO,
        console_output: bool = True,
        file_output: bool = True
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.level = level
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level.value)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Add console handler if requested
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self._create_formatter())
            self.logger.addHandler(console_handler)
            
        # Add file handler if requested
        if file_output:
            log_file = self.log_dir / f"{name}_{int(time.time())}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self._create_formatter())
            self.logger.addHandler(file_handler)
            
        # Metrics logging setup
        self.metrics_file = self.log_dir / f"{name}_metrics.jsonl"
        self.metrics_queue: Queue = Queue()
        self.metrics_thread = threading.Thread(
            target=self._metrics_writer,
            daemon=True
        )
        self.metrics_thread.start()
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
        
        self.logger.info(f"Logger initialized: {name}")
        
    def _create_formatter(self) -> logging.Formatter:
        """Create log formatter"""
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def _metrics_writer(self) -> None:
        """Background thread for writing metrics to file"""
        with open(self.metrics_file, 'a') as f:
            while True:
                try:
                    metrics = self.metrics_queue.get()
                    if metrics is None:  # Shutdown signal
                        break
                    json.dump(metrics, f)
                    f.write('\n')
                    f.flush()
                except Exception as e:
                    self.logger.error(f"Error writing metrics: {e}")
                    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.metrics_queue.put(None)  # Signal thread to stop
        self.metrics_thread.join()
        
    def log_event(
        self,
        event_type: str,
        message: str,
        level: LogLevel = LogLevel.INFO,
        **kwargs
    ) -> None:
        """Log a simulation event"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'message': message,
            **kwargs
        }
        
        self.logger.log(level.value, json.dumps(log_data))
        
    def log_metrics(self, metrics: Dict) -> None:
        """Log metrics data"""
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        self.metrics_queue.put(metrics_data)
        
    def log_resource_state(
        self,
        step: int,
        device_id: str,
        memory_used: float,
        memory_total: float,
        compute_used: float,
        compute_total: float
    ) -> None:
        """Log resource utilization state"""
        self.log_metrics({
            'step': step,
            'device_id': device_id,
            'resource_state': {
                'memory': {
                    'used': memory_used,
                    'total': memory_total,
                    'utilization': memory_used / memory_total
                },
                'compute': {
                    'used': compute_used,
                    'total': compute_total,
                    'utilization': compute_used / compute_total
                }
            }
        })
        
    def log_component_assignment(
        self,
        step: int,
        component_id: str,
        device_id: str,
        assignment_type: str
    ) -> None:
        """Log component assignment decision"""
        self.log_event(
            'component_assignment',
            f"Component {component_id} assigned to device {device_id}",
            level=LogLevel.INFO,
            step=step,
            component_id=component_id,
            device_id=device_id,
            assignment_type=assignment_type
        )
        
    def log_migration(
        self,
        step: int,
        component_id: str,
        source_device: str,
        target_device: str,
        reason: str,
        cost: float
    ) -> None:
        """Log component migration"""
        self.log_event(
            'migration',
            f"Migrating {component_id} from {source_device} to {target_device}",
            level=LogLevel.INFO,
            step=step,
            component_id=component_id,
            source_device=source_device,
            target_device=target_device,
            reason=reason,
            cost=cost
        )
        
    def log_communication(
        self,
        step: int,
        source_component: str,
        target_component: str,
        data_size: float,
        transfer_time: float
    ) -> None:
        """Log communication events"""
        self.log_metrics({
            'step': step,
            'communication': {
                'source': source_component,
                'target': target_component,
                'data_size': data_size,
                'transfer_time': transfer_time
            }
        })
        
    def log_error(
        self,
        error_type: str,
        message: str,
        **kwargs
    ) -> None:
        """Log error events"""
        self.log_event(
            'error',
            message,
            level=LogLevel.ERROR,
            error_type=error_type,
            **kwargs
        )
        
    def log_warning(
        self,
        warning_type: str,
        message: str,
        **kwargs
    ) -> None:
        """Log warning events"""
        self.log_event(
            'warning',
            message,
            level=LogLevel.WARNING,
            warning_type=warning_type,
            **kwargs
        )
        
    def log_performance(
        self,
        step: int,
        metrics: Dict
    ) -> None:
        """Log performance metrics"""
        self.log_metrics({
            'step': step,
            'performance': metrics
        })

def setup_logging(
    experiment_name: str,
    log_dir: Union[str, Path],
    console_level: LogLevel = LogLevel.DEBUG,
    file_level: LogLevel = LogLevel.DEBUG
) -> SimulationLogger:
    """Set up logging for an experiment"""
    numeric_level = min(console_level.value, file_level.value)
    return SimulationLogger(
        name=experiment_name,
        log_dir=log_dir,
        level=LogLevel(numeric_level),
        console_output=True,
        file_output=True
    )

class NullLogger(SimulationLogger):
    """Null logger for testing or when logging is disabled"""
    
    def __init__(self):
        pass
        
    def log_event(self, *args, **kwargs) -> None:
        pass
        
    def log_metrics(self, *args, **kwargs) -> None:
        pass
        
    def cleanup(self) -> None:
        pass