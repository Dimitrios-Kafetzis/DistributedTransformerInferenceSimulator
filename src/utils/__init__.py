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
# File:    src/utils/__init__.py
# Description:
#   Initializes the utilities package for configuration, logging,
#   and visualization functionalities used throughout the
#   Transformer Inference Simulator.
#
# ---------------------------------------------------------------------------

"""
Initializes the utility subpackage, providing configuration file management,
structured logging, and data visualization tools to support distributed
Transformer inference simulations.
"""


from .config import (
    SimulationConfig,
    load_config,
    save_config,
    validate_config
)

from .logging import (
    setup_logging,
    SimulationLogger,
    LogLevel
)

from .visualization import (
    VisualizationManager
)

# Version information
__version__ = '0.1.0'

# Define public interface
__all__ = [
    # Configuration management
    'SimulationConfig',
    'load_config',
    'save_config',
    'validate_config',
    
    # Logging utilities
    'setup_logging',
    'SimulationLogger',
    'LogLevel',
    
    # Visualization tools
    'VisualizationManager'
]

# Module level documentation
SimulationConfig.__doc__ = """
Configuration management for simulation parameters and settings.
Handles loading, saving, and validating configuration files.
"""

SimulationLogger.__doc__ = """
Structured logging system for simulation events and metrics.
Provides different logging levels and output formats.
"""

VisualizationManager.__doc__ = """
Manages visualization and plotting utilities for simulation results.
Provides methods for creating plots and generating performance reports.
"""