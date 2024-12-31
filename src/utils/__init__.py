"""
Utility functions and helpers for transformer inference simulation.
Provides configuration management, logging, and visualization tools.
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