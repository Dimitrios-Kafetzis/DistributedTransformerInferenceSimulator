"""
Utility functions for data loading and processing in analysis.
"""

from .data_loading import (
    load_experimental_data,
    load_metrics_data,
    load_configuration,
    ensure_directory
)

from .processing import (
    process_raw_data,
    clean_metrics_data,
    aggregate_results,
    normalize_metrics
)

__version__ = '0.1.0'

__all__ = [
    # Data loading utilities
    'load_experimental_data',
    'load_metrics_data',
    'load_configuration',
    'ensure_directory',
    
    # Data processing utilities
    'process_raw_data',
    'clean_metrics_data',
    'aggregate_results',
    'normalize_metrics'
]