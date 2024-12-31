"""
Analysis package for transformer inference experiments.
Provides tools for analyzing results, generating visualizations,
and computing metrics.
"""

from .analyze_results import analyze_all_results
from .metrics import (
    calculate_performance_metrics,
    calculate_resource_metrics,
    calculate_network_metrics,
    calculate_statistics
)
from .plotting import (
    create_performance_plots,
    create_resource_plots,
    create_network_plots,
    create_comparison_plots
)

__version__ = '0.1.0'

__all__ = [
    'analyze_all_results',
    'calculate_performance_metrics',
    'calculate_resource_metrics',
    'calculate_network_metrics',
    'calculate_statistics',
    'create_performance_plots',
    'create_resource_plots',
    'create_network_plots',
    'create_comparison_plots'
]