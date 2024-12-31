# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Dimitrios Kafetzis
#
# This file is part of the Transformer Inference Simulator project.
# Licensed under the MIT License; you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#   https://opensource.org/licenses/MIT
#
# Author: Dimitrios Kafetzis (dimitrioskafetzis@gmail.com)
# File: analysis/metrics/__init__.py
# Description:
#   Metrics calculation and analysis for transformer inference experiments.
#   This package provides modules for performance, resource, network,
#   and statistical metrics, facilitating a comprehensive analysis of 
#   distributed transformer inference.
#
# ---------------------------------------------------------------------------

"""
This __init__.py aggregates and exposes the metrics modules within the
`analysis.metrics` package. It defines common imports and sets the public API
for performance, resource usage, network analysis, and statistical calculations.
"""

from .performance import (
    calculate_performance_metrics,
    calculate_latency_metrics,
    calculate_throughput_metrics,
    analyze_performance_trends
)

from .resource import (
    calculate_resource_metrics,
    calculate_memory_metrics,
    calculate_cpu_metrics,
    analyze_resource_efficiency
)

from .network import (
    calculate_network_metrics,
    calculate_communication_overhead,
    calculate_bandwidth_utilization,
    analyze_network_patterns
)

from .statistics import (
    calculate_statistics,
    perform_significance_tests,
    calculate_confidence_intervals,
    analyze_distributions
)

__all__ = [
    # Performance metrics
    'calculate_performance_metrics',
    'calculate_latency_metrics',
    'calculate_throughput_metrics',
    'analyze_performance_trends',
    
    # Resource metrics
    'calculate_resource_metrics',
    'calculate_memory_metrics',
    'calculate_cpu_metrics',
    'analyze_resource_efficiency',
    
    # Network metrics
    'calculate_network_metrics',
    'calculate_communication_overhead',
    'calculate_bandwidth_utilization',
    'analyze_network_patterns',
    
    # Statistical analysis
    'calculate_statistics',
    'perform_significance_tests',
    'calculate_confidence_intervals',
    'analyze_distributions'
]