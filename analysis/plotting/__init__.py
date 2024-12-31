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
# File: analysis/plotting/__init__.py
# Description:
#   Plotting utilities for visualizing experiment results.
#   This module provides consolidated imports and version info for
#   the performance, resource, network, and comparison plots.
#
# ---------------------------------------------------------------------------

"""
Initializes the plotting module for the Transformer Inference Simulator.
Exposes various plotting functionalities for performance metrics, resource
utilization, network analysis, and algorithmic comparisons.
"""

from .performance_plots import (
    create_performance_plots,
    plot_latency_distribution,
    plot_throughput_comparison,
    plot_latency_over_time
)

from .resource_plots import (
    create_resource_plots,
    plot_memory_utilization,
    plot_cpu_utilization,
    plot_resource_efficiency
)

from .network_plots import (
    create_network_plots,
    plot_network_topology,
    plot_communication_overhead,
    plot_bandwidth_utilization
)

from .comparison_plots import (
    create_comparison_plots,
    plot_algorithm_comparison,
    plot_scenario_comparison,
    plot_scaling_behavior
)

__all__ = [
    'create_performance_plots',
    'create_resource_plots',
    'create_network_plots',
    'create_comparison_plots',
    'plot_latency_distribution',
    'plot_throughput_comparison',
    'plot_latency_over_time',
    'plot_memory_utilization',
    'plot_cpu_utilization',
    'plot_resource_efficiency',
    'plot_network_topology',
    'plot_communication_overhead',
    'plot_bandwidth_utilization',
    'plot_algorithm_comparison',
    'plot_scenario_comparison',
    'plot_scaling_behavior'
]