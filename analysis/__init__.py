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
# File:    analysis/__init__.py
# Description:
#   Analysis package initialization. Provides modules for metrics calculation,
#   plotting, and result analysis for distributed transformer inference
#   simulations.
#
# ---------------------------------------------------------------------------

"""
Initializes the analysis package, which includes submodules for metrics
calculation, visualization tools, and comprehensive results analysis
of distributed transformer inference experiments.
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