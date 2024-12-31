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
# File:    analysis/utils/__init__.py
# Description:
#   Utility functions for data loading, processing, and other
#   supporting tasks in the analysis of distributed transformer
#   inference metrics.
#
# ---------------------------------------------------------------------------

"""
Utility subpackage initialization for the analysis module.
Exposes data loading, data processing, and other helper functions
used to prepare and analyze metrics from distributed transformer
inference experiments.
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