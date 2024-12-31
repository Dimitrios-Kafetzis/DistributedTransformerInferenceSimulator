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
# File:    experiments/scenarios/__init__.py
# Description:
#   Initializes the experiment scenarios package, making common scenario
#   classes and baseline distributions available for usage across the
#   codebase.
#
# ---------------------------------------------------------------------------

"""
Initializes the scenarios module for different experiment configurations.
Provides classes and functions for running baseline comparisons, distributed
edge scenarios, edge cluster scenarios, and hybrid cloud-edge setups.
"""

from .common import (
    BaseScenario,
    ScenarioResult,
    run_scenario,
    load_scenario_config
)

from .edge_cluster_scenarios import (
    EdgeClusterBasicScenario,
    EdgeClusterScalabilityScenario,
    EdgeClusterFailureScenario
)

from .distributed_edge_scenarios import (
    DistributedEdgeBasicScenario,
    DistributedEdgeCommunicationScenario,
    DistributedEdgeHeterogeneityScenario
)

from .hybrid_cloud_scenarios import (
    HybridCloudBasicScenario,
    HybridCloudTierBalancingScenario,
    HybridCloudLatencyScenario
)

from .baseline_scenarios import (
    GreedyBaselineScenario,
    RoundRobinBaselineScenario,
    StaticBaselineScenario,
    DynamicMigrationBaselineScenario
)

# Version information
__version__ = '0.1.0'

# Define public interface
__all__ = [
    # Base components
    'BaseScenario',
    'ScenarioResult',
    'run_scenario',
    'load_scenario_config',
    
    # Edge cluster scenarios
    'EdgeClusterBasicScenario',
    'EdgeClusterScalabilityScenario',
    'EdgeClusterFailureScenario',
    
    # Distributed edge scenarios
    'DistributedEdgeBasicScenario',
    'DistributedEdgeCommunicationScenario',
    'DistributedEdgeHeterogeneityScenario',
    
    # Hybrid cloud scenarios
    'HybridCloudBasicScenario',
    'HybridCloudTierBalancingScenario',
    'HybridCloudLatencyScenario',
    
    # Baseline scenarios
    'GreedyBaselineScenario',
    'RoundRobinBaselineScenario',
    'StaticBaselineScenario',
    'DynamicMigrationBaselineScenario'
]